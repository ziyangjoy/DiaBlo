"""
DiaBlo training script.

Usage:
    python train.py --task GSM8k --model_name Llama2-7B --num_blocks 64 --lr 5e-4
"""

import os
import math
import argparse

import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from datasets import load_from_disk
import datasets.features.features as _dff
if "List" not in _dff._FEATURE_TYPES:
    _dff._FEATURE_TYPES["List"] = _dff._FEATURE_TYPES["Sequence"]
from torch.nn.utils.rnn import pad_sequence

from diablo import replace_linear_with_blocklinear, CustomTrainer, CustomTrainingArguments

# ── Argument parsing ────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Train with DiaBlo (Diagonal Block adapters)")

# Model
parser.add_argument("--model_name", type=str, default="Llama2-7B",
                     choices=["Llama2-7B", "Llama2-13B", "Llama3-8B", "Llama3-3B", "Mistral-7B", "Qwen2.5-7B"])
parser.add_argument("--model_cache_dir", type=str, default=None, help="HuggingFace model cache directory")
parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token (or set HF_TOKEN env var)")

# DiaBlo hyperparameters
parser.add_argument("--num_blocks", type=int, default=64, help="Number of diagonal blocks")
parser.add_argument("--dropout", type=float, default=0.05, help="Dropout rate")
parser.add_argument("--target_modules", type=str, default="qkvud",
                     help="Target modules: q=q_proj, k=k_proj, v=v_proj, u=up_proj, d=down_proj, o=o_proj, g=gate_proj")

# Training
parser.add_argument("--task", type=str, default="GSM8k",
                     choices=["GSM8k", "GSM8k_lori", "metamath", "commonsense", "commonsense_new"])
parser.add_argument("--data_dir", type=str, default="./datasets", help="Root directory for processed datasets")
parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
parser.add_argument("--per_device_batch_size", type=int, default=4, help="Per-device batch size")
parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
parser.add_argument("--max_epochs", type=int, default=3, help="Number of training epochs")
parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
parser.add_argument("--scheduler", type=str, default="linear", help="LR scheduler type")
parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"],
                     help="Training precision")
parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")

# Saving
parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
parser.add_argument("--save_adapter_only", action="store_true", default=True,
                     help="Save only adapter weights (default: True)")
parser.add_argument("--save_strategy", type=str, default="epoch", help="Save strategy")
parser.add_argument("--save_total_limit", type=int, default=3, help="Max checkpoints to keep")

# Layer selection
parser.add_argument("--layer_min", type=int, default=0, help="Min layer index to apply DiaBlo")
parser.add_argument("--layer_max", type=int, default=31, help="Max layer index to apply DiaBlo")

parser.add_argument("--seed", type=int, default=42, help="Random seed")

args = parser.parse_args()

# ── Setup ───────────────────────────────────────────────────────────────────

# HuggingFace login
hf_token = args.hf_token or os.environ.get("HF_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(hf_token)

model_dir = args.model_cache_dir or os.environ.get("MODEL_CACHE_DIR", None)

# Precision
if args.precision == "fp16":
    compute_precision = torch.float16
    adapter_precision = torch.float32
    fp16, bf16 = True, False
elif args.precision == "bf16":
    compute_precision = torch.bfloat16
    adapter_precision = torch.bfloat16
    fp16, bf16 = False, True
else:
    compute_precision = torch.float32
    adapter_precision = torch.float32
    fp16, bf16 = False, False

# Model name mapping
MODEL_MAP = {
    "Llama2-7B": "meta-llama/Llama-2-7b-hf",
    "Llama2-13B": "meta-llama/Llama-2-13b-hf",
    "Llama3-8B": "meta-llama/Meta-Llama-3-8B",
    "Llama3-3B": "meta-llama/Llama-3.2-3B",
    "Mistral-7B": "mistralai/Mistral-7B-v0.1",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
}
load_name = MODEL_MAP[args.model_name]

# Parse target modules
MODULE_MAP = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "u": "up_proj", "d": "down_proj", "o": "o_proj", "g": "gate_proj"}
target_modules_list = [MODULE_MAP[c] for c in args.target_modules if c in MODULE_MAP]

print(f"Config: {args}")

# ── Load model ──────────────────────────────────────────────────────────────

model = AutoModelForCausalLM.from_pretrained(load_name, cache_dir=model_dir, torch_dtype=compute_precision)
tokenizer = AutoTokenizer.from_pretrained(load_name, cache_dir=model_dir)
tokenizer.add_special_tokens({'pad_token': '<PAD>'})
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = tokenizer.pad_token_id

# Apply DiaBlo adapters
replace_linear_with_blocklinear(
    model.model.layers[args.layer_min : args.layer_max + 1],
    num_blocks=args.num_blocks,
    drop_out=args.dropout,
    target_modules=target_modules_list,
)

device = "cuda"
model.to(device)

# Freeze all parameters, then unfreeze adapter parameters
for param in model.parameters():
    param.requires_grad = False

for name, module in model.named_modules():
    if type(module).__name__ == "BlockLinear":
        module.block_A.requires_grad = True
        module.block_A.data = module.block_A.data.to(adapter_precision)
        module.linear.weight.requires_grad = False
        if hasattr(module.linear, "bias") and module.linear.bias is not None:
            module.linear.bias.requires_grad = False

# Print trainable parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total_params:,} | Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

# ── Load data ───────────────────────────────────────────────────────────────

pad_token_id = tokenizer.pad_token_id
padding_side = "left"

if args.task == "GSM8k":
    training_data = load_from_disk(os.path.join(args.data_dir, "GSM8k_processed/train"))
    validation_data = None
    length = args.max_length
elif args.task == "GSM8k_lori":
    training_data = load_from_disk(f'/network/rit/lab/ziyang_lab/ziyang/github/Kronecker_LoRA/datasets/gsm8k_lori_shuffle_{args.model_name}_special_token')
    # Truncate to 7456 so that with drop_last and grad_accum=8, we get exactly 233 steps/epoch (matching run_BlockLoRA.py)
    n_keep = (len(training_data) // args.per_device_batch_size // args.gradient_accumulation_steps) * args.per_device_batch_size * args.gradient_accumulation_steps
    training_data = training_data.select(range(n_keep))
    validation_data = None
    length = args.max_length
elif args.task == "metamath":
    training_data = load_from_disk(
        os.path.join(args.data_dir, f"metamath/train_all_{args.max_length}_OnlyOutput")
    )
    validation_data = None
    length = args.max_length
elif args.task == "commonsense":
    training_data = load_from_disk(
        os.path.join(args.data_dir, f"commonsense_170k_new/train_all_{args.max_length}_OnlyOutput_{args.model_name}")
    )
    validation_data = load_from_disk(
        os.path.join(args.data_dir, "dataset_commonsense/combined_val_0.1_512")
    )
    length = args.max_length
elif args.task == "commonsense_new":
    training_data = load_from_disk(
        os.path.join(args.data_dir, f"commonsense_170k_new/train_val120_{args.max_length}_OnlyOutput_{args.model_name}")
    )
    validation_data = None
    length = args.max_length
else:
    raise ValueError(f"Unknown task: {args.task}")


def collate_fn(batch):
    input_ids = [torch.tensor(b["input_ids"]) for b in batch]
    attention_mask = [torch.tensor(b["attention_mask"]) for b in batch]
    target_ids = [torch.tensor(b["labels"]) for b in batch]

    input_ids = torch.swapaxes(pad_sequence(input_ids, padding_value=pad_token_id, padding_side=padding_side), 0, 1)
    attention_mask = torch.swapaxes(pad_sequence(attention_mask, padding_value=0, padding_side=padding_side), 0, 1)
    target_ids = torch.swapaxes(pad_sequence(target_ids, padding_value=-100, padding_side=padding_side), 0, 1)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": target_ids}


# ── Training ────────────────────────────────────────────────────────────────

batch_size = args.per_device_batch_size * args.gradient_accumulation_steps
name = (
    f"{args.model_name}_block{args.num_blocks}"
    f"_lr{args.lr}_bs{batch_size}_ep{args.max_epochs}_{args.task}"
)
output_dir = os.path.join(args.output_dir, args.model_name, args.task, name)

wandb.init(
    project=f"DiaBlo-{args.task}",
    name=name,
    config=vars(args),
)

has_eval = validation_data is not None
training_args = CustomTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=args.max_epochs,
    per_device_train_batch_size=args.per_device_batch_size,
    per_device_eval_batch_size=args.per_device_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.lr,
    lr_scheduler_type=args.scheduler,
    warmup_steps=args.warmup_steps,
    weight_decay=args.weight_decay,
    optim="adamw_torch",
    logging_dir="./logs",
    logging_steps=10,
    eval_strategy="epoch" if has_eval else "no",
    save_strategy=args.save_strategy,
    load_best_model_at_end=has_eval,
    bf16=bf16,
    fp16=fp16,
    save_total_limit=args.save_total_limit,
    metric_for_best_model="loss",
    greater_is_better=False,
    label_names=["labels"],
    save_adapter_only=args.save_adapter_only,
    gradient_checkpointing=args.gradient_checkpointing,
    dataloader_drop_last=True,
)

model.train()
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    eval_dataset=validation_data,
    data_collator=collate_fn,
)

trainer.train()
print(f"Training complete. Model saved to {output_dir}")
