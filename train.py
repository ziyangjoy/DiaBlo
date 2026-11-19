"""
DiaBlo training script (with LoRA baseline, resumable adapter checkpoints, 4-bit bases).

Usage:
    python train.py --task GSM8k --model_name Qwen2.5-1.5B --method diablo --num_blocks 64 --lr 5e-4 --precision fp16
    python train.py --task GSM8k --model_name Qwen2.5-1.5B --method lora --lora_r 16 --lr 2e-4 --precision fp16
    python train.py ... --resume            # continue from the latest checkpoint in output_dir
"""

import os
import sys
import json
import time
import shutil
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback, set_seed
from datasets import load_from_disk
import datasets.features.features as _dff
if "List" not in _dff._FEATURE_TYPES:
    _dff._FEATURE_TYPES["List"] = _dff._FEATURE_TYPES["Sequence"]
from torch.nn.utils.rnn import pad_sequence

from diablo import (CustomTrainer, CustomTrainingArguments, resolve_model, parse_target_modules,
                    setup_pad_token, apply_adapter, mark_trainable)
from diablo.common import train_data_paths

# ── Argument parsing ────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Train with DiaBlo (Diagonal Block adapters) or LoRA")

# Model
parser.add_argument("--model_name", type=str, default="Llama2-7B",
                    help="Short name from diablo/common.py MODEL_MAP, or any HF model id")
parser.add_argument("--model_cache_dir", type=str, default=None, help="HuggingFace model cache directory")
parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token (or set HF_TOKEN env var)")
parser.add_argument("--load_in_4bit", action="store_true", help="Load base in 4-bit NF4 (bitsandbytes)")

# Method
parser.add_argument("--method", type=str, default="diablo", choices=["diablo", "lora"])
parser.add_argument("--num_blocks", type=int, default=64, help="DiaBlo: number of diagonal blocks")
parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha (default 2*r)")
parser.add_argument("--dropout", type=float, default=0.05, help="Adapter dropout rate")
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
parser.add_argument("--max_steps", type=int, default=-1, help="Override epochs (smoke tests)")
parser.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
parser.add_argument("--scheduler", type=str, default="linear", help="LR scheduler type")
parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"],
                    help="Training precision (use fp16 or fp32 on T4)")
parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")
parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")

# Saving / logging
parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
parser.add_argument("--run_name", type=str, default=None, help="Override auto-generated run name")
parser.add_argument("--save_adapter_only", action="store_true", default=True,
                    help="Save only adapter weights (default: True)")
parser.add_argument("--save_strategy", type=str, default="steps", help="Save strategy (steps|epoch)")
parser.add_argument("--save_steps", type=int, default=100, help="Checkpoint every N steps")
parser.add_argument("--save_total_limit", type=int, default=2, help="Max checkpoints to keep")
parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output dir")
parser.add_argument("--report_to", type=str, default="none", help="none | wandb | tensorboard")
parser.add_argument("--deadline", type=float, default=None,
                    help="Unix time. Checkpoint and exit with code 3 before it (Slurm segments of long runs)")

# Layer selection
parser.add_argument("--layer_min", type=int, default=0, help="Min layer index to apply adapters")
parser.add_argument("--layer_max", type=int, default=None, help="Max layer index (default: last layer)")

parser.add_argument("--seed", type=int, default=42, help="Random seed")

args = parser.parse_args()
set_seed(args.seed)

# ── Setup ───────────────────────────────────────────────────────────────────

hf_token = args.hf_token or os.environ.get("HF_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(hf_token)

model_dir = args.model_cache_dir or os.environ.get("MODEL_CACHE_DIR", None)

if args.precision == "bf16" and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
    raise SystemExit("This GPU does not support bf16 (e.g. T4). Use --precision fp16 or fp32.")

# Precision: base weights in compute_precision, adapters in adapter_precision
if args.precision == "fp16":
    compute_precision, adapter_precision = torch.float16, torch.float32
    fp16, bf16 = True, False
elif args.precision == "bf16":
    compute_precision, adapter_precision = torch.bfloat16, torch.bfloat16
    fp16, bf16 = False, True
else:
    compute_precision, adapter_precision = torch.float32, torch.float32
    fp16, bf16 = False, False

load_name = resolve_model(args.model_name)
target_modules_list = parse_target_modules(args.target_modules)

print(f"Config: {args}")

# ── Load model ──────────────────────────────────────────────────────────────

load_kwargs = dict(cache_dir=model_dir, torch_dtype=compute_precision)
if args.load_in_4bit:
    from transformers import BitsAndBytesConfig
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_precision, bnb_4bit_use_double_quant=False,
    )
    load_kwargs["device_map"] = {"": 0}

model = AutoModelForCausalLM.from_pretrained(load_name, **load_kwargs)
tokenizer = AutoTokenizer.from_pretrained(load_name, cache_dir=model_dir)
setup_pad_token(tokenizer, model)

model = apply_adapter(
    model, args.method, target_modules_list,
    num_blocks=args.num_blocks, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
    dropout=args.dropout, layer_min=args.layer_min, layer_max=args.layer_max,
    quantized=args.load_in_4bit,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
if not args.load_in_4bit:
    model.to(device)

total_params, trainable_params = mark_trainable(model, adapter_precision)
print(f"Total params: {total_params:,} | Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

if args.gradient_checkpointing:
    model.enable_input_require_grads()

if args.load_in_4bit:
    # HF Trainer refuses to train "purely quantized" models unless it sees a PEFT
    # adapter. Our DiaBlo adapters are trainable, so tell it so.
    model._hf_peft_config_loaded = True

# ── Load data ───────────────────────────────────────────────────────────────

pad_token_id = tokenizer.pad_token_id
padding_side = "left"

train_path, val_path = train_data_paths(args.data_dir, args.task, args.model_name, args.max_length)
training_data = load_from_disk(train_path)
validation_data = load_from_disk(val_path) if val_path else None
if args.task == "GSM8k_lori":
    n_keep = (len(training_data) // args.per_device_batch_size // args.gradient_accumulation_steps) * args.per_device_batch_size * args.gradient_accumulation_steps
    training_data = training_data.select(range(n_keep))


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
size_tag = f"block{args.num_blocks}" if args.method == "diablo" else f"lora_r{args.lora_r}"
name = args.run_name or (
    f"{args.model_name.replace('/', '_')}_{size_tag}"
    f"_lr{args.lr}_bs{batch_size}_ep{args.max_epochs}_{args.task}_seed{args.seed}"
)
output_dir = os.path.join(args.output_dir, name)
os.makedirs(output_dir, exist_ok=True)

if args.report_to == "wandb":
    import wandb
    wandb.init(project=f"DiaBlo-{args.task}", name=name, config=vars(args))

has_eval = validation_data is not None
training_args = CustomTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=args.max_epochs,
    max_steps=args.max_steps,
    per_device_train_batch_size=args.per_device_batch_size,
    per_device_eval_batch_size=args.per_device_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    learning_rate=args.lr,
    lr_scheduler_type=args.scheduler,
    warmup_steps=args.warmup_steps,
    weight_decay=args.weight_decay,
    optim="adamw_torch",
    logging_steps=10,
    eval_strategy="epoch" if has_eval else "no",
    save_strategy=args.save_strategy,
    save_steps=args.save_steps,
    load_best_model_at_end=has_eval,
    bf16=bf16,
    fp16=fp16,
    save_total_limit=args.save_total_limit,
    metric_for_best_model="loss",
    greater_is_better=False,
    label_names=["labels"],
    save_adapter_only=args.save_adapter_only,
    gradient_checkpointing=args.gradient_checkpointing,
    gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
    dataloader_drop_last=True,
    report_to=args.report_to,
    seed=args.seed,
)

# Record the exact config next to the checkpoints (used by merge/eval scripts).
with open(os.path.join(output_dir, "run_config.json"), "w") as f:
    json.dump({**vars(args), "hf_model_id": load_name, "trainable_params": trainable_params,
               "total_params": total_params}, f, indent=2)

model.train()
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=training_data,
    eval_dataset=validation_data,
    data_collator=collate_fn,
)

class DeadlineCallback(TrainerCallback):
    """Save a checkpoint and stop when the next two steps might not finish before the deadline.

    A job killed by Slurm in the middle of a checkpoint save leaves a broken checkpoint. Stopping
    ourselves first means every segment of a multi-day run ends on a complete checkpoint.
    """

    def __init__(self, deadline):
        self.deadline, self.t_prev, self.step_s, self.stopped = deadline, None, 0.0, False

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        if self.t_prev is not None:
            self.step_s = max(self.step_s, now - self.t_prev)
        self.t_prev = now
        if now + 2 * self.step_s + 60 > self.deadline:
            control.should_save = True
            control.should_training_stop = True
            self.stopped = True
            print(f"Deadline reached at step {state.global_step}: saving checkpoint and stopping.", flush=True)
        return control


deadline_cb = DeadlineCallback(args.deadline) if args.deadline else None
if deadline_cb:
    trainer.add_callback(deadline_cb)

resume = None
if args.resume:
    from transformers.trainer_utils import get_last_checkpoint
    # HF writes trainer_state.json last, so a checkpoint without it was cut off mid-save (killed job).
    for d in os.listdir(output_dir):
        p = os.path.join(output_dir, d)
        if d.startswith("checkpoint-") and os.path.isdir(p) and not os.path.exists(os.path.join(p, "trainer_state.json")):
            print(f"Removing incomplete checkpoint {p}")
            shutil.rmtree(p)
    resume = get_last_checkpoint(output_dir)
    print(f"Resuming from: {resume}" if resume else "No checkpoint found, starting fresh.")

start_step = 0
if resume:
    with open(os.path.join(resume, "trainer_state.json")) as f:
        start_step = json.load(f)["global_step"]
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
out = trainer.train(resume_from_checkpoint=resume)

# Timing/memory of this process (one Slurm segment); scripts/slurm/bench.sbatch projects run cost from it.
steps_per_epoch = len(training_data) // (args.per_device_batch_size * args.gradient_accumulation_steps)
stats = {"steps_this_process": trainer.state.global_step - start_step, "global_step": trainer.state.global_step,
         "max_steps": trainer.state.max_steps, "steps_per_epoch": steps_per_epoch, "n_train": len(training_data),
         "train_runtime_s": out.metrics.get("train_runtime"),
         "peak_mem_gb": torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else None,
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
if stats["steps_this_process"] > 0 and stats["train_runtime_s"]:
    stats["s_per_step"] = stats["train_runtime_s"] / stats["steps_this_process"]
print("TRAIN_STATS " + json.dumps(stats))
with open(os.path.join(output_dir, "train_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)

if trainer.state.global_step < trainer.state.max_steps:
    # Stopped by the deadline: the checkpoint is saved, but the run is not finished. No final adapter,
    # no log_history.json (those mark a finished run). The next segment resumes with --resume.
    print(f"Segment finished at step {trainer.state.global_step}/{trainer.state.max_steps}; resume to continue.")
    sys.exit(3)

trainer.save_model(output_dir)  # final adapter.chkpt at output_dir root
with open(os.path.join(output_dir, "log_history.json"), "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)
print(f"Training complete. Adapter saved to {os.path.join(output_dir, 'adapter.chkpt')}")
