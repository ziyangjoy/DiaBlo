"""
Evaluate a model on GSM8k with HF generate (greedy, 512 new tokens).

Same prompt / scoring as the original script. Additions: LoRA + base-model support,
4-bit bases, resumable per-example JSONL output, full 1319-example evaluation.

Note: like the original, the answer is extracted from the decoded prompt+completion.

Usage:
    python evaluate_gsm8k.py --model_name Qwen2.5-1.5B --method diablo --num_blocks 64 \
        --adapter_path results/.../adapter.chkpt --output_file eval/run.jsonl --precision fp16
    python evaluate_gsm8k.py --model_name Qwen2.5-1.5B --method none --output_file eval/base.jsonl
    python evaluate_gsm8k.py --model_path merged_dir --method none ...   # pre-merged model
"""

import os
import json
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

from diablo import resolve_model, parse_target_modules, setup_pad_token, apply_adapter, load_adapter
from gsm8k_utils import build_prompt, is_correct, load_jsonl, summarize

parser = argparse.ArgumentParser(description="Evaluate on GSM8k (HF generate)")
parser.add_argument("--model_name", type=str, default="Llama2-7B")
parser.add_argument("--model_path", type=str, default=None, help="Local/merged model dir (overrides model_name)")
parser.add_argument("--method", type=str, default="diablo", choices=["diablo", "lora", "none"])
parser.add_argument("--adapter_path", type=str, default=None, help="Path to adapter.chkpt")
parser.add_argument("--data_dir", type=str, default="./datasets", help="Dataset directory")
parser.add_argument("--num_blocks", type=int, default=64)
parser.add_argument("--lora_r", type=int, default=16)
parser.add_argument("--lora_alpha", type=int, default=None)
parser.add_argument("--target_modules", type=str, default="qkvud")
parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
parser.add_argument("--load_in_4bit", action="store_true")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--limit", type=int, default=None, help="Only first N test examples")
parser.add_argument("--output_file", type=str, required=True, help="Per-example JSONL (resumable)")
parser.add_argument("--model_cache_dir", type=str, default=None)
parser.add_argument("--hf_token", type=str, default=None)
args = parser.parse_args()

hf_token = args.hf_token or os.environ.get("HF_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(hf_token)

model_dir = args.model_cache_dir or os.environ.get("MODEL_CACHE_DIR", None)
device = "cuda" if torch.cuda.is_available() else "cpu"
compute_precision = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
load_name = args.model_path or resolve_model(args.model_name)

# ── Model ────────────────────────────────────────────────────────────────────
load_kwargs = dict(cache_dir=model_dir, torch_dtype=compute_precision)
if args.load_in_4bit:
    from transformers import BitsAndBytesConfig
    load_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=compute_precision)
    load_kwargs["device_map"] = {"": 0}

model = AutoModelForCausalLM.from_pretrained(load_name, **load_kwargs)
tokenizer = AutoTokenizer.from_pretrained(load_name, cache_dir=model_dir, padding_side="left", use_fast=False)
setup_pad_token(tokenizer, model)

if args.method != "none":
    model = apply_adapter(model, args.method, parse_target_modules(args.target_modules),
                          num_blocks=args.num_blocks, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                          dropout=0.0, quantized=args.load_in_4bit)
    load_adapter(model, args.adapter_path)

if not args.load_in_4bit:
    model.to(compute_precision).to(device)
model.eval()
for m in model.modules():
    if isinstance(m, torch.nn.Dropout):
        m.p = 0.0

# ── Data ─────────────────────────────────────────────────────────────────────
dataset = load_from_disk(os.path.join(args.data_dir, "GSM8k_raw"))
test_data = list(dataset["test"])
n_test = len(test_data)
if args.limit:
    test_data = test_data[: args.limit]

os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
done = {r["idx"] for r in load_jsonl(args.output_file)}
todo = [i for i in range(len(test_data)) if i not in done]
print(f"{len(done)} already evaluated, {len(todo)} to go.")

gen_kwargs = {"max_new_tokens": args.max_new_tokens, "do_sample": False,
              "pad_token_id": tokenizer.pad_token_id}

import time
t_gen = time.time()
with torch.no_grad(), open(args.output_file, "a") as fout:
    for start in tqdm(range(0, len(todo), args.batch_size), desc="Evaluating GSM8k"):
        idxs = todo[start: start + args.batch_size]
        prompts = [build_prompt(test_data[i]["question"]) for i in idxs]
        inputs = tokenizer(prompts, return_tensors="pt", padding="longest", add_special_tokens=False)
        out = model.generate(input_ids=inputs.input_ids.to(device),
                             attention_mask=inputs.attention_mask.to(device), **gen_kwargs)
        texts = tokenizer.batch_decode(out, skip_special_tokens=True)
        for i, text in zip(idxs, texts):
            fout.write(json.dumps({"idx": i, "output": text, "correct": is_correct(test_data[i], text)}) + "\n")
        fout.flush()

records = load_jsonl(args.output_file)
summary = summarize(records, n_test if not args.limit else len(test_data))
summary.update({"model": load_name, "method": args.method, "adapter": args.adapter_path, "backend": "hf",
                "seconds": round(time.time() - t_gen, 1), "generated": len(todo)})
print(json.dumps(summary, indent=2))
with open(args.output_file.replace(".jsonl", "") + "_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
