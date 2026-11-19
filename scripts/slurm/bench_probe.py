"""Pick the per-device batch size and gradient checkpointing for one run, from its worst-case memory.

Builds the model exactly like train.py (same dtype rules, adapter, trainable params in the adapter dtype,
AdamW state allocated) and runs one forward/backward/optimizer step on the run's LONGEST training examples.
A real batch can never be longer, so a configuration that fits here fits for the whole run.

Candidates are tried fastest-first: no checkpointing with batch 4, 2, then checkpointing with 4, 2, 1.
Only batch sizes that divide the effective batch are tried. Prints PROBE lines and writes
{"bs", "ga", "gc", "peak_gb", "total_gb"} to --out.

    python scripts/slurm/bench_probe.py --model Llama2-7B --task commonsense_new --data_dir $DATA \
        --method diablo --size 64 --modules qkvud --precision fp16 --max_length 256 --eff 16 --out bench/C1_probe.json
"""

import argparse
import gc as pygc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pyarrow.compute as pc
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

from diablo import apply_adapter, mark_trainable, parse_target_modules, resolve_model, setup_pad_token
from diablo.common import train_data_paths

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--task", required=True)
ap.add_argument("--data_dir", required=True)
ap.add_argument("--method", default="diablo")
ap.add_argument("--size", type=int, default=64)
ap.add_argument("--modules", default="qkvud")
ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16"])
ap.add_argument("--max_length", type=int, default=512)
ap.add_argument("--eff", type=int, default=16, help="effective batch (per-device x accumulation)")
ap.add_argument("--headroom_gb", type=float, default=1.0)
ap.add_argument("--only_gc", action="store_true", help="skip the no-checkpointing candidates")
ap.add_argument("--out", required=True)
args = ap.parse_args()

assert torch.cuda.is_available(), "no GPU visible"
dev = torch.device("cuda")
total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"GPU: {torch.cuda.get_device_name(0)} | {total_gb:.1f} GB | torch {torch.__version__}", flush=True)

train_path, _ = train_data_paths(args.data_dir, args.task, args.model, args.max_length)
train = load_from_disk(train_path)
lengths = pc.list_value_length(train.data.column("input_ids")).to_numpy()
order = lengths.argsort()[::-1]
print(f"{len(train)} training examples | longest {int(lengths[order[0]])} | mean {lengths.mean():.1f}", flush=True)

name = resolve_model(args.model)
dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16
adapter_dtype = torch.float32 if args.precision == "fp16" else torch.bfloat16
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
setup_pad_token(tok, model)
kw = {"num_blocks": args.size} if args.method == "diablo" else {"lora_r": args.size}
model = apply_adapter(model, args.method, parse_target_modules(args.modules), dropout=0.05, **kw)
model.to(dev)
_, trainable = mark_trainable(model, adapter_dtype)
model.enable_input_require_grads()
model.config.use_cache = False
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-6)
scaler = torch.amp.GradScaler("cuda", enabled=args.precision == "fp16")
print(f"{trainable / 1e6:.2f}M trainable", flush=True)


def longest_batch(bs):
    rows = [train[int(i)] for i in order[:bs]]
    width = max(len(r["input_ids"]) for r in rows)

    def left_pad(key, value):
        return torch.tensor([[value] * (width - len(r[key])) + list(r[key]) for r in rows], device=dev)

    return width, dict(input_ids=left_pad("input_ids", tok.pad_token_id),
                       attention_mask=left_pad("attention_mask", 0), labels=left_pad("labels", -100))


cands = [(False, 4), (False, 2), (True, 4), (True, 2), (True, 1)]
if args.only_gc:
    cands = [c for c in cands if c[0]]
cands = [(g, b) for g, b in cands if args.eff % b == 0]
chosen = None
for use_gc, bs in cands:
    if use_gc:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_disable()
    pygc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        width, batch = longest_batch(bs)
        with torch.autocast("cuda", dtype=dtype):
            loss = model(**batch).loss
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        peak = torch.cuda.max_memory_reserved() / 2**30
        ok = peak <= total_gb - args.headroom_gb
        print(f"PROBE gc={int(use_gc)} batch={bs} x {width} tokens: peak {peak:.2f} GB of {total_gb:.1f} -> "
              f"{'OK' if ok else 'TOO BIG'}", flush=True)
        del batch, loss
        if ok:
            chosen = dict(bs=bs, ga=args.eff // bs, gc=use_gc, peak_gb=round(peak, 2), total_gb=round(total_gb, 1))
            break
    except torch.OutOfMemoryError:
        opt.zero_grad(set_to_none=True)
        print(f"PROBE gc={int(use_gc)} batch={bs}: OUT OF MEMORY", flush=True)

if chosen is None:
    sys.exit("PROBE nothing fits on this GPU, not even batch 1 with gradient checkpointing")
print("PROBE chosen " + json.dumps(chosen))
json.dump(chosen, open(args.out, "w"), indent=2)
