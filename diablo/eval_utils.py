"""Shared pieces of the generation-based evaluators (commonsense, math).

The model is built exactly like train.py builds it (same pad token, same adapter wrapper),
the adapter is loaded with a strict key check (upstream used strict=False, so a wrong
--num_blocks or --target_modules silently evaluated the base model), results are appended
per example to a JSONL file so a killed job resumes where it stopped, and a CUDA OOM halves the
batch instead of killing the job (beam search on a 7B model needs a lot of KV cache).
"""

import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .adapters import apply_adapter, load_adapter
from .common import parse_target_modules, resolve_model, setup_pad_token

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def add_model_args(parser):
    parser.add_argument("--model_name", type=str, default="Llama2-7B")
    parser.add_argument("--model_path", type=str, default=None, help="Local/merged model dir (overrides model_name)")
    parser.add_argument("--method", type=str, default="diablo", choices=["diablo", "lora", "none"])
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to adapter.chkpt")
    parser.add_argument("--num_blocks", type=int, default=64)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--target_modules", type=str, default="qkvud")
    parser.add_argument("--precision", type=str, default="bf16", choices=list(DTYPES))
    parser.add_argument("--model_cache_dir", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=None)


def load_model_for_eval(args):
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(hf_token)
    cache = args.model_cache_dir or os.environ.get("MODEL_CACHE_DIR", None)
    dtype = DTYPES[args.precision]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = args.model_path or resolve_model(args.model_name)

    # Load straight onto the GPU in the compute dtype (upstream loaded fp32 on CPU first: 27 GB of RAM for 7B).
    model = AutoModelForCausalLM.from_pretrained(name, cache_dir=cache, torch_dtype=dtype,
                                                 device_map={"": device} if device == "cuda" else None)
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache, padding_side="left")
    setup_pad_token(tokenizer, model)

    if args.method != "none":
        if not args.adapter_path:
            raise SystemExit("--adapter_path is required unless --method none")
        model = apply_adapter(model, args.method, parse_target_modules(args.target_modules),
                              num_blocks=args.num_blocks, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                              dropout=0.0)
        load_adapter(model, args.adapter_path)
    model.to(dtype).to(device)
    model.eval()
    return tokenizer, model, name


def load_jsonl(path):
    """Read a JSONL file, dropping a half-written last line left by a killed job, and rewrite it clean."""
    if not os.path.exists(path):
        return []
    records, clean = [], True
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                clean = False
    if not clean:
        with open(path, "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in records)
    return records


@torch.no_grad()
def generate_new_text(model, tokenizer, prompts, **gen_kwargs):
    """Generate for a batch of prompts and return only the newly generated text of each.

    Upstream decoded prompt+completion and then split on a marker. Decoding only the new tokens gives the
    same text whenever the marker is present, and cannot pick an answer out of the prompt when it is not
    (e.g. a BoolQ prompt contains 'true/false').
    """
    device = next(model.parameters()).device
    enc = tokenizer(prompts, return_tensors="pt", padding=True)
    out = model.generate(input_ids=enc.input_ids.to(device), attention_mask=enc.attention_mask.to(device),
                         pad_token_id=tokenizer.pad_token_id, **gen_kwargs)
    return tokenizer.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)


def run_resumable(items, out_file, batch_size, make_prompt, score, model, tokenizer, desc="eval", **gen_kwargs):
    """Evaluate `items` (list of dicts) in order, appending one JSON line per item to out_file.

    score(item, generated_text) -> dict with at least 'correct'.
    Returns (all records sorted by idx, {"seconds": generation time in this process, "generated": examples}).
    """
    import time
    from tqdm import tqdm

    os.makedirs(os.path.dirname(os.path.abspath(out_file)), exist_ok=True)
    done = {r["idx"] for r in load_jsonl(out_file)}
    todo = [i for i in range(len(items)) if i not in done]
    print(f"[{desc}] {len(done)} already evaluated, {len(todo)} to go (batch {batch_size})", flush=True)

    bs = batch_size
    pos = 0
    t0 = time.time()
    bar = tqdm(total=len(todo), desc=desc, mininterval=30)
    with open(out_file, "a") as fout:
        while pos < len(todo):
            idxs = todo[pos: pos + bs]
            try:
                texts = generate_new_text(model, tokenizer, [make_prompt(items[i]) for i in idxs], **gen_kwargs)
            except torch.OutOfMemoryError:
                if bs == 1:
                    raise
                bs = max(1, bs // 2)
                torch.cuda.empty_cache()
                print(f"[{desc}] CUDA OOM, retrying with batch {bs}", flush=True)
                continue
            for i, text in zip(idxs, texts):
                fout.write(json.dumps({"idx": i, "output": text, **score(items[i], text)}) + "\n")
            fout.flush()
            pos += len(idxs)
            bar.update(len(idxs))
    bar.close()
    timing = {"seconds": round(time.time() - t0, 1), "generated": len(todo)}
    return sorted(load_jsonl(out_file), key=lambda r: r["idx"]), timing
