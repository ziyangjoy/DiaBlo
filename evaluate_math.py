"""
Evaluate on GSM8K and MATH after MetaMathQA training (paper Table 2).

Decoding and scoring are the released script's: beam search (4 beams), 256 new tokens for GSM8K and 512 for
MATH; GSM8K = last number in the response (tolerance 1e-3), MATH = MetaMath's "The answer is: " extraction
and string normalisation (utils_math.py). Changes: DiaBlo/LoRA/base support with a strict adapter check,
resumable per-example JSONL, only generated tokens are scored, and the prompt is the exact training prefix
(--prompt_style upstream gives the released script's prompt, which is indented differently from training).

Test files:
    <data_dir>/dataset_math/gsm8k/test.json   (LLM-Adapters dataset/gsm8k/test.json, 1319 questions)
    <data_dir>/MATH_test.json                 (MetaMath data/test/MATH_test.jsonl as a JSON list, 5000 problems)

Usage:
    python evaluate_math.py --model_name Llama2-7B --method diablo --num_blocks 64 --precision bf16 \
        --adapter_path results/M1/adapter.chkpt --data_dir datasets --output_dir eval/M1
"""

import argparse
import json
import os
import re

from diablo.eval_utils import add_model_args, load_model_for_eval, run_resumable
from diablo.prompts import metamath_eval_prompt
from utils_math import last_boxed_only_string, process_results, remove_boxed

DATASETS = ["gsm8k", "MATH"]
MAX_NEW = {"gsm8k": 256, "MATH": 512}

parser = argparse.ArgumentParser()
add_model_args(parser)
parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
parser.add_argument("--data_dir", type=str, default="./datasets")
parser.add_argument("--output_dir", type=str, required=True, help="<name>.jsonl per dataset + summary.json")
parser.add_argument("--batch_size", type=int, default=8, help="Halved automatically on CUDA OOM")
parser.add_argument("--num_beams", type=int, default=4)
parser.add_argument("--prompt_style", type=str, default="train", choices=["train", "upstream"])
parser.add_argument("--limit", type=int, default=None, help="Only the first N examples per dataset (timing runs)")
args = parser.parse_args()


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(",", "")
    pred = re.findall(r"-?\d+\.?\d*", sentence)
    if not pred:
        return float("inf")
    try:
        return float(pred[-1])
    except ValueError:
        return float("inf")


def score_gsm8k(item, text):
    label = item["answer"]
    try:
        label = float(label)
    except (TypeError, ValueError):
        label = float("inf")
    pred = extract_answer_number(text)
    return {"pred": pred if pred != float("inf") else None, "correct": abs(label - pred) <= 0.001}


def score_math(item, text):
    label = remove_boxed(last_boxed_only_string(item["output"]))
    ok, pred = process_results(text, label)
    return {"pred": pred, "answer": label, "correct": bool(ok)}


def load_items(name):
    path = (os.path.join(args.data_dir, "MATH_test.json") if name == "MATH"
            else os.path.join(args.data_dir, "dataset_math", name, "test.json"))
    return json.load(open(path))


tokenizer, model, load_name = load_model_for_eval(args)
os.makedirs(args.output_dir, exist_ok=True)

summary = {"model": load_name, "method": args.method, "adapter": args.adapter_path, "num_blocks": args.num_blocks,
           "prompt_style": args.prompt_style, "num_beams": args.num_beams, "datasets": {}}
for name in args.datasets:
    items = load_items(name)
    n_total = len(items)
    if args.limit:
        items = items[: args.limit]
    records, timing = run_resumable(
        items, os.path.join(args.output_dir, f"{name}.jsonl"), args.batch_size,
        make_prompt=lambda it: metamath_eval_prompt(it["instruction"], args.prompt_style),
        score=score_math if name == "MATH" else score_gsm8k,
        model=model, tokenizer=tokenizer, desc=name,
        num_beams=args.num_beams, do_sample=False, max_new_tokens=MAX_NEW[name])
    acc = sum(r["correct"] for r in records) / max(len(records), 1)
    summary["datasets"][name] = {"n": len(records), "n_total": n_total, "accuracy": acc, **timing}
    print(f"{name}: {acc * 100:.2f} ({len(records)} examples)", flush=True)

complete = (len(summary["datasets"]) == len(DATASETS)
            and all(d["n"] == d["n_total"] for d in summary["datasets"].values()))
summary["average"] = sum(d["accuracy"] for d in summary["datasets"].values()) / len(summary["datasets"])
summary["complete"] = complete
print(json.dumps(summary, indent=2))
with open(os.path.join(args.output_dir, "summary.json" if complete else "summary_partial.json"), "w") as f:
    json.dump(summary, f, indent=2)
