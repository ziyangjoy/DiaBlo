"""
Evaluate on the 8 commonsense reasoning benchmarks of LLM-Adapters (paper Table 1 / Table 6).

Decoding and scoring are the released script's: beam search (4 beams), 32 new tokens, the first
"true|false" / "solution1|2" / "answer1-5" / "ending1-4" / "option1|2" after "the correct answer is".
Changes: one model load for all datasets, DiaBlo/LoRA/base support with a strict adapter check,
resumable per-example JSONL, only generated tokens are scored, and the prompt is the training template
(--prompt_style upstream gives the released script's bare "<instruction>\\n" prompt).

Test files: <data_dir>/dataset_commonsense/<name>/test.json (LLM-Adapters dataset/<name>/test.json).

Usage:
    python evaluate_commonsense.py --model_name Llama2-7B --method diablo --num_blocks 64 --precision fp16 \
        --adapter_path results/C1/adapter.chkpt --data_dir datasets --output_dir eval/C1
"""

import argparse
import json
import os
import re

from diablo.eval_utils import add_model_args, load_model_for_eval, run_resumable
from diablo.prompts import commonsense_eval_prompt

DATASETS = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"]
PATTERNS = {
    "boolq": r"true|false",
    "piqa": r"solution1|solution2",
    "social_i_qa": r"answer1|answer2|answer3|answer4|answer5",
    "ARC-Challenge": r"answer1|answer2|answer3|answer4|answer5",
    "ARC-Easy": r"answer1|answer2|answer3|answer4|answer5",
    "openbookqa": r"answer1|answer2|answer3|answer4|answer5",
    "hellaswag": r"ending1|ending2|ending3|ending4",
    "winogrande": r"option1|option2",
}

parser = argparse.ArgumentParser()
add_model_args(parser)
parser.add_argument("--datasets", nargs="+", default=DATASETS, choices=DATASETS)
parser.add_argument("--data_dir", type=str, default="./datasets")
parser.add_argument("--output_dir", type=str, required=True, help="<name>.jsonl per dataset + summary.json")
parser.add_argument("--batch_size", type=int, default=16, help="Halved automatically on CUDA OOM")
parser.add_argument("--num_beams", type=int, default=4)
parser.add_argument("--max_new_tokens", type=int, default=32)
parser.add_argument("--prompt_style", type=str, default="train", choices=["train", "upstream"])
parser.add_argument("--limit", type=int, default=None, help="Only the first N examples per dataset (timing runs)")
args = parser.parse_args()


def extract_answer(dataset, text):
    text = text.split("the correct answer is")[-1].strip()
    pred = re.findall(PATTERNS[dataset], text)
    return pred[0] if pred else ""


tokenizer, model, load_name = load_model_for_eval(args)
gen = dict(num_beams=args.num_beams, do_sample=False, max_new_tokens=args.max_new_tokens)
os.makedirs(args.output_dir, exist_ok=True)

summary = {"model": load_name, "method": args.method, "adapter": args.adapter_path, "num_blocks": args.num_blocks,
           "prompt_style": args.prompt_style, "num_beams": args.num_beams, "datasets": {}}
for name in args.datasets:
    items = json.load(open(os.path.join(args.data_dir, "dataset_commonsense", name, "test.json")))
    n_total = len(items)
    if args.limit:
        items = items[: args.limit]
    records, timing = run_resumable(
        items, os.path.join(args.output_dir, f"{name}.jsonl"), args.batch_size,
        make_prompt=lambda it: commonsense_eval_prompt(it["instruction"], args.prompt_style),
        score=lambda it, text, name=name: {"pred": extract_answer(name, text), "answer": it["answer"],
                                           "correct": extract_answer(name, text) == it["answer"]},
        model=model, tokenizer=tokenizer, desc=name, **gen)
    acc = sum(r["correct"] for r in records) / max(len(records), 1)
    summary["datasets"][name] = {"n": len(records), "n_total": n_total, "accuracy": acc, **timing}
    print(f"{name}: {acc * 100:.2f} ({len(records)} examples)", flush=True)

complete = all(d["n"] == d["n_total"] for d in summary["datasets"].values()) and len(summary["datasets"]) == len(DATASETS)
accs = [d["accuracy"] for d in summary["datasets"].values()]
summary["average"] = sum(accs) / len(accs)
summary["complete"] = complete
print(json.dumps(summary, indent=2))
# summary.json is the "finished" marker that the Slurm scripts look for, so write it only for a full run.
fname = "summary.json" if complete else "summary_partial.json"
with open(os.path.join(args.output_dir, fname), "w") as f:
    json.dump(summary, f, indent=2)
