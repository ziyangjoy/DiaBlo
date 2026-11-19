"""
Preprocess the Commonsense 170k dataset for DiaBlo training.

Requires: commonsense_170k.json (from https://github.com/AGI-Edgerunners/LLM-Adapters, ft-training_set/)

Like LLM-Adapters (val_set_size=120), 120 shuffled examples are held out, and the rest is saved as
  <output_dir>/commonsense_170k_new/train_val120_<max_length>_OnlyOutput_<model_name>
which is what `train.py --task commonsense_new` loads. Prompt/response tokenization and label masking
are unchanged from upstream (templates now live in diablo/prompts.py).

Usage:
    python data_processing/process_commonsense.py --data_path ./datasets/commonsense_170k.json \
        --output_dir ./datasets --model_name Llama2-7B
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow.compute as pc
from transformers import AutoTokenizer
from datasets import load_dataset

from diablo.common import model_tag, resolve_model
from diablo.prompts import commonsense_train_prompt

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to commonsense_170k.json")
parser.add_argument("--output_dir", type=str, default="./datasets")
parser.add_argument("--model_name", type=str, default="Llama2-7B", help="Short name (diablo/common.py) or HF id")
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--val_size", type=int, default=120, help="Held-out examples (LLM-Adapters uses 120)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_proc", type=int, default=4)
parser.add_argument("--model_cache_dir", type=str, default=None)
parser.add_argument("--hf_token", type=str, default=None)
args = parser.parse_args()

hf_token = args.hf_token or os.environ.get("HF_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(hf_token)

model_dir = args.model_cache_dir or os.environ.get("MODEL_CACHE_DIR", None)
tokenizer = AutoTokenizer.from_pretrained(resolve_model(args.model_name), cache_dir=model_dir)
cutoff_len = args.max_length


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
    if result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len and add_eos_token:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point):
    full = tokenize(commonsense_train_prompt(data_point["instruction"], data_point["input"], data_point["output"]))
    # Mask input tokens in labels (train only on output)
    user_len = len(tokenize(commonsense_train_prompt(data_point["instruction"], data_point["input"], ""),
                            add_eos_token=False)["input_ids"])
    full["labels"] = [-100] * user_len + full["labels"][user_len:]
    return full


# datasets >= 5 starts a worker pool for any num_proc >= 1; spawned workers (Windows/macOS) re-import this file,
# so the work below must sit under the __main__ guard, and num_proc=1 means "no pool".
num_proc = args.num_proc if args.num_proc > 1 else None

if __name__ == "__main__":
    data = load_dataset("json", data_files=args.data_path)["train"]
    split = data.train_test_split(test_size=args.val_size, shuffle=True, seed=args.seed)
    cols = data.column_names
    train_data = split["train"].shuffle(seed=args.seed).map(generate_and_tokenize_prompt, num_proc=num_proc,
                                                             remove_columns=cols)
    val_data = split["test"].map(generate_and_tokenize_prompt, remove_columns=cols)

    base = os.path.join(args.output_dir, "commonsense_170k_new")
    tag = f"{cutoff_len}_OnlyOutput_{model_tag(args.model_name)}"
    train_data.save_to_disk(os.path.join(base, f"train_val{args.val_size}_{tag}"))
    val_data.save_to_disk(os.path.join(base, f"val{args.val_size}_{tag}"))
    lens = pc.list_value_length(train_data.data.column("input_ids")).to_numpy()
    print(f"{len(train_data)} train / {len(val_data)} val examples | mean len {lens.mean():.1f} | "
          f"truncated at {cutoff_len}: {int((lens >= cutoff_len).sum())}")
    print(f"Saved to {base}/train_val{args.val_size}_{tag}")
