"""
Preprocess the MetaMathQA dataset (395k) for DiaBlo training.

Downloads from HuggingFace: meta-math/MetaMathQA. Saves <output_dir>/metamath/train_all_<max_length>_OnlyOutput,
which `train.py --task metamath` loads. The tokenizer is model-specific, so use one output_dir per model.
Prompt/response tokenization and label masking are unchanged from upstream (template in diablo/prompts.py).

Usage:
    python data_processing/process_metamath.py --output_dir ./datasets/Llama2-7B --model_name Llama2-7B
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow.compute as pc
from transformers import AutoTokenizer
from datasets import load_dataset

from diablo.common import resolve_model
from diablo.prompts import metamath_train_prompt

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="./datasets")
parser.add_argument("--model_name", type=str, default="Llama2-7B", help="Short name (diablo/common.py) or HF id")
parser.add_argument("--max_length", type=int, default=2048)
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
data_cache = os.environ.get("DATA_CACHE_DIR", None)
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
    full = tokenize(metamath_train_prompt(data_point["query"], data_point["response"]))
    # Mask input tokens in labels (train only on output)
    user_len = len(tokenize(metamath_train_prompt(data_point["query"], ""), add_eos_token=False)["input_ids"])
    full["labels"] = [-100] * user_len + full["labels"][user_len:]
    return full


# datasets >= 5 starts a worker pool for any num_proc >= 1; spawned workers (Windows/macOS) re-import this file,
# so the work below must sit under the __main__ guard, and num_proc=1 means "no pool".
num_proc = args.num_proc if args.num_proc > 1 else None

if __name__ == "__main__":
    data = load_dataset("meta-math/MetaMathQA", cache_dir=data_cache)["train"]
    train_data = data.shuffle(seed=args.seed).map(generate_and_tokenize_prompt, num_proc=num_proc,
                                                  remove_columns=data.column_names)

    save_path = os.path.join(args.output_dir, f"metamath/train_all_{cutoff_len}_OnlyOutput")
    train_data.save_to_disk(save_path)
    lens = pc.list_value_length(train_data.data.column("input_ids")).to_numpy()
    print(f"{len(train_data)} examples | mean len {lens.mean():.1f} | truncated: {int((lens >= cutoff_len).sum())}")
    print(f"Dataset saved to {save_path}")
