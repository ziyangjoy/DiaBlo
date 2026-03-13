"""
Preprocess the Commonsense 170k dataset for DiaBlo training.

Requires: commonsense_170k.json (from https://github.com/AGI-Edgerunners/LLM-Adapters)

Usage:
    python data_processing/process_commonsense.py --data_path ./datasets/commonsense_170k.json \
        --output_dir ./datasets --model_name Llama2-7B
"""

import argparse
import os

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to commonsense_170k.json")
parser.add_argument("--output_dir", type=str, default="./datasets")
parser.add_argument("--model_name", type=str, default="Llama2-7B")
parser.add_argument("--max_length", type=int, default=256)
parser.add_argument("--model_cache_dir", type=str, default=None)
parser.add_argument("--hf_token", type=str, default=None)
args = parser.parse_args()

hf_token = args.hf_token or os.environ.get("HF_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(hf_token)

model_dir = args.model_cache_dir or os.environ.get("MODEL_CACHE_DIR", None)

MODEL_MAP = {
    "Llama2-7B": "meta-llama/Llama-2-7b-hf",
    "Llama2-13B": "meta-llama/Llama-2-13b-hf",
    "Llama3-8B": "meta-llama/Meta-Llama-3-8B",
    "Llama3-3B": "meta-llama/Llama-3.2-3B",
}
load_name = MODEL_MAP[args.model_name]

tokenizer = AutoTokenizer.from_pretrained(load_name, cache_dir=model_dir)
tokenizer.pad_token_id = 0
tokenizer.padding_side = "left"

cutoff_len = args.max_length


def generate_prompt(data_point):
    if data_point["input"]:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {data_point["instruction"]}

                ### Input:
                {data_point["input"]}

                ### Response:
                {data_point["output"]}"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                {data_point["instruction"]}

                ### Response:
                {data_point["output"]}"""


def tokenize(prompt, add_eos_token=True):
    result = tokenizer(prompt, truncation=True, max_length=cutoff_len, padding=False, return_tensors=None)
    if result["input_ids"][-1] != tokenizer.eos_token_id and len(result["input_ids"]) < cutoff_len and add_eos_token:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
    result["labels"] = result["input_ids"].copy()
    return result


def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)

    # Mask input tokens in labels (train only on output)
    user_prompt = generate_prompt({**data_point, "output": ""})
    tokenized_user_prompt = tokenize(user_prompt, add_eos_token=False)
    user_prompt_len = len(tokenized_user_prompt["input_ids"])
    tokenized_full_prompt["labels"] = [-100] * user_prompt_len + tokenized_full_prompt["labels"][user_prompt_len:]

    return {key: torch.tensor(val) for key, val in tokenized_full_prompt.items()}


data = load_dataset("json", data_files=args.data_path)
train_data = data["train"].shuffle().map(generate_and_tokenize_prompt, num_proc=32)

save_path = os.path.join(args.output_dir, f"commonsense_170k_new/train_all_{cutoff_len}_OnlyOutput_{args.model_name}")
train_data.save_to_disk(save_path)
print(f"Dataset saved to {save_path}")
