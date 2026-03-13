"""
Preprocess the GSM8k dataset for DiaBlo training.

Tokenizes prompt and response separately, masks prompt tokens in labels.
Follows the LoRI processing pipeline.

Usage:
    python data_processing/process_gsm8k.py --output_dir ./datasets --model_name meta-llama/Meta-Llama-3-8B
"""

import argparse
import os
import random

import numpy as np
import torch
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", type=str, default="./datasets")
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf")
parser.add_argument("--max_length", type=int, default=512)
parser.add_argument("--max_prompt_length", type=int, default=256)
parser.add_argument("--model_cache_dir", type=str, default=None)
parser.add_argument("--hf_token", type=str, default=None)
parser.add_argument("--seed", type=int, default=0, help="Random seed for shuffling training data")
args = parser.parse_args()

hf_token = args.hf_token or os.environ.get("HF_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(hf_token)

model_dir = args.model_cache_dir or os.environ.get("MODEL_CACHE_DIR", None)
data_cache = os.environ.get("DATA_CACHE_DIR", None)

tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=model_dir)
tokenizer.add_special_tokens({'pad_token': '<PAD>'})

QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"
ANSWER_PROMPT = "The final answer is: "


def tokenize_example(example):
    prompt = example["question"] + QUESTION_PROMPT
    response = example["answer"].replace("####", ANSWER_PROMPT)

    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    response_tokens = tokenizer(response, add_special_tokens=False)

    # Append EOS token to response
    response_tokens['input_ids'].append(tokenizer.eos_token_id)
    response_tokens['attention_mask'].append(1)

    # Truncate prompt if combined is too long
    if len(prompt_tokens['input_ids']) + len(response_tokens['input_ids']) > args.max_length:
        prompt_tokens = {k: v[:args.max_prompt_length] for k, v in prompt_tokens.items()}

    # Truncate response if still too long
    if len(prompt_tokens['input_ids']) + len(response_tokens['input_ids']) > args.max_length:
        response_tokens = {k: v[:args.max_length - len(prompt_tokens['input_ids'])] for k, v in response_tokens.items()}

    # Concatenate prompt + response
    input_ids = prompt_tokens['input_ids'] + response_tokens['input_ids']
    attention_mask = prompt_tokens['attention_mask'] + response_tokens['attention_mask']

    # Labels: mask prompt tokens with -100
    labels = [-100] * len(prompt_tokens['input_ids']) + response_tokens['input_ids']

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
    }


# Load and process dataset
dataset = load_dataset("openai/gsm8k", "main", cache_dir=data_cache)

for split in ['train', 'test']:
    examples = list(dataset[split])

    # Shuffle training data with deterministic seed (matches LoRI pipeline)
    if split == 'train':
        np.random.seed(args.seed)
        shuffle_seed = int(np.random.randint(0, 2**32))
        random.seed(shuffle_seed)
        random.shuffle(examples)
        print(f"Shuffled {len(examples)} training examples with seed={args.seed}")

    data = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for example in tqdm(examples, desc=f'Processing {split}'):
        tokens = tokenize_example(example)
        data['input_ids'].append(tokens['input_ids'])
        data['attention_mask'].append(tokens['attention_mask'])
        data['labels'].append(tokens['labels'])

    hf_dataset = Dataset.from_dict(data)
    print(f"{split}: {len(hf_dataset)} examples")

    save_path = os.path.join(args.output_dir, f"GSM8k_processed/{split}")
    hf_dataset.save_to_disk(save_path)
    print(f"Saved to {save_path}")

# Also save original dataset for evaluation (need question/answer text)
dataset.save_to_disk(os.path.join(args.output_dir, "GSM8k_raw"))
print("Raw dataset saved for evaluation.")
