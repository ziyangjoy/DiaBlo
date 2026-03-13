"""
Merge DiaBlo adapter weights back into the base model.

Usage:
    python merge.py --model_name Llama2-7B --adapter_path results/.../adapter.chkpt --output_dir merged_model/
"""

import os
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from diablo import replace_linear_with_blocklinear, replace_blocklinear_with_linear

parser = argparse.ArgumentParser(description="Merge DiaBlo adapter into base model")
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--adapter_path", type=str, required=True, help="Path to adapter.chkpt file")
parser.add_argument("--output_dir", type=str, required=True, help="Directory to save merged model")
parser.add_argument("--num_blocks", type=int, default=64)
parser.add_argument("--target_modules", type=str, default="qkvud")
parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
parser.add_argument("--model_cache_dir", type=str, default=None)
parser.add_argument("--hf_token", type=str, default=None)
args = parser.parse_args()

# HuggingFace login
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
    "Mistral-7B": "mistralai/Mistral-7B-v0.1",
    "Qwen2.5-7B": "Qwen/Qwen2.5-7B",
}
load_name = MODEL_MAP[args.model_name]

if args.precision == "bf16":
    compute_precision = torch.bfloat16
elif args.precision == "fp16":
    compute_precision = torch.float16
else:
    compute_precision = torch.float32

MODULE_MAP = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "u": "up_proj", "d": "down_proj", "o": "o_proj", "g": "gate_proj"}
target_modules_list = [MODULE_MAP[c] for c in args.target_modules if c in MODULE_MAP]

# Load base model
model = AutoModelForCausalLM.from_pretrained(load_name, cache_dir=model_dir, torch_dtype=compute_precision)
tokenizer = AutoTokenizer.from_pretrained(load_name, cache_dir=model_dir)

# Add BlockLinear layers
replace_linear_with_blocklinear(
    model.model,
    num_blocks=args.num_blocks,
    target_modules=target_modules_list,
)

# Load adapter weights
model.load_state_dict(torch.load(args.adapter_path, map_location="cpu"), strict=False)

# Merge and replace with standard Linear
replace_blocklinear_with_linear(model.model)

# Save
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(f"Merged model saved to {args.output_dir}")
