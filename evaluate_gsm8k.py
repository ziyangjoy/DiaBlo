"""
Evaluate DiaBlo model on the GSM8k benchmark.

Usage:
    python evaluate_gsm8k.py --model_name Llama2-7B --num_blocks 64 --adapter_path results/.../adapter.chkpt
"""

import os
import math
import argparse

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_from_disk

from diablo import replace_linear_with_blocklinear

parser = argparse.ArgumentParser(description="Evaluate DiaBlo on GSM8k")
parser.add_argument("--model_name", type=str, default="Llama2-7B")
parser.add_argument("--adapter_path", type=str, required=True, help="Path to adapter.chkpt")
parser.add_argument("--data_dir", type=str, default="./datasets", help="Dataset directory")
parser.add_argument("--num_blocks", type=int, default=64)
parser.add_argument("--target_modules", type=str, default="qkvud")
parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--model_cache_dir", type=str, default=None)
parser.add_argument("--hf_token", type=str, default=None)
args = parser.parse_args()

# Setup
hf_token = args.hf_token or os.environ.get("HF_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(hf_token)

model_dir = args.model_cache_dir or os.environ.get("MODEL_CACHE_DIR", None)
device = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_MAP = {
    "Llama2-7B": "meta-llama/Llama-2-7b-hf",
    "Llama2-13B": "meta-llama/Llama-2-13b-hf",
    "Llama3-8B": "meta-llama/Meta-Llama-3-8B",
    "Llama3-3B": "meta-llama/Llama-3.2-3B",
    "Mistral-7B": "mistralai/Mistral-7B-v0.1",
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

# Load model
model = AutoModelForCausalLM.from_pretrained(load_name, cache_dir=model_dir, torch_dtype=compute_precision)
tokenizer = AutoTokenizer.from_pretrained(load_name, cache_dir=model_dir, padding_side="left", use_fast=False)
tokenizer.add_special_tokens({'pad_token': '<PAD>'})
model.resize_token_embeddings(len(tokenizer))

replace_linear_with_blocklinear(
    model.model,
    num_blocks=args.num_blocks,
    target_modules=target_modules_list,
)

model.load_state_dict(torch.load(args.adapter_path, map_location="cpu"), strict=False)
model.to(compute_precision).to(device)
model.eval()
for m in model.modules():
    if isinstance(m, torch.nn.Dropout):
        m.p = 0.0

# Load dataset
dataset = load_from_disk(os.path.join(args.data_dir, "GSM8k_raw"))
test_data = list(dataset["test"])

QUESTION_PROMPT = "\nAnswer the above question. First think step by step and then answer the final number.\n"

gen_kwargs = {
    "max_new_tokens": 512,
    "do_sample": False,
    "pad_token_id": model.config.pad_token_id,
}


ANSWER_PROMPT = "The final answer is: "


def extract_answer_gsm8k(sentence):
    """Extract answer from generated text. Try 'The final answer is:' first, then last number."""
    import re
    sentence = sentence.replace(',', '')
    pred = [s for s in re.findall(r'-?\d+\.?\d*', sentence)]
    if not pred:
        return float('inf')
    segment = sentence.split(ANSWER_PROMPT)
    if len(segment) > 1:
        pred_answer = segment[1]
        pred_answer = [s for s in re.findall(r'-?\d+\.?\d*', pred_answer)]
        if len(pred_answer) > 0:
            pred_answer = pred_answer[0]
        else:
            pred_answer = str(pred[-1])
    else:
        pred_answer = str(pred[-1])
    try:
        return float(pred_answer)
    except ValueError:
        return float('inf')


def evaluate(answers, preds):
    correct = 0
    for i in range(len(preds)):
        n_target = float(answers[i]["answer"].split("####")[-1].replace(",", ""))
        n_pred = extract_answer_gsm8k(preds[i])
        if n_target == n_pred:
            correct += 1
    return correct / len(preds)


outputs = []
batch = args.batch_size
# Drop last partial batch to match evaluate_lori.py behavior (uses floor, not ceil)
eval_steps = math.floor(len(dataset["test"]) / batch)
n_eval = eval_steps * batch

with torch.no_grad():
    for i in tqdm(range(eval_steps), desc="Evaluating GSM8k"):
        prompts = []
        for k in range(batch):
            idx = i * batch + k
            prompts.append(test_data[idx]["question"] + QUESTION_PROMPT)

        inputs = tokenizer(prompts, return_tensors="pt", padding="longest", add_special_tokens=False)
        gen_kwargs["input_ids"] = inputs.input_ids.to(device)
        gen_kwargs["attention_mask"] = inputs.attention_mask.to(device)

        generate_ids = model.generate(**gen_kwargs)
        result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True)
        outputs.extend(result)

        acc = evaluate(test_data[:len(outputs)], outputs)
        tqdm.write(f"Running accuracy: {acc:.4f} ({int(round(acc * len(outputs)))}/{len(outputs)})")

final_acc = evaluate(test_data[:n_eval], outputs)
correct = sum(1 for i in range(n_eval) if evaluate([test_data[i]], [outputs[i]]) > 0)
print(f"\nFinal accuracy: {final_acc:.4f} ({correct}/{n_eval})")
