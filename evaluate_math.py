"""
Evaluate DiaBlo model on math reasoning benchmarks (GSM8k, MATH, etc.).

Usage:
    python evaluate_math.py --model_name Llama2-7B --dataset gsm8k \
        --adapter_path results/.../adapter.chkpt --output_dir eval_results/
"""

import copy
import json
import os
import re
import argparse

import torch
from tqdm import tqdm
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer

from diablo import replace_linear_with_blocklinear
from utils_math import process_results, remove_boxed, last_boxed_only_string

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, help="Dataset name (e.g. gsm8k, MATH)")
parser.add_argument("--model_name", type=str, default="Llama2-7B")
parser.add_argument("--adapter_path", type=str, default=None, help="Path to adapter.chkpt (None for zero-shot)")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--data_dir", type=str, default="./datasets", help="Dataset directory")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_blocks", type=int, default=64)
parser.add_argument("--target_modules", type=str, default="qkvud")
parser.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
parser.add_argument("--model_cache_dir", type=str, default=None)
parser.add_argument("--hf_token", type=str, default=None)
args = parser.parse_args()

# Setup
hf_token = args.hf_token or os.environ.get("HF_TOKEN")
if hf_token:
    from huggingface_hub import login
    login(hf_token)

model_dir = args.model_cache_dir or os.environ.get("MODEL_CACHE_DIR", None)

MODEL_MAP = {
    "Llama2-7B": "meta-llama/Llama-2-7b-hf",
    "Llama2-13B": "meta-llama/Llama-2-13b-hf",
    "Llama3-8B": "meta-llama/Meta-Llama-3-8B",
}
load_name = MODEL_MAP[args.model_name]

if args.precision == "bf16":
    load_precision = torch.bfloat16
elif args.precision == "fp16":
    load_precision = torch.float32  # load fp32, compute fp16
else:
    load_precision = torch.float32

compute_precision = torch.bfloat16 if args.precision == "bf16" else torch.float16

MODULE_MAP = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "u": "up_proj", "d": "down_proj", "o": "o_proj", "g": "gate_proj"}
target_modules_list = [MODULE_MAP[c] for c in args.target_modules if c in MODULE_MAP]


def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Input:
                {input}

                ### Response: Let's think step by step.
                """
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

                ### Instruction:
                {instruction}

                ### Response: Let's think step by step.
                """


def load_data():
    if args.dataset == "MATH":
        file_path = os.path.join(args.data_dir, "MATH_test.json")
    else:
        file_path = os.path.join(args.data_dir, f"dataset_math/{args.dataset}/test.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find dataset file: {file_path}")
    return json.load(open(file_path, "r"))


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(load_name, cache_dir=model_dir)
    tokenizer.padding_side = "left"
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    model = AutoModelForCausalLM.from_pretrained(load_name, cache_dir=model_dir, torch_dtype=load_precision)

    if args.num_blocks > 0:
        replace_linear_with_blocklinear(
            model.model,
            num_blocks=args.num_blocks,
            target_modules=target_modules_list,
        )

    model.resize_token_embeddings(len(tokenizer))

    if args.adapter_path:
        model.load_state_dict(torch.load(args.adapter_path, map_location="cpu"), strict=False)

    model.to(compute_precision).to(device)
    return tokenizer, model


def extract_answer_number(sentence: str) -> float:
    sentence = sentence.replace(",", "")
    pred = [s for s in re.findall(r"-?\d+\.?\d*", sentence)]
    if not pred:
        return float("inf")
    try:
        return float(pred[-1])
    except ValueError:
        return float("inf")


def main():
    dataset = load_data()
    tokenizer, model = load_model()
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    save_file = os.path.join(args.output_dir, f"{args.dataset}.json")

    # Create batches
    batches = [dataset[i : i + args.batch_size] for i in range(0, len(dataset), args.batch_size)]

    correct = 0
    current = 0
    output_data = []
    miss = 0.001

    for idx, batch in enumerate(tqdm(batches, desc=f"Evaluating {args.dataset}")):
        current += len(batch)
        instructions = [data.get("instruction") for data in batch]

        max_new_tokens = 512 if args.dataset == "MATH" else 256
        prompts = [generate_prompt(inst) for inst in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)

        generation_config = GenerationConfig(temperature=0.1, top_p=0.75, top_k=40, num_beams=4, do_sample=False)
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        outputs = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
        outputs = [o.split("### Response:")[-1].strip() for o in outputs]

        for data, output in zip(batch, outputs):
            if args.dataset != "MATH":
                label = data.get("answer")
                if isinstance(label, str):
                    try:
                        label = float(label)
                    except ValueError:
                        label = float("inf")
                predict = extract_answer_number(output)
                flag = abs(label - predict) <= miss
            else:
                label = data.get("output")
                label = remove_boxed(last_boxed_only_string(label))
                flag, predict = process_results(output, label)

            if flag:
                correct += 1

            new_data = copy.deepcopy(data)
            new_data["output_pred"] = output
            new_data["pred"] = predict
            new_data["flag"] = flag
            output_data.append(new_data)

        print(f"  {idx + 1}/{len(batches)} | accuracy: {correct}/{current} = {correct / current:.4f}")

        with open(save_file, "w") as f:
            json.dump(output_data, f, indent=4)

    print(f"\nFinal accuracy: {correct}/{current} = {correct / current:.4f}")


if __name__ == "__main__":
    main()
