"""
Evaluate DiaBlo model on commonsense reasoning benchmarks.

Supported datasets: boolq, piqa, social_i_qa, hellaswag, winogrande, ARC-Challenge, ARC-Easy, openbookqa

Usage:
    python evaluate_commonsense.py --model_name Llama2-7B --dataset boolq \
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

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True,
                     choices=["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa"])
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
    "Llama3-3B": "meta-llama/Llama-3.2-3B",
}
load_name = MODEL_MAP[args.model_name]

if args.precision == "bf16":
    load_precision = torch.bfloat16
elif args.precision == "fp16":
    load_precision = torch.float32
else:
    load_precision = torch.float32

compute_precision = torch.bfloat16 if args.precision == "bf16" else torch.float16

MODULE_MAP = {"q": "q_proj", "k": "k_proj", "v": "v_proj", "u": "up_proj", "d": "down_proj", "o": "o_proj", "g": "gate_proj"}
target_modules_list = [MODULE_MAP[c] for c in args.target_modules if c in MODULE_MAP]


def generate_prompt(instruction, input=None):
    if input:
        return f"""{instruction}\n{input}"""
    else:
        return f"""{instruction}\n"""


def load_data():
    file_path = os.path.join(args.data_dir, f"dataset_commonsense/{args.dataset}/test.json")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find dataset file: {file_path}")
    return json.load(open(file_path, "r"))


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(load_name, cache_dir=model_dir)
    tokenizer.padding_side = "left"
    tokenizer.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained(load_name, cache_dir=model_dir, torch_dtype=load_precision)

    if args.num_blocks > 0:
        replace_linear_with_blocklinear(
            model.model,
            num_blocks=args.num_blocks,
            target_modules=target_modules_list,
        )

    if args.adapter_path:
        D = torch.load(args.adapter_path, map_location="cpu")
        # Remove score parameters if present in old checkpoints
        D = {k: v for k, v in D.items() if "scores" not in k}
        model.load_state_dict(D, strict=False)

    model.to(compute_precision).to(device)
    return tokenizer, model


def extract_answer(sentence: str) -> str:
    sentence_ = sentence.strip()
    if args.dataset == "boolq":
        pred = re.findall(r"true|false", sentence_)
    elif args.dataset == "piqa":
        pred = re.findall(r"solution1|solution2", sentence_)
    elif args.dataset in ["social_i_qa", "ARC-Challenge", "ARC-Easy", "openbookqa"]:
        pred = re.findall(r"answer1|answer2|answer3|answer4|answer5", sentence_)
    elif args.dataset == "hellaswag":
        pred = re.findall(r"ending1|ending2|ending3|ending4", sentence_)
    elif args.dataset == "winogrande":
        pred = re.findall(r"option1|option2", sentence_)
    else:
        pred = []
    return pred[0] if pred else ""


def main():
    dataset = load_data()
    tokenizer, model = load_model()
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    save_file = os.path.join(args.output_dir, f"{args.dataset}.json")

    batches = [dataset[i : i + args.batch_size] for i in range(0, len(dataset), args.batch_size)]

    correct = 0
    current = 0
    output_data = []

    for idx, batch in enumerate(tqdm(batches, desc=f"Evaluating {args.dataset}")):
        current += len(batch)
        instructions = [data.get("instruction") for data in batch]
        prompts = [generate_prompt(inst) for inst in instructions]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True)

        generation_config = GenerationConfig(
            temperature=0.1, top_p=0.75, top_k=40, num_beams=4, do_sample=False
        )
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=32,
            )

        outputs = tokenizer.batch_decode(generation_output.sequences, skip_special_tokens=True)
        outputs = [o.split("the correct answer is")[-1].strip() for o in outputs]

        for data, output in zip(batch, outputs):
            label = data.get("answer")
            predict = extract_answer(output)
            flag = label == predict
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
