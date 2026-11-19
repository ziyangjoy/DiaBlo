"""End-to-end pipeline test on a tiny random Qwen2 model (CPU or GPU, ~1-2 min).

train -> resume -> merge -> eval(adapter) vs eval(merged) for both DiaBlo and LoRA.

    python tests/e2e_tiny.py [--workdir /tmp/diablo_e2e] [--precision fp32|fp16]
"""

import argparse
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", default=os.path.join(REPO, "_e2e_tmp"))
ap.add_argument("--precision", default="fp32")
ap.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
a = ap.parse_args()

import torch
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

W = a.workdir
os.makedirs(W, exist_ok=True)
model_dir, data_dir, out_dir = (os.path.join(W, d) for d in ("tiny_model", "datasets", "results"))


def run(*cmd):
    print("\n$", " ".join(cmd), flush=True)
    r = subprocess.run([sys.executable, *cmd], cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-6000:])
        raise SystemExit(f"FAILED: {' '.join(cmd)}")
    return r.stdout + r.stderr


# 1. tiny model + real tokenizer
tok = AutoTokenizer.from_pretrained(a.tokenizer)
cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                  tie_word_embeddings=True)
torch.manual_seed(0)
Qwen2ForCausalLM(cfg).save_pretrained(model_dir)
tok.save_pretrained(model_dir)

# 2. tiny datasets in the layout process_gsm8k.py produces
qs = [f"Tom has {i} apples and buys {i + 2} more. How many apples?" for i in range(40)]
ans = [f"He has {i} + {i + 2} = {2 * i + 2}.\n#### {2 * i + 2}" for i in range(40)]
rows = {"input_ids": [], "attention_mask": [], "labels": []}
for q, s in zip(qs, ans):
    p = tok(q + "\nAnswer:\n", add_special_tokens=False)["input_ids"]
    r = tok(s.replace("####", "The final answer is: "), add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    rows["input_ids"].append(p + r)
    rows["attention_mask"].append([1] * (len(p) + len(r)))
    rows["labels"].append([-100] * len(p) + r)
Dataset.from_dict(rows).save_to_disk(os.path.join(data_dir, "GSM8k_processed/train"))
DatasetDict({"train": Dataset.from_dict({"question": qs[:4], "answer": ans[:4]}),
             "test": Dataset.from_dict({"question": qs[:6], "answer": ans[:6]})}
            ).save_to_disk(os.path.join(data_dir, "GSM8k_raw"))

common = ["--model_name", model_dir, "--data_dir", data_dir, "--output_dir", out_dir,
          "--precision", a.precision, "--per_device_batch_size", "2", "--gradient_accumulation_steps", "2",
          "--save_steps", "3", "--warmup_steps", "1", "--lr", "1e-2"]

for method, extra in [("diablo", ["--num_blocks", "4"]), ("lora", ["--lora_r", "2"])]:
    name = f"e2e_{method}"
    run_dir = os.path.join(out_dir, name)
    run("train.py", *common, "--method", method, *extra, "--run_name", name, "--max_steps", "3")
    log = run("train.py", *common, "--method", method, *extra, "--run_name", name, "--max_steps", "6", "--resume")
    assert "Resumed adapter weights" in log, "resume did not load adapter"
    state = json.load(open(os.path.join(run_dir, "log_history.json")))
    cfg_run = json.load(open(os.path.join(run_dir, "run_config.json")))
    print(f"[{method}] trainable params: {cfg_run['trainable_params']:,}; steps logged: {len(state)}")

    run("merge.py", "--run_dir", run_dir, "--output_dir", os.path.join(W, f"merged_{method}"), "--precision", "fp32")

    ev = ["evaluate_gsm8k.py", "--data_dir", data_dir, "--precision", "fp32", "--batch_size", "3",
          "--max_new_tokens", "12"]
    f_ad, f_mg = (os.path.join(W, f"eval_{method}_{k}.jsonl") for k in ("adapter", "merged"))
    for f in (f_ad, f_mg):
        if os.path.exists(f):
            os.remove(f)
    run(*ev, "--model_name", model_dir, "--method", method, *extra, "--adapter_path",
        os.path.join(run_dir, "adapter.chkpt"), "--output_file", f_ad)
    run(*ev, "--model_path", os.path.join(W, f"merged_{method}"), "--method", "none", "--output_file", f_mg)
    o_ad = [json.loads(l)["output"] for l in open(f_ad)]
    o_mg = [json.loads(l)["output"] for l in open(f_mg)]
    assert len(o_ad) == 6, len(o_ad)
    same = sum(x == y for x, y in zip(o_ad, o_mg))
    print(f"[{method}] adapter vs merged generations identical: {same}/6")
    assert same == 6, "merged model diverges from adapter model"

# 3. Slurm segments: a past deadline stops after one step on a saved checkpoint (exit 3), a half-written checkpoint
#    is removed, and --resume finishes the run.
import shutil
import time

name = "e2e_deadline"
rd = os.path.join(out_dir, name)
shutil.rmtree(rd, ignore_errors=True)
args_d = [*common, "--method", "diablo", "--num_blocks", "4", "--run_name", name, "--max_steps", "6"]
r = subprocess.run([sys.executable, "train.py", *args_d, "--deadline", str(time.time() - 1)], cwd=REPO,
                   capture_output=True, text=True)
assert r.returncode == 3, f"expected exit 3, got {r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}"
assert not os.path.exists(os.path.join(rd, "log_history.json")), "a stopped segment must not look finished"
assert sorted(d for d in os.listdir(rd) if d.startswith("checkpoint-")) == ["checkpoint-1"]
os.makedirs(os.path.join(rd, "checkpoint-2"))          # cut off mid-save: no trainer_state.json
log = run("train.py", *args_d, "--resume")
assert "Removing incomplete checkpoint" in log and "Resumed adapter weights" in log, log[-2000:]
assert json.load(open(os.path.join(rd, "train_stats.json")))["global_step"] == 6
print("[deadline] stop -> exit 3 -> cleanup -> resume OK")

# 4. Commonsense: process_commonsense -> train --task commonsense_new -> evaluate_commonsense (all 8, resumable)
cs = [{"instruction": f"Please answer the following question with true or false, question: is {i} even?\n\n"
                      "Answer format: true/false", "input": "",
       "output": f"the correct answer is {'true' if i % 2 == 0 else 'false'}",
       "answer": "true" if i % 2 == 0 else "false"} for i in range(130)]
json.dump(cs, open(os.path.join(W, "cs170k.json"), "w"))
run("data_processing/process_commonsense.py", "--data_path", os.path.join(W, "cs170k.json"), "--output_dir", data_dir,
    "--model_name", model_dir, "--val_size", "120", "--num_proc", "1")
shutil.rmtree(os.path.join(out_dir, "e2e_cs"), ignore_errors=True)
run("train.py", "--model_name", model_dir, "--data_dir", data_dir, "--output_dir", out_dir, "--task", "commonsense_new",
    "--max_length", "256", "--precision", a.precision, "--per_device_batch_size", "2", "--gradient_accumulation_steps", "2",
    "--warmup_steps", "1", "--lr", "1e-2", "--method", "diablo", "--num_blocks", "4", "--run_name", "e2e_cs",
    "--max_steps", "3")
shared = os.path.join(W, "shared")
answers = {"boolq": "true", "piqa": "solution1", "social_i_qa": "answer2", "hellaswag": "ending3",
           "winogrande": "option1", "ARC-Easy": "answer1", "ARC-Challenge": "answer4", "openbookqa": "answer3"}
for d, ans in answers.items():
    os.makedirs(os.path.join(shared, "dataset_commonsense", d), exist_ok=True)
    json.dump([{"instruction": f"{d} question {i}?", "input": "", "output": f"the correct answer is {ans}", "answer": ans}
               for i in range(3)], open(os.path.join(shared, "dataset_commonsense", d, "test.json"), "w"))
cs_adapter = ["--adapter_path", os.path.join(out_dir, "e2e_cs", "adapter.chkpt")]
ev_cs = ["--model_name", model_dir, "--method", "diablo", *cs_adapter, "--data_dir", shared, "--precision", "fp32",
         "--batch_size", "2", "--output_dir", os.path.join(W, "eval_cs")]
shutil.rmtree(os.path.join(W, "eval_cs"), ignore_errors=True)
run("evaluate_commonsense.py", *ev_cs, "--num_blocks", "4")
s = json.load(open(os.path.join(W, "eval_cs", "summary.json")))
assert s["complete"] and all(v["n"] == 3 for v in s["datasets"].values()), s
run("evaluate_commonsense.py", *ev_cs, "--num_blocks", "4")            # resume: nothing left to generate
s = json.load(open(os.path.join(W, "eval_cs", "summary.json")))
assert all(v["generated"] == 0 for v in s["datasets"].values()), s
bad = subprocess.run([sys.executable, "evaluate_commonsense.py", *ev_cs, "--num_blocks", "8",
                      "--output_dir", os.path.join(W, "eval_cs_bad")], cwd=REPO, capture_output=True, text=True)
assert bad.returncode != 0, "a wrong --num_blocks must fail the strict adapter check"
print(f"[commonsense] process -> train -> eval OK, avg {s['average']:.2f}; resume OK; wrong N rejected")

# 5. Math eval (GSM8K + MATH files in the LLM-Adapters / MetaMath formats), base model, both prompt styles
os.makedirs(os.path.join(shared, "dataset_math", "gsm8k"), exist_ok=True)
json.dump([{"instruction": f"What is {i} + {i}?", "input": "", "output": "", "answer": str(2 * i)} for i in range(3)],
          open(os.path.join(shared, "dataset_math", "gsm8k", "test.json"), "w"))
json.dump([{"instruction": f"Compute {i}*2.", "output": f"It is $\\boxed{{{2 * i}}}$."} for i in range(2)],
          open(os.path.join(shared, "MATH_test.json"), "w"))
for style in ("train", "upstream"):
    od = os.path.join(W, f"eval_math_{style}")
    shutil.rmtree(od, ignore_errors=True)
    run("evaluate_math.py", "--model_name", model_dir, "--method", "none", "--data_dir", shared, "--precision", "fp32",
        "--batch_size", "2", "--num_beams", "2", "--prompt_style", style, "--output_dir", od)
    s = json.load(open(os.path.join(od, "summary.json")))
    assert s["complete"] and s["datasets"]["gsm8k"]["n"] == 3 and s["datasets"]["MATH"]["n"] == 2, s
print("[math] evaluate_math OK (train and upstream prompts)")

print("\nE2E OK")
