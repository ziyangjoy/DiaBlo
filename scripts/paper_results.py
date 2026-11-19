"""Collect the paper-scale runs into paper-vs-ours tables (markdown), with progress for unfinished runs.

    python scripts/paper_results.py                 # on deep21, from the repo root (reads ~/diablo)
    python scripts/paper_results.py --work /path/to/copy/of/diablo > paper_results.md
"""

import argparse
import glob
import json
import os
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("--work", default=os.environ.get("DIABLO_WORK", os.path.expanduser("~/diablo")))
args = ap.parse_args()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dump = subprocess.run([os.environ.get("DIABLO_BASH", "bash"), "-c", 'source scripts/slurm/runs.sh && dump_runs "${ALL_PAPER[@]}"'],
                      cwd=REPO, capture_output=True, text=True, check=True).stdout
RUNS = {}
for line in dump.splitlines():
    if not line.strip():
        continue
    rid, model, task, method, size, lr, ev, paper, paper_num = (line.split("\t") + [""] * 9)[:9]
    RUNS[rid] = dict(model=model, task=task, method=method, size=size, lr=lr, eval=ev, paper=paper,
                     paper_num=float(paper_num) if paper_num else None)


def load(path):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return None


def status(rid):
    """'done', or a short progress string for unfinished runs."""
    r = RUNS[rid]
    res, ev = f"{args.work}/results-{r['model']}/{rid}", f"{args.work}/eval-{r['model']}"
    final = load(f"{ev}/{rid}_hf_summary.json") if r["eval"] == "gsm8k_lori" else load(f"{ev}/{rid}/summary.json")
    if final:
        return "done", final
    if r["method"] != "none" and not os.path.exists(f"{res}/log_history.json"):
        ckpts = sorted(glob.glob(f"{res}/checkpoint-*/trainer_state.json"), key=os.path.getmtime)
        if not ckpts:
            return "not started", None
        st = load(ckpts[-1])
        return f"training {st['global_step']}/{st['max_steps']} ({100 * st['global_step'] / st['max_steps']:.0f}%)", None
    started = glob.glob(f"{ev}/{rid}/*.jsonl") + glob.glob(f"{ev}/{rid}_hf.jsonl")
    if r["method"] == "none" and not started:
        return "not started", None
    n = 0
    for p in started:
        with open(p) as f:
            n += sum(1 for line in f if line.strip())
    return f"evaluating ({n} examples so far)", None


def pct(x):
    return f"{100 * x:.2f}"


CS = ["boolq", "piqa", "social_i_qa", "hellaswag", "winogrande", "ARC-Easy", "ARC-Challenge", "openbookqa"]
CS_HEAD = ["BoolQ", "PIQA", "SIQA", "HellaS.", "WinoG.", "ARC-e", "ARC-c", "OBQA"]


def table(title, ids, kind):
    print(f"\n### {title}\n")
    if kind == "commonsense":
        print("| Run | Model | Method | LR | " + " | ".join(CS_HEAD) + " | Avg | Paper |")
        print("|---" * (len(CS_HEAD) + 6) + "|")
    elif kind == "math":
        print("| Run | Model | Method | LR | GSM8K | MATH | Avg | Paper |\n|---|---|---|---|---|---|---|---|")
    else:
        print("| Run | Model | Method | LR | GSM8K acc | Paper |\n|---|---|---|---|---|---|")
    for rid in ids:
        r = RUNS[rid]
        meth = "base" if r["method"] == "none" else (f"DiaBlo N={r['size']}" if r["method"] == "diablo" else f"LoRA r={r['size']}")
        st, s = status(rid)
        head = f"| {rid} | {r['model']} | {meth} | {r['lr'] if r['method'] != 'none' else '-'} |"
        if st != "done":
            n = {"commonsense": len(CS) + 1, "math": 3}.get(kind, 1)
            print(head + f" _{st}_ |" + " |" * (n - 1) + f" {r['paper']} |")
        elif kind == "commonsense":
            print(head + "".join(f" {pct(s['datasets'][d]['accuracy'])} |" for d in CS) + f" **{pct(s['average'])}** | {r['paper']} |")
        elif kind == "math":
            d = s["datasets"]
            print(head + f" {pct(d['gsm8k']['accuracy'])} | {pct(d['MATH']['accuracy'])} | **{pct(s['average'])}** | {r['paper']} |")
        else:
            print(head + f" **{pct(s['accuracy'])}** | {r['paper']} |")


print(f"# DiaBlo paper-scale reproduction: results so far\n\n(source: `{args.work}`)")

print("\n### Minimum viable reproduction: our number vs the paper's\n")
print("| Run | Experiment | Ours | Paper | Ours - paper | Status |\n|---|---|---|---|---|---|")
for rid in ["G1", "A64", "A512", "A1024", "A16", "AL32"]:
    r = RUNS[rid]
    st, s = status(rid)
    ours = 100 * (s["accuracy"] if r["eval"] == "gsm8k_lori" else s["average"]) if s else None
    paper = r["paper_num"]
    print(f"| {rid} | {r['paper']} | {f'{ours:.2f}' if ours is not None else '-'} | "
          f"{f'{paper:.2f}' if paper else '-'} | "
          f"{f'{ours - paper:+.2f}' if ours is not None and paper else '-'} | {st} |")
table("Table 5 / Fig. 5: Llama3-8B fine-tuned on GSM8K (greedy, 1319 questions)", ["G0", "G1", "G2", "G3", "G4", "G5", "G6", "G7"], "gsm8k")
table("Table 1 (FP16): Commonsense-170k, 8 benchmarks (beam 4)", ["C1", "C2", "C5", "C3", "C4"], "commonsense")
table("Table 10: N ablation, Llama3.2-3B, commonsense", ["A16", "A64", "A256", "A512", "A1024", "AL32"], "commonsense")
table("Table 2: Llama2-7B on MetaMathQA (beam 4)", ["M1", "M2", "M1r"], "math")

bench = sorted(glob.glob(f"{args.work}/bench/*.json"))
bench = [b for b in bench if not b.endswith("_probe.json")]
if bench:
    print("\n### Cost projection from the bench job (GPU-hours, before the 30% margin)\n")
    print("| Run | GPU | batch x accum | grad. ckpt | s/step | train h | eval h | total h |\n|---|---|---|---|---|---|---|---|")
    tot = 0
    for p in bench:
        b = load(p)
        rid = os.path.basename(p)[:-5]
        tot += b["total_h"]
        print(f"| {rid} | {b['gpu']} | {b['bs']} x {b['ga']} | {'yes' if b['gc'] else 'no'} | {b['s_per_step']} | "
              f"{b['train_h']} | {b['eval_h']} | {b['total_h']} |")
    print(f"\nSum: {tot:.0f} GPU-hours.")
