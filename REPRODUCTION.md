# Reproducing DiaBlo on a free Colab T4

Paper: *DiaBlo: Diagonal Blocks Are Sufficient For Finetuning* (arXiv:2506.03230, ICLR 2026).
Upstream code: https://github.com/ziyangjoy/DiaBlo @ `83ffb6c`. This branch: `colab-repro`.

## What is being reproduced
The paper's full experiments (Llama2/3 7–8B, MetaMathQA 395k × 2048 tokens, Commonsense-170k) are far beyond a free T4.
This study is a **method-level reproduction at 1.5B scale**: GSM8k-train → GSM8k-test on **Qwen2.5-1.5B**. It tests
the paper's relative claims with matched baselines:

| Claim | Paper evidence | Runs |
|---|---|---|
| **C1** DiaBlo ≥ LoRA at a comparable trainable-parameter budget | Table 2: Llama2-7B/MetaMathQA, GSM8k 66.5 (DiaBlo N=64, 1.04%) vs 60.6 (LoRA r=64, 1.67%) | R1 vs R2, R3 vs R8 |
| **C2** Accuracy improves as N decreases but plateaus around N=64 | App. A.3.5 (Table 10, Fig. 5) | R3 / R1 / R4 |
| **C3** Robust across learning rates | App. A.3.5, Fig. 5 (GSM8k, Llama3-8B) | R1 vs R7 |

It does **not** reproduce the paper's absolute numbers.

Trainable parameters on Qwen2.5-1.5B (q,k,v,up,down × 28 layers):

| DiaBlo | params | LoRA | params |
|---|---|---|---|
| N=32 | 26.84M | r=32 | 24.77M |
| N=64 | 13.42M | r=16 | 12.39M |
| N=128 | 6.71M | r=8 | 6.19M |

The optional stretch (`notebooks/04_stretch_llama2_4bit.ipynb`) compares NF4 DiaBlo with QLoRA on Llama2-7B. The paper's quantized table
uses GPTQ/MagR and Math10K, which aren't in the repo, so this is a related experiment rather than a reproduction of Table 4.

## How to run
1. Push this branch to your GitHub fork. Set `REPO_URL` in the first cell of each notebook. Alternatively, copy the repo folder to
   `MyDrive/diablo/DiaBlo`.
2. In Colab, open `notebooks/00_setup_smoketest.ipynb` on a T4 and run all cells. This runs the tests, tokenizes GSM8k, checks zero-init and
   runs the fp16 smoke test with resume, and prints s/step → hours per run.
3. Open `notebooks/01_train.ipynb`, set `RUN_ID`, and run all cells. Re-run after a disconnect and it resumes from Drive.
   Priority: R1, R2, R3, R4, then R6/R7, then R8/R5.
4. Run `notebooks/02_eval.ipynb`: the parity check once, then the full eval of R0 (base model) and the finished runs.
5. Run `notebooks/03_results.ipynb` for the table, loss curves, accuracy-vs-params plot, McNemar tests and write-up template.

Everything persistent lives in `MyDrive/diablo/` (datasets, results/<RUN_ID>, eval). The HF model cache is kept on local disk (`/content/hf_cache`), because Drive corrupts multi-GB safetensors files.

## Running on the SZTE Deep Cluster (Slurm)
Branch `deepcluster` adds `scripts/slurm/`, which wraps the same `train.py` / `evaluate_gsm8k.py` calls as the notebooks.
Cluster rules that shape it: 20 GPU-hours per course account, 2 running jobs, 24 h per job, mostly 11–12 GB GPUs
(see https://github.com/szegedai/deepcluster_onboarding).

```bash
# on deep21, once
git clone -b deepcluster https://github.com/Mrazakos/DiaBlo.git ~/projects/DiaBlo && cd ~/projects/DiaBlo
bash scripts/slurm/setup_env.sh           # conda env "diablo" (python 3.12, torch cu126)
sbatch scripts/slurm/prep.sbatch          # CPU-only: model + data download, unit tests, tiny e2e
sbatch scripts/slurm/smoke.sbatch         # ~10 GPU-min: memory probe, s/step, eval timing -> "SMOKE ..." lines
# then the whole matrix, 2 jobs at a time (TIME / BS / GA from the smoke output)
TIME=02:00:00 bash scripts/slurm/submit_all.sh
```

Each run is one job: train with `--resume`, then HF-generate eval. Re-running `submit_all.sh` skips finished runs and resumes the
rest. All jobs are pinned to one GPU type (`rtx2080ti`). Everything persistent lives in `~/diablo/` with the same layout as on
Drive (`datasets/`, `results/<RUN_ID>/`, `eval/<RUN_ID>_hf.jsonl`), so `notebooks/03_results.ipynb` works on a copy with `BACKEND = "hf"`.

Run matrix on the cluster (shared settings as in the notebooks; R0 is the base model):

| ID | Method | Setting | Trainable | LR | Seed | Claim tested |
|---|---|---|---|---|---|---|
| R1 | DiaBlo | N=64 | 13.42M | 5e-4 | 42 | C1, C2, C3 |
| R2 | LoRA | r=16 | 12.39M | 2e-4 | 42 | C1 |
| R3 | DiaBlo | N=32 | 26.84M | 5e-4 | 42 | C1, C2 |
| R8 | LoRA | r=32 | 24.77M | 2e-4 | 42 | C1 |
| R4 | DiaBlo | N=128 | 6.71M | 5e-4 | 42 | C2 |
| R9 | DiaBlo | N=512 | 1.76M | 5e-4 | 42 | C2 |
| R10 | DiaBlo | N=16 | 53.67M | 5e-4 | 42 | C2 |
| R7 | DiaBlo | N=64 | 13.42M | 2e-4 | 42 | C3 |
| R11 | DiaBlo | N=64 | 13.42M | 1e-4 | 42 | C3 |
| R12 | DiaBlo | N=64 | 13.42M | 1e-3 | 42 | C3 |
| R5a | DiaBlo | N=64 | 13.42M | 5e-4 | 1 | variance |
| R5b | LoRA | r=16 | 12.39M | 2e-4 | 1 | variance |

Compared with the Colab matrix, R9/R10 widen the N sweep and R11/R12 widen the LR sweep. The paper's N ablation (LLaMA3.2-3B,
commonsense) differs by only 0.3 points between N=32, 64 and 128 and drops visibly only from N=512, while one GSM8k accuracy has
about ±2.6 points of sampling noise (1319 questions), so three adjacent N values cannot test C2. R6 (LoRA at lr 5e-4) is dropped.

At N=512, Qwen's `k_proj`/`v_proj` (256 outputs) get 1-wide output blocks and only the first 256 blocks reach the output. This is
upstream `BlockLinear` behaviour when N exceeds the layer width, and it affects only those two modules.

### Second wave: closer to the paper (run table in `scripts/slurm/runs.sh`)
The first wave's fine-tuned runs all scored below the base model, probably because Qwen2.5-1.5B is already strong at GSM8k
(see `RESULTS.md`). Two follow-ups:

| Group | Runs | What changes |
|---|---|---|
| `paper` | P1 (DiaBlo N=64), P2 (LoRA r=16) | Paper Table 20 recipe on Qwen2.5-1.5B: 2 epochs, effective batch 32, DiaBlo lr 4e-4, warmup 20, all seven modules. LoRA keeps lr 2e-4 (no LoRA lr is given for this setup) |
| `paper` | S0 (base), S1, S2 | Same recipe on Qwen2.5-0.5B, a weaker base. Results go to `~/diablo/{results,eval}-Qwen2.5-0.5B/` |
| `llama` | L0-L2 (Llama-3.2-3B), H1/H2 (Llama3-8B) | Paper's model family, 4-bit NF4 base. Needs more GPU hours than the course budget, gated HF access (`HF_TOKEN`), probably `GPU=a5000` and a larger `--mem`, and a memory probe for BS/GA first |

```bash
TIME=02:00:00 bash scripts/slurm/submit_all.sh paper     # P1 P2 S0 S1 S2, 2 jobs at a time
MODEL=Qwen2.5-0.5B sbatch scripts/slurm/prep.sbatch      # once, before the S runs: model + tokenized data
```

Budget order if hours are short: finish R12/R5b, then P1/P2, then S0 (base check, ~0.2 GPU-h) and S1/S2 only if S0 is well below the 1.5B base.

**Bug in the first two waves' `run.sbatch` (fixed on branch `paper-repro`):** it passed `--max_epochs 3 --warmup_steps 100
--target_modules qkvud` to every run, ignoring `EPOCHS`/`WARMUP`/`MODULES` from `runs.sh`. The R runs are unaffected (that *is*
their recipe). Any P/S/L/H run trained before the fix used the first-wave recipe instead of Table 20, so delete and rerun it.

## Paper-scale reproduction (branch `paper-repro`, 1000 GPU-hours)
The paper's own experiments: Llama models in 16-bit, the paper's datasets and its appendix hyper-parameters, evaluated with the
released evaluation code. The paper used one A100-80GB. Here, the only cluster GPU that can hold a 7–8B model in 16-bit is the
**single a5000 (24 GB)**, so all 7–8B runs are a serial queue on that card. The 3B ablation runs in parallel on an rtx2080ti
(fp16, which Turing supports; the per-person limit is 2 running jobs).

### Minimum viable reproduction (start here)
The smallest set of runs that each have a **published number to match**, finishing in about a week of wall clock:

| Run | Paper number | Model / data | GPU | Rough cost |
|---|---|---|---|---|
| G1 | Table 5, DiaBlo 1/64 fine-tuned on GSM8K: **67.68** | Llama3-8B, GSM8K, Table 20 recipe | a5000 | ~3 GPU-h |
| A64 | Table 10, N=64: **84.44** (8-task commonsense avg) | Llama3.2-3B, Commonsense-170k, 3 epochs | rtx2080ti | ~40 GPU-h |
| A512 | Table 10, N=512: **82.54** | same | rtx2080ti | ~40 GPU-h |
| A1024 | Table 10, N=1024: **81.60** | same | rtx2080ti | ~40 GPU-h |

Together they reproduce the paper's Table 5 fine-tuning number on its own 8B model, and the shape of its N ablation
(flat at N=64, then dropping at 512 and 1024) on its own 3B model. About 125 GPU-hours. With two jobs running at once, A64 and A512
run side by side and everything is done in about 5 days. Next if time remains: A16 (84.38) and AL32, a same-pipeline LoRA r=32
baseline on the 3B model (the paper has no LoRA number there). Bench A16 separately first, because it has the largest adapter.

```bash
sbatch scripts/slurm/prep_paper.sbatch                                                    # CPU only, 0 GPU-h
sbatch --gres=gpu:a5000:1 --mem=48G --time=01:00:00 scripts/slurm/bench.sbatch G1
sbatch --gres=gpu:rtx2080ti:1 --mem=16G --time=01:00:00 scripts/slurm/bench.sbatch A64
bash scripts/slurm/submit_all.sh mvr
python scripts/paper_results.py          # first table: ours vs paper, with the difference
```
Expect agreement within about ±1 point, not exact equality:
- Table 10's learning rate is not published (1e-4 is used for every N).
- The eval prompt matches training rather than the released eval script (see the deviations below).
- One seed only. The 8 commonsense test sets total about 22k questions, so the average's sampling noise is about ±0.3 points.

### What is reproduced

| Group | Runs | Paper | Setting (paper appendix table) | GPU |
|---|---|---|---|---|
| `gsm8k` | G1 (N=64), G0 (base), G2–G5 (lr 1e-4…1.6e-3), G6/G7 (N=32/128) | Table 5 DiaBlo row (67.68), Fig. 5 | Llama3-8B, GSM8K-train, Table 20: lr 4e-4, 2 ep, batch 32, warmup 20, len 512, qkvogud, bf16 | a5000 |
| `cs` | C1/C2 (Llama2-7B N=64/128), C3/C4 (Llama3-8B N=64/128) | Table 1 FP16 (83.4 / 83.5 / 87.3 / 87.2) | Commonsense-170k, Table 12: 3 ep, batch 16, warmup 100, len 256, qkvud, fp16 base + fp32 adapters | a5000 |
| `ablate` | A16, A64, A256, A512, A1024 | Table 10 (84.38 … 81.60) | Llama3.2-3B, Commonsense-170k, Table 12 recipe (the table's parameter counts imply q,k,v,up,down), lr 1e-4 for all N | rtx2080ti |
| `mm` | M1 (N=64), M2 (N=32) | Table 2 (GSM8K 66.5 / 66.3, MATH 17.6 / 20.4) | Llama2-7B, MetaMathQA-395k, Table 14: lr 2e-4, 3 ep, batch 16, warmup 100, len 2048, qkvud, bf16 | a5000 |
| `optional` | C5 (LoRA r=32), M1r | same-pipeline LoRA baseline; MetaMath with the released script's recipe | C5: lr 3e-4 (LLM-Adapters default). M1r: 1 ep, lr 1.2e-3, batch 32, cosine (`scripts/train_metamath.sh`) | a5000 |

**Not reproduced, and why:** LLaMA-13B / Llama2-13B (26 GB of fp16 weights do not fit 24 GB); Table 6 BF16 Mistral/Llama3 commonsense
(fits, about 15 GPU-h each, add if hours remain); Table 3 code/safety (HEx-PHI refusal is judged by GPT-4); Table 4 quantized models
(GPTQ/MagR and Math10K are not in the repo); Table 5's random-sparsity patterns (they need new adapter types); MT-Bench and GLUE.

### Plan, budget and wall clock
Rough estimates. They assume the A5000 is about 3.5× slower than the paper's A100, based on its 170 min/epoch commonsense timing.
The **bench job measures the real numbers first**, and `submit_all.sh` sizes every job from them.

| Phase | Runs | GPU-h (rough) | Gate |
|---|---|---|---|
| 0 prep | models, data, tests | 0 (CPU job) | `PREP-PAPER OK` |
| 0 bench | G6 C1 C3 M2 on a5000, A16 on rtx2080ti | ~3 | read the `BENCH` lines, re-plan |
| 1 | `gsm8k` (8 runs) | ~15–20 | G1 within a few points of 67.68 → the pipeline works at 8B |
| 2 | `cs` (4 runs) | ~150–180 | |
| 2′ parallel | `ablate` (5 runs, rtx2080ti) | ~125–200 | |
| 3 | `mm`: M1, then M2 | ~150–200 each | if the bench projects M1 > ~220 h, run M1r instead (~1/3 of the cost) |
| reserve | reruns, optional C5 | ~100 | |

Total ≈ 600–800 GPU-hours. The a5000 track is about 450–600 h, which is **3–4 weeks of exclusive use** of the one A5000. See who holds it with
`squeue -h -o "%u %b %T %M" | grep a5000`, and start Phase 2 early.

### How to run (on deep21, from the repo root, branch `paper-repro`)
```bash
git fetch && git checkout paper-repro && git pull
# once: approved access to meta-llama/Llama-2-7b-hf, Meta-Llama-3-8B and Llama-3.2-3B on huggingface.co, then
source scripts/slurm/env.sh && huggingface-cli login       # token goes to $HF_HOME, which jobs read

sbatch scripts/slurm/prep_paper.sbatch                      # CPU only: models, test sets, tokenized data, tests
sbatch --gres=gpu:a5000:1 --mem=48G --time=02:30:00 scripts/slurm/bench.sbatch G6 C1 C3 M2
sbatch --gres=gpu:rtx2080ti:1 --mem=16G --time=01:00:00 scripts/slurm/bench.sbatch A16
bash scripts/slurm/submit_all.sh gsm8k                      # Phase 1
bash scripts/slurm/submit_all.sh cs ablate                  # Phase 2 (a5000 chain + rtx2080ti chain in parallel)
bash scripts/slurm/submit_all.sh mm                         # Phase 3
python scripts/paper_results.py                             # paper-vs-ours tables at any time
```
Runs longer than 24 h are submitted as chains of 24 h segments (`--dependency=afterok`). Each segment stops training on a complete
checkpoint 15 min before its limit and the next one resumes. Evaluations append per example and resume the same way. A crash
cancels the rest of that run's chain. `submit_all.sh` skips finished and queued runs, so re-running it is always safe.

### Deviations from the paper in this part
- **Hardware:** a 24 GB A5000 instead of an 80 GB A100. The effective batch size is the paper's, but the per-device batch is smaller,
  with gradient accumulation (and gradient checkpointing where the bench needs it). This changes speed, not the math, apart from
  floating-point summation order.
- **Evaluation prompt:** the released eval scripts do not use the training template. `evaluate_commonsense.py` sent only
  `"<instruction>\n"`, and `evaluate_math.py` indents the template with 16 spaces where training uses 12. Both now give the model the
  exact training prefix (`diablo/prompts.py`). `--prompt_style upstream` reproduces the released prompts, for example to check how
  much this matters on one run. Decoding (beam 4) and answer extraction are unchanged. Only generated tokens are scored.
- **Pad token:** no new `<PAD>` token or embedding resize. Llama-2 pads with `<unk>` and Llama-3 with EOS. Padding is always masked,
  and this avoids a randomly initialised extra lm_head row that differed between training and evaluation.
- **Commonsense split:** like LLM-Adapters, 120 examples are held out with a fixed seed (upstream shuffled without one).
- **Unstated hyper-parameters:** Fig. 5's learning rates (a doubling grid around 4e-4 is used), Table 10's learning rate (1e-4 for every
  N), adapter dropout (0.05, the released scripts' value) and GSM8K precision (bf16).
- **MetaMathQA:** Table 14 (3 epochs, lr 2e-4, batch 16, linear) contradicts the released `scripts/train_metamath.sh` (1 epoch, lr 1.2e-3,
  batch 32, cosine). M1/M2 follow the paper table; M1r follows the script.

## Changes vs upstream (all backwards compatible for the original Llama scripts)
- `diablo/common.py`: one shared `MODEL_MAP` (adds Qwen2.5-0.5B/1.5B, accepts raw HF ids) and `setup_pad_token`.
  Upstream always added `<PAD>` and resized embeddings to `len(tokenizer)`, which **shrinks** Qwen's padded embedding matrix.
  Since `paper-repro`, no token is ever added: the pad token is the model's own, else `<unk>`, else EOS.
- `diablo/prompts.py`, `diablo/eval_utils.py` (paper-repro): shared train/eval prompts; one model loader for all evaluators
  (strict adapter check, resumable JSONL, OOM backoff). `evaluate_commonsense.py` and `evaluate_math.py` were rewritten on top of them.
- `diablo/block_linear.py` (paper-repro): `replace_linear_with_blocklinear` reuses the frozen layer instead of allocating and copying
  a new one (same result; on 7B it saved minutes and GBs of RAM).
- `train.py` (paper-repro): `--deadline` (checkpoint and exit 3 before a Slurm limit), removal of half-written checkpoints on
  `--resume`, `train_stats.json` (s/step, peak memory).
  Now a pad token is added only if missing (Llama-2 behaviour is unchanged).
- `diablo/adapters.py`: `apply_adapter` (DiaBlo / LoRA via peft / none), `mark_trainable`, `load_adapter` (strict key check),
  `merge_adapter`, and `BlockAdapter`, a DiaBlo wrapper for 4-bit bitsandbytes layers that reuses the upstream einsum.
- `diablo/trainer.py`: adapter-only checkpoints also cover LoRA. Adds `_load_from_checkpoint`, so `--resume` works.
  Upstream's adapter-only saving broke the HF Trainer's resume.
- `train.py`: `--method {diablo,lora}`, `--lora_r`, `--resume`, `--save_steps` (default steps/100), `--max_steps`, `--report_to`
  (wandb is now opt-in), `--load_in_4bit`, `--run_name`, and a bf16-on-T4 guard. Writes `run_config.json` and `log_history.json`.
- `evaluate_gsm8k.py`: same prompt and scoring, now via `gsm8k_utils.py`. Adds LoRA/base/merged/4-bit support, resumable JSONL output, and
  reports all 1319 questions plus the original first-1312 protocol.
- `eval_gsm8k_vllm.py` (new): fast eval of merged models with identical scoring (the answer is extracted from prompt+completion, as upstream does).
- `merge.py`: also merges LoRA, merges in fp32, and can read `run_config.json` via `--run_dir`.
- `tests/`: unit tests (zero-init identity, merge == adapter forward incl. non-divisible shapes, block-diagonal structure,
  `BlockAdapter` == `BlockLinear`) and `tests/e2e_tiny.py`, which runs train → resume → merge → eval on a tiny random Qwen2 for both methods
  and asserts that the merged and adapter generations are identical.

Not touched: commonsense/MetaMath/MATH scripts, which are out of scope here.

## Known deviations from the paper
Smaller model (Qwen2.5-1.5B), GSM8k-train instead of MetaMathQA, fp16 on T4 (the paper uses bf16), LoRA at a matched parameter count rather than
r=64, a single seed by default (R5 adds a second), and repo-default hyperparameters (`scripts/train_gsm8k.sh`: lr 5e-4, effective batch 16,
3 epochs, warmup 100, dropout 0.05, qkvud) with only a small LR check for both methods (R6, R7).

The paper's only stated settings for fine-tuning on GSM8k-train (Table 20, Llama3-8B) are 2 epochs, batch 32, lr 4e-4, 20 warmup
steps and all seven modules (q,k,v,o,gate,up,down). This study uses the released script's settings instead. The paper's LR-robustness
figure (Fig. 5) is on Llama3-8B and has no LoRA comparison, and its N ablation is on commonsense reasoning, not GSM8k.
