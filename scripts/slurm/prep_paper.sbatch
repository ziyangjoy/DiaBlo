#!/usr/bin/env bash
# CPU-only preparation for the paper-scale runs (no GPU requested: costs 0 GPU-hours).
#   sbatch scripts/slurm/prep_paper.sbatch            (from the repo root)
# If compute nodes have no internet, run it on deep21 instead:  bash scripts/slurm/prep_paper.sbatch
# Safe to re-run: every step skips what already exists.
#
# Needs approved access to the gated meta-llama repos and a saved token:
#   source scripts/slurm/env.sh && huggingface-cli login      (stores it under $HF_HOME, which jobs read)
#
#SBATCH --job-name=prep-paper
#SBATCH --account=dl_2026_fall
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

echo "prep-paper started on $(hostname) at $(date)"
MODEL=Llama2-7B source scripts/slurm/env.sh || exit 1
set -eo pipefail
NPROC="${SLURM_CPUS_PER_TASK:-4}"
LLMA="https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main"
MM="https://raw.githubusercontent.com/meta-math/MetaMath/main"
MODELS=(Llama2-7B:meta-llama/Llama-2-7b-hf Llama3-8B:meta-llama/Meta-Llama-3-8B Llama3.2-3B:meta-llama/Llama-3.2-3B)

# 1. Internet + gated access
python -c "import urllib.request; urllib.request.urlopen('https://huggingface.co', timeout=20)" || {
    echo "ERROR: no internet on $(hostname). Run 'bash scripts/slurm/prep_paper.sbatch' on deep21 instead." >&2; exit 1; }
python - "${MODELS[@]}" <<'EOF'
import sys
from huggingface_hub import HfApi, hf_hub_download
try:
    print("HF user:", HfApi().whoami()["name"])
except Exception as e:
    sys.exit(f"ERROR: no valid HF token ({e}). Run: source scripts/slurm/env.sh && huggingface-cli login")
bad = []
for spec in sys.argv[1:]:
    repo = spec.split(":", 1)[1]
    try:
        hf_hub_download(repo, "config.json")
        print("access OK:", repo)
    except Exception as e:
        bad.append(repo)
        print(f"NO ACCESS: {repo} ({type(e).__name__}). Request it at https://huggingface.co/{repo}")
if bad:
    sys.exit("ERROR: gated model access missing; approval usually takes minutes to hours.")
EOF

# 2. Models (safetensors only; skips the duplicate .bin / original/ consolidated checkpoints)
for spec in "${MODELS[@]}"; do
    python - "${spec#*:}" <<'EOF'
import sys
from huggingface_hub import snapshot_download
p = snapshot_download(sys.argv[1], allow_patterns=["*.json", "*.safetensors", "tokenizer.model"],
                      ignore_patterns=["original/*"])
print("model cached at", p)
EOF
done

# 3. Model-independent test sets (+ the commonsense training file)
mkdir -p "$SHARED_DATA/dataset_commonsense" "$SHARED_DATA/dataset_math/gsm8k"
fetch() { [ -s "$2" ] || { mkdir -p "$(dirname "$2")"; curl -fsSL --retry 3 -o "$2.tmp" "$1" && mv "$2.tmp" "$2"; }; }
fetch "$LLMA/ft-training_set/commonsense_170k.json" "$SHARED_DATA/commonsense_170k.json"
for d in boolq piqa social_i_qa hellaswag winogrande ARC-Easy ARC-Challenge openbookqa; do
    fetch "$LLMA/dataset/$d/test.json" "$SHARED_DATA/dataset_commonsense/$d/test.json"
done
fetch "$LLMA/dataset/gsm8k/test.json" "$SHARED_DATA/dataset_math/gsm8k/test.json"
fetch "$MM/data/test/MATH_test.jsonl" "$SHARED_DATA/MATH_test.jsonl"
python - "$SHARED_DATA" <<'EOF'
import json, os, sys
d = sys.argv[1]
if not os.path.exists(f"{d}/MATH_test.json"):
    rows = [json.loads(l) for l in open(f"{d}/MATH_test.jsonl") if l.strip()]
    json.dump(rows, open(f"{d}/MATH_test.json", "w"))
sizes = {n: len(json.load(open(f"{d}/dataset_commonsense/{n}/test.json"))) for n in sorted(os.listdir(f"{d}/dataset_commonsense"))}
print("commonsense test sizes:", sizes, "total", sum(sizes.values()))
print("GSM8K test:", len(json.load(open(f"{d}/dataset_math/gsm8k/test.json"))),
      "| MATH test:", len(json.load(open(f"{d}/MATH_test.json"))),
      "| commonsense_170k train:", len(json.load(open(f"{d}/commonsense_170k.json"))))
EOF

# 4. Tokenized training data (tokenizer-specific, one folder per model)
for M in Llama2-7B Llama3-8B Llama3.2-3B; do
    D="$WORK/datasets/$M"
    if [ ! -d "$D/commonsense_170k_new/train_val120_256_OnlyOutput_$M" ]; then
        python data_processing/process_commonsense.py --data_path "$SHARED_DATA/commonsense_170k.json" \
            --output_dir "$D" --model_name "$M" --max_length 256 --num_proc "$NPROC"
    fi
done
if [ ! -d "$WORK/datasets/Llama2-7B/metamath/train_all_2048_OnlyOutput" ]; then
    python data_processing/process_metamath.py --output_dir "$WORK/datasets/Llama2-7B" --model_name Llama2-7B \
        --max_length 2048 --num_proc "$NPROC"
fi
if [ ! -d "$WORK/datasets/Llama3-8B/GSM8k_processed/train" ]; then
    python data_processing/process_gsm8k.py --model_name meta-llama/Meta-Llama-3-8B \
        --output_dir "$WORK/datasets/Llama3-8B" --max_length 512
fi

# 5. Unit tests + tiny end-to-end pipeline on CPU (catches library-version problems before any GPU time)
python -m pytest -q tests
python tests/e2e_tiny.py --workdir "$WORK/e2e" --precision fp32

du -sh "$HF_HOME" "$WORK/datasets" 2>/dev/null || true
echo "PREP-PAPER OK at $(date)"
