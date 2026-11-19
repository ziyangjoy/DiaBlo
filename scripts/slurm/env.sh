#!/usr/bin/env bash
# Shared settings for every cluster script. Source it, don't run it:
#   source scripts/slurm/env.sh
# Override any of these by exporting them before sbatch / bash.

CONDA_ENV="${CONDA_ENV:-diablo}"
source /shared/anaconda/anaconda/bin/activate
conda activate "$CONDA_ENV"

export MODEL="${MODEL:-Qwen2.5-1.5B}"            # short name in diablo/common.py MODEL_MAP
case "$MODEL" in                                 # keep in sync with MODEL_MAP in diablo/common.py
    Qwen2.5-0.5B) _HF=Qwen/Qwen2.5-0.5B ;;
    Qwen2.5-1.5B) _HF=Qwen/Qwen2.5-1.5B ;;
    Llama3-3B|Llama3.2-3B) _HF=meta-llama/Llama-3.2-3B ;;   # gated: needs approved HF access + a saved HF token
    Llama3-8B)    _HF=meta-llama/Meta-Llama-3-8B ;;   # gated
    Llama2-7B)    _HF=meta-llama/Llama-2-7b-hf ;;     # gated
    *)            _HF="$MODEL" ;;
esac
export HF_ID="${HF_ID:-$_HF}"

# Everything persistent lives here (same layout as MyDrive/diablo in the Colab notebooks).
export WORK="${DIABLO_WORK:-$HOME/diablo}"
export DATA="$WORK/datasets/$MODEL"              # tokenized data is tokenizer-specific
# The first-wave Qwen2.5-1.5B runs keep the original flat folders; every other model gets its own.
if [ "$MODEL" = "Qwen2.5-1.5B" ]; then
    export RESULTS="$WORK/results"
    export EVALS="$WORK/eval"
else
    export RESULTS="$WORK/results-$MODEL"
    export EVALS="$WORK/eval-$MODEL"
fi
export HF_HOME="$WORK/hf_cache"                 # `huggingface-cli login` after sourcing this stores the token here
export SHARED_DATA="$WORK/datasets/shared"       # model-independent test sets (commonsense, GSM8K/MATH json)

export WANDB_MODE=disabled
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # less fragmentation, no effect on results
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$DATA" "$RESULTS" "$EVALS" "$SHARED_DATA" "$WORK/bench"

# Jobs must be submitted from the repo root (sbatch copies the script, so its own path is useless).
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f train.py ]; then
    echo "ERROR: run this from the DiaBlo repo root (train.py not found in $PWD)" >&2
    return 1 2>/dev/null || exit 1
fi
