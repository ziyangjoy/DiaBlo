#!/usr/bin/env bash
# Submit runs in priority order. Run on deep21 from the repo root:
#   bash scripts/slurm/submit_all.sh mvr              # a group (see runs.sh): mvr first paper llama gsm8k cs ablate mm optional
#   bash scripts/slurm/submit_all.sh C1 C3            # or single runs
# Safe to run again at any time: finished runs and runs already in the queue are skipped, unfinished ones resume.
#
# Each run is submitted as a chain of segments (sbatch --dependency=afterok), each at most 24 h. A segment
# stops training on a checkpoint before its time limit and exits 0, so the next one continues. A crash exits
# non-zero, and the rest of that run's chain is cancelled instead of crashing again on your budget.
# Segments and --time come from $WORK/bench/<ID>.json (bench.sbatch) when it exists, else from runs.sh.
#
# Overrides (environment): TIME (--time per segment), SEGMENTS, GPU (GPU type), BS GA GC (batch layout).
set -eo pipefail
GPU_ENV="${GPU:-}"; SEGMENTS_ENV="${SEGMENTS:-}"     # load_run overwrites GPU/SEGMENTS, so keep the user's values

source scripts/slurm/runs.sh

if [ ! -f train.py ]; then
    echo "ERROR: run this from the DiaBlo repo root" >&2
    exit 1
fi
mkdir -p logs

expand=()
for a in "$@"; do
    case "$a" in
        first)    expand+=("${DEFAULT_ORDER[@]}") ;;
        paper)    expand+=("${PAPER_ORDER[@]}") ;;
        llama)    expand+=("${LLAMA_ORDER[@]}") ;;
        gsm8k)    expand+=("${GSM8K_ORDER[@]}") ;;
        cs)       expand+=("${CS_ORDER[@]}") ;;
        ablate)   expand+=("${ABLATE_ORDER[@]}") ;;
        mm)       expand+=("${MM_ORDER[@]}") ;;
        optional) expand+=("${OPTIONAL_ORDER[@]}") ;;
        mvr)      expand+=("${MVR_ORDER[@]}") ;;
        *)        expand+=("$a") ;;
    esac
done
IDS=("${expand[@]}")
if [ ${#IDS[@]} -eq 0 ]; then
    echo "usage: bash scripts/slurm/submit_all.sh <group|ID>...   (groups: mvr first paper llama gsm8k cs ablate mm optional)" >&2
    exit 2
fi

[ -n "${BS:-}" ] && export BS
[ -n "${GA:-}" ] && export GA
[ -n "${GC:-}" ] && export GC
queued="$(squeue --me -h -o %j)"
work="${DIABLO_WORK:-$HOME/diablo}"

for id in "${IDS[@]}"; do
    summary="$(eval_summary_path "$id")" || exit 2
    if [ -f "$summary" ]; then
        echo "$id: finished, skipping"
        continue
    fi
    if grep -qx "$id" <<<"$queued"; then
        echo "$id: already in the queue, skipping"
        continue
    fi
    load_run "$id"

    # Segments and time per segment: bench projection (+30 % margin) > runs.sh default.
    segs="$SEGMENTS"; time="24:00:00"
    bench="$work/bench/$id.json"
    if [ -f "$bench" ]; then
        read -r segs time < <(python3 - "$bench" <<'EOF'
import json, math, sys
b = json.load(open(sys.argv[1]))
total = b["total_h"] * 1.3 + 0.25
segs = max(1, math.ceil(total / 23.5))
hours = 24 if segs > 1 else min(24, math.ceil(total * 2) / 2)
print(segs, f"{int(hours):02d}:{int(hours % 1 * 60):02d}:00")
EOF
)
    elif [[ "$MODEL" == Qwen* ]]; then
        time="03:00:00"   # first-wave / second-wave Qwen runs
    fi
    segs="${SEGMENTS_ENV:-$segs}"
    [ -n "${TIME:-}" ] && time="$TIME"
    gpu="${GPU_ENV:-$GPU}"

    prev=""
    jids=()
    for ((s = 1; s <= segs; s++)); do
        opts=(--parsable --export=ALL -J "$id" --gres="gpu:${gpu}:1" --mem="$MEM" --time="$time")
        [ -n "$prev" ] && opts+=(--dependency="afterok:$prev" --kill-on-invalid-dep=yes)
        prev="$(sbatch "${opts[@]}" scripts/slurm/run.sbatch "$id")"
        jids+=("$prev")
    done
    echo "$id: submitted ${segs} segment(s) of $time on $gpu, mem $MEM: jobs ${jids[*]}  (logs: logs/${id}-<job>.out)"
done

echo
squeue --me
