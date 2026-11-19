#!/usr/bin/env bash
# One run = train (resumable) + evaluation (resumable), as one Slurm job or a chain of 24 h segments.
#   bash scripts/slurm/submit_all.sh C1          (sets GPU type, memory, time and the segment chain from runs.sh)
#   sbatch -J C1 --gres=gpu:a5000:1 --mem=40G --time=24:00:00 scripts/slurm/run.sbatch C1   (one segment by hand)
# Training stops ~15 min before the job's time limit, on a complete checkpoint; the next segment (or the same
# command again) continues from it. Evaluation appends per example, so it also resumes.
#
# Environment overrides: BS / GA / GC (per-device batch, accumulation, gradient checkpointing 0/1),
# EVAL_BS (eval batch; halved automatically on OOM).
#
#SBATCH --job-name=diablo-run
#SBATCH --account=dl_2026_fall
#SBATCH --gres=gpu:rtx2080ti:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=6G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

RUN_ID="${1:?usage: sbatch -J <ID> scripts/slurm/run.sbatch <ID>   (ids: see scripts/slurm/runs.sh)}"

# Run table lives in runs.sh. It decides MODEL, so load it before env.sh.
source scripts/slurm/runs.sh
load_run "$RUN_ID" || exit 2
export MODEL
BS="$RUN_BS"; GA="$RUN_GA"; GC="$RUN_GC"

echo "Job ${SLURM_JOB_ID:-local} ($RUN_ID) started on $(hostname) at $(date)"
source scripts/slurm/env.sh || exit 1
set -eo pipefail
export HF_HUB_OFFLINE=1          # models and data were cached by prep_paper.sbatch
unset HF_TOKEN                   # a token makes the scripts call login(), which fails offline
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

RUN_DIR="$RESULTS/$RUN_ID"
SIZE_FLAG=()
[ "$METHOD" = diablo ] && SIZE_FLAG=(--num_blocks "$SIZE")
[ "$METHOD" = lora ]   && SIZE_FLAG=(--lora_r "$SIZE")
QUANT=(); [ "$LOAD4" = 1 ] && QUANT=(--load_in_4bit)
GC_FLAG=(); [ "$GC" = 1 ] && GC_FLAG=(--gradient_checkpointing)

# Stop training 15 min before the job ends, so it exits on a complete checkpoint.
DEADLINE_FLAG=()
end_ts=""
if [ -n "${SLURM_JOB_ID:-}" ]; then
    end="$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null || true)"
    end_ts="$(date -d "$end" +%s 2>/dev/null || true)"
    if [ -n "$end_ts" ]; then
        DEADLINE_FLAG=(--deadline "$((end_ts - 900))")
        echo "Job ends at $end; training stops by $(date -d "@$((end_ts - 900))")"
    fi
fi

# ── Train ────────────────────────────────────────────────────────────────────
if [ "$METHOD" = none ]; then
    echo "$RUN_ID: base model, nothing to train"
elif [ -f "$RUN_DIR/log_history.json" ] && [ -f "$RUN_DIR/adapter.chkpt" ]; then
    echo "$RUN_ID: training already finished, skipping to eval"
else
    mkdir -p "$RUN_DIR"
    if [ -f "$RUN_DIR/run_config.json" ]; then
        # Resuming: refuse to continue a run with different settings than it was started with.
        python - <<EOF
import json, sys
have = json.load(open("$RUN_DIR/run_config.json"))
want = dict(task="$TASK", method="$METHOD", lr=float("$LR"), seed=$SEED, max_epochs=$EPOCHS, warmup_steps=$WARMUP,
            target_modules="$MODULES", per_device_batch_size=$BS, gradient_accumulation_steps=$GA,
            precision="$PREC", max_length=$MAXLEN, scheduler="$SCHED")
want["num_blocks" if "$METHOD" == "diablo" else "lora_r"] = int("$SIZE")
bad = {k: f"started with {have.get(k)!r}, now {v!r}" for k, v in want.items() if have.get(k) != v}
if bad:
    sys.exit(f"ERROR: $RUN_ID was started with different settings: {bad}. "
             f"Use the same BS/GA, or delete $RUN_DIR to restart it.")
EOF
    fi
    echo "$RUN_ID: $MODEL $TASK $METHOD $SIZE lr=$LR epochs=$EPOCHS bs=$BS x ga=$GA gc=$GC $PREC len=$MAXLEN"
    set +e
    python train.py --model_name "$MODEL" --task "$TASK" --data_dir "$DATA" \
        --output_dir "$RESULTS" --run_name "$RUN_ID" --resume \
        --method "$METHOD" "${SIZE_FLAG[@]}" --lr "$LR" --seed "$SEED" \
        --per_device_batch_size "$BS" --gradient_accumulation_steps "$GA" \
        --max_epochs "$EPOCHS" --warmup_steps "$WARMUP" --dropout 0.05 --target_modules "$MODULES" \
        --precision "$PREC" --max_length "$MAXLEN" --scheduler "$SCHED" --save_steps "$SAVE_STEPS" \
        "${GC_FLAG[@]}" "${QUANT[@]}" "${DEADLINE_FLAG[@]}" 2>&1 | tee -a "$RUN_DIR/train.log"
    rc=${PIPESTATUS[0]}
    set -e
    if [ "$rc" = 3 ]; then
        echo "RUN $RUN_ID SEGMENT DONE at $(date): training continues in the next segment (or submit again)."
        exit 0
    elif [ "$rc" != 0 ]; then
        echo "ERROR: training of $RUN_ID failed with exit code $rc" >&2
        exit "$rc"
    fi
fi

# ── Evaluate ─────────────────────────────────────────────────────────────────
SUMMARY="$(eval_summary_path "$RUN_ID")"
ADAPTER=()
[ "$METHOD" != none ] && ADAPTER=(--method "$METHOD" "${SIZE_FLAG[@]}" --target_modules "$MODULES" --adapter_path "$RUN_DIR/adapter.chkpt")
[ "$METHOD" = none ] && ADAPTER=(--method none)

# Stop the evaluation 3 min before the job ends (exit 0), so the next segment resumes it instead of the chain
# breaking on a TIMEOUT. Results are appended per batch, so at most one batch is redone.
EVAL_CMD=()
if [ -f "$SUMMARY" ]; then
    echo "$RUN_ID: eval already finished"
elif [ "$EVAL" = gsm8k_lori ]; then
    # 1319 GSM8k test questions, greedy, 512 new tokens (upstream evaluate_gsm8k.py protocol)
    EVAL_CMD=(evaluate_gsm8k.py --model_name "$MODEL" "${ADAPTER[@]}" "${QUANT[@]}" --data_dir "$DATA"
              --precision "$PREC" --batch_size "${EVAL_BS:-32}" --output_file "$EVALS/${RUN_ID}_hf.jsonl")
elif [ "$EVAL" = commonsense ]; then
    # 8 LLM-Adapters benchmarks, beam 4, 32 new tokens (upstream evaluate_commonsense.py protocol)
    EVAL_CMD=(evaluate_commonsense.py --model_name "$MODEL" "${ADAPTER[@]}" --data_dir "$SHARED_DATA"
              --precision "$PREC" --batch_size "${EVAL_BS:-16}" --output_dir "$EVALS/$RUN_ID")
elif [ "$EVAL" = math ]; then
    # GSM8K (256 new tokens) + MATH (512), beam 4 (upstream evaluate_math.py protocol)
    EVAL_CMD=(evaluate_math.py --model_name "$MODEL" "${ADAPTER[@]}" --data_dir "$SHARED_DATA"
              --precision "$PREC" --batch_size "${EVAL_BS:-8}" --output_dir "$EVALS/$RUN_ID")
fi
if [ ${#EVAL_CMD[@]} -gt 0 ]; then
    TIMEOUT=()
    if [ -n "$end_ts" ]; then
        left=$((end_ts - $(date +%s) - 180))
        if [ "$left" -lt 600 ]; then
            echo "RUN $RUN_ID SEGMENT DONE at $(date): too little time left to evaluate; the next segment does it."
            exit 0
        fi
        TIMEOUT=(timeout --signal=TERM "${left}s")
    fi
    set +e
    "${TIMEOUT[@]}" python "${EVAL_CMD[@]}"
    rc=$?
    set -e
    if [ "$rc" = 124 ]; then
        echo "RUN $RUN_ID SEGMENT DONE at $(date): evaluation continues in the next segment (or submit again)."
        exit 0
    elif [ "$rc" != 0 ]; then
        echo "ERROR: evaluation of $RUN_ID failed with exit code $rc" >&2
        exit "$rc"
    fi
fi

echo "RUN $RUN_ID DONE at $(date)"
cat "$SUMMARY" 2>/dev/null || echo "(no final summary yet: submit the run again to finish the evaluation)"
