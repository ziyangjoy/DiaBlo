#!/usr/bin/env bash
# Phase 0: measure before spending the budget. For each representative run it
#   1. picks per-device batch / accumulation / gradient checkpointing from worst-case memory (bench_probe.py),
#   2. times 12 real training steps with that layout (6 for MetaMath),
#   3. times a slice of the evaluation,
# then writes $WORK/bench/<ID>.json for the run and its siblings (same model + task) and prints BENCH lines with
# the projected GPU-hours. run.sbatch and submit_all.sh read those files (batch layout, segments, --time).
#
#   sbatch --gres=gpu:a5000:1 --mem=48G --time=02:30:00 scripts/slurm/bench.sbatch G6 C1 C3 M2
#   sbatch --gres=gpu:rtx2080ti:1 --mem=16G --time=01:00:00 scripts/slurm/bench.sbatch A16
# Minimum viable reproduction: bench.sbatch G1 on the a5000 and bench.sbatch A64 on an rtx2080ti (~30 min each).
# A family's representative should have the most trainable parameters (worst memory). G1 / A64 are the cheaper
# representatives for the minimum viable reproduction; A16 then needs its own bench before it runs.
# Re-running is safe: finished steps are skipped (delete $WORK/bench/<ID>* to redo one).
#
#SBATCH --job-name=bench
#SBATCH --account=dl_2026_fall
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:30:00
#SBATCH --output=logs/%x-%j.out

[ $# -gt 0 ] || { echo "usage: sbatch [--gres=...] scripts/slurm/bench.sbatch <rep ID>..." >&2; exit 2; }

# Siblings share the representative's measurements (same model, task, sequence length and batch).
declare -A FAMILY=(
    [G6]="G0 G1 G2 G3 G4 G5 G7"
    [G1]="G0"
    [A64]="A256 A512 A1024 AL32"
    [C1]="C2 C5"
    [C3]="C4"
    [A16]="A64 A256 A512 A1024"
    [M2]="M1 M1r"
)

source scripts/slurm/runs.sh
echo "bench job ${SLURM_JOB_ID:-local} on $(hostname) at $(date): $*"
for REP in "$@"; do
    load_run "$REP" || exit 2
    export MODEL
    # env.sh depends on MODEL, so source it per run (cheap)
    source scripts/slurm/env.sh || exit 1
    set -eo pipefail
    export HF_HUB_OFFLINE=1
    unset HF_TOKEN
    B="$WORK/bench"; mkdir -p "$B/runs" "$B/eval"
    [ "$METHOD" = none ] && { echo "ERROR: $REP is an eval-only run; bench a trained sibling instead" >&2; exit 2; }
    SIZE_FLAG=(--num_blocks "$SIZE"); [ "$METHOD" = lora ] && SIZE_FLAG=(--lora_r "$SIZE")
    echo; echo "════ $REP: $MODEL $TASK $METHOD $SIZE $PREC len=$MAXLEN eff=$EFF on $(nvidia-smi --query-gpu=name --format=csv,noheader)"

    # 1. memory probe
    ONLY_GC=(); [ "$GC_DEF" = 1 ] && ONLY_GC=(--only_gc)
    if [ ! -f "$B/${REP}_probe.json" ]; then
        python scripts/slurm/bench_probe.py --model "$MODEL" --task "$TASK" --data_dir "$DATA" --method "$METHOD" \
            --size "$SIZE" --modules "$MODULES" --precision "$PREC" --max_length "$MAXLEN" --eff "$EFF" \
            "${ONLY_GC[@]}" --out "$B/${REP}_probe.json"
    fi
    read -r PBS PGA PGC < <(python -c "import json;d=json.load(open('$B/${REP}_probe.json'));print(d['bs'],d['ga'],int(d['gc']))")

    # 2. training speed with that layout
    STEPS=12; [ "$TASK" = metamath ] && STEPS=6
    if [ ! -f "$B/runs/$REP/train_stats.json" ]; then
        rm -rf "$B/runs/$REP"
        GC_FLAG=(); [ "$PGC" = 1 ] && GC_FLAG=(--gradient_checkpointing)
        python train.py --model_name "$MODEL" --task "$TASK" --data_dir "$DATA" --output_dir "$B/runs" --run_name "$REP" \
            --method "$METHOD" "${SIZE_FLAG[@]}" --lr "$LR" --per_device_batch_size "$PBS" --gradient_accumulation_steps "$PGA" \
            --max_epochs "$EPOCHS" --warmup_steps 2 --target_modules "$MODULES" --precision "$PREC" --max_length "$MAXLEN" \
            --save_steps 100000 --max_steps "$STEPS" "${GC_FLAG[@]}" > "$B/runs/${REP}.log" 2>&1 \
            || { tail -40 "$B/runs/${REP}.log"; echo "ERROR: bench training of $REP failed" >&2; exit 1; }
    fi
    grep -E "Trainable|TRAIN_STATS" "$B/runs/${REP}.log" || true

    # 3. evaluation speed on a slice (the adapter from step 2 is near zero, but speed does not depend on it)
    rm -rf "$B/eval/$REP"     # timing needs a fresh slice (a resumed one would generate nothing)
    EV=(--model_name "$MODEL" --method "$METHOD" "${SIZE_FLAG[@]}" --target_modules "$MODULES"
        --adapter_path "$B/runs/$REP/adapter.chkpt" --precision "$PREC")
    case "$EVAL" in
        gsm8k_lori)  EVCMD=(evaluate_gsm8k.py "${EV[@]}" --data_dir "$DATA" --batch_size 32 --limit 64
                            --output_file "$B/eval/$REP/gsm8k.jsonl") ;;
        commonsense) EVCMD=(evaluate_commonsense.py "${EV[@]}" --data_dir "$SHARED_DATA" --batch_size 16 --limit 32
                            --output_dir "$B/eval/$REP") ;;
        math)        EVCMD=(evaluate_math.py "${EV[@]}" --data_dir "$SHARED_DATA" --batch_size 8 --limit 16
                            --output_dir "$B/eval/$REP") ;;
    esac
    python "${EVCMD[@]}" > "$B/eval/${REP}.log" 2>&1 \
        || { tail -40 "$B/eval/${REP}.log"; echo "ERROR: bench evaluation of $REP failed" >&2; exit 1; }
    grep -E "^[A-Za-z_-]+: [0-9.]+ \(|accuracy" "$B/eval/${REP}.log" | tail -10 || true

    # 4. projection for the representative and its siblings
    for ID in "$REP" ${FAMILY[$REP]:-}; do
        load_run "$ID"
        python - "$B" "$REP" "$ID" "$EVAL" "$EPOCHS" "$METHOD" "$SHARED_DATA" "$EFF" <<'EOF'
import glob, json, math, os, sys
B, rep, rid, ev, epochs, method, shared, eff = sys.argv[1:9]
probe = json.load(open(f"{B}/{rep}_probe.json"))
st = json.load(open(f"{B}/runs/{rep}/train_stats.json"))
steps = st["steps_per_epoch"] * int(epochs)
train_h = 0.0 if method == "none" else steps * st["s_per_step"] / 3600

# eval: seconds per generated example per dataset, times the full test size, + 3 min model load
if ev == "gsm8k_lori":
    s = json.load(open(f"{B}/eval/{rep}/gsm8k_summary.json"))
    eval_h = 1319 * s["seconds"] / max(s["generated"], 1) / 3600
else:
    s = json.load(open(glob.glob(f"{B}/eval/{rep}/summary*.json")[0]))
    eval_h = sum(d["n_total"] * d["seconds"] / max(d["generated"], 1) for d in s["datasets"].values()) / 3600
eval_h += 0.05
total = train_h + eval_h
out = dict(rep=rep, bs=probe["bs"], ga=int(eff) // probe["bs"], gc=probe["gc"], peak_gb=probe["peak_gb"],
           gpu=st["gpu"], s_per_step=round(st["s_per_step"], 2), steps=steps, train_h=round(train_h, 2),
           eval_h=round(eval_h, 2), total_h=round(total, 2))
json.dump(out, open(f"{B}/{rid}.json", "w"), indent=2)
print(f"BENCH {rid:6s} bs={out['bs']} ga={out['ga']} gc={int(out['gc'])} peak={out['peak_gb']}GB | "
      f"{out['s_per_step']} s/step x {steps} steps = {train_h:.1f} h train + {eval_h:.1f} h eval = {total:.1f} GPU-h")
EOF
    done
    set +eo pipefail
done

echo
echo "BENCH summary of everything measured so far (all *.json in $WORK/bench):"
python - "$WORK/bench" <<'EOF'
import glob, json, os, sys
rows = [json.load(open(p)) | {"id": os.path.basename(p)[:-5]} for p in sorted(glob.glob(f"{sys.argv[1]}/*.json"))
        if not p.endswith("_probe.json")]
for r in rows:
    print(f"BENCH-TOTAL {r['id']:6s} {r['total_h']:7.1f} GPU-h  ({r['gpu']})")
print(f"BENCH-TOTAL sum of benched runs: {sum(r['total_h'] for r in rows):.0f} GPU-h (before the ~30% margin)")
EOF
echo "BENCH OK at $(date)"
