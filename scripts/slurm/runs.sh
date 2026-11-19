#!/usr/bin/env bash
# Run table, sourced by run.sbatch, bench.sbatch and submit_all.sh (single source of truth).
#   source scripts/slurm/runs.sh; load_run C1     -> sets every field below for run C1
#
# Fields: MODEL TASK METHOD SIZE LR SEED EPOCHS WARMUP MODULES EFF (effective batch) MAXLEN PREC SCHED
#         GPU MEM EVAL (gsm8k_lori | commonsense | math) LOAD4 SAVE_STEPS SEGMENTS (24 h jobs to chain)
#         PAPER (the number to compare against, for the results table)
# Per-device batch size / accumulation / gradient checkpointing are NOT fixed here: bench.sbatch measures what
# fits on the GPU and writes them to $WORK/bench/<ID>.json. Before that, BS=2 and GA=EFF/2. BS/GA/GC from
# the environment win over both.
#
# Groups (argument to submit_all.sh / bench.sbatch):
#   first   R*  first wave: Qwen2.5-1.5B, repo's train_gsm8k.sh recipe (3 epochs, batch 16, warmup 100, qkvud)
#   paper   P* S*  Table 20 recipe on Qwen2.5-1.5B / 0.5B
#   llama   L* H*  4-bit NF4 Llama (superseded by the full-precision groups below)
#   --- the paper's own experiments (Llama, full 16-bit base, a5000 unless noted) ---
#   gsm8k   G*  Table 5 DiaBlo row + Fig. 5 (LR and N sweep): Llama3-8B on GSM8K, Table 20 recipe
#   cs      C*  Table 1 (FP16): Llama2-7B and Llama3-8B on Commonsense-170k, Table 12 recipe
#   ablate  A*  Table 10: N ablation on Llama3.2-3B, Commonsense-170k, FP16 recipe (rtx2080ti)
#   mm      M*  Table 2: Llama2-7B on MetaMathQA, Table 14 recipe
#   mvr     minimum viable reproduction: only runs with a published number to match, ~1 week of wall clock:
#           G1 (Table 5, Llama3-8B GSM8K 67.68) + A64/A512/A1024 (Table 10, Llama3.2-3B: 84.44 / 82.54 / 81.60)
#   optional  AL32 (LoRA r=32 on Llama3.2-3B, no paper number), A16, C5 (LoRA, Llama2-7B), M1r (MetaMath, released recipe)

DEFAULT_ORDER=(R0 R1 R2 R3 R8 R4 R9 R10 R7 R11 R12 R5a R5b)
PAPER_ORDER=(P1 P2 S0 S1 S2)
LLAMA_ORDER=(L0 L1 L2 H1 H2)
GSM8K_ORDER=(G1 G0 G2 G3 G4 G5 G6 G7)
CS_ORDER=(C3 C1 C4 C2)
ABLATE_ORDER=(A64 A512 A1024 A256 A16)
MM_ORDER=(M1 M2)
OPTIONAL_ORDER=(AL32 A16 C5 M1r)
MVR_ORDER=(G1 A64 A512 A1024)
ALL_PAPER=("${GSM8K_ORDER[@]}" "${CS_ORDER[@]}" "${ABLATE_ORDER[@]}" "${MM_ORDER[@]}" "${OPTIONAL_ORDER[@]}")

load_run() {
    local id="$1"
    # First-wave defaults
    MODEL=Qwen2.5-1.5B; TASK=GSM8k; METHOD=; SIZE=; LR=; SEED=42; EPOCHS=3; WARMUP=100; MODULES=qkvud; EFF=16
    MAXLEN=512; PREC=fp16; SCHED=linear; GPU=rtx2080ti; MEM=6G; EVAL=gsm8k_lori; LOAD4=0; SAVE_STEPS=100
    SEGMENTS=1; PAPER=; PAPER_NUM=; GC_DEF=0
    case "$id" in
        R0)  METHOD=none ;;                                   # base model, eval only
        R1)  METHOD=diablo; SIZE=64;  LR=5e-4 ;;
        R2)  METHOD=lora;   SIZE=16;  LR=2e-4 ;;
        R3)  METHOD=diablo; SIZE=32;  LR=5e-4 ;;
        R4)  METHOD=diablo; SIZE=128; LR=5e-4 ;;
        R5a) METHOD=diablo; SIZE=64;  LR=5e-4; SEED=1 ;;
        R5b) METHOD=lora;   SIZE=16;  LR=2e-4; SEED=1 ;;
        R7)  METHOD=diablo; SIZE=64;  LR=2e-4 ;;
        R8)  METHOD=lora;   SIZE=32;  LR=2e-4 ;;
        R9)  METHOD=diablo; SIZE=512; LR=5e-4 ;;
        R10) METHOD=diablo; SIZE=16;  LR=5e-4 ;;
        R11) METHOD=diablo; SIZE=64;  LR=1e-4 ;;
        R12) METHOD=diablo; SIZE=64;  LR=1e-3 ;;
        # paper recipe on the same model (LoRA keeps the first-wave lr 2e-4: the paper gives no LoRA lr for this setup)
        P1)  METHOD=diablo; SIZE=64;  LR=4e-4; EPOCHS=2; WARMUP=20; MODULES=qkvogud; EFF=32 ;;
        P2)  METHOD=lora;   SIZE=16;  LR=2e-4; EPOCHS=2; WARMUP=20; MODULES=qkvogud; EFF=32 ;;
        S0)  MODEL=Qwen2.5-0.5B; METHOD=none ;;
        S1)  MODEL=Qwen2.5-0.5B; METHOD=diablo; SIZE=64; LR=4e-4; EPOCHS=2; WARMUP=20; MODULES=qkvogud; EFF=32 ;;
        S2)  MODEL=Qwen2.5-0.5B; METHOD=lora;   SIZE=16; LR=2e-4; EPOCHS=2; WARMUP=20; MODULES=qkvogud; EFF=32 ;;
        L0)  MODEL=Llama3-3B; METHOD=none; LOAD4=1 ;;
        L1)  MODEL=Llama3-3B; METHOD=diablo; SIZE=64; LR=4e-4; EPOCHS=2; WARMUP=20; MODULES=qkvogud; EFF=32; LOAD4=1 ;;
        L2)  MODEL=Llama3-3B; METHOD=lora;   SIZE=16; LR=2e-4; EPOCHS=2; WARMUP=20; MODULES=qkvogud; EFF=32; LOAD4=1 ;;
        H1)  MODEL=Llama3-8B; METHOD=diablo; SIZE=64; LR=4e-4; EPOCHS=2; WARMUP=20; MODULES=qkvogud; EFF=32; LOAD4=1 ;;
        H2)  MODEL=Llama3-8B; METHOD=lora;   SIZE=16; LR=2e-4; EPOCHS=2; WARMUP=20; MODULES=qkvogud; EFF=32; LOAD4=1 ;;

        # ── Paper Table 5 / Fig. 5: Llama3-8B on GSM8K (Table 20: lr 4e-4, 2 epochs, batch 32, warmup 20,
        #    max len 512, all 7 modules). bf16 (the A5000 supports it; Table 20 gives no precision).
        #    The paper does not list Fig. 5's LR values; this is a doubling grid around 4e-4.
        G0|G1|G2|G3|G4|G5|G6|G7)
            MODEL=Llama3-8B; TASK=GSM8k; METHOD=diablo; SIZE=64; LR=4e-4; EPOCHS=2; WARMUP=20; MODULES=qkvogud
            EFF=32; MAXLEN=512; PREC=bf16; GPU=a5000; MEM=40G; SAVE_STEPS=50; EVAL=gsm8k_lori
            case "$id" in
                G0) METHOD=none; PAPER="base model (not in paper)" ;;
                G1) PAPER="Table 5 DiaBlo 1/64: 67.68"; PAPER_NUM=67.68 ;;
                G2) LR=1e-4; PAPER="Fig. 5 (LR sweep)" ;;
                G3) LR=2e-4; PAPER="Fig. 5 (LR sweep)" ;;
                G4) LR=8e-4; PAPER="Fig. 5 (LR sweep)" ;;
                G5) LR=1.6e-3; PAPER="Fig. 5 (LR sweep)" ;;
                G6) SIZE=32; PAPER="Fig. 5 (N sweep); Table 5 1/32 retained-update: 67.30" ;;
                G7) SIZE=128; PAPER="Fig. 5 (N sweep); Table 5 1/128 retained-update: 63.11" ;;
            esac ;;

        # ── Paper Table 1 (FP16) with Table 12 hyper-parameters: 3 epochs, batch 16, warmup 100, max len 256,
        #    q k v up down, linear schedule. fp16 = fp16 base + fp32 adapters, as in scripts/train_commonsense.sh.
        C1|C2|C3|C4|C5)
            TASK=commonsense_new; EPOCHS=3; WARMUP=100; MODULES=qkvud; EFF=16; MAXLEN=256; PREC=fp16
            GPU=a5000; MEM=40G; SAVE_STEPS=200; EVAL=commonsense; SEGMENTS=3
            case "$id" in
                C1) MODEL=Llama2-7B; METHOD=diablo; SIZE=64;  LR=2e-4; PAPER="Table 1 LLaMA2-7B N=64: avg 83.4"; PAPER_NUM=83.4 ;;
                C2) MODEL=Llama2-7B; METHOD=diablo; SIZE=128; LR=3e-4; PAPER="Table 1 LLaMA2-7B N=128: avg 83.5"; PAPER_NUM=83.5 ;;
                C3) MODEL=Llama3-8B; METHOD=diablo; SIZE=64;  LR=5e-5; PAPER="Table 1 LLaMA3-8B N=64: avg 87.3"; PAPER_NUM=87.3 ;;
                C4) MODEL=Llama3-8B; METHOD=diablo; SIZE=128; LR=1e-4; PAPER="Table 1 LLaMA3-8B N=128: avg 87.2"; PAPER_NUM=87.2 ;;
                # Optional same-pipeline baseline. lr 3e-4 is LLM-Adapters' LoRA default; alpha = 2r.
                C5) MODEL=Llama2-7B; METHOD=lora;   SIZE=32;  LR=3e-4; PAPER="Table 1 LLaMA2-7B LoRA r=32: avg 77.6 (from DoRA)" ;;
            esac ;;

        # ── Paper Table 10: N ablation on Llama3.2-3B, commonsense. The table's parameter counts match q k v up down
        #    (e.g. N=64 -> 27.56 Mi), i.e. the Table 12 FP16 recipe; its LR is not given, so one LR for every N.
        A16|A64|A256|A512|A1024|AL32)
            MODEL=Llama3.2-3B; TASK=commonsense_new; METHOD=diablo; SIZE="${id#A}"; LR=1e-4; EPOCHS=3; WARMUP=100
            MODULES=qkvud; EFF=16; MAXLEN=256; PREC=fp16; GPU=rtx2080ti; MEM=16G; SAVE_STEPS=200
            EVAL=commonsense; SEGMENTS=3
            case "$id" in
                A16) PAPER="Table 10 N=16: avg 84.38"; PAPER_NUM=84.38 ;;
                A64) PAPER="Table 10 N=64: avg 84.44"; PAPER_NUM=84.44 ;;
                A256) PAPER="Table 10 N=256: avg 83.92"; PAPER_NUM=83.92 ;;
                A512) PAPER="Table 10 N=512: avg 82.54"; PAPER_NUM=82.54 ;;
                A1024) PAPER="Table 10 N=1024: avg 81.60"; PAPER_NUM=81.60 ;;
                # Same-pipeline LoRA baseline (~33M params vs 28.9M for N=64). lr 3e-4 = LLM-Adapters' LoRA default.
                AL32) METHOD=lora; SIZE=32; LR=3e-4; PAPER="no paper number (LoRA baseline for DiaBlo N=64)" ;;
            esac ;;

        # ── Paper Table 2 with Table 14 hyper-parameters: Llama2-7B on MetaMathQA (395k), lr 2e-4, 3 epochs, batch 16,
        #    warmup 100, max len 2048, q k v up down, linear. bf16 as in scripts/train_metamath.sh.
        #    M1r is the fallback if the bench shows Table 14 is too expensive: the authors' released
        #    scripts/train_metamath.sh recipe (1 epoch, lr 1.2e-3, batch 32, cosine), about 1/3 of the cost.
        M1|M2|M1r)
            MODEL=Llama2-7B; TASK=metamath; METHOD=diablo; LR=2e-4; EPOCHS=3; WARMUP=100; MODULES=qkvud; EFF=16
            MAXLEN=2048; PREC=bf16; GPU=a5000; MEM=48G; SAVE_STEPS=200; EVAL=math; SEGMENTS=9; GC_DEF=1
            case "$id" in
                M1) SIZE=64; PAPER="Table 2 N=64: GSM8K 66.5, MATH 17.6" ;;
                M2) SIZE=32; PAPER="Table 2 N=32: GSM8K 66.3, MATH 20.4" ;;
                M1r) SIZE=64; LR=1.2e-3; EPOCHS=1; EFF=32; SCHED=cosine; SEGMENTS=4
                     PAPER="Table 2 N=64: GSM8K 66.5, MATH 17.6 (released-script recipe)" ;;
            esac ;;

        *)   echo "ERROR: unknown run id '$id'" >&2; return 2 ;;
    esac

    # Batch layout: environment > bench result > default (BS=2 for the old first-wave runs' GA_DEF behaviour).
    local work="${DIABLO_WORK:-$HOME/diablo}" bench="${DIABLO_WORK:-$HOME/diablo}/bench/$id.json"
    local b_bs="" b_gc=""
    if [ -f "$bench" ]; then
        b_bs="$(python3 -c "import json;print(json.load(open('$bench'))['bs'])" 2>/dev/null || true)"
        b_gc="$(python3 -c "import json;print(int(json.load(open('$bench'))['gc']))" 2>/dev/null || true)"
    fi
    RUN_BS="${BS:-${b_bs:-2}}"
    RUN_GA="${GA:-$((EFF / RUN_BS))}"
    RUN_GC="${GC:-${b_gc:-$GC_DEF}}"
}

# Where a run's MODEL-specific folders live (same rule as env.sh), without sourcing conda.
run_dirs() {
    local work="${DIABLO_WORK:-$HOME/diablo}"
    if [ "$MODEL" = "Qwen2.5-1.5B" ]; then R_RESULTS="$work/results"; R_EVALS="$work/eval"
    else R_RESULTS="$work/results-$MODEL"; R_EVALS="$work/eval-$MODEL"; fi
}

# The file whose existence means "this run is completely finished" (train + eval).
eval_summary_path() {
    local id="$1"
    load_run "$id" || return 2
    run_dirs
    if [ "$EVAL" = gsm8k_lori ]; then echo "$R_EVALS/${id}_hf_summary.json"
    else echo "$R_EVALS/$id/summary.json"; fi
}

# Tab-separated run table for scripts/paper_results.py: id model task method size lr eval paper paper_num
dump_runs() {
    local id
    for id in "$@"; do
        load_run "$id" >/dev/null || continue
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$MODEL" "$TASK" "$METHOD" "$SIZE" "$LR" "$EVAL" "$PAPER" "$PAPER_NUM"
    done
}
