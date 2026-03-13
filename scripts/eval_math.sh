#!/bin/bash
# Evaluate DiaBlo on math reasoning benchmarks (gsm8k, MATH)
# Set ADAPTER_PATH to your trained checkpoint

ADAPTER_PATH=${1:?"Usage: bash scripts/eval_math.sh <adapter_path>"}
OUTPUT_DIR=${2:-"./eval_results"}

model_name="Llama2-7B"
num_blocks=64
precision="bf16"
target_modules="qkvud"
batch_size=4

mkdir -p ${OUTPUT_DIR}

for dataset in gsm8k MATH; do
    echo "Evaluating on ${dataset}..."
    python evaluate_math.py \
        --model_name ${model_name} \
        --dataset ${dataset} \
        --adapter_path ${ADAPTER_PATH} \
        --output_dir ${OUTPUT_DIR} \
        --num_blocks ${num_blocks} \
        --target_modules ${target_modules} \
        --precision ${precision} \
        --batch_size ${batch_size} \
        | tee ${OUTPUT_DIR}/${dataset}.txt
done
