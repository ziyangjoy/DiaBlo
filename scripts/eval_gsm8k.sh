#!/bin/bash
# Evaluate DiaBlo on GSM8k
# Set ADAPTER_PATH to your trained checkpoint

ADAPTER_PATH=${1:?"Usage: bash scripts/eval_gsm8k.sh <adapter_path>"}

model_name="Llama2-7B"
num_blocks=64
precision="bf16"
target_modules="qkvud"
batch_size=16

python evaluate_gsm8k.py \
    --model_name ${model_name} \
    --adapter_path ${ADAPTER_PATH} \
    --num_blocks ${num_blocks} \
    --target_modules ${target_modules} \
    --precision ${precision} \
    --batch_size ${batch_size}
