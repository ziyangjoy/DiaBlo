#!/bin/bash
# Train DiaBlo on MetaMath dataset

model_name="Llama2-7B"
num_blocks=64
lr=12e-4
max_epochs=1
dropout=0.05
per_device_batch_size=4
gradient_accumulation_steps=8
target_modules="qkvud"
precision="bf16"
warmup_steps=100
max_length=2048
scheduler="cosine"

python train.py \
    --model_name ${model_name} \
    --task metamath \
    --num_blocks ${num_blocks} \
    --lr ${lr} \
    --max_epochs ${max_epochs} \
    --dropout ${dropout} \
    --per_device_batch_size ${per_device_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --target_modules ${target_modules} \
    --precision ${precision} \
    --warmup_steps ${warmup_steps} \
    --max_length ${max_length} \
    --scheduler ${scheduler} \
    --save_adapter_only \
    --save_strategy epoch \
    --gradient_checkpointing
