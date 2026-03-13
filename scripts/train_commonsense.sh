#!/bin/bash
# Train DiaBlo on Commonsense 170k dataset

model_name="Llama2-7B"
num_blocks=64
lr=2.5e-4
max_epochs=3
dropout=0.05
per_device_batch_size=4
gradient_accumulation_steps=4
target_modules="qkvud"
precision="fp16"
warmup_steps=100
max_length=256

python train.py \
    --model_name ${model_name} \
    --task commonsense_new \
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
    --save_adapter_only \
    --save_strategy epoch
