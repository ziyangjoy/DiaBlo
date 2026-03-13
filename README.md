# DiaBlo: Diagonal Blocks Are Sufficient For Fine-tuning

This repository contains the source code for **DiaBlo**, a parameter-efficient fine-tuning method that uses block-diagonal weight updates to adapt large language models.

## Method Overview

DiaBlo replaces the full weight update matrix with a block-diagonal structure. Each linear layer receives a trainable block-diagonal adapter `BlockLinear`, which adds a block-diagonal low-rank update to the frozen pretrained weights. This achieves competitive performance with significantly fewer trainable parameters compared to full fine-tuning.

Key features:
- **Block-diagonal adapters**: Structured weight updates with `num_blocks` diagonal blocks
- **Adapter merging**: Trained adapters can be merged back into the base model for zero-overhead inference
- **Multiple benchmarks**: Evaluated on GSM8k, MetaMath, MATH, and 8 commonsense reasoning tasks

## Installation

```bash
pip install -r requirements.txt
```

Set your HuggingFace token:
```bash
export HF_TOKEN="your_token_here"
```

Optionally set a model cache directory:
```bash
export MODEL_CACHE_DIR="/path/to/model/cache"
```

## Repository Structure

```
DiaBlo/
├── diablo/                         # Core DiaBlo module
│   ├── __init__.py
│   ├── block_linear.py             # BlockLinear layer implementation
│   └── trainer.py                  # Custom Trainer with adapter saving
├── train.py                        # Main training script
├── merge.py                        # Merge adapters into base model
├── evaluate_gsm8k.py               # GSM8k evaluation
├── evaluate_math.py                # Math reasoning evaluation
├── evaluate_commonsense.py         # Commonsense reasoning evaluation
├── utils_math.py                   # MATH dataset evaluation utilities
├── data_processing/                # Dataset preprocessing scripts
│   ├── process_gsm8k.py
│   ├── process_commonsense.py
│   └── process_metamath.py
├── scripts/                        # Example training & evaluation scripts
│   ├── train_gsm8k.sh
│   ├── train_metamath.sh
│   ├── train_commonsense.sh
│   ├── eval_gsm8k.sh
│   ├── eval_math.sh
│   └── eval_commonsense.sh
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Prepare Data

```bash
# GSM8k (downloads automatically from HuggingFace)
python data_processing/process_gsm8k.py --output_dir ./datasets

# Commonsense 170k (requires commonsense_170k.json from LLM-Adapters)
python data_processing/process_commonsense.py \
    --data_path ./datasets/commonsense_170k.json \
    --output_dir ./datasets \
    --model_name Llama2-7B

# MetaMath (downloads automatically from HuggingFace)
python data_processing/process_metamath.py --output_dir ./datasets
```

### 2. Train

```bash
# Train DiaBlo on GSM8k with 64 diagonal blocks
bash scripts/train_gsm8k.sh

# Or run directly with custom settings:
python train.py \
    --model_name Llama2-7B \
    --task GSM8k \
    --num_blocks 64 \
    --lr 5e-4 \
    --max_epochs 3 \
    --precision bf16
```

### 3. Evaluate

```bash
# Evaluate on GSM8k
bash scripts/eval_gsm8k.sh path/to/adapter.chkpt

# Evaluate on commonsense benchmarks
bash scripts/eval_commonsense.sh path/to/adapter.chkpt ./eval_results

# Evaluate on math benchmarks
bash scripts/eval_math.sh path/to/adapter.chkpt ./eval_results
```

### 4. Merge Adapter (Optional)

Merge the trained adapter into the base model for deployment:

```bash
python merge.py \
    --model_name Llama2-7B \
    --adapter_path path/to/adapter.chkpt \
    --output_dir ./merged_model \
    --num_blocks 64
```

## Key Arguments

| Argument | Description | Default |
|---|---|---|
| `--model_name` | Base model (Llama2-7B, Llama2-13B, Llama3-8B, etc.) | Llama2-7B |
| `--num_blocks` | Number of diagonal blocks | 64 |
| `--target_modules` | Modules to adapt (q/k/v/u/d/o/g) | qkvud |
| `--lr` | Learning rate | 5e-4 |
| `--precision` | Training precision (fp32/fp16/bf16) | bf16 |
| `--save_adapter_only` | Save only adapter weights | True |


## Citation

```bibtex
@article{gurses2025diablo,
  title={DiaBlo: Diagonal Blocks Are Sufficient For Finetuning},
  author={Gurses, Selcuk and Zhang, Aozhong and Deng, Yanxia and Dong, Xun and Li, Xin and Wang, Naigang and Yin, Penghang and Yang, Zi},
  journal={International Conference on Learning Representations},
  year={2026}
}
```
