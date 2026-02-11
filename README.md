# OpenPI Fine-tuning Training

This repository provides a fine-tuning pipeline for OpenPI based on the official implementation.

## Environment Setup

```bash
cd /home
git clone https://github.com/D-Robotics/openpi.git
cd openpi

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

wandb online
```

## Dataset Download

```bash
wget https://huggingface.co/D-Robotics/openpi/tree/main/base_models
```

## Model Download

```bash
# Create model directory
mkdir -p models/openpi_touch_models

# Clone model to specified directory
git clone https://huggingface.co/D-Robotics/openpi models/openpi_touch_models/pi0_put_the_yellow_mango_on_the_blue_plate
```

## Precomputing Normalization Statistics

```bash
# Compute normalization statistics
CUDA_VISIBLE_DEVICES=0 uv run scripts/compute_norm_stats.py --config-name pi0_piper_finetune_put_the_yellow_mango_on_the_blue_plate
```

## Fine-tuning Training

```bash
# Single GPU training
CUDA_VISIBLE_DEVICES=0 uv run scripts/train_pytorch.py pi0_piper_lora_finetune_put_the_yellow_mango_on_the_blue_plate --exp_name single_gpu_test

# Multi-GPU training (4 GPUs)
torchrun --standalone --nnodes=1 --nproc_per_node=4 scripts/train_pytorch.py pi0_piper_lora_finetune_put_the_yellow_mango_on_the_blue_plate --exp_name multi_gpu_test
```

