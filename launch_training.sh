#!/bin/bash

# Simple Multi-GPU Training Launcher
# Usage: ./launch_training.sh [num_gpus] [data_dir]

NUM_GPUS=${1:-2}  # Default to 2 GPUs
DATA_DIR=${2:-"shakespeare_processed"}  # Default dataset path

echo "Launching training on $NUM_GPUS GPUs with dataset: $DATA_DIR"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    train_llada_pretrained.py \
    --data_dir "$DATA_DIR" \
    --batch_size 1 \
    --epochs 5 \
    --fp16
