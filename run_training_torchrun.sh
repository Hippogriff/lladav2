#!/bin/bash

# Multi-GPU Training Launcher using torchrun
# This script launches the LLaDA training on multiple GPUs

# Configuration
DATA_DIR="shakespeare_processed"  # Change this to your dataset path
MODEL_NAME="GSAI-ML/LLaDA-8B-Instruct"
BATCH_SIZE=1  # Batch size per GPU
EPOCHS=5
LEARNING_RATE=1e-5
SAVE_DIR="checkpoints_pretrained"
NUM_GPUS=2  # Change this to match your available GPUs

# Check if CUDA is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: CUDA not available. Please check your NVIDIA drivers and CUDA installation."
    exit 1
fi

# Check available GPUs
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "Available GPUs: $AVAILABLE_GPUS"

if [ $AVAILABLE_GPUS -lt $NUM_GPUS ]; then
    echo "Warning: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS are available."
    echo "Adjusting to use $AVAILABLE_GPUS GPUs."
    NUM_GPUS=$AVAILABLE_GPUS
fi

# Check if dataset directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Dataset directory '$DATA_DIR' not found."
    echo "Please ensure you have processed the Shakespeare dataset first."
    exit 1
fi

echo "Starting multi-GPU training with torchrun..."
echo "=============================================="
echo "Dataset: $DATA_DIR"
echo "Model: $MODEL_NAME"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Total batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "Epochs: $EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Save directory: $SAVE_DIR"
echo "Number of GPUs: $NUM_GPUS"
echo "=============================================="

# Launch training with torchrun
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    train_llada_pretrained.py \
    --data_dir "$DATA_DIR" \
    --model_name "$MODEL_NAME" \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --learning_rate $LEARNING_RATE \
    --save_dir "$SAVE_DIR" \
    --fp16

echo "Training completed!"
