#!/bin/bash

# Script to run LLaDA training with different Accelerate configurations

echo "LLaDA Training Scripts"
echo "======================"
echo ""

# Check if data directory is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <data_directory> [options]"
    echo ""
    echo "Examples:"
    echo "  $0 ./shakespeare_data"
    echo "  $0 ./shakespeare_data --fp16 --batch_size 2"
    echo "  $0 ./shakespeare_data --bf16 --gradient_accumulation_steps 4"
    echo ""
    echo "For distributed training:"
    echo "  accelerate launch --config_file accelerate_config.yaml train_llada_pretrained.py --data_dir ./shakespeare_data"
    exit 1
fi

DATA_DIR=$1
shift

echo "Data directory: $DATA_DIR"
echo "Additional options: $@"
echo ""

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Data directory '$DATA_DIR' does not exist!"
    exit 1
fi

echo "Starting training with Accelerate..."
echo ""

# Run training with provided options
python train_llada_pretrained.py --data_dir "$DATA_DIR" "$@"

echo ""
echo "Training completed!"
echo ""
echo "To run with distributed training on multiple GPUs:"
echo "  accelerate launch --config_file accelerate_config.yaml train_llada_pretrained.py --data_dir $DATA_DIR"
echo ""
echo "To run with specific GPU configuration:"
echo "  CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config.yaml train_llada_pretrained.py --data_dir $DATA_DIR"
