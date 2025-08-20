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
    echo "For distributed training (if you encounter NCCL issues, use single-machine config):"
    echo "  # Multi-GPU distributed (may have NCCL issues):"
    echo "  accelerate launch --config_file accelerate_config.yaml train_llada_pretrained.py --data_dir ./shakespeare_data"
    echo ""
    echo "  # Single-machine multi-GPU (more reliable, avoids NCCL issues):"
    echo "  accelerate launch --config_file accelerate_config_single_machine.yaml train_llada_pretrained.py --data_dir ./shakespeare_data"
    echo ""
    echo "  # Single GPU (most reliable):"
    echo "  CUDA_VISIBLE_DEVICES=0 python train_llada_pretrained.py --data_dir ./shakespeare_data"
    echo ""
    echo "Troubleshooting NCCL issues:"
    echo "  1. Try single-machine config first: accelerate_config_single_machine.yaml"
    echo "  2. If still having issues, use single GPU: CUDA_VISIBLE_DEVICES=0"
    echo "  3. Check GPU drivers and CUDA installation"
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

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
    echo ""
fi

echo "Starting training with Accelerate..."
echo ""

# Run training with provided options
python train_llada_pretrained.py --data_dir "$DATA_DIR" "$@"

echo ""
echo "Training completed!"
echo ""
echo "If you encountered NCCL errors, try these alternatives:"
echo ""
echo "1. Single-machine multi-GPU (recommended for most setups):"
echo "   accelerate launch --config_file accelerate_config_single_machine.yaml train_llada_pretrained.py --data_dir $DATA_DIR"
echo ""
echo "2. Single GPU (most reliable):"
echo "   CUDA_VISIBLE_DEVICES=0 python train_llada_pretrained.py --data_dir $DATA_DIR"
echo ""
echo "3. Multi-GPU with specific devices:"
echo "   CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file accelerate_config_single_machine.yaml train_llada_pretrained.py --data_dir $DATA_DIR"
echo ""
echo "4. Debug NCCL issues:"
echo "   NCCL_DEBUG=INFO accelerate launch --config_file accelerate_config.yaml train_llada_pretrained.py --data_dir $DATA_DIR"
