# Multi-GPU Training with torchrun

This project now supports multi-GPU training using PyTorch's Distributed Data Parallel (DDP) and `torchrun`.

## Key Changes

The training script has been updated to:
- Use PyTorch DDP instead of Accelerate
- Support distributed data sampling across GPUs
- Handle mixed precision training with `torch.cuda.amp.GradScaler`
- Automatically detect multi-GPU vs single-GPU mode

## Requirements

- PyTorch 1.12+ with CUDA support
- Multiple NVIDIA GPUs
- NCCL backend (automatically used for CUDA)

## Usage

### Method 1: Using the launcher script

```bash
# Train on 2 GPUs (default)
./launch_training.sh

# Train on specific number of GPUs
./launch_training.sh 4

# Train on specific GPUs with custom dataset path
./launch_training.sh 2 /path/to/dataset
```

### Method 2: Using torchrun directly

```bash
# Train on 2 GPUs
torchrun \
    --nproc_per_node=2 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    train_llada_pretrained.py \
    --data_dir shakespeare_processed \
    --batch_size 1 \
    --epochs 5 \
    --fp16

# Train on 4 GPUs
torchrun \
    --nproc_per_node=4 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=12355 \
    train_llada_pretrained.py \
    --data_dir shakespeare_processed \
    --batch_size 1 \
    --epochs 5 \
    --fp16
```

### Method 3: Single GPU training (fallback)

The script automatically detects single-GPU mode if `torchrun` is not used:

```bash
python train_llada_pretrained.py \
    --data_dir shakespeare_processed \
    --batch_size 1 \
    --epochs 5 \
    --fp16
```

## Configuration

### Environment Variables

The script automatically reads these environment variables set by `torchrun`:
- `WORLD_SIZE`: Total number of processes
- `RANK`: Process rank (0 to WORLD_SIZE-1)

### Command Line Arguments

- `--data_dir`: Path to processed dataset
- `--model_name`: Hugging Face model name
- `--batch_size`: Batch size per GPU (total batch size = batch_size × num_gpus)
- `--epochs`: Number of training epochs
- `--learning_rate`: Learning rate
- `--save_dir`: Directory to save checkpoints
- `--fp16`: Enable mixed precision training (default: True)
- `--no_fp16`: Disable mixed precision training

## Performance Benefits

- **Data Parallelism**: Each GPU processes different batches simultaneously
- **Gradient Synchronization**: Gradients are automatically averaged across GPUs
- **Mixed Precision**: 16-bit training for faster computation and lower memory usage
- **Distributed Sampling**: Data is properly distributed across GPUs without overlap

## Memory Considerations

- **Batch Size**: The `--batch_size` argument specifies batch size per GPU
- **Total Memory**: Total memory usage ≈ batch_size × num_gpus × model_size
- **Gradient Accumulation**: For very large models, consider reducing per-GPU batch size

## Troubleshooting

### Common Issues

1. **NCCL Errors**: Ensure all GPUs are visible and CUDA drivers are up to date
2. **Port Conflicts**: Change `--master_port` if port 12355 is already in use
3. **Memory Issues**: Reduce `--batch_size` or disable `--fp16`

### Debug Mode

Add environment variable for more verbose output:
```bash
NCCL_DEBUG=INFO torchrun --nproc_per_node=2 ...
```

### Single Node vs Multi-Node

This setup is configured for single-node multi-GPU training. For multi-node training, modify:
- `--nnodes`: Number of nodes
- `--node_rank`: Rank of current node
- `--master_addr`: IP address of master node

## Example Output

```
Available GPUs: 4
Starting multi-GPU training with torchrun...
==============================================
Dataset: shakespeare_processed
Model: GSAI-ML/LLaDA-8B-Instruct
Batch size per GPU: 1
Total batch size: 4
Epochs: 5
Learning rate: 1e-05
Save directory: checkpoints_pretrained
Number of GPUs: 4
==============================================
Loading pre-trained model: GSAI-ML/LLaDA-8B-Instruct
✓ Tokenizer loaded successfully
✓ Model loaded successfully
✓ 16-bit mixed precision training enabled
Vocabulary size: 126336
Starting training for 5 epochs...
Model parameters: 8,000,000,000
Training on 4 GPUs
```

## Checkpoint Loading

When resuming training, the script automatically handles DDP model loading:

```python
checkpoint = torch.load('checkpoint_epoch_5.pt')
model.module.load_state_dict(checkpoint['model_state_dict'])
```

The `model.module` access is required because the model is wrapped with DDP.
