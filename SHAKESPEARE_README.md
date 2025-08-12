# Shakespeare Dataset for LLaDA Training

This directory contains scripts to download, preprocess, and train LLaDA (Large Language Diffusion Models) on the Shakespeare dataset.

## Overview

The scripts implement the LLaDA training process as described in the [GUIDELINES.md](../GUIDELINES.md) file. LLaDA is a diffusion-based language model that uses masking instead of autoregressive generation.

## Files

- `download_shakespeare_dataset.py` - Downloads and preprocesses the Shakespeare dataset
- `train_llada_shakespeare.py` - Training script for LLaDA (from scratch)
- `train_llada_pretrained.py` - Fine-tuning script for pre-trained LLaDA models
- `test_transformer.py` - Test script for the custom transformer implementation
- `test_pretrained_model.py` - Test script for pre-trained model loading
- `requirements.txt` - Python dependencies for training from scratch
- `requirements_pretrained.txt` - Python dependencies for pre-trained models
- `SHAKESPEARE_README.md` - This file

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test Transformer Implementation

```bash
# Test custom transformer
python test_transformer.py

# Test pre-trained model loading
python test_pretrained_model.py
```

### 3. Download and Preprocess Dataset

```bash
python download_shakespeare_dataset.py --output_dir shakespeare_dataset
```

This will:
- Download the Shakespeare dataset from Andrej Karpathy's repository
- Build a character-level vocabulary
- Create training sequences of length 4096 (configurable)
- Split data into training (90%) and validation (10%) sets
- Save processed data in numpy format

### 4. Train LLaDA

```bash
# Train from scratch
python train_llada_shakespeare.py --data_dir shakespeare_dataset --epochs 10 --batch_size 4

# Fine-tune pre-trained model
python train_llada_pretrained.py --data_dir shakespeare_dataset --epochs 5 --batch_size 1
```

## Dataset Format

The processed dataset follows the LLaDA requirements:

- **Sequence Length**: 4096 tokens (configurable)
- **Vocabulary**: Character-level with special tokens
- **Special Tokens**:
  - `<PAD>` (0): Padding token
  - `<BOS>` (1): Beginning of sequence
  - `<EOS>` (2): End of sequence
  - `<MASK>` (126336): Mask token for diffusion

## LLaDA Training Process

The training implements the forward process described in GUIDELINES.md:

1. **Random Masking**: Each sequence gets a random masking ratio between 0 and 1
2. **Noise Addition**: Tokens are replaced with `<MASK>` tokens based on the masking probability
3. **Model Prediction**: The model predicts the original tokens at masked positions
4. **Loss Computation**: Cross-entropy loss with importance weighting by masking probability

## Training Options

### **1. Training from Scratch**
Use `train_llada_shakespeare.py` to train a custom transformer from scratch.

### **2. Fine-tuning Pre-trained Models**
Use `train_llada_pretrained.py` to fine-tune the official LLaDA-8B-Instruct model.

**Advantages of Fine-tuning:**
- **Pre-trained Knowledge**: Leverages 8B parameters of pre-trained knowledge
- **Faster Convergence**: Requires fewer epochs (5 vs 10+)
- **Better Performance**: Higher quality results with less data
- **Production Ready**: Based on the official LLaDA implementation

**Requirements:**
- More GPU memory (16GB+ recommended)
- Smaller batch sizes (1-2)
- Lower learning rates (1e-5)
- Transformers library

## Model Architecture

- **Custom Attention**: Uses `scaled_dot_product_attention` without causal masking
- **Bidirectional Processing**: Full attention across all positions (non-autoregressive)
- **Embedding Layer**: Token and positional embeddings with layer normalization
- **Transformer Encoder**: Multi-head self-attention layers with GELU activation
- **Layer Normalization**: Applied to embeddings, attention, and final output
- **Dropout**: Configurable dropout for regularization
- **Output Projection**: Linear layer to vocabulary size

**Default Architecture:**
- **Layers**: 24 transformer layers
- **Dimensions**: 768 model dimensions
- **Heads**: 12 attention heads
- **Dropout**: 0.1

**Key Features:**
- **No Causal Masking**: Perfect for LLaDA's diffusion approach
- **Bidirectional Attention**: Each token can attend to all other tokens
- **Efficient Implementation**: Uses PyTorch's optimized `scaled_dot_product_attention`
- **Residual Connections**: Proper skip connections throughout the network

This architecture provides a good balance between model capacity and training efficiency. For production use, you can scale up to match the full LLaDA architecture.

## Configuration Options

### Dataset Download

- `--output_dir`: Output directory for processed dataset
- `--sequence_length`: Sequence length for training (default: 4096)

### Training

- `--data_dir`: Directory containing processed dataset
- `--batch_size`: Training batch size (default: 4)
- `--epochs`: Number of training epochs (default: 10)
- `--d_model`: Model dimension (default: 768)
- `--n_heads`: Number of attention heads (default: 12)
- `--n_layers`: Number of transformer layers (default: 24)
- `--dropout`: Dropout rate (default: 0.1)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--device`: Device to use (auto, cpu, cuda)

## Output Files

After running the download script, you'll get:

- `shakespeare_raw.txt` - Raw Shakespeare text
- `vocab.json` - Vocabulary mapping and metadata
- `train_data.npy` - Training sequences
- `val_data.npy` - Validation sequences
- `metadata.json` - Dataset statistics and configuration

After training, checkpoints are saved to the `checkpoints/` directory.

## Example Usage

### Training from Scratch

```bash
# Download dataset
python download_shakespeare_dataset.py

# Train with default architecture (24 layers, 768 dimensions)
python train_llada_shakespeare.py \
    --data_dir shakespeare_dataset \
    --epochs 20 \
    --batch_size 8

# Train with custom architecture
python train_llada_shakespeare.py \
    --data_dir shakespeare_dataset \
    --epochs 20 \
    --batch_size 8 \
    --d_model 1024 \
    --n_heads 16 \
    --n_layers 32 \
    --dropout 0.15
```

### Fine-tuning Pre-trained Model

```bash
# Download dataset
python download_shakespeare_dataset.py

# Fine-tune with default settings
python train_llada_pretrained.py \
    --data_dir shakespeare_dataset \
    --epochs 5 \
    --batch_size 1

# Fine-tune with custom settings
python train_llada_pretrained.py \
    --data_dir shakespeare_dataset \
    --epochs 10 \
    --batch_size 2 \
    --learning_rate 2e-5 \
    --model_name GSAI-ML/LLaDA-8B-Base
```

### Custom Sequence Length

```bash
# Use shorter sequences for faster training
python download_shakespeare_dataset.py --sequence_length 1024
python train_llada_shakespeare.py --data_dir shakespeare_dataset --epochs 5
```

## Integration with Existing LLaDA Code

To use this with your existing LLaDA training code:

1. Replace the data loading section with the processed dataset
2. Use the vocabulary and token mappings from `vocab.json`
3. Ensure your model architecture matches the sequence length and vocabulary size
4. Implement the forward process as shown in the training script

## Notes

- The Shakespeare dataset is relatively small (~1MB), making it good for experimentation
- For production training, consider using larger datasets
- The character-level approach may not be optimal for all use cases
- Adjust hyperparameters based on your specific requirements and hardware constraints

## Model Scaling Considerations

### Memory Requirements
- **24 layers, 768 dimensions**: ~15M parameters, suitable for most GPUs
- **32 layers, 1024 dimensions**: ~45M parameters, requires 8GB+ GPU
- **48 layers, 1536 dimensions**: ~150M parameters, requires 16GB+ GPU

### Training Tips
- Start with smaller models for initial experiments
- Use gradient accumulation for larger models with limited memory
- Monitor GPU memory usage and adjust batch size accordingly
- Consider using mixed precision training for larger models

### Model Size Calculator

Use the included `model_size_calculator.py` script to estimate parameters and memory requirements:

```bash
# Calculate for default architecture
python model_size_calculator.py

# Calculate for custom architecture
python model_size_calculator.py --layers 48 --d_model 1536 --n_heads 24

# Get suggestions for your GPU memory
python model_size_calculator.py --hardware_memory 8
```

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or sequence length
2. **Slow Training**: Use GPU acceleration or reduce model size
3. **Poor Convergence**: Adjust learning rate or model architecture

### Dependencies

Ensure you have the required packages:
- PyTorch >= 1.9.0
- NumPy >= 1.20.0
- Requests >= 2.25.0
- TQDM >= 4.60.0

## References

- [LLaDA Paper](https://arxiv.org/abs/2502.09992)
- [GUIDELINES.md](../GUIDELINES.md) - LLaDA training guidelines
- [Shakespeare Dataset](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare)
