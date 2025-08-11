# Shakespeare Dataset for LLaDA Training

This directory contains scripts to download, preprocess, and train LLaDA (Large Language Diffusion Models) on the Shakespeare dataset.

## Overview

The scripts implement the LLaDA training process as described in the [GUIDELINES.md](../GUIDELINES.md) file. LLaDA is a diffusion-based language model that uses masking instead of autoregressive generation.

## Files

- `download_shakespeare_dataset.py` - Downloads and preprocesses the Shakespeare dataset
- `train_llada_shakespeare.py` - Training script for LLaDA
- `requirements.txt` - Python dependencies
- `SHAKESPEARE_README.md` - This file

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download and Preprocess Dataset

```bash
python download_shakespeare_dataset.py --output_dir shakespeare_dataset
```

This will:
- Download the Shakespeare dataset from Andrej Karpathy's repository
- Build a character-level vocabulary
- Create training sequences of length 4096 (configurable)
- Split data into training (90%) and validation (10%) sets
- Save processed data in numpy format

### 3. Train LLaDA

```bash
python train_llada_shakespeare.py --data_dir shakespeare_dataset --epochs 10 --batch_size 4
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

## Model Architecture

The training script includes a simple Transformer Encoder implementation:

- **Embedding Layer**: Token and positional embeddings
- **Transformer Encoder**: Multi-head self-attention layers
- **Output Projection**: Linear layer to vocabulary size

For production use, you should replace this with a more sophisticated architecture.

## Configuration Options

### Dataset Download

- `--output_dir`: Output directory for processed dataset
- `--sequence_length`: Sequence length for training (default: 4096)

### Training

- `--data_dir`: Directory containing processed dataset
- `--batch_size`: Training batch size (default: 4)
- `--epochs`: Number of training epochs (default: 10)
- `--d_model`: Model dimension (default: 512)
- `--n_heads`: Number of attention heads (default: 8)
- `--n_layers`: Number of transformer layers (default: 6)
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

### Basic Training

```bash
# Download dataset
python download_shakespeare_dataset.py

# Train for 20 epochs with larger model
python train_llada_shakespeare.py \
    --data_dir shakespeare_dataset \
    --epochs 20 \
    --batch_size 8 \
    --d_model 768 \
    --n_heads 12 \
    --n_layers 8
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
