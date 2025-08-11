#!/usr/bin/env python3
"""
Shakespeare Dataset Downloader and Preprocessor for LLaDA Training

This script downloads the Shakespeare dataset and prepares it for training LLaDA
(Large Language Diffusion Models) according to the specifications in GUIDELINES.md.

The script:
1. Downloads the Shakespeare dataset from Andrej Karpathy's repository
2. Preprocesses the text data into the required format for LLaDA training
3. Creates training and validation splits
4. Saves the processed data in the format expected by LLaDA

Usage:
    python download_shakespeare_dataset.py [--output_dir OUTPUT_DIR] [--sequence_length SEQUENCE_LENGTH]

Requirements:
    - requests
    - tqdm
    - numpy
    - torch
"""

import os
import argparse
import requests
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import tqdm
import numpy as np
import torch
from collections import Counter


class ShakespeareDatasetDownloader:
    """Downloads and preprocesses Shakespeare dataset for LLaDA training."""
    
    def __init__(self, output_dir: str = "shakespeare_dataset", sequence_length: int = 4096):
        self.output_dir = Path(output_dir)
        self.sequence_length = sequence_length
        self.shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # LLaDA specific tokens (from GUIDELINES.md)
        self.mask_token = 126336  # [MASK] token
        self.bos_token = 1        # Beginning of sequence
        self.eos_token = 2        # End of sequence
        self.pad_token = 0        # Padding token
        
        # Vocabulary mapping for Shakespeare text
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
        
    def download_dataset(self) -> str:
        """Download the Shakespeare dataset."""
        print(f"Downloading Shakespeare dataset from {self.shakespeare_url}...")
        
        response = requests.get(self.shakespeare_url)
        response.raise_for_status()
        
        # Save raw text
        raw_file = self.output_dir / "shakespeare_raw.txt"
        with open(raw_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Downloaded {len(response.text)} characters to {raw_file}")
        return response.text
    
    def build_vocabulary(self, text: str) -> None:
        """Build character-level vocabulary from the text."""
        print("Building vocabulary...")
        
        # Get unique characters and sort them for consistent mapping
        unique_chars = sorted(set(text))
        
        # Create character to ID mapping
        self.char_to_id = {char: idx + 3 for idx, char in enumerate(unique_chars)}  # Start from 3 to reserve special tokens
        self.id_to_char = {idx: char for char, idx in self.char_to_id.items()}
        
        # Add special tokens
        self.char_to_id['<PAD>'] = self.pad_token
        self.char_to_id['<BOS>'] = self.bos_token
        self.char_to_id['<EOS>'] = self.eos_token
        self.char_to_id['<MASK>'] = self.mask_token
        
        self.id_to_char[self.pad_token] = '<PAD>'
        self.id_to_char[self.bos_token] = '<BOS>'
        self.id_to_char[self.eos_token] = '<EOS>'
        self.id_to_char[self.mask_token] = '<MASK>'
        
        self.vocab_size = len(self.char_to_id)
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Special tokens: {list(self.char_to_id.keys())[:4]}")
        
        # Save vocabulary
        vocab_file = self.output_dir / "vocab.json"
        with open(vocab_file, 'w') as f:
            json.dump({
                'char_to_id': self.char_to_id,
                'id_to_char': {str(k): v for k, v in self.id_to_char.items()},
                'vocab_size': self.vocab_size,
                'mask_token': self.mask_token,
                'bos_token': self.bos_token,
                'eos_token': self.eos_token,
                'pad_token': self.pad_token
            }, f, indent=2)
        
        print(f"Vocabulary saved to {vocab_file}")
    
    def text_to_ids(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        return [self.char_to_id.get(char, self.pad_token) for char in text]
    
    def ids_to_text(self, ids: List[int]) -> str:
        """Convert token IDs back to text."""
        return ''.join([self.id_to_char.get(idx, '<UNK>') for idx in ids])
    
    def create_sequences(self, text_ids: List[int]) -> List[List[int]]:
        """Create fixed-length sequences for training."""
        print("Creating training sequences...")
        
        sequences = []
        total_tokens = len(text_ids)
        
        # Create sequences with overlap for better training
        stride = self.sequence_length // 2  # 50% overlap between sequences
        
        for i in range(0, total_tokens - self.sequence_length + 1, stride):
            sequence = text_ids[i:i + self.sequence_length]
            sequences.append(sequence)
        
        # Handle the last sequence if it's shorter than sequence_length
        if total_tokens % stride != 0:
            last_sequence = text_ids[-self.sequence_length:]
            if len(last_sequence) == self.sequence_length:
                sequences.append(last_sequence)
        
        print(f"Created {len(sequences)} sequences of length {self.sequence_length}")
        return sequences
    
    def create_llada_training_data(self, sequences: List[List[int]]) -> Tuple[List[List[int]], List[int]]:
        """Create LLaDA training data format as specified in GUIDELINES.md."""
        print("Creating LLaDA training data format...")
        
        # For pre-training, we need:
        # 1. input_ids: (batch_size, sequence_length) with random masking
        # 2. masked_indices: boolean mask indicating which tokens are masked
        
        # Add BOS and EOS tokens to each sequence
        processed_sequences = []
        for seq in sequences:
            # Add BOS at the beginning and EOS at the end
            processed_seq = [self.bos_token] + seq + [self.eos_token]
            
            # Pad to sequence_length if necessary
            if len(processed_seq) < self.sequence_length:
                processed_seq.extend([self.pad_token] * (self.sequence_length - len(processed_seq)))
            elif len(processed_seq) > self.sequence_length:
                processed_seq = processed_seq[:self.sequence_length]
            
            processed_sequences.append(processed_seq)
        
        # Convert to numpy arrays for easier manipulation
        input_ids = np.array(processed_sequences, dtype=np.int64)
        
        # Create a simple validation split (10% of data)
        n_val = max(1, len(input_ids) // 10)
        val_indices = np.random.choice(len(input_ids), n_val, replace=False)
        train_indices = np.setdiff1d(np.arange(len(input_ids)), val_indices)
        
        train_data = input_ids[train_indices]
        val_data = input_ids[val_indices]
        
        print(f"Training sequences: {len(train_data)}")
        print(f"Validation sequences: {len(val_data)}")
        
        return train_data, val_data
    
    def save_training_data(self, train_data: np.ndarray, val_data: np.ndarray) -> None:
        """Save training data in the format expected by LLaDA."""
        print("Saving training data...")
        
        # Save as numpy arrays
        train_file = self.output_dir / "train_data.npy"
        val_file = self.output_dir / "val_data.npy"
        
        np.save(train_file, train_data)
        np.save(val_file, val_data)
        
        print(f"Training data saved to {train_file}")
        print(f"Validation data saved to {val_file}")
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'vocab_size': self.vocab_size,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'total_tokens': len(train_data) * self.sequence_length,
            'mask_token': self.mask_token,
            'bos_token': self.bos_token,
            'eos_token': self.eos_token,
            'pad_token': self.pad_token
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to {metadata_file}")
    
    def create_sample_batch(self, data: np.ndarray, batch_size: int = 4) -> Dict[str, torch.Tensor]:
        """Create a sample batch to demonstrate the data format."""
        print(f"Creating sample batch of size {batch_size}...")
        
        # Sample random indices
        indices = np.random.choice(len(data), min(batch_size, len(data)), replace=False)
        batch_data = data[indices]
        
        # Convert to torch tensors
        input_ids = torch.tensor(batch_data, dtype=torch.long)
        
        # Create sample batch dictionary
        batch = {
            'input_ids': input_ids,
            'batch_size': input_ids.shape[0],
            'sequence_length': input_ids.shape[1]
        }
        
        print(f"Sample batch shape: {input_ids.shape}")
        print(f"Sample sequence (first 50 tokens): {input_ids[0, :50].tolist()}")
        
        return batch
    
    def demonstrate_llada_forward_process(self, batch: Dict[str, torch.Tensor]) -> None:
        """Demonstrate the LLaDA forward process as described in GUIDELINES.md."""
        print("\nDemonstrating LLaDA forward process...")
        
        input_ids = batch['input_ids']
        b, l = input_ids.shape
        
        # LLaDA forward process (from GUIDELINES.md)
        eps = 1e-3
        t = torch.rand(b, device=input_ids.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)
        
        masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
        noisy_batch = torch.where(masked_indices, self.mask_token, input_ids)
        
        print(f"Original input shape: {input_ids.shape}")
        print(f"Masking probability range: {p_mask.min().item():.4f} - {p_mask.max().item():.4f}")
        print(f"Masked tokens: {masked_indices.sum().item()}")
        print(f"Noisy batch shape: {noisy_batch.shape}")
        
        # Show example of masking
        if b > 0 and l > 20:
            print(f"\nExample masking (first 20 tokens of first sequence):")
            print(f"Original: {input_ids[0, :20].tolist()}")
            print(f"Masked:   {noisy_batch[0, :20].tolist()}")
            print(f"Mask:     {masked_indices[0, :20].tolist()}")
    
    def process(self) -> None:
        """Main processing pipeline."""
        print("Starting Shakespeare dataset processing for LLaDA...")
        
        # Download dataset
        text = self.download_dataset()
        
        # Build vocabulary
        self.build_vocabulary(text)
        
        # Convert text to token IDs
        text_ids = self.text_to_ids(text)
        print(f"Converted text to {len(text_ids)} token IDs")
        
        # Create sequences
        sequences = self.create_sequences(text_ids)
        
        # Create LLaDA training data
        train_data, val_data = self.create_llada_training_data(sequences)
        
        # Save training data
        self.save_training_data(train_data, val_data)
        
        # Create and demonstrate sample batch
        sample_batch = self.create_sample_batch(train_data)
        self.demonstrate_llada_forward_process(sample_batch)
        
        print(f"\nDataset processing complete!")
        print(f"Output directory: {self.output_dir}")
        print(f"Files created:")
        for file in self.output_dir.glob("*"):
            print(f"  - {file.name}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Download and preprocess Shakespeare dataset for LLaDA training")
    parser.add_argument("--output_dir", type=str, default="shakespeare_dataset", 
                       help="Output directory for processed dataset")
    parser.add_argument("--sequence_length", type=int, default=4096,
                       help="Sequence length for training (default: 4096)")
    
    args = parser.parse_args()
    
    # Create downloader and process dataset
    downloader = ShakespeareDatasetDownloader(
        output_dir=args.output_dir,
        sequence_length=args.sequence_length
    )
    
    try:
        downloader.process()
    except Exception as e:
        print(f"Error processing dataset: {e}")
        raise


if __name__ == "__main__":
    main()
