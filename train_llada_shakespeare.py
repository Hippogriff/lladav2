#!/usr/bin/env python3
"""
LLaDA Training Script for Shakespeare Dataset

This script demonstrates how to train LLaDA using the processed Shakespeare dataset.
It implements the training loop as described in GUIDELINES.md.

Usage:
    python train_llada_shakespeare.py --data_dir shakespeare_dataset [--batch_size BATCH_SIZE] [--epochs EPOCHS]

Requirements:
    - torch
    - numpy
    - transformers (for the model architecture)
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import time
from tqdm import tqdm


class SimpleTransformerEncoder(nn.Module):
    """Simple Transformer Encoder for demonstration purposes.
    
    This is a minimal implementation to show the LLaDA training process.
    In practice, you would use a more sophisticated architecture.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 768, n_heads: int = 12, 
                 n_layers: int = 24, max_seq_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Layer normalization for embeddings
        self.embed_norm = nn.LayerNorm(d_model)
        
        # Dropout after embeddings
        self.embed_dropout = nn.Dropout(dropout)
        
        # Create transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, dropout) 
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights for main model
        self.apply(self._init_weights)
        
        # Initialize weights for transformer layers
        for layer in self.layers:
            layer.apply(layer._init_weights)
        
        # Print model info
        print(f"Created Transformer with {n_layers} layers, {d_model} dimensions, {n_heads} heads")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self, module):
        """Initialize weights for the main transformer model."""
        if isinstance(module, nn.Linear):
            # Use Xavier/Glorot initialization for better stability
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Use smaller std for embeddings to prevent instability
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, input_ids):
        """Forward pass through the transformer encoder."""
        # input_ids: (batch_size, seq_len)
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply embedding normalization and dropout
        x = self.embed_norm(x)
        x = self.embed_dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Apply final layer normalization
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)  # (batch_size, seq_len, vocab_size)
        
        return type('Output', (), {'logits': logits})()


class TransformerLayer(nn.Module):
    """Custom transformer layer using scaled_dot_product_attention without causal masking."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Layer normalization
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _init_weights(self, module):
        """Initialize weights for the transformer layer."""
        if isinstance(module, nn.Linear):
            # Use Xavier/Glorot initialization for better stability
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Multi-head attention
        residual = x
        x = self.attn_norm(x)
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Use scaled_dot_product_attention without causal masking
        attn_output = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.o_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        # Add residual connection
        x = residual + attn_output
        
        # Feed-forward network
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        
        # Add residual connection
        x = residual + x
        
        return x
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
class LLADATrainer:
    """LLaDA trainer implementing the training process from GUIDELINES.md."""
    
    def __init__(self, model, vocab_size: int, mask_token: int = 126336, 
                 device: str = 'cpu', learning_rate: float = 1e-4):
        self.model = model.to(device)
        self.device = device
        self.mask_token = mask_token
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=learning_rate,
            total_steps=1000,  # Will be updated during training
            pct_start=0.1,     # 10% warmup
            anneal_strategy='cos'
        )
        
        # Training stats
        self.train_losses = []
        self.val_losses = []
        
    def forward_process(self, input_ids, eps=1e-3):
        """LLaDA forward process as described in GUIDELINES.md."""
        b, l = input_ids.shape
        
        # Sample random time t for each sequence in batch
        t = torch.rand(b, device=input_ids.device)
        
        # Calculate masking probability
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, l)
        
        # Create mask
        masked_indices = torch.rand((b, l), device=input_ids.device) < p_mask
        
        # Apply masking
        noisy_batch = torch.where(masked_indices, self.mask_token, input_ids)
        
        return noisy_batch, masked_indices, p_mask
    
    def compute_loss(self, input_ids, noisy_batch, masked_indices, p_mask):
        """Compute LLaDA loss as described in GUIDELINES.md."""
        # Get model predictions
        outputs = self.model(noisy_batch)
        logits = outputs.logits
        
        # Add numerical stability to logits
        logits = torch.clamp(logits, min=-100, max=100)
        
        # Compute token-level loss
        token_loss = F.cross_entropy(
            logits[masked_indices], 
            input_ids[masked_indices], 
            reduction='none'
        )
        
        # Apply importance weighting with numerical stability
        # Add small epsilon to prevent division by zero
        eps = 1e-8
        p_mask_stable = torch.clamp(p_mask, min=eps, max=1.0)
        token_loss = token_loss / p_mask_stable[masked_indices]
        
        # Check for NaN in token loss
        if torch.isnan(token_loss).any():
            print(f"Warning: NaN detected in token loss")
            token_loss = torch.nan_to_num(token_loss, nan=0.0, posinf=100.0, neginf=-100.0)
        
        # Compute final loss with numerical stability
        loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
        
        # Final NaN check
        if torch.isnan(loss):
            print(f"Warning: NaN detected in final loss, using fallback loss")
            loss = torch.tensor(10.0, device=loss.device, requires_grad=True)
        
        return loss
    
    def train_epoch(self, train_loader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch[0].to(self.device)
            
            # Check input for NaN
            if torch.isnan(input_ids).any():
                print(f"Warning: NaN detected in input_ids at batch {batch_idx}")
                continue
            
            # Apply LLaDA forward process
            noisy_batch, masked_indices, p_mask = self.forward_process(input_ids)
            
            # Check forward process outputs
            if torch.isnan(noisy_batch).any() or torch.isnan(p_mask).any():
                print(f"Warning: NaN detected in forward process at batch {batch_idx}")
                continue
            
            # Compute loss
            loss = self.compute_loss(input_ids, noisy_batch, masked_indices, p_mask)
            
            # Check loss for NaN
            if torch.isnan(loss):
                print(f"Warning: NaN loss detected at batch {batch_idx}")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Enhanced gradient clipping with NaN detection
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Check for NaN gradients
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"Warning: Invalid gradient norm detected: {grad_norm}")
                # Skip this update
                continue
            
            # Check if any parameters have NaN gradients
            has_nan_grad = False
            for param in self.model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    break
            
            if has_nan_grad:
                print(f"Warning: NaN gradients detected, skipping update")
                continue
            
            self.optimizer.step()
            
            # Check model weights for NaN after update
            has_nan_weights = False
            for param in self.model.parameters():
                if torch.isnan(param).any():
                    has_nan_weights = True
                    break
            
            if has_nan_weights:
                print(f"Warning: NaN weights detected after update, stopping training")
                return float('inf')  # Return high loss to indicate failure
            
            # Step scheduler for OneCycleLR (step after each batch)
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0].to(self.device)
                
                # Apply LLaDA forward process
                noisy_batch, masked_indices, p_mask = self.forward_process(input_ids)
                
                # Compute loss
                loss = self.compute_loss(input_ids, noisy_batch, masked_indices, p_mask)
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, epoch: int, save_dir: str):
        """Save model checkpoint."""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'vocab_size': self.vocab_size,
            'mask_token': self.mask_token
        }
        
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def train(self, train_loader, val_loader, num_epochs: int, save_dir: str = "checkpoints"):
        """Main training loop."""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Update scheduler total steps
        total_steps = len(train_loader) * num_epochs
        self.scheduler.total_steps = total_steps
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            # Learning rate scheduling (step after each epoch for OneCycleLR)
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            print(f"\nEpoch {epoch}/{num_epochs} completed in {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, save_dir)
            
            # Save checkpoint every 5 epochs
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, save_dir)
        
        print(f"\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")


def load_dataset(data_dir: str):
    """Load the processed Shakespeare dataset."""
    data_dir = Path(data_dir)
    
    # Load metadata
    with open(data_dir / "metadata.json", 'r') as f:
        metadata = json.load(f)
    
    # Load training and validation data
    train_data = np.load(data_dir / "train_data.npy")
    val_data = np.load(data_dir / "val_data.npy")
    
    print(f"Dataset loaded:")
    print(f"  - Training samples: {len(train_data)}")
    print(f"  - Validation samples: {len(val_data)}")
    print(f"  - Sequence length: {metadata['sequence_length']}")
    print(f"  - Vocabulary size: {metadata['vocab_size']}")
    
    return train_data, val_data, metadata


def create_data_loaders(train_data, val_data, batch_size: int):
    """Create PyTorch data loaders."""
    # Convert to tensors
    train_tensor = torch.tensor(train_data, dtype=torch.long)
    val_tensor = torch.tensor(val_data, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Set to higher value for multi-GPU training
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Train LLaDA on Shakespeare dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing processed dataset")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--d_model", type=int, default=768,
                       help="Model dimension (default: 768)")
    parser.add_argument("--n_heads", type=int, default=12,
                       help="Number of attention heads (default: 12)")
    parser.add_argument("--n_layers", type=int, default=24,
                       help="Number of transformer layers (default: 24)")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate (default: 0.1)")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate (default: 5e-5)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load dataset
    train_data, val_data, metadata = load_dataset(args.data_dir)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_data, val_data, args.batch_size)
    
    # Create model
    model = SimpleTransformerEncoder(
        vocab_size=metadata['vocab_size'],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=metadata['sequence_length'],
        dropout=args.dropout
    )
    
    # Create trainer
    trainer = LLADATrainer(
        model=model,
        vocab_size=metadata['vocab_size'],
        mask_token=metadata['mask_token'],
        device=device,
        learning_rate=args.learning_rate
    )
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_dir="checkpoints"
    )


if __name__ == "__main__":
    main()
