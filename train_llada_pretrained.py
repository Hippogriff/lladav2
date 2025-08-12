#!/usr/bin/env python3
"""
LLaDA Pre-trained Model Training Script

This script loads the pre-trained LLaDA-8B-Instruct model from Hugging Face
and fine-tunes it on the Shakespeare dataset using the LLaDA training process.
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import time
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed


class LLADAPretrainedTrainer:
    """LLaDA trainer for pre-trained models implementing the training process from GUIDELINES.md."""
    
    def __init__(self, model, tokenizer, mask_token: int = 126336, 
                 learning_rate: float = 1e-5, mixed_precision: str = 'fp16'):
        self.tokenizer = tokenizer
        self.mask_token = mask_token
        self.learning_rate = learning_rate
        
        # Get vocabulary size from tokenizer
        self.vocab_size = len(tokenizer)
        print(f"Vocabulary size: {self.vocab_size}")
        
        # Initialize accelerator for mixed precision and device management
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            log_with="tensorboard",
            project_dir="./logs"
        )
        
        # Move model to device and prepare for training
        self.model = self.accelerator.prepare(model)
        
        # Optimizer - use different learning rates for different parameter groups
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            max_lr=learning_rate,
            total_steps=1000,  # Will be updated during training
            pct_start=0.1,     # 10% warmup
            anneal_strategy='cos'
        )
        
        # Prepare optimizer and scheduler with accelerator
        self.optimizer, self.scheduler = self.accelerator.prepare(self.optimizer, self.scheduler)
        
        # Training stats
        self.train_losses = []
        self.val_losses = []
        
        # Log mixed precision status
        if mixed_precision == 'fp16':
            print(f"✓ 16-bit mixed precision training enabled via Accelerate")
        elif mixed_precision == 'bf16':
            print(f"✓ BFloat16 mixed precision training enabled via Accelerate")
        else:
            print(f"ℹ️  Using 32-bit precision training")
        
    def _create_optimizer(self):
        """Create optimizer with different learning rates for different parameter groups."""
        # Separate parameters into different groups
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01,
                'lr': self.learning_rate
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
                'lr': self.learning_rate
            }
        ]
        
        return torch.optim.AdamW(optimizer_grouped_parameters, lr=self.learning_rate)
        
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
        outputs = self.model(input_ids=noisy_batch)
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
            input_ids = batch[0]
            
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
            
            # Backward pass with accelerator
            self.accelerator.backward(loss)
            
            # Enhanced gradient clipping with NaN detection
            # Use accelerator's unwrapped model for gradient clipping
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            
            # Debug: Check gradient statistics before clipping
            total_params = 0
            total_grads = 0
            nan_grads = 0
            inf_grads = 0
            
            for param in unwrapped_model.parameters():
                if param.grad is not None:
                    total_params += 1
                    total_grads += param.grad.numel()
                    nan_grads += torch.isnan(param.grad).sum().item()
                    inf_grads += torch.isinf(param.grad).sum().item()
            
            if total_params > 0:
                print(f"Debug: {total_params} params, {total_grads} gradients, {nan_grads} NaN, {inf_grads} Inf")
            
            # Only clip if we have valid gradients
            if total_grads > 0 and nan_grads == 0 and inf_grads == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(unwrapped_model.parameters(), max_norm=1.0)
                
                # Check for NaN gradients after clipping
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: Invalid gradient norm detected after clipping: {grad_norm}")
                    # Skip this update
                    continue
            else:
                print(f"Warning: Skipping gradient clipping due to {nan_grads} NaN and {inf_grads} Inf gradients")
                # Skip this update
                continue
            
            # Optimizer step
            self.optimizer.step()
            
            # Check model weights for NaN after update
            has_nan_weights = False
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            for param in unwrapped_model.parameters():
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
                input_ids = batch[0]
                
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
        
        # Save model state using accelerator
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}"
        self.accelerator.save_state(checkpoint_path)
        
        # Save training stats separately
        stats_path = save_dir / f"stats_epoch_{epoch}.json"
        stats = {
            'epoch': epoch,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'vocab_size': self.vocab_size,
            'mask_token': self.mask_token
        }
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_path}")
    
    def train(self, train_loader, val_loader, num_epochs: int, save_dir: str = "checkpoints"):
        """Main training loop."""
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Update scheduler total steps
        total_steps = len(train_loader) * num_epochs
        self.scheduler.total_steps = total_steps
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Check if training failed
            if train_loss == float('inf'):
                print(f"Training failed at epoch {epoch}, stopping...")
                break
            
            # Validation
            val_loss = self.validate(val_loader)
            
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


def load_pretrained_model(model_name: str = 'GSAI-ML/LLaDA-8B-Instruct'):
    """Load the pre-trained LLaDA model and tokenizer."""
    print(f"Loading pre-trained model: {model_name}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ Tokenizer loaded successfully")
        
        # Load model
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ Model loaded successfully")
        
        # Set model to evaluation mode initially
        model.eval()
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise


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
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained LLaDA on Shakespeare dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing processed dataset")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                       help="Pre-trained model name from Hugging Face")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size (default: 1 for large models)")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs (default: 5 for fine-tuning)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate (default: 1e-5 for fine-tuning)")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--save_dir", type=str, default="checkpoints_pretrained",
                       help="Directory to save checkpoints")
    parser.add_argument("--fp16", action="store_true",
                       help="Enable 16-bit mixed precision training")
    parser.add_argument("--no_fp16", action="store_true",
                       help="Disable 16-bit mixed precision training")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load pre-trained model
    model, tokenizer = load_pretrained_model(args.model_name)
    
    # Load dataset
    train_data, val_data, metadata = load_dataset(args.data_dir)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_data, val_data, args.batch_size)
    
    # Create trainer
    mixed_precision = 'fp16' if device == "cuda" and not args.no_fp16 else 'no'
    trainer = LLADAPretrainedTrainer(
        model=model,
        tokenizer=tokenizer,
        mask_token=metadata['mask_token'],
        learning_rate=args.learning_rate,
        mixed_precision=mixed_precision
    )
    
    # Prepare data loaders with accelerator
    train_loader, val_loader = trainer.accelerator.prepare(train_loader, val_loader)
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
