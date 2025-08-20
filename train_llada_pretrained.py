#!/usr/bin/env python3
"""
LLaDA Pre-trained Model Training Script

This script loads the pre-trained LLaDA-8B-Instruct model from Hugging Face
and fine-tunes it on the Shakespeare dataset using the LLaDA training process.
Supports multi-GPU training using torchrun and Distributed Data Parallel (DDP).
"""

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import time
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def setup_ddp(rank, world_size):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Clean up distributed training."""
    dist.destroy_process_group()


class LLADAPretrainedTrainer:
    """LLaDA trainer for pre-trained models implementing the training process from GUIDELINES.md."""
    
    def __init__(self, model, tokenizer, rank, world_size, mask_token: int = 126336, 
                 learning_rate: float = 1e-5, mixed_precision: bool = True):
        self.tokenizer = tokenizer
        self.mask_token = mask_token
        self.learning_rate = learning_rate
        self.rank = rank
        self.world_size = world_size
        self.is_main_process = rank == 0
        
        # Get vocabulary size from tokenizer
        self.vocab_size = len(tokenizer)
        if self.is_main_process:
            print(f"Vocabulary size: {self.vocab_size}")
        
        # Wrap model with DDP
        self.model = DDP(model, device_ids=[rank], output_device=rank)
        
        # Enable mixed precision if available
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        
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
        
        # Training stats
        self.train_losses = []
        self.val_losses = []
        
        # Log mixed precision status
        if self.is_main_process:
            if mixed_precision:
                print(f"✓ 16-bit mixed precision training enabled")
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
        # Forward pass through the model
        outputs = self.model(inputs_embeds=noisy_batch)
        logits = outputs.logits if hasattr(outputs, 'logits') else outputs.last_hidden_state
        
        # Ensure logits have the right shape for loss computation
        if len(logits.shape) == 3:
            # If logits are (batch_size, seq_len, vocab_size), we need to reshape
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(-1, vocab_size)
            target_ids = input_ids.view(-1)
        else:
            # If logits are already flattened
            target_ids = input_ids.view(-1)
        
        # Clamp logits for numerical stability
        logits = torch.clamp(logits, min=-100, max=100)
        
        # Compute cross-entropy loss
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        token_loss = loss_fct(logits, target_ids)
        
        # Reshape token loss back to (batch_size, seq_len)
        token_loss = token_loss.view(batch_size, seq_len)
        
        # Apply importance weighting: 1 / p_mask for masked tokens
        p_mask_stable = torch.clamp(p_mask, min=1e-8, max=1.0)
        importance_weights = 1.0 / p_mask_stable
        
        # Apply weights only to masked tokens
        weighted_loss = token_loss * importance_weights * masked_indices.float()
        
        # Average over masked tokens
        num_masked = masked_indices.sum()
        if num_masked > 0:
            loss = weighted_loss.sum() / num_masked
        else:
            loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        
        # Handle NaN in loss
        if torch.isnan(loss):
            if self.is_main_process:
                print("Warning: NaN loss detected, using fallback")
            loss = torch.tensor(10.0, device=input_ids.device, requires_grad=True)
        
        return loss
    
    def train_epoch(self, train_loader, epoch: int):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        # Only show progress bar on main process
        if self.is_main_process:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        else:
            progress_bar = train_loader
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch[0]
            
            # Check input for NaN
            if torch.isnan(input_ids).any():
                if self.is_main_process:
                    print(f"Warning: NaN detected in input_ids at batch {batch_idx}")
                continue
            
            # Apply LLaDA forward process
            noisy_batch, masked_indices, p_mask = self.forward_process(input_ids)
            
            # Check forward process outputs
            if torch.isnan(noisy_batch).any() or torch.isnan(p_mask).any():
                if self.is_main_process:
                    print(f"Warning: NaN detected in forward process at batch {batch_idx}")
                continue
            
            # Compute loss
            loss = self.compute_loss(input_ids, noisy_batch, masked_indices, p_mask)
            
            # Check loss for NaN
            if torch.isnan(loss):
                if self.is_main_process:
                    print(f"Warning: NaN loss detected at batch {batch_idx}")
                continue
            
            # Backward pass with mixed precision
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Enhanced gradient clipping with NaN detection
            total_params = 0
            total_grads = 0
            nan_grads = 0
            inf_grads = 0
            
            for param in self.model.parameters():
                if param.grad is not None:
                    total_params += 1
                    total_grads += param.grad.numel()
                    nan_grads += torch.isnan(param.grad).sum().item()
                    inf_grads += torch.isinf(param.grad).sum().item()
            
            if total_params > 0 and self.is_main_process:
                print(f"Debug: {total_params} params, {total_grads} gradients, {nan_grads} NaN, {inf_grads} Inf")
            
            # Only clip if we have valid gradients
            if total_grads > 0 and nan_grads == 0 and inf_grads == 0:
                try:
                    if self.scaler is not None:
                        # Unscale gradients for clipping
                        self.scaler.unscale_(self.optimizer)
                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    
                    if self.is_main_process:
                        print(f"Debug: Gradient norm after clipping: {grad_norm:.6f}")
                    
                    # Check for NaN gradients after clipping
                    if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                        if self.is_main_process:
                            print(f"Warning: Invalid gradient norm detected after clipping: {grad_norm}")
                        # Skip this update
                        continue
                except Exception as e:
                    if self.is_main_process:
                        print(f"Warning: Error during gradient clipping: {e}")
                    # Skip this update
                    continue
            else:
                if self.is_main_process:
                    print(f"Warning: Skipping gradient clipping due to {nan_grads} NaN and {inf_grads} Inf gradients")
                # Skip this update
                continue
            
            # Optimizer step with mixed precision
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Check model weights for NaN after update
            has_nan_weights = False
            for param in self.model.parameters():
                if torch.isnan(param).any():
                    has_nan_weights = True
                    break
            
            if has_nan_weights:
                if self.is_main_process:
                    print(f"Warning: NaN weights detected after update, stopping training")
                return float('inf')  # Return high loss to indicate failure
            
            # Step scheduler for OneCycleLR (step after each batch)
            self.scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar on main process
            if self.is_main_process:
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
        if not self.is_main_process:
            return
            
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Save model state
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'vocab_size': self.vocab_size,
            'mask_token': self.mask_token
        }, checkpoint_path)
        
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
        if self.is_main_process:
            print(f"Starting training for {num_epochs} epochs...")
            print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"Training on {self.world_size} GPUs")
        
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
                if self.is_main_process:
                    print(f"Training failed at epoch {epoch}, stopping...")
                break
            
            # Validation
            val_loss = self.validate(val_loader)
            
            epoch_time = time.time() - start_time
            
            if self.is_main_process:
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
        
        if self.is_main_process:
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


def create_data_loaders(train_data, val_data, batch_size: int, rank: int, world_size: int):
    """Create PyTorch data loaders with distributed sampling."""
    # Convert to tensors
    train_tensor = torch.tensor(train_data, dtype=torch.long)
    val_tensor = torch.tensor(val_data, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    
    # Create distributed samplers
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=4,  # Higher value for multi-GPU training
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, train_sampler, val_sampler


def main_worker(rank, world_size, args):
    """Main worker function for distributed training."""
    # Setup distributed training
    setup_ddp(rank, world_size)
    
    # Set device
    device = torch.device(f'cuda:{rank}')
    torch.cuda.set_device(device)
    
    if rank == 0:
        print(f"Using device: {device}")
        print(f"World size: {world_size}")
    
    # Load pre-trained model
    model, tokenizer = load_pretrained_model(args.model_name)
    model = model.to(device)
    
    # Load dataset
    train_data, val_data, metadata = load_dataset(args.data_dir)
    
    # Create data loaders
    train_loader, val_loader, train_sampler, val_sampler = create_data_loaders(
        train_data, val_data, args.batch_size, rank, world_size
    )
    
    # Create trainer
    trainer = LLADAPretrainedTrainer(
        model=model,
        tokenizer=tokenizer,
        rank=rank,
        world_size=world_size,
        mask_token=metadata['mask_token'],
        learning_rate=args.learning_rate,
        mixed_precision=args.fp16
    )
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_dir=args.save_dir
    )
    
    # Cleanup
    cleanup_ddp()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained LLaDA on Shakespeare dataset")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing processed dataset")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                       help="Pre-trained model name from Hugging Face")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size per GPU (default: 1 for large models)")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs (default: 5 for fine-tuning)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate (default: 1e-5 for fine-tuning)")
    parser.add_argument("--save_dir", type=str, default="checkpoints_pretrained",
                       help="Directory to save checkpoints")
    parser.add_argument("--fp16", action="store_true", default=True,
                       help="Enable 16-bit mixed precision training (default: True)")
    parser.add_argument("--no_fp16", action="store_true",
                       help="Disable 16-bit mixed precision training")
    
    args = parser.parse_args()
    
    # Handle fp16 flag
    if args.no_fp16:
        args.fp16 = False
    
    # Get world size from environment (set by torchrun)
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    if world_size > 1:
        # Multi-GPU training
        main_worker(rank, world_size, args)
    else:
        # Single GPU training
        print("Single GPU training mode")
        main_worker(0, 1, args)


if __name__ == "__main__":
    main()
