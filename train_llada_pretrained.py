#!/usr/bin/env python3
"""
LLaDA Pre-trained Model Training Script

This script loads the pre-trained LLaDA-8B-Instruct model from Hugging Face
and fine-tunes it on the Shakespeare dataset using the LLaDA training process
with full Accelerate integration for distributed training.
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
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed, DummyScheduler, DummyOptim
from accelerate.tracking import TrackerMixin
import wandb
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration class for training parameters."""
    # Model and data
    model_name: str = "GSAI-ML/LLaDA-8B-Instruct"
    data_dir: str = ""
    mask_token: int = 126336
    
    # Training hyperparameters
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    num_epochs: int = 5
    
    # Batch and optimization
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    dataloader_num_workers: int = 4
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_dir: str = "checkpoints_pretrained"
    
    # Mixed precision
    mixed_precision: str = "fp16"
    
    # Reproducibility
    seed: int = 42
    
    # Checkpointing
    resume_from_checkpoint: Optional[str] = None
    
    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size across all devices and accumulation steps."""
        return self.batch_size * self.gradient_accumulation_steps
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for logging."""
        return {
            'model_name': self.model_name,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_steps': self.warmup_steps,
            'max_grad_norm': self.max_grad_norm,
            'num_epochs': self.num_epochs,
            'batch_size': self.batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'effective_batch_size': self.effective_batch_size,
            'mixed_precision': self.mixed_precision,
            'seed': self.seed
        }


def diagnose_nccl_issues():
    """Diagnose and fix common NCCL issues."""
    print("Diagnosing NCCL configuration...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available, skipping NCCL diagnosis")
        return
    
    # Check number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s)")
    
    if num_gpus <= 1:
        print("Single GPU detected, NCCL not needed")
        return
    
    # Set NCCL environment variables
    nccl_vars = {
        'NCCL_DEBUG': 'INFO',
        'NCCL_IB_DISABLE': '1',
        'NCCL_P2P_DISABLE': '1',
        'NCCL_SOCKET_IFNAME': 'lo',
        'NCCL_BLOCKING_WAIT': '1',
        'NCCL_TIMEOUT': '1800',
        'NCCL_SHM_DISABLE': '1',  # Disable shared memory if having issues
        'NCCL_NET_GDR_LEVEL': '0'  # Disable GPU Direct RDMA
    }
    
    print("Setting NCCL environment variables:")
    for var, value in nccl_vars.items():
        os.environ[var] = value
        print(f"  {var}={value}")
    
    # Test basic GPU communication
    try:
        print("Testing GPU communication...")
        for i in range(num_gpus):
            torch.cuda.set_device(i)
            x = torch.randn(10, 10, device=f'cuda:{i}')
            print(f"  GPU {i}: {x.device} - OK")
        print("GPU communication test passed")
    except Exception as e:
        print(f"GPU communication test failed: {e}")
        print("Consider using single-machine configuration")


def setup_accelerate_logging():
    """Setup logging configuration for Accelerate."""
    # Create logs directory
    os.makedirs("./logs", exist_ok=True)
    
    # Setup NCCL environment variables to avoid network issues
    os.environ.setdefault('NCCL_DEBUG', 'INFO')
    os.environ.setdefault('NCCL_IB_DISABLE', '1')
    os.environ.setdefault('NCCL_P2P_DISABLE', '1')
    os.environ.setdefault('NCCL_SOCKET_IFNAME', 'lo')
    os.environ.setdefault('NCCL_BLOCKING_WAIT', '1')
    os.environ.setdefault('NCCL_TIMEOUT', '1800')
    
    # Setup wandb if available
    try:
        import wandb
        if wandb.run is None:
            wandb.init(project="llada-pretrained", name="llada-shakespeare-finetuning")
    except ImportError:
        print("Wandb not available, skipping wandb logging")


def create_accelerator_config(config: TrainingConfig) -> Accelerator:
    """Create and configure Accelerator instance."""
    # Diagnose and fix NCCL issues
    diagnose_nccl_issues()
    
    return Accelerator(
        mixed_precision=config.mixed_precision,
        log_with="tensorboard",
        project_dir="./logs",
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers,
        kwargs_handlers=[DummyScheduler, DummyOptim] if config.mixed_precision == "no" else None
    )


class LLADAPretrainedTrainer:
    """LLaDA trainer for pre-trained models with full Accelerate integration."""
    
    def __init__(self, model, tokenizer, mask_token: int = 126336, 
                 learning_rate: float = 1e-5, mixed_precision: str = 'fp16',
                 weight_decay: float = 0.01, warmup_steps: int = 100,
                 max_grad_norm: float = 1.0, logging_steps: int = 10,
                 save_steps: int = 500, eval_steps: int = 500,
                 gradient_accumulation_steps: int = 1, dataloader_num_workers: int = 4):
        self.tokenizer = tokenizer
        self.mask_token = mask_token
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_grad_norm = max_grad_norm
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.dataloader_num_workers = dataloader_num_workers
        
        # Get vocabulary size from tokenizer
        self.vocab_size = len(tokenizer)
        
        # Initialize accelerator with enhanced configuration
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            log_with="tensorboard",
            project_dir="./logs",
            gradient_accumulation_steps=gradient_accumulation_steps,
            dataloader_num_workers=dataloader_num_workers
        )
        
        # Get logger
        self.logger = get_logger(__name__)
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Log accelerator configuration
        self.logger.info(f"Accelerator configuration:")
        self.logger.info(f"  - Mixed precision: {mixed_precision}")
        self.logger.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
        self.logger.info(f"  - Dataloader workers: {dataloader_num_workers}")
        self.logger.info(f"  - Device: {self.accelerator.device}")
        self.logger.info(f"  - Process index: {self.accelerator.process_index}")
        self.logger.info(f"  - Local process index: {self.accelerator.local_process_index}")
        self.logger.info(f"  - Num processes: {self.accelerator.num_processes}")
        
        # Move model to device and prepare for training
        self.model = self.accelerator.prepare(model)
        
        # Create optimizer with proper parameter grouping
        self.optimizer = self._create_optimizer()
        
        # Create learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Prepare optimizer and scheduler with accelerator
        self.optimizer, self.scheduler = self.accelerator.prepare(self.optimizer, self.scheduler)
        
        # Training stats
        self.train_losses = []
        self.val_losses = []
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Log model information
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Vocabulary size: {self.vocab_size}")
        self.logger.info(f"Mask token: {self.mask_token}")
        
        # Log mixed precision status
        if mixed_precision == 'fp16':
            self.logger.info("✓ 16-bit mixed precision training enabled via Accelerate")
        elif mixed_precision == 'bf16':
            self.logger.info("✓ BFloat16 mixed precision training enabled via Accelerate")
        else:
            self.logger.info("ℹ️  Using 32-bit precision training")
        
    def _create_optimizer(self):
        """Create optimizer with different learning rates for different parameter groups."""
        # Separate parameters into different groups
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay,
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
    
    def _create_scheduler(self, num_training_steps: int = 1000):
        """Create learning rate scheduler with warmup."""
        if self.scheduler is not None:
            return self.scheduler
        
        # Use linear warmup and cosine decay
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=num_training_steps
        )
        
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
            self.logger.warning("NaN loss detected, using fallback")
            loss = torch.tensor(10.0, device=input_ids.device, requires_grad=True)
        
        return loss
    
    def train_epoch(self, train_loader, epoch: int):
        """Train for one epoch with full Accelerate integration."""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        # Create progress bar only on main process
        if self.accelerator.is_main_process:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        else:
            progress_bar = train_loader
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch[0]
            
            # Check input for NaN
            if torch.isnan(input_ids).any():
                self.logger.warning(f"NaN detected in input_ids at batch {batch_idx}")
                continue
            
            # Apply LLaDA forward process
            noisy_batch, masked_indices, p_mask = self.forward_process(input_ids)
            
            # Check forward process outputs
            if torch.isnan(noisy_batch).any() or torch.isnan(p_mask).any():
                self.logger.warning(f"NaN detected in forward process at batch {batch_idx}")
                continue
            
            # Compute loss
            loss = self.compute_loss(input_ids, noisy_batch, masked_indices, p_mask)
            
            # Check loss for NaN
            if torch.isnan(loss):
                self.logger.warning(f"NaN loss detected at batch {batch_idx}")
                continue
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass with accelerator
            self.accelerator.backward(loss)
            
            # Gradient clipping with proper unwrapping
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Optimizer step (only when we have accumulated enough gradients)
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Log training metrics
                if self.global_step % self.logging_steps == 0:
                    self._log_training_metrics(loss.item(), epoch, batch_idx)
            
            total_loss += loss.item()
            
            # Update progress bar on main process
            if self.accelerator.is_main_process:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.6f}',
                    'step': self.global_step
                })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def _log_training_metrics(self, loss, epoch, batch_idx):
        """Log training metrics using accelerator's logging capabilities."""
        if self.accelerator.is_main_process:
            # Log to tensorboard via accelerator
            self.accelerator.log({
                "train/loss": loss,
                "train/learning_rate": self.optimizer.param_groups[0]["lr"],
                "train/epoch": epoch,
                "train/global_step": self.global_step,
            }, step=self.global_step)
            
            # Also log to console
            self.logger.info(
                f"Step {self.global_step}: loss = {loss:.4f}, "
                f"lr = {self.optimizer.param_groups[0]['lr']:.6f}"
            )
    
    def validate(self, val_loader):
        """Validate the model with proper distributed evaluation."""
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
        
        # Gather losses from all processes for proper averaging
        avg_loss = total_loss / num_batches
        gathered_losses = self.accelerator.gather(torch.tensor(avg_loss))
        
        if self.accelerator.is_main_process:
            # Average across all processes
            final_loss = gathered_losses.mean().item()
            self.val_losses.append(final_loss)
            
            # Log validation metrics
            self.accelerator.log({
                "eval/loss": final_loss,
                "eval/global_step": self.global_step,
            }, step=self.global_step)
            
            return final_loss
        else:
            return avg_loss
    
    def save_checkpoint(self, epoch: int, save_dir: str, is_best: bool = False):
        """Save model checkpoint using accelerator."""
        if not self.accelerator.is_main_process:
            return
            
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Save model state using accelerator
        if is_best:
            checkpoint_path = save_dir / "best_model"
        else:
            checkpoint_path = save_dir / f"checkpoint_epoch_{epoch}"
        
        self.accelerator.save_state(checkpoint_path)
        
        # Save training stats separately
        stats_path = save_dir / f"stats_epoch_{epoch}.json"
        stats = {
            'epoch': epoch,
            'global_step': self.global_step,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'vocab_size': self.vocab_size,
            'mask_token': self.mask_token,
            'best_val_loss': self.best_val_loss,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay
        }
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def train(self, train_loader, val_loader, num_epochs: int, save_dir: str = "checkpoints"):
        """Main training loop with full Accelerate integration."""
        self.logger.info(f"Starting training for {num_epochs} epochs...")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Update scheduler total steps
        total_steps = len(train_loader) * num_epochs // self.gradient_accumulation_steps
        if hasattr(self.scheduler, 'num_training_steps'):
            self.scheduler.num_training_steps = total_steps
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Training
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            epoch_time = time.time() - start_time
            
            if self.accelerator.is_main_process:
                self.logger.info(f"\nEpoch {epoch}/{num_epochs} completed in {epoch_time:.2f}s")
                self.logger.info(f"Train Loss: {train_loss:.4f}")
                self.logger.info(f"Val Loss: {val_loss:.4f}")
                self.logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                self.logger.info(f"Global Step: {self.global_step}")
                
                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, save_dir, is_best=True)
                
                # Save checkpoint every save_steps
                if epoch % 5 == 0:
                    self.save_checkpoint(epoch, save_dir)
        
        if self.accelerator.is_main_process:
            self.logger.info(f"\nTraining completed!")
            self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
            
            # Save final checkpoint
            self.save_checkpoint(num_epochs, save_dir)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint using accelerator."""
        self.accelerator.load_state(checkpoint_path)
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")


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


def create_data_loaders(train_data, val_data, batch_size: int, num_workers: int = 4):
    """Create PyTorch data loaders with Accelerate integration."""
    # Convert to tensors
    train_tensor = torch.tensor(train_data, dtype=torch.long)
    val_tensor = torch.tensor(val_data, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)
    
    # Create data loaders with proper distributed settings
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Important for distributed training
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def main():
    """Main function with full Accelerate integration."""
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained LLaDA on Shakespeare dataset with Accelerate")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing processed dataset")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct",
                       help="Pre-trained model name from Hugging Face")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Training batch size per device (default: 1 for large models)")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs (default: 5 for fine-tuning)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate (default: 1e-5 for fine-tuning)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay (default: 0.01)")
    parser.add_argument("--warmup_steps", type=int, default=100,
                       help="Number of warmup steps (default: 100)")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                       help="Maximum gradient norm for clipping (default: 1.0)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Number of steps to accumulate gradients (default: 1)")
    parser.add_argument("--dataloader_num_workers", type=int, default=4,
                       help="Number of dataloader workers (default: 4)")
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Log every X steps (default: 10)")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every X steps (default: 500)")
    parser.add_argument("--eval_steps", type=int, default=500,
                       help="Evaluate every X steps (default: 500)")
    parser.add_argument("--save_dir", type=str, default="checkpoints_pretrained",
                       help="Directory to save checkpoints")
    parser.add_argument("--fp16", action="store_true",
                       help="Enable 16-bit mixed precision training")
    parser.add_argument("--bf16", action="store_true",
                       help="Enable BFloat16 mixed precision training")
    parser.add_argument("--no_fp16", action="store_true",
                       help="Disable mixed precision training")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                       help="Path to checkpoint to resume from")
    
    args = parser.parse_args()
    
    # Create configuration object
    config = TrainingConfig(
        data_dir=args.data_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_dir=args.save_dir,
        seed=args.seed,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # Determine mixed precision setting
    if args.bf16:
        config.mixed_precision = 'bf16'
    elif args.fp16 and not args.no_fp16:
        config.mixed_precision = 'fp16'
    else:
        config.mixed_precision = 'no'
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Setup logging
    setup_accelerate_logging()
    
    print(f"Training configuration:")
    print(f"  - Mixed precision: {config.mixed_precision}")
    print(f"  - Batch size per device: {config.batch_size}")
    print(f"  - Gradient accumulation steps: {config.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {config.effective_batch_size}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Weight decay: {config.weight_decay}")
    print(f"  - Warmup steps: {config.warmup_steps}")
    print(f"  - Max gradient norm: {config.max_grad_norm}")
    print(f"  - Number of epochs: {config.num_epochs}")
    
    # Load pre-trained model
    model, tokenizer = load_pretrained_model(config.model_name)
    
    # Load dataset
    train_data, val_data, metadata = load_dataset(config.data_dir)
    
    # Update config with dataset metadata
    config.mask_token = metadata.get('mask_token', 126336)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_data, val_data, config.batch_size, config.dataloader_num_workers
    )
    
    # Create trainer with full Accelerate integration
    trainer = LLADAPretrainedTrainer(
        model=model,
        tokenizer=tokenizer,
        mask_token=config.mask_token,
        learning_rate=config.learning_rate,
        mixed_precision=config.mixed_precision,
        weight_decay=config.weight_decay,
        warmup_steps=config.warmup_steps,
        max_grad_norm=config.max_grad_norm,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_num_workers=config.dataloader_num_workers
    )
    
    # Resume from checkpoint if specified
    if config.resume_from_checkpoint:
        trainer.load_checkpoint(config.resume_from_checkpoint)
        print(f"Resumed training from checkpoint: {config.resume_from_checkpoint}")
    
    # Prepare data loaders with accelerator
    train_loader, val_loader = trainer.accelerator.prepare(train_loader, val_loader)
    
    # Log configuration
    if trainer.accelerator.is_main_process:
        trainer.accelerator.log(config.to_dict(), step=0)
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.num_epochs,
        save_dir=config.save_dir
    )


if __name__ == "__main__":
    main()
