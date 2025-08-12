#!/usr/bin/env python3
"""
Test script to demonstrate memory savings with 16-bit mixed precision training.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
import gc


def create_test_model(vocab_size=1000, d_model=768, n_layers=24, n_heads=12):
    """Create a test transformer model similar to our LLaDA implementation."""
    class TestTransformer(nn.Module):
        def __init__(self, vocab_size, d_model, n_layers, n_heads):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, d_model)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=d_model * 4,
                    batch_first=True
                ) for _ in range(n_layers)
            ])
            self.output = nn.Linear(d_model, vocab_size)
        
        def forward(self, x):
            x = self.embedding(x)
            for layer in self.layers:
                x = layer(x)
            return self.output(x)
    
    return TestTransformer(vocab_size, d_model, n_layers, n_heads)


def measure_memory_usage(model, batch_size, seq_len, vocab_size, use_fp16=False):
    """Measure memory usage for a given configuration."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    device = "cuda"
    model = model.to(device)
    
    # Clear cache and measure initial memory
    torch.cuda.empty_cache()
    gc.collect()
    initial_memory = torch.cuda.memory_allocated() / (1024**3)
    
    print(f"  Initial GPU memory: {initial_memory:.2f} GB")
    
    try:
        # Create test input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        
        # Measure memory after input creation
        input_memory = torch.cuda.memory_allocated() / (1024**3)
        print(f"  After input creation: {input_memory:.2f} GB (+{input_memory - initial_memory:.2f} GB)")
        
        # Forward pass
        if use_fp16:
            with autocast():
                outputs = model(input_ids)
        else:
            outputs = model(input_ids)
        
        # Measure memory after forward pass
        forward_memory = torch.cuda.memory_allocated() / (1024**3)
        print(f"  After forward pass: {forward_memory:.2f} GB (+{forward_memory - initial_memory:.2f} GB)")
        
        # Compute loss
        loss = nn.functional.cross_entropy(
            outputs.view(-1, vocab_size), 
            targets.view(-1)
        )
        
        # Backward pass
        if use_fp16:
            scaler = GradScaler()
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Measure memory after backward pass
        backward_memory = torch.cuda.memory_allocated() / (1024**3)
        print(f"  After backward pass: {backward_memory:.2f} GB (+{backward_memory - initial_memory:.2f} GB)")
        
        # Clean up
        del outputs, loss, input_ids, targets
        if use_fp16:
            del scaler
        torch.cuda.empty_cache()
        gc.collect()
        
        return {
            'input_memory': input_memory - initial_memory,
            'forward_memory': forward_memory - initial_memory,
            'backward_memory': backward_memory - initial_memory,
            'total_memory': backward_memory - initial_memory
        }
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def compare_precision_modes():
    """Compare memory usage between FP32 and FP16."""
    print("=" * 60)
    print("16-bit vs 32-bit Memory Usage Comparison")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, cannot run memory comparison")
        return
    
    # Test configurations
    configs = [
        {"batch_size": 1, "seq_len": 512, "name": "Small"},
        {"batch_size": 2, "seq_len": 1024, "name": "Medium"},
        {"batch_size": 4, "seq_len": 2048, "name": "Large"},
    ]
    
    vocab_size = 1000
    d_model = 768
    n_layers = 24
    n_heads = 12
    
    print(f"Model configuration:")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Model dimensions: {d_model}")
    print(f"  - Number of layers: {n_layers}")
    print(f"  - Number of heads: {n_heads}")
    print(f"  - Total parameters: {vocab_size * d_model + n_layers * (4 * d_model * d_model + 2 * d_model):,}")
    
    for config in configs:
        print(f"\n--- {config['name']} Configuration ---")
        print(f"Batch size: {config['batch_size']}, Sequence length: {config['seq_len']}")
        
        # Create model
        model = create_test_model(vocab_size, d_model, n_layers, n_heads)
        
        # Test FP32
        print(f"\n  FP32 (32-bit) Training:")
        fp32_results = measure_memory_usage(
            model, config['batch_size'], config['seq_len'], vocab_size, use_fp16=False
        )
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        # Test FP16
        print(f"\n  FP16 (16-bit) Training:")
        model = create_test_model(vocab_size, d_model, n_layers, n_heads)
        fp16_results = measure_memory_usage(
            model, config['batch_size'], config['seq_len'], vocab_size, use_fp16=True
        )
        
        # Clean up
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        # Compare results
        if fp32_results and fp16_results:
            print(f"\n  Memory Savings:")
            total_savings = fp32_results['total_memory'] - fp16_results['total_memory']
            savings_percent = (total_savings / fp32_results['total_memory']) * 100
            print(f"    - Total memory saved: {total_savings:.2f} GB ({savings_percent:.1f}%)")
            print(f"    - FP32 total: {fp32_results['total_memory']:.2f} GB")
            print(f"    - FP16 total: {fp16_results['total_memory']:.2f} GB")


def main():
    """Main function."""
    print("Testing 16-bit mixed precision training memory usage...")
    
    try:
        compare_precision_modes()
        
        print(f"\n" + "=" * 60)
        print("USAGE INSTRUCTIONS")
        print("=" * 60)
        print("To use 16-bit training in your scripts:")
        print(f"\n1. Pre-trained model training:")
        print(f"   python train_llada_pretrained.py --data_dir shakespeare_dataset --fp16")
        print(f"\n2. Scratch training:")
        print(f"   python train_llada_shakespeare.py --data_dir shakespeare_dataset --fp16")
        print(f"\n3. Disable 16-bit (use 32-bit):")
        print(f"   python train_llada_pretrained.py --data_dir shakespeare_dataset --no_fp16")
        
        print(f"\nBenefits of 16-bit training:")
        print(f"  - ~50% memory reduction")
        print(f"  - Faster training (especially on modern GPUs)")
        print(f"  - Larger batch sizes possible")
        print(f"  - Maintains training quality")
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
