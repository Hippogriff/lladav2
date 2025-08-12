#!/usr/bin/env python3
"""
Debug script for NaN issues in LLaDA training.

This script helps identify common causes of NaN during training and provides
solutions to fix them.
"""

import torch
import torch.nn as nn
import numpy as np


def check_model_weights(model, name="Model"):
    """Check if model weights contain NaN or infinite values."""
    print(f"\n=== {name} Weight Check ===")
    
    total_params = 0
    nan_params = 0
    inf_params = 0
    zero_params = 0
    
    for param_name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            
            if torch.isnan(param).any():
                nan_params += param.numel()
                print(f"  ❌ {param_name}: NaN detected")
            
            if torch.isinf(param).any():
                inf_params += param.numel()
                print(f"  ❌ {param_name}: Inf detected")
            
            if torch.all(param == 0):
                zero_params += param.numel()
                print(f"  ⚠️  {param_name}: All zeros")
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  NaN parameters: {nan_params:,}")
    print(f"  Inf parameters: {inf_params:,}")
    print(f"  Zero parameters: {zero_params:,}")
    
    return nan_params == 0 and inf_params == 0


def check_gradients(model, name="Model"):
    """Check if model gradients contain NaN or infinite values."""
    print(f"\n=== {name} Gradient Check ===")
    
    total_grads = 0
    nan_grads = 0
    inf_grads = 0
    zero_grads = 0
    
    for param_name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            total_grads += param.grad.numel()
            
            if torch.isnan(param.grad).any():
                nan_grads += param.grad.numel()
                print(f"  ❌ {param_name}: NaN gradients")
            
            if torch.isinf(param.grad).any():
                inf_grads += param.grad.numel()
                print(f"  ❌ {param_name}: Inf gradients")
            
            if torch.all(param.grad == 0):
                zero_grads += param.grad.numel()
                print(f"  ⚠️  {param_name}: Zero gradients")
    
    print(f"  Total gradients: {total_grads:,}")
    print(f"  NaN gradients: {nan_grads:,}")
    print(f"  Inf gradients: {inf_grads:,}")
    print(f"  Zero gradients: {zero_grads:,}")
    
    return nan_grads == 0 and inf_grads == 0


def check_loss_computation(input_ids, model, mask_token=126336):
    """Check loss computation for numerical stability."""
    print(f"\n=== Loss Computation Check ===")
    
    # Test forward pass
    try:
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            print(f"  ✓ Forward pass successful")
            print(f"  Logits shape: {logits.shape}")
            print(f"  Logits range: [{logits.min().item():.4f}, {logits.max().item():.4f}]")
            
            if torch.isnan(logits).any():
                print(f"  ❌ NaN detected in logits")
                return False
            if torch.isinf(logits).any():
                print(f"  ❌ Inf detected in logits")
                return False
                
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        return False
    
    # Test LLaDA forward process
    try:
        batch_size, seq_len = input_ids.shape
        eps = 1e-3
        
        # Sample random time t
        t = torch.rand(batch_size, device=input_ids.device)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, seq_len)
        
        # Create mask
        masked_indices = torch.rand((batch_size, seq_len), device=input_ids.device) < p_mask
        noisy_batch = torch.where(masked_indices, mask_token, input_ids)
        
        print(f"  ✓ LLaDA forward process successful")
        print(f"  Masking probability range: [{p_mask.min().item():.4f}, {p_mask.max().item():.4f}]")
        print(f"  Masked tokens: {masked_indices.sum().item()}")
        
        if torch.isnan(p_mask).any():
            print(f"  ❌ NaN detected in masking probability")
            return False
            
    except Exception as e:
        print(f"  ❌ LLaDA forward process failed: {e}")
        return False
    
    # Test loss computation
    try:
        model.train()
        outputs = model(noisy_batch)
        logits = outputs.logits
        
        # Compute token-level loss
        token_loss = nn.functional.cross_entropy(
            logits[masked_indices], 
            input_ids[masked_indices], 
            reduction='none'
        )
        
        # Apply importance weighting
        eps = 1e-8
        p_mask_stable = torch.clamp(p_mask, min=eps, max=1.0)
        token_loss = token_loss / p_mask_stable[masked_indices]
        
        # Final loss
        loss = token_loss.sum() / (input_ids.shape[0] * input_ids.shape[1])
        
        print(f"  ✓ Loss computation successful")
        print(f"  Loss value: {loss.item():.4f}")
        
        if torch.isnan(loss):
            print(f"  ❌ NaN detected in final loss")
            return False
        if torch.isinf(loss):
            print(f"  ❌ Inf detected in final loss")
            return False
            
    except Exception as e:
        print(f"  ❌ Loss computation failed: {e}")
        return False
    
    return True


def suggest_fixes():
    """Suggest fixes for common NaN issues."""
    print(f"\n=== Suggested Fixes for NaN Issues ===")
    
    print(f"\n1. Learning Rate Issues:")
    print(f"   - Reduce learning rate (try 1e-5 instead of 1e-4)")
    print(f"   - Use learning rate warmup")
    print(f"   - Implement gradient clipping")
    
    print(f"\n2. Weight Initialization:")
    print(f"   - Use Xavier/Glorot initialization")
    print(f"   - Reduce initial weight variance")
    print(f"   - Check for zero or NaN in initial weights")
    
    print(f"\n3. Numerical Stability:")
    print(f"   - Add epsilon to prevent division by zero")
    print(f"   - Clamp logits to reasonable range")
    print(f"   - Use mixed precision training")
    
    print(f"\n4. Model Architecture:")
    print(f"   - Reduce model size initially")
    print(f"   - Add layer normalization")
    print(f"   - Use residual connections properly")
    
    print(f"\n5. Data Issues:")
    print(f"   - Check input data for NaN")
    print(f"   - Normalize input data")
    print(f"   - Use smaller batch size")


def main():
    """Main debugging function."""
    print("=" * 60)
    print("LLaDA Training NaN Debugger")
    print("=" * 60)
    
    print(f"\nThis script helps identify common causes of NaN during training.")
    print(f"Run this after encountering NaN issues to diagnose the problem.")
    
    suggest_fixes()
    
    print(f"\n" + "=" * 60)
    print("To use this debugger in your training script:")
    print("1. Import the check functions")
    print("2. Call them at key points during training")
    print("3. Monitor the output for warnings")
    print("4. Apply the suggested fixes")


if __name__ == "__main__":
    main()
