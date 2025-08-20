#!/usr/bin/env python3
"""
NCCL Troubleshooting Script

This script helps diagnose and fix common NCCL issues in distributed training.
"""

import os
import torch
import subprocess
import sys


def check_cuda_installation():
    """Check CUDA installation and version."""
    print("=== CUDA Installation Check ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available in PyTorch")
        return False
    
    print(f"✅ CUDA available in PyTorch")
    print(f"   PyTorch CUDA version: {torch.version.cuda}")
    print(f"   CUDA device count: {torch.cuda.device_count()}")
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ nvidia-smi available")
            # Extract driver version
            for line in result.stdout.split('\n'):
                if 'Driver Version:' in line:
                    driver_version = line.split('Driver Version:')[1].strip()
                    print(f"   Driver version: {driver_version}")
                    break
        else:
            print("❌ nvidia-smi failed")
    except FileNotFoundError:
        print("❌ nvidia-smi not found")
    
    return True


def check_gpu_communication():
    """Test basic GPU communication."""
    print("\n=== GPU Communication Test ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping GPU test")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPU(s)")
    
    if num_gpus <= 1:
        print("ℹ️  Single GPU detected, NCCL not needed")
        return True
    
    # Test basic operations on each GPU
    for i in range(num_gpus):
        try:
            torch.cuda.set_device(i)
            x = torch.randn(100, 100, device=f'cuda:{i}')
            y = torch.randn(100, 100, device=f'cuda:{i}')
            z = torch.mm(x, y)
            print(f"✅ GPU {i}: Basic operations OK")
        except Exception as e:
            print(f"❌ GPU {i}: Failed - {e}")
            return False
    
    # Test inter-GPU communication if multiple GPUs
    if num_gpus > 1:
        try:
            print("Testing inter-GPU communication...")
            x = torch.randn(100, 100, device='cuda:0')
            y = x.to('cuda:1')
            z = y.to('cuda:0')
            print("✅ Inter-GPU communication OK")
        except Exception as e:
            print(f"❌ Inter-GPU communication failed: {e}")
            return False
    
    return True


def setup_nccl_environment():
    """Setup NCCL environment variables."""
    print("\n=== NCCL Environment Setup ===")
    
    nccl_vars = {
        'NCCL_DEBUG': 'INFO',
        'NCCL_IB_DISABLE': '1',
        'NCCL_P2P_DISABLE': '1',
        'NCCL_SOCKET_IFNAME': 'lo',
        'NCCL_BLOCKING_WAIT': '1',
        'NCCL_TIMEOUT': '1800',
        'NCCL_SHM_DISABLE': '1',
        'NCCL_NET_GDR_LEVEL': '0'
    }
    
    print("Setting NCCL environment variables:")
    for var, value in nccl_vars.items():
        os.environ[var] = value
        print(f"  {var}={value}")
    
    print("✅ NCCL environment variables set")


def test_distributed_training():
    """Test basic distributed training setup."""
    print("\n=== Distributed Training Test ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping distributed test")
        return False
    
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        print("ℹ️  Single GPU detected, skipping distributed test")
        return True
    
    try:
        # Test basic distributed setup
        from accelerate import Accelerator
        
        accelerator = Accelerator(
            mixed_precision='no',
            gradient_accumulation_steps=1
        )
        
        # Create a simple model
        model = torch.nn.Linear(10, 1).to(accelerator.device)
        # model = torch.nn.parallel.DistributedDataParallel(model)
        optimizer = torch.optim.Adam(model.parameters())
        
        # Prepare model and optimizer
        model, optimizer = accelerator.prepare(model, optimizer)
        
        # Test forward pass
        x = torch.randn(5, 10).to(accelerator.device)
        y = model(x)
        loss = y.sum()
        
        # Test backward pass
        accelerator.backward(loss)
        optimizer.step()
        
        print("✅ Basic distributed training test passed")
        return True
        
    except Exception as e:
        print(f"❌ Distributed training test failed: {e}")
        return False


def recommend_solution():
    """Recommend solution based on test results."""
    print("\n=== Recommendations ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available. Please install CUDA and PyTorch with CUDA support.")
        return
    
    num_gpus = torch.cuda.device_count()
    
    if num_gpus <= 1:
        print("ℹ️  Single GPU detected. Use standard training:")
        print("   python train_llada_pretrained.py --data_dir <your_data_dir>")
        return
    
    print("🔧 Multi-GPU detected. For NCCL issues, try these solutions in order:")
    print("")
    print("1. Single-machine multi-GPU (most reliable):")
    print("   accelerate launch --config_file accelerate_config_single_machine.yaml train_llada_pretrained.py --data_dir <your_data_dir>")
    print("")
    print("2. Multi-GPU with NCCL fixes:")
    print("   accelerate launch --config_file accelerate_config.yaml train_llada_pretrained.py --data_dir <your_data_dir>")
    print("")
    print("3. Single GPU (fallback):")
    print("   CUDA_VISIBLE_DEVICES=0 python train_llada_pretrained.py --data_dir <your_data_dir>")
    print("")
    print("4. Debug mode:")
    print("   NCCL_DEBUG=INFO accelerate launch --config_file accelerate_config.yaml train_llada_pretrained.py --data_dir <your_data_dir>")


def main():
    """Main troubleshooting function."""
    print("NCCL Troubleshooting Script")
    print("=" * 40)
    
    # Run all checks
    cuda_ok = check_cuda_installation()
    gpu_comm_ok = check_gpu_communication()
    
    if cuda_ok and gpu_comm_ok:
        setup_nccl_environment()
        dist_ok = test_distributed_training()
    
    # Provide recommendations
    recommend_solution()
    
    print("\n" + "=" * 40)
    print("Troubleshooting complete!")


if __name__ == "__main__":
    main()
