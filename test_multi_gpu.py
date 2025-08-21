#!/usr/bin/env python3
"""
Test script for multi-GPU training setup
"""

import torch
import torch.distributed as dist
import os


def test_device_setup():
    """Test basic device setup."""
    print("Testing device setup...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    num_gpus = torch.cuda.device_count()
    print(f"✓ Found {num_gpus} CUDA devices")
    
    # Check each GPU
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    return True


def test_ddp_setup():
    """Test DDP setup."""
    print("\nTesting DDP setup...")
    
    # Check if we're in a distributed environment
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    print(f"World size: {world_size}, Rank: {rank}")
    
    if world_size > 1:
        print("✓ Multi-GPU environment detected")
        
        # Test DDP initialization
        try:
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            torch.cuda.set_device(rank)
            print(f"✓ DDP initialized successfully on rank {rank}")
            
            # Cleanup
            dist.destroy_process_group()
            return True
            
        except Exception as e:
            print(f"❌ DDP initialization failed: {e}")
            return False
    else:
        print("ℹ️  Single GPU mode")
        return True


def test_tensor_creation():
    """Test tensor creation on correct devices."""
    print("\nTesting tensor creation...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available")
        return False
    
    # Test single GPU
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    
    # Create tensors
    tensor_cpu = torch.tensor([1, 2, 3])
    tensor_gpu = torch.tensor([1, 2, 3], device=device)
    
    print(f"✓ CPU tensor device: {tensor_cpu.device}")
    print(f"✓ GPU tensor device: {tensor_gpu.device}")
    
    # Test operations
    try:
        result = tensor_gpu + tensor_gpu
        print(f"✓ GPU operation successful: {result.device}")
    except Exception as e:
        print(f"❌ GPU operation failed: {e}")
        return False
    
    return True


def main():
    """Main test function."""
    print("Multi-GPU Training Setup Test")
    print("=" * 40)
    
    tests = [
        test_device_setup,
        test_ddp_setup,
        test_tensor_creation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 40)
    print("Test Results:")
    print(f"Passed: {sum(results)}/{len(results)}")
    
    if all(results):
        print("🎉 All tests passed! Multi-GPU setup is ready.")
    else:
        print("❌ Some tests failed. Please check the setup.")
    
    return all(results)


if __name__ == "__main__":
    main()
