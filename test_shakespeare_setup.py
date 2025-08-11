#!/usr/bin/env python3
"""
Test script to verify Shakespeare dataset setup for LLaDA.

This script tests that all dependencies are available and the basic
functionality can be imported without errors.
"""

import sys
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError as e:
        print(f"✗ NumPy import failed: {e}")
        return False
    
    try:
        import requests
        print(f"✓ Requests: {requests.__version__}")
    except ImportError as e:
        print(f"✗ Requests import failed: {e}")
        return False
    
    try:
        import tqdm
        print(f"✓ TQDM: {tqdm.__version__}")
    except ImportError as e:
        print(f"✗ TQDM import failed: {e}")
        return False
    
    try:
        import json
        print(f"✓ JSON: Built-in module")
    except ImportError as e:
        print(f"✗ JSON import failed: {e}")
        return False
    
    return True


def test_downloader_import():
    """Test that the downloader script can be imported."""
    print("\nTesting downloader script import...")
    
    try:
        # This will test that the script can be parsed without syntax errors
        with open("download_shakespeare_dataset.py", "r") as f:
            content = f.read()
        
        # Basic syntax check
        compile(content, "download_shakespeare_dataset.py", "exec")
        print("✓ download_shakespeare_dataset.py syntax is valid")
        return True
        
    except Exception as e:
        print(f"✗ download_shakespeare_dataset.py import failed: {e}")
        return False


def test_training_script_import():
    """Test that the training script can be imported."""
    print("\nTesting training script import...")
    
    try:
        # This will test that the script can be parsed without syntax errors
        with open("train_llada_shakespeare.py", "r") as f:
            content = f.read()
        
        # Basic syntax check
        compile(content, "train_llada_shakespeare.py", "exec")
        print("✓ train_llada_shakespeare.py syntax is valid")
        return True
        
    except Exception as e:
        print(f"✗ train_llada_shakespeare.py import failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        "download_shakespeare_dataset.py",
        "train_llada_shakespeare.py",
        "requirements.txt",
        "SHAKESPEARE_README.md"
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"✓ {file} exists")
        else:
            print(f"✗ {file} missing")
            all_exist = False
    
    return all_exist


def test_basic_functionality():
    """Test basic functionality by creating a small test."""
    print("\nTesting basic functionality...")
    
    try:
        # Test that we can create a simple transformer model
        import torch
        import torch.nn as nn
        
        # Simple test model
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(100, 64)
                self.linear = nn.Linear(64, 100)
            
            def forward(self, x):
                return self.linear(self.embedding(x))
        
        model = TestModel()
        x = torch.randint(0, 100, (2, 10))
        output = model(x)
        
        print(f"✓ Test model created successfully")
        print(f"  - Input shape: {x.shape}")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def test_llada_concepts():
    """Test LLaDA-specific concepts."""
    print("\nTesting LLaDA concepts...")
    
    try:
        import torch
        
        # Test LLaDA masking process
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 100, (batch_size, seq_len))
        mask_token = 126336
        
        # LLaDA forward process
        eps = 1e-3
        t = torch.rand(batch_size)
        p_mask = (1 - eps) * t + eps
        p_mask = p_mask[:, None].repeat(1, seq_len)
        
        masked_indices = torch.rand((batch_size, seq_len)) < p_mask
        noisy_batch = torch.where(masked_indices, mask_token, input_ids)
        
        print(f"✓ LLaDA masking process works")
        print(f"  - Original shape: {input_ids.shape}")
        print(f"  - Masked shape: {noisy_batch.shape}")
        print(f"  - Masked tokens: {masked_indices.sum().item()}")
        
        return True
        
    except Exception as e:
        print(f"✗ LLaDA concepts test failed: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("Shakespeare Dataset Setup Test for LLaDA")
    print("=" * 60)
    
    tests = [
        ("Import Dependencies", test_imports),
        ("Downloader Script", test_downloader_import),
        ("Training Script", test_training_script_import),
        ("File Structure", test_file_structure),
        ("Basic Functionality", test_basic_functionality),
        ("LLaDA Concepts", test_llada_concepts),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:20} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your Shakespeare dataset setup is ready.")
        print("\nNext steps:")
        print("1. Run: python download_shakespeare_dataset.py")
        print("2. Run: python train_llada_shakespeare.py --data_dir shakespeare_dataset")
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
