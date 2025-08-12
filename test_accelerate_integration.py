#!/usr/bin/env python3
"""
Test script to verify that the accelerate integration works correctly.
"""

def test_pretrained_training_import():
    """Test importing the pre-trained training script with accelerate."""
    try:
        print("Testing pre-trained training script with accelerate...")
        import train_llada_pretrained
        print("✓ train_llada_pretrained.py imported successfully with accelerate")
        return True
    except Exception as e:
        print(f"✗ train_llada_pretrained.py import failed: {e}")
        return False


def test_scratch_training_import():
    """Test importing the scratch training script with accelerate."""
    try:
        print("Testing scratch training script with accelerate...")
        import train_llada_shakespeare
        print("✓ train_llada_shakespeare.py imported successfully with accelerate")
        return True
    except Exception as e:
        print(f"✗ train_llada_shakespeare.py import failed: {e}")
        return False


def test_accelerate_import():
    """Test that accelerate can be imported."""
    try:
        print("Testing accelerate library import...")
        from accelerate import Accelerator
        from accelerate.logging import get_logger
        from accelerate.utils import set_seed
        print("✓ accelerate library imported successfully")
        return True
    except Exception as e:
        print(f"✗ accelerate library import failed: {e}")
        return False


def test_accelerate_basic_functionality():
    """Test basic accelerate functionality."""
    try:
        print("Testing basic accelerate functionality...")
        from accelerate import Accelerator
        
        # Test accelerator initialization
        accelerator = Accelerator(mixed_precision='fp16')
        print("✓ Accelerator initialized successfully")
        
        # Test device detection
        device = accelerator.device
        print(f"✓ Device detected: {device}")
        
        # Test mixed precision
        mixed_precision = accelerator.mixed_precision
        print(f"✓ Mixed precision: {mixed_precision}")
        
        return True
    except Exception as e:
        print(f"✗ Basic accelerate functionality failed: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 70)
    print("Accelerate Integration Test")
    print("=" * 70)
    
    tests = [
        ("Accelerate Library", test_accelerate_import),
        ("Basic Functionality", test_accelerate_basic_functionality),
        ("Pre-trained Training", test_pretrained_training_import),
        ("Scratch Training", test_scratch_training_import),
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
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{test_name:25} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Accelerate integration successful!")
        print("\nBenefits of using Accelerate:")
        print("✓ Automatic mixed precision training (FP16/BF16)")
        print("✓ Automatic device management (CPU/GPU/TPU)")
        print("✓ Built-in gradient accumulation")
        print("✓ Distributed training support")
        print("✓ Automatic checkpoint saving/loading")
        print("✓ TensorBoard logging integration")
        print("\nNext steps:")
        print("1. Train from scratch: python train_llada_shakespeare.py --data_dir shakespeare_dataset --fp16")
        print("2. Fine-tune pre-trained: python train_llada_pretrained.py --data_dir shakespeare_dataset --fp16")
        print("3. Use BFloat16: python train_llada_shakespeare.py --data_dir shakespeare_dataset --bf16")
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
