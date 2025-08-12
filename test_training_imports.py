#!/usr/bin/env python3
"""
Test script to verify that training scripts can be imported without errors.
"""

def test_pretrained_training_import():
    """Test importing the pre-trained training script."""
    try:
        print("Testing pre-trained training script import...")
        import train_llada_pretrained
        print("✓ train_llada_pretrained.py imported successfully")
        return True
    except Exception as e:
        print(f"✗ train_llada_pretrained.py import failed: {e}")
        return False


def test_scratch_training_import():
    """Test importing the scratch training script."""
    try:
        print("Testing scratch training script import...")
        import train_llada_shakespeare
        print("✓ train_llada_shakespeare.py imported successfully")
        return True
    except Exception as e:
        print(f"✗ train_llada_shakespeare.py import failed: {e}")
        return False


def test_fp16_memory_import():
    """Test importing the FP16 memory test script."""
    try:
        print("Testing FP16 memory test script import...")
        import test_fp16_memory
        print("✓ test_fp16_memory.py imported successfully")
        return True
    except Exception as e:
        print(f"✗ test_fp16_memory.py import failed: {e}")
        return False


def main():
    """Main test function."""
    print("=" * 60)
    print("Training Script Import Test")
    print("=" * 60)
    
    tests = [
        ("Pre-trained Training", test_pretrained_training_import),
        ("Scratch Training", test_scratch_training_import),
        ("FP16 Memory Test", test_fp16_memory_import),
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
        print(f"{test_name:25} : {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All training scripts can be imported successfully!")
        print("\nNext steps:")
        print("1. Test 16-bit memory usage: python test_fp16_memory.py")
        print("2. Train from scratch: python train_llada_shakespeare.py --data_dir shakespeare_dataset")
        print("3. Fine-tune pre-trained: python train_llada_pretrained.py --data_dir shakespeare_dataset")
    else:
        print(f"\n❌ {total - passed} test(s) failed. Please check the errors above.")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
