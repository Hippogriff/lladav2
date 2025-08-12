#!/usr/bin/env python3
"""
Quick test to verify that the device attribute error has been fixed.
"""

def test_imports():
    """Test that both training scripts can be imported without device errors."""
    try:
        print("Testing pre-trained training script import...")
        import train_llada_pretrained
        print("✓ train_llada_pretrained.py imported successfully")
    except AttributeError as e:
        if "device" in str(e):
            print(f"✗ Device attribute error still exists: {e}")
            return False
        else:
            print(f"✗ Other attribute error: {e}")
            return False
    except Exception as e:
        print(f"✗ Import failed with error: {e}")
        return False
    
    try:
        print("Testing scratch training script import...")
        import train_llada_shakespeare
        print("✓ train_llada_shakespeare.py imported successfully")
    except AttributeError as e:
        if "device" in str(e):
            print(f"✗ Device attribute error still exists: {e}")
            return False
        else:
            print(f"✗ Other attribute error: {e}")
            return False
    except Exception as e:
        print(f"✗ Import failed with error: {e}")
        return False
    
    return True


def main():
    """Main test function."""
    print("=" * 50)
    print("Device Attribute Fix Test")
    print("=" * 50)
    
    if test_imports():
        print("\n🎉 Device attribute error has been fixed!")
        print("\nBoth training scripts can now be imported successfully.")
        print("\nNext steps:")
        print("1. Test accelerate integration: python test_accelerate_integration.py")
        print("2. Train from scratch: python train_llada_shakespeare.py --data_dir shakespeare_dataset --fp16")
        print("3. Fine-tune pre-trained: python train_llada_pretrained.py --data_dir shakespeare_dataset --fp16")
        return 0
    else:
        print("\n❌ Device attribute error still exists. Please check the errors above.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
