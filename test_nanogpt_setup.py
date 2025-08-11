#!/usr/bin/env python3
"""
Test script to verify nanoGPT dataset downloader setup.

This script checks that all dependencies are available and the basic
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
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError as e:
        print(f"✗ Transformers import failed: {e}")
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
    
    return True


def test_data_loader_import():
    """Test that the data_loader module can be imported."""
    print("\nTesting data_loader import...")
    
    try:
        # Add current directory to path
        sys.path.append(str(Path(__file__).parent))
        
        from data_loader import LLADADataLoader, PackedDatasetBuilder
        print("✓ data_loader module imported successfully")
        print(f"  - LLADADataLoader: {LLADADataLoader}")
        print(f"  - PackedDatasetBuilder: {PackedDatasetBuilder}")
        return True
        
    except ImportError as e:
        print(f"✗ data_loader import failed: {e}")
        return False


def test_downloader_import():
    """Test that the downloader script can be imported."""
    print("\nTesting downloader script import...")
    
    try:
        # This will test that the script can be parsed without syntax errors
        with open("download_nanogpt_dataset.py", "r") as f:
            content = f.read()
        
        # Basic syntax check
        compile(content, "download_nanogpt_dataset.py", "exec")
        print("✓ download_nanogpt_dataset.py syntax is valid")
        return True
        
    except Exception as e:
        print(f"✗ download_nanogpt_dataset.py import failed: {e}")
        return False


def test_example_import():
    """Test that the example script can be imported."""
    print("\nTesting example script import...")
    
    try:
        # This will test that the script can be parsed without syntax errors
        with open("example_nanogpt_usage.py", "r") as f:
            content = f.read()
        
        # Basic syntax check
        compile(content, "example_nanogpt_usage.py", "exec")
        print("✓ example_nanogpt_usage.py syntax is valid")
        return True
        
    except Exception as e:
        print(f"✗ example_nanogpt_usage.py import failed: {e}")
        return False


def test_file_structure():
    """Test that all required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        "download_nanogpt_dataset.py",
        "example_nanogpt_usage.py", 
        "NANOGPT_DATASET_README.md",
        "data_loader.py",
        "requirements.txt"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} (missing)")
            all_exist = False
    
    return all_exist


def test_basic_functionality():
    """Test basic functionality without downloading data."""
    print("\nTesting basic functionality...")
    
    try:
        # Test that we can create basic objects
        from data_loader import PackedDatasetBuilder
        from pathlib import Path
        
        # Create a temporary directory for testing
        test_dir = Path("./test_temp")
        test_dir.mkdir(exist_ok=True)
        
        # Test PackedDatasetBuilder creation
        builder = PackedDatasetBuilder(
            outdir=test_dir,
            prefix="test",
            chunk_size=1024,
            sep_token=0,
            dtype="auto"
        )
        
        print("✓ PackedDatasetBuilder created successfully")
        
        # Test adding a small array
        import numpy as np
        test_array = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        builder.add_array(test_array)
        builder.write_reminder()
        
        print("✓ Basic array processing works")
        
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("nanoGPT Dataset Downloader Setup Test")
    print("=" * 60)
    
    tests = [
        ("Import Dependencies", test_imports),
        ("Data Loader Import", test_data_loader_import),
        ("Downloader Script", test_downloader_import),
        ("Example Script", test_example_import),
        ("File Structure", test_file_structure),
        ("Basic Functionality", test_basic_functionality),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The setup is ready to use.")
        print("\nNext steps:")
        print("1. Run: python download_nanogpt_dataset.py --max_files 1000")
        print("2. Test with: python example_nanogpt_usage.py")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Check that all files are in the correct location")
        print("3. Verify Python version compatibility (3.8+)")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
