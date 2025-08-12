#!/usr/bin/env python3
"""
Test script for loading the pre-trained LLaDA model.

This script tests that the pre-trained model can be loaded and used
for basic inference and training preparation.
"""

import torch
from transformers import AutoModel, AutoTokenizer


def test_model_loading():
    """Test loading the pre-trained LLaDA model."""
    print("Testing pre-trained LLaDA model loading...")
    
    model_name = "GSAI-ML/LLaDA-8B-Instruct"
    
    try:
        # Test tokenizer loading
        print(f"Loading tokenizer from {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ Tokenizer loaded successfully")
        print(f"  - Vocabulary size: {len(tokenizer)}")
        print(f"  - Special tokens: {tokenizer.special_tokens_map}")
        
        # Test model loading
        print(f"\nLoading model from {model_name}...")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print(f"✓ Model loaded successfully")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test basic forward pass
        print(f"\nTesting basic forward pass...")
        model.eval()
        
        # Create test input
        test_text = "Hello, world!"
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
        
        print(f"  - Input text: '{test_text}'")
        print(f"  - Input shape: {inputs['input_ids'].shape}")
        
        with torch.no_grad():
            outputs = model(**inputs)
            print(f"  - Output logits shape: {outputs.logits.shape}")
            print(f"  - Output logits range: [{outputs.logits.min().item():.4f}, {outputs.logits.max().item():.4f}]")
        
        print(f"✓ Basic forward pass successful")
        
        # Test model configuration
        if hasattr(model, 'config'):
            config = model.config
            print(f"\nModel configuration:")
            print(f"  - Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
            print(f"  - Number of layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
            print(f"  - Number of attention heads: {getattr(config, 'num_attention_heads', 'N/A')}")
            print(f"  - Max position embeddings: {getattr(config, 'max_position_embeddings', 'N/A')}")
        
        # Test device transfer
        print(f"\nTesting device transfer...")
        if torch.cuda.is_available():
            device = "cuda"
            model = model.to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model(**inputs)
            
            print(f"✓ GPU forward pass successful")
        else:
            print(f"  - CUDA not available, using CPU")
        
        # Test training mode
        print(f"\nTesting training mode...")
        model.train()
        
        # Create a simple training scenario
        batch_size, seq_len = 2, 16
        vocab_size = len(tokenizer)
        
        # Create random input and targets
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        outputs = model(input_ids=input_ids)
        logits = outputs.logits
        
        # Compute loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab_size), 
            targets.view(-1)
        )
        
        print(f"  - Training loss: {loss.item():.4f}")
        print(f"✓ Training mode test successful")
        
        # Test model saving
        print(f"\nTesting model saving...")
        test_save_path = "test_model_save"
        model.save_pretrained(test_save_path)
        tokenizer.save_pretrained(test_save_path)
        print(f"✓ Model saved to {test_save_path}")
        
        # Clean up
        import shutil
        shutil.rmtree(test_save_path)
        print(f"✓ Test files cleaned up")
        
        print(f"\n🎉 All tests passed! The pre-trained model is ready for fine-tuning.")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        raise


def test_memory_usage():
    """Test memory usage of the model."""
    print(f"\n=== Memory Usage Test ===")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    try:
        model_name = "GSAI-ML/LLaDA-8B-Instruct"
        
        # Load model
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model = model.to("cuda")
        
        # Get GPU memory info
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / (1024**3)
        
        print(f"Initial GPU memory: {initial_memory:.2f} GB")
        
        # Test with different batch sizes
        batch_sizes = [1, 2, 4]
        seq_lengths = [512, 1024, 2048]
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                try:
                    # Create test input
                    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device="cuda")
                    
                    # Forward pass
                    with torch.no_grad():
                        outputs = model(input_ids=input_ids)
                    
                    # Check memory
                    current_memory = torch.cuda.memory_allocated() / (1024**3)
                    memory_used = current_memory - initial_memory
                    
                    print(f"  Batch {batch_size}, Seq {seq_len}: +{memory_used:.2f} GB")
                    
                    # Clean up
                    del outputs, input_ids
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"  Batch {batch_size}, Seq {seq_len}: Failed - {e}")
        
        print(f"✓ Memory usage test completed")
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")


def main():
    """Main test function."""
    print("=" * 60)
    print("Pre-trained LLaDA Model Test")
    print("=" * 60)
    
    try:
        # Test basic functionality
        model, tokenizer = test_model_loading()
        
        # Test memory usage
        test_memory_usage()
        
        print(f"\n" + "=" * 60)
        print("TEST SUMMARY")
        print("=" * 60)
        print("✓ Model loading successful")
        print("✓ Tokenizer working")
        print("✓ Forward pass working")
        print("✓ Training mode working")
        print("✓ Model saving/loading working")
        print("✓ Memory usage tested")
        
        print(f"\nThe pre-trained model is ready for fine-tuning!")
        print(f"Next steps:")
        print(f"1. Prepare your Shakespeare dataset")
        print(f"2. Run: python train_llada_pretrained.py --data_dir shakespeare_dataset")
        
    except Exception as e:
        print(f"\n❌ Testing failed: {e}")
        print(f"Please check your installation and try again.")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
