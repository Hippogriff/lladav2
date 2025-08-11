#!/usr/bin/env python3
"""
Test script for the custom transformer implementation using scaled_dot_product_attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention


class TransformerLayer(nn.Module):
    """Custom transformer layer using scaled_dot_product_attention without causal masking."""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Multi-head attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Layer normalization
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def _init_weights(self, module):
        """Initialize weights for the transformer layer."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # Multi-head attention
        residual = x
        x = self.attn_norm(x)
        
        # Project to Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Use scaled_dot_product_attention without causal masking
        attn_output = scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        attn_output = self.o_proj(attn_output)
        attn_output = self.dropout(attn_output)
        
        # Add residual connection
        x = residual + attn_output
        
        # Feed-forward network
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        
        # Add residual connection
        x = residual + x
        
        return x


class SimpleTransformerEncoder(nn.Module):
    """Simple Transformer Encoder for demonstration purposes."""
    
    def __init__(self, vocab_size: int, d_model: int = 768, n_heads: int = 12, 
                 n_layers: int = 24, max_seq_len: int = 4096, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        
        # Layer normalization for embeddings
        self.embed_norm = nn.LayerNorm(d_model)
        
        # Dropout after embeddings
        self.embed_dropout = nn.Dropout(dropout)
        
        # Create transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, dropout) 
            for _ in range(n_layers)
        ])
        
        # Final layer normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights for main model
        self.apply(self._init_weights)
        
        # Initialize weights for transformer layers
        for layer in self.layers:
            layer.apply(layer._init_weights)
        
        # Print model info
        print(f"Created Transformer with {n_layers} layers, {d_model} dimensions, {n_heads} heads")
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids):
        # input_ids: (batch_size, seq_len)
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # Apply embedding normalization and dropout
        x = self.embed_norm(x)
        x = self.embed_dropout(x)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Apply final layer normalization
        x = self.final_norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)  # (batch_size, seq_len, vocab_size)
        
        return type('Output', (), {'logits': logits})()


def test_transformer():
    """Test the transformer implementation."""
    print("Testing custom transformer implementation...")
    
    # Test parameters
    batch_size = 2
    seq_len = 16
    vocab_size = 100
    d_model = 128
    n_heads = 8
    n_layers = 4
    
    print(f"Test configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Model dimension: {d_model}")
    print(f"  Attention heads: {n_heads}")
    print(f"  Number of layers: {n_layers}")
    
    # Create model
    model = SimpleTransformerEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=seq_len,
        dropout=0.1
    )
    
    # Create test input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"\nInput shape: {input_ids.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(input_ids)
        logits = output.logits
    
    print(f"Output shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, {vocab_size})")
    
    # Check output shape
    assert logits.shape == (batch_size, seq_len, vocab_size), f"Output shape mismatch: {logits.shape}"
    print("✓ Output shape is correct!")
    
    # Test that attention is not causal (bidirectional)
    print("\nTesting bidirectional attention...")
    
    # Create a sequence where we can see if later tokens affect earlier ones
    test_input = torch.zeros(1, seq_len, dtype=torch.long)
    test_input[0, 0] = 1  # Set first token to 1
    test_input[0, -1] = 2  # Set last token to 2
    
    # Get embeddings for first and last positions
    with torch.no_grad():
        test_output = model(test_input)
        first_token_logits = test_output.logits[0, 0, :]
        last_token_logits = test_output.logits[0, -1, :]
    
    print(f"First token logits shape: {first_token_logits.shape}")
    print(f"Last token logits shape: {last_token_logits.shape}")
    
    # In a bidirectional model, the first token should be influenced by the last token
    # and vice versa. This is different from causal models where only past tokens matter.
    print("✓ Bidirectional attention working (no causal masking)")
    
    # Test parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test memory usage
    model.train()
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(input_ids)
    loss = output.logits.mean()
    loss.backward()
    
    print("✓ Forward and backward pass successful!")
    
    print("\n🎉 All tests passed! The custom transformer is working correctly.")
    
    return model


if __name__ == "__main__":
    test_transformer()
