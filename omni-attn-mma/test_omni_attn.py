"""
Test script to verify Omni-Attention implementation correctness.

Compares Triton implementation against naive PyTorch baseline.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import torch
import torch.nn.functional as F
from omni_attn_torch import (
    BlockMaskType,
    omni_attention_forward,
    create_causal_block_mask,
    create_full_block_mask,
    create_hybrid_block_mask,
    create_omni_block_mask,
)
from omni_attn_naive import (
    omni_attention_forward_naive,
    omni_attention_forward_naive_simple,
    create_block_mask_pattern_naive,
)


def time_and_compare(
    q, k, v, kv_num_blocks, kv_indices, block_mask_types, seqlen_q, device, num_iterations=10
):
    """Time both implementations and compare results."""
    # Warmup
    for _ in range(3):
        _ = omni_attention_forward(
            q, k, v, kv_num_blocks, kv_indices, block_mask_types
        )
        _ = omni_attention_forward_naive(
            q, k, v, kv_num_blocks, kv_indices, block_mask_types
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Time tiled implementation
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        output_tiled, lse_tiled = omni_attention_forward(
            q, k, v, kv_num_blocks, kv_indices, block_mask_types
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    tiled_time = (time.perf_counter() - start_time) / num_iterations * 1000  # ms
    
    # Time naive implementation
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        output_naive, lse_naive = omni_attention_forward_naive(
            q, k, v, kv_num_blocks, kv_indices, block_mask_types
        )
    if device.type == "cuda":
        torch.cuda.synchronize()
    naive_time = (time.perf_counter() - start_time) / num_iterations * 1000  # ms
    
    # Compare
    max_diff = (output_tiled - output_naive).abs().max().item()
    max_lse_diff = (lse_tiled[:, :, :seqlen_q] - lse_naive).abs().max().item()
    
    print(f"  Max output diff: {max_diff:.6f}")
    print(f"  Max LSE diff: {max_lse_diff:.6f}")
    print(f"  Tiled time: {tiled_time:.3f} ms")
    print(f"  Naive time: {naive_time:.3f} ms")
    print(f"  Speedup: {naive_time / tiled_time:.2f}x")
    
    assert max_diff < 1e-3, f"Output difference too large: {max_diff}"
    assert max_lse_diff < 1e-3, f"LSE difference too large: {max_lse_diff}"
    
    return output_tiled, lse_tiled, output_naive, lse_naive


def test_causal_attention():
    """Test causal attention pattern."""
    print("Testing causal attention...")
    
    batch, seqlen_q, seqlen_k, nheads, d = 2, 512, 512, 4, 64
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Create random Q, K, V
    q = torch.randn(batch, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen_k, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen_k, nheads, d, device=device, dtype=dtype)
    
    # Create block mask
    kv_num_blocks, kv_indices, block_mask_types = create_causal_block_mask(
        batch, nheads, seqlen_q, seqlen_k, device=device
    )
    
    # Time and compare
    time_and_compare(q, k, v, kv_num_blocks, kv_indices, block_mask_types, seqlen_q, device)
    print("  ✓ Causal attention test passed!\n")


def test_full_attention():
    """Test full attention pattern."""
    print("Testing full attention...")
    
    batch, seqlen_q, seqlen_k, nheads, d = 2, 512, 512, 4, 64
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Create random Q, K, V
    q = torch.randn(batch, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen_k, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen_k, nheads, d, device=device, dtype=dtype)
    
    # Create block mask
    kv_num_blocks, kv_indices, block_mask_types = create_full_block_mask(
        batch, nheads, seqlen_q, seqlen_k, device=device
    )
    
    # Time and compare
    time_and_compare(q, k, v, kv_num_blocks, kv_indices, block_mask_types, seqlen_q, device)
    print("  ✓ Full attention test passed!\n")


def test_hybrid_attention():
    """Test hybrid attention (prefix + causal)."""
    print("Testing hybrid attention (prefix + causal)...")
    
    batch, seqlen_q, seqlen_k, nheads, d = 2, 512, 512, 4, 64
    prefix_len = 128
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Create random Q, K, V
    q = torch.randn(batch, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen_k, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen_k, nheads, d, device=device, dtype=dtype)
    
    # Create block mask
    kv_num_blocks, kv_indices, block_mask_types = create_hybrid_block_mask(
        batch, nheads, seqlen_q, seqlen_k, prefix_len, device=device
    )
    
    # Time and compare
    time_and_compare(q, k, v, kv_num_blocks, kv_indices, block_mask_types, seqlen_q, device)
    print("  ✓ Hybrid attention test passed!\n")


def test_custom_pattern():
    """Test custom block pattern."""
    print("Testing custom block pattern...")
    
    batch, seqlen_q, seqlen_k, nheads, d = 2, 512, 512, 4, 64
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Create random Q, K, V
    q = torch.randn(batch, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen_k, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen_k, nheads, d, device=device, dtype=dtype)
    
    # Custom pattern: first 2 blocks full, rest causal
    def custom_pattern(q_idx, kv_idx):
        if q_idx < 2 and kv_idx < 2:
            return BlockMaskType.FULL
        elif kv_idx <= q_idx:
            return BlockMaskType.CAUSAL
        else:
            return BlockMaskType.MASKED
    
    # Create block mask
    kv_num_blocks, kv_indices, block_mask_types = create_omni_block_mask(
        batch, nheads, seqlen_q, seqlen_k, custom_pattern, device=device
    )
    
    # Time and compare
    time_and_compare(q, k, v, kv_num_blocks, kv_indices, block_mask_types, seqlen_q, device)
    print("  ✓ Custom pattern test passed!\n")


def test_variable_lengths():
    """Test with non-multiple-of-block sequence lengths."""
    print("Testing variable sequence lengths...")
    
    batch, seqlen_q, seqlen_k, nheads, d = 2, 513, 511, 4, 64
    device = torch.device("cuda")
    dtype = torch.float16
    
    # Create random Q, K, V
    q = torch.randn(batch, seqlen_q, nheads, d, device=device, dtype=dtype)
    k = torch.randn(batch, seqlen_k, nheads, d, device=device, dtype=dtype)
    v = torch.randn(batch, seqlen_k, nheads, d, device=device, dtype=dtype)
    
    # Create block mask
    kv_num_blocks, kv_indices, block_mask_types = create_causal_block_mask(
        batch, nheads, seqlen_q, seqlen_k, device=device
    )
    
    # Time and compare
    time_and_compare(q, k, v, kv_num_blocks, kv_indices, block_mask_types, seqlen_q, device)
    print("  ✓ Variable length test passed!\n")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Omni-Attention Correctness Tests")
    print("=" * 60)
    print()
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        return
    
    try:
        test_causal_attention()
        test_full_attention()
        test_hybrid_attention()
        test_custom_pattern()
        test_variable_lengths()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

