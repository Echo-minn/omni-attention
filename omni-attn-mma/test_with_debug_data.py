#!/usr/bin/env python3
"""
Test omni_attention kernels with fixed debug data.

Loads the fixed debug data and compares kernel output with reference.
"""

import torch
import sys
import os
import time
from typing import Optional
from omni_attn_torch import (
    omni_attention_shared_kv_swizzle,
    omni_attention_simple,
    omni_attention_cp_async,
    omni_attention_shared_kv,
    omni_attention_preftech,
    check_correctness,
)

# Try to import flex_attention
try:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from transfusion_pytorch.transfusion import flex_attention, create_block_mask
    HAS_FLEX_ATTN = True
    
    def causal(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx
except ImportError:
    HAS_FLEX_ATTN = False
    print("Warning: flex_attention not available")


def compute_dense_attn_tflops(Q: torch.Tensor, elapsed_sec: Optional[float]) -> Optional[float]:
    """Estimate TFLOPS assuming dense attention work."""
    if elapsed_sec is None or elapsed_sec <= 0:
        return None

    B, H, N, D = Q.shape

    flops_qk = B * H * N * N * (2 * D - 1)
    flops_softmax = (
        B * H * N * (N - 1)  # row max
        + B * H * N * N  # subtract max
        + B * H * N * N  # exp
        + B * H * N * (N - 1)  # row sum
        + B * H * N * N  # normalization
    )
    flops_scaling = B * H * N * N
    flops_pv = B * H * N * D * (2 * N - 1)

    total_flops = flops_qk + flops_scaling + flops_softmax + flops_pv
    return total_flops * 1e-12 / elapsed_sec


def log_time_and_tflops(tag: str, Q: torch.Tensor, elapsed_sec: Optional[float]) -> Optional[float]:
    """Print time and TFLOPS for a run."""
    if elapsed_sec is None:
        print(f"  Time: N/A for {tag}")
        return None

    tflops = compute_dense_attn_tflops(Q, elapsed_sec)
    if tflops is None:
        print(f"  Time: {elapsed_sec*1000:.2f}ms for {tag}")
    else:
        print(f"  Time: {elapsed_sec*1000:.2f}ms, TFLOPS: {tflops:.2f} for {tag}")
    return tflops


def test_flex_attention(Q, K, V, dense_mask, reference_output, q_block_size, kv_block_size, device="cuda", is_causal=False):
    """Test flex_attention with fixed debug data."""
    passed = False
    flex_output = None
    flex_time = None
    
    if not HAS_FLEX_ATTN:
        print("\nSkipping flex_attention (not available)")
        return passed, flex_output, flex_time
    
    if dense_mask is None and not is_causal:
        print("\nSkipping flex_attention (dense_mask not in debug data)")
        return passed, flex_output, flex_time
    
    print("\n" + "="*60)
    print("Testing flex_attention...")
    print("="*60)
    q_block_size = max(128, q_block_size)
    kv_block_size = max(128, kv_block_size)
    
    try:
        B, H, seq_len, head_dim = Q.shape
        
        if is_causal:
            flex_block_mask = create_block_mask(
                causal,
                B=None,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                BLOCK_SIZE=(q_block_size, kv_block_size),
                device=device,
                _compile=False,
            )
        else:
            dense_mask = dense_mask.to(device)
            def mask_mod(b, h, q_idx, kv_idx):
                return dense_mask[b, h, q_idx, kv_idx]
            
            flex_block_mask = create_block_mask(
                mask_mod,
                B=B,
                H=H,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                BLOCK_SIZE=(q_block_size, kv_block_size),
                device=device,
                _compile=False,
            )
        
        _ = flex_attention(Q, K, V, block_mask=flex_block_mask)
        torch.cuda.synchronize()
        
        start = time.time()
        flex_output = flex_attention(Q, K, V, block_mask=flex_block_mask)
        torch.cuda.synchronize()
        flex_time = time.time() - start
        
        passed = check_correctness(
            reference_output,
            flex_output,
            rtol=1e-1,
            atol=1e-1,
            name="flex_attention"
        )
        log_time_and_tflops("flex_attention", Q, flex_time)
    
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

    return passed, flex_output, flex_time

def test_omni_attention_simple(Q, K, V, omni_block_mask, reference_output):
    """Test omni_attention_simple with fixed debug data."""

    print("\n" + "="*60)
    print("Testing omni_attention_simple...")
    print("="*60)
    
    passed = False
    omni_output = None
    omni_time = None
    try:
        _ = omni_attention_simple(Q, K, V, omni_block_mask)
        torch.cuda.synchronize()
        
        start = time.time()
        omni_output = omni_attention_simple(Q, K, V, omni_block_mask)
        torch.cuda.synchronize()
        omni_time = time.time() - start
        
        passed = check_correctness(
            reference_output,
            omni_output,
            rtol=1e-1,
            atol=1e-1,
            name="omni_attention_simple"
        )
        log_time_and_tflops("omni_attention_simple", Q, omni_time)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
      
    return passed, omni_output, omni_time

def test_omni_attention_cp_async(Q, K, V, omni_block_mask, reference_output):
    """Test omni_attention_cp_async with fixed debug data."""
    print("\n" + "="*60)
    print("Testing omni_attention_cp_async...")
    print("="*60)
    
    omni_output = None
    omni_time = None
    try:
        _ = omni_attention_cp_async(Q, K, V, omni_block_mask)
        torch.cuda.synchronize()
        
        start = time.time()
        omni_output = omni_attention_cp_async(Q, K, V, omni_block_mask)
        torch.cuda.synchronize()
        omni_time = time.time() - start
        
        check_correctness(
            reference_output,
            omni_output,
            rtol=1e-1,
            atol=1e-1,
            name="omni_attention_cp_async"
        )
        log_time_and_tflops("omni_attention_cp_async", Q, omni_time)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
      
    return omni_output, omni_time

def test_omni_attention_shared_kv(Q, K, V, omni_block_mask, reference_output):
    """Test omni_attn_shared_kv with fixed debug data."""
    print("\n" + "="*60)
    print("Testing omni_attn_shared_kv...")
    print("="*60)
    
    passed = False
    omni_output = None
    omni_time = None
    try:
        _ = omni_attention_shared_kv(Q, K, V, omni_block_mask)
        torch.cuda.synchronize()
        
        start = time.time()
        omni_output = omni_attention_shared_kv(Q, K, V, omni_block_mask)
        torch.cuda.synchronize()
        omni_time = time.time() - start

        passed = check_correctness(
            reference_output,
            omni_output,
            rtol=1e-1,
            atol=1e-1,
            name="omni_attn_shared_kv"
        )
        log_time_and_tflops("omni_attn_shared_kv", Q, omni_time)
        
       
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
      
    return passed, omni_output, omni_time

def test_omni_attention_shared_kv_swizzle(Q, K, V, omni_block_mask, reference_output):

    print("\n" + "="*60)
    print("Testing omni_attention_shared_kv_swizzle...")
    print("="*60)
    
    passed = False
    omni_output = None
    omni_time = None

    try:
        _ = omni_attention_shared_kv_swizzle(Q, K, V, omni_block_mask)
        torch.cuda.synchronize()
        
        start = time.time()
        omni_output = omni_attention_shared_kv_swizzle(Q, K, V, omni_block_mask)
        torch.cuda.synchronize()
        omni_time = time.time() - start

        passed = check_correctness(
            reference_output,
            omni_output,
            rtol=1e-1,
            atol=1e-1,
            name="omni_attention_shared_kv_swizzle"
        )
        log_time_and_tflops("omni_attention_shared_kv_swizzle", Q, omni_time)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
      
    return passed, omni_output, omni_time

def test_omni_attention_preftech(Q, K, V, omni_block_mask, reference_output):
    """Test omni_attention_preftech with fixed debug data."""
    print("\n" + "="*60)
    print("Testing omni_attention_preftech...")
    print("="*60)
    
    passed = False
    omni_output = None
    omni_time = None
    try:
        _ = omni_attention_preftech(Q, K, V, omni_block_mask)
        torch.cuda.synchronize()
        
        start = time.time()
        omni_output = omni_attention_preftech(Q, K, V, omni_block_mask)
        torch.cuda.synchronize()
        omni_time = time.time() - start
        
        passed = check_correctness(
            reference_output,
            omni_output,
            rtol=1e-1,
            atol=1e-1,
            name="omni_attention_preftech"
        )
        log_time_and_tflops("omni_attention_preftech", Q, omni_time)
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
      
    return passed, omni_output, omni_time

def test_with_debug_data(debug_data_file="debug_data.pt", device="cuda"):
    """Test kernels with fixed debug data."""
    print(f"Loading debug data from {debug_data_file}...")
    # Use weights_only=False for custom objects (OmniBlockMask)
    data = torch.load(debug_data_file, map_location=device, weights_only=False)
    
    Q = data['Q'].to(device)
    K = data['K'].to(device)
    V = data['V'].to(device)
    reference_output = data['reference_output'].to(device)
    omni_block_mask = data['omni_block_mask']
    dense_mask = data.get('dense_mask', None)
    is_causal = 'causal_mask' in data and dense_mask is None
    metadata = data['metadata']
    sparsity = metadata['sparsity'] if 'sparsity' in metadata else 0
    pattern = metadata['pattern'] if 'pattern' in metadata else None
    q_block_size = omni_block_mask.Q_BLOCK_SIZE
    kv_block_size = omni_block_mask.KV_BLOCK_SIZE
    
    print(f"\nTesting with fixed data:")
    print(f"  {metadata}")
    print(f"  Reference output shape: {reference_output.shape}")
    
    simple_passed, _, simple_time = test_omni_attention_simple(Q, K, V, omni_block_mask, reference_output)
    # omni_output, omni_time = test_omni_attention_cp_async(Q, K, V, omni_block_mask, reference_output)
    prefetch_passed, _, preftech_time = test_omni_attention_preftech(Q, K, V, omni_block_mask, reference_output)
    shared_kv_passed, _, shared_kv_time = test_omni_attention_shared_kv(Q, K, V, omni_block_mask, reference_output)
    swizzle_passed, _, swizzle_time = test_omni_attention_shared_kv_swizzle(Q, K, V, omni_block_mask, reference_output)
    flex_passed, _, flex_time = test_flex_attention(Q, K, V, dense_mask, reference_output, q_block_size, kv_block_size, device, is_causal=is_causal)

    print("\n" + "="*60)
    print(f"Testing with data: {debug_data_file} ...")
    print(f"  Q shape: {Q.shape}; Sparsity: {sparsity:.4f}; Pattern: {pattern}")
    print("="*60)

    simple_tflops = compute_dense_attn_tflops(Q, simple_time)
    preftech_tflops = compute_dense_attn_tflops(Q, preftech_time)
    shared_kv_tflops = compute_dense_attn_tflops(Q, shared_kv_time)
    swizzle_tflops = compute_dense_attn_tflops(Q, swizzle_time)
    flex_tflops = compute_dense_attn_tflops(Q, flex_time)

    def format_tflops(val):
        return f", TFLOPS: {val:.2f}" if val is not None else ""

    if flex_time is not None:
        simple_speedup = flex_time / simple_time
        prefetch_speedup = flex_time / preftech_time
        shared_kv_speedup = flex_time / shared_kv_time
        swizzle_speedup = flex_time / swizzle_time
        print(f"  Simple {'PASSED✅' if simple_passed else 'FAILED❌'}, time: {simple_time*1000:.2f}ms{format_tflops(simple_tflops)}, speedup: {simple_speedup:.2f}x")
        print(f"  Prefetch {'PASSED✅' if prefetch_passed else 'FAILED❌'}, time: {preftech_time*1000:.2f}ms{format_tflops(preftech_tflops)}, speedup: {prefetch_speedup:.2f}x")
        print(f"  Shared_kv {'PASSED✅' if shared_kv_passed else 'FAILED❌'}, time: {shared_kv_time*1000:.2f}ms{format_tflops(shared_kv_tflops)}, speedup: {shared_kv_speedup:.2f}x")
        print(f"  Swizzle {'PASSED✅' if swizzle_passed else 'FAILED❌'}, time: {swizzle_time*1000:.2f}ms{format_tflops(swizzle_tflops)}, speedup: {swizzle_speedup:.2f}x")
        print(f"  Flex {'PASSED✅' if flex_passed else 'FAILED❌'}, time: {flex_time*1000:.2f}ms{format_tflops(flex_tflops)}")
    else:
        print(f"  Simple {'PASSED✅' if simple_passed else 'FAILED❌'}, time: {simple_time*1000:.2f}ms{format_tflops(simple_tflops)}")
        print(f"  Prefetch {'PASSED✅' if prefetch_passed else 'FAILED❌'}, time: {preftech_time*1000:.2f}ms{format_tflops(preftech_tflops)}")
        print(f"  Shared_kv {'PASSED✅' if shared_kv_passed else 'FAILED❌'}, time: {shared_kv_time*1000:.2f}ms{format_tflops(shared_kv_tflops)}")
        print(f"  Swizzle {'PASSED✅' if swizzle_passed else 'FAILED❌'}, time: {swizzle_time*1000:.2f}ms{format_tflops(swizzle_tflops)}")
        print(f"  Flex: Skipped")
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test with fixed debug data')
    parser.add_argument('--input', type=str, default='debug_data.pt', help='Debug data file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    test_with_debug_data(debug_data_file=args.input, device=args.device)

