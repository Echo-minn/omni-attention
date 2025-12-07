#!/usr/bin/env python3
"""
Test omni_attention kernels with fixed debug data.

Loads the fixed debug data and compares kernel output with reference.
"""

import torch
import sys
import os
import time
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
except ImportError:
    HAS_FLEX_ATTN = False
    print("Warning: flex_attention not available")

def test_flex_attention(Q, K, V, dense_mask, reference_output, q_block_size, kv_block_size, device="cuda"):
    """Test flex_attention with fixed debug data."""
    if HAS_FLEX_ATTN and dense_mask is not None:
        print("\n" + "="*60)
        print("Testing flex_attention...")
        print("="*60)
        
        try:
            B, H, seq_len, head_dim = Q.shape
            dense_mask = dense_mask.to(device)
            
            # Create mask_mod function from dense_mask
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
                atol=1e-2,
                name="flex_attention"
            )
            print(f"  Time: {flex_time*1000:.2f}ms")
        
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    elif not HAS_FLEX_ATTN:
        print("\nSkipping flex_attention (not available)")
    elif dense_mask is None:
        print("\nSkipping flex_attention (dense_mask not in debug data)")

    return passed, flex_output, flex_time

def test_omni_attention_simple(Q, K, V, omni_block_mask, reference_output):
    """Test omni_attention_simple with fixed debug data."""

    print("\n" + "="*60)
    print("Testing omni_attention_simple...")
    print("="*60)
    
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
            atol=1e-2,
            name="omni_attention_simple"
        )
        print(f"  Time: {omni_time*1000:.2f}ms")
        
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
            atol=1e-2,
            name="omni_attention_cp_async"
        )
        print(f"  Time: {omni_time*1000:.2f}ms")
        
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
            atol=1e-2,
            name="omni_attn_shared_kv"
        )
        print(f"  Time: {omni_time*1000:.2f}ms")
        
       
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
      
    return passed, omni_output, omni_time

def test_omni_attention_shared_kv_swizzle(Q, K, V, omni_block_mask, reference_output):

    print("\n" + "="*60)
    print("Testing omni_attention_shared_kv_swizzle...")
    print("="*60)
    
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
            atol=1e-2,
            name="omni_attention_shared_kv_swizzle"
        )
        print(f"  Time: {omni_time*1000:.2f}ms")
        
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
            atol=1e-2,
            name="omni_attention_preftech"
        )
        print(f"  Time: {omni_time*1000:.2f}ms")
        
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
    metadata = data['metadata']
    q_block_size = omni_block_mask.Q_BLOCK_SIZE
    kv_block_size = omni_block_mask.KV_BLOCK_SIZE
    
    print(f"\nTesting with fixed data:")
    print(f"  {metadata}")
    print(f"  Q shape: {Q.shape}, dtype: {Q.dtype}; Q_BLOCK_SIZE: {q_block_size}, KV_BLOCK_SIZE: {kv_block_size}")
    print(f"  Reference output shape: {reference_output.shape}")
    
    simple_passed, _, simple_time = test_omni_attention_simple(Q, K, V, omni_block_mask, reference_output)
    # omni_output, omni_time = test_omni_attention_cp_async(Q, K, V, omni_block_mask, reference_output)
    prefetch_passed, _, preftech_time = test_omni_attention_preftech(Q, K, V, omni_block_mask, reference_output)
    shared_kv_passed, _, shared_kv_time = test_omni_attention_shared_kv(Q, K, V, omni_block_mask, reference_output)
    swizzle_passed, _, swizzle_time = test_omni_attention_shared_kv_swizzle(Q, K, V, omni_block_mask, reference_output)
    flex_passed, _, flex_time = test_flex_attention(Q, K, V, dense_mask, reference_output, q_block_size, kv_block_size, device)


    simple_speedup = flex_time / simple_time
    prefetch_speedup = flex_time / preftech_time
    shared_kv_speedup = flex_time / shared_kv_time
    swizzle_speedup = flex_time / swizzle_time

    print("\n" + "="*40)
    print(f"Testing with data: {debug_data_file} ...")
    print("="*40)

    print(f"  Simple {'PASSED✅' if simple_passed else 'FAILED❌'}, time: {simple_time*1000:.2f}ms, speedup: {simple_speedup:.2f}x")
    print(f"  Prefetch {'PASSED✅' if prefetch_passed else 'FAILED❌'}, time: {preftech_time*1000:.2f}ms, speedup: {prefetch_speedup:.2f}x")
    print(f"  Shared_kv {'PASSED✅' if shared_kv_passed else 'FAILED❌'}, time: {shared_kv_time*1000:.2f}ms, speedup: {shared_kv_speedup:.2f}x")
    print(f"  Swizzle {'PASSED✅' if swizzle_passed else 'FAILED❌'}, time: {swizzle_time*1000:.2f}ms, speedup: {swizzle_speedup:.2f}x")
    print(f"  Flex {'PASSED✅' if flex_passed else 'FAILED❌'}, time: {flex_time*1000:.2f}ms")
        
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test with fixed debug data')
    parser.add_argument('--input', type=str, default='debug_data.pt', help='Debug data file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    test_with_debug_data(debug_data_file=args.input, device=args.device)

