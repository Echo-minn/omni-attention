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
    omni_attention_simple,
    omni_attention_cp_async,
    omni_attention_mma,
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

def test_flex_attention(Q, K, V, dense_mask, reference_output, device="cuda"):
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
                device=device,
                _compile=False,
            )
            
            _ = flex_attention(Q, K, V, block_mask=flex_block_mask)
            torch.cuda.synchronize()
            
            start = time.time()
            flex_output = flex_attention(Q, K, V, block_mask=flex_block_mask)
            torch.cuda.synchronize()
            flex_time = time.time() - start
            
            check_correctness(
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

    return flex_output, flex_time

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
        
        check_correctness(
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
      
    return omni_output, omni_time

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
        _ = omni_attention_mma(Q, K, V, omni_block_mask)
        torch.cuda.synchronize()
        
        start = time.time()
        omni_output = omni_attention_mma(Q, K, V, omni_block_mask)
        torch.cuda.synchronize()
        omni_time = time.time() - start

        passed =check_correctness(
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
      
    return omni_output, omni_time


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
        
        check_correctness(
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
      
    return omni_output, omni_time

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
    
    print(f"\nTesting with fixed data:")
    print(f"  {metadata}")
    print(f"  Q shape: {Q.shape}, dtype: {Q.dtype}")
    print(f"  Reference output shape: {reference_output.shape}")
    
    flex_output, flex_time = test_flex_attention(Q, K, V, dense_mask, reference_output, device)
    # omni_output, omni_time = test_omni_attention_simple(Q, K, V, omni_block_mask, reference_output)
    # omni_output, omni_time = test_omni_attention_cp_async(Q, K, V, omni_block_mask, reference_output)
    # omni_output, omni_time = test_omni_attention_shared_kv(Q, K, V, omni_block_mask, reference_output)
    omni_output, omni_time = test_omni_attention_preftech(Q, K, V, omni_block_mask, reference_output)

    # Compare omni vs flex
    if omni_output is not None:
        print("\n" + "="*60)
        print("Comparing omni_attention_simple vs flex_attention...")
        print("="*60)
        
        check_correctness(
            omni_output,
            flex_output,
            rtol=1e-1,
            atol=1e-2,
            name="omni vs flex"
        )
        
        if omni_time and flex_time:
            speedup = omni_time / flex_time
            faster = "omni" if speedup < 1 else "flex"
            print(f"  Speedup: {speedup:.2f}x ({faster} faster)")
            print(f"    omni: {omni_time*1000:.2f}ms")
            print(f"    flex: {flex_time*1000:.2f}ms")
        

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test with fixed debug data')
    parser.add_argument('--input', type=str, default='debug_data.pt', help='Debug data file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    test_with_debug_data(debug_data_file=args.input, device=args.device)

