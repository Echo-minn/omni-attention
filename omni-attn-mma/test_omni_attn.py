#!/usr/bin/env python3
"""
Omni-Attention Test Suite

Tests correctness of naive attention and flex_attention with modality padding.
"""

import argparse
from typing import Tuple, Union

import torch
from torch import randn

# Import from omni_attn_torch
from omni_attn_torch import (
    OmniBlockMask,
    create_causal_omni_block_mask,
    create_omni_block_mask_from_modality_positions,
    generate_causal_input,
    generate_random_input_with_modalities,
    get_embeddings_from_text_and_images,
    naive_attention,
    pad_modalities_to_block_size,
    print_block_structure,
    check_correctness,
    omni_attention_shared_kv,
    omni_block_mask_from_flex_block_mask,
)

# Import test input utilities
from test_inputs import load_test_input, TestInput

# Import from transfusion for flex_attention
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
try:
    from transfusion_pytorch.transfusion import flex_attention, create_block_mask, transfusion_attn_mask, naive_attn_mask, exists
    HAS_FLEX_ATTN = True
except ImportError:
    HAS_FLEX_ATTN = False
    print("Warning: flex_attention not available")

def create_flex_block_mask(
    modality_positions: torch.Tensor,
    B: int,
    H: int,
    Q_LEN: int,
    KV_LEN: int,
    device: str = "cuda",
    compile_mask: bool = False,  # Disable compilation by default
) -> object:
    """Create block mask for flex_attention using transfusion logic."""
    if not HAS_FLEX_ATTN:
        return None
    
    mask_fn = transfusion_attn_mask(modality_positions)
    
    block_mask = create_block_mask(
        mask_fn,
        B=B,
        H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        _compile=compile_mask,  # Disable compilation to reduce overhead
        device=device
    )
    
    return block_mask


def compare_masks(
    mask1: torch.Tensor,
    mask2: torch.Tensor,
    name1: str = "mask1",
    name2: str = "mask2",
) -> bool:
    """Compare two boolean masks and print differences."""
    if mask1.shape != mask2.shape:
        print(f"  ✗ Shape mismatch: {mask1.shape} vs {mask2.shape}")
        return False
    
    diff = (mask1 != mask2).sum().item()
    total = mask1.numel()
    match_pct = 100.0 * (1.0 - diff / total)
    
    if diff == 0:
        print(f"  ✓ Masks match perfectly ({match_pct:.2f}%)")
        return True
    else:
        print(f"  ✗ Masks differ: {diff}/{total} positions ({match_pct:.2f}% match)")
        # Print some example differences
        diff_positions = torch.nonzero(mask1 != mask2, as_tuple=False)
        if len(diff_positions) > 0:
            print(f"    First 5 differences:")
            for i in range(min(5, len(diff_positions))):
                pos = diff_positions[i]
                print(f"      Position {pos.tolist()}: {name1}={mask1[tuple(pos)].item()}, {name2}={mask2[tuple(pos)].item()}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Omni-Attention Test Suite')
    parser.add_argument('--B', type=int, default=1, help='Batch size')
    parser.add_argument('--H', type=int, default=8, help='Number of heads')
    parser.add_argument('--N', type=int, default=512, help='Target sequence length')
    parser.add_argument('--D', type=int, default=512, help='Model dimension (H * head_dim)')
    parser.add_argument('--block_size', type=int, default=128, help='Block size (default: 128)')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--num_segments', type=int, default=None, help='Number of modality segments (2-5)')
    parser.add_argument('--use_saved_input', type=int, default=None, 
                        help='Use saved test input by index (0-2). Requires test_inputs.pkl file.')
    parser.add_argument('--input_file', type=str, default='test_inputs.pkl',
                        help='Path to saved test inputs file (default: test_inputs.pkl)')

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load from saved input or generate new
    if args.use_saved_input is not None:
        print(f"Loading saved test input {args.use_saved_input} from {args.input_file}...")
        test_input = load_test_input(args.input_file, index=args.use_saved_input)
        
        B = test_input.B
        H = test_input.H
        D = test_input.D
        BLOCK_SIZE = test_input.BLOCK_SIZE
        seq_len = test_input.seq_len
        
        text_and_images = test_input.text_and_images
        modality_positions = test_input.modality_positions
        Q = test_input.Q
        K = test_input.K
        V = test_input.V
        
        print(f"Loaded: B={B}, H={H}, seq_len={seq_len}, D={D}, BLOCK_SIZE={BLOCK_SIZE}, seed={test_input.seed}")
    else:
        B, H, N, D = args.B, args.H, args.N, args.D
        BLOCK_SIZE = args.block_size
        
        print(f"B={B}, H={H}, N={N}, D={D}, BLOCK_SIZE={BLOCK_SIZE}, device={device}")
        
        # Generate random input with 2-5 modality segments
        text_and_images, modality_positions = generate_random_input_with_modalities(
            B=B, N=N, D=D, device=device,
        )
        
        # Get embeddings
        x = get_embeddings_from_text_and_images(text_and_images, D=D, device=device)
        seq_len = x.shape[1]
        
        # Create Q, K, V
        head_dim = D // H
        assert D % H == 0, f"Model dimension D ({D}) must be divisible by number of heads H ({H})"
        
        W_Q = randn(D, D, device=device)
        W_K = randn(D, D, device=device)
        W_V = randn(D, D, device=device)
        
        # Project and reshape to [B, H, N, head_dim]
        x_proj_q = torch.matmul(x, W_Q)  # [B, N, D]
        x_proj_k = torch.matmul(x, W_K)
        x_proj_v = torch.matmul(x, W_V)
        
        Q = x_proj_q.view(B, seq_len, H, head_dim).transpose(1, 2)  # [B, H, N, head_dim]
        K = x_proj_k.view(B, seq_len, H, head_dim).transpose(1, 2)
        V = x_proj_v.view(B, seq_len, H, head_dim).transpose(1, 2)
    
    # Print block structure BEFORE padding
    print("\nBEFORE padding (seq_len={}):".format(seq_len))
    print_block_structure(modality_positions, q_len=seq_len, BLOCK_SIZE=BLOCK_SIZE, batch_idx=0)
    
    # Create masks - use SAME mask logic for both naive and flex_attention
    if HAS_FLEX_ATTN:
        dense_mask = naive_attn_mask(seq_len, modality_positions, device=device)  # [B, N, N]
        dense_mask = dense_mask.unsqueeze(1).expand(B, H, seq_len, seq_len)  # [B, H, N, N]
        flex_block_mask = create_flex_block_mask(
            modality_positions, B=B, H=H, Q_LEN=seq_len, KV_LEN=seq_len,
            device=device, compile_mask=False,
        )
    else:
        dense_mask = naive_attn_mask(seq_len, modality_positions, device=device)  # [B, N, N]
        dense_mask = dense_mask.unsqueeze(1).expand(B, H, seq_len, seq_len)  # [B, H, N, N]
        flex_block_mask = None
    
    # Run correctness tests
    print("\nCorrectness tests:")
    from omni_attn_torch import naive_attention_with_dense_mask
    score_mask = torch.where(dense_mask, 0.0, float('-inf')).to(torch.float32)
    naive_output = naive_attention_with_dense_mask(Q, K, V, mask=score_mask)
    
    if HAS_FLEX_ATTN and flex_block_mask is not None:
        print(f"flex_block_mask: {flex_block_mask.shape}")
        
        _ = flex_attention(Q, K, V, block_mask=flex_block_mask)  # Warmup
        flex_output = flex_attention(Q, K, V, block_mask=flex_block_mask)
        check_correctness(naive_output, flex_output, rtol=1e-2, atol=1e-3, name="flex_attention")
    else:
        print("Skipping flex_attention (not available)")
    
    # Test CUDA kernels
    print("\nTesting CUDA kernels...")
    try:
        # FIXME
        omni_block_mask = create_omni_block_mask_from_modality_positions(
            modality_positions=modality_positions,
            batch=B,
            nheads=H,
            q_len=seq_len,
            kv_len=seq_len,
            BLOCK_SIZE=BLOCK_SIZE,
            device=device,
        )
        
        print(f"  Created OmniBlockMask: {omni_block_mask.shape}, sparsity={omni_block_mask.sparsity():.2f}%")
        
        # Ensure Q, K, V are contiguous (the wrapper handles dtype conversion to half)
        Q_cuda = Q.contiguous()
        K_cuda = K.contiguous()
        V_cuda = V.contiguous()
        
        # Reference output in half precision for comparison
        naive_output_fp16 = naive_output.half()
        
        # Track kernel times for performance comparison
        kernel_times = {}
        
        # Test 1: Simple kernel (correctness baseline)
        print("\n  Testing omni_attention_simple (correctness baseline)...")
        try:
            from omni_attn_torch import omni_attention_simple
            
            # Create fresh copies of input tensors to avoid memory corruption
            Q_simple = Q_cuda.clone()
            K_simple = K_cuda.clone()
            V_simple = V_cuda.clone()
            
            # Warmup
            _ = omni_attention_simple(Q_simple, K_simple, V_simple, omni_block_mask)
            torch.cuda.synchronize()  # This will raise if there's a CUDA error
            
            # Time the kernel
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            
            simple_output = omni_attention_simple(Q_simple, K_simple, V_simple, omni_block_mask)
            
            end.record()
            torch.cuda.synchronize()  # This will raise if there's a CUDA error
            
            simple_time = start.elapsed_time(end)
            kernel_times['simple'] = simple_time
            
            # Check correctness
            check_correctness(
                naive_output_fp16, 
                simple_output, 
                rtol=1e-1,  # More relaxed tolerance for half precision
                atol=1e-2, 
                name="omni_attention_simple"
            )
            print(f"    ✓ Simple kernel time: {simple_time:.2f} ms")
            
        except ImportError as e:
            print(f"    ✗ Simple kernel not available: {e}")
        except Exception as e:
            print(f"    ✗ Error running simple kernel: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 2: cp.async kernel
        # print("\n  Testing omni_attention_cp_async (cp.async optimized)...")
        # try:
        #     from omni_attn_torch import omni_attention_cp_async
            
        #     # Create fresh copies of input tensors to avoid memory corruption
        #     Q_cp = Q_cuda.clone()
        #     K_cp = K_cuda.clone()
        #     V_cp = V_cuda.clone()
            
        #     # Warmup
        #     _ = omni_attention_cp_async(Q_cp, K_cp, V_cp, omni_block_mask)
        #     torch.cuda.synchronize()  # This will raise if there's a CUDA error
            
        #     # Time the kernel
        #     torch.cuda.synchronize()
        #     start = torch.cuda.Event(enable_timing=True)
        #     end = torch.cuda.Event(enable_timing=True)
        #     start.record()
            
        #     cp_async_output = omni_attention_cp_async(Q_cp, K_cp, V_cp, omni_block_mask)
            
        #     end.record()
        #     torch.cuda.synchronize()  # This will raise if there's a CUDA error
            
        #     cp_async_time = start.elapsed_time(end)
        #     kernel_times['cp_async'] = cp_async_time
            
        #     # Check correctness
        #     check_correctness(
        #         naive_output_fp16, 
        #         cp_async_output, 
        #         rtol=1e-1,  # More relaxed tolerance for half precision
        #         atol=1e-2, 
        #         name="omni_attention_cp_async"
        #     )
        #     print(f"    ✓ cp.async kernel time: {cp_async_time:.2f} ms")
            
        #     # Compare performance if both kernels ran
        #     if 'simple' in kernel_times:
        #         speedup = kernel_times['simple'] / cp_async_time if cp_async_time > 0 else 0
        #         print(f"    Speedup vs simple: {speedup:.2f}x")
            
        # except ImportError as e:
        #     print(f"    ✗ cp.async kernel not available: {e}")
        # except Exception as e:
        #     print(f"    ✗ Error running cp.async kernel: {e}")
        #     import traceback
        #     traceback.print_exc()
        
        # Test 3: MMA kernel (for comparison)
        print("\n  Testing omni_attention_shared_kv (MMA optimized)...")
        try:
            from omni_attn_torch import omni_attention_shared_kv
            
            # Create fresh copies of input tensors to avoid memory corruption
            Q_mma = Q_cuda.clone()
            K_mma = K_cuda.clone()
            V_mma = V_cuda.clone()
            
            # Warmup
            _ = omni_attention_shared_kv(Q_mma, K_mma, V_mma, omni_block_mask, stages=2)
            torch.cuda.synchronize()  # This will raise if there's a CUDA error
            
            # Time the kernel
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            
            mma_output = omni_attention_shared_kv(Q_mma, K_mma, V_mma, omni_block_mask, stages=2)
            
            end.record()
            torch.cuda.synchronize()  # This will raise if there's a CUDA error
            
            mma_time = start.elapsed_time(end)
            kernel_times['mma'] = mma_time
            
            # Check correctness
            check_correctness(
                naive_output_fp16, 
                mma_output, 
                rtol=1e-1,  # More relaxed tolerance for half precision
                atol=1e-2, 
                name="omni_attention_shared_kv"
            )
            print(f"    ✓ MMA kernel time: {mma_time:.2f} ms")
            
            # Compare performance if other kernels ran
            if 'simple' in kernel_times:
                speedup = kernel_times['simple'] / mma_time if mma_time > 0 else 0
                print(f"    Speedup vs simple: {speedup:.2f}x")
            if 'cp_async' in kernel_times:
                speedup = kernel_times['cp_async'] / mma_time if mma_time > 0 else 0
                print(f"    Speedup vs cp.async: {speedup:.2f}x")
            
        except ImportError as e:
            print(f"    ✗ MMA kernel not available: {e}")
        except Exception as e:
            print(f"    ✗ Error running MMA kernel: {e}")
            import traceback
            traceback.print_exc()
        
    except ImportError as e:
        print(f"  ✗ CUDA kernels not available: {e}")
        print("  To build the kernels, run: cd omni-attn-mma && python setup.py build_ext --inplace")
    except Exception as e:
        print(f"  ✗ Error running CUDA kernels: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
