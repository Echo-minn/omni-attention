#!/usr/bin/env python3
"""
Omni-Attention Test Suite

Tests correctness of naive attention and flex_attention with modality padding.
"""

import argparse
from typing import Tuple

import torch
from torch import randn

# Import from omni_attn_torch
from omni_attn_torch import (
    OmniBlockMask,
    create_omni_block_mask_from_modality_positions,
    generate_random_input_with_modalities,
    get_embeddings_from_text_and_images,
    naive_attention,
    pad_modalities_to_block_size,
    print_block_structure,
    check_correctness,
)

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

    args = parser.parse_args()
    B, H, N, D = args.B, args.H, args.N, args.D
    BLOCK_SIZE = args.block_size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"B={B}, H={H}, N={N}, D={D}, BLOCK_SIZE={BLOCK_SIZE}, device={device}")
    
    # Generate random input with 2-5 modality segments
    text_and_images, modality_positions = generate_random_input_with_modalities(
        B=B, N=N, D=D, device=device, seed=args.seed,
    )
    
    # Get embeddings
    x = get_embeddings_from_text_and_images(text_and_images, D=D, device=device)
    original_seq_len = x.shape[1]
    
    # Print block structure BEFORE padding
    print("\nBEFORE padding (seq_len={}):".format(original_seq_len))
    print_block_structure(modality_positions, q_len=original_seq_len, BLOCK_SIZE=BLOCK_SIZE, batch_idx=0)
    
    # Pad each modality separately
    x_padded, modality_positions_padded = pad_modalities_to_block_size(x, modality_positions, BLOCK_SIZE=BLOCK_SIZE)
    padded_seq_len = x_padded.shape[1]
    
    # Print block structure AFTER padding
    print("\nAFTER padding (seq_len={}, added={}):".format(padded_seq_len, padded_seq_len - original_seq_len))
    print_block_structure(modality_positions_padded, q_len=padded_seq_len, BLOCK_SIZE=BLOCK_SIZE, batch_idx=0)
    
    # Create Q, K, V
    head_dim = D // H
    assert D % H == 0, f"Model dimension D ({D}) must be divisible by number of heads H ({H})"
    
    W_Q = randn(D, D, device=device)
    W_K = randn(D, D, device=device)
    W_V = randn(D, D, device=device)
    
    # Project and reshape to [B, H, N, head_dim]
    x_proj_q = torch.matmul(x_padded, W_Q)  # [B, N, D]
    x_proj_k = torch.matmul(x_padded, W_K)
    x_proj_v = torch.matmul(x_padded, W_V)
    
    Q = x_proj_q.view(B, padded_seq_len, H, head_dim).transpose(1, 2)  # [B, H, N, head_dim]
    K = x_proj_k.view(B, padded_seq_len, H, head_dim).transpose(1, 2)
    V = x_proj_v.view(B, padded_seq_len, H, head_dim).transpose(1, 2)
    
    # Create masks - use SAME mask logic for both naive and flex_attention
    if HAS_FLEX_ATTN:
        dense_mask = naive_attn_mask(padded_seq_len, modality_positions_padded, device=device)  # [B, N, N]
        dense_mask = dense_mask.unsqueeze(1).expand(B, H, padded_seq_len, padded_seq_len)  # [B, H, N, N]
        flex_block_mask = create_flex_block_mask(
            modality_positions_padded, B=B, H=H, Q_LEN=padded_seq_len, KV_LEN=padded_seq_len,
            device=device, compile_mask=False,
        )
    else:
        dense_mask = naive_attn_mask(padded_seq_len, modality_positions_padded, device=device)  # [B, N, N]
        dense_mask = dense_mask.unsqueeze(1).expand(B, H, padded_seq_len, padded_seq_len)  # [B, H, N, N]
        flex_block_mask = None
    
    # Run correctness tests
    print("\nCorrectness tests:")
    from omni_attn_torch import naive_attention_with_dense_mask
    score_mask = torch.where(dense_mask, 0.0, float('-inf')).to(torch.float32)
    naive_output = naive_attention_with_dense_mask(Q, K, V, mask=score_mask)
    
    if HAS_FLEX_ATTN and flex_block_mask is not None:
        _ = flex_attention(Q, K, V, block_mask=flex_block_mask)  # Warmup
        flex_output = flex_attention(Q, K, V, block_mask=flex_block_mask)
        check_correctness(naive_output, flex_output, rtol=1e-2, atol=1e-3, name="flex_attention")
    else:
        print("Skipping flex_attention (not available)")


if __name__ == "__main__":
    main()
