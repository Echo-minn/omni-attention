#!/usr/bin/env python3
"""
Generate fixed debug data for omni_attention debugging.

Creates:
1. Fixed Q, K, V tensors (with fixed seed for reproducibility)
2. Causal mask
3. Reference output from naive attention
4. Saves everything to a file for debugging
"""

import torch
import numpy as np
import os
from omni_attn_torch import (
    create_causal_omni_block_mask,
    naive_attention_with_dense_mask,
    naive_attention,
    omni_attention_simple,
    OmniBlockMask,
    BlockMaskType,
    build_partial_block_data,
)

def generate_fixed_debug_causal_data(
    B=1,
    H=8,
    seq_len=512,
    head_dim=64,
    BLOCK_SIZE=128,
    device="cuda",
    output_file="debug_data.pt",
    seed=42
):
    """Generate fixed debug data for causal attention.
    
    Args:
        B: Batch size
        H: Number of heads
        seq_len: Sequence length
        head_dim: Head dimension
        BLOCK_SIZE: Block size for mask
        device: Device
        output_file: Output file path
        seed: Random seed for reproducibility
        
    Returns:
        dict with keys: Q, K, V, mask, reference_output, block_mask, metadata
    """
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Generating fixed debug data:")
    print(f"  B={B}, H={H}, seq_len={seq_len}, head_dim={head_dim}, BLOCK_SIZE={BLOCK_SIZE}")
    print(f"  seed={seed}, device={device}")
    
    # Generate fixed Q, K, V tensors
    # Use a simple pattern that's easy to verify
    Q = torch.randn(B, H, seq_len, head_dim, device=device, dtype=torch.float32)
    K = torch.randn(B, H, seq_len, head_dim, device=device, dtype=torch.float32)
    V = torch.randn(B, H, seq_len, head_dim, device=device, dtype=torch.float32)
    
    # Create causal mask (dense)
    # Causal: q_idx >= kv_idx
    causal_mask = torch.zeros(B, H, seq_len, seq_len, device=device, dtype=torch.bool)
    for q_idx in range(seq_len):
        for kv_idx in range(seq_len):
            if q_idx >= kv_idx:
                causal_mask[:, :, q_idx, kv_idx] = True
    
    # Create score mask (0 for attend, -inf for masked)
    score_mask = torch.where(causal_mask, 0.0, float('-inf')).to(torch.float32)

    # Create modality positions
    # Format: [B, M, 3] where M is number of modalities per batch
    # For causal (all text): [B, 1, 3] with (modality_type=1, offset=0, length=seq_len)
    modality_positions = torch.zeros(B, 1, 3, dtype=torch.int32, device=device)
    modality_positions[:, 0, :] = torch.tensor([1, 0, seq_len], dtype=torch.int32, device=device)
    
    # Create OmniBlockMask for CUDA kernels
    omni_block_mask = create_causal_omni_block_mask(
        batch=B,
        nheads=H,
        q_len=seq_len,
        kv_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
        device=device,
    )
    
    # Compute reference output using naive attention
    print("\nComputing reference output with naive attention...")
    reference_output = naive_attention_with_dense_mask(Q, K, V, mask=score_mask)
    
    # Convert to half precision for CUDA kernels
    Q_half = Q.half()
    K_half = K.half()
    V_half = V.half()
    reference_output_half = reference_output.half()
    
    # Prepare data dict
    debug_data = {
        'Q': Q_half,  # Half precision for CUDA
        'K': K_half,
        'V': V_half,
        'Q_fp32': Q,  # Full precision for reference
        'K_fp32': K,
        'V_fp32': V,
        'causal_mask': causal_mask,
        'score_mask': score_mask,
        'reference_output': reference_output_half,
        'reference_output_fp32': reference_output,
        'omni_block_mask': omni_block_mask,
        'modality_positions': modality_positions,
        'metadata': {
            'B': B,
            'H': H,
            'seq_len': seq_len,
            'head_dim': head_dim,
            'BLOCK_SIZE': BLOCK_SIZE,
            'seed': seed,
            'device': str(device),
        }
    }
    
    # Save to file
    print(f"\nSaving to {output_file}...")
    # Use weights_only=False for custom objects (OmniBlockMask)
    torch.save(debug_data, output_file, _use_new_zipfile_serialization=False)
    
    # Print summary
    print("\nDebug data summary:")
    print(f"  Q shape: {Q_half.shape}, dtype: {Q_half.dtype}")
    print(f"  K shape: {K_half.shape}, dtype: {K_half.dtype}")
    print(f"  V shape: {V_half.shape}, dtype: {V_half.dtype}")
    print(f"  Reference output shape: {reference_output_half.shape}, dtype: {reference_output_half.dtype}")
    print(f"  Block mask: {omni_block_mask.q_len}x{omni_block_mask.kv_len}, sparsity: {omni_block_mask.sparsity():.2f}%")
    
    # Print some sample values for verification
    print("\nSample values (batch=0, head=0):")
    print(f"  Q[0,0,0,:3] = {Q_half[0,0,0,:3].cpu().tolist()}")
    print(f"  K[0,0,0,:3] = {K_half[0,0,0,:3].cpu().tolist()}")
    print(f"  V[0,0,0,:3] = {V_half[0,0,0,:3].cpu().tolist()}")
    print(f"  Reference[0,0,0,:3] = {reference_output_half[0,0,0,:3].cpu().tolist()}")
    
    print(f"\n✓ Debug data saved to {output_file}")
    print(f"  File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    return debug_data

def generate_fixed_debug_omni_data(
    B=1,
    H=8,
    seq_len=512,
    head_dim=64,
    BLOCK_SIZE=128,
    device="cuda",
    output_file="debug_data_omni.pt",
    seed=42
):
    """Generate fixed debug data with random FULL/CAUSAL block pattern.
    
    Each q_block attends to all previous kv_blocks (0 to q_block),
    with each block randomly assigned as FULL or CAUSAL.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Generating random FULL/CAUSAL debug data:")
    print(f"  B={B}, H={H}, seq_len={seq_len}, head_dim={head_dim}, BLOCK_SIZE={BLOCK_SIZE}")
    
    Q = torch.randn(B, H, seq_len, head_dim, device=device, dtype=torch.float32)
    K = torch.randn(B, H, seq_len, head_dim, device=device, dtype=torch.float32)
    V = torch.randn(B, H, seq_len, head_dim, device=device, dtype=torch.float32)
    
    # Create OmniBlockMask with random pattern
    q_len_padded = ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    kv_len_padded = q_len_padded
    num_q_blocks = q_len_padded // BLOCK_SIZE
    num_kv_blocks = kv_len_padded // BLOCK_SIZE
    max_blocks = num_kv_blocks
    
    kv_num_blocks = torch.zeros(B, H, num_q_blocks, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(B, H, num_q_blocks, max_blocks, dtype=torch.int32, device=device)
    block_mask_types = torch.zeros(B, H, num_q_blocks, max_blocks, dtype=torch.int32, device=device)
    
    # Generate random pattern: for each q_block, randomly assign FULL/CAUSAL to each kv_block
    for q_block in range(num_q_blocks):
        num_active = 0
        for kv_block in range(q_block + 1):
            # Randomly choose FULL or CAUSAL (50/50)
            mask_type = BlockMaskType.FULL if torch.rand(1).item() < 0.5 else BlockMaskType.CAUSAL
            kv_indices[:, :, q_block, num_active] = kv_block
            block_mask_types[:, :, q_block, num_active] = mask_type
            num_active += 1
        kv_num_blocks[:, :, q_block] = num_active
        block_mask_types[:, :, q_block, num_active:] = BlockMaskType.MASKED
    
    omni_block_mask = OmniBlockMask(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        block_mask_types=block_mask_types,
        q_len=seq_len,
        kv_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Create dense mask from block pattern
    dense_mask = omni_block_mask.to_dense_mask()
    score_mask = torch.where(dense_mask, 0.0, float('-inf')).to(torch.float32)
    
    print("\nComputing reference output...")
    reference_output = naive_attention_with_dense_mask(Q, K, V, mask=score_mask)
    
    Q_half = Q.half()
    K_half = K.half()
    V_half = V.half()
    reference_output_half = reference_output.half()
    
    debug_data = {
        'Q': Q_half,
        'K': K_half,
        'V': V_half,
        'Q_fp32': Q,
        'K_fp32': K,
        'V_fp32': V,
        'dense_mask': dense_mask,
        'score_mask': score_mask,
        'reference_output': reference_output_half,
        'reference_output_fp32': reference_output,
        'omni_block_mask': omni_block_mask,
        'metadata': {
            'B': B, 'H': H, 'seq_len': seq_len, 'head_dim': head_dim,
            'BLOCK_SIZE': BLOCK_SIZE, 'seed': seed, 'device': str(device),
            'pattern': 'random_full_causal',
        }
    }
    
    print(f"\nSaving to {output_file}...")
    torch.save(debug_data, output_file, _use_new_zipfile_serialization=False)
    
    print(f"\n✓ Saved. Random block mask pattern (head=0):")
    for q_block in range(min(4, num_q_blocks)):
        num_active = kv_num_blocks[0, 0, q_block].item()
        pattern = []
        for i in range(num_active):
            mask_type = block_mask_types[0, 0, q_block, i].item()
            pattern.append(f"{'F' if mask_type == BlockMaskType.FULL else 'C'}")
        print(f"  q_block={q_block}: {' '.join(pattern)}")
    
    return debug_data

def generate_fixed_debug_partial_data(
    B=1,
    H=8,
    seq_len=512,
    head_dim=64,
    BLOCK_SIZE=128,
    device="cuda",
    output_file="debug_data_partial.pt",
    seed=42
):
    """Generate fixed debug data with PARTIAL block pattern."""
    """Generate fixed debug data with random FULL/CAUSAL block pattern.
    
    Each q_block attends to all previous kv_blocks (0 to q_block),
    with each block randomly assigned as FULL or CAUSAL.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    print(f"Generating random FULL/CAUSAL debug data:")
    print(f"  B={B}, H={H}, seq_len={seq_len}, head_dim={head_dim}, BLOCK_SIZE={BLOCK_SIZE}")
    
    Q = torch.randn(B, H, seq_len, head_dim, device=device, dtype=torch.float32)
    K = torch.randn(B, H, seq_len, head_dim, device=device, dtype=torch.float32)
    V = torch.randn(B, H, seq_len, head_dim, device=device, dtype=torch.float32)
    
    # Create OmniBlockMask with random pattern
    q_len_padded = ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    kv_len_padded = q_len_padded
    num_q_blocks = q_len_padded // BLOCK_SIZE
    num_kv_blocks = kv_len_padded // BLOCK_SIZE
    max_blocks = num_kv_blocks
    
    kv_num_blocks = torch.zeros(B, H, num_q_blocks, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(B, H, num_q_blocks, max_blocks, dtype=torch.int32, device=device)
    block_mask_types = torch.zeros(B, H, num_q_blocks, max_blocks, dtype=torch.int32, device=device)
    
    # Generate random pattern: for each q_block, randomly assign FULL/CAUSAL/PARTIAL to each kv_block
    for q_block in range(num_q_blocks):
        num_active = 0
        for kv_block in range(q_block + 1):
            # Randomly choose FULL, CAUSAL, or PARTIAL (1/3 each)
            rand_val = torch.rand(1).item()
            if rand_val < 0.333:
                mask_type = BlockMaskType.FULL
            elif rand_val < 0.666:
                mask_type = BlockMaskType.CAUSAL
            else:
                mask_type = BlockMaskType.PARTIAL
            kv_indices[:, :, q_block, num_active] = kv_block
            block_mask_types[:, :, q_block, num_active] = mask_type
            num_active += 1
        kv_num_blocks[:, :, q_block] = num_active
        block_mask_types[:, :, q_block, num_active:] = BlockMaskType.MASKED
    
    omni_block_mask = OmniBlockMask(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        block_mask_types=block_mask_types,
        q_len=seq_len,
        kv_len=seq_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Create dense mask from block pattern
    # For PARTIAL blocks, we need to manually create half causal + half full pattern
    dense_mask = torch.zeros(B, H, seq_len, seq_len, device=device, dtype=torch.bool)
    
    for q_block in range(num_q_blocks):
        q_start = q_block * BLOCK_SIZE
        q_end = min(q_start + BLOCK_SIZE, seq_len)
        num_active = kv_num_blocks[0, 0, q_block].item()
        
        for i in range(num_active):
            kv_block = kv_indices[0, 0, q_block, i].item()
            mask_type = block_mask_types[0, 0, q_block, i].item()
            kv_start = kv_block * BLOCK_SIZE
            kv_end = min(kv_start + BLOCK_SIZE, seq_len)
            
            if mask_type == BlockMaskType.FULL:
                # Full attention: all q can attend to all kv in this block
                dense_mask[:, :, q_start:q_end, kv_start:kv_end] = True
            elif mask_type == BlockMaskType.CAUSAL:
                # Causal: q_idx >= kv_idx within block
                for q_idx in range(q_start, q_end):
                    kv_start_causal = kv_start
                    kv_end_causal = min(q_idx + 1, kv_end)
                    dense_mask[:, :, q_idx, kv_start_causal:kv_end_causal] = True
            elif mask_type == BlockMaskType.PARTIAL:
                # PARTIAL: half causal, half full
                q_mid = (q_start + q_end) // 2
                dense_mask[:, :, q_start:q_mid, kv_start:kv_end] = True
                for q_idx in range(q_mid, q_end):
                    kv_start_causal = kv_start
                    kv_end_causal = min(q_idx + 1, kv_end)
                    dense_mask[:, :, q_idx, kv_start_causal:kv_end_causal] = True
    
    score_mask = torch.where(dense_mask, 0.0, float('-inf')).to(torch.float32)
    
    print("Building partial block data...")
    partial_indices, partial_masks = build_partial_block_data(omni_block_mask, dense_mask)
    omni_block_mask.partial_block_mask_indices = partial_indices
    omni_block_mask.partial_block_masks = partial_masks

    print("\nComputing reference output...")
    reference_output = naive_attention_with_dense_mask(Q, K, V, mask=score_mask)
    
    Q_half = Q.half()
    K_half = K.half()
    V_half = V.half()
    reference_output_half = reference_output.half()
    
    debug_data = {
        'Q': Q_half,
        'K': K_half,
        'V': V_half,
        'Q_fp32': Q,
        'K_fp32': K,
        'V_fp32': V,
        'dense_mask': dense_mask,
        'score_mask': score_mask,
        'reference_output': reference_output_half,
        'reference_output_fp32': reference_output,
        'omni_block_mask': omni_block_mask,
        'metadata': {
            'B': B, 'H': H, 'seq_len': seq_len, 'head_dim': head_dim,
            'BLOCK_SIZE': BLOCK_SIZE, 'seed': seed, 'device': str(device),
            'pattern': 'random_full_causal_partial',
        }
    }
    
    print(f"\nSaving to {output_file}...")
    torch.save(debug_data, output_file, _use_new_zipfile_serialization=False)
    
    print(f"\n✓ Saved. Random block mask pattern (head=0, F=FULL, C=CAUSAL, P=PARTIAL):")
    for q_block in range(min(4, num_q_blocks)):
        num_active = kv_num_blocks[0, 0, q_block].item()
        pattern = []
        for i in range(num_active):
            mask_type = block_mask_types[0, 0, q_block, i].item()
            if mask_type == BlockMaskType.FULL:
                pattern.append('F')
            elif mask_type == BlockMaskType.CAUSAL:
                pattern.append('C')
            else:  # PARTIAL
                pattern.append('P')
        print(f"  q_block={q_block}: {' '.join(pattern)}")
    
    return debug_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate fixed debug data for omni_attention')
    parser.add_argument('--B', type=int, default=1, help='Batch size')
    parser.add_argument('--H', type=int, default=8, help='Number of heads')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length')
    parser.add_argument('--head_dim', type=int, default=64, help='Head dimension')
    parser.add_argument('--BLOCK_SIZE', type=int, default=128, help='Block size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='debug_data.pt', help='Output file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    
    # generate_fixed_debug_causal_data(
    #     B=args.B,
    #     H=args.H,
    #     seq_len=args.seq_len,
    #     head_dim=args.head_dim,
    #     BLOCK_SIZE=args.BLOCK_SIZE,
    #     device=args.device,
    #     output_file='debug_data.pt',
    #     seed=args.seed,
    # )

    # generate_fixed_debug_omni_data(
    #     B=args.B,
    #     H=args.H,
    #     seq_len=args.seq_len,
    #     head_dim=args.head_dim,
    #     BLOCK_SIZE=args.BLOCK_SIZE,
    #     device=args.device,
    #     output_file='debug_data_omni.pt',
    #     seed=args.seed,
    # )

    generate_fixed_debug_partial_data(
        B=args.B,
        H=args.H,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        BLOCK_SIZE=args.BLOCK_SIZE,
        device=args.device,
        output_file='debug_data_partial.pt',
        seed=args.seed,
    )

