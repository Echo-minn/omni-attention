#!/usr/bin/env python3

import torch
import numpy as np
import os
from omni_attn_torch import (
    create_causal_omni_block_mask,
    naive_attention_with_dense_mask,
    OmniBlockMask,
    BlockMaskType,
    build_partial_block_data,
)

def _set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _generate_qkv(B, H, seq_len, head_dim, device):
    return (
        torch.randn(B, H, seq_len, head_dim, device=device, dtype=torch.float32),
        torch.randn(B, H, seq_len, head_dim, device=device, dtype=torch.float32),
        torch.randn(B, H, seq_len, head_dim, device=device, dtype=torch.float32),
    )

def _compute_block_counts(seq_len, Q_BLOCK_SIZE, KV_BLOCK_SIZE):
    num_q_blocks = ((seq_len + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE)
    num_kv_blocks = ((seq_len + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE)
    return num_q_blocks, num_kv_blocks

def _compute_sparsity(dense_mask):
    """Compute sparsity ratio from dense mask.
    
    Sparsity = 1 - (number of active elements / total elements)
    Returns a float between 0.0 (dense) and 1.0 (fully sparse).
    """
    total_elements = dense_mask.numel()
    active_elements = dense_mask.sum().item()
    sparsity = 1.0 - (active_elements / total_elements)
    return sparsity

def _create_dense_mask_from_blocks(B, H, seq_len, num_q_blocks, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
                                    kv_num_blocks, kv_indices, block_mask_types, device):
    dense_mask = torch.zeros(B, H, seq_len, seq_len, device=device, dtype=torch.bool)
    
    for q_block in range(num_q_blocks):
        q_start = q_block * Q_BLOCK_SIZE
        q_end = min(q_start + Q_BLOCK_SIZE, seq_len)
        num_active = kv_num_blocks[0, 0, q_block].item()
        
        for i in range(num_active):
            kv_block = kv_indices[0, 0, q_block, i].item()
            mask_type = block_mask_types[0, 0, q_block, i].item()
            kv_start = kv_block * KV_BLOCK_SIZE
            kv_end = min(kv_start + KV_BLOCK_SIZE, seq_len)
            
            if mask_type == BlockMaskType.FULL:
                dense_mask[:, :, q_start:q_end, kv_start:kv_end] = True
            elif mask_type == BlockMaskType.CAUSAL:
                for q_idx in range(q_start, q_end):
                    dense_mask[:, :, q_idx, kv_start:min(q_idx + 1, kv_end)] = True
            elif mask_type == BlockMaskType.PARTIAL:
                q_mid = (q_start + q_end) // 2
                dense_mask[:, :, q_start:q_mid, kv_start:kv_end] = True
                for q_idx in range(q_mid, q_end):
                    dense_mask[:, :, q_idx, kv_start:min(q_idx + 1, kv_end)] = True
    
    return dense_mask

def _save_debug_data(debug_data, output_file, pattern_name=None):
    print(f"\nSaving to {output_file}...")
    torch.save(debug_data, output_file, _use_new_zipfile_serialization=False)
    print(f"✓ Saved. File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    # Print sparsity if available in metadata
    if 'sparsity' in debug_data.get('metadata', {}):
        sparsity = debug_data['metadata']['sparsity']
        print(f"Sparsity: {sparsity:.4f} ({sparsity*100:.2f}% sparse)")
    
    if pattern_name:
        print(f"Block mask pattern (head=0, F=FULL, C=CAUSAL, P=PARTIAL):")
        print_block_pattern(debug_data['omni_block_mask'].block_mask_types,
                          debug_data['omni_block_mask'].kv_num_blocks)

def print_block_pattern(block_mask_types, kv_num_blocks, number_of_blocks_to_print=None):
    if number_of_blocks_to_print is None:
        number_of_blocks_to_print = kv_num_blocks.shape[-1]
    for q_block in range(number_of_blocks_to_print):
        num_active = kv_num_blocks[0, 0, q_block].item()
        pattern = []
        for i in range(num_active):
            mask_type = block_mask_types[0, 0, q_block, i].item()
            pattern.append('F' if mask_type == BlockMaskType.FULL else
                          'C' if mask_type == BlockMaskType.CAUSAL else
                          'P' if mask_type == BlockMaskType.PARTIAL else 'M')
        print(f"  q_block={q_block}: {' '.join(pattern)}")

def generate_fixed_debug_causal_data(
    B=1, H=8, seq_len=512, head_dim=64, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128,
    device="cuda", output_file="debug_data.pt", seed=42
):
    _set_seed(seed)
    print(f"Generating fixed debug data: B={B}, H={H}, seq_len={seq_len}, head_dim={head_dim}")
    print(f"  Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}, seed={seed}, device={device}")
    
    Q, K, V = _generate_qkv(B, H, seq_len, head_dim, device)
    
    causal_mask = torch.zeros(B, H, seq_len, seq_len, device=device, dtype=torch.bool)
    for q_idx in range(seq_len):
        causal_mask[:, :, q_idx, :q_idx+1] = True
    
    score_mask = torch.where(causal_mask, 0.0, float('-inf')).to(torch.float32)
    modality_positions = torch.zeros(B, 1, 3, dtype=torch.int32, device=device)
    modality_positions[:, 0, :] = torch.tensor([1, 0, seq_len], dtype=torch.int32, device=device)
    
    omni_block_mask = create_causal_omni_block_mask(
        batch=B, nheads=H, q_len=seq_len, kv_len=seq_len,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE, KV_BLOCK_SIZE=KV_BLOCK_SIZE, device=device
    )
    
    print("\nComputing reference output...")
    reference_output = naive_attention_with_dense_mask(Q, K, V, mask=score_mask)
    
    sparsity = _compute_sparsity(causal_mask)
    
    debug_data = {
        'Q': Q.half(), 'K': K.half(), 'V': V.half(),
        'Q_fp32': Q, 'K_fp32': K, 'V_fp32': V,
        'causal_mask': causal_mask, 'score_mask': score_mask,
        'reference_output': reference_output.half(),
        'reference_output_fp32': reference_output,
        'omni_block_mask': omni_block_mask,
        'modality_positions': modality_positions,
        'metadata': {
            'B': B, 'H': H, 'seq_len': seq_len, 'head_dim': head_dim,
            'Q_BLOCK_SIZE': Q_BLOCK_SIZE, 'KV_BLOCK_SIZE': KV_BLOCK_SIZE,
            'seed': seed, 'device': str(device), 'sparsity': sparsity,
        }
    }
    
    _save_debug_data(debug_data, output_file, "Causal")
    return debug_data

def generate_fixed_debug_F_C_data(
    B=1, H=8, seq_len=512, head_dim=64, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128,
    device="cuda", output_file="debug_data_omni.pt", seed=42
):
    _set_seed(seed)
    print(f"Generating random FULL/PARTIAL debug data: B={B}, H={H}, seq_len={seq_len}, head_dim={head_dim}")
    print(f"  Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}")
    
    Q, K, V = _generate_qkv(B, H, seq_len, head_dim, device)
    num_q_blocks, num_kv_blocks = _compute_block_counts(seq_len, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
    
    kv_num_blocks = torch.zeros(B, H, num_q_blocks, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(B, H, num_q_blocks, num_kv_blocks, dtype=torch.int32, device=device)
    block_mask_types = torch.zeros(B, H, num_q_blocks, num_kv_blocks, dtype=torch.int32, device=device)
    
    ratio = 0.75
    for q_block in range(num_q_blocks):
        num_active = 0
        q_end = min((q_block + 1) * Q_BLOCK_SIZE, seq_len)
        
        for kv_block in range(num_kv_blocks):
            kv_end = min((kv_block + 1) * KV_BLOCK_SIZE, seq_len)
            if kv_end <= q_end:
                mask_type = BlockMaskType.FULL if torch.rand(1).item() < ratio else BlockMaskType.PARTIAL
                kv_indices[:, :, q_block, num_active] = kv_block
                block_mask_types[:, :, q_block, num_active] = mask_type
                num_active += 1
        
        kv_num_blocks[:, :, q_block] = num_active
        block_mask_types[:, :, q_block, num_active:] = BlockMaskType.MASKED
    
    omni_block_mask = OmniBlockMask(
        kv_num_blocks=kv_num_blocks, kv_indices=kv_indices, block_mask_types=block_mask_types,
        q_len=seq_len, kv_len=seq_len, Q_BLOCK_SIZE=Q_BLOCK_SIZE, KV_BLOCK_SIZE=KV_BLOCK_SIZE
    )
    
    dense_mask = _create_dense_mask_from_blocks(
        B, H, seq_len, num_q_blocks, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
        kv_num_blocks, kv_indices, block_mask_types, device
    )
    score_mask = torch.where(dense_mask, 0.0, float('-inf')).to(torch.float32)
    
    print("\nComputing reference output...")
    reference_output = naive_attention_with_dense_mask(Q, K, V, mask=score_mask)
    
    sparsity = _compute_sparsity(dense_mask)
    
    debug_data = {
        'Q': Q.half(), 'K': K.half(), 'V': V.half(),
        'Q_fp32': Q, 'K_fp32': K, 'V_fp32': V,
        'dense_mask': dense_mask, 'score_mask': score_mask,
        'reference_output': reference_output.half(),
        'reference_output_fp32': reference_output,
        'omni_block_mask': omni_block_mask,
        'metadata': {
            'B': B, 'H': H, 'seq_len': seq_len, 'head_dim': head_dim,
            'Q_BLOCK_SIZE': Q_BLOCK_SIZE, 'KV_BLOCK_SIZE': KV_BLOCK_SIZE,
            'seed': seed, 'device': str(device), 'pattern': 'random_full_partial',
            'sparsity': sparsity,
        }
    }
    
    _save_debug_data(debug_data, output_file, "Random FULL/PARTIAL")
    return debug_data

def generate_fixed_debug_F_data(
    B=1, H=8, seq_len=512, head_dim=64, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128,
    device="cuda", output_file="debug_data_omni.pt", seed=42
):
    _set_seed(seed)
    print(f"Generating fixed FULL debug data: B={B}, H={H}, seq_len={seq_len}, head_dim={head_dim}")
    print(f"  Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}")
    
    Q, K, V = _generate_qkv(B, H, seq_len, head_dim, device)
    num_q_blocks, num_kv_blocks = _compute_block_counts(seq_len, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
    
    kv_num_blocks = torch.zeros(B, H, num_q_blocks, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(B, H, num_q_blocks, num_kv_blocks, dtype=torch.int32, device=device)
    block_mask_types = torch.zeros(B, H, num_q_blocks, num_kv_blocks, dtype=torch.int32, device=device)
    
    for q_block in range(num_q_blocks):
        num_active = 0
        # FIX: Use block index comparison to match create_causal_omni_block_mask logic
        for kv_block in range(q_block + 1):  # Q block can attend to KV blocks 0 through q_block
            kv_indices[:, :, q_block, num_active] = kv_block
            block_mask_types[:, :, q_block, num_active] = BlockMaskType.FULL
            num_active += 1
        
        kv_num_blocks[:, :, q_block] = num_active
        block_mask_types[:, :, q_block, num_active:] = BlockMaskType.MASKED
    
    omni_block_mask = OmniBlockMask(
        kv_num_blocks=kv_num_blocks, kv_indices=kv_indices, block_mask_types=block_mask_types,
        q_len=seq_len, kv_len=seq_len, Q_BLOCK_SIZE=Q_BLOCK_SIZE, KV_BLOCK_SIZE=KV_BLOCK_SIZE
    )
    
    dense_mask = _create_dense_mask_from_blocks(
        B, H, seq_len, num_q_blocks, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
        kv_num_blocks, kv_indices, block_mask_types, device
    )
    score_mask = torch.where(dense_mask, 0.0, float('-inf')).to(torch.float32)
    
    print("\nComputing reference output...")
    reference_output = naive_attention_with_dense_mask(Q, K, V, mask=score_mask)
    
    sparsity = _compute_sparsity(dense_mask)
    
    debug_data = {
        'Q': Q.half(), 'K': K.half(), 'V': V.half(),
        'Q_fp32': Q, 'K_fp32': K, 'V_fp32': V,
        'dense_mask': dense_mask, 'score_mask': score_mask,
        'reference_output': reference_output.half(),
        'reference_output_fp32': reference_output,
        'omni_block_mask': omni_block_mask,
        'metadata': {
            'B': B, 'H': H, 'seq_len': seq_len, 'head_dim': head_dim,
            'Q_BLOCK_SIZE': Q_BLOCK_SIZE, 'KV_BLOCK_SIZE': KV_BLOCK_SIZE,
            'seed': seed, 'device': str(device), 'pattern': 'full',
            'sparsity': sparsity,
        }
    }
    
    _save_debug_data(debug_data, output_file, "FULL")
    return debug_data

def generate_fixed_debug_F_C_P_data(
    B=1, H=8, seq_len=512, head_dim=64, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128,
    device="cuda", output_file="debug_data_partial.pt", seed=42
):
    _set_seed(seed)
    print(f"Generating random FULL/CAUSAL/PARTIAL debug data: B={B}, H={H}, seq_len={seq_len}, head_dim={head_dim}")
    print(f"  Q_BLOCK_SIZE={Q_BLOCK_SIZE}, KV_BLOCK_SIZE={KV_BLOCK_SIZE}")
    
    Q, K, V = _generate_qkv(B, H, seq_len, head_dim, device)
    num_q_blocks, num_kv_blocks = _compute_block_counts(seq_len, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
    
    kv_num_blocks = torch.zeros(B, H, num_q_blocks, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(B, H, num_q_blocks, num_kv_blocks, dtype=torch.int32, device=device)
    block_mask_types = torch.zeros(B, H, num_q_blocks, num_kv_blocks, dtype=torch.int32, device=device)
    
    for q_block in range(num_q_blocks):
        num_active = 0
        # FIX: Use block index comparison to match create_causal_omni_block_mask logic
        for kv_block in range(q_block + 1):  # Q block can attend to KV blocks 0 through q_block
            rand_val = torch.rand(1).item()
            if kv_block < q_block:
                # Past blocks: can be FULL, CAUSAL, or PARTIAL
                mask_type = (BlockMaskType.FULL if rand_val < 0.333 else
                            BlockMaskType.CAUSAL if rand_val < 0.666 else
                            BlockMaskType.PARTIAL)
            else:
                # Current block (diagonal): CAUSAL or PARTIAL (not FULL for causal pattern)
                mask_type = (BlockMaskType.CAUSAL if rand_val < 0.5 else
                            BlockMaskType.PARTIAL)
            
            kv_indices[:, :, q_block, num_active] = kv_block
            block_mask_types[:, :, q_block, num_active] = mask_type
            num_active += 1
        
        kv_num_blocks[:, :, q_block] = num_active
        block_mask_types[:, :, q_block, num_active:] = BlockMaskType.MASKED
    
    omni_block_mask = OmniBlockMask(
        kv_num_blocks=kv_num_blocks, kv_indices=kv_indices, block_mask_types=block_mask_types,
        q_len=seq_len, kv_len=seq_len, Q_BLOCK_SIZE=Q_BLOCK_SIZE, KV_BLOCK_SIZE=KV_BLOCK_SIZE
    )
    
    dense_mask = _create_dense_mask_from_blocks(
        B, H, seq_len, num_q_blocks, Q_BLOCK_SIZE, KV_BLOCK_SIZE,
        kv_num_blocks, kv_indices, block_mask_types, device
    )
    score_mask = torch.where(dense_mask, 0.0, float('-inf')).to(torch.float32)
    
    print("Building partial block data...")
    partial_indices, partial_masks = build_partial_block_data(omni_block_mask, dense_mask)
    omni_block_mask.partial_block_mask_indices = partial_indices
    omni_block_mask.partial_block_masks = partial_masks
    
    print("\nComputing reference output...")
    reference_output = naive_attention_with_dense_mask(Q, K, V, mask=score_mask)
    
    sparsity = _compute_sparsity(dense_mask)
    
    debug_data = {
        'Q': Q.half(), 'K': K.half(), 'V': V.half(),
        'Q_fp32': Q, 'K_fp32': K, 'V_fp32': V,
        'dense_mask': dense_mask, 'score_mask': score_mask,
        'reference_output': reference_output.half(),
        'reference_output_fp32': reference_output,
        'omni_block_mask': omni_block_mask,
        'metadata': {
            'B': B, 'H': H, 'seq_len': seq_len, 'head_dim': head_dim,
            'Q_BLOCK_SIZE': Q_BLOCK_SIZE, 'KV_BLOCK_SIZE': KV_BLOCK_SIZE,
            'seed': seed, 'device': str(device), 'pattern': 'random_full_causal_partial',
            'sparsity': sparsity,
        }
    }
    
    _save_debug_data(debug_data, output_file, "Random FULL/CAUSAL/PARTIAL")
    return debug_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate fixed debug data for omni_attention')
    parser.add_argument('--B', type=int, default=1, help='Batch size')
    parser.add_argument('--H', type=int, default=8, help='Number of heads')
    parser.add_argument('--seq_len', type=int, default=512, help='Sequence length')
    parser.add_argument('--head_dim', type=int, default=64, help='Head dimension')
    parser.add_argument('--Q_BLOCK_SIZE', type=int, default=64, help='Block size for query')
    parser.add_argument('--KV_BLOCK_SIZE', type=int, default=64, help='Block size for key/value')
    parser.add_argument('--seed', type=int, default=40, help='Random seed')
    parser.add_argument('--output', type=str, default='debug_data.pt', help='Output file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    kwargs = {
        'B': args.B, 'H': args.H, 'seq_len': args.seq_len, 'head_dim': args.head_dim,
        'Q_BLOCK_SIZE': args.Q_BLOCK_SIZE, 'KV_BLOCK_SIZE': args.KV_BLOCK_SIZE,
        'device': args.device, 'seed': args.seed
    }
    
    generate_fixed_debug_F_C_P_data(
        **kwargs, output_file=f'data/{args.seq_len}/{args.Q_BLOCK_SIZE}_{args.KV_BLOCK_SIZE}_F_C_P.pt'
    )

