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
                          debug_data['omni_block_mask'].kv_num_blocks,
                          debug_data['omni_block_mask'].kv_indices)

def print_block_pattern(block_mask_types, kv_num_blocks, kv_indices=None, number_of_blocks_to_print=None):
    """Print per-block mask types including masked columns.
    
    Uses kv_indices to place mask types in the correct KV block columns.
    """
    if number_of_blocks_to_print is None:
        number_of_blocks_to_print = block_mask_types.shape[-2]  # num_q_blocks
    num_kv_blocks = block_mask_types.shape[-1]
    for q_block in range(number_of_blocks_to_print):
        row = [' '] * num_kv_blocks
        num_active = kv_num_blocks[0, 0, q_block].item()
        for i in range(num_active):
            kv_block = kv_indices[0, 0, q_block, i].item() if kv_indices is not None else i
            mask_type = block_mask_types[0, 0, q_block, i].item()
            row[kv_block] = (
                'F' if mask_type == BlockMaskType.FULL else
                'C' if mask_type == BlockMaskType.CAUSAL else
                'P' if mask_type == BlockMaskType.PARTIAL else
                ' '
            )
        print(f"  q_block={q_block}: {' '.join(row)}")

def generate_causal_mask(
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
    # modality_positions = torch.zeros(B, 1, 3, dtype=torch.int32, device=device)
    # modality_positions[:, 0, :] = torch.tensor([1, 0, seq_len], dtype=torch.int32, device=device)
    
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
        # 'modality_positions': modality_positions,
        'metadata': {
            'B': B, 'H': H, 'seq_len': seq_len, 'head_dim': head_dim,
            'Q_BLOCK_SIZE': Q_BLOCK_SIZE, 'KV_BLOCK_SIZE': KV_BLOCK_SIZE,
            'seed': seed, 'device': str(device), 'sparsity': sparsity,
        }
    }
    
    _save_debug_data(debug_data, output_file, "Causal")
    return debug_data

def generate_interleaved_mask(
    B=1, H=8, seq_len=512, head_dim=64, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128,
    device="cuda", output_file="debug_data.pt", seed=42
):
    _set_seed(seed)
    # Pick 3 random positive segments that sum to seq_len
    while True:
        segment_len1 = torch.randint(1, seq_len - 1, (1,)).item()
        remaining = seq_len - segment_len1
        segment_len3 = torch.randint(30, remaining - 30, (1,)).item()
        segment_len2 = seq_len - segment_len1 - segment_len3
        if segment_len3 > 0:
            break
    print(f"Segment lengths: {segment_len1}, {segment_len2}, {segment_len3}")
    seg1_end = segment_len1
    seg2_end = segment_len1 + segment_len2
    boundaries = [0, seg1_end, seg2_end, seq_len]
    
    # Q/K/V
    Q, K, V = _generate_qkv(B, H, seq_len, head_dim, device)
    
    # Interleaved mask
    dense_mask = torch.zeros(B, H, seq_len, seq_len, device=device, dtype=torch.bool)
    for q_idx in range(seq_len):
        if q_idx < seg1_end:
            dense_mask[:, :, q_idx, :q_idx + 1] = True
        elif q_idx < seg2_end:
            # Segment 2 attends fully to segment 1 and fully within segment 2
            dense_mask[:, :, q_idx, :seg2_end] = True
        else:
            dense_mask[:, :, q_idx, :seg2_end] = True
            within_seg3 = q_idx - seg2_end
            dense_mask[:, :, q_idx, seg2_end:seg2_end + within_seg3 + 1] = True
    
    score_mask = torch.where(dense_mask, 0.0, float('-inf')).to(torch.float32)
    
    # Convert dense mask to block-sparse OmniBlockMask
    num_q_blocks, num_kv_blocks = _compute_block_counts(seq_len, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
    max_blocks = num_kv_blocks
    kv_num_blocks = torch.zeros(B, H, num_q_blocks, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(B, H, num_q_blocks, max_blocks, dtype=torch.int32, device=device)
    block_mask_types = torch.full(
        (B, H, num_q_blocks, max_blocks), BlockMaskType.MASKED, dtype=torch.int32, device=device
    )
    
    for q_block in range(num_q_blocks):
        q_start = q_block * Q_BLOCK_SIZE
        q_end = min(q_start + Q_BLOCK_SIZE, seq_len)
        num_active = 0
        # Detect if this q block crosses a segment boundary
        q_cross_boundary = any((b > q_start) and (b < q_end) for b in boundaries)
        for kv_block in range(num_kv_blocks):
            kv_start = kv_block * KV_BLOCK_SIZE
            kv_end = min(kv_start + KV_BLOCK_SIZE, seq_len)
            kv_cross_boundary = any((b > kv_start) and (b < kv_end) for b in boundaries)
            
            block = dense_mask[0, 0, q_start:q_end, kv_start:kv_end]
            if not block.any():
                continue
            
            if block.all():
                mask_type = BlockMaskType.FULL
            else:
                q_positions = torch.arange(q_start, q_end, device=device)
                kv_positions = torch.arange(kv_start, kv_end, device=device)
                causal_pattern = kv_positions.unsqueeze(0) <= q_positions.unsqueeze(1)
                # If the block straddles a segment boundary and is not fully unmasked,
                # treat it as PARTIAL even if the pattern looks causal.
                if (q_cross_boundary or kv_cross_boundary) and not block.all():
                    mask_type = BlockMaskType.PARTIAL
                elif torch.equal(block, causal_pattern):
                    mask_type = BlockMaskType.CAUSAL
                else:
                    mask_type = BlockMaskType.PARTIAL
            
            kv_indices[:, :, q_block, num_active] = kv_block
            block_mask_types[:, :, q_block, num_active] = mask_type
            num_active += 1
        
        kv_num_blocks[:, :, q_block] = num_active
        if num_active < max_blocks:
            block_mask_types[:, :, q_block, num_active:] = BlockMaskType.MASKED
    
    omni_block_mask = OmniBlockMask(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        block_mask_types=block_mask_types,
        q_len=seq_len,
        kv_len=seq_len,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
    )
    
    # Attach partial block data when needed
    if (block_mask_types == BlockMaskType.PARTIAL).any():
        partial_indices, partial_masks = build_partial_block_data(omni_block_mask, dense_mask)
        omni_block_mask.partial_block_mask_indices = partial_indices
        omni_block_mask.partial_block_masks = partial_masks
    
    print("\nComputing reference output...")
    reference_output = naive_attention_with_dense_mask(Q, K, V, mask=score_mask)
    
    sparsity = _compute_sparsity(dense_mask)
    
    # modality_positions = torch.zeros(B, 3, 3, dtype=torch.int32, device=device)
    # modality_positions[:, 0, :] = torch.tensor([1, 0, segment_len1], dtype=torch.int32, device=device)
    # modality_positions[:, 1, :] = torch.tensor([2, segment_len1, segment_len2], dtype=torch.int32, device=device)
    # modality_positions[:, 2, :] = torch.tensor([1, segment_len1 + segment_len2, segment_len3], dtype=torch.int32, device=device)
    
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
            'seed': seed, 'device': str(device), 'pattern': 'interleaved',
            'segments': [segment_len1, segment_len2, segment_len3],
            'sparsity': sparsity,
        }
    }
    
    _save_debug_data(debug_data, output_file, "Interleaved")
    return debug_data


def generate_document_mask(
    B=1, H=8, seq_len=512, head_dim=64, Q_BLOCK_SIZE=128, KV_BLOCK_SIZE=128,
    device="cuda", output_file="debug_data.pt", seed=42
):
    _set_seed(seed)
    # Pick 3 random positive segments that sum to seq_len
    while True:
        segment_len1 = torch.randint(10, seq_len - 10, (1,)).item()
        remaining = seq_len - segment_len1
        segment_len2 = torch.randint(1, remaining - 1, (1,)).item()
        segment_len3 = seq_len - segment_len1 - segment_len2
        if segment_len3 > 0:
            break
    print(f"Document segments: {segment_len1}, {segment_len2}, {segment_len3}")
    seg1_end = segment_len1
    seg2_end = segment_len1 + segment_len2
    boundaries = [0, seg1_end, seg2_end, seq_len]
    
    Q, K, V = _generate_qkv(B, H, seq_len, head_dim, device)
    
    # Dense mask: only within each segment (three disjoint squares)
    dense_mask = torch.zeros(B, H, seq_len, seq_len, device=device, dtype=torch.bool)
    segments = [(0, seg1_end), (seg1_end, seg2_end), (seg2_end, seq_len)]
    for start, end in segments:
        dense_mask[:, :, start:end, start:end] = True
    
    score_mask = torch.where(dense_mask, 0.0, float('-inf')).to(torch.float32)
    
    # Convert dense mask to block-sparse OmniBlockMask
    num_q_blocks, num_kv_blocks = _compute_block_counts(seq_len, Q_BLOCK_SIZE, KV_BLOCK_SIZE)
    max_blocks = num_kv_blocks
    kv_num_blocks = torch.zeros(B, H, num_q_blocks, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(B, H, num_q_blocks, max_blocks, dtype=torch.int32, device=device)
    block_mask_types = torch.full(
        (B, H, num_q_blocks, max_blocks), BlockMaskType.MASKED, dtype=torch.int32, device=device
    )
    
    for q_block in range(num_q_blocks):
        q_start = q_block * Q_BLOCK_SIZE
        q_end = min(q_start + Q_BLOCK_SIZE, seq_len)
        num_active = 0
        q_cross_boundary = any((b > q_start) and (b < q_end) for b in boundaries)
        for kv_block in range(num_kv_blocks):
            kv_start = kv_block * KV_BLOCK_SIZE
            kv_end = min(kv_start + KV_BLOCK_SIZE, seq_len)
            kv_cross_boundary = any((b > kv_start) and (b < kv_end) for b in boundaries)
            
            block = dense_mask[0, 0, q_start:q_end, kv_start:kv_end]
            if not block.any():
                continue
            
            if block.all() and not (q_cross_boundary or kv_cross_boundary):
                mask_type = BlockMaskType.FULL
            else:
                mask_type = BlockMaskType.PARTIAL
            
            kv_indices[:, :, q_block, num_active] = kv_block
            block_mask_types[:, :, q_block, num_active] = mask_type
            num_active += 1
        
        kv_num_blocks[:, :, q_block] = num_active
        if num_active < max_blocks:
            block_mask_types[:, :, q_block, num_active:] = BlockMaskType.MASKED
    
    omni_block_mask = OmniBlockMask(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        block_mask_types=block_mask_types,
        q_len=seq_len,
        kv_len=seq_len,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
    )
    
    if (block_mask_types == BlockMaskType.PARTIAL).any():
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
            'seed': seed, 'device': str(device), 'pattern': 'document',
            'segments': [segment_len1, segment_len2, segment_len3],
            'sparsity': sparsity,
        }
    }
    
    _save_debug_data(debug_data, output_file, "Document")
    return debug_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate fixed debug data for omni_attention')
    parser.add_argument('--B', type=int, default=1, help='Batch size')
    parser.add_argument('--H', type=int, default=8, help='Number of heads')
    parser.add_argument('--seq_len', type=int, default=1024, help='Sequence length')
    parser.add_argument('--head_dim', type=int, default=1024, help='Head dimension')
    parser.add_argument('--Q_BLOCK_SIZE', type=int, default=64, help='Block size for query')
    parser.add_argument('--KV_BLOCK_SIZE', type=int, default=64, help='Block size for key/value')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output', type=str, default='debug_data.pt', help='Output file')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    args = parser.parse_args()
    kwargs = {
        'B': args.B, 'H': args.H, 'seq_len': args.seq_len, 'head_dim': args.head_dim,
        'Q_BLOCK_SIZE': args.Q_BLOCK_SIZE, 'KV_BLOCK_SIZE': args.KV_BLOCK_SIZE,
        'device': args.device, 'seed': args.seed
    }

    print(f"B: {args.B}, H: {args.H}, seq_len: {args.seq_len}, head_dim: {args.head_dim}, Q_BLOCK_SIZE: {args.Q_BLOCK_SIZE}, KV_BLOCK_SIZE: {args.KV_BLOCK_SIZE}")
    
    generate_causal_mask(**kwargs, output_file=f'data/attn_data/{args.seq_len}_causal_{args.Q_BLOCK_SIZE}_{args.KV_BLOCK_SIZE}.pt')

    # generate_interleaved_mask(**kwargs, output_file=f'data/attn_data/{args.seq_len}_interleaved_{args.Q_BLOCK_SIZE}_{args.KV_BLOCK_SIZE}.pt')
    # generate_document_mask(**kwargs, output_file=f'data/attn_data/{args.seq_len}_document_{args.Q_BLOCK_SIZE}_{args.KV_BLOCK_SIZE}.pt')

