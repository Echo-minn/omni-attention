#!/usr/bin/env python3
"""Print VLM block mask matrix."""

import torch
from omni_attn_torch import create_random_vlm_block_mask, BlockMaskType

def print_block_structure(block_mask, batch_idx=0, head_idx=0):
    """Print block-level mask structure."""
    B, H, num_q_blocks, max_blocks = block_mask.kv_indices.shape
    kv_num_blocks = block_mask.kv_num_blocks[batch_idx, head_idx].cpu()
    kv_indices = block_mask.kv_indices[batch_idx, head_idx].cpu()
    block_mask_types = block_mask.block_mask_types[batch_idx, head_idx].cpu()
    
    print("=" * 80)
    print("BLOCK-LEVEL MASK STRUCTURE")
    print("=" * 80)
    print(f"Q blocks: {num_q_blocks}, KV blocks: {block_mask.num_kv_blocks}, BLOCK_SIZE: {block_mask.BLOCK_SIZE}")
    print(f"Legend: F=FULL, C=CAUSAL, M=MASKED, .=no block\n")
    
    # Print block matrix
    block_matrix = []
    for q_block in range(num_q_blocks):
        row = ['M'] * block_mask.num_kv_blocks
        num_active = kv_num_blocks[q_block].item()
        for idx in range(num_active):
            kv_block = kv_indices[q_block, idx].item()
            mask_type = block_mask_types[q_block, idx].item()
            if mask_type == BlockMaskType.FULL:
                row[kv_block] = 'F'
            elif mask_type == BlockMaskType.CAUSAL:
                row[kv_block] = 'C'
        block_matrix.append(row)
    
    # Print header
    print("Q\\KV", end=" ")
    for kv in range(min(block_mask.num_kv_blocks, 20)):
        print(f"{kv:2d}", end="")
    if block_mask.num_kv_blocks > 20:
        print("...")
    else:
        print()
    print("-" * (4 + min(block_mask.num_kv_blocks, 20) * 2))
    
    # Print rows
    for q_block in range(min(num_q_blocks, 20)):
        print(f"{q_block:3d} ", end="")
        for kv_block in range(min(block_mask.num_kv_blocks, 20)):
            print(f" {block_matrix[q_block][kv_block]}", end="")
        if block_mask.num_kv_blocks > 20:
            print(" ...")
        else:
            print()
    if num_q_blocks > 20:
        print("...")
    print()

def print_token_types(q_len, num_image_regions, image_region_size, BLOCK_SIZE):
    """Recreate and print token type information."""
    # Recreate the same logic as in create_random_vlm_block_mask
    spacing = q_len // (num_image_regions + 1)
    image_regions = []
    for i in range(num_image_regions):
        start = (i + 1) * spacing - image_region_size // 2
        start = max(0, min(start, q_len - image_region_size))
        end = start + image_region_size
        image_regions.append((start, end))
    
    token_types = torch.zeros(q_len, dtype=torch.int32)
    for i, (start, end) in enumerate(image_regions):
        token_types[start:end] = i + 1
    
    print("=" * 80)
    print("TOKEN TYPE DISTRIBUTION")
    print("=" * 80)
    print(f"Image regions: {image_regions}")
    print(f"0 = text, 1+ = image region ID\n")
    
    # Show which blocks contain which types
    num_q_blocks = (q_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    print("Block token types (dominant type per block):")
    for q_block in range(num_q_blocks):
        q_start = q_block * BLOCK_SIZE
        q_end = min(q_start + BLOCK_SIZE, q_len)
        q_types = token_types[q_start:q_end]
        q_type = q_types.mode().values.item()
        type_str = f"img{q_type-1}" if q_type > 0 else "text"
        print(f"  Q block {q_block} [{q_start:3d}:{q_end:3d}]: {type_str}")
    print()

def main():
    # Default parameters
    batch = 1
    nheads = 1
    q_len = 512
    kv_len = 512
    num_image_regions = 2
    image_region_size = 128
    BLOCK_SIZE = 128
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Creating VLM block mask...")
    print(f"  q_len={q_len}, kv_len={kv_len}")
    print(f"  num_image_regions={num_image_regions}, image_region_size={image_region_size}")
    print(f"  BLOCK_SIZE={BLOCK_SIZE}\n")
    
    # Create mask
    block_mask = create_random_vlm_block_mask(
        batch=batch,
        nheads=nheads,
        q_len=q_len,
        kv_len=kv_len,
        num_image_regions=num_image_regions,
        image_region_size=image_region_size,
        BLOCK_SIZE=BLOCK_SIZE,
        device=device,
        seed=42  # For reproducibility
    )
    
    # Print token types
    print_token_types(q_len, num_image_regions, image_region_size, BLOCK_SIZE)
    
    # Print block structure
    print_block_structure(block_mask, batch_idx=0, head_idx=0)

if __name__ == "__main__":
    main()

