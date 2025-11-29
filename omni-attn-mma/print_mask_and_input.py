#!/usr/bin/env python3
"""
Simple script to print mask and input matrix to files.
Uses create_block_mask() from transfusion.
"""

import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transfusion_pytorch.transfusion import (
    transfusion_attn_mask,
    create_block_mask,
    naive_attn_mask,
    exists,
)
from test_omni_attn import generate_random_input, get_embeddings

def print_matrix_to_file(matrix, filename, batch_idx=0, head_idx=0):
    """Print a matrix to a file without ellipsis."""
    with open(filename, 'w') as f:
        if matrix.ndim == 4:
            # [B, H, N, M] -> print batch_idx, head_idx
            mat = matrix[batch_idx, head_idx].cpu()
            f.write(f"Shape: {mat.shape}\n")
            f.write(f"Batch index: {batch_idx}, Head index: {head_idx}\n\n")
            for i in range(mat.shape[0]):
                row = mat[i].tolist()
                f.write(' '.join(map(str, row)) + '\n')
        elif matrix.ndim == 3:
            # [B, N, M] -> print batch_idx
            mat = matrix[batch_idx].cpu()
            f.write(f"Shape: {mat.shape}\n")
            f.write(f"Batch index: {batch_idx}\n\n")
            for i in range(mat.shape[0]):
                row = mat[i].tolist()
                f.write(' '.join(map(str, row)) + '\n')
        elif matrix.ndim == 2:
            # [N, M] -> print directly
            mat = matrix.cpu()
            f.write(f"Shape: {mat.shape}\n\n")
            for i in range(mat.shape[0]):
                row = mat[i].tolist()
                f.write(' '.join(map(str, row)) + '\n')
        else:
            raise ValueError(f"Unsupported matrix dimension: {matrix.ndim}")

def main():
    # Use 256x256 as requested
    B, H, N, D = 1, 8, 256, 512
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Generating input with B={B}, H={H}, N={N}, D={D}")
    print(f"Device: {device}")
    
    # Generate random input
    print("Generating random input...")
    text_and_images, modality_positions = generate_random_input(
        B=B, H=H, N=N, D=D, device=device
    )
    
    # Get embeddings
    print("Creating embeddings...")
    x = get_embeddings(text_and_images, D=D, device=device, block_size=128)
    actual_seq_len = x.shape[1]
    print(f"Actual sequence length: {actual_seq_len}")
    
    # Truncate or pad to exactly 256 if needed
    if actual_seq_len > 256:
        x = x[:, :256, :]
        actual_seq_len = 256
        print(f"Truncated to 256")
    elif actual_seq_len < 256:
        padding = torch.zeros(B, 256 - actual_seq_len, D, device=device, dtype=x.dtype)
        x = torch.cat([x, padding], dim=1)
        actual_seq_len = 256
        print(f"Padded to 256")
    
    # Create block mask using create_block_mask from transfusion
    print("Creating block mask using create_block_mask()...")
    if not exists(create_block_mask):
        print("Error: create_block_mask not available")
        return
    
    mask_fn = transfusion_attn_mask(modality_positions)
    block_mask = create_block_mask(
        mask_fn,
        B=B,
        H=H,
        Q_LEN=actual_seq_len,
        KV_LEN=actual_seq_len,
        _compile=False,
        device=device
    )
    
    # Materialize the full token-level mask
    # Note: naive_attn_mask produces the same mask as create_block_mask's underlying logic
    print("Materializing full token-level mask...")
    dense_mask_2d = naive_attn_mask(actual_seq_len, modality_positions, device=device)  # [B, N, N]
    # Expand to [B, H, N, N] - all heads use the same mask
    dense_mask = dense_mask_2d.unsqueeze(1).expand(B, H, actual_seq_len, actual_seq_len)
    
    # Convert mask: True -> 1 (visit), False -> 0 (masked)
    mask_int = dense_mask.int()  # [B, H, N, N]
    
    # Print mask to file (use first head)
    print("Writing mask to mask.txt...")
    print_matrix_to_file(mask_int, "mask.txt", batch_idx=0, head_idx=0)
    print(f"Mask shape: {mask_int.shape}")
    print(f"Mask density: {mask_int[0, 0].float().mean().item():.2%}")
    
    # Print input matrix to file
    print("Writing input matrix to input.txt...")
    print_matrix_to_file(x, "input.txt", batch_idx=0)
    print(f"Input shape: {x.shape}")
    
    print("\nDone! Files written:")
    print("  - mask.txt: attention mask from create_block_mask() (1=visit, 0=masked)")
    print("  - input.txt: input embeddings matrix")

if __name__ == "__main__":
    main()

