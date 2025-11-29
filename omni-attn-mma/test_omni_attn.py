#!/usr/bin/env python3
"""
Omni-Attention Benchmark and Correctness Test Suite

This script benchmarks and verifies the correctness of:
1. Naive PyTorch MHA (baseline)
2. FlexAttention (PyTorch 2.5+)
3. Omni-Attention CUDA MMA kernel

Usage:
    python test_omni_attn.py                    # Run all tests
    python test_omni_attn.py --correctness      # Run correctness tests only
    python test_omni_attn.py --benchmark        # Run benchmarks only
    python test_omni_attn.py --profile          # Run with profiling
"""

import argparse
import time
import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
from functools import partial

import torch
import torch.nn.functional as F
from torch import randint, randn, tensor
from naive_attn import naive_attention

# Import from transfusion
import sys
import os
# Add parent directory to path to import transfusion_pytorch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from transfusion_pytorch.transfusion import (
    flex_attention,
    create_block_mask,
    transfusion_attn_mask,
    modality_positions_to_tensor,
    naive_attn_mask,
    exists,
)
# pad_sequence from transfusion is already batch_first=True
from torch.nn.utils.rnn import pad_sequence
pad_seq = partial(pad_sequence, batch_first=True)

# Constants
BLOCK_SIZE = 128

# ============================================================================
# Input Generation
# ============================================================================

def generate_random_input(
    B: int,
    H: int,
    N: int,
    D: int,
    vab_size: int = 512,
    min_text_len: int = 5,
    max_text_len: int = 64,
    min_image_size: int = 4,
    max_image_size: int = 48,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[List, torch.Tensor]:
    """
    Generate random text and image input similar to transfusion format.
    
    Args:
        N: Target sequence length (will generate sequences up to this length)
        D: Model dimension (H * head_dim) for embeddings
    
    Returns:
        text_and_images: List of B samples, each containing alternating text and images
        modality_positions: Tensor of shape [B, M, 3] where each entry is (modality_type, offset, length)
    """
    text_and_images = []
    all_modality_positions = []
    
    for b in range(B):
        batch_sample = []
        batch_modality_positions = []
        offset = 0
        current_length = 0
        
        # Randomly decide number of segments (text/image pairs)
        num_segments = torch.randint(2, 6, (1,)).item()
        
        # Calculate how to distribute N tokens across segments
        segments_with_images = num_segments - 1
        # Reserve some tokens for text (at least min_text_len per segment)
        min_text_total = num_segments * min_text_len
        # Remaining tokens go to images
        image_tokens_budget = max(0, N - min_text_total)
        
        for seg_idx in range(num_segments):
            # Add text - use reasonable size, save space for images
            remaining = N - current_length
            remaining_images = segments_with_images - seg_idx
            
            if remaining > 0 and seg_idx < num_segments - 1:
                # Reserve space for remaining images
                # Allocate text tokens: use remaining / (remaining_images + 1) for text
                text_alloc = remaining // (remaining_images + 1)
                text_len = max(min_text_len, min(max_text_len, text_alloc))
            elif remaining > 0:
                # Last segment - use all remaining for text
                text_len = min(max_text_len, max(min_text_len, remaining))
            else:
                text_len = torch.randint(min_text_len, max_text_len + 1, (1,)).item()
            
            text_tokens = randint(0, vab_size, (text_len,), device=device)
            batch_sample.append(text_tokens)
            offset += text_len
            current_length += text_len
            
            # Add image (modality) - generate directly in target dimension D
            if seg_idx < num_segments - 1:  # Don't add image after last text segment
                # Scale image size aggressively to reach target N
                remaining = N - current_length
                if remaining > 0:
                    # Distribute remaining tokens among remaining images
                    # Be aggressive: use most of remaining space for this image
                    if remaining_images > 1:
                        # Use 70% of remaining for this image, leave 30% for others
                        target_image_tokens = int(remaining * 0.7)
                    else:
                        # Last image, use all remaining
                        target_image_tokens = remaining
                    
                    # Clamp to max possible
                    target_image_tokens = min(target_image_tokens, max_image_size * max_image_size)
                    
                    # Find h, w such that h*w ≈ target_image_tokens
                    sqrt_tokens = int(target_image_tokens ** 0.5)
                    h_size = min(max_image_size, max(min_image_size, sqrt_tokens))
                    w_size = min(max_image_size, max(min_image_size, target_image_tokens // h_size))
                    
                    # If still too small and we have room, make it bigger
                    if h_size * w_size < target_image_tokens * 0.8 and remaining > target_image_tokens:
                        # Try to increase size
                        while h_size < max_image_size and w_size < max_image_size and h_size * w_size < target_image_tokens:
                            if h_size <= w_size:
                                h_size = min(max_image_size, h_size + 1)
                            else:
                                w_size = min(max_image_size, w_size + 1)
                    
                    # Ensure we don't exceed remaining
                    if h_size * w_size > remaining:
                        # Scale down to fit
                        scale = (remaining / (h_size * w_size)) ** 0.5
                        h_size = max(min_image_size, int(h_size * scale))
                        w_size = max(min_image_size, int(w_size * scale))
                        # Fine-tune to get as close as possible
                        while h_size * w_size < remaining and (h_size < max_image_size or w_size < max_image_size):
                            if h_size <= w_size and h_size < max_image_size:
                                h_size += 1
                            elif w_size < max_image_size:
                                w_size += 1
                            else:
                                break
                        # Final clamp
                        while h_size * w_size > remaining:
                            if h_size > w_size:
                                h_size -= 1
                            else:
                                w_size -= 1
                else:
                    h_size = torch.randint(min_image_size, max_image_size + 1, (1,)).item()
                    w_size = torch.randint(min_image_size, max_image_size + 1, (1,)).item()
                
                # Generate image as [H, W, D] where D is the embedding dimension
                image = randn(h_size, w_size, D, device=device)
                modality_length = h_size * w_size
                
                batch_sample.append(image)
                
                batch_modality_positions.append((0, offset, modality_length))
                offset += modality_length
                current_length += modality_length
        
        # Final check: if we're far from target, add more tokens to last image
        if current_length < N * 0.8 and len(batch_modality_positions) > 0:
            last_image_idx = len(batch_sample) - 1
            if last_image_idx >= 0 and batch_sample[last_image_idx].dtype != torch.long:
                old_image = batch_sample[last_image_idx]
                old_h, old_w = old_image.shape[0], old_image.shape[1]
                remaining = N - current_length
                if remaining > 0:
                    target_tokens = old_h * old_w + remaining
                    new_h = min(max_image_size, max(old_h, int(target_tokens ** 0.5)))
                    new_w = min(max_image_size, max(old_w, target_tokens // new_h))
                    if new_h * new_w > old_h * old_w:
                        batch_sample[last_image_idx] = randn(new_h, new_w, D, device=device)
                        last_modality = batch_modality_positions[-1]
                        batch_modality_positions[-1] = (last_modality[0], last_modality[1], new_h * new_w)
                        current_length = current_length - (old_h * old_w) + (new_h * new_w)
        
        text_and_images.append(batch_sample)
        all_modality_positions.append(batch_modality_positions)
    
    # Convert modality positions to tensor
    modality_positions = modality_positions_to_tensor(all_modality_positions, device=device)
    
    return text_and_images, modality_positions

# ============================================================================
# Embedding Generation
# ============================================================================

def get_embeddings(
    text_and_images: List,
    D: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    block_size: int = 128
) -> torch.Tensor:
    """
    Convert text_and_images to embeddings tensor [B, N, D].
    Simplified version - creates random embeddings for text and flattens images.
    
    Args:
        D: Model dimension (H * head_dim) for embeddings
        block_size: Block size for block-sparse attention
    """
    batch_embeddings = []
    
    for batch_sample in text_and_images:
        batch_tokens = []
        
        for item in batch_sample:
            if item.dtype == torch.long or item.dtype == torch.int:
                # Text tokens - create random embeddings
                text_len = item.shape[0]
                text_emb = randn(text_len, D, device=device)
                batch_tokens.append(text_emb)
            else:
                # Image - already in [H, W, D] format, just flatten to [H*W, D]
                h, w = item.shape[0], item.shape[1]
                image_emb = item.reshape(h * w, D)
                batch_tokens.append(image_emb)
        
        # Concatenate all tokens in sequence
        batch_seq = torch.cat(batch_tokens, dim=0)
        batch_embeddings.append(batch_seq)
    
    # Pad to max length
    x = pad_seq(batch_embeddings, batch_first=True, padding_value=0.0)
    
    # pad to multiple of block_size for block-sparse efficiency
    seq_len = x.shape[1]
    padded_len = ((seq_len + block_size - 1) // block_size) * block_size
    if padded_len > seq_len:
        padding = torch.zeros(x.shape[0], padded_len - seq_len, x.shape[2], device=device, dtype=x.dtype)
        x = torch.cat([x, padding], dim=1)
    
    return x

def omni_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    block_mask,
    scale: float = None
) -> torch.Tensor:
    """
    Placeholder for Omni-Attention CUDA kernel.
    Currently returns zeros - to be implemented.
    """
    B, H, N, D = Q.shape
    # Placeholder implementation
    return torch.zeros_like(Q)

# ============================================================================
# Block Mask Creation
# ============================================================================

def create_omni_block_mask(
    modality_positions: torch.Tensor,
    B: int,
    H: int,
    Q_LEN: int,
    KV_LEN: int,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    compile_mask: bool = False
):
    """
    Create block mask for omni attention using transfusion logic.
    
    Args:
        modality_positions: [B, M, 3] tensor of (modality_type, offset, length)
        B: batch size
        H: number of heads
        Q_LEN: query sequence length
        KV_LEN: key/value sequence length
        compile_mask: Whether to compile the mask (adds overhead, disable for benchmarks)
    
    Returns:
        block_mask: BlockMask object for flex_attention
    """
    # Use transfusion's transfusion_attn_mask function
    mask_fn = transfusion_attn_mask(modality_positions)
    
    # Create block mask using transfusion's create_block_mask
    block_mask = create_block_mask(
        mask_fn,
        B=B,
        H=H,
        Q_LEN=Q_LEN,
        KV_LEN=KV_LEN,
        _compile=compile_mask,
        device=device
    )
    
    return block_mask

# ============================================================================
# Correctness Checking
# ============================================================================

def check_correctness(
    ref_output: torch.Tensor,
    test_output: torch.Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-3,
    name: str = "test"
) -> bool:
    """
    Check if test output matches reference output within tolerance.
    """
    if torch.allclose(ref_output, test_output, rtol=rtol, atol=atol):
        print(f"✓ {name}: PASSED (rtol={rtol}, atol={atol})")
        return True
    else:
        max_diff = (ref_output - test_output).abs().max().item()
        mean_diff = (ref_output - test_output).abs().mean().item()
        print(f"✗ {name}: FAILED")
        print(f"  Max diff: {max_diff:.6f}, Mean diff: {mean_diff:.6f}")
        return False

# ============================================================================
# Main Test Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Omni-Attention Test Suite')
    parser.add_argument('--correctness', action='store_true', help='Run correctness tests only')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks only')
    parser.add_argument('--profile', action='store_true', help='Run with profiling')
    parser.add_argument('--B', type=int, default=2, help='Batch size')
    parser.add_argument('--H', type=int, default=8, help='Number of heads')
    parser.add_argument('--N', type=int, default=1024, help='Max sequence length')
    parser.add_argument('--D', type=int, default=512, help='Model dimension (H * head_dim) for embeddings')

    args = parser.parse_args()
    
    B, H, N, D = args.B, args.H, args.N, args.D
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Running tests with B={B}, H={H}, N={N}, D={D}")
    print(f"Device: {device}")
    print("-" * 60)
    
    # Generate random input
    print("Generating random input...")
    text_and_images, modality_positions = generate_random_input(
        B=B, H=H, N=N, D=D, device=device
    )
    
    # Get embeddings
    print("Creating embeddings...")
    x = get_embeddings(text_and_images, D=D, device=device, block_size=BLOCK_SIZE)
    actual_seq_len = x.shape[1]
    print(f"Sequence length: {actual_seq_len} (target: {N})")
    if actual_seq_len < N * 0.8:
        print(f"  ⚠ Warning: Sequence length is {actual_seq_len/N*100:.1f}% of target N={N}")
        print(f"    Consider increasing max_image_size or max_text_len parameters")
    
    # Create Q, K, V
    print("Creating Q, K, V...")
    # D is model_dim (H * head_dim), so head_dim = D // H
    head_dim = D // H
    assert D % H == 0, f"Model dimension D ({D}) must be divisible by number of heads H ({H})"
    
    # Standard attention: project from model_dim to model_dim, then split into heads
    W_Q = randn(D, D, device=device)  # [model_dim, model_dim]
    W_K = randn(D, D, device=device)
    W_V = randn(D, D, device=device)
    
    # Project x: [B, N, model_dim] -> [B, N, model_dim]
    x_proj_q = torch.matmul(x, W_Q)  # [B, N, model_dim]
    x_proj_k = torch.matmul(x, W_K)
    x_proj_v = torch.matmul(x, W_V)
    
    # Reshape to [B, H, N, head_dim]: [B, N, model_dim] -> [B, N, H, head_dim] -> [B, H, N, head_dim]
    Q = x_proj_q.view(B, actual_seq_len, H, head_dim).transpose(1, 2)  # [B, H, N, head_dim]
    K = x_proj_k.view(B, actual_seq_len, H, head_dim).transpose(1, 2)
    V = x_proj_v.view(B, actual_seq_len, H, head_dim).transpose(1, 2)
    
    # Create attention masks
    print("Creating attention masks...")
    naive_mask = naive_attn_mask(actual_seq_len, modality_positions, device=device)  # [B, N, N]
    
    # Create block mask for flex_attention
    if exists(flex_attention):
        print("Creating block mask for flex_attention...")
        # Disable compilation for block mask during benchmarks to avoid overhead
        # In production, you'd want compile_mask=True, but for testing it adds overhead
        compile_mask = args.correctness  # Only compile if running correctness tests
        block_mask = create_omni_block_mask(
            modality_positions, B, H, actual_seq_len, actual_seq_len, device, compile_mask=compile_mask
        )
        
        # Print sparsity info for debugging
        try:
            if hasattr(block_mask, 'kv_num_blocks') and hasattr(block_mask, 'kv_indices'):
                num_q_blocks = block_mask.kv_num_blocks.shape[-2]
                max_kv_blocks = block_mask.kv_num_blocks.shape[-1]
                total_possible = B * H * num_q_blocks * max_kv_blocks
                active_blocks = block_mask.kv_num_blocks.sum().item()
                sparsity = 1.0 - (active_blocks / total_possible) if total_possible > 0 else 0
                density = active_blocks / total_possible if total_possible > 0 else 0
                print(f"Block mask sparsity: {sparsity:.2%} (density: {density:.2%}, active: {active_blocks}/{total_possible})")
                if density > 0.5:
                    print(f"  ⚠ Warning: Block mask is too dense ({density:.1%}). Block-sparse attention may be slower than dense.")
        except Exception as e:
            print(f"  Could not compute sparsity: {e}")
    else:
        block_mask = None
        print("Warning: flex_attention not available")
    
    # Run correctness tests
    if not args.benchmark:
        print("\n" + "=" * 60)
        print("CORRECTNESS TESTS")
        print("=" * 60)
        
        # Naive attention (reference)
        print("\nRunning naive_attention...")
        naive_output = naive_attention(Q, K, V, naive_mask)
        
        # Flex attention
        if exists(flex_attention) and block_mask is not None:
            print("Running flex_attention...")
            try:
                flex_output = flex_attention(Q, K, V, block_mask=block_mask)
                # Use relaxed tolerance for block-sparse attention due to block-level approximations
                check_correctness(naive_output, flex_output, rtol=1e-2, atol=1e-3, name="flex_attention")
            except Exception as e:
                print(f"✗ flex_attention: FAILED with error: {e}")
        else:
            print("Skipping flex_attention (not available)")
        
        # Omni attention (placeholder)
        print("Running omni_attention (placeholder)...")
        omni_output = omni_attention(Q, K, V, block_mask)
        print("⚠ omni_attention: PLACEHOLDER (returns zeros)")
    
    # Run benchmarks
    if not args.correctness:
        print("\n" + "=" * 60)
        print("BENCHMARKS")
        print("=" * 60)
        
        num_iterations = 10
        warmup = 10  # Increased warmup for compiled code
        
        # Warmup naive
        for _ in range(warmup):
            _ = naive_attention(Q, K, V, naive_mask)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Benchmark naive
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(num_iterations):
                _ = naive_attention(Q, K, V, naive_mask)
            end_event.record()
            torch.cuda.synchronize()
            naive_time = start_event.elapsed_time(end_event) / num_iterations  # ms
        else:
            start = time.time()
            for _ in range(num_iterations):
                _ = naive_attention(Q, K, V, naive_mask)
            naive_time = (time.time() - start) / num_iterations * 1000  # ms
        
        print(f"\nnaive_attention: {naive_time:.2f} ms/iter")
        
        # Benchmark flex_attention
        if exists(flex_attention) and block_mask is not None:
            # Warmup - enough to ensure compilation but not excessive
            for _ in range(warmup):
                _ = flex_attention(Q, K, V, block_mask=block_mask)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Benchmark with same timing method as naive for fair comparison
            if torch.cuda.is_available():
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                
                start_event.record()
                for _ in range(num_iterations):
                    _ = flex_attention(Q, K, V, block_mask=block_mask)
                end_event.record()
                torch.cuda.synchronize()
                flex_time = start_event.elapsed_time(end_event) / num_iterations  # ms
            else:
                start = time.time()
                for _ in range(num_iterations):
                    _ = flex_attention(Q, K, V, block_mask=block_mask)
                flex_time = (time.time() - start) / num_iterations * 1000  # ms
            
            print(f"flex_attention: {flex_time:.2f} ms/iter")
            print(f"Speedup: {naive_time/flex_time:.2f}x")
            
            if flex_time > naive_time * 1.5:
                print(f"\n⚠ Warning: flex_attention is {flex_time/naive_time:.1f}x slower than naive_attention")
                print(f"  This is expected if block mask density >50% (see sparsity info above)")
                print(f"  Block-sparse attention only helps with high sparsity (<30% active blocks)")
                print(f"  For dense masks, use naive_attention instead")
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
