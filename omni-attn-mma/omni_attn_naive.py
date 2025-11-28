"""
Naive PyTorch reference implementation for Omni-Attention.

This is a baseline implementation that materializes the full attention matrix.
Used for verifying the correctness of the Triton implementation.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

# Import BlockMaskType - handle both direct import and relative import
try:
    from omni_attn_torch import BlockMaskType
except ImportError:
    # If running as script, use relative import
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from omni_attn_torch import BlockMaskType


def omni_attention_forward_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_num_blocks: torch.Tensor,
    kv_indices: torch.Tensor,
    block_mask_types: torch.Tensor,
    softmax_scale: Optional[float] = None,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Naive PyTorch reference implementation for Omni-Attention.
    
    This implementation:
    - Materializes the full attention matrix S = Q @ K^T
    - Applies block-wise masking based on block_mask_types
    - Computes softmax and attention output
    
    Args:
        q: Query tensor [batch, seqlen_q, nheads, headdim]
        k: Key tensor [batch, seqlen_k, nheads, headdim]
        v: Value tensor [batch, seqlen_k, nheads, headdim]
        kv_num_blocks: [batch, nheads, num_q_blocks] - number of active KV blocks per Q block
        kv_indices: [batch, nheads, num_q_blocks, max_blocks] - which KV block columns to process
        block_mask_types: [batch, nheads, num_q_blocks, max_blocks] - mask type per block
        softmax_scale: Scaling factor (default: 1/sqrt(headdim))
        BLOCK_M: Q block size (default: 128)
        BLOCK_N: KV block size (default: 128)
    
    Returns:
        output: [batch, seqlen_q, nheads, headdim]
        lse: [batch, nheads, seqlen_q] - log-sum-exp for numerical stability
    """
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    
    softmax_scale = softmax_scale or (1.0 / math.sqrt(d))
    
    # Compute attention scores: [batch, nheads, seqlen_q, seqlen_k]
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * softmax_scale
    
    # Create block mask matrix: [batch, nheads, seqlen_q, seqlen_k]
    block_mask = torch.full(
        (batch, nheads, seqlen_q, seqlen_k),
        float("-inf"),
        dtype=scores.dtype,
        device=scores.device
    )
    
    num_q_blocks = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (seqlen_k + BLOCK_N - 1) // BLOCK_N
    
    # Build block mask from block_mask_types
    for b in range(batch):
        for h in range(nheads):
            for q_block_idx in range(num_q_blocks):
                num_active = kv_num_blocks[b, h, q_block_idx].item()
                
                q_start = q_block_idx * BLOCK_M
                q_end = min(q_start + BLOCK_M, seqlen_q)
                
                for kv_block_idx in range(num_active):
                    kv_col_idx = kv_indices[b, h, q_block_idx, kv_block_idx].item()
                    mask_type = block_mask_types[b, h, q_block_idx, kv_block_idx].item()
                    
                    kv_start = kv_col_idx * BLOCK_N
                    kv_end = min(kv_start + BLOCK_N, seqlen_k)
                    
                    # Apply mask based on type
                    if mask_type == BlockMaskType.MASKED:
                        # Fully masked - already set to -inf, skip
                        continue
                    elif mask_type == BlockMaskType.CAUSAL:
                        # Causal mask: q_idx >= kv_idx (using global indices)
                        q_indices = torch.arange(q_start, q_end, device=q.device)
                        kv_indices_block = torch.arange(kv_start, kv_end, device=k.device)
                        # Create meshgrid for this block with global indices
                        q_grid, kv_grid = torch.meshgrid(
                            q_indices, kv_indices_block, indexing="ij"
                        )
                        # Causal: q_idx >= kv_idx
                        causal_mask = q_grid >= kv_grid
                        block_mask[b, h, q_start:q_end, kv_start:kv_end] = torch.where(
                            causal_mask,
                            0.0,
                            float("-inf")
                        )
                    elif mask_type == BlockMaskType.FULL:
                        # Full attention - no masking (set to 0.0)
                        block_mask[b, h, q_start:q_end, kv_start:kv_end] = 0.0
    
    # Apply block mask to scores
    scores = scores + block_mask
    
    # Compute softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Compute output: [batch, nheads, seqlen_q, headdim]
    output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)
    
    # Compute LSE for numerical stability check
    lse = torch.logsumexp(scores, dim=-1)  # [batch, nheads, seqlen_q]
    
    return output, lse


def create_block_mask_pattern_naive(
    seqlen_q: int,
    seqlen_k: int,
    pattern_fn,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Create a dense block mask pattern for naive implementation.
    
    Args:
        seqlen_q: Query sequence length
        seqlen_k: Key sequence length
        pattern_fn: Function (q_block_idx, kv_block_idx) -> BlockMaskType
        BLOCK_M: Q block size
        BLOCK_N: KV block size
        device: Device to create tensor on
    
    Returns:
        block_mask: [seqlen_q, seqlen_k] tensor with mask values
                   0.0 for allowed, -inf for masked
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    num_q_blocks = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (seqlen_k + BLOCK_N - 1) // BLOCK_N
    
    block_mask = torch.full(
        (seqlen_q, seqlen_k),
        float("-inf"),
        dtype=torch.float32,
        device=device
    )
    
    for q_block_idx in range(num_q_blocks):
        q_start = q_block_idx * BLOCK_M
        q_end = min(q_start + BLOCK_M, seqlen_q)
        
        for kv_block_idx in range(num_kv_blocks):
            kv_start = kv_block_idx * BLOCK_N
            kv_end = min(kv_start + BLOCK_N, seqlen_k)
            
            mask_type = pattern_fn(q_block_idx, kv_block_idx)
            
            if mask_type == BlockMaskType.MASKED:
                continue  # Already -inf
            elif mask_type == BlockMaskType.CAUSAL:
                # Apply causal mask within block
                q_indices = torch.arange(q_start, q_end, device=device)
                kv_indices = torch.arange(kv_start, kv_end, device=device)
                q_grid, kv_grid = torch.meshgrid(q_indices, kv_indices, indexing="ij")
                causal_mask = q_grid >= kv_grid
                block_mask[q_start:q_end, kv_start:kv_end] = torch.where(
                    causal_mask,
                    0.0,
                    float("-inf")
                )
            elif mask_type == BlockMaskType.FULL:
                # Full attention
                block_mask[q_start:q_end, kv_start:kv_end] = 0.0
    
    return block_mask


def omni_attention_forward_naive_simple(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simplified naive implementation using a dense block mask.
    
    Args:
        q: Query tensor [batch, seqlen_q, nheads, headdim]
        k: Key tensor [batch, seqlen_k, nheads, headdim]
        v: Value tensor [batch, seqlen_k, nheads, headdim]
        block_mask: [seqlen_q, seqlen_k] mask (0.0 for allowed, -inf for masked)
        softmax_scale: Scaling factor (default: 1/sqrt(headdim))
    
    Returns:
        output: [batch, seqlen_q, nheads, headdim]
        lse: [batch, nheads, seqlen_q]
    """
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    
    softmax_scale = softmax_scale or (1.0 / math.sqrt(d))
    
    # Compute attention scores: [batch, nheads, seqlen_q, seqlen_k]
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * softmax_scale
    
    # Expand block mask: [seqlen_q, seqlen_k] -> [batch, nheads, seqlen_q, seqlen_k]
    block_mask_expanded = block_mask.unsqueeze(0).unsqueeze(0).expand(batch, nheads, -1, -1)
    
    # Apply mask
    scores = scores + block_mask_expanded
    
    # Compute softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Compute output
    output = torch.einsum("bhqk,bkhd->bqhd", attn_weights, v)
    
    # Compute LSE
    lse = torch.logsumexp(scores, dim=-1)  # [batch, nheads, seqlen_q]
    
    return output, lse

