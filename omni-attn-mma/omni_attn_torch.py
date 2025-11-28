"""
Omni-Attention: Combining FlashAttention-2's split-Q strategy with FlexAttention's block_mask.

Key features:
- Uses FlashAttention-2's efficient split-Q tiling (BLOCK_M x BLOCK_N)
- Supports block-sparse attention via BlockMask (from FlexAttention)
- Per-block mask types: FULL (no mask), CAUSAL (causal mask), MASKED (skip block)
- Efficient online softmax with tiling to avoid materializing full S matrix
"""

import math
from enum import IntEnum
from typing import Optional, Tuple

import torch

# Try to import triton, but make it optional
try:
    import triton
    import triton.language as tl
    # TRITON_AVAILABLE = True
except ImportError:
    triton = None
    tl = None

TRITON_AVAILABLE = False

class BlockMaskType(IntEnum):
    """Mask type for each block in the attention matrix."""
    MASKED = 0  # Block is fully masked out (skip computation)
    CAUSAL = 1  # Block uses causal masking
    FULL = 2    # Block has no masking (full attention)


def omni_attention_forward_pytorch(
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
    Pure PyTorch tiled implementation of Omni-Attention (no Triton required).
    
    Uses the same tiling strategy as the Triton version:
    - Splits Q into blocks of BLOCK_M
    - Iterates over KV blocks specified by BlockMask
    - Uses online softmax to avoid materializing full attention matrix
    
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
    
    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    num_q_blocks = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    
    # Initialize output and LSE
    o = torch.zeros_like(q)
    lse = torch.full((batch, nheads, seqlen_q), float("-inf"), device=q.device, dtype=torch.float32)
    
    # Process each Q block
    for q_block_idx in range(num_q_blocks):
        q_start = q_block_idx * BLOCK_M
        q_end = min(q_start + BLOCK_M, seqlen_q)
        q_block_size = q_end - q_start
        
        # Load Q block: [batch, nheads, BLOCK_M, headdim]
        q_block = q[:, q_start:q_end, :, :]  # [batch, q_block_size, nheads, headdim]
        q_block = q_block.transpose(1, 2)  # [batch, nheads, q_block_size, headdim]
        
        # Initialize accumulators for online softmax
        # Use float32 for numerical stability
        m_i = torch.full((batch, nheads, q_block_size), float("-inf"), device=q.device, dtype=torch.float32)
        l_i = torch.zeros((batch, nheads, q_block_size), device=q.device, dtype=torch.float32)
        acc_o = torch.zeros((batch, nheads, q_block_size, d), device=q.device, dtype=torch.float32)
        
        # Get number of KV blocks for this Q block
        num_active_blocks = kv_num_blocks[:, :, q_block_idx]  # [batch, nheads]
        
        # Process each active KV block
        for b in range(batch):
            for h in range(nheads):
                num_kv_blocks = num_active_blocks[b, h].item()
                
                for kv_block_idx in range(num_kv_blocks):
                    # Get KV block column index
                    kv_col_idx = kv_indices[b, h, q_block_idx, kv_block_idx].item()
                    mask_type = block_mask_types[b, h, q_block_idx, kv_block_idx].item()
                    
                    # Skip if fully masked
                    if mask_type == BlockMaskType.MASKED:
                        continue
                    
                    kv_start = kv_col_idx * BLOCK_N
                    kv_end = min(kv_start + BLOCK_N, seqlen_k)
                    kv_block_size = kv_end - kv_start
                    
                    # Load K, V blocks: [kv_block_size, headdim]
                    k_block = k[b, kv_start:kv_end, h, :]  # [kv_block_size, headdim]
                    v_block = v[b, kv_start:kv_end, h, :]  # [kv_block_size, headdim]
                    
                    # Compute QK^T: [q_block_size, kv_block_size]
                    # Cast to float32 for numerical stability
                    q_block_bh = q_block[b, h, :, :].float()  # [q_block_size, headdim]
                    k_block_f = k_block.float()  # [kv_block_size, headdim]
                    qk = torch.matmul(q_block_bh, k_block_f.t()) * softmax_scale  # [q_block_size, kv_block_size]
                    
                    # Apply masking based on block type
                    if mask_type == BlockMaskType.CAUSAL:
                        # Causal mask: q_idx >= kv_idx (using global indices)
                        q_indices = torch.arange(q_start, q_end, device=q.device)
                        kv_indices_block = torch.arange(kv_start, kv_end, device=k.device)
                        q_grid, kv_grid = torch.meshgrid(q_indices, kv_indices_block, indexing="ij")
                        causal_mask = q_grid >= kv_grid
                        qk = torch.where(causal_mask, qk, float("-inf"))
                    # FULL type: no masking needed
                    
                    # Online softmax update
                    # m_ij = max(max(qk), m_i) - new maximum
                    m_ij = torch.maximum(torch.max(qk, dim=1)[0], m_i[b, h, :])  # [q_block_size]
                    
                    # Compute probabilities with new maximum
                    p = torch.exp(qk - m_ij[:, None])  # [q_block_size, kv_block_size]
                    l_ij = torch.sum(p, dim=1)  # [q_block_size] - sum of exp(qk - m_ij)
                    
                    # Scale accumulator: exp(m_i - m_ij) * acc_o
                    # When m_i = -inf, exp(-inf - m_ij) = 0, which is correct
                    acc_o_scale = torch.exp(m_i[b, h, :] - m_ij)  # [q_block_size]
                    acc_o[b, h, :, :] = acc_o[b, h, :, :] * acc_o_scale[:, None]
                    
                    # Update output accumulator
                    # Keep computation in float32 for numerical stability
                    v_block_f = v_block.float()
                    acc_o_update = torch.matmul(p, v_block_f)  # [q_block_size, headdim]
                    acc_o[b, h, :, :] = acc_o[b, h, :, :] + acc_o_update
                    
                    # Update statistics: l_i_new = exp(m_i - m_ij) * l_i + l_ij
                    # When m_i = -inf, exp(-inf - m_ij) = 0, so l_i_new = 0 * l_i + l_ij = l_ij
                    l_i_new = torch.exp(m_i[b, h, :] - m_ij) * l_i[b, h, :] + l_ij
                    
                    m_i[b, h, :] = m_ij
                    l_i[b, h, :] = l_i_new
        
        # Compute final LSE and scaling
        # lse = m_i + log(l_i), but handle case where l_i might be 0 (all masked)
        # Use log(l_i + eps) to avoid log(0), but if l_i is truly 0, lse should be -inf
        l_i_safe = torch.clamp(l_i, min=1e-10)  # Avoid log(0), but preserve -inf for truly zero
        lse_block = m_i + torch.log(l_i_safe)  # [batch, nheads, q_block_size]
        # If l_i was 0, set lse to -inf
        lse_block = torch.where(l_i > 0, lse_block, torch.full_like(lse_block, float("-inf")))
        lse[:, :, q_start:q_end] = lse_block
        
        # Final scaling: o_scale = exp(m_i - lse) = 1/l_i
        # Handle case where l_i is 0
        o_scale = torch.where(
            l_i > 0,
            torch.exp(m_i - lse_block),
            torch.zeros_like(l_i)
        )  # [batch, nheads, q_block_size]
        acc_o = acc_o * o_scale[:, :, :, None]
        
        # Write back output (cast back to original dtype)
        o[:, q_start:q_end, :, :] = acc_o.transpose(1, 2).to(q.dtype)  # [batch, q_block_size, nheads, headdim]
    
    return o, lse


def omni_attention_forward(
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
    Forward pass for Omni-Attention.
    
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
        lse: [batch, nheads, seqlen_q_rounded] - log-sum-exp for numerical stability
    """
    # Use PyTorch implementation if Triton is not available
    if not TRITON_AVAILABLE:
        return omni_attention_forward_pytorch(
            q, k, v, kv_num_blocks, kv_indices, block_mask_types, softmax_scale, BLOCK_M, BLOCK_N
        )
    
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert d <= 128, "Omni-Attention supports head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype in [torch.float16, torch.bfloat16]
    assert q.is_cuda and k.is_cuda and v.is_cuda
    
    softmax_scale = softmax_scale or (1.0 / math.sqrt(d))
    
    # Ensure contiguous
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    
    seqlen_q_rounded = math.ceil(seqlen_q / BLOCK_M) * BLOCK_M
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)
    
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    num_warps = 4 if d <= 64 else 8
    
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    
    _omni_attn_fwd_kernel[grid](
        q,
        k,
        v,
        o,
        lse,
        tmp,
        kv_num_blocks,
        kv_indices,
        block_mask_types,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        o.stride(0),
        o.stride(2),
        o.stride(1),
        # BlockMask strides
        kv_num_blocks.stride(0),
        kv_num_blocks.stride(1),
        kv_num_blocks.stride(2),
        kv_indices.stride(0),
        kv_indices.stride(1),
        kv_indices.stride(2),
        kv_indices.stride(3),
        block_mask_types.stride(0),
        block_mask_types.stride(1),
        block_mask_types.stride(2),
        block_mask_types.stride(3),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        num_warps=num_warps,
        num_stages=1,
    )
    
    return o, lse


def create_omni_block_mask(
    batch: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    block_mask_pattern: torch.Tensor,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create block mask tensors for Omni-Attention.
    
    Args:
        batch: Batch size
        nheads: Number of heads
        seqlen_q: Query sequence length
        seqlen_k: Key sequence length
        block_mask_pattern: [num_q_blocks, num_kv_blocks] tensor with BlockMaskType values
                           or a callable that takes (q_block_idx, kv_block_idx) -> BlockMaskType
        BLOCK_M: Q block size
        BLOCK_N: KV block size
        device: Device to create tensors on
    
    Returns:
        kv_num_blocks: [batch, nheads, num_q_blocks] - number of active blocks per Q block
        kv_indices: [batch, nheads, num_q_blocks, max_blocks] - KV block column indices
        block_mask_types: [batch, nheads, num_q_blocks, max_blocks] - mask type per block
    """
    if device is None:
        device = torch.device("cuda")
    
    num_q_blocks = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (seqlen_k + BLOCK_N - 1) // BLOCK_N
    
    # If pattern is a tensor, use it directly
    if isinstance(block_mask_pattern, torch.Tensor):
        assert block_mask_pattern.shape == (num_q_blocks, num_kv_blocks)
        pattern = block_mask_pattern
    else:
        # If it's a callable, build the pattern
        pattern = torch.zeros((num_q_blocks, num_kv_blocks), dtype=torch.int32, device=device)
        for q_idx in range(num_q_blocks):
            for kv_idx in range(num_kv_blocks):
                pattern[q_idx, kv_idx] = block_mask_pattern(q_idx, kv_idx)
    
    # Count active blocks per Q block (non-MASKED blocks)
    kv_num_blocks = (pattern != BlockMaskType.MASKED).sum(dim=1)  # [num_q_blocks]
    max_blocks = kv_num_blocks.max().item()
    
    # Create indices and types tensors
    kv_indices = torch.zeros(
        (batch, nheads, num_q_blocks, max_blocks),
        dtype=torch.int32,
        device=device
    )
    block_mask_types = torch.zeros(
        (batch, nheads, num_q_blocks, max_blocks),
        dtype=torch.int32,
        device=device
    )
    
    # Fill in indices and types
    for q_idx in range(num_q_blocks):
        active_kv_blocks = (pattern[q_idx] != BlockMaskType.MASKED).nonzero(as_tuple=True)[0]
        num_active = len(active_kv_blocks)
        
        for b in range(batch):
            for h in range(nheads):
                for i, kv_idx in enumerate(active_kv_blocks):
                    kv_indices[b, h, q_idx, i] = kv_idx.item()
                    block_mask_types[b, h, q_idx, i] = pattern[q_idx, kv_idx].item()
    
    # Expand kv_num_blocks to [batch, nheads, num_q_blocks]
    kv_num_blocks = kv_num_blocks.unsqueeze(0).unsqueeze(0).expand(batch, nheads, -1)
    
    return kv_num_blocks, kv_indices, block_mask_types


# Example usage patterns
def create_causal_block_mask(
    batch: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a causal block mask (lower triangular pattern)."""
    num_q_blocks = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (seqlen_k + BLOCK_N - 1) // BLOCK_N
    
    def causal_pattern(q_idx, kv_idx):
        # Causal: q_block can attend to kv_blocks where kv_block <= q_block
        if kv_idx <= q_idx:
            return BlockMaskType.CAUSAL
        else:
            return BlockMaskType.MASKED
    
    return create_omni_block_mask(
        batch, nheads, seqlen_q, seqlen_k, causal_pattern, BLOCK_M, BLOCK_N, device
    )


def create_full_block_mask(
    batch: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a full attention block mask (all blocks are FULL type)."""
    num_q_blocks = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (seqlen_k + BLOCK_N - 1) // BLOCK_N
    
    pattern = torch.full(
        (num_q_blocks, num_kv_blocks),
        BlockMaskType.FULL,
        dtype=torch.int32,
        device=device or torch.device("cuda")
    )
    
    return create_omni_block_mask(
        batch, nheads, seqlen_q, seqlen_k, pattern, BLOCK_M, BLOCK_N, device
    )


def create_hybrid_block_mask(
    batch: int,
    nheads: int,
    seqlen_q: int,
    seqlen_k: int,
    prefix_len: int,
    BLOCK_M: int = 128,
    BLOCK_N: int = 128,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a hybrid block mask: full attention for prefix, causal for rest.
    
    Useful for prefix-LM or retrieval-augmented generation.
    """
    num_q_blocks = (seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (seqlen_k + BLOCK_N - 1) // BLOCK_N
    prefix_q_blocks = (prefix_len + BLOCK_M - 1) // BLOCK_M
    prefix_kv_blocks = (prefix_len + BLOCK_N - 1) // BLOCK_N
    
    def hybrid_pattern(q_idx, kv_idx):
        # Prefix region: full attention
        if q_idx < prefix_q_blocks and kv_idx < prefix_kv_blocks:
            return BlockMaskType.FULL
        # After prefix: causal
        elif kv_idx <= q_idx:
            return BlockMaskType.CAUSAL
        else:
            return BlockMaskType.MASKED
    
    return create_omni_block_mask(
        batch, nheads, seqlen_q, seqlen_k, hybrid_pattern, BLOCK_M, BLOCK_N, device
    )

