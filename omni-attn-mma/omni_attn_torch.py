"""
Omni-Attention: Combining FlashAttention-2's split-Q strategy with FlexAttention's block_mask.

Key features:
- Uses FlashAttention-2's efficient split-Q tiling (BLOCK_M x BLOCK_N)
- Supports block-sparse attention via BlockMask (from FlexAttention)
- Per-block mask types: FULL (no mask), CAUSAL (causal mask), MASKED (skip block)
- Efficient online softmax with tiling to avoid materializing full S matrix

This module provides:
1. Block mask creation utilities for VLM-style interleaved patterns
2. Naive PyTorch reference implementation
3. FlexAttention wrapper with block mask support
"""

import math
from dataclasses import dataclass
from enum import IntEnum
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, randint, randn


# ============================================================================
# Block Mask Types
# ============================================================================

class BlockMaskType(IntEnum):
    """Block mask types for Omni-Attention.
    
    MASKED (0): Block is fully masked out - skip computation entirely
    CAUSAL (1): Block uses causal masking (q_idx >= kv_idx)
    FULL (2): Block has no masking - full attention
    PARTIAL (3): Block is partially masked (cross-modality or mixed content)
    """
    MASKED = 0  # Skip block entirely (don't load K/V)
    CAUSAL = 1  # Apply causal masking within block
    FULL = 2    # No masking (dense attention)
    PARTIAL = 3  # Partially masked (requires per-token mask check)


# ============================================================================
# Block Mask Data Structure
# ============================================================================

@dataclass
class OmniBlockMask:
    """Block mask for Omni-Attention.
    
    Attributes:
        kv_num_blocks: [batch, nheads, num_q_blocks] - number of active KV blocks per Q block
        kv_indices: [batch, nheads, num_q_blocks, max_blocks] - which KV block columns to process
        block_mask_types: [batch, nheads, num_q_blocks, max_blocks] - mask type per block
        q_len: Original query sequence length
        kv_len: Original key/value sequence length
        Q_BLOCK_SIZE: Block size for query (default 128)
        KV_BLOCK_SIZE: Block size for key/value (default 128)
    """
    kv_num_blocks: Tensor  # [B, H, num_q_blocks]
    kv_indices: Tensor     # [B, H, num_q_blocks, max_blocks]
    block_mask_types: Tensor  # [B, H, num_q_blocks, max_blocks]
    q_len: int
    kv_len: int
    Q_BLOCK_SIZE: int = 128
    KV_BLOCK_SIZE: int = 128
    partial_block_mask_indices: Optional[Tensor] = None  # [B, H, num_q_blocks, max_blocks] or None
    partial_block_masks: Optional[Tensor] = None  # [num_partial_blocks, Q_BLOCK_SIZE, KV_BLOCK_SIZE] or None
    
    @property
    def num_q_blocks(self) -> int:
        return (self.q_len + self.Q_BLOCK_SIZE - 1) // self.Q_BLOCK_SIZE
    
    @property
    def num_kv_blocks(self) -> int:
        return (self.kv_len + self.KV_BLOCK_SIZE - 1) // self.KV_BLOCK_SIZE
    
    @property
    def device(self) -> torch.device:
        return self.kv_num_blocks.device
    
    @property
    def shape(self) -> Tuple[int, int, int, int]:
        """Returns (batch, nheads, q_len, kv_len)."""
        B, H, _ = self.kv_num_blocks.shape
        return (B, H, self.q_len, self.kv_len)
    
    def to_dense_mask(self) -> Tensor:
        """Convert block mask to dense boolean mask [B, H, q_len, kv_len].
        
        Returns:
            mask: Boolean tensor where True = attend, False = masked out
        """
        B, H, num_q_blocks, max_blocks = self.kv_indices.shape
        device = self.device
        
        # Create dense mask
        dense_mask = torch.zeros(B, H, self.q_len, self.kv_len, dtype=torch.bool, device=device)
        
        for b in range(B):
            for h in range(H):
                for q_block in range(num_q_blocks):
                    q_start = q_block * self.Q_BLOCK_SIZE
                    q_end = min(q_start + self.Q_BLOCK_SIZE, self.q_len)
                    
                    num_active = self.kv_num_blocks[b, h, q_block].item()
                    
                    for idx in range(num_active):
                        kv_block = self.kv_indices[b, h, q_block, idx].item()
                        mask_type = self.block_mask_types[b, h, q_block, idx].item()
                        
                        kv_start = kv_block * self.KV_BLOCK_SIZE
                        kv_end = min(kv_start + self.KV_BLOCK_SIZE, self.kv_len)
                        
                        if mask_type == BlockMaskType.MASKED:
                            # Skip - already zeros
                            pass
                        elif mask_type == BlockMaskType.FULL:
                            # Full attention
                            dense_mask[b, h, q_start:q_end, kv_start:kv_end] = True
                        elif mask_type == BlockMaskType.CAUSAL:
                            # Causal mask within block
                            for qi in range(q_start, q_end):
                                for ki in range(kv_start, min(qi + 1, kv_end)):
                                    dense_mask[b, h, qi, ki] = True
                        elif mask_type == BlockMaskType.PARTIAL:
                            # Use stored per-block masks when available
                            if (
                                self.partial_block_mask_indices is not None
                                and self.partial_block_masks is not None
                            ):
                                partial_idx = self.partial_block_mask_indices[b, h, q_block, idx].item()
                                if partial_idx >= 0 and partial_idx < self.partial_block_masks.shape[0]:
                                    block = self.partial_block_masks[partial_idx]
                                    qq = q_end - q_start
                                    kk = kv_end - kv_start
                                    dense_mask[b, h, q_start:q_end, kv_start:kv_end] = block[:qq, :kk]
                                    continue
                            # Fallback: treat as causal within block
                            for qi in range(q_start, q_end):
                                for ki in range(kv_start, min(qi + 1, kv_end)):
                                    dense_mask[b, h, qi, ki] = True
        
        return dense_mask


def build_partial_block_data(
    block_mask: OmniBlockMask,
    dense_mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Build per-block bitmasks for PARTIAL blocks from a dense mask."""
    B, H, num_q_blocks, max_blocks = block_mask.block_mask_types.shape
    device = dense_mask.device
    Q_BLOCK_SIZE = block_mask.Q_BLOCK_SIZE
    KV_BLOCK_SIZE = block_mask.KV_BLOCK_SIZE
    
    # Ensure dense_mask matches padded lengths
    if dense_mask.shape[2] != block_mask.q_len or dense_mask.shape[3] != block_mask.kv_len:
        padded = torch.zeros(
            B, H, block_mask.q_len, block_mask.kv_len, dtype=dense_mask.dtype, device=device
        )
        q_len = min(dense_mask.shape[2], block_mask.q_len)
        kv_len = min(dense_mask.shape[3], block_mask.kv_len)
        padded[:, :, :q_len, :kv_len] = dense_mask[:, :, :q_len, :kv_len]
        dense_mask = padded
    
    partial_indices = torch.full(
        (B, H, num_q_blocks, max_blocks), -1, dtype=torch.int32, device=device
    )
    partial_masks: List[Tensor] = []
    
    for b in range(B):
        for h in range(H):
            for q_block in range(num_q_blocks):
                num_active = block_mask.kv_num_blocks[b, h, q_block].item()
                q_start = q_block * Q_BLOCK_SIZE
                q_end = min(q_start + Q_BLOCK_SIZE, block_mask.q_len)
                for idx in range(num_active):
                    mask_type = block_mask.block_mask_types[b, h, q_block, idx].item()
                    if mask_type != BlockMaskType.PARTIAL:
                        continue
                    kv_block = block_mask.kv_indices[b, h, q_block, idx].item()
                    kv_start = kv_block * KV_BLOCK_SIZE
                    kv_end = min(kv_start + KV_BLOCK_SIZE, block_mask.kv_len)
                    
                    block = torch.zeros(
                        Q_BLOCK_SIZE, KV_BLOCK_SIZE, dtype=torch.bool, device=device
                    )
                    q_len_block = q_end - q_start
                    kv_len_block = kv_end - kv_start
                    block[:q_len_block, :kv_len_block] = dense_mask[
                        b, h, q_start:q_end, kv_start:kv_end
                    ]
                    partial_indices[b, h, q_block, idx] = len(partial_masks)
                    partial_masks.append(block)
    
    if partial_masks:
        partial_mask_tensor = torch.stack(partial_masks, dim=0)
    else:
        partial_mask_tensor = torch.empty(
            0, Q_BLOCK_SIZE, KV_BLOCK_SIZE, dtype=torch.bool, device=device
        )
    
    return partial_indices, partial_mask_tensor
    
def to_score_mask(self) -> Tensor:
    """Convert to score mask with -inf for masked positions.
    
    Returns:
        mask: Float tensor with 0 for attend, -inf for masked
    """
    bool_mask = self.to_dense_mask()
    score_mask = torch.where(bool_mask, 0.0, float('-inf'))
    return score_mask.to(torch.float32)

def sparsity(self) -> float:
    """Compute the sparsity ratio (percentage of skipped computation)."""
    total_blocks = self.num_q_blocks * self.num_kv_blocks
    active_blocks = self.kv_num_blocks.float().sum().item()
    B, H = self.kv_num_blocks.shape[:2]
    avg_active = active_blocks / (B * H)
    return 100.0 * (1.0 - avg_active / total_blocks)


def omni_block_mask_from_flex_block_mask(flex_block_mask) -> OmniBlockMask:
    """Convert flex_attention's BlockMask to OmniBlockMask.
    
    This is useful for debugging - allows using flex_attention's block_mask
    (which passes correctness) with CUDA kernels to isolate bugs.
    
    Args:
        flex_block_mask: flex_attention BlockMask object
        
    Returns:
        OmniBlockMask with same kv_num_blocks and kv_indices as flex_block_mask
    """
    # Get dimensions
    q_len, kv_len = flex_block_mask.seq_lengths
    
    q_block_size = flex_block_mask.Q_BLOCK_SIZE
    kv_block_size = flex_block_mask.KV_BLOCK_SIZE
    
    # Get kv_num_blocks and kv_indices
    kv_num_blocks = flex_block_mask.kv_num_blocks.clone()
    kv_indices = flex_block_mask.kv_indices.clone()
    
    # Ensure correct shape [B, H, num_q_blocks] for kv_num_blocks
    # and [B, H, num_q_blocks, max_blocks] for kv_indices
    if kv_num_blocks.dim() == 1:
        # Shape is [num_q_blocks], need to add B and H dimensions
        # Assume B=1, H=1 (will need to expand later)
        kv_num_blocks = kv_num_blocks.unsqueeze(0).unsqueeze(0)  # [1, 1, num_q_blocks]
        kv_indices = kv_indices.unsqueeze(0).unsqueeze(0)  # [1, 1, num_q_blocks, max_blocks]
    elif kv_num_blocks.dim() == 2:
        # Shape is [H, num_q_blocks], add B dimension
        kv_num_blocks = kv_num_blocks.unsqueeze(0)  # [1, H, num_q_blocks]
        kv_indices = kv_indices.unsqueeze(0)  # [1, H, num_q_blocks, max_blocks]
    # else: already [B, H, num_q_blocks] or [B, H, num_q_blocks, max_blocks]
    
    B, H, num_q_blocks = kv_num_blocks.shape
    _, _, _, max_blocks = kv_indices.shape
    
    # Create block_mask_types - set all to PARTIAL since flex_attention uses mask_mod
    # for per-token masking (we don't know the exact mask type per block)
    block_mask_types = torch.full(
        (B, H, num_q_blocks, max_blocks),
        BlockMaskType.PARTIAL,
        dtype=torch.int32,
        device=kv_num_blocks.device
    )
    
    # Set mask types to MASKED for positions beyond kv_num_blocks
    for b in range(B):
        for h in range(H):
            for q_block in range(num_q_blocks):
                num_active = kv_num_blocks[b, h, q_block].item()
                # Positions beyond num_active are already invalid (won't be accessed)
                # but we can mark them as MASKED for clarity
                block_mask_types[b, h, q_block, num_active:] = BlockMaskType.MASKED
    
    return OmniBlockMask(
        kv_num_blocks=kv_num_blocks.to(torch.int32),
        kv_indices=kv_indices.to(torch.int32),
        block_mask_types=block_mask_types,
        q_len=q_len,
        kv_len=kv_len,
        Q_BLOCK_SIZE=q_block_size,
        KV_BLOCK_SIZE=kv_block_size,
    )


# ============================================================================
# Block Mask Creation Utilities
# ============================================================================


# ============================================================================
# Naive PyTorch Attention Implementation
# ============================================================================

def naive_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_mask: Optional[OmniBlockMask] = None,
    scale: Optional[float] = None,
) -> Tensor:
    """Naive PyTorch attention implementation with block mask support.
    
    This materializes the full attention matrix and applies the mask.
    Used as a reference for correctness testing.
    
    Args:
        query: [batch, nheads, q_len, head_dim]
        key: [batch, nheads, kv_len, head_dim]
        value: [batch, nheads, kv_len, head_dim]
        block_mask: Optional OmniBlockMask for sparse attention
        scale: Attention scale (default: 1/sqrt(head_dim))
        
    Returns:
        output: [batch, nheads, q_len, head_dim]
    """
    B, H, Q, D = query.shape
    _, _, KV, _ = key.shape
    
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    # Compute attention scores: [B, H, Q, KV]
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # Apply mask if provided
    if block_mask is not None:
        mask = block_mask.to_score_mask()
        # Broadcast mask to match scores shape
        if mask.shape != scores.shape:
            mask = mask.to(scores.dtype).to(scores.device)
        scores = scores + mask
    
    # Softmax and output
    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(attn_weights, value)
    
    return output


def naive_attention_with_dense_mask(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Optional[Tensor] = None,
    scale: Optional[float] = None,
) -> Tensor:
    """Naive PyTorch attention with dense mask tensor.
    
    Args:
        query: [batch, nheads, q_len, head_dim]
        key: [batch, nheads, kv_len, head_dim]
        value: [batch, nheads, kv_len, head_dim]
        mask: Optional [batch, nheads, q_len, kv_len] with 0 for attend, -inf for masked
        scale: Attention scale (default: 1/sqrt(head_dim))
        
    Returns:
        output: [batch, nheads, q_len, head_dim]
    """
    B, H, Q, D = query.shape
    
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    
    # Apply mask
    if mask is not None:
        scores = scores + mask.to(scores.dtype)
    
    # Softmax and output
    attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    output = torch.matmul(attn_weights, value)
    
    return output


# ============================================================================
# FlexAttention Wrapper
# ============================================================================

def flex_attention_with_block_mask(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_mask: OmniBlockMask,
    scale: Optional[float] = None,
) -> Tensor:
    """FlexAttention wrapper that converts OmniBlockMask to FlexAttention format.
    
    Args:
        query: [batch, nheads, q_len, head_dim]
        key: [batch, nheads, kv_len, head_dim]
        value: [batch, nheads, kv_len, head_dim]
        block_mask: OmniBlockMask for sparse attention
        scale: Attention scale (default: 1/sqrt(head_dim))
        
    Returns:
        output: [batch, nheads, q_len, head_dim]
    """
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    except ImportError:
        raise ImportError(
            "FlexAttention requires PyTorch 2.5+ with flex_attention support. "
            "Falling back to naive implementation."
        )
    
    B, H, Q, D = query.shape
    _, _, KV, _ = key.shape
    
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    # Convert OmniBlockMask to dense boolean mask
    dense_mask = block_mask.to_dense_mask()  # [B, H, Q, KV]
    
    # Create mask_mod function from dense mask
    # FlexAttention expects mask_mod(b, h, q_idx, kv_idx) -> bool
    def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        # Note: This is a simplified version - in practice you'd want to
        # create a proper block mask using FlexAttention's create_block_mask
        return dense_mask[b, h, q_idx, kv_idx]
    
    # Create FlexAttention block mask
    flex_block_mask = create_block_mask(
        mask_mod,
        B=B,
        H=H,
        Q_LEN=Q,
        KV_LEN=KV,
        device=query.device,
        BLOCK_SIZE=(block_mask.Q_BLOCK_SIZE, block_mask.KV_BLOCK_SIZE),
    )
    
    # Run FlexAttention
    # Note: flex_attention expects BHSD format
    output = flex_attention(
        query,
        key,
        value,
        block_mask=flex_block_mask,
        scale=scale,
    )
    
    return output


def flex_attention_compiled(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_mask: OmniBlockMask,
    scale: Optional[float] = None,
) -> Tensor:
    """Compiled FlexAttention for better performance.
    
    Uses torch.compile to optimize FlexAttention.
    """
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    except ImportError:
        raise ImportError("FlexAttention requires PyTorch 2.5+")
    
    B, H, Q, D = query.shape
    _, _, KV, _ = key.shape
    
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    
    # Convert to dense mask for mask_mod
    dense_mask = block_mask.to_dense_mask()
    
    def mask_mod(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
        return dense_mask[b, h, q_idx, kv_idx]
    
    flex_block_mask = create_block_mask(
        mask_mod,
        B=B,
        H=H,
        Q_LEN=Q,
        KV_LEN=KV,
        device=query.device,
        BLOCK_SIZE=(block_mask.Q_BLOCK_SIZE, block_mask.KV_BLOCK_SIZE),
    )
    
    # Compile flex_attention
    compiled_flex = torch.compile(flex_attention)
    
    output = compiled_flex(
        query,
        key,
        value,
        block_mask=flex_block_mask,
        scale=scale,
    )
    
    return output


# ============================================================================
# CUDA MMA Kernel Wrapper (placeholder until kernel is built)
# ============================================================================

def omni_attention_shared_kv(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_mask: OmniBlockMask,
    scale: Optional[float] = None,
) -> Tensor:
    """Omni-Attention with CUDA MMA kernel.
    
    This is a wrapper for the CUDA MMA implementation.
    
    Args:
        query: [batch, nheads, q_len, head_dim]
        key: [batch, nheads, kv_len, head_dim]
        value: [batch, nheads, kv_len, head_dim]
        block_mask: OmniBlockMask for sparse attention
        scale: Attention scale (default: 1/sqrt(head_dim))
        stages: Number of pipeline stages (1 or 2)
        
    Returns:
        output: [batch, nheads, q_len, head_dim]
    """
    try:
        from omni_attn import omni_attn_mma_stages_split_q_shared_kv
    except ImportError:
        # Try loading from current directory if not installed
        import sys
        import os
        import importlib.util
        
        # Look for .so file in current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        so_files = [
            os.path.join(current_dir, 'omni_attn.cpython-312-x86_64-linux-gnu.so'),
            os.path.join(current_dir, 'omni_attn.so'),
        ]
        
        module = None
        for so_path in so_files:
            if os.path.exists(so_path):
                try:
                    # Use the module name that matches setup.py
                    spec = importlib.util.spec_from_file_location('omni_attn', so_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        omni_attn_mma_stages_split_q_shared_kv = getattr(module, 'omni_attn_mma_stages_split_q_shared_kv')
                        break
                except Exception as e:
                    continue
        
        if module is None:
            raise ImportError(
                "CUDA MMA kernel not found. Please build the extension first:\n"
                "cd omni-attn-mma && python setup.py build_ext --inplace\n"
                f"Or ensure the .so file is in: {current_dir}"
            )
    
    B, H, Q, D = query.shape
    _, _, KV, _ = key.shape
    
    # Validate shapes
    assert key.shape == (B, H, KV, D), f"Key shape mismatch: {key.shape} vs expected {(B, H, KV, D)}"
    assert value.shape == (B, H, KV, D), f"Value shape mismatch: {value.shape} vs expected {(B, H, KV, D)}"
    assert Q == KV, f"Q and KV sequence lengths must match: Q={Q}, KV={KV}"
    
    # Pad to Q_BLOCK_SIZE and KV_BLOCK_SIZE multiples (like flex_attention does)
    # The mask is already created with padded length, so we just need to pad Q, K, V
    query_padded, q_orig_len = _pad_to_block_size(query, block_mask.Q_BLOCK_SIZE)
    key_padded, kv_orig_len = _pad_to_block_size(key, block_mask.KV_BLOCK_SIZE)
    value_padded, _ = _pad_to_block_size(value, block_mask.KV_BLOCK_SIZE)
    
    Q_padded = query_padded.shape[2]
    KV_padded = key_padded.shape[2]
    
    # Validate that padded lengths match mask (mask should already be padded)
    assert Q_padded == block_mask.q_len, \
        f"Padded Q length ({Q_padded}) doesn't match mask q_len ({block_mask.q_len})"
    assert KV_padded == block_mask.kv_len, \
        f"Padded KV length ({KV_padded}) doesn't match mask kv_len ({block_mask.kv_len})"
    
    # Validate block mask shapes
    num_q_blocks = Q_padded // block_mask.Q_BLOCK_SIZE
    assert block_mask.kv_num_blocks.shape == (B, H, num_q_blocks), \
        f"kv_num_blocks shape mismatch: {block_mask.kv_num_blocks.shape} vs expected {(B, H, num_q_blocks)}"
    max_blocks = block_mask.kv_indices.shape[3]
    assert block_mask.kv_indices.shape == (B, H, num_q_blocks, max_blocks), \
        f"kv_indices shape mismatch: {block_mask.kv_indices.shape} vs expected {(B, H, num_q_blocks, max_blocks)}"
    assert block_mask.block_mask_types.shape == (B, H, num_q_blocks, max_blocks), \
        f"block_mask_types shape mismatch: {block_mask.block_mask_types.shape} vs expected {(B, H, num_q_blocks, max_blocks)}"
    
    # Ensure contiguous and correct dtype for Q, K, V
    query_padded = query_padded.contiguous().half()
    key_padded = key_padded.contiguous().half()
    value_padded = value_padded.contiguous().half()
    
    # Create output tensor (padded size)
    output_padded = torch.empty_like(query_padded)
    
    # Ensure block mask tensors are int32 and contiguous
    kv_num_blocks = block_mask.kv_num_blocks.contiguous().to(torch.int32)
    kv_indices = block_mask.kv_indices.contiguous().to(torch.int32)
    block_mask_types = block_mask.block_mask_types.contiguous().to(torch.int32)
    
    # Validate head_dim is supported (32, 64, 128)
    if D not in [32, 64, 128]:
        raise ValueError(
            f"Unsupported head_dim={D}. Supported values: 32, 64, 128. "
            f"Please use D that results in head_dim in [32, 64, 128] (e.g., D=256 with H=8 gives head_dim=32)."
        )
    
    # Validate Q_BLOCK_SIZE is supported (64 or 128)
    if block_mask.Q_BLOCK_SIZE not in [64, 128]:
        raise ValueError(
            f"Unsupported Q_BLOCK_SIZE={block_mask.Q_BLOCK_SIZE}. Supported values: 64, 128. "
            f"Please use --q_block_size 64 or --q_block_size 128."
        )
    
    if block_mask.KV_BLOCK_SIZE not in [64, 128]:
        raise ValueError(
            f"Unsupported KV_BLOCK_SIZE={block_mask.KV_BLOCK_SIZE}. Supported values: 64, 128. "
            f"Please use --kv_block_size 64 or --kv_block_size 128."
        )
    
    has_partial = (block_mask.partial_block_mask_indices is not None and 
                   block_mask.partial_block_masks is not None)
    partial_block_mask_indices = None
    partial_block_masks = None
    if has_partial:
        partial_block_mask_indices = block_mask.partial_block_mask_indices.contiguous().to(torch.int32)
        partial_block_masks = block_mask.partial_block_masks.contiguous().to(torch.bool)

    # print(f"has_partial: {has_partial} {partial_block_mask_indices is not None} {partial_block_masks is not None}")
    
    # Call CUDA kernel
    omni_attn_mma_stages_split_q_shared_kv(
        query_padded, key_padded, value_padded, output_padded,
        kv_num_blocks,
        kv_indices,
        block_mask_types,
        block_mask.Q_BLOCK_SIZE,
        block_mask.KV_BLOCK_SIZE,
        q_orig_len,
        partial_block_mask_indices if has_partial else torch.empty(0, dtype=torch.int32, device=query_padded.device),
        partial_block_masks if has_partial else torch.empty(0, dtype=torch.bool, device=query_padded.device),
        has_partial,
    )
    
    # Synchronize to catch any CUDA errors immediately
    torch.cuda.synchronize()
    
    # Slice back to original length
    output = output_padded[:, :, :q_orig_len, :]
    
    return output

def omni_attention_simple(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_mask: OmniBlockMask,
) -> Tensor:
    """Omni-Attention with simple CUDA kernel (correctness baseline).
    
    This is a wrapper for the simple CUDA kernel that serves as a correctness
    baseline. It's not performance-optimized but is easier to reason about.
    
    Args:
        query: [batch, nheads, q_len, head_dim]
        key: [batch, nheads, kv_len, head_dim]
        value: [batch, nheads, kv_len, head_dim]
        block_mask: OmniBlockMask for sparse attention
        scale: Attention scale (default: 1/sqrt(head_dim))
        
    Returns:
        output: [batch, nheads, q_len, head_dim]
    """
    try:
        from omni_attn import omni_attn_simple_kernel
    except ImportError:
        # Try loading from current directory if not installed
        import sys
        import os
        import importlib.util
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        so_files = [
            os.path.join(current_dir, 'omni_attn.cpython-312-x86_64-linux-gnu.so'),
            os.path.join(current_dir, 'omni_attn.so'),
        ]
        
        module = None
        for so_path in so_files:
            if os.path.exists(so_path):
                try:
                    spec = importlib.util.spec_from_file_location('omni_attn', so_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        omni_attn_simple_kernel = getattr(module, 'omni_attn_simple_kernel')
                        break
                except Exception as e:
                    continue
        
        if module is None:
            raise ImportError(
                "CUDA simple kernel not found. Please build the extension first:\n"
                "cd omni-attn-mma && python setup.py build_ext --inplace\n"
                f"Or ensure the .so file is in: {current_dir}"
            )
    
    B, H, Q, D = query.shape
    _, _, KV, _ = key.shape
    
    # Validate shapes
    assert key.shape == (B, H, KV, D), f"Key shape mismatch: {key.shape} vs expected {(B, H, KV, D)}"
    assert value.shape == (B, H, KV, D), f"Value shape mismatch: {value.shape} vs expected {(B, H, KV, D)}"
    assert Q == KV, f"Q and KV sequence lengths must match: Q={Q}, KV={KV}"
    
    # Pad to Q_BLOCK_SIZE and KV_BLOCK_SIZE multiples (like flex_attention does)
    # The mask is already created with padded length, so we just need to pad Q, K, V
    query_padded, q_orig_len = _pad_to_block_size(query, block_mask.Q_BLOCK_SIZE)
    key_padded, kv_orig_len = _pad_to_block_size(key, block_mask.KV_BLOCK_SIZE)
    value_padded, _ = _pad_to_block_size(value, block_mask.KV_BLOCK_SIZE)
    
    Q_padded = query_padded.shape[2]
    KV_padded = key_padded.shape[2]
    
    # Validate that padded lengths match mask (mask should already be padded)
    assert Q_padded == block_mask.q_len, \
        f"Padded Q length ({Q_padded}) doesn't match mask q_len ({block_mask.q_len})"
    assert KV_padded == block_mask.kv_len, \
        f"Padded KV length ({KV_padded}) doesn't match mask kv_len ({block_mask.kv_len})"
    
    # Ensure contiguous and correct dtype for Q, K, V
    query_padded = query_padded.contiguous().half()
    key_padded = key_padded.contiguous().half()
    value_padded = value_padded.contiguous().half()
    
    # Create output tensor (padded size)
    output_padded = torch.empty_like(query_padded)
    
    # Ensure block mask tensors are int32 and contiguous
    kv_num_blocks = block_mask.kv_num_blocks.contiguous().to(torch.int32)
    kv_indices = block_mask.kv_indices.contiguous().to(torch.int32)
    block_mask_types = block_mask.block_mask_types.contiguous().to(torch.int32)
    has_partial = ((block_mask_types == BlockMaskType.PARTIAL).any().item() and
                   block_mask.partial_block_mask_indices is not None and
                   block_mask.partial_block_masks is not None)
    
    B, H, num_q_blocks, max_blocks = block_mask_types.shape
    partial_offsets = torch.full(
        (B, H, num_q_blocks, max_blocks),
        -1,
        dtype=torch.int32,
        device=query_padded.device
    )
    partial_masks = torch.empty(
        0,
        block_mask.Q_BLOCK_SIZE,
        block_mask.KV_BLOCK_SIZE,
        dtype=torch.bool,
        device=query_padded.device,
    )

    if has_partial:
        print("has_partial")
        partial_offsets = block_mask.partial_block_mask_indices
        partial_masks = block_mask.partial_block_masks
        partial_offsets = partial_offsets.contiguous().to(torch.int32)
        partial_masks = partial_masks.contiguous().to(torch.bool)
    
    # Call CUDA kernel (pass original length to bound KV access)
    omni_attn_simple_kernel(
        query_padded, key_padded, value_padded, output_padded,
        kv_num_blocks,
        kv_indices,
        block_mask_types,
        block_mask.Q_BLOCK_SIZE,
        block_mask.KV_BLOCK_SIZE,
        q_orig_len,  # Original sequence length (before padding)
        partial_offsets,
        partial_masks,
        has_partial,
    )
    
    # Synchronize to catch any CUDA errors immediately
    torch.cuda.synchronize()
    
    # Slice back to original length
    output = output_padded[:, :, :q_orig_len, :]
    
    return output


def omni_attention_cp_async(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_mask: OmniBlockMask,
    scale: Optional[float] = None,
) -> Tensor:
    """Omni-Attention with cp.async CUDA kernel.
    
    This is a wrapper for the cp.async optimized CUDA kernel that uses
    copy-prefetch async and Q tiling strategy.
    
    Args:
        query: [batch, nheads, q_len, head_dim]
        key: [batch, nheads, kv_len, head_dim]
        value: [batch, nheads, kv_len, head_dim]
        block_mask: OmniBlockMask for sparse attention
        scale: Attention scale (default: 1/sqrt(head_dim))
        
    Returns:
        output: [batch, nheads, q_len, head_dim]
    """
    try:
        from omni_attn import omni_attn_cp_async
    except ImportError:
        # Try loading from current directory if not installed
        import sys
        import os
        import importlib.util
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        so_files = [
            os.path.join(current_dir, 'omni_attn.cpython-312-x86_64-linux-gnu.so'),
            os.path.join(current_dir, 'omni_attn.so'),
        ]
        
        module = None
        for so_path in so_files:
            if os.path.exists(so_path):
                try:
                    spec = importlib.util.spec_from_file_location('omni_attn', so_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        omni_attn_cp_async = getattr(module, 'omni_attn_cp_async')
                        break
                except Exception as e:
                    continue
        
        if module is None:
            raise ImportError(
                "CUDA cp.async kernel not found. Please build the extension first:\n"
                "cd omni-attn-mma && python setup.py build_ext --inplace\n"
                f"Or ensure the .so file is in: {current_dir}"
            )
    
    B, H, Q, D = query.shape
    _, _, KV, _ = key.shape
    
    # Validate shapes
    assert key.shape == (B, H, KV, D), f"Key shape mismatch: {key.shape} vs expected {(B, H, KV, D)}"
    assert value.shape == (B, H, KV, D), f"Value shape mismatch: {value.shape} vs expected {(B, H, KV, D)}"
    assert Q == KV, f"Q and KV sequence lengths must match: Q={Q}, KV={KV}"
    
    # Pad to Q_BLOCK_SIZE and KV_BLOCK_SIZE multiples (like flex_attention does)
    # The mask is already created with padded length, so we just need to pad Q, K, V
    query_padded, q_orig_len = _pad_to_block_size(query, block_mask.Q_BLOCK_SIZE)
    key_padded, kv_orig_len = _pad_to_block_size(key, block_mask.KV_BLOCK_SIZE)
    value_padded, _ = _pad_to_block_size(value, block_mask.KV_BLOCK_SIZE)
    
    Q_padded = query_padded.shape[2]
    KV_padded = key_padded.shape[2]
    
    # Validate that padded lengths match mask (mask should already be padded)
    assert Q_padded == block_mask.q_len, \
        f"Padded Q length ({Q_padded}) doesn't match mask q_len ({block_mask.q_len})"
    assert KV_padded == block_mask.kv_len, \
        f"Padded KV length ({KV_padded}) doesn't match mask kv_len ({block_mask.kv_len})"
    
    # Validate head_dim is supported (32, 64, 128)
    if D not in [32, 64, 128]:
        raise ValueError(
            f"Unsupported head_dim={D}. Supported values: 32, 64, 128. "
            f"Please use D that results in head_dim in [32, 64, 128] (e.g., D=256 with H=8 gives head_dim=32)."
        )
    
    # Validate Q_BLOCK_SIZE is supported (64 or 128)
    if block_mask.Q_BLOCK_SIZE not in [64, 128]:
        raise ValueError(
            f"Unsupported Q_BLOCK_SIZE={block_mask.Q_BLOCK_SIZE}. Supported values: 64, 128."
        )
    # Validate KV_BLOCK_SIZE is supported (64 or 128)
    if block_mask.KV_BLOCK_SIZE not in [64, 128]:
        raise ValueError(
            f"Unsupported KV_BLOCK_SIZE={block_mask.KV_BLOCK_SIZE}. Supported values: 64, 128."
        )

    # Ensure contiguous and correct dtype for Q, K, V
    query_padded = query_padded.contiguous().half()
    key_padded = key_padded.contiguous().half()
    value_padded = value_padded.contiguous().half()
    
    # Create output tensor (padded size)
    output_padded = torch.empty_like(query_padded)
    
    # Ensure block mask tensors are int32 and contiguous
    kv_num_blocks = block_mask.kv_num_blocks.contiguous().to(torch.int32)
    kv_indices = block_mask.kv_indices.contiguous().to(torch.int32)
    block_mask_types = block_mask.block_mask_types.contiguous().to(torch.int32)

    # Get partial mask information
    has_partial = (block_mask_types == BlockMaskType.PARTIAL).any().item()
    
    B, H, num_q_blocks, max_blocks = block_mask_types.shape
    partial_offsets = torch.full(
        (B, H, num_q_blocks, max_blocks),
        -1,
        dtype=torch.int32,
        device=query_padded.device
    )
    partial_masks = torch.empty(
        0,
        block_mask.Q_BLOCK_SIZE,
        block_mask.KV_BLOCK_SIZE,
        dtype=torch.bool,
        device=query_padded.device,
    )
    if has_partial:
        partial_offsets = block_mask.partial_block_mask_indices
        partial_masks = block_mask.partial_block_masks

        partial_offsets = partial_offsets.contiguous().to(torch.int32)
        partial_masks = partial_masks.contiguous().to(torch.bool)
    
    omni_attn_cp_async(
        query_padded, key_padded, value_padded, output_padded,
        kv_num_blocks,
        kv_indices,
        block_mask_types,
        block_mask.Q_BLOCK_SIZE,
        block_mask.KV_BLOCK_SIZE,
        q_orig_len,
        partial_offsets,
        partial_masks,
        has_partial,
    )
    
    # Synchronize to catch any CUDA errors immediately
    torch.cuda.synchronize()
    
    # Slice back to original length
    output = output_padded[:, :, :q_orig_len, :]
    
    return output

def omni_attention_preftech(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_mask: OmniBlockMask,
) -> Tensor:
    """Omni-Attention with preftech CUDA kernel.
    
    This is a wrapper for the preftech optimized CUDA kernel that uses
    """
    try:
        from omni_attn import omni_attn_preftech
    except ImportError:
        # Try loading from current directory if not installed
        import sys
        import os
        import importlib.util
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        so_files = [
            os.path.join(current_dir, 'omni_attn.cpython-312-x86_64-linux-gnu.so'),
            os.path.join(current_dir, 'omni_attn.so'),
        ]
        
        module = None
        for so_path in so_files:
            if os.path.exists(so_path):
                try:
                    spec = importlib.util.spec_from_file_location('omni_attn', so_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        omni_attn_preftech = getattr(module, 'omni_attn_preftech_kernel')
                        break
                except Exception as e:
                    continue
        
        if module is None:
            raise ImportError(
                "CUDA preftech kernel not found. Please build the extension first:\n"
                "cd omni-attn-mma && python setup.py build_ext --inplace\n"
                f"Or ensure the .so file is in: {current_dir}"
            )
    
    B, H, Q, D = query.shape
    _, _, KV, _ = key.shape
    
    # Validate shapes
    assert key.shape == (B, H, KV, D), f"Key shape mismatch: {key.shape} vs expected {(B, H, KV, D)}"
    assert value.shape == (B, H, KV, D), f"Value shape mismatch: {value.shape} vs expected {(B, H, KV, D)}"
    assert Q == KV, f"Q and KV sequence lengths must match: Q={Q}, KV={KV}"
    
    # Pad to Q_BLOCK_SIZE and KV_BLOCK_SIZE multiples (like flex_attention does)
    # The mask is already created with padded length, so we just need to pad Q, K, V
    query_padded, q_orig_len = _pad_to_block_size(query, block_mask.Q_BLOCK_SIZE)
    key_padded, kv_orig_len = _pad_to_block_size(key, block_mask.KV_BLOCK_SIZE)
    value_padded, _ = _pad_to_block_size(value, block_mask.KV_BLOCK_SIZE)
    
    Q_padded = query_padded.shape[2]
    KV_padded = key_padded.shape[2]
    
    # Validate that padded lengths match mask (mask should already be padded)
    assert Q_padded == block_mask.q_len, \
        f"Padded Q length ({Q_padded}) doesn't match mask q_len ({block_mask.q_len})"
    assert KV_padded == block_mask.kv_len, \
        f"Padded KV length ({KV_padded}) doesn't match mask kv_len ({block_mask.kv_len})"
    
    # Validate head_dim is supported (32, 64, 128)
    if D not in [32, 64, 128]:
        raise ValueError(
            f"Unsupported head_dim={D}. Supported values: 32, 64, 128. "
            f"Please use D that results in head_dim in [32, 64, 128] (e.g., D=256 with H=8 gives head_dim=32)."
        )
    
    # Validate Q_BLOCK_SIZE is supported (64 or 128)
    if block_mask.Q_BLOCK_SIZE not in [64, 128]:
        raise ValueError(
            f"Unsupported Q_BLOCK_SIZE={block_mask.Q_BLOCK_SIZE}. Supported values: 64, 128."
        )
    
    # Validate KV_BLOCK_SIZE is supported (64 or 128)
    if block_mask.KV_BLOCK_SIZE not in [64, 128]:
        raise ValueError(
            f"Unsupported KV_BLOCK_SIZE={block_mask.KV_BLOCK_SIZE}. Supported values: 64, 128."
        )
    
    # Ensure contiguous and correct dtype for Q, K, V
    query_padded = query_padded.contiguous().half()
    key_padded = key_padded.contiguous().half()
    value_padded = value_padded.contiguous().half()
    
    # Create output tensor (padded size)
    output_padded = torch.empty_like(query_padded)
    
    # Ensure block mask tensors are int32 and contiguous
    kv_num_blocks = block_mask.kv_num_blocks.contiguous().to(torch.int32)
    kv_indices = block_mask.kv_indices.contiguous().to(torch.int32)
    block_mask_types = block_mask.block_mask_types.contiguous().to(torch.int32)

    # Get partial mask information
    has_partial = ((block_mask_types == BlockMaskType.PARTIAL).any().item() and
                   block_mask.partial_block_mask_indices is not None and
                   block_mask.partial_block_masks is not None)
    
    B, H, num_q_blocks, max_blocks = block_mask_types.shape
    partial_offsets = torch.full(
        (B, H, num_q_blocks, max_blocks),
        -1,
        dtype=torch.int32,
        device=query_padded.device
    )
    partial_masks = torch.empty(
        0,
        block_mask.Q_BLOCK_SIZE,
        block_mask.KV_BLOCK_SIZE,
        dtype=torch.bool,
        device=query_padded.device,
    )
    if has_partial:
        partial_offsets = block_mask.partial_block_mask_indices
        partial_masks = block_mask.partial_block_masks

        partial_offsets = partial_offsets.contiguous().to(torch.int32)
        partial_masks = partial_masks.contiguous().to(torch.bool)
    
    omni_attn_preftech(
        query_padded, key_padded, value_padded, output_padded,
        kv_num_blocks,
        kv_indices,
        block_mask_types,
        block_mask.Q_BLOCK_SIZE,
        block_mask.KV_BLOCK_SIZE,
        q_orig_len,
        partial_offsets,
        partial_masks,
        has_partial,
    )
    
    # Synchronize to catch any CUDA errors immediately
    torch.cuda.synchronize()
    
    # Slice back to original length
    output = output_padded[:, :, :q_orig_len, :]
    
    return output

def omni_attention_shared_kv_swizzle(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_mask: OmniBlockMask,
) -> Tensor:
    """Omni-Attention with shared KV CUDA kernel with swizzling.
    """
    try:
        from omni_attn import omni_attn_mma_stages_split_q_shared_kv_swizzle
    except ImportError:
        # Try loading from current directory if not installed
        import sys
        import os
        import importlib.util
        
        # Look for .so file in current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        so_files = [
            os.path.join(current_dir, 'omni_attn.cpython-312-x86_64-linux-gnu.so'),
            os.path.join(current_dir, 'omni_attn.so'),
        ]
        
        module = None
        for so_path in so_files:
            if os.path.exists(so_path):
                try:
                    # Use the module name that matches setup.py
                    spec = importlib.util.spec_from_file_location('omni_attn', so_path)
                    if spec and spec.loader:
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        omni_attn_mma_stages_split_q_shared_kv_swizzle = getattr(module, 'omni_attn_mma_stages_split_q_shared_kv_swizzle')
                        break
                except Exception as e:
                    continue
        
        if module is None:
            raise ImportError(
                "CUDA MMA kernel not found. Please build the extension first:\n"
                "cd omni-attn-mma && python setup.py build_ext --inplace\n"
                f"Or ensure the .so file is in: {current_dir}"
            )
    
    B, H, Q, D = query.shape
    _, _, KV, _ = key.shape
    
    # Validate shapes
    assert key.shape == (B, H, KV, D), f"Key shape mismatch: {key.shape} vs expected {(B, H, KV, D)}"
    assert value.shape == (B, H, KV, D), f"Value shape mismatch: {value.shape} vs expected {(B, H, KV, D)}"
    assert Q == KV, f"Q and KV sequence lengths must match: Q={Q}, KV={KV}"
    
    # Pad to Q_BLOCK_SIZE and KV_BLOCK_SIZE multiples (like flex_attention does)
    # The mask is already created with padded length, so we just need to pad Q, K, V
    query_padded, q_orig_len = _pad_to_block_size(query, block_mask.Q_BLOCK_SIZE)
    key_padded, kv_orig_len = _pad_to_block_size(key, block_mask.KV_BLOCK_SIZE)
    value_padded, _ = _pad_to_block_size(value, block_mask.KV_BLOCK_SIZE)
    
    Q_padded = query_padded.shape[2]
    KV_padded = key_padded.shape[2]
    
    # Validate that padded lengths match mask (mask should already be padded)
    assert Q_padded == block_mask.q_len, \
        f"Padded Q length ({Q_padded}) doesn't match mask q_len ({block_mask.q_len})"
    assert KV_padded == block_mask.kv_len, \
        f"Padded KV length ({KV_padded}) doesn't match mask kv_len ({block_mask.kv_len})"
    
    # Validate block mask shapes
    num_q_blocks = Q_padded // block_mask.Q_BLOCK_SIZE
    assert block_mask.kv_num_blocks.shape == (B, H, num_q_blocks), \
        f"kv_num_blocks shape mismatch: {block_mask.kv_num_blocks.shape} vs expected {(B, H, num_q_blocks)}"
    max_blocks = block_mask.kv_indices.shape[3]
    assert block_mask.kv_indices.shape == (B, H, num_q_blocks, max_blocks), \
        f"kv_indices shape mismatch: {block_mask.kv_indices.shape} vs expected {(B, H, num_q_blocks, max_blocks)}"
    assert block_mask.block_mask_types.shape == (B, H, num_q_blocks, max_blocks), \
        f"block_mask_types shape mismatch: {block_mask.block_mask_types.shape} vs expected {(B, H, num_q_blocks, max_blocks)}"
    
    # Ensure contiguous and correct dtype for Q, K, V
    query_padded = query_padded.contiguous().half()
    key_padded = key_padded.contiguous().half()
    value_padded = value_padded.contiguous().half()
    
    # Create output tensor (padded size)
    output_padded = torch.empty_like(query_padded)
    
    # Ensure block mask tensors are int32 and contiguous
    kv_num_blocks = block_mask.kv_num_blocks.contiguous().to(torch.int32)
    kv_indices = block_mask.kv_indices.contiguous().to(torch.int32)
    block_mask_types = block_mask.block_mask_types.contiguous().to(torch.int32)
    
    # Validate head_dim is supported (32, 64, 128)
    if D not in [32, 64, 128]:
        raise ValueError(
            f"Unsupported head_dim={D}. Supported values: 32, 64, 128. "
            f"Please use D that results in head_dim in [32, 64, 128] (e.g., D=256 with H=8 gives head_dim=32)."
        )
    
    # Validate Q_BLOCK_SIZE is supported (64 or 128)
    if block_mask.Q_BLOCK_SIZE not in [64, 128]:
        raise ValueError(
            f"Unsupported Q_BLOCK_SIZE={block_mask.Q_BLOCK_SIZE}. Supported values: 64, 128. "
            f"Please use --q_block_size 64 or --q_block_size 128."
        )
    
    if block_mask.KV_BLOCK_SIZE not in [64, 128]:
        raise ValueError(
            f"Unsupported KV_BLOCK_SIZE={block_mask.KV_BLOCK_SIZE}. Supported values: 64, 128. "
            f"Please use --kv_block_size 64 or --kv_block_size 128."
        )
    
    has_partial = (block_mask.partial_block_mask_indices is not None and 
                   block_mask.partial_block_masks is not None)
    partial_block_mask_indices = None
    partial_block_masks = None
    if has_partial:
        partial_block_mask_indices = block_mask.partial_block_mask_indices.contiguous().to(torch.int32)
        partial_block_masks = block_mask.partial_block_masks.contiguous().to(torch.bool)

    # print(f"has_partial: {has_partial} {partial_block_mask_indices is not None} {partial_block_masks is not None}")
    
    # Call CUDA kernel
    omni_attn_mma_stages_split_q_shared_kv_swizzle(
        query_padded, key_padded, value_padded, output_padded,
        kv_num_blocks,
        kv_indices,
        block_mask_types,
        block_mask.Q_BLOCK_SIZE,
        block_mask.KV_BLOCK_SIZE,
        q_orig_len,
        partial_block_mask_indices if has_partial else torch.empty(0, dtype=torch.int32, device=query_padded.device),
        partial_block_masks if has_partial else torch.empty(0, dtype=torch.bool, device=query_padded.device),
        has_partial,
    )
    
    # Synchronize to catch any CUDA errors immediately
    torch.cuda.synchronize()
    
    # Slice back to original length
    output = output_padded[:, :, :q_orig_len, :]
    
    return output


# ============================================================================
# Utility Functions
# ============================================================================


def _pad_to_block_size(x: Tensor, BLOCK_SIZE: int) -> Tuple[Tensor, int]:
    """Pad tensor's sequence dimension to multiple of BLOCK_SIZE.
    
    Args:
        x: [..., seq_len, ...] tensor
        BLOCK_SIZE: Block size to pad to
        
    Returns:
        padded_x: Padded tensor
        original_len: Original sequence length
    """
    *leading_dims, seq_len = x.shape[:-1]
    head_dim = x.shape[-1]
    
    padded_len = ((seq_len + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    if padded_len == seq_len:
        return x, seq_len
    
    padding_size = padded_len - seq_len
    # Pad with zeros on the sequence dimension
    padding = torch.zeros(*leading_dims, padding_size, head_dim, 
                         device=x.device, dtype=x.dtype)
    padded_x = torch.cat([x, padding], dim=-2)
    return padded_x, seq_len


def check_correctness(
    ref_output: Tensor,
    test_output: Tensor,
    rtol: float = 1e-2,
    atol: float = 1e-3,
    name: str = "test",
) -> bool:
    """Check if test output matches reference output.
    
    Args:
        ref_output: Reference output tensor
        test_output: Test output tensor
        rtol: Relative tolerance
        atol: Absolute tolerance
        name: Name for error messages
        
    Returns:
        True if outputs match within tolerance
    """
    if ref_output.shape != test_output.shape:
        print(f"[{name}] Shape mismatch: {ref_output.shape} vs {test_output.shape}")
        return False
    
    # Check for None or invalid outputs
    if test_output is None:
        print(f"[{name}] test_output is None")
        return False
    
    ref_output = ref_output.float()
    test_output = test_output.float()
    
    # Check for NaN values in outputs
    ref_nan_count = torch.isnan(ref_output).sum().item()
    test_nan_count = torch.isnan(test_output).sum().item()
    ref_inf_count = torch.isinf(ref_output).sum().item()
    test_inf_count = torch.isinf(test_output).sum().item()
    
    # Check for uninitialized output (all zeros or all NaN)
    test_zero_count = (test_output == 0).sum().item()
    test_total = test_output.numel()
    if test_zero_count == test_total:
        print(f"[{name}] WARNING: test_output is all zeros (possibly uninitialized)")
    elif test_nan_count == test_total:
        print(f"[{name}] WARNING: test_output is all NaN (kernel may have failed)")
    
    if ref_nan_count > 0:
        print(f"[{name}] WARNING: ref_output contains {ref_nan_count} NaN values")
    if test_nan_count > 0:
        print(f"[{name}] WARNING: test_output contains {test_nan_count} NaN values (out of {test_total}({test_nan_count/test_total:.2%}) total)")
    if ref_inf_count > 0:
        print(f"[{name}] WARNING: ref_output contains {ref_inf_count} Inf values")
    if test_inf_count > 0:
        print(f"[{name}] WARNING: test_output contains {test_inf_count} Inf values")
    
    # Compute differences (will be nan if either tensor has nan)
    diff = (ref_output - test_output).abs()
    max_diff = diff.max().item()
    min_diff = diff.min().item()
    mean_diff = diff.mean().item()
    
    passed = torch.allclose(ref_output, test_output, rtol=rtol, atol=atol)
    
    # print max and min diff of some rows of batch[0, 0, ]
    # for i in range(16):
    #     rows_ref = ref_output[0, 0, i*64, :]
    #     rows_test = test_output[0, 0, i*64, :]
    #     rows_diff = (rows_ref - rows_test).abs()
    #     rows_max_diff = rows_diff.max().item()
    #     rows_min_diff = rows_diff.min().item()
    #     rows_mean_diff = rows_diff.mean().item()
    #     print(f"First {i*64} rows (batch[0], head[0]): max_diff={rows_max_diff:.6f}, min_diff={rows_min_diff:.6f}, mean_diff={rows_mean_diff:.6f}")

    if passed:
        print(f"[{name}] PASSED (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}), min_diff={min_diff:.6f}")
    else:
        print(f"[{name}] FAILED (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}), min_diff={min_diff:.6f}")
    
    return passed


def create_random_qkv(
    batch: int,
    nheads: int,
    q_len: int,
    kv_len: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: Union[str, torch.device] = "cuda",
    seed: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Create random Q, K, V tensors.
    
    Args:
        batch: Batch size
        nheads: Number of attention heads
        q_len: Query sequence length
        kv_len: Key/value sequence length
        head_dim: Head dimension
        dtype: Tensor dtype
        device: Device
        seed: Random seed
        
    Returns:
        (query, key, value) tensors
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    query = torch.randn(batch, nheads, q_len, head_dim, dtype=dtype, device=device)
    key = torch.randn(batch, nheads, kv_len, head_dim, dtype=dtype, device=device)
    value = torch.randn(batch, nheads, kv_len, head_dim, dtype=dtype, device=device)
    
    return query, key, value


def pad_modalities_to_block_size(
    embeddings: Tensor,
    modality_positions: Tensor,
    BLOCK_SIZE: int = 128,
) -> Tuple[Tensor, Tensor]:
    """Pad each modality segment separately to multiple of BLOCK_SIZE.
    
    Args:
        embeddings: [B, N, D] tensor of embeddings
        modality_positions: [B, M, 3] tensor of (modality_type, offset, length)
        BLOCK_SIZE: Block size for padding
        
    Returns:
        padded_embeddings: [B, new_N, D] padded embeddings
        new_modality_positions: [B, M, 3] updated modality positions
    """
    B, N, D = embeddings.shape
    device, dtype = embeddings.device, embeddings.dtype
    
    def round_up(x):
        return ((x + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE
    
    padded_segments = []
    new_modality_positions_list = []
    
    for b in range(B):
        mods = modality_positions[b]
        valid_mask = mods[:, 2] > 0
        mods = mods[valid_mask]
        
        if len(mods) == 0:
            # No modalities - pad whole sequence if needed
            if N % BLOCK_SIZE != 0:
                padded_len = round_up(N)
                padding = torch.zeros(padded_len - N, D, device=device, dtype=dtype)
                padded_segments.append(torch.cat([embeddings[b], padding], dim=0))
            else:
                padded_segments.append(embeddings[b])
            new_modality_positions_list.append(torch.zeros(0, 3, dtype=torch.int64, device=device))
            continue
        
        # Sort by offset
        mods = mods[mods[:, 1].argsort()]
        segments = []
        new_positions = []
        offset = 0
        
        # Process each modality with text before it
        for i, (mod_type, mod_start, mod_len) in enumerate(mods):
            mod_start, mod_len = int(mod_start), int(mod_len)
            
            # Text segment before this modality
            text_start = 0 if i == 0 else int(mods[i-1, 1] + mods[i-1, 2])
            if mod_start > text_start:
                text_seg = embeddings[b, text_start:mod_start]
                text_len = text_seg.shape[0]
                if text_len % BLOCK_SIZE != 0:
                    padded_len = round_up(text_len)
                    padding = torch.zeros(padded_len - text_len, D, device=device, dtype=dtype)
                    text_seg = torch.cat([text_seg, padding], dim=0)
                segments.append(text_seg)
                offset += text_seg.shape[0]
            
            # Modality segment
            mod_seg = embeddings[b, mod_start:mod_start+mod_len]
            if mod_len % BLOCK_SIZE != 0:
                padded_len = round_up(mod_len)
                padding = torch.zeros(padded_len - mod_len, D, device=device, dtype=dtype)
                mod_seg = torch.cat([mod_seg, padding], dim=0)
                # Important: modality_positions should only contain actual modality length, not padding
                # Padding will be treated as text (causal) by the mask
                new_positions.append((mod_type, offset, mod_len))
            else:
                new_positions.append((mod_type, offset, mod_len))
            segments.append(mod_seg)
            offset += mod_seg.shape[0]
        
        # Final text segment
        last_mod_end = int(mods[-1, 1] + mods[-1, 2])
        if last_mod_end < N:
            text_seg = embeddings[b, last_mod_end:]
            text_len = text_seg.shape[0]
            if text_len > 0:
                if text_len % BLOCK_SIZE != 0:
                    padded_len = round_up(text_len)
                    padding = torch.zeros(padded_len - text_len, D, device=device, dtype=dtype)
                    text_seg = torch.cat([text_seg, padding], dim=0)
                segments.append(text_seg)
                offset += text_seg.shape[0]
        
        # Concatenate
        padded_seq = torch.cat(segments, dim=0) if segments else embeddings[b]
        padded_segments.append(padded_seq)
        
        # Convert positions to tensor
        if new_positions:
            new_modality_positions_list.append(torch.tensor(new_positions, dtype=torch.int64, device=device))
        else:
            new_modality_positions_list.append(torch.zeros(0, 3, dtype=torch.int64, device=device))
    
    # Pad to max length across batch
    max_len = max(seg.shape[0] for seg in padded_segments)
    padded_embeddings = torch.zeros(B, max_len, D, device=device, dtype=dtype)
    for b, seg in enumerate(padded_segments):
        padded_embeddings[b, :seg.shape[0]] = seg
    
    # Pad modality positions to same M
    max_m = max(pos.shape[0] for pos in new_modality_positions_list)
    new_modality_positions = torch.zeros(B, max_m, 3, dtype=torch.int64, device=device)
    for b, pos in enumerate(new_modality_positions_list):
        if pos.shape[0] > 0:
            new_modality_positions[b, :pos.shape[0]] = pos
    
    return padded_embeddings, new_modality_positions


def create_omni_block_mask_from_modality_positions(
    modality_positions: Tensor,
    batch: int,
    nheads: int,
    q_len: int,
    kv_len: int,
    Q_BLOCK_SIZE: int = 128,
    KV_BLOCK_SIZE: int = 128,
    device: Union[str, torch.device] = "cuda",
) -> OmniBlockMask:
    """Create OmniBlockMask from modality positions.
    
    Mask rules:
    - Text segments: CAUSAL attention
    - Image modalities: FULL attention within same modality, FULL to all previous tokens
    - Different image modalities: MASKED (no attention between different images)
    
    Like flex_attention, this function pads the sequence length to Q_BLOCK_SIZE and KV_BLOCK_SIZE multiples
    internally. The mask treats padding positions as masked (no attention).
    
    Args:
        modality_positions: [B, M, 3] tensor of (modality_type, offset, length)
        batch: Batch size
        nheads: Number of attention heads
        q_len: Query sequence length (will be padded to Q_BLOCK_SIZE multiple)
        kv_len: Key/value sequence length (will be padded to KV_BLOCK_SIZE multiple)
        Q_BLOCK_SIZE: Block size for query
        KV_BLOCK_SIZE: Block size for key/value
        device: Device
        
    Returns:
        OmniBlockMask with padded lengths
    """
    # Pad to BLOCK_SIZE multiples (like flex_attention does)
    q_len_padded = ((q_len + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE) * Q_BLOCK_SIZE
    kv_len_padded = ((kv_len + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE) * KV_BLOCK_SIZE
    
    num_q_blocks = q_len_padded // Q_BLOCK_SIZE
    num_kv_blocks = kv_len_padded // KV_BLOCK_SIZE
    max_blocks = num_kv_blocks
    
    # Create token type array: 0 = text, 1+ = modality id
    # Use padded length - padding positions will remain 0 (text/masked)
    token_types = torch.zeros(batch, q_len_padded, dtype=torch.int32, device=device)
    modality_ids = {}  # Track which modality id each (b, mod_idx) maps to
    
    for b in range(batch):
        mods = modality_positions[b]  # [M, 3]
        valid_mask = mods[:, 2] > 0
        mods = mods[valid_mask]
        
        if len(mods) == 0:
            continue
        
        # Sort by offset
        sorted_indices = mods[:, 1].argsort()
        mods = mods[sorted_indices]
        
        modality_id = 1
        for mod_idx, (mod_type, mod_offset, mod_length) in enumerate(mods):
            mod_offset = int(mod_offset)
            mod_length = int(mod_length)
            mod_end = mod_offset + mod_length
            token_types[b, mod_offset:mod_end] = modality_id
            modality_ids[(b, mod_idx)] = modality_id
            modality_id += 1
    
    # Build block mask
    kv_num_blocks = torch.zeros(batch, nheads, num_q_blocks, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(batch, nheads, num_q_blocks, max_blocks, dtype=torch.int32, device=device)
    block_mask_types = torch.zeros(batch, nheads, num_q_blocks, max_blocks, dtype=torch.int32, device=device)
    
    for q_block in range(num_q_blocks):
        q_start = q_block * Q_BLOCK_SIZE
        q_end = min(q_start + Q_BLOCK_SIZE, q_len_padded)
        q_effective_end = min(q_end, q_len)  # Effective end (excluding padding)
        
        for b in range(batch):
            # Get Q block token types and check if it's "pure" (single type) or "mixed"
            q_vals = token_types[b, q_start:q_end]  # [block_size]
            q_unique = torch.unique(q_vals)
            q_is_pure = len(q_unique) == 1 or (len(q_unique) == 2 and 0 in q_unique and (q_vals != 0).sum() == 0)
            q_dominant_type = q_unique[0].item() if len(q_unique) > 0 else 0
            if not q_is_pure and len(q_unique) > 0:
                # Get dominant non-zero type
                non_zero = q_vals[q_vals != 0]
                if len(non_zero) > 0:
                    q_unique_nonzero, q_counts = torch.unique(non_zero, return_counts=True)
                    q_dominant_type = q_unique_nonzero[q_counts.argmax()].item()
            
            num_active = 0
            for kv_block in range(num_kv_blocks):
                kv_start = kv_block * KV_BLOCK_SIZE
                kv_end = min(kv_start + KV_BLOCK_SIZE, kv_len_padded)
                kv_effective_end = min(kv_end, kv_len)  # Effective end (excluding padding)
                
                # Mask padding blocks (beyond original sequence length)
                if kv_start >= kv_len or q_start >= q_len:
                    continue  # Skip this KV block
                
                # Get KV block token types and check if it's "pure" or "mixed"
                kv_vals = token_types[b, kv_start:kv_end]  # [block_size]
                kv_unique = torch.unique(kv_vals)
                kv_is_pure = len(kv_unique) == 1 or (len(kv_unique) == 2 and 0 in kv_unique and (kv_vals != 0).sum() == 0)
                kv_dominant_type = kv_unique[0].item() if len(kv_unique) > 0 else 0
                if not kv_is_pure and len(kv_unique) > 0:
                    # Get dominant non-zero type
                    non_zero = kv_vals[kv_vals != 0]
                    if len(non_zero) > 0:
                        kv_unique_nonzero, kv_counts = torch.unique(non_zero, return_counts=True)
                        kv_dominant_type = kv_unique_nonzero[kv_counts.argmax()].item()
                
                # Check if blocks overlap (for same-block detection)
                blocks_overlap = not (kv_effective_end <= q_start or q_effective_end <= kv_start)
                same_block = (q_block == kv_block)
                
                # Check if block pair crosses modality boundaries (PARTIAL case)
                # This happens when:
                # 1. Either block is mixed (contains multiple token types)
                # 2. Blocks overlap and have different types
                is_partial = False
                if not q_is_pure or not kv_is_pure:
                    is_partial = True
                elif blocks_overlap and q_dominant_type != kv_dominant_type:
                    is_partial = True
                elif same_block and not q_is_pure:
                    is_partial = True
                
                # Determine mask type based on dominant types and block properties
                if is_partial:
                    # PARTIAL: Block contains multiple types or crosses boundaries
                    # Use conservative mask type based on dominant types
                    if q_dominant_type == 0 and kv_dominant_type == 0:
                        # Both text (dominant) - CAUSAL
                        mask_type = BlockMaskType.PARTIAL  # Will need per-token check
                    elif q_dominant_type > 0 and kv_dominant_type > 0:
                        if q_dominant_type == kv_dominant_type:
                            # Same modality but mixed - PARTIAL
                            mask_type = BlockMaskType.PARTIAL
                        elif kv_effective_end <= q_start:
                            # KV before Q - PARTIAL (may have mixed content)
                            mask_type = BlockMaskType.PARTIAL
                        else:
                            mask_type = BlockMaskType.MASKED
                    else:
                        # Mixed text/modality - PARTIAL
                        mask_type = BlockMaskType.PARTIAL
                elif q_dominant_type > 0 and kv_dominant_type > 0:
                    # Both are pure modalities
                    if q_dominant_type == kv_dominant_type:
                        # Same modality - FULL attention
                        mask_type = BlockMaskType.FULL
                    elif kv_effective_end <= q_start:
                        # KV is before Q - FULL (image can attend to all previous)
                        mask_type = BlockMaskType.FULL
                    else:
                        # Different modalities, KV not before - MASKED
                        mask_type = BlockMaskType.MASKED
                elif q_dominant_type == 0 and kv_dominant_type > 0:
                    # Q is text, KV is modality
                    if kv_effective_end <= q_start:
                        # KV is before Q - FULL (text can attend to past images)
                        mask_type = BlockMaskType.FULL
                    else:
                        # KV is after Q - MASKED (text is causal, can't see future)
                        mask_type = BlockMaskType.MASKED
                elif q_dominant_type > 0 and kv_dominant_type == 0:
                    # Q is modality, KV is text
                    if kv_effective_end <= q_start:
                        mask_type = BlockMaskType.FULL
                    else:
                        mask_type = BlockMaskType.MASKED
                else:
                    # Both are text (pure)
                    if same_block:
                        # Same block - CAUSAL (text within same block is causal)
                        mask_type = BlockMaskType.CAUSAL
                    elif kv_effective_end <= q_start:
                        # KV is entirely before Q - FULL
                        mask_type = BlockMaskType.FULL
                    elif q_effective_end <= kv_start:
                        # Q is entirely before KV - MASKED
                        mask_type = BlockMaskType.MASKED
                    else:
                        # Overlapping blocks - CAUSAL
                        mask_type = BlockMaskType.CAUSAL
                
                if mask_type != BlockMaskType.MASKED:
                    kv_indices[b, :, q_block, num_active] = kv_block
                    block_mask_types[b, :, q_block, num_active] = mask_type
                    num_active += 1
            
            kv_num_blocks[b, :, q_block] = num_active
    
    omni_block_mask = OmniBlockMask(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        block_mask_types=block_mask_types,
        q_len=q_len_padded,  # Return padded length
        kv_len=kv_len_padded,  # Return padded length
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
    )
    
    if (block_mask_types == BlockMaskType.PARTIAL).any():
        from transfusion_pytorch.transfusion import naive_attn_mask
        
        dense_mask_tokens = naive_attn_mask(q_len, modality_positions, device=device)  # [B, q_len, q_len]
        dense_mask_tokens = dense_mask_tokens.unsqueeze(1).expand(-1, nheads, -1, -1)
        dense_mask = torch.zeros(
            batch, nheads, q_len_padded, kv_len_padded, dtype=torch.bool, device=device
        )
        dense_mask[:, :, :q_len, :kv_len] = dense_mask_tokens
        
        partial_indices, partial_masks = build_partial_block_data(omni_block_mask, dense_mask)
        omni_block_mask.partial_block_mask_indices = partial_indices
        omni_block_mask.partial_block_masks = partial_masks
    
    return omni_block_mask


def generate_random_input_with_modalities(
    B: int,
    N: int,
    D: int,
    min_text_len: int = 32,
    max_text_len: int = 128,
    min_image_size: int = 4,
    max_image_size: int = 32,
    vocab_size: int = 512,
    device: Union[str, torch.device] = "cuda",
    seed: Optional[int] = None,
) -> Tuple[List, Tensor]:
    """Generate random input with 2-5 modality segments.
    
    Args:
        B: Batch size
        N: Target sequence length
        D: Model dimension
        min_text_len: Minimum text segment length
        max_text_len: Maximum text segment length
        min_image_size: Minimum image dimension (h or w)
        max_image_size: Maximum image dimension (h or w)
        num_modality_segments: Number of modality segments (2-5, random if None)
        device: Device
        seed: Random seed
        
    Returns:
        text_and_images: List of B samples, each containing alternating text and images
        modality_positions: [B, M, 3] tensor of (modality_type, offset, length)
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    text_and_images = []
    all_modality_positions = []
    
    for b in range(B):
        batch_sample = []
        batch_modality_positions = []
        offset = 0
        current_length = 0
        text_in_row = 0
        image_in_row = 0

        is_last_segment = False

        while current_length < N:
            remaining = N - current_length
            if remaining <= 0:
                break

            if text_in_row >= 2:
                text_or_image = 1
            elif image_in_row >= 2:
                text_or_image = 0
            else:
                text_or_image = torch.randint(0, 2, ()).item()
            
            if text_or_image == 0:
                # Generate text segment
                text_in_row += 1
                image_in_row = 0
                if remaining > max_text_len:
                    text_len = torch.randint(min_text_len, max_text_len + 1, (1,)).item()
                elif remaining >= min_text_len:
                    text_len = torch.randint(min_text_len, remaining + 1, (1,)).item()
                else:
                    # Use all remaining tokens for text (even if < min_text_len)
                    text_len = remaining
                
                text_tokens = randint(0, vocab_size, (text_len,), device=device)
                batch_sample.append(text_tokens)
                batch_modality_positions.append((1, offset, text_len))  # 1 = text modality
                offset += text_len
                current_length += text_len
                
                if current_length >= N:
                    break
            else:
                # Generate image segment
                image_in_row += 1
                text_in_row = 0
                max_image_len = max_image_size * max_image_size
                min_image_len = min_image_size * min_image_size
                
                if remaining > max_image_len:
                    image_len = torch.randint(min_image_len, max_image_len + 1, (1,)).item()
                elif remaining >= min_image_len:
                    image_len = torch.randint(min_image_len, remaining + 1, (1,)).item()
                else:
                    # Use all remaining tokens for image (even if < min_image_len)
                    # Find the closest valid h*w that fits
                    image_len = remaining
                    # Adjust to valid dimensions
                    sqrt_tokens = max(1, int(image_len ** 0.5))
                    h_size = max(1, min(sqrt_tokens, max_image_size))
                    w_size = max(1, min(image_len // h_size, max_image_size))
                    image_len = h_size * w_size
                    # If still doesn't match, use remaining directly
                    if image_len > remaining:
                        image_len = remaining
                        h_size = 1
                        w_size = image_len
                
                # Find h, w such that h*w = image_len (if not already set)
                if image_len >= min_image_len:
                    sqrt_tokens = int(image_len ** 0.5)
                    h_size = max(min_image_size, min(sqrt_tokens, max_image_size))
                    w_size = max(min_image_size, min(image_len // h_size, max_image_size))
                    image_len = h_size * w_size
                
                image = randn(h_size, w_size, D, device=device)
                batch_sample.append(image)
                batch_modality_positions.append((0, offset, image_len))  # 0 = image modality
                offset += image_len
                current_length += image_len
                
                if current_length >= N:
                    break
        
            print(f"text_or_image: ({'text' if text_or_image == 0 else 'image'}), current_length: {current_length}, remaining: {remaining}")

        # Ensure we have exactly N tokens (pad if needed, truncate if over)
        total_generated = sum(
            len(item) if item.dim() == 1 else item.shape[0] * item.shape[1]
            for item in batch_sample
        )
        
        if total_generated < N:
            # Pad with text tokens
            pad_len = N - total_generated
            pad_tokens = randint(0, vocab_size, (pad_len,), device=device)
            batch_sample.append(pad_tokens)
            # Note: Padding text is NOT added to modality_positions (text is default)
        elif total_generated > N:
            # Truncate last segment
            excess = total_generated - N
            last_item = batch_sample[-1]
            if last_item.dim() == 1:
                # Text: truncate
                batch_sample[-1] = last_item[:len(last_item) - excess]
                batch_modality_positions[-1] = (batch_modality_positions[-1][0], 
                                                batch_modality_positions[-1][1],
                                                batch_modality_positions[-1][2] - excess)
            else:
                # Image: adjust dimensions
                h, w = last_item.shape[:2]
                total_tokens = h * w
                new_total = total_tokens - excess
                # Find new h, w
                sqrt_tokens = max(1, int(new_total ** 0.5))
                new_h = max(1, min(sqrt_tokens, h))
                new_w = max(1, min(new_total // new_h, w))
                new_total = new_h * new_w
                batch_sample[-1] = last_item[:new_h, :new_w]
                batch_modality_positions[-1] = (batch_modality_positions[-1][0],
                                                batch_modality_positions[-1][1],
                                                new_total)
    
        text_and_images.append(batch_sample)
        all_modality_positions.append(batch_modality_positions)
    
    # Convert modality positions to tensor
    try:
        from transfusion_pytorch.transfusion import modality_positions_to_tensor
        modality_positions = modality_positions_to_tensor(all_modality_positions, device=device)
    except ImportError:
        # Fallback: create tensor manually
        max_m = max(len(pos) for pos in all_modality_positions)
        modality_positions = torch.zeros(B, max_m, 3, dtype=torch.int64, device=device)
        for b, pos_list in enumerate(all_modality_positions):
            for m, (mod_type, offset, length) in enumerate(pos_list):
                modality_positions[b, m, 0] = mod_type
                modality_positions[b, m, 1] = offset
                modality_positions[b, m, 2] = length
    
    return text_and_images, modality_positions


def generate_causal_input(
    B: int,
    N: int,
    D: int,
    device: Union[str, torch.device] = "cuda",
) -> Tuple[List, Tensor]:
    """Generate causal input with modalities.
    
    Creates a simple text-only input for causal attention testing.
    
    Args:
        B: Batch size
        N: Sequence length
        D: Model dimension (not used, kept for compatibility)
        device: Device
        
    Returns:
        text_and_images: List of B samples, each with one text segment
        modality_positions: [B, 1, 3] tensor of (modality_type, offset, length)
    """
    text_and_images = []
    modality_positions = []

    # one segment of each batch, which is causal
    for b in range(B):
        # Create text tokens (random integers)
        text_tokens = torch.randint(0, 512, (N,), device=device, dtype=torch.long)
        text_and_images.append([text_tokens])
        modality_positions.append([(1, 0, N)])  # modality_type=1 (text), offset=0, length=N
    
    return text_and_images, torch.tensor(modality_positions, device=device)

def create_causal_omni_block_mask(
    batch: int,
    nheads: int,
    q_len: int,
    kv_len: int,
    Q_BLOCK_SIZE: int = 128,
    KV_BLOCK_SIZE: int = 128,
    device: Union[str, torch.device] = "cuda",
) -> OmniBlockMask:
    """Create a simple causal mask in OmniBlockMask format.
    
    Pattern:
    - Diagonal blocks: CAUSAL
    - Below diagonal (left-bottom): FULL
    - Above diagonal (top-right): MASKED
    
    Args:
        batch: Batch size
        nheads: Number of attention heads
        q_len: Query sequence length (will be padded to Q_BLOCK_SIZE multiple)
        kv_len: Key/value sequence length (will be padded to KV_BLOCK_SIZE multiple)
        Q_BLOCK_SIZE: Block size for query
        KV_BLOCK_SIZE: Block size for key/value
        device: Device
        
    Returns:
        OmniBlockMask with causal attention pattern
    """
    # Pad to BLOCK_SIZE multiple
    q_len_padded = ((q_len + Q_BLOCK_SIZE - 1) // Q_BLOCK_SIZE) * Q_BLOCK_SIZE
    kv_len_padded = ((kv_len + KV_BLOCK_SIZE - 1) // KV_BLOCK_SIZE) * KV_BLOCK_SIZE
    
    num_q_blocks = q_len_padded // Q_BLOCK_SIZE
    num_kv_blocks = kv_len_padded // KV_BLOCK_SIZE
    max_blocks = num_kv_blocks  # Maximum number of KV blocks any Q block can attend to
    
    # Initialize tensors
    kv_num_blocks = torch.zeros(batch, nheads, num_q_blocks, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(batch, nheads, num_q_blocks, max_blocks, dtype=torch.int32, device=device)
    block_mask_types = torch.zeros(batch, nheads, num_q_blocks, max_blocks, dtype=torch.int32, device=device)
    
    for q_block in range(num_q_blocks):
        num_active = 0
        for kv_block in range(q_block + 1):
            if kv_block < q_block:
                # Below diagonal: FULL
                mask_type = BlockMaskType.FULL
            else:
                # Diagonal: CAUSAL
                mask_type = BlockMaskType.CAUSAL
            
            kv_indices[:, :, q_block, num_active] = kv_block
            block_mask_types[:, :, q_block, num_active] = mask_type
            num_active += 1
        
        kv_num_blocks[:, :, q_block] = num_active
    
    return OmniBlockMask(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        block_mask_types=block_mask_types,
        q_len=q_len_padded,
        kv_len=kv_len_padded,
        Q_BLOCK_SIZE=Q_BLOCK_SIZE,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
    )

def get_embeddings_from_text_and_images(
    text_and_images: List,
    D: int,
    device: Union[str, torch.device] = "cuda",
) -> Tensor:
    """Convert text_and_images to embeddings tensor [B, N, D].
    
    Args:
        text_and_images: List of B samples, each containing alternating text and images
        D: Model dimension
        device: Device
        
    Returns:
        embeddings: [B, N, D] tensor
    """
    from torch.nn.utils.rnn import pad_sequence
    
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
    x = pad_sequence(batch_embeddings, batch_first=True, padding_value=0.0)
    
    return x


def print_block_structure(
    modality_positions: Tensor,
    q_len: int,
    batch_idx: int = 0,
    Q_BLOCK_SIZE: int = 128,
    KV_BLOCK_SIZE: int = 128,
) -> None:
    """Print block structure in the format shown in the terminal.
    
    Args:
        modality_positions: [B, M, 3] tensor of (modality_type, offset, length)
        q_len: Sequence length
        batch_idx: Which batch to print
        Q_BLOCK_SIZE: Block size for query
        KV_BLOCK_SIZE: Block size for key/value

    print something like(not in Q_BLOCK_SIZE and KV_BLOCK_SIZE chunks):

    Modality ranges: 
    text[0:70] (len=70) 
    text[71:126] (len=55) 
    img0[126:230] (len=104) 
    text[230:294] (len=64) 
    img1[294:380] (len=86) 

    Block types:
    Q0[0:70]=text
    Q1[70:126]=text
    Q2[126:230]=img0
    Q3[230:294]=text
    Q4[294:380]=img1

    Mask (Q blocks: 4, F=FULL, C=CAUSAL, M=MASKED):
    Q\\KV   0  1  2  3  4
    --------------------
    0   C  M  M  M  M
    1   F  C  M  M  M
    2   F  F  F  M  M
    3   F  F  F  C  M
    4   F  F  F  F  F
    """
    device = modality_positions.device
    
    # Get image modalities
    mods = modality_positions[batch_idx]
    valid_mask = mods[:, 2] > 0
    mods = mods[valid_mask]
    
    # Sort by offset
    if len(mods) > 0:
        sorted_indices = mods[:, 1].argsort()
        mods = mods[sorted_indices]
    
    # Build list of all segments (text and image) in order
    # Note: mod_type: 0 = image, 1 = text
    segments = []
    img_counter = 0
    
    # Process each modality segment in order (already sorted by offset)
    for mod_type, mod_offset, mod_length in mods:
        mod_type = int(mod_type)
        mod_offset = int(mod_offset)
        mod_length = int(mod_length)
        mod_end = mod_offset + mod_length
        
        if mod_type == 0:
            # Image modality
            segments.append(("img", img_counter, mod_offset, mod_end))
            img_counter += 1
        elif mod_type == 1:
            # Text modality
            segments.append(("text", mod_offset, mod_end))
        # else: unknown type, skip
    
    # Fill in any gaps (regions not covered by explicit modality segments)
    # This handles padding or regions before/after all segments
    if len(segments) == 0:
        # No segments at all, assume all text
        if q_len > 0:
            segments.append(("text", 0, q_len))
    else:
        # Check for gap before first segment
        first_seg = segments[0]
        if first_seg[0] == "text":
            first_start = first_seg[1]
        else:
            first_start = first_seg[2]  # img: (name, idx, start, end)
        if first_start > 0:
            segments.insert(0, ("text", 0, first_start))
        
        # Check for gaps between segments
        i = 0
        while i < len(segments) - 1:
            seg1 = segments[i]
            seg2 = segments[i + 1]
            
            # Get end of seg1
            if seg1[0] == "text":
                seg1_end = seg1[2]
            else:
                seg1_end = seg1[3]  # img: (name, idx, start, end)
            
            # Get start of seg2
            if seg2[0] == "text":
                seg2_start = seg2[1]
            else:
                seg2_start = seg2[2]  # img: (name, idx, start, end)
            
            # If there's a gap, insert text segment
            if seg2_start > seg1_end:
                segments.insert(i + 1, ("text", seg1_end, seg2_start))
                i += 2  # Skip the newly inserted segment
            else:
                i += 1
        
        # Check for gap after last segment
        last_seg = segments[-1]
        if last_seg[0] == "text":
            last_end = last_seg[2]
        else:
            last_end = last_seg[3]  # img: (name, idx, start, end)
        if last_end < q_len:
            segments.append(("text", last_end, q_len))
    
    # Print block types (each segment is a block)
    print("Block types:")
    for q_idx, seg in enumerate(segments):
        if seg[0] == "text":
            start, end = seg[1], seg[2]
            print(f"Q{q_idx}[{start}:{end}]=text, len={end-start}")
        else:
            start, end = seg[2], seg[3]
            img_idx = seg[1]
            print(f"Q{q_idx}[{start}:{end}]=img{img_idx}, len={end-start}")
    
    # Print mask based on segment types and positions
    num_q_blocks = len(segments)
    print(f"Mask (Q blocks: {num_q_blocks}, F=FULL, C=CAUSAL, M=MASKED):")
    print(r"Q\KV  ", end="")
    for kv_block in range(num_q_blocks):
        print(f"{kv_block:2d} ", end="")
    print()
    print("-" * (6 + 3 * num_q_blocks))
    
    for q_seg_idx in range(num_q_blocks):
        q_seg = segments[q_seg_idx]
        q_is_text = q_seg[0] == "text"
        if q_is_text:
            q_start, q_end = q_seg[1], q_seg[2]
        else:
            q_start, q_end = q_seg[2], q_seg[3]
            q_img_idx = q_seg[1]
        
        print(f"  {q_seg_idx:2d}  ", end="")
        
        for kv_seg_idx in range(num_q_blocks):
            kv_seg = segments[kv_seg_idx]
            kv_is_text = kv_seg[0] == "text"
            if kv_is_text:
                kv_start, kv_end = kv_seg[1], kv_seg[2]
            else:
                kv_start, kv_end = kv_seg[2], kv_seg[3]
                kv_img_idx = kv_seg[1]
 
            
            if q_is_text and kv_is_text:
                if q_seg_idx == kv_seg_idx:
                    mask_type = BlockMaskType.CAUSAL
                elif kv_end <= q_start:
                    mask_type = BlockMaskType.FULL
                else:
                    mask_type = BlockMaskType.MASKED
            elif not q_is_text and kv_is_text:
                if kv_end <= q_start:
                    mask_type = BlockMaskType.FULL
                else:
                    mask_type = BlockMaskType.MASKED
            elif q_is_text and not kv_is_text:
                if kv_end <= q_start:
                    mask_type = BlockMaskType.FULL
                else:
                    mask_type = BlockMaskType.MASKED
            else:
                if q_img_idx == kv_img_idx:
                    mask_type = BlockMaskType.FULL
                elif kv_end <= q_start:
                    mask_type = BlockMaskType.FULL
                else:
                    mask_type = BlockMaskType.MASKED
            
            if mask_type == BlockMaskType.FULL:
                print(" F ", end="")
            elif mask_type == BlockMaskType.CAUSAL:
                print(" C ", end="")
            else:
                print(" M ", end="")
        print()
