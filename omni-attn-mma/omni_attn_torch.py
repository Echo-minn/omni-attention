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
from torch import Tensor


# ============================================================================
# Block Mask Types
# ============================================================================

class BlockMaskType(IntEnum):
    """Block mask types for Omni-Attention.
    
    MASKED (0): Block is fully masked out - skip computation entirely
    CAUSAL (1): Block uses causal masking (q_idx >= kv_idx)
    FULL (2): Block has no masking - full attention
    """
    MASKED = 0  # Skip block entirely (don't load K/V)
    CAUSAL = 1  # Apply causal masking within block
    FULL = 2    # No masking (dense attention)


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
        BLOCK_SIZE: Block size (default 128)
    """
    kv_num_blocks: Tensor  # [B, H, num_q_blocks]
    kv_indices: Tensor     # [B, H, num_q_blocks, max_blocks]
    block_mask_types: Tensor  # [B, H, num_q_blocks, max_blocks]
    q_len: int
    kv_len: int
    BLOCK_SIZE: int = 128
    
    @property
    def num_q_blocks(self) -> int:
        return (self.q_len + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
    
    @property
    def num_kv_blocks(self) -> int:
        return (self.kv_len + self.BLOCK_SIZE - 1) // self.BLOCK_SIZE
    
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
                    q_start = q_block * self.BLOCK_SIZE
                    q_end = min(q_start + self.BLOCK_SIZE, self.q_len)
                    
                    num_active = self.kv_num_blocks[b, h, q_block].item()
                    
                    for idx in range(num_active):
                        kv_block = self.kv_indices[b, h, q_block, idx].item()
                        mask_type = self.block_mask_types[b, h, q_block, idx].item()
                        
                        kv_start = kv_block * self.BLOCK_SIZE
                        kv_end = min(kv_start + self.BLOCK_SIZE, self.kv_len)
                        
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
        
        return dense_mask
    
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


# ============================================================================
# Block Mask Creation Utilities
# ============================================================================

def create_causal_block_mask(
    batch: int,
    nheads: int,
    q_len: int,
    kv_len: int,
    BLOCK_SIZE: int = 128,
    device: Union[str, torch.device] = "cuda",
) -> OmniBlockMask:
    """Create a causal (lower triangular) block mask.
    
    Args:
        batch: Batch size
        nheads: Number of attention heads
        q_len: Query sequence length
        kv_len: Key/value sequence length
        BLOCK_SIZE: Block size
        device: Device to create tensors on
        
    Returns:
        OmniBlockMask with causal pattern
    """
    num_q_blocks = (q_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_kv_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    max_blocks = num_kv_blocks
    
    kv_num_blocks = torch.zeros(batch, nheads, num_q_blocks, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(batch, nheads, num_q_blocks, max_blocks, dtype=torch.int32, device=device)
    block_mask_types = torch.zeros(batch, nheads, num_q_blocks, max_blocks, dtype=torch.int32, device=device)
    
    for q_block in range(num_q_blocks):
        q_block_end = (q_block + 1) * BLOCK_SIZE - 1  # Last Q index in this block
        
        # Determine which KV blocks this Q block can attend to
        num_active = 0
        for kv_block in range(num_kv_blocks):
            kv_block_start = kv_block * BLOCK_SIZE
            kv_block_end = (kv_block + 1) * BLOCK_SIZE - 1
            
            if kv_block_start > q_block_end:
                # KV block is entirely after Q block - masked out
                continue
            
            if kv_block_end <= q_block_end:
                # KV block is entirely at or before Q block end - FULL attention
                mask_type = BlockMaskType.FULL
            else:
                # KV block overlaps Q block end - CAUSAL
                mask_type = BlockMaskType.CAUSAL
            
            kv_indices[:, :, q_block, num_active] = kv_block
            block_mask_types[:, :, q_block, num_active] = mask_type
            num_active += 1
        
        kv_num_blocks[:, :, q_block] = num_active
    
    return OmniBlockMask(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        block_mask_types=block_mask_types,
        q_len=q_len,
        kv_len=kv_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def create_full_block_mask(
    batch: int,
    nheads: int,
    q_len: int,
    kv_len: int,
    BLOCK_SIZE: int = 128,
    device: Union[str, torch.device] = "cuda",
) -> OmniBlockMask:
    """Create a full (dense) attention block mask.
    
    Args:
        batch: Batch size
        nheads: Number of attention heads
        q_len: Query sequence length
        kv_len: Key/value sequence length
        BLOCK_SIZE: Block size
        device: Device to create tensors on
        
    Returns:
        OmniBlockMask with full attention pattern
    """
    num_q_blocks = (q_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_kv_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    max_blocks = num_kv_blocks
    
    kv_num_blocks = torch.full((batch, nheads, num_q_blocks), num_kv_blocks, dtype=torch.int32, device=device)
    kv_indices = torch.arange(num_kv_blocks, dtype=torch.int32, device=device).expand(batch, nheads, num_q_blocks, -1).contiguous()
    block_mask_types = torch.full((batch, nheads, num_q_blocks, max_blocks), BlockMaskType.FULL, dtype=torch.int32, device=device)
    
    return OmniBlockMask(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        block_mask_types=block_mask_types,
        q_len=q_len,
        kv_len=kv_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def create_random_vlm_block_mask(
    batch: int,
    nheads: int,
    q_len: int,
    kv_len: int,
    num_image_regions: int = 2,
    image_region_size: int = 256,
    BLOCK_SIZE: int = 128,
    device: Union[str, torch.device] = "cuda",
    seed: Optional[int] = None,
) -> OmniBlockMask:
    """Create a random VLM-style interleaved block mask.
    
    This simulates a Vision-Language Model pattern where:
    - Image token regions have FULL attention within themselves
    - Image regions are MASKED from each other
    - Text regions have CAUSAL attention
    - Text can attend to previous image regions (FULL)
    
    Args:
        batch: Batch size
        nheads: Number of attention heads
        q_len: Query sequence length
        kv_len: Key/value sequence length  
        num_image_regions: Number of image regions
        image_region_size: Size of each image region in tokens
        BLOCK_SIZE: Block size
        device: Device to create tensors on
        seed: Random seed for reproducibility
        
    Returns:
        OmniBlockMask with VLM-style pattern
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    num_q_blocks = (q_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_kv_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    max_blocks = num_kv_blocks
    
    # Generate random image region positions
    total_image_size = num_image_regions * image_region_size
    text_size = q_len - total_image_size
    
    if text_size < 0:
        # Not enough space for image regions, fall back to causal
        return create_causal_block_mask(batch, nheads, q_len, kv_len, BLOCK_SIZE, device)
    
    # Randomly place image regions
    # For simplicity, place them at evenly spaced intervals
    spacing = q_len // (num_image_regions + 1)
    image_regions = []
    for i in range(num_image_regions):
        start = (i + 1) * spacing - image_region_size // 2
        start = max(0, min(start, q_len - image_region_size))
        end = start + image_region_size
        image_regions.append((start, end))
    
    # Create token type array: 0 = text, 1+ = image region id
    token_types = torch.zeros(q_len, dtype=torch.int32, device=device)
    for i, (start, end) in enumerate(image_regions):
        token_types[start:end] = i + 1
    
    # Build block mask
    kv_num_blocks = torch.zeros(batch, nheads, num_q_blocks, dtype=torch.int32, device=device)
    kv_indices = torch.zeros(batch, nheads, num_q_blocks, max_blocks, dtype=torch.int32, device=device)
    block_mask_types = torch.zeros(batch, nheads, num_q_blocks, max_blocks, dtype=torch.int32, device=device)
    
    for q_block in range(num_q_blocks):
        q_start = q_block * BLOCK_SIZE
        q_end = min(q_start + BLOCK_SIZE, q_len)
        
        # Get dominant token type in this Q block
        q_types = token_types[q_start:q_end]
        q_type = q_types.mode().values.item()  # Most common type
        
        num_active = 0
        for kv_block in range(num_kv_blocks):
            kv_start = kv_block * BLOCK_SIZE
            kv_end = min(kv_start + BLOCK_SIZE, kv_len)
            
            # Get dominant token type in this KV block
            kv_types = token_types[kv_start:kv_end]
            kv_type = kv_types.mode().values.item()
            
            # Determine mask type based on token types
            if q_type > 0 and kv_type > 0:
                # Both are image regions
                if q_type == kv_type:
                    # Same image region - FULL attention
                    mask_type = BlockMaskType.FULL
                else:
                    # Different image regions - MASKED
                    mask_type = BlockMaskType.MASKED
            elif q_type == 0 and kv_type > 0:
                # Q is text, KV is image - FULL attention to past images
                if kv_end <= q_start:
                    mask_type = BlockMaskType.FULL
                else:
                    mask_type = BlockMaskType.MASKED
            elif q_type > 0 and kv_type == 0:
                # Q is image, KV is text - MASKED (images don't attend to text)
                mask_type = BlockMaskType.MASKED
            else:
                # Both are text - CAUSAL
                if kv_start > q_end:
                    mask_type = BlockMaskType.MASKED
                elif kv_end <= q_start:
                    mask_type = BlockMaskType.FULL
                else:
                    mask_type = BlockMaskType.CAUSAL
            
            if mask_type != BlockMaskType.MASKED:
                kv_indices[:, :, q_block, num_active] = kv_block
                block_mask_types[:, :, q_block, num_active] = mask_type
                num_active += 1
        
        kv_num_blocks[:, :, q_block] = num_active
    
    return OmniBlockMask(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        block_mask_types=block_mask_types,
        q_len=q_len,
        kv_len=kv_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def create_random_sparse_block_mask(
    batch: int,
    nheads: int,
    q_len: int,
    kv_len: int,
    sparsity: float = 0.5,
    BLOCK_SIZE: int = 128,
    device: Union[str, torch.device] = "cuda",
    seed: Optional[int] = None,
) -> OmniBlockMask:
    """Create a random sparse block mask with approximate sparsity level.
    
    Args:
        batch: Batch size
        nheads: Number of attention heads
        q_len: Query sequence length
        kv_len: Key/value sequence length
        sparsity: Target sparsity ratio (0.0 = dense, 1.0 = fully masked)
        BLOCK_SIZE: Block size
        device: Device to create tensors on
        seed: Random seed for reproducibility
        
    Returns:
        OmniBlockMask with random sparse pattern
    """
    if seed is not None:
        torch.manual_seed(seed)
    
    num_q_blocks = (q_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_kv_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    max_blocks = num_kv_blocks
    
    # Generate random block mask
    # Sample which blocks are active (1 - sparsity probability)
    active_prob = 1.0 - sparsity
    active_mask = torch.rand(batch, nheads, num_q_blocks, num_kv_blocks, device=device) < active_prob
    
    # Randomly assign mask types to active blocks
    # 50% FULL, 50% CAUSAL for active blocks (for variety)
    type_random = torch.rand(batch, nheads, num_q_blocks, num_kv_blocks, device=device)
    block_types = torch.where(type_random < 0.5, BlockMaskType.FULL, BlockMaskType.CAUSAL)
    block_types = torch.where(active_mask, block_types, BlockMaskType.MASKED)
    
    # Convert to sparse format
    kv_num_blocks = active_mask.sum(dim=-1).to(torch.int32)
    max_active = kv_num_blocks.max().item()
    
    kv_indices = torch.zeros(batch, nheads, num_q_blocks, max_blocks, dtype=torch.int32, device=device)
    block_mask_types = torch.zeros(batch, nheads, num_q_blocks, max_blocks, dtype=torch.int32, device=device)
    
    # Fill in sparse format (this is slow but only for test setup)
    for b in range(batch):
        for h in range(nheads):
            for q in range(num_q_blocks):
                idx = 0
                for kv in range(num_kv_blocks):
                    if active_mask[b, h, q, kv]:
                        kv_indices[b, h, q, idx] = kv
                        block_mask_types[b, h, q, idx] = block_types[b, h, q, kv]
                        idx += 1
    
    return OmniBlockMask(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        block_mask_types=block_mask_types,
        q_len=q_len,
        kv_len=kv_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )


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
        BLOCK_SIZE=block_mask.BLOCK_SIZE,
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
        BLOCK_SIZE=block_mask.BLOCK_SIZE,
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

def omni_attention_mma(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_mask: OmniBlockMask,
    scale: Optional[float] = None,
    stages: int = 2,
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
        import omni_attn_mma_cuda
    except ImportError:
        raise ImportError(
            "CUDA MMA kernel not found. Please build the extension first:\n"
            "cd omni-attn-mma && python setup.py install"
        )
    
    B, H, Q, D = query.shape
    
    # Ensure contiguous and correct dtype
    query = query.contiguous().half()
    key = key.contiguous().half()
    value = value.contiguous().half()
    
    # Create output tensor
    output = torch.empty_like(query)
    
    # Call CUDA kernel
    omni_attn_mma_cuda.omni_attn_mma_stages_split_q_shared_kv(
        query, key, value, output,
        block_mask.kv_num_blocks,
        block_mask.kv_indices,
        block_mask.block_mask_types,
        stages,
    )
    
    return output


# ============================================================================
# Utility Functions
# ============================================================================

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
    
    ref_output = ref_output.float()
    test_output = test_output.float()
    
    max_diff = (ref_output - test_output).abs().max().item()
    rel_diff = ((ref_output - test_output).abs() / (ref_output.abs() + 1e-8)).max().item()
    
    passed = torch.allclose(ref_output, test_output, rtol=rtol, atol=atol)
    
    if passed:
        print(f"[{name}] PASSED (max_diff={max_diff:.6f}, rel_diff={rel_diff:.6f})")
    else:
        print(f"[{name}] FAILED (max_diff={max_diff:.6f}, rel_diff={rel_diff:.6f})")
    
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
