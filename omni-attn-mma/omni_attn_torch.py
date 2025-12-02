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
    # Set up LD_LIBRARY_PATH for PyTorch libraries if needed
    import os
    try:
        import torch
        torch_lib_path = os.path.join(os.path.dirname(torch.__file__), 'lib')
        if os.path.exists(torch_lib_path):
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            if torch_lib_path not in current_ld_path:
                os.environ['LD_LIBRARY_PATH'] = f"{torch_lib_path}:{current_ld_path}" if current_ld_path else torch_lib_path
    except Exception:
        pass  # Continue even if torch import fails
    
    try:
        from omni_attn import omni_attn_mma_stages_split_q_shared_kv
    except (ImportError, OSError) as e:
        # Try loading from current directory if not installed
        import sys
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
                except Exception as load_err:
                    continue
        
        if module is None:
            raise ImportError(
                "CUDA MMA kernel not found. Please build the extension first:\n"
                "cd omni-attn-mma && python setup.py build_ext --inplace\n"
                f"Or ensure the .so file is in: {current_dir}\n"
                f"Original error: {e}"
            )
    
    B, H, Q, D = query.shape
    _, _, KV, _ = key.shape
    
    # Validate shapes
    assert key.shape == (B, H, KV, D), f"Key shape mismatch: {key.shape} vs expected {(B, H, KV, D)}"
    assert value.shape == (B, H, KV, D), f"Value shape mismatch: {value.shape} vs expected {(B, H, KV, D)}"
    assert Q == KV, f"Q and KV sequence lengths must match: Q={Q}, KV={KV}"
    assert Q == block_mask.q_len, f"Query length mismatch: Q={Q}, block_mask.q_len={block_mask.q_len}"
    assert KV == block_mask.kv_len, f"KV length mismatch: KV={KV}, block_mask.kv_len={block_mask.kv_len}"
    
    # Validate block mask shapes
    num_q_blocks = (Q + block_mask.BLOCK_SIZE - 1) // block_mask.BLOCK_SIZE
    assert block_mask.kv_num_blocks.shape == (B, H, num_q_blocks), \
        f"kv_num_blocks shape mismatch: {block_mask.kv_num_blocks.shape} vs expected {(B, H, num_q_blocks)}"
    max_blocks = block_mask.kv_indices.shape[3]
    assert block_mask.kv_indices.shape == (B, H, num_q_blocks, max_blocks), \
        f"kv_indices shape mismatch: {block_mask.kv_indices.shape} vs expected {(B, H, num_q_blocks, max_blocks)}"
    assert block_mask.block_mask_types.shape == (B, H, num_q_blocks, max_blocks), \
        f"block_mask_types shape mismatch: {block_mask.block_mask_types.shape} vs expected {(B, H, num_q_blocks, max_blocks)}"
    
    # Ensure contiguous and correct dtype for Q, K, V
    query = query.contiguous().half()
    key = key.contiguous().half()
    value = value.contiguous().half()
    
    # Create output tensor
    output = torch.empty_like(query)
    
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
    
    # Validate BLOCK_SIZE is supported (64 or 128)
    if block_mask.BLOCK_SIZE not in [64, 128]:
        raise ValueError(
            f"Unsupported BLOCK_SIZE={block_mask.BLOCK_SIZE}. Supported values: 64, 128. "
            f"Please use --block_size 64 or --block_size 128."
        )
    
    # Call CUDA kernel
    omni_attn_mma_stages_split_q_shared_kv(
        query, key, value, output,
        kv_num_blocks,
        kv_indices,
        block_mask_types,
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
    BLOCK_SIZE: int = 128,
    device: Union[str, torch.device] = "cuda",
) -> OmniBlockMask:
    """Create OmniBlockMask from modality positions.
    
    Mask rules:
    - Text segments: CAUSAL attention
    - Image modalities: FULL attention within same modality, FULL to all previous tokens
    - Different image modalities: MASKED (no attention between different images)
    
    Args:
        modality_positions: [B, M, 3] tensor of (modality_type, offset, length)
        batch: Batch size
        nheads: Number of attention heads
        q_len: Query sequence length
        kv_len: Key/value sequence length
        BLOCK_SIZE: Block size
        device: Device
        
    Returns:
        OmniBlockMask
    """
    num_q_blocks = (q_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_kv_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    max_blocks = num_kv_blocks
    
    # Create token type array: 0 = text, 1+ = modality id
    token_types = torch.zeros(batch, q_len, dtype=torch.int32, device=device)
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
        q_start = q_block * BLOCK_SIZE
        q_end = min(q_start + BLOCK_SIZE, q_len)
        
        # Get dominant token type in this Q block
        q_types = token_types[:, q_start:q_end]  # [B, block_size]
        # For each batch, get mode
        q_type_per_batch = []
        for b in range(batch):
            q_vals = q_types[b]
            # Get most common non-zero value, or 0 if all zeros
            unique_vals, counts = torch.unique(q_vals, return_counts=True)
            if len(unique_vals) > 0:
                mode_idx = counts.argmax()
                q_type_per_batch.append(unique_vals[mode_idx].item())
            else:
                q_type_per_batch.append(0)
        
        for b in range(batch):
            q_type = q_type_per_batch[b]
            
            num_active = 0
            for kv_block in range(num_kv_blocks):
                kv_start = kv_block * BLOCK_SIZE
                kv_end = min(kv_start + BLOCK_SIZE, kv_len)
                
                # Get dominant token type in this KV block
                kv_vals = token_types[b, kv_start:kv_end]
                unique_vals, counts = torch.unique(kv_vals, return_counts=True)
                if len(unique_vals) > 0:
                    mode_idx = counts.argmax()
                    kv_type = unique_vals[mode_idx].item()
                else:
                    kv_type = 0
                
                # Determine mask type
                # Rules:
                # - Text: CAUSAL only (Q >= KV)
                # - Image: FULL within same modality OR FULL for all previous blocks
                if q_type > 0 and kv_type > 0:
                    # Both are modalities
                    if q_type == kv_type:
                        # Same modality - FULL attention
                        mask_type = BlockMaskType.FULL
                    else:
                        # Different modalities - check if KV is before Q
                        if kv_end <= q_start:
                            # KV is before Q - FULL (image can attend to all previous)
                            mask_type = BlockMaskType.FULL
                        else:
                            # Different modalities, KV not before - MASKED
                            mask_type = BlockMaskType.MASKED
                elif q_type == 0 and kv_type > 0:
                    # Q is text, KV is modality - CAUSAL only (text can't attend to future)
                    if kv_end <= q_start:
                        # KV is before Q - FULL (text can attend to past images)
                        mask_type = BlockMaskType.FULL
                    else:
                        # KV is after Q - MASKED (text is causal, can't see future)
                        mask_type = BlockMaskType.MASKED
                elif q_type > 0 and kv_type == 0:
                    # Q is modality, KV is text - FULL if KV is before Q
                    if kv_end <= q_start:
                        mask_type = BlockMaskType.FULL
                    else:
                        mask_type = BlockMaskType.MASKED
                else:
                    # Both are text - STRICT CAUSAL (Q >= KV)
                    if kv_end <= q_start:
                        # KV is entirely before Q - FULL
                        mask_type = BlockMaskType.FULL
                    elif kv_start >= q_end:
                        # KV is entirely after Q - MASKED
                        mask_type = BlockMaskType.MASKED
                    else:
                        # Overlapping - CAUSAL (Q can attend to KV if Q >= KV)
                        # For blocks, if they overlap, use CAUSAL
                        mask_type = BlockMaskType.CAUSAL
                
                if mask_type != BlockMaskType.MASKED:
                    kv_indices[b, :, q_block, num_active] = kv_block
                    block_mask_types[b, :, q_block, num_active] = mask_type
                    num_active += 1
            
            kv_num_blocks[b, :, q_block] = num_active
    
    return OmniBlockMask(
        kv_num_blocks=kv_num_blocks,
        kv_indices=kv_indices,
        block_mask_types=block_mask_types,
        q_len=q_len,
        kv_len=kv_len,
        BLOCK_SIZE=BLOCK_SIZE,
    )


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

        while not is_last_segment:
            remaining = N - current_length
            if remaining <= 0:
                break
            
            # Alternate between text and image, either 0 or 1
            text_or_image = torch.randint(0, 2, ()).item()

            if text_in_row > 1:
                text_or_image = 1
            if image_in_row > 1:
                text_or_image = 0

            if text_or_image == 0:
                # Generate text segment
                text_in_row += 1
                image_in_row = 0
                if remaining > max_text_len:
                    text_len = torch.randint(min_text_len, max_text_len + 1, (1,)).item()
                elif remaining > min_text_len:
                    text_len = torch.randint(min_text_len, remaining + 1, (1,)).item()
                else:
                    is_last_segment = True
                    break
                
                text_tokens = randint(0, vocab_size, (text_len,), device=device)
                batch_sample.append(text_tokens)
                offset += text_len
                current_length += text_len
                print(f"text_len: {text_len}")
            else:
                # Generate image segment
                image_in_row += 1
                text_in_row = 0
                max_image_len = max_image_size * max_image_size
                min_image_len = min_image_size * min_image_size
                
                if remaining > max_image_len:
                    image_len = torch.randint(min_image_len, max_image_len + 1, (1,)).item()
                elif remaining > min_image_len:
                    image_len = torch.randint(min_image_len, remaining + 1, (1,)).item()
                else:
                    # image_len = remaining
                    is_last_segment = True
                    break
                
                # Find h, w such that h*w = image_len
                sqrt_tokens = int(image_len ** 0.5)
                h_size = max(min_image_size, min(sqrt_tokens, max_image_size))
                w_size = max(min_image_size, min(image_len // h_size, max_image_size))
                image_len = h_size * w_size
                
                image = randn(h_size, w_size, D, device=device)
                batch_sample.append(image)
                batch_modality_positions.append((0, offset, image_len))
                offset += image_len
                print(f"image_len: {image_len}")
                current_length += image_len
    
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
    BLOCK_SIZE: int = 128,
) -> None:
    """Print block structure in the format shown in the terminal.
    
    Args:
        modality_positions: [B, M, 3] tensor of (modality_type, offset, length)
        q_len: Sequence length
        batch_idx: Which batch to print
        BLOCK_SIZE: Block size for mask computation

    print something like(not in block size chunks):

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
    Q\KV   0  1  2  3  4
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
    segments = []
    img_counter = 0
    
    # Add text segment before first image (if any)
    if len(mods) > 0:
        first_img_start = int(mods[0, 1])
        if first_img_start > 0:
            segments.append(("text", 0, first_img_start))
    
    # Process each image and text segments between them
    for i, (mod_type, mod_offset, mod_length) in enumerate(mods):
        mod_offset = int(mod_offset)
        mod_length = int(mod_length)
        mod_end = mod_offset + mod_length
        
        # Add image segment
        segments.append(("img", img_counter, mod_offset, mod_end))
        img_counter += 1
        
        # Add text segment after this image (if not last)
        if i < len(mods) - 1:
            next_img_start = int(mods[i + 1, 1])
            if next_img_start > mod_end:
                segments.append(("text", mod_end, next_img_start))
    
    # Add final text segment (if any)
    if len(mods) > 0:
        last_img_end = int(mods[-1, 1] + mods[-1, 2])
        if last_img_end < q_len:
            segments.append(("text", last_img_end, q_len))
    else:
        # No images, all text
        if q_len > 0:
            segments.append(("text", 0, q_len))
    
    # Print modality ranges
    print("Modality ranges: ", end="")
    for seg in segments:
        if seg[0] == "text":
            start, end = seg[1], seg[2]
            length = end - start
            print(f"text[{start}:{end}] (len={length}) ", end="")
        else:
            start, end = seg[2], seg[3]
            length = end - start
            img_idx = seg[1]
            print(f"img{img_idx}[{start}:{end}] (len={length}) ", end="")
    print()
    
    # Print block types (each segment is a block)
    print("Block types:")
    for q_idx, seg in enumerate(segments):
        if seg[0] == "text":
            start, end = seg[1], seg[2]
            print(f"Q{q_idx}[{start}:{end}]=text")
        else:
            start, end = seg[2], seg[3]
            img_idx = seg[1]
            print(f"Q{q_idx}[{start}:{end}]=img{img_idx}")
    
    # Print mask based on segment types and positions
    num_q_blocks = len(segments)
    print(f"Mask (Q blocks: {num_q_blocks}, F=FULL, C=CAUSAL, M=MASKED):")
    print("Q\\KV  ", end="")
    for kv_block in range(num_q_blocks):
        print(f"{kv_block:2d} ", end="")
    print()
    print("-" * (6 + 3 * num_q_blocks))
    
    # Determine mask type for each segment pair based on rules:
    # - Text segments: CAUSAL attention (can attend to previous text, same segment)
    # - Image modalities: FULL attention within same modality, FULL to all previous tokens
    # - Different image modalities: MASKED (no attention between different images, unless KV is before Q)
    
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
            
            # Determine mask type based on segment positions
            # Rules:
            # - Text can attend to all previous tokens (text and image) with FULL
            # - Text within same segment: CAUSAL
            # - Text cannot attend to future tokens: MASKED
            # - Image can attend to all previous tokens: FULL
            # - Image same modality: FULL
            # - Image cannot attend to future different images: MASKED
            
            if q_is_text and kv_is_text:
                # Text to text
                if q_seg_idx == kv_seg_idx:
                    # Same text segment: CAUSAL
                    mask_type = BlockMaskType.CAUSAL
                elif kv_end <= q_start:
                    # Previous text segment: FULL
                    mask_type = BlockMaskType.FULL
                else:
                    # Future text segment: MASKED
                    mask_type = BlockMaskType.MASKED
            elif not q_is_text and kv_is_text:
                # Image to text: FULL if text is before image
                if kv_end <= q_start:
                    mask_type = BlockMaskType.FULL
                else:
                    mask_type = BlockMaskType.MASKED
            elif q_is_text and not kv_is_text:
                # Text to image
                if kv_end <= q_start:
                    # Previous image: FULL (text can attend to previous images)
                    mask_type = BlockMaskType.FULL
                else:
                    # Future image: MASKED
                    mask_type = BlockMaskType.MASKED
            else:
                # Image to image
                if q_img_idx == kv_img_idx:
                    # Same image: FULL
                    mask_type = BlockMaskType.FULL
                elif kv_end <= q_start:
                    # Previous different image: FULL
                    mask_type = BlockMaskType.FULL
                else:
                    # Future different image: MASKED
                    mask_type = BlockMaskType.MASKED
            
            if mask_type == BlockMaskType.FULL:
                print(" F ", end="")
            elif mask_type == BlockMaskType.CAUSAL:
                print(" C ", end="")
            else:
                print(" M ", end="")
        print()
