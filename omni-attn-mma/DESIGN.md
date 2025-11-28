# Omni-Attention Design Document

## Overview

Omni-Attention combines the best of both worlds:
- **FlashAttention-2's split-Q strategy**: Efficient tiling with online softmax to avoid materializing the full attention matrix
- **FlexAttention's block_mask**: Block-sparse attention with per-block mask types

## Key Design Decisions

### 1. Block Mask Types

Each block in the attention matrix can have one of three types:

- **MASKED (0)**: Block is fully masked out - skip computation entirely
- **CAUSAL (1)**: Block uses causal masking (q_idx >= kv_idx)
- **FULL (2)**: Block has no masking - full attention

### 2. Split-Q Strategy (from FlashAttention-2)

- Split Q into blocks of size `BLOCK_M` (typically 128)
- For each Q block, iterate over KV blocks specified by `BlockMask`
- Use online softmax with tiling to avoid materializing full S = Q@K^T matrix
- Maintain running statistics (m_i, lse_i) for numerical stability

### 3. BlockMask Format (from FlexAttention)

- `kv_num_blocks`: [batch, nheads, num_q_blocks] - number of active KV blocks per Q block
- `kv_indices`: [batch, nheads, num_q_blocks, max_blocks] - which KV block columns to process
- `block_mask_types`: [batch, nheads, num_q_blocks, max_blocks] - mask type per block

## Implementation Details

### Kernel Structure

```
For each Q block (start_m):
  1. Load Q block into SRAM (stays there)
  2. Read kv_num_blocks[q_block_idx] to know how many KV blocks to process
  3. For each active KV block:
     a. Read kv_indices to get which KV block column
     b. Read block_mask_types to get mask type
     c. If MASKED: skip
     d. If CAUSAL: apply causal mask to qk scores
     e. If FULL: no masking
     f. Update online softmax statistics
     g. Accumulate output
  4. Write back output and LSE
```

### Memory Efficiency

- **Q blocks**: Loaded once per Q block, stay in SRAM
- **K/V blocks**: Loaded on-demand based on BlockMask
- **S matrix**: Never materialized (computed on-the-fly with tiling)
- **Output**: Accumulated incrementally, scaled at the end

### Performance Optimizations

1. **Skip masked blocks**: Don't load K/V for fully masked blocks
2. **Separate full blocks**: Can optimize FULL blocks (no mask computation needed)
3. **Online softmax**: Avoids storing full attention matrix
4. **Tiling**: Reduces memory bandwidth

## Usage Examples

### Causal Attention
```python
kv_num_blocks, kv_indices, block_mask_types = create_causal_block_mask(
    batch=1, nheads=8, seqlen_q=1024, seqlen_k=1024
)
output, lse = omni_attention_forward(q, k, v, kv_num_blocks, kv_indices, block_mask_types)
```

### Full Attention
```python
kv_num_blocks, kv_indices, block_mask_types = create_full_block_mask(
    batch=1, nheads=8, seqlen_q=1024, seqlen_k=1024
)
output, lse = omni_attention_forward(q, k, v, kv_num_blocks, kv_indices, block_mask_types)
```

### Hybrid (Prefix-LM)
```python
# Full attention for prefix, causal for rest
kv_num_blocks, kv_indices, block_mask_types = create_hybrid_block_mask(
    batch=1, nheads=8, seqlen_q=1024, seqlen_k=1024, prefix_len=256
)
output, lse = omni_attention_forward(q, k, v, kv_num_blocks, kv_indices, block_mask_types)
```

### Custom Pattern
```python
def custom_pattern(q_idx, kv_idx):
    if q_idx < 2 and kv_idx < 2:
        return BlockMaskType.FULL  # First 2 blocks: full attention
    elif kv_idx <= q_idx:
        return BlockMaskType.CAUSAL  # Rest: causal
    else:
        return BlockMaskType.MASKED

kv_num_blocks, kv_indices, block_mask_types = create_omni_block_mask(
    batch=1, nheads=8, seqlen_q=1024, seqlen_k=1024, 
    block_mask_pattern=custom_pattern
)
```

## Comparison with Alternatives

### vs FlashAttention-2
- ✅ Supports block-sparse patterns (not just causal)
- ✅ Can skip blocks entirely (better for very sparse patterns)
- ✅ Per-block mask types (more flexible)
- ⚠️ Slightly more complex kernel (but still efficient)

### vs FlexAttention
- ✅ Uses proven FlashAttention-2 tiling strategy
- ✅ Simpler block mask format (just indices + types)
- ✅ More explicit about mask types (FULL/CAUSAL/MASKED)
- ⚠️ Less general than FlexAttention's score_mod (but more efficient)

## Future Enhancements

1. **Backward pass**: Implement gradient computation
2. **Variable block sizes**: Support different BLOCK_M/BLOCK_N per region
3. **Attention bias**: Add support for ALiBi or other biases
4. **Multi-query attention**: Support MQA/GQA
5. **Autotuning**: Add Triton autotune for optimal block sizes

