# Omni Attention Kernel Implementation Guide

## 1. How flex_attn Uses block_mask

### BlockMask Structure
`block_mask.as_tuple()` returns:
```python
(
    q_length, kv_length,           # seq_lengths
    kv_num_blocks,                 # [B, H, num_q_blocks] - int32
    kv_indices,                    # [B, H, num_q_blocks, max_blocks] - int32
    full_kv_num_blocks,            # Optional: for optimization
    full_kv_indices,               # Optional: for optimization
    q_num_blocks,                  # For backward pass
    q_indices,                     # For backward pass
    full_q_num_blocks,             # For backward pass
    full_q_indices,                # For backward pass
    Q_BLOCK_SIZE, KV_BLOCK_SIZE,   # Block sizes
    mask_mod                       # Function for per-element masking
)
```

### Usage Pattern
1. **Iterate over Q blocks**: Each thread block processes one Q tile (Br rows)
2. **For each Q block, iterate over active KV blocks**: Use `kv_num_blocks[b, h, q_block]` to get count
3. **Get KV block indices**: Use `kv_indices[b, h, q_block, kv_idx]` to get which KV block to process
4. **Apply masking**: Use `block_mask_types` to determine mask type (MASKED/CAUSAL/FULL)

## 2. Arguments to Pass to omni_attn Kernel

Your current signature is **correct**:
```cpp
void omni_attn_mma_blockmask(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor kv_num_blocks,      // [B, H, num_q_blocks] - int32
    torch::Tensor kv_indices,         // [B, H, num_q_blocks, max_blocks] - int32
    torch::Tensor block_mask_types,   // [B, H, num_q_blocks, max_blocks] - int32
    int stages)
```

### Block Mask Type Values
- `0` = MASKED: Skip this block entirely
- `1` = CAUSAL: Apply causal mask within block (q_idx >= kv_idx)
- `2` = FULL: No masking, dense attention

## 3. Required Dimension Assertions

Add these runtime checks at the start of your kernel:

```cpp
// Dimension checks
TORCH_CHECK(Q.dim() == 4, "Q must be 4D [B, H, seqlen, head_dim]");
TORCH_CHECK(K.dim() == 4, "K must be 4D [B, H, seqlen, head_dim]");
TORCH_CHECK(V.dim() == 4, "V must be 4D [B, H, seqlen, head_dim]");
TORCH_CHECK(O.dim() == 4, "O must be 4D [B, H, seqlen, head_dim]");

TORCH_CHECK(kv_num_blocks.dim() == 3, "kv_num_blocks must be 3D [B, H, num_q_blocks]");
TORCH_CHECK(kv_indices.dim() == 4, "kv_indices must be 4D [B, H, num_q_blocks, max_blocks]");
TORCH_CHECK(block_mask_types.dim() == 4, "block_mask_types must be 4D [B, H, num_q_blocks, max_blocks]");

// Shape consistency
TORCH_CHECK(Q.size(0) == K.size(0) && Q.size(0) == V.size(0), "Batch size mismatch");
TORCH_CHECK(Q.size(1) == K.size(1) && Q.size(1) == V.size(1), "Head count mismatch");
TORCH_CHECK(Q.size(2) == K.size(2) && Q.size(2) == V.size(2), "Sequence length mismatch");
TORCH_CHECK(Q.size(3) == K.size(3) && K.size(3) == V.size(3), "Head dim mismatch");

// Block mask shape consistency
TORCH_CHECK(kv_num_blocks.size(0) == Q.size(0), "kv_num_blocks batch mismatch");
TORCH_CHECK(kv_num_blocks.size(1) == Q.size(1), "kv_num_blocks head mismatch");
TORCH_CHECK(kv_indices.size(0) == kv_num_blocks.size(0), "kv_indices batch mismatch");
TORCH_CHECK(kv_indices.size(1) == kv_num_blocks.size(1), "kv_indices head mismatch");
TORCH_CHECK(kv_indices.size(2) == kv_num_blocks.size(2), "kv_indices num_q_blocks mismatch");
TORCH_CHECK(block_mask_types.sizes() == kv_indices.sizes(), "block_mask_types shape must match kv_indices");

// Data type checks
TORCH_CHECK(Q.dtype() == torch::kFloat16, "Q must be float16");
TORCH_CHECK(K.dtype() == torch::kFloat16, "K must be float16");
TORCH_CHECK(V.dtype() == torch::kFloat16, "V must be float16");
TORCH_CHECK(kv_num_blocks.dtype() == torch::kInt32, "kv_num_blocks must be int32");
TORCH_CHECK(kv_indices.dtype() == torch::kInt32, "kv_indices must be int32");
TORCH_CHECK(block_mask_types.dtype() == torch::kInt32, "block_mask_types must be int32");

// Block size alignment
const int BLOCK_SIZE = 128; // or from block_mask if available
TORCH_CHECK(Q.size(2) % BLOCK_SIZE == 0 || Q.size(2) <= BLOCK_SIZE, 
            "Q seqlen must be aligned to BLOCK_SIZE or <= BLOCK_SIZE");
```

## 4. Implementation Structure

### Main Loop Structure
```cpp
// Get Q block index
const int q_block = Q_tile_id;  // Current Q block being processed
const int q_block_start = q_block * Br;
const int q_block_end = min(q_block_start + Br, QKV_seqlen);

// Get number of active KV blocks for this Q block
const int* kv_num_blocks_ptr = kv_num_blocks.data_ptr<int>();
const int* kv_indices_ptr = kv_indices.data_ptr<int>();
const int* block_mask_types_ptr = block_mask_types.data_ptr<int>();

// Compute strides (assuming row-major)
const int kv_num_blocks_stride0 = kv_num_blocks.stride(0);
const int kv_num_blocks_stride1 = kv_num_blocks.stride(1);
const int kv_num_blocks_stride2 = kv_num_blocks.stride(2);

const int kv_indices_stride0 = kv_indices.stride(0);
const int kv_indices_stride1 = kv_indices.stride(1);
const int kv_indices_stride2 = kv_indices.stride(2);
const int kv_indices_stride3 = kv_indices.stride(3);

const int block_mask_types_stride0 = block_mask_types.stride(0);
const int block_mask_types_stride1 = block_mask_types.stride(1);
const int block_mask_types_stride2 = block_mask_types.stride(2);
const int block_mask_types_stride3 = block_mask_types.stride(3);

// Get active KV block count for this (batch, head, q_block)
int kv_nb = kv_num_blocks_ptr[
    QKV_batch_id * kv_num_blocks_stride0 +
    QKV_head_id * kv_num_blocks_stride1 +
    q_block * kv_num_blocks_stride2
];

// Main loop over active KV blocks
for (int kv_idx = 0; kv_idx < kv_nb; ++kv_idx) {
    // Get KV block index
    int kv_block = kv_indices_ptr[
        QKV_batch_id * kv_indices_stride0 +
        QKV_head_id * kv_indices_stride1 +
        q_block * kv_indices_stride2 +
        kv_idx * kv_indices_stride3
    ];
    
    // Get mask type
    int mask_type = block_mask_types_ptr[
        QKV_batch_id * block_mask_types_stride0 +
        QKV_head_id * block_mask_types_stride1 +
        q_block * block_mask_types_stride2 +
        kv_idx * block_mask_types_stride3
    ];
    
    // Skip if masked
    if (mask_type == 0) {  // MASKED
        continue;
    }
    
    // Compute KV block boundaries
    int kv_block_start = kv_block * Bc;
    int kv_block_end = min(kv_block_start + Bc, QKV_seqlen);
    
    // Apply causal masking if needed
    int kv_start = kv_block_start;
    int kv_end = kv_block_end;
    if (mask_type == 1) {  // CAUSAL
        // For each Q position in this tile, only attend to KV positions <= q_idx
        // This needs to be applied per-Q-position during S computation
    }
    
    // Load K tile [Bc, head_dim] from gmem -> smem
    // Compute S = Q @ K^T with mask application
    // Apply softmax to get P
    // Load V tile [Bc, head_dim] from gmem -> smem
    // Compute O += P @ V
    // Update online softmax statistics
}
```

## 5. Critical Correctness Points

### A. Online Softmax with Block Masking
- **Row max tracking**: Track max per Q row across ALL active KV blocks (not per block)
- **Row sum tracking**: Track sum per Q row across ALL active KV blocks
- **Online correction**: When processing a new KV block, correct old statistics:
  ```cpp
  float m_new = max(m_old, m_new_block);
  float alpha = exp(m_old - m_new);
  sum_new = alpha * sum_old + sum_new_block;
  ```

### B. Causal Mask Application
- **Within-block causal**: For CAUSAL blocks, mask positions where `q_idx < kv_idx`
- **Block-level causal**: If `q_block_start < kv_block_start`, entire block is masked
- **Per-MMA-tile masking**: Apply causal mask at the MMA tile level (m16n8k16)

### C. Memory Synchronization
- **Q loading**: Load Q once at start, wait for completion before use
- **K/V loading**: Load K/V per KV block, sync before MMA operations
- **Shared memory reuse**: K and V can share smem (since K is consumed before V is loaded)

### D. Grid/Block Configuration
```cpp
// Grid: (num_q_blocks, batch * num_heads)
// Block: (kNumThreads, 1, 1)
dim3 grid(div_ceil(QKV_seqlen, Br), QKV_batch * QKV_head);
dim3 block(kNumThreads, 1, 1);
```

### E. Boundary Conditions
- **Q tile overflow**: Check `load_gmem_Q_Br >= QKV_seqlen` (already done)
- **KV block overflow**: Check `kv_block_start < QKV_seqlen`
- **Empty blocks**: Handle case where `kv_nb == 0` (output zeros)

## 6. Testing Checklist

1. **Dimension assertions**: All checks pass
2. **Block mask iteration**: Correctly iterates over active blocks only
3. **Mask type handling**: MASKED blocks are skipped, CAUSAL/FULL handled correctly
4. **Online softmax**: Statistics correctly accumulated across blocks
5. **Causal masking**: Correctly masks future positions
6. **Boundary conditions**: Handles sequence lengths not divisible by block size
7. **Memory correctness**: No out-of-bounds accesses
8. **Numerical correctness**: Matches reference implementation (e.g., omni_attn_simple_kernel)

## 7. Performance Considerations

1. **Prefetching**: Use `cp.async` for K/V loading while computing
2. **Register pressure**: Monitor register usage, may need to reduce prefetch depth
3. **Shared memory**: Ensure K and V smem reuse is correct
4. **Warp efficiency**: Ensure all warps are active during computation
5. **Bank conflicts**: Check smem access patterns for bank conflicts

