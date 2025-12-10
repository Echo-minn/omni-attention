# Precision Loss Analysis: Masked vs Unmasked Attention

## Summary

The masked kernel (`omni_attn_shared_kv.cu`) suffers from **both precision loss AND critical bugs**. The large max_diff (3.6) indicates a **logic bug**, not just numerical precision issues. This document identifies the root causes and provides solutions.

## ⚠️ CRITICAL BUG FIXED: Incorrect Causal Mask Column Indexing

**Status**: ✅ FIXED

The causal mask was using **wrong column indices**, causing incorrect masking. This was the primary cause of the large max_diff (3.6).

**Root Cause**: 
- Causal mask used `col_pair = col_in_tile / 2` where `col_in_tile = lane_id % 8`
- This gave wrong column mapping: `{0,0,1,1,2,2,3,3}` instead of `{0,2,4,6}`
- Partial mask correctly uses `col_pair = (lane_id / 8) * 2`

**Fix Applied**: Causal mask now uses the same column indexing as partial mask for consistency.

## Root Causes

### 1. **Max Reduction Includes -INFINITY Values** (Lines 393-405)

**Problem**: The max reduction doesn't explicitly filter out -INFINITY values before reduction. While `max(-INFINITY, valid_value) = valid_value` is correct, when all values in a row are masked, the max becomes -INFINITY, which then propagates through the softmax computation.

**Current Code**:
```cuda
float tmp_max_0 = max(S[0], S[1]);  // If both are -INFINITY, result is -INFINITY
lane_row_max_new[0][0] = max(lane_row_max_new[0][0], tmp_max_0);
```

**Issue**: When computing max across warps, if a row has all masked values, `lane_row_max_new[0][0]` becomes -INFINITY, which then causes issues in the global max computation (lines 416-432).

### 2. **Inconsistent Max Handling for All-Masked Rows** (Lines 416-432)

**Problem**: The logic for handling all-masked blocks is complex and can lead to precision issues:

```cuda
if (block_row_max_new_0 > MASKED_THRESHOLD) {
  if (block_row_max_old_0 > MASKED_THRESHOLD) {
    block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
  }
} else {
  // Current block is all masked, keep old max
  block_row_max_new_0 = (block_row_max_old_0 > MASKED_THRESHOLD) ? block_row_max_old_0 : block_row_max_new_0;
}
```

**Issue**: When a block is all-masked, the code tries to use the old max from previous blocks. However, this can cause precision issues because:
- The old max might be from a completely different set of values
- The exp computation (line 442) uses `block_row_max_new_0`, which might be stale
- The rescaling logic (lines 512-528) assumes consistent max values

### 3. **Exp Computation on Masked Values** (Lines 442-445)

**Problem**: The code computes `exp(S - max)` even when S is -INFINITY:

```cuda
S[0] = __expf(S[0] - block_row_max_new_0);  // If S[0] is -INFINITY, exp(-INFINITY - max) = 0
```

**Issue**: While mathematically correct (exp(-INFINITY) = 0), this can cause numerical issues when:
- The max is computed from a mix of valid and masked values
- The max is stale (from previous blocks when current block is all-masked)
- The subtraction `-INFINITY - max` can cause overflow/underflow in edge cases

### 4. **Rescaling Logic with Masked Values** (Lines 512-528)

**Problem**: The rescaling factor computation doesn't properly handle the case when old or new max is from masked values:

```cuda
if (block_row_max_old_0 > MASKED_THRESHOLD && block_row_max_new_0 > MASKED_THRESHOLD) {
  float max_diff_0 = block_row_max_old_0 - block_row_max_new_0;
  // ...
  rescale_o_factor_0 = __expf(max_diff_0);
}
```

**Issue**: If either max is from masked values (even if > MASKED_THRESHOLD due to previous valid blocks), the rescaling factor might be incorrect.

### 5. **Comparison with Unmasked Version**

**Unmasked version** (flash_attn_mma_share_kv.cu, lines 521-522):
```cuda
float tmp_max_0 = __half2float(__hmax(t_hptr_S_0_1[0], t_hptr_S_0_1[1])) * scale;
```

**Key Difference**: 
- Unmasked: Max computed from half values, then scaled
- Masked: Max computed from float values after scaling

**Why unmasked works better**:
- No -INFINITY values to handle
- Simpler max reduction logic
- No edge cases with all-masked rows

## Solutions

### Solution 1: Filter -INFINITY in Max Reduction

**Fix**: Explicitly exclude -INFINITY values from max reduction:

```cuda
// Row max reduction - properly exclude masked values (-INFINITY)
{
  const float MASKED_THRESHOLD = -1e8f;
  #pragma unroll
  for (int j = 0; j < kWarpTileSeqLenK; ++j) {
    float *S = reinterpret_cast<float *>(&(R_S[0][j][0]));
    float tmp_max_0 = -INFINITY;
    float tmp_max_1 = -INFINITY;
    
    // Only consider non-masked values
    if (S[0] > MASKED_THRESHOLD) tmp_max_0 = max(tmp_max_0, S[0]);
    if (S[1] > MASKED_THRESHOLD) tmp_max_0 = max(tmp_max_0, S[1]);
    if (S[2] > MASKED_THRESHOLD) tmp_max_1 = max(tmp_max_1, S[2]);
    if (S[3] > MASKED_THRESHOLD) tmp_max_1 = max(tmp_max_1, S[3]);
    
    lane_row_max_new[0][0] = max(lane_row_max_new[0][0], tmp_max_0);
    lane_row_max_new[0][1] = max(lane_row_max_new[0][1], tmp_max_1);
  }
  // Warp level reduce max, warp_size = 4
  lane_row_max_new[0][0] = warp_reduce_max<float, 4>(lane_row_max_new[0][0]);
  lane_row_max_new[0][1] = warp_reduce_max<float, 4>(lane_row_max_new[0][1]);
}
```

### Solution 2: Simplify Max Update Logic

**Fix**: Simplify the max update to match the unmasked version more closely:

```cuda
// Compute exp and row sum
{
  float block_row_max_new_0 = lane_row_max_new[0][0];
  float block_row_max_new_1 = lane_row_max_new[0][1];
  float block_row_max_old_0 = lane_block_row_max_old[0][0];
  float block_row_max_old_1 = lane_block_row_max_old[0][1];
  
  const float MASKED_THRESHOLD = -1e8f;
  
  // Simple max update: if new max is valid, use it; otherwise keep old
  // This matches the unmasked version: max(m_old, m_new)
  if (block_row_max_new_0 > MASKED_THRESHOLD) {
    block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
  } else {
    // All masked in this block, keep old max (or -INFINITY if first block)
    block_row_max_new_0 = (block_row_max_old_0 > MASKED_THRESHOLD) ? block_row_max_old_0 : block_row_max_new_0;
  }
  
  if (block_row_max_new_1 > MASKED_THRESHOLD) {
    block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);
  } else {
    block_row_max_new_1 = (block_row_max_old_1 > MASKED_THRESHOLD) ? block_row_max_old_1 : block_row_max_new_1;
  }
  
  // ... rest of exp computation
}
```

### Solution 3: Skip Exp Computation for Masked Values

**Fix**: Explicitly skip exp computation for masked values to avoid numerical issues:

```cuda
#pragma unroll
for (int j = 0; j < kWarpTileSeqLenK; ++j) {
  float *S = reinterpret_cast<float *>(&(R_S[0][j][0]));
  half *h_S = reinterpret_cast<half *>(&(R_S[0][j][0]));
  const float MASKED_THRESHOLD = -1e8f;
  
  // Only compute exp for non-masked values
  if (S[0] > MASKED_THRESHOLD) {
    S[0] = __expf(S[0] - block_row_max_new_0);
    lane_row_sum_new[0][0] += S[0];
    h_S[0] = __float2half_rn(S[0]);
  } else {
    S[0] = 0.0f;
    h_S[0] = __float2half_rn(0.0f);
  }
  
  // Similar for S[1], S[2], S[3]
  // ...
}
```

### Solution 4: Use Fused Multiply-Add for Exp (Match Unmasked)

**Fix**: Use FMA for exp computation to match the unmasked version's precision:

```cuda
// Instead of: S[0] = __expf(S[0] - block_row_max_new_0);
// Where S[0] is already scaled, use:
S[0] = __expf(__fmaf_rn(S[0], 1.0f, -block_row_max_new_0));
```

**Note**: This is less critical since S is already scaled, but it ensures consistency.

### Solution 5: Fix Rescaling Logic

**Fix**: Ensure rescaling only happens when both old and new maxes are valid:

```cuda
// rescale factor for O and l, exp(m_old - m_new)
float rescale_o_factor_0 = 1.0f;
float rescale_o_factor_1 = 1.0f;

const float MASKED_THRESHOLD = -1e9f;  // Use same threshold

if (block_row_max_old_0 > MASKED_THRESHOLD && block_row_max_new_0 > MASKED_THRESHOLD) {
  float max_diff_0 = block_row_max_old_0 - block_row_max_new_0;
  const float MIN_EXP_DIFF = 1e-5f;
  if (fabsf(max_diff_0) > MIN_EXP_DIFF) {  // Only rescale if significant difference
    max_diff_0 = fmaxf(-88.0f, fminf(88.0f, max_diff_0));  // Clamp to avoid overflow
    rescale_o_factor_0 = __expf(max_diff_0);
  }
}
// Similar for rescale_o_factor_1
```

## Recommended Implementation Order

1. **Start with Solution 1** (Filter -INFINITY in max reduction) - This is the most critical fix
2. **Then Solution 3** (Skip exp for masked values) - Prevents numerical issues
3. **Then Solution 2** (Simplify max update) - Makes logic clearer
4. **Finally Solution 5** (Fix rescaling) - Ensures correct accumulation

## Testing

After implementing fixes, compare results with:
- Unmasked FA2 implementation
- Reference implementation (e.g., PyTorch's standard attention)
- Check for precision errors: `max_error < 1e-3`, `mean_error < 1e-5`

## Additional Notes

- The unmasked version doesn't have these issues because it never deals with -INFINITY
- The precision loss is likely most noticeable when:
  - Many rows have masked values
  - Mixed masked/unmasked blocks
  - Long sequences with many blocks
- Consider using higher precision (FP32) for intermediate computations if needed

