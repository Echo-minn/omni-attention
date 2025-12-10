# Plan: Switch to FP32 Accumulation for Q@K^T

## Overview
Change from `HMMA16816` (FP16 accumulation) to `HMMA16816F32` (FP32 accumulation) for Q@K^T computation to improve precision.

## Key Changes Required

### 1. Register Storage Changes

**Current (FP16):**
```cuda
uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][2]; // [1][8][2] - FP16
```

**New (FP32):**
```cuda
uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][4]; // [1][8][4] - FP32
```

**Impact:**
- **2x more registers** per R_S element (from 2 to 4 uint32_t)
- Total: `1 * 8 * 4 = 32` uint32_t registers per thread (was 16)
- Need to verify register pressure doesn't exceed limits

### 2. Initialize R_S to Zero

**Current:**
```cuda
// R_S is implicitly zero-initialized or accumulates
```

**New:**
```cuda
// Need explicit initialization before MMA loop
fill_3D_regs<uint32_t, kWarpTileSeqLenQ, kWarpTileSeqLenK, 4>(R_S, 0);
```

**Location:** Before the loop over `tile_K_d` (around line 203)

### 3. MMA Instruction Change

**Current (line 228):**
```cuda
HMMA16816(R_S[0][j][0], R_S[0][j][1], 
          R_Q[0][0][0], R_Q[0][0][1], R_Q[0][0][2], R_Q[0][0][3], 
          R_K[j][0], R_K[j][1], 
          R_S[0][j][0], R_S[0][j][1]);
```

**New:**
```cuda
HMMA16816F32(R_S[0][j][0], R_S[0][j][1], R_S[0][j][2], R_S[0][j][3],
             R_Q[0][0][0], R_Q[0][0][1], R_Q[0][0][2], R_Q[0][0][3],
             R_K[j][0], R_K[j][1],
             R_S[0][j][0], R_S[0][j][1], R_S[0][j][2], R_S[0][j][3]);
```

**Key Differences:**
- Takes 4 output registers (RD0-RD3) instead of 2
- Takes 4 input accumulator registers (RC0-RC3) instead of 2
- Q and K inputs remain the same (FP16)

### 4. Scaling (Line 233-241)

**Current:**
```cuda
half *S = reinterpret_cast<half *>(&R_S[0][j][0]);
S[0] = __hmul(S[0], __float2half(scale));
S[1] = __hmul(S[1], __float2half(scale));
S[2] = __hmul(S[2], __float2half(scale));
S[3] = __hmul(S[3], __float2half(scale));
```

**New:**
```cuda
float *S = reinterpret_cast<float *>(&R_S[0][j][0]);
S[0] = S[0] * scale;  // Direct FP32 multiplication
S[1] = S[1] * scale;
S[2] = S[2] * scale;
S[3] = S[3] * scale;
```

**Benefit:** No precision loss from half conversion!

### 5. Masking (Lines 243-339)

**Current:**
```cuda
half *S = reinterpret_cast<half *>(&R_S[0][j][0]);
if (row0 < kv_col_0) S[0] = __float2half(-INFINITY);
```

**New:**
```cuda
float *S = reinterpret_cast<float *>(&R_S[0][j][0]);
if (row0 < kv_col_0) S[0] = -INFINITY;  // Direct FP32 assignment
```

**Changes needed in:**
- Causal mask section (lines 243-266)
- Partial mask section (lines 268-339)

### 6. Max Computation (Lines 366-406) - **MAJOR IMPROVEMENT**

**Current:**
```cuda
half *t_hptr_S_0_1 = reinterpret_cast<half *>(&(R_S[0][j][0]));
float s0 = __half2float(t_hptr_S_0_1[0]);  // Already rounded!
```

**New:**
```cuda
float *t_fptr_S_0_1 = reinterpret_cast<float *>(&(R_S[0][j][0]));
// Direct FP32 access - no rounding errors!
float s0 = t_fptr_S_0_1[0];
float s1 = t_fptr_S_0_1[1];
float s2 = t_fptr_S_0_1[2];
float s3 = t_fptr_S_0_1[3];
```

**Benefit:** Max computed from full FP32 precision values!

### 7. Exp Computation (Lines 408-500)

**Current:**
```cuda
half *t_hptr_S_0_1 = reinterpret_cast<half *>(&(R_S[0][j][0]));
float s_val_0 = __half2float(t_hptr_S_0_1[0]);  // Convert from half
t_reg_S_0_1.x = __expf(s_val_0 - block_row_max_new_0);
t_hptr_S_0_1[0] = __float2half_rn(t_reg_S_0_1.x);  // Convert back to half
```

**New:**
```cuda
float *t_fptr_S_0_1 = reinterpret_cast<float *>(&(R_S[0][j][0]));
// Direct FP32 computation
t_fptr_S_0_1[0] = __expf(__fmaf_rn(t_fptr_S_0_1[0], scale, -block_row_max_new_0));
t_fptr_S_0_1[1] = __expf(__fmaf_rn(t_fptr_S_0_1[1], scale, -block_row_max_new_0));
t_fptr_S_0_1[2] = __expf(__fmaf_rn(t_fptr_S_0_1[2], scale, -block_row_max_new_1));
t_fptr_S_0_1[3] = __expf(__fmaf_rn(t_fptr_S_0_1[3], scale, -block_row_max_new_1));

// Convert to FP16 for P@V (P@V still uses FP16)
half *t_hptr_S_0_1 = reinterpret_cast<half *>(&(R_S[0][j][0]));
t_hptr_S_0_1[0] = __float2half_rn(t_fptr_S_0_1[0]);
t_hptr_S_0_1[1] = __float2half_rn(t_fptr_S_0_1[1]);
t_hptr_S_0_1[2] = __float2half_rn(t_fptr_S_0_1[2]);
t_hptr_S_0_1[3] = __float2half_rn(t_fptr_S_0_1[3]);
```

**Note:** Scale is applied during exp computation (not before), matching flash-attn pattern.

### 8. P@V Computation (Line 530)

**No changes needed!** P@V still uses FP16, and we convert exp values to FP16 before this step.

## Implementation Order

1. **Step 1: Change R_S declaration** (line 123)
   - Change `[2]` to `[4]`
   - Add initialization

2. **Step 2: Update MMA instruction** (line 228)
   - Change to `HMMA16816F32`
   - Update all 4 register arguments

3. **Step 3: Update scaling** (line 233-241)
   - Change from half to float pointer
   - Direct FP32 multiplication

4. **Step 4: Update masking** (lines 243-339)
   - Change from half to float pointer
   - Direct FP32 assignment

5. **Step 5: Update max computation** (lines 366-406)
   - Change from half to float pointer
   - Direct FP32 access

6. **Step 6: Update exp computation** (lines 408-500)
   - Change from half to float pointer
   - Apply scale during exp (fused)
   - Convert to FP16 at the end

## Critical Correctness Checks

### 1. Register Layout Verification
- FP32 MMA produces 4 FP32 values per thread
- Layout: `(x,y) -> {c0, c1}`, `(z,w) -> {c2, c3}` (rows 0-7 and 8-15)
- Must match the row indexing used in masking

### 2. Scale Application
- **Current**: Scale applied before masking
- **Flash-Attn F32F16F16F32**: Scale applied during exp computation
- **Decision**: Follow flash-attn pattern (apply during exp) for consistency

### 3. Masking with FP32
- `-INFINITY` in FP32 is the same value
- Masking logic remains the same, just using FP32 values

### 4. Max Computation
- Now computed from FP32 values (much better precision!)
- No conversion errors

### 5. Exp to FP16 Conversion
- Must convert to FP16 before P@V
- Conversion happens after exp computation
- R_S is reused for P (exp values)

## Expected Benefits

1. **Better precision in Q@K^T**: FP32 accumulation reduces rounding errors
2. **Better precision in max computation**: Computed from FP32 values
3. **Better precision in exp computation**: All in FP32 until final conversion
4. **Reduced error accumulation**: Especially important for masked blocks

## Potential Issues

1. **Register pressure**: 2x more registers for R_S
   - Current: 16 uint32_t per thread
   - New: 32 uint32_t per thread
   - Need to verify this doesn't cause register spilling

2. **Scale application timing**: 
   - Current code applies scale before masking
   - Flash-attn applies scale during exp
   - Need to verify this doesn't break correctness

3. **Masking row/column indexing**: 
   - Must ensure FP32 layout matches expected row/column mapping
   - Verify causal and partial mask logic still works correctly

## Testing Strategy

1. **Compare with naive kernel**: Should see much better agreement
2. **Compare row_max/row_sum**: Should match naive kernel more closely
3. **Test with different mask types**: F, C, P combinations
4. **Verify no regressions**: All existing tests should still pass

