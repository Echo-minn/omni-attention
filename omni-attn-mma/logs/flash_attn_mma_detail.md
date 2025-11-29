# FlashAttention-MMA shared KV
```python
# ========== 阶段1: 加载Q (一次性加载，后续复用) ==========
async_copy_Q_g2s():
    for each thread:
        gmem_addr = Q_gmem_offset + (Q_tile_id * Br + row) * head_dim + col
        smem_addr = smem_Q_base + row * (head_dim + padQ) + col
        CP_ASYNC_CG(smem_addr, gmem_addr, 16 bytes)  # 128位对齐加载
    CP_ASYNC_COMMIT_GROUP()
    CP_ASYNC_WAIT_GROUP(0)
    __syncthreads()

# ========== 阶段2: 循环处理K序列 (Tc = ceil(seqlen / Bc)) ==========
for tile_K_seqlen in range(0, Tc):  # 例如: seqlen=256, Bc=64, Tc=4
    
    # 2.1 加载K tile (Stage 1: 同步加载)
    async_copy_K_g2s():
        K_Bc_offset = tile_K_seqlen * Bc  # 0, 64, 128, 192...
        gmem_addr = K_gmem_offset + (K_Bc_offset + row) * head_dim + col
        smem_addr = smem_K_base + row * (head_dim + padK) + col
        CP_ASYNC_CG(smem_addr, gmem_addr, 16 bytes)
    CP_ASYNC_COMMIT_GROUP()
    CP_ASYNC_WAIT_GROUP(0)  # 等待K加载完成
    __syncthreads()
    
    # 2.2 计算 S = Q @ K^T (分块矩阵乘法)
    R_S = zeros([Br, Bc])  # 寄存器中的S矩阵
    
    for tile_K_d in range(0, head_dim / 16):  # 按k=16分块
        # 从smem加载Q到寄存器 (m16k16)
        load_Q_s2r(R_Q, smem_Q, tile_K_d, warp_QP)
        
        # 从smem加载K到寄存器 (k16n8)  
        load_K_s2r(R_K, smem_K, tile_K_d, warp_KV)
        
        # MMA计算: R_S += R_Q @ R_K^T (m16n8k16)
        for j in range(kWarpTileSeqLenK):  # 8次MMA
            HMMA16816(R_S[0][j], R_Q, R_K[j], R_S[0][j])
    
    __syncthreads()
    
    # 2.3 加载V tile (Stage 1: 在计算完Q@K^T后加载)
    async_copy_V_g2s():
        V_Bc_offset = tile_K_seqlen * Bc
        gmem_addr = V_gmem_offset + (V_Bc_offset + row) * head_dim + col
        smem_addr = smem_V_base + row * (head_dim + padV) + col  # 复用K的smem
        CP_ASYNC_CG(smem_addr, gmem_addr, 16 bytes)
    CP_ASYNC_COMMIT_GROUP()
    CP_ASYNC_WAIT_GROUP(0)
    __syncthreads()
    
    # 2.4 Online Softmax: 计算 P = softmax(S)
    # 2.4.1 计算行最大值
    row_max_new = reduce_max(S * scale)  # Thread -> Warp -> Block reduce
    row_max_global = max(row_max_old, row_max_new)  # 全局最大值
    
    # 2.4.2 计算exp和行和
    P = exp(S * scale - row_max_global)  # 更新R_S为P
    row_sum_new = reduce_sum(P)  # Thread -> Warp -> Block reduce
    
    # 2.5 计算 O_partial = P @ V
    R_O = zeros([Br, head_dim])  # 当前tile的输出
    
    for tile_V_Bc in range(0, Bc / 16):  # 按Bc=16分块
        # 从smem加载V到寄存器 (k16n8, 转置)
        load_V_s2r_trans(R_V, smem_V, tile_V_Bc, warp_KV)
        
        # MMA计算: R_O += P[:, Bc_slice] @ V[Bc_slice, :]
        w = tile_V_Bc * 2  # 选择P的列块
        for j in range(kWarpTileHeadDimV):  # 8次MMA
            HMMA16816(R_O[0][j], R_S[0][w], R_S[0][w+1], R_V[j], R_O[0][j])
    
    __syncthreads()
    
    # 2.6 Online Rescaling (FlashAttention-2核心)
    # 更新全局统计量
    rescale_factor = exp(row_max_old - row_max_global)
    
    # Rescale旧输出并累加新输出
    O_global = rescale_factor * O_global + R_O
    
    # 更新统计量
    row_sum_global = rescale_factor * row_sum_old + row_sum_new
    row_max_old = row_max_global
    row_sum_old = row_sum_global

# ========== 阶段3: 最终归一化并写回 ==========
# 最终rescale: O_final = O_global / row_sum_global
rescale_factor_final = 1.0 / row_sum_global
O_final = O_global * rescale_factor_final

# 写回全局内存 (使用warp shuffle进行collective store)
collective_store_O_gmem(O_final, O_gmem_offset, Q_tile_id)
```

# OmniAttention-MMA prefetch
```python
# ========== 阶段1: 加载Q (一次性加载，后续复用) ==========
async_copy_Q_g2s():
    # 与Stage 1相同
    CP_ASYNC_CG(smem_Q, gmem_Q, 16 bytes)
    CP_ASYNC_COMMIT_GROUP()
    CP_ASYNC_WAIT_GROUP(0)
    __syncthreads()

# ========== 阶段2: 循环处理K序列 (带预取优化) ==========
for tile_K_seqlen in range(0, Tc):
    
    # ========== 2.1 预取策略 (Stage 2核心优化) ==========
    if tile_K_seqlen == 0:
        # 第一次迭代：同步加载第一个K tile
        async_copy_K_g2s(buffer_id=0):
            K_Bc_offset = 0
            CP_ASYNC_CG(smem_K[0], gmem_K, 16 bytes)
        CP_ASYNC_COMMIT_GROUP()
        CP_ASYNC_WAIT_GROUP(0)  # 等待K加载完成
        __syncthreads()
    
    # 预取当前tile的V (在计算Q@K^T之前)
    async_copy_V_g2s(buffer_id=1):  # 使用buffer 1
        V_Bc_offset = tile_K_seqlen * Bc
        CP_ASYNC_CG(smem_V[1], gmem_V, 16 bytes)
    CP_ASYNC_COMMIT_GROUP()  # 异步加载，不等待
    
    # ========== 2.2 计算 S = Q @ K^T ==========
    # 使用buffer 0中的K (已在smem中)
    R_S = zeros([Br, Bc])
    
    for tile_K_d in range(0, head_dim / 16):
        load_Q_s2r(R_Q, smem_Q, tile_K_d, warp_QP)
        load_K_s2r(R_K, smem_K[0], tile_K_d, warp_KV)  # 从buffer 0读取
        for j in range(kWarpTileSeqLenK):
            HMMA16816(R_S[0][j], R_Q, R_K[j], R_S[0][j])
    
    __syncthreads()
    
    # ========== 2.3 预取下一个K tile (在计算P@V之前) ==========
    if (tile_K_seqlen + 1) < Tc:
        async_copy_K_g2s(buffer_id=0):  # 预取到buffer 0
            K_Bc_offset = (tile_K_seqlen + 1) * Bc
            CP_ASYNC_CG(smem_K[0], gmem_K, 16 bytes)
        CP_ASYNC_COMMIT_GROUP()  # 异步加载，不等待
    
    # ========== 2.4 Online Softmax: 计算 P = softmax(S) ==========
    row_max_new = reduce_max(S * scale)
    row_max_global = max(row_max_old, row_max_new)
    P = exp(S * scale - row_max_global)
    row_sum_new = reduce_sum(P)
    
    # ========== 2.5 等待V就绪，然后计算 O_partial = P @ V ==========
    if (tile_K_seqlen + 1) < Tc:
        CP_ASYNC_WAIT_GROUP(1)  # 等待V加载完成（group 1）
    else:
        CP_ASYNC_WAIT_GROUP(0)  # 最后一个tile，等待V
    __syncthreads()
    
    # 使用buffer 1中的V (已在smem中)
    R_O = zeros([Br, head_dim])
    
    for tile_V_Bc in range(0, Bc / 16):
        load_V_s2r_trans(R_V, smem_V[1], tile_V_Bc, warp_KV)  # 从buffer 1读取
        w = tile_V_Bc * 2
        for j in range(kWarpTileHeadDimV):
            HMMA16816(R_O[0][j], R_S[0][w], R_S[0][w+1], R_V[j], R_O[0][j])
    
    __syncthreads()
    
    # ========== 2.6 Online Rescaling ==========
    rescale_factor = exp(row_max_old - row_max_global)
    O_global = rescale_factor * O_global + R_O
    row_sum_global = rescale_factor * row_sum_old + row_sum_new
    row_max_old = row_max_global
    row_sum_old = row_sum_global
    
    # ========== 2.7 等待下一个K tile就绪 (为下次迭代准备) ==========
    if (tile_K_seqlen + 1) < Tc:
        CP_ASYNC_WAIT_GROUP(0)  # 等待下一个K加载完成
        __syncthreads()
        # 下次迭代将使用buffer 0中的K

# ========== 阶段3: 最终归一化并写回 ==========
rescale_factor_final = 1.0 / row_sum_global
O_final = O_global * rescale_factor_final
collective_store_O_gmem(O_final, O_gmem_offset, Q_tile_id)

```

# OmniAttention-MMA blockmask

```python
// causal mask
// [1,0,0,0,0,0,0,0,
// 1,1,0,0,0,0,0,0,
// 1,1,1,0,0,0,0,0,
// 1,1,1,1,0,0,0,0,
// 1,1,1,1,1,0,0,0,
// 1,1,1,1,1,1,0,0,
// 1,1,1,1,1,1,1,0,
// 1,1,1,1,1,1,1,1]

// full mask
// [1,1,1,1,1,1,1,1,
// 1,1,1,1,1,1,1,1,
// 1,1,1,1,1,1,1,1,
// 1,1,1,1,1,1,1,1,
// 1,1,1,1,1,1,1,1,
// 1,1,1,1,1,1,1,1,
// 1,1,1,1,1,1,1,1,
// 1,1,1,1,1,1,1,1]

// partial mask(some is causal, some is full)
// [1,0,0,0,0,0,0,0,
// 1,1,0,0,0,0,0,0,
// 1,1,1,1,1,0,0,0,
// 1,1,1,1,1,0,0,0,
// 1,1,1,1,1,0,0,0,
// 1,1,1,1,1,1,1,1,
// 1,1,1,1,1,1,1,1,
// 1,1,1,1,1,1,1,1]
// or
// [1,1,1,1,1,0,0,0,
// 1,1,1,1,1,0,0,0,
// 1,1,1,1,1,0,0,0,
// 1,1,1,1,1,0,0,0,
// 1,1,1,1,1,0,0,0,
// 1,1,1,1,1,1,0,0,
// 1,1,1,1,1,1,1,0,
// 1,1,1,1,1,1,1,1]
```

```python
# OmniAttention-MMA blockmask implementation(shared KV)
# ============================================================================
# Kernel Arguments Design
# ============================================================================
# Input tensors:
#   Q: [batch, nheads, seqlen, head_dim] - Query tensor
#   K: [batch, nheads, seqlen, head_dim] - Key tensor  
#   V: [batch, nheads, seqlen, head_dim] - Value tensor
#
# Output tensor:
#   O: [batch, nheads, seqlen, head_dim] - Output tensor
#
# Block mask metadata (sparse format):
#   kv_num_blocks: [batch, nheads, num_q_blocks] - int32
#                  Number of active KV blocks for each Q block
#   kv_indices: [batch, nheads, num_q_blocks, max_blocks] - int32
#               KV block indices to process (0-indexed, relative to KV sequence)
#   block_mask_types: [batch, nheads, num_q_blocks, max_blocks] - int32
#                     Mask type per block: 0=MASKED, 1=CAUSAL, 2=FULL, 3=PARTIAL
#
# Stride information (for flexible tensor layouts):
#   kv_num_blocks_stride[0,1,2]: strides for batch, head, q_block dimensions
#   kv_indices_stride[0,1,2,3]: strides for batch, head, q_block, kv_idx dimensions
#   block_mask_types_stride[0,1,2,3]: same as kv_indices_stride
#
# Dimensions:
#   seqlen: Sequence length (Q and KV have same length in shared KV case)
#   head_dim: Head dimension (32, 64, 96, 128, 256)
#   BLOCK_SIZE: Block size used in block mask (typically 128, must be >= Bc)
#   num_q_blocks: Number of Q blocks = ceil(seqlen / BLOCK_SIZE)
#   max_blocks: Maximum number of active KV blocks per Q block
#
# Grid/Block configuration:
#   grid.x = num_q_blocks
#   grid.y = batch * nheads
#   block.x = kNumThreads (e.g., 128 for 4 warps)
#
# Template parameters (compile-time):
#   kHeadDim: Head dimension
#   kStage: Pipeline stage (1=no prefetch, 2=with prefetch)
#   kMmaAtomM, kMmaAtomN, kMmaAtomK: MMA atom sizes (16, 8, 16)
#   kMmaTileSeqLenQ, kMmaTileSeqLenK: MMA tile sizes
#   kWarpTileSeqLenQ, kWarpTileSeqLenK: Warp tile sizes
#   Br: Q block size = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ
#   Bc: KV tile size = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK

# ============================================================================
# Stage 1: Load Q Block (load once, reuse for all KV blocks)
# ============================================================================
# Get block indices
batch_id = blockIdx.y / nheads
head_id = blockIdx.y % nheads
q_block_id = blockIdx.x  # [0, num_q_blocks)

# Compute Q block boundaries
q_block_start = q_block_id * BLOCK_SIZE
q_block_end = min(q_block_start + BLOCK_SIZE, seqlen)
q_tile_start = q_block_start  # Q tile aligned with Q block in this design
q_tile_end = min(q_tile_start + Br, seqlen)

# Load Q block from global memory to shared memory
async_copy_Q_g2s():
    for each thread:
        q_row = thread_mapped_row  # Within Br
        q_col = thread_mapped_col  # Within head_dim
        if (q_tile_start + q_row) < q_tile_end:
            gmem_addr = Q_gmem_offset + (q_tile_start + q_row) * head_dim + q_col
            smem_addr = smem_Q_base + q_row * (head_dim + padQ) + q_col
            CP_ASYNC_CG(smem_addr, gmem_addr, 16 bytes)
    CP_ASYNC_COMMIT_GROUP()
    CP_ASYNC_WAIT_GROUP(0)
    __syncthreads()

# Initialize online softmax statistics (per Q row)
row_max_old = -infinity  # Running max per row
row_sum_old = 0.0        # Running sum of exp per row
O_global = zeros([Br, head_dim])  # Accumulated output per row

# ============================================================================
# Stage 2: Iterate over Active KV Blocks (from block mask)
# ============================================================================
# Get number of active KV blocks for this Q block
num_kv_blocks = kv_num_blocks[
    batch_id * kv_num_blocks_stride0 +
    head_id * kv_num_blocks_stride1 +
    q_block_id * kv_num_blocks_stride2
]

# Outer loop: iterate over active KV blocks
for kv_idx in range(0, num_kv_blocks):
    
    # Get KV block index and mask type
    kv_block_idx_off = (
        batch_id * kv_indices_stride0 +
        head_id * kv_indices_stride1 +
        q_block_id * kv_indices_stride2 +
        kv_idx * kv_indices_stride3
    )
    kv_block = kv_indices[kv_block_idx_off]  # KV block index (0-indexed)
    
    mask_type_off = (
        batch_id * block_mask_types_stride0 +
        head_id * block_mask_types_stride1 +
        q_block_id * block_mask_types_stride2 +
        kv_idx * block_mask_types_stride3
    )
    mask_type = block_mask_types[mask_type_off]
    
    # Skip fully masked blocks
    if mask_type == MASKED:
        continue
    
    # Compute KV block boundaries (in sequence indices)
    kv_block_start = kv_block * BLOCK_SIZE
    kv_block_end = min(kv_block_start + BLOCK_SIZE, seqlen)
    
    # Determine which tiles within this KV block to process
    # A KV block may span multiple tiles if BLOCK_SIZE > Bc
    first_tile = kv_block_start // Bc  # First tile index
    last_tile = (kv_block_end + Bc - 1) // Bc  # Last tile index (exclusive)
    
    # Inner loop: iterate over tiles within this KV block
    for tile_K_seqlen in range(first_tile, last_tile):
        
        # Compute tile boundaries
        tile_kv_start = tile_K_seqlen * Bc
        tile_kv_end = min(tile_kv_start + Bc, seqlen)
        
        # Determine effective KV range for this tile (may be partial)
        effective_kv_start = max(tile_kv_start, kv_block_start)
        effective_kv_end = min(tile_kv_end, kv_block_end)
        
        # Check if this tile is fully within the KV block
        is_full_tile = (tile_kv_start >= kv_block_start and 
                       tile_kv_end <= kv_block_end)
        
        # ========== 2.1 Load K Tile ==========
        async_copy_K_g2s():
            for each thread:
                k_row = thread_mapped_row  # Within Bc
                k_col = thread_mapped_col  # Within head_dim
                kv_idx_abs = tile_kv_start + k_row
                if kv_idx_abs < tile_kv_end:
                    gmem_addr = K_gmem_offset + kv_idx_abs * head_dim + k_col
                    smem_addr = smem_K_base + k_row * (head_dim + padK) + k_col
                    CP_ASYNC_CG(smem_addr, gmem_addr, 16 bytes)
        CP_ASYNC_COMMIT_GROUP()
        CP_ASYNC_WAIT_GROUP(0)
        __syncthreads()
        
        # ========== 2.2 Compute S = Q @ K^T (with mask application) ==========
        R_S = zeros([Br, Bc])  # Score matrix in registers
        
        for tile_K_d in range(0, head_dim / kMmaAtomK):
            # Load Q from smem to registers (m16k16)
            load_Q_s2r(R_Q, smem_Q, tile_K_d, warp_QP)
            
            # Load K from smem to registers (k16n8)
            load_K_s2r(R_K, smem_K, tile_K_d, warp_KV)
            
            # MMA computation: R_S += R_Q @ R_K^T (m16n8k16)
            for j in range(kWarpTileSeqLenK):
                HMMA16816(R_S[0][j], R_Q, R_K[j], R_S[0][j])
        
        __syncthreads()
        
        # ========== 2.3 Apply Block Mask to S Matrix ==========
        # Apply mask based on mask_type and tile boundaries
        for q_row_local in range(0, Br):
            q_idx_abs = q_tile_start + q_row_local
            if q_idx_abs >= q_tile_end:
                continue
            
            for kv_col_local in range(0, Bc):
                kv_idx_abs = tile_kv_start + kv_col_local
                
                # Skip if outside effective KV range
                if kv_idx_abs < effective_kv_start or kv_idx_abs >= effective_kv_end:
                    R_S[q_row_local][kv_col_local] = -infinity
                    continue
                
                # Apply mask based on type
                if mask_type == FULL:
                    # No masking needed
                    pass
                elif mask_type == CAUSAL:
                    # Causal mask: q_idx >= kv_idx
                    if q_idx_abs < kv_idx_abs:
                        R_S[q_row_local][kv_col_local] = -infinity
                elif mask_type == PARTIAL:
                    # Partial mask: handle cross-boundary cases
                    # This requires additional metadata or computation
                    # For now, treat as CAUSAL if crossing boundary
                    if not is_full_tile:
                        # Check if this position should be masked
                        # (Implementation depends on partial mask definition)
                        if should_mask_partial(q_idx_abs, kv_idx_abs, 
                                             q_block_start, q_block_end,
                                             kv_block_start, kv_block_end):
                            R_S[q_row_local][kv_col_local] = -infinity
        
        # ========== 2.4 Load V Tile ==========
        async_copy_V_g2s():
            for each thread:
                v_row = thread_mapped_row  # Within Bc
                v_col = thread_mapped_col  # Within head_dim
                kv_idx_abs = tile_kv_start + v_row
                if kv_idx_abs < tile_kv_end:
                    gmem_addr = V_gmem_offset + kv_idx_abs * head_dim + v_col
                    smem_addr = smem_V_base + v_row * (head_dim + padV) + v_col
                    CP_ASYNC_CG(smem_addr, gmem_addr, 16 bytes)
        CP_ASYNC_COMMIT_GROUP()
        CP_ASYNC_WAIT_GROUP(0)
        __syncthreads()
        
        # ========== 2.5 Online Softmax: Compute P = softmax(S) ==========
        # 2.5.1 Compute row max (with mask applied)
        row_max_new = reduce_max(R_S * scale)  # Thread -> Warp -> Block reduce
        row_max_global = max(row_max_old, row_max_new)
        
        # 2.5.2 Compute exp and row sum
        P = exp(R_S * scale - row_max_global)  # Update R_S to P
        row_sum_new = reduce_sum(P)  # Thread -> Warp -> Block reduce
        
        # ========== 2.6 Compute O_partial = P @ V ==========
        R_O = zeros([Br, head_dim])  # Current tile output
        
        for tile_V_Bc in range(0, Bc / kMmaAtomK):
            # Load V from smem to registers (k16n8, transposed)
            load_V_s2r_trans(R_V, smem_V, tile_V_Bc, warp_KV)
            
            # MMA computation: R_O += P[:, Bc_slice] @ V[Bc_slice, :]
            w = tile_V_Bc * 2  # Select P column block
            for j in range(kWarpTileHeadDimV):
                HMMA16816(R_O[0][j], R_S[0][w], R_S[0][w+1], R_V[j], R_O[0][j])
        
        __syncthreads()
        
        # ========== 2.7 Online Rescaling (FlashAttention-2 core) ==========
        rescale_factor = exp(row_max_old - row_max_global)
        
        # Rescale old output and accumulate new output
        O_global = rescale_factor * O_global + R_O
        
        # Update statistics
        row_sum_global = rescale_factor * row_sum_old + row_sum_new
        row_max_old = row_max_global
        row_sum_old = row_sum_global

# ============================================================================
# Stage 3: Final Normalization and Write Back
# ============================================================================
# Final rescale: O_final = O_global / row_sum_global
rescale_factor_final = 1.0 / row_sum_global
O_final = O_global * rescale_factor_final

# Write back to global memory (collective store with warp shuffle)
collective_store_O_gmem(O_final, O_gmem_offset, q_tile_start)

# ============================================================================
# Helper Function: should_mask_partial
# ============================================================================
# Determines if a (q_idx, kv_idx) position should be masked in a PARTIAL block
# This handles cases where blocks cross tile boundaries
def should_mask_partial(q_idx, kv_idx, q_block_start, q_block_end,
                       kv_block_start, kv_block_end):
    # Example: VLM-style interleaved pattern
    # - Some rows in Q block attend to full KV block
    # - Some rows in Q block attend to partial KV block (causal-like)
    # 
    # Implementation depends on specific partial mask pattern:
    # - Could use additional metadata (e.g., row_mask_ranges)
    # - Could compute based on position relative to boundaries
    # - Could use a lookup table or bitmask
    
    # For now, simple heuristic: treat as causal if near boundaries
    q_rel = q_idx - q_block_start
    kv_rel = kv_idx - kv_block_start
    
    # Example: first half of Q block is causal, second half is full
    if q_rel < (q_block_end - q_block_start) // 2:
        return q_idx < kv_idx  # Causal
    else:
        return False  # Full
    
    # Or: use row-specific mask ranges stored in additional tensor
    # row_mask_start = row_mask_ranges[q_idx, 0]
    # row_mask_end = row_mask_ranges[q_idx, 1]
    # return kv_idx < row_mask_start or kv_idx >= row_mask_end
```

# ============================================================================
# Design Notes
# ============================================================================
# 1. Block Boundary Handling:
#    - A KV block (BLOCK_SIZE) may span multiple tiles (Bc)
#    - Each tile is processed independently with proper boundary checks
#    - Mask is applied per-tile based on effective_kv_start/end
#
# 2. Mask Types:
#    - MASKED: Skip entire block (no K/V load, no computation)
#    - FULL: No masking, process all positions in block
#    - CAUSAL: Apply causal mask (q_idx >= kv_idx) within block
#    - PARTIAL: Custom mask pattern (requires additional logic/metadata)
#
# 3. Cross-Boundary Blocks:
#    - When BLOCK_SIZE > Bc, a block spans multiple tiles
#    - Each tile processes its portion with proper mask application
#    - effective_kv_start/end ensure correct masking at boundaries
#
# 4. Memory Efficiency:
#    - Q loaded once per Q block, stays in smem
#    - K/V loaded per tile within active blocks
#    - S matrix never fully materialized (computed on-the-fly)
#    - Output accumulated incrementally with online softmax
#
# 5. Performance Considerations:
#    - Stage 1 (no prefetch): Simpler, lower memory usage
#    - Stage 2 (with prefetch): Overlap compute and memory, higher throughput
#    - Block mask sparsity reduces unnecessary K/V loads
#    - Mask application adds minimal overhead (element-wise operations)
```
