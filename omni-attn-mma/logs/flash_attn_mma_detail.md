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

# OmniAttention-MMA shared KV

```python
# ========== 阶段1: 加载Q (一次性加载，后续复用) ==========
async_copy_Q_g2s():
    for each thread:
        gmem_addr = Q_gmem_offset + (Q_tile_id * Br + row) * head_dim + col
        smem_addr = smem_Q_base + row * (head_dim + padQ) + col
        CP_ASYNC_CG(smem_addr, gmem_addr, 16 bytes)
    CP_ASYNC_COMMIT_GROUP()
    CP_ASYNC_WAIT_GROUP(0)
    __syncthreads()

# ========== 阶段2: 循环处理活跃的KV块 (基于block mask, 共享KV内存) ==========
kv_nb = kv_num_blocks[batch_id, head_id, q_block]  # 获取活跃KV块数量

for kv_idx in range(0, kv_nb):
    # 2.1 获取KV块索引和mask类型
    kv_block = kv_indices[batch_id, head_id, q_block, kv_idx]
    mask_type = block_mask_types[batch_id, head_id, q_block, kv_idx]
    
    if mask_type == BLOCK_MASK_MASKED:
        continue  # 跳过被mask的块
    
    # 2.2 处理PARTIAL mask
    partial_block_index = -1
    if mask_type == BLOCK_MASK_PARTIAL:
        # BUG_MAYBE: 使用block_mask_types_stride访问partial_block_mask_indices可能错误
        partial_block_index = partial_block_mask_indices[batch_id, head_id, q_block, kv_idx]
        if partial_block_index < 0:
            continue
    
    # 2.3 计算KV块边界
    kv_block_start = kv_block * KV_BLOCK_SIZE
    # BUG_MAYBE: 使用QKV_seqlen_orig而非QKV_seqlen可能导致越界
    kv_block_end = min(kv_block_start + KV_BLOCK_SIZE, QKV_seqlen_orig)
    
    # 2.4 加载K tile (顺序加载, 无预取)
    async_copy_K_g2s():
        for each thread:
            load_gmem_K_Bc = kv_block_start + load_smem_K_Bc
            if load_smem_K_Bc < Bc and load_gmem_K_Bc < kv_block_end:
                gmem_addr = K_gmem_offset + load_gmem_K_Bc * head_dim + col
                smem_addr = smem_K_base + load_smem_K_Bc * (head_dim + padK) + col
                CP_ASYNC_CG(smem_addr, gmem_addr, 16 bytes)
        CP_ASYNC_COMMIT_GROUP()
    CP_ASYNC_WAIT_GROUP(0)
    __syncthreads()
    
    # 2.5 计算 S = Q @ K^T
    R_S = zeros([Br, Bc])
    for tile_K_d in range(0, head_dim / 16):
        # 从smem加载Q到寄存器 (m16k16)
        load_Q_s2r(R_Q, smem_Q, tile_K_d, warp_QP)
        # 从smem加载K到寄存器 (k16n8)
        load_K_s2r(R_K, smem_K, tile_K_d, warp_KV)
        # MMA计算: R_S += R_Q @ R_K^T (m16n8k16)
        for j in range(kWarpTileSeqLenK):
            HMMA16816(R_S[0][j], R_Q, R_K[j], R_S[0][j])
    __syncthreads()
    
    # 2.6 应用scale
    S = S * scale
    
    # 2.7 应用CAUSAL mask
    if mask_type == BLOCK_MASK_CAUSAL:
        q_base = Q_tile_id * Br
        kv_base = kv_block * KV_BLOCK_SIZE
        # BUG_MAYBE: causal mask计算可能不正确, q_row和kv_col的索引计算
        for each score in R_S:
            q_row = q_base + row_in_tile
            kv_col = kv_base + col_in_tile
            if q_row < kv_col:
                S = -INFINITY
    
    # 2.8 应用PARTIAL mask
    if mask_type == BLOCK_MASK_PARTIAL:
        q_base = Q_tile_id * Br
        kv_base = kv_block * KV_BLOCK_SIZE
        q_block_start = q_block * Q_BLOCK_SIZE
        # BUG_MAYBE: partial mask索引计算复杂, 边界检查可能有问题
        for each score in R_S:
            q_row = q_base + row_in_tile
            kv_col = kv_base + col_in_tile
            local_q = q_row - q_block_start
            local_kv = kv_col - kv_base
            if valid_range(local_q, local_kv):
                mask_offset = partial_block_index * block_area + local_q * KV_BLOCK_SIZE + local_kv
                if !partial_block_masks[mask_offset]:
                    S = -INFINITY
    
    # 2.9 加载V tile (覆盖K的smem, K已使用完毕)
    # NOTE: V_tile_smem = K_tile_smem (共享内存), V覆盖K的位置
    async_copy_V_g2s():
        for each thread:
            load_gmem_V_Bc = kv_block_start + load_smem_V_Bc
            if load_smem_V_Bc < Bc and load_gmem_V_Bc < kv_block_end:
                gmem_addr = V_gmem_offset + load_gmem_V_Bc * head_dim + col
                smem_addr = smem_V_base + load_smem_V_Bc * (head_dim + padV) + col  # 复用K的smem
                CP_ASYNC_CG(smem_addr, gmem_addr, 16 bytes)
        CP_ASYNC_COMMIT_GROUP()
    CP_ASYNC_WAIT_GROUP(0)
    __syncthreads()
    
    # 2.10 Online Softmax: 计算 P = softmax(S)
    row_max_new = reduce_max(S)  # Warp reduce
    row_max_global = max(row_max_old, row_max_new)
    
    P = exp(S - row_max_global)  # 更新R_S为P
    row_sum_new = reduce_sum(P)  # Warp reduce
    
    # 2.11 计算 O_partial = P @ V
    R_O = zeros([Br, head_dim])
    for tile_V_Bc in range(0, Bc / 16):
        # 从smem加载V到寄存器 (k16n8, 转置)
        load_V_s2r_trans(R_V, smem_V, tile_V_Bc, warp_KV)
        w = tile_V_Bc * 2
        for j in range(kWarpTileHeadDimV):
            HMMA16816(R_O[0][j], R_S[0][w], R_S[0][w+1], R_V[j], R_O[0][j])
    __syncthreads()
    
    # 2.12 Online Rescaling
    rescale_factor = exp(row_max_old - row_max_global)
    O_global = rescale_factor * O_global + R_O
    row_sum_global = rescale_factor * row_sum_old + row_sum_new
    row_max_old = row_max_global
    row_sum_old = row_sum_global

# ========== 阶段3: 最终归一化并写回 ==========
rescale_factor_final = 1.0 / row_sum_global
O_final = O_global * rescale_factor_final

# 写回全局内存
for each thread:
    row0 = Q_tile_id * Br + warp_QP * kMmaAtomM + row_in_tile
    row1 = row0 + 8
    col0 = j * kMmaAtomN + col_pair
    col1 = col0 + 1
    if row0 < QKV_seqlen:
        O[O_gmem_offset + row0 * head_dim + col0] = O_final[0]
        O[O_gmem_offset + row0 * head_dim + col1] = O_final[1]
    if row1 < QKV_seqlen:
        O[O_gmem_offset + row1 * head_dim + col0] = O_final[2]
        O[O_gmem_offset + row1 * head_dim + col1] = O_final[3]
```

# OmniAttention-MMA prefetch (double-buffering)

```python
# ========== 阶段1: 加载Q (一次性加载，后续复用) ==========
async_copy_Q_g2s():
    for each thread:
        gmem_addr = Q_gmem_offset + (Q_tile_id * Br + row) * head_dim + col
        smem_addr = smem_Q_base + row * (head_dim + padQ) + col
        CP_ASYNC_CG(smem_addr, gmem_addr, 16 bytes)
    CP_ASYNC_COMMIT_GROUP()
    CP_ASYNC_WAIT_GROUP(0)
    __syncthreads()

# ========== 阶段2: 循环处理活跃的KV块 (基于block mask, 双缓冲预取) ==========
kv_nb = kv_num_blocks[batch_id, head_id, q_block]  # 获取活跃KV块数量

for kv_idx in range(0, kv_nb):
    # 2.1 获取KV块索引和mask类型
    kv_block = kv_indices[batch_id, head_id, q_block, kv_idx]
    mask_type = block_mask_types[batch_id, head_id, q_block, kv_idx]
    
    if mask_type == BLOCK_MASK_MASKED:
        continue  # 跳过被mask的块
    
    # 2.2 处理PARTIAL mask
    partial_block_index = -1
    if mask_type == BLOCK_MASK_PARTIAL:
        partial_block_index = partial_block_mask_indices[batch_id, head_id, q_block, kv_idx]
        if partial_block_index < 0:
            continue
    
    # 2.3 计算KV块边界
    kv_block_start = kv_block * KV_BLOCK_SIZE
    kv_block_end = min(kv_block_start + KV_BLOCK_SIZE, QKV_seqlen_orig)
    
    # 2.4 加载K tile (第一次迭代同步加载, 后续迭代使用预取的K)
    if kv_idx == 0:
        # 第一次迭代: 同步加载K到buffer 0
        async_copy_K_g2s(buffer_id=0):
            for each thread:
                load_gmem_K_Bc = kv_block_start + load_smem_K_Bc
                if load_smem_K_Bc < Bc and load_gmem_K_Bc < kv_block_end:
                    gmem_addr = K_gmem_offset + load_gmem_K_Bc * head_dim + col
                    smem_addr = smem_K_base + (0 * K_tile_size + load_smem_K_Bc * (head_dim + padK) + col)
                    CP_ASYNC_CG(smem_addr, gmem_addr, 16 bytes)
            CP_ASYNC_COMMIT_GROUP()
        CP_ASYNC_WAIT_GROUP(0)
        __syncthreads()
    
    # 2.5 预取V tile (异步加载到buffer 1, 在计算Q@K^T之前)
    async_copy_V_g2s(buffer_id=1):  # 使用buffer 1 (与K的buffer 0分离)
        for each thread:
            load_gmem_V_Bc = kv_block_start + load_smem_V_Bc
            if load_smem_V_Bc < Bc and load_gmem_V_Bc < kv_block_end:
                gmem_addr = V_gmem_offset + load_gmem_V_Bc * head_dim + col
                smem_addr = smem_V_base + (1 * V_tile_size + load_smem_V_Bc * (head_dim + padV) + col)
                CP_ASYNC_CG(smem_addr, gmem_addr, 16 bytes)
        CP_ASYNC_COMMIT_GROUP()  # 异步加载, 不等待
    
    # 2.6 计算 S = Q @ K^T (使用buffer 0中的K)
    R_S = zeros([Br, Bc])
    for tile_K_d in range(0, head_dim / 16):
        # 从smem加载Q到寄存器 (m16k16)
        load_Q_s2r(R_Q, smem_Q, tile_K_d, warp_QP)
        # 从smem buffer 0加载K到寄存器 (k16n8)
        load_K_s2r(R_K, smem_K[buffer_id=0], tile_K_d, warp_KV)
        # MMA计算: R_S += R_Q @ R_K^T (m16n8k16)
        for j in range(kWarpTileSeqLenK):
            HMMA16816(R_S[0][j], R_Q, R_K[j], R_S[0][j])
    __syncthreads()
    
    # 2.7 预取下一个K tile (如果存在, 异步加载到buffer 0, 在计算P@V之前)
    if (kv_idx + 1) < kv_nb:
        next_kv_block = kv_indices[batch_id, head_id, q_block, kv_idx + 1]
        next_mask_type = block_mask_types[batch_id, head_id, q_block, kv_idx + 1]
        if next_mask_type != BLOCK_MASK_MASKED:
            async_copy_K_g2s(buffer_id=0):  # 预取到buffer 0
                next_kv_block_start = next_kv_block * KV_BLOCK_SIZE
                next_kv_block_end = min(next_kv_block_start + KV_BLOCK_SIZE, QKV_seqlen_orig)
                for each thread:
                    load_gmem_K_Bc = next_kv_block_start + load_smem_K_Bc
                    if load_smem_K_Bc < Bc and load_gmem_K_Bc < next_kv_block_end:
                        gmem_addr = K_gmem_offset + load_gmem_K_Bc * head_dim + col
                        smem_addr = smem_K_base + (0 * K_tile_size + load_smem_K_Bc * (head_dim + padK) + col)
                        CP_ASYNC_CG(smem_addr, gmem_addr, 16 bytes)
                CP_ASYNC_COMMIT_GROUP()  # 异步加载, 不等待
    
    # 2.8 应用scale
    S = S * scale
    
    # 2.9 应用CAUSAL mask
    if mask_type == BLOCK_MASK_CAUSAL:
        q_base = Q_tile_id * Br
        kv_base = kv_block * KV_BLOCK_SIZE
        for each score in R_S:
            q_row = q_base + row_in_tile
            kv_col = kv_base + col_in_tile
            if q_row < kv_col:
                S = -INFINITY
    
    # 2.10 应用PARTIAL mask
    if mask_type == BLOCK_MASK_PARTIAL:
        q_base = Q_tile_id * Br
        kv_base = kv_block * KV_BLOCK_SIZE
        q_block_start = q_block * Q_BLOCK_SIZE
        for each score in R_S:
            q_row = q_base + row_in_tile
            kv_col = kv_base + col_in_tile
            local_q = q_row - q_block_start
            local_kv = kv_col - kv_base
            if valid_range(local_q, local_kv):
                mask_offset = partial_block_index * block_area + local_q * KV_BLOCK_SIZE + local_kv
                if !partial_block_masks[mask_offset]:
                    S = -INFINITY
    
    # 2.11 等待V就绪 (预取在buffer 1中)
    CP_ASYNC_WAIT_GROUP(1)  # 等待V加载完成
    __syncthreads()
    
    # 2.12 Online Softmax: 计算 P = softmax(S)
    row_max_new = reduce_max(S)  # Warp reduce
    row_max_global = max(row_max_old, row_max_new)
    
    P = exp(S - row_max_global)  # 更新R_S为P
    row_sum_new = reduce_sum(P)  # Warp reduce
    
    # 2.13 计算 O_partial = P @ V (使用buffer 1中的V)
    R_O = zeros([Br, head_dim])
    for tile_V_Bc in range(0, Bc / 16):
        # 从smem buffer 1加载V到寄存器 (k16n8, 转置)
        load_V_s2r_trans(R_V, smem_V[buffer_id=1], tile_V_Bc, warp_KV)
        w = tile_V_Bc * 2
        for j in range(kWarpTileHeadDimV):
            HMMA16816(R_O[0][j], R_S[0][w], R_S[0][w+1], R_V[j], R_O[0][j])
    __syncthreads()
    
    # 2.14 等待下一个K就绪 (如果预取了)
    if (kv_idx + 1) < kv_nb:
        CP_ASYNC_WAIT_GROUP(0)  # 等待下一个K加载完成
        __syncthreads()
    
    # 2.15 Online Rescaling
    rescale_factor = exp(row_max_old - row_max_global)
    O_global = rescale_factor * O_global + R_O
    row_sum_global = rescale_factor * row_sum_old + row_sum_new
    row_max_old = row_max_global
    row_sum_old = row_sum_global

# ========== 阶段3: 最终归一化并写回 ==========
rescale_factor_final = 1.0 / row_sum_global
O_final = O_global * rescale_factor_final

# 写回全局内存
for each thread:
    row0 = Q_tile_id * Br + warp_QP * kMmaAtomM + row_in_tile
    row1 = row0 + 8
    col0 = j * kMmaAtomN + col_pair
    col1 = col0 + 1
    if row0 < QKV_seqlen:
        O[O_gmem_offset + row0 * head_dim + col0] = O_final[0]
        O[O_gmem_offset + row0 * head_dim + col1] = O_final[1]
    if row1 < QKV_seqlen:
        O[O_gmem_offset + row1 * head_dim + col0] = O_final[2]
        O[O_gmem_offset + row1 * head_dim + col1] = O_final[3]
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
