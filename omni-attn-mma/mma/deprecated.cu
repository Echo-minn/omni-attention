#include "utils.h"

// Omni-Attention: Block-sparse attention with FA2 split-Q tiling
// Supports MASKED (skip), CAUSAL (lower-triangular), FULL (dense) block types

#define BLOCK_MASK_MASKED 0
#define BLOCK_MASK_CAUSAL 1
#define BLOCK_MASK_FULL 2

// -----------------------------------------------------------------------------
// Optimized MMA kernel (kept for future re-enablement once correctness
// is fully validated with the simple kernel above).
// -----------------------------------------------------------------------------

template <
    const int kHeadDim, const int kMmaAtomM, const int kMmaAtomN, const int kMmaAtomK,
    const int kMmaTileSeqLenQ, const int kMmaTileSeqLenK,
    const int kMmaTileSeqLenP, const int kMmaTileHeadDimV,
    const int kWarpTileSeqLenQ, const int kWarpTileSeqLenK,
    const int kWarpTileSeqLenP, const int kWarpTileHeadDimV,
    const int kOStorageAccFloat32, const int kStage,
    const int kPadQ, const int kPadK, const int kPadV>
__global__ void __launch_bounds__(WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK)
    omni_attn_mma_stages_split_q_shared_kv_kernel(
        half *__restrict__ Q, half *__restrict__ K, half *__restrict__ V,
        half *__restrict__ O,
        const int *__restrict__ kv_num_blocks,
        const int *__restrict__ kv_indices,
        const int *__restrict__ block_mask_types,
        int QKV_seqlen, int QKV_head, int QKV_batch,
        int num_q_blocks, int max_blocks,
        int kv_num_blocks_stride0, int kv_num_blocks_stride1, int kv_num_blocks_stride2,
        int kv_indices_stride0, int kv_indices_stride1, int kv_indices_stride2, int kv_indices_stride3,
        int block_mask_types_stride0, int block_mask_types_stride1, int block_mask_types_stride2, int block_mask_types_stride3) {

  static_assert(kMmaAtomM == 16 && kMmaAtomN == 8 && kMmaAtomK == 16);
  static_assert(kMmaTileSeqLenQ <= 8 && kMmaTileSeqLenK == 1);
  static_assert(kMmaTileSeqLenP <= 8 && kMmaTileHeadDimV == 1);
  static_assert(kWarpTileSeqLenQ == 1 && kWarpTileSeqLenK <= 16);
  static_assert(kWarpTileSeqLenP == 1 && kWarpTileHeadDimV == (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV)));
  static_assert(kStage >= 1 && kStage <= 2);
  static_assert(kPadQ % 8 == 0 && kPadK % 8 == 0 && kPadV % 8 == 0);

  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  const float scale = 1.0f / sqrtf((float)kHeadDim);

  const int batch_id = blockIdx.y / QKV_head;
  const int head_id = blockIdx.y % QKV_head;
  const int q_tile_id = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;
  const int warp_QP = warp_id;

  const int qkv_offset = (batch_id * QKV_head + head_id) * QKV_seqlen * kHeadDim;

  // Thread-to-tile mapping for coalesced loads
  const int load_row = tid / (kNumThreads / Br);
  const int load_col = (tid % (kNumThreads / Br)) * (kHeadDim / (kNumThreads / Br));
  const int gmem_q_row = q_tile_id * Br + load_row;
  
  if (gmem_q_row >= QKV_seqlen) return;

  extern __shared__ half smem[];
  constexpr int Q_tile_size = Br * (kHeadDim + kPadQ);
  constexpr int K_tile_size = Bc * (kHeadDim + kPadK);
  constexpr int V_tile_size = Bc * (kHeadDim + kPadV);
  half *Q_smem = smem;
  half *K_smem = Q_smem + Q_tile_size;
  half *V_smem = K_smem;  // KV share smem

  uint32_t Q_smem_ptr = __cvta_generic_to_shared(Q_smem);
  uint32_t K_smem_ptr = __cvta_generic_to_shared(K_smem);
  uint32_t V_smem_ptr = __cvta_generic_to_shared(V_smem);

  // Online softmax state
  float row_max_old[kWarpTileSeqLenQ][2];
  float row_sum_old[kWarpTileSeqLenQ][2];
  fill_2D_regs<float, kWarpTileSeqLenQ, 2>(row_max_old, -INFINITY);
  fill_2D_regs<float, kWarpTileSeqLenQ, 2>(row_sum_old, 0.0f);

  // Register allocation
  constexpr bool kPrefetchQ = ((kHeadDim / kMmaAtomK) <= 8) && (kHeadDim < 64);
  constexpr bool kPrefetchKV = (kStage == 2);
  constexpr int kNumQRegs = kPrefetchQ ? (kHeadDim / kMmaAtomK) : 1;
  
  uint32_t R_Q[kNumQRegs][kWarpTileSeqLenQ][4];
  uint32_t R_K[kWarpTileSeqLenK][2];
  uint32_t R_V[kWarpTileHeadDimV][2];
  uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][2];
  uint32_t R_O[kWarpTileSeqLenP][kWarpTileHeadDimV][2];
  uint32_t R_D[kWarpTileSeqLenP][kWarpTileHeadDimV][kOStorageAccFloat32 ? 4 : 2];
  fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV, (kOStorageAccFloat32 ? 4 : 2)>(R_D, 0);

  // Load Q tile once
  {
    int addr = qkv_offset + gmem_q_row * kHeadDim + load_col;
    uint32_t sptr = Q_smem_ptr + (load_row * (kHeadDim + kPadQ) + load_col) * sizeof(half);
    #pragma unroll
    for (int i = 0; i < (kHeadDim / (kNumThreads / Br)); i += 8) {
      CP_ASYNC_CG(sptr + i * 2, &Q[addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();
  }

  // Get active KV block count
  int num_kv = kv_num_blocks[batch_id * kv_num_blocks_stride0 + 
                              head_id * kv_num_blocks_stride1 + 
                              q_tile_id * kv_num_blocks_stride2];

  // Main loop over active KV blocks
  #pragma unroll 1
  for (int kv_idx = 0; kv_idx < num_kv; ++kv_idx) {
    int indices_off = batch_id * kv_indices_stride0 + head_id * kv_indices_stride1 + 
                      q_tile_id * kv_indices_stride2 + kv_idx * kv_indices_stride3;
    int kv_block = kv_indices[indices_off];
    int mask_type = block_mask_types[batch_id * block_mask_types_stride0 + 
                                      head_id * block_mask_types_stride1 +
                                      q_tile_id * block_mask_types_stride2 + 
                                      kv_idx * block_mask_types_stride3];

    if (mask_type == BLOCK_MASK_MASKED) continue;

    int kv_offset = kv_block * Bc;

    // Load K
    if constexpr (kPrefetchKV) {
      if (kv_idx == 0 && load_row < Bc) {
        int addr = qkv_offset + (kv_offset + load_row) * kHeadDim + load_col;
        uint32_t sptr = K_smem_ptr + (load_row * (kHeadDim + kPadK) + load_col) * sizeof(half);
        #pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(sptr + i * 2, &K[addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      }
      if (kv_idx == 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
      }
      // Prefetch V (alternate between two buffers for double-buffering)
      if (load_row < Bc) {
        int addr = qkv_offset + (kv_offset + load_row) * kHeadDim + load_col;
        int v_buffer_offset = (kv_idx & 1) * K_tile_size;  // 0 or K_tile_size
        uint32_t sptr = V_smem_ptr + (v_buffer_offset + load_row * (kHeadDim + kPadV) + load_col) * sizeof(half);
        #pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(sptr + i * 2, &V[addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      }
    } else {
      if (load_row < Bc) {
        int addr = qkv_offset + (kv_offset + load_row) * kHeadDim + load_col;
        uint32_t sptr = K_smem_ptr + (load_row * (kHeadDim + kPadK) + load_col) * sizeof(half);
        #pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(sptr + i * 2, &K[addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      }
      CP_ASYNC_WAIT_GROUP(0);
      __syncthreads();
    }

    // Load Q into registers on first iteration
    if constexpr (kPrefetchQ) {
      if (kv_idx == 0) {
        if constexpr (!kPrefetchKV) { CP_ASYNC_WAIT_GROUP(0); }
        else { CP_ASYNC_WAIT_GROUP(1); }
        __syncthreads();
        #pragma unroll
        for (int d = 0; d < (kHeadDim / kMmaAtomK); ++d) {
          int row = warp_QP * kMmaAtomM + lane_id % 16;
          int col = d * kMmaAtomK + (lane_id / 16) * 8;
          uint32_t ptr = Q_smem_ptr + (row * (kHeadDim + kPadQ) + col) * sizeof(half);
          LDMATRIX_X4(R_Q[d][0][0], R_Q[d][0][1], R_Q[d][0][2], R_Q[d][0][3], ptr);
        }
        __syncthreads();
      }
    }

    // Compute S = Q @ K^T
    fill_3D_regs<uint32_t, kWarpTileSeqLenQ, kWarpTileSeqLenK, 2>(R_S, 0);
    #pragma unroll
    for (int d = 0; d < (kHeadDim / kMmaAtomK); ++d) {
      if constexpr (!kPrefetchQ) {
        int row = warp_QP * kMmaAtomM + lane_id % 16;
        int col = d * kMmaAtomK + (lane_id / 16) * 8;
        uint32_t ptr = Q_smem_ptr + (row * (kHeadDim + kPadQ) + col) * sizeof(half);
        LDMATRIX_X4(R_Q[0][0][0], R_Q[0][0][1], R_Q[0][0][2], R_Q[0][0][3], ptr);
      }

      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        int k_row = j * kMmaAtomN + lane_id % 8;
        int k_col = d * kMmaAtomK + ((lane_id / 8) % 2) * 8;
        uint32_t ptr = K_smem_ptr + (k_row * (kHeadDim + kPadK) + k_col) * sizeof(half);
        LDMATRIX_X2(R_K[j][0], R_K[j][1], ptr);
      }

      int q_idx = kPrefetchQ ? d : 0;
      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        HMMA16816(R_S[0][j][0], R_S[0][j][1],
                  R_Q[q_idx][0][0], R_Q[q_idx][0][1], R_Q[q_idx][0][2], R_Q[q_idx][0][3],
                  R_K[j][0], R_K[j][1], R_S[0][j][0], R_S[0][j][1]);
      }
    }
    
    // Apply scale to QK^T scores
    #pragma unroll
    for (int j = 0; j < kWarpTileSeqLenK; ++j) {
      half *S = reinterpret_cast<half *>(&R_S[0][j][0]);
      S[0] = __hmul(S[0], __float2half(scale));
      S[1] = __hmul(S[1], __float2half(scale));
      S[2] = __hmul(S[2], __float2half(scale));
      S[3] = __hmul(S[3], __float2half(scale));
    }
    __syncthreads();

    // Apply causal mask - correct MMA m16n8k16 fragment layout
    if (mask_type == BLOCK_MASK_CAUSAL) {
      int q_base = q_tile_id * Br + warp_QP * kMmaAtomM;
      int kv_base = kv_block * Bc;
      
      // MMA m16n8k16 output layout: thread lane_id holds elements at:
      // row_0 = lane_id % 8,     row_1 = lane_id % 8 + 8
      // col = (lane_id / 8) * 2, col + 1
      int row_in_tile = lane_id % 8;
      int col_pair = (lane_id / 8) * 2;  // 0, 2, 4, or 6
      
      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        int q_row_0 = q_base + row_in_tile;
        int q_row_1 = q_base + row_in_tile + 8;
        int kv_col = kv_base + j * kMmaAtomN + col_pair;
        
        half *S = reinterpret_cast<half *>(&R_S[0][j][0]);
        // c0: (row_0, col), c1: (row_0, col+1), c2: (row_1, col), c3: (row_1, col+1)
        if (q_row_0 < kv_col)     S[0] = __float2half(-INFINITY);
        if (q_row_0 < kv_col + 1) S[1] = __float2half(-INFINITY);
        if (q_row_1 < kv_col)     S[2] = __float2half(-INFINITY);
        if (q_row_1 < kv_col + 1) S[3] = __float2half(-INFINITY);
      }
    }

    // Load V if not prefetched
    if constexpr (!kPrefetchKV) {
      if (load_row < Bc) {
        int addr = qkv_offset + (kv_offset + load_row) * kHeadDim + load_col;
        uint32_t sptr = V_smem_ptr + (load_row * (kHeadDim + kPadV) + load_col) * sizeof(half);
        #pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(sptr + i * 2, &V[addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      }
    }

    // Prefetch next K
    if constexpr (kPrefetchKV) {
      if (kv_idx + 1 < num_kv && load_row < Bc) {
        int next_block = kv_indices[batch_id * kv_indices_stride0 + head_id * kv_indices_stride1 +
                                     q_tile_id * kv_indices_stride2 + (kv_idx + 1) * kv_indices_stride3];
        int next_off = next_block * Bc;
        int addr = qkv_offset + (next_off + load_row) * kHeadDim + load_col;
        uint32_t sptr = K_smem_ptr + (load_row * (kHeadDim + kPadK) + load_col) * sizeof(half);
        #pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(sptr + i * 2, &K[addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      }
    }

    // Online softmax: compute per-block row max and exp sum (tile-local),
    // then later merge with running (m_old, l_old, O_old).
    float row_max_new[kWarpTileSeqLenQ][2];
    float row_sum_new[kWarpTileSeqLenQ][2];
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(row_max_new, -INFINITY);
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(row_sum_new, 0.0f);

    // Row max
    // For m16n8k16: each thread processes 2 rows (row_0 = lane_id % 8, row_1 = lane_id % 8 + 8)
    // and 2 columns. For row-wise reduction, we need to reduce across threads with same lane_id % 8
    // There are 4 threads per row group: lane_id % 8 == 0, 8, 16, 24 for row_0
    int row_group = lane_id % 8;
    int base_lane = row_group;
    
    #pragma unroll
    for (int j = 0; j < kWarpTileSeqLenK; ++j) {
      half *S = reinterpret_cast<half *>(&R_S[0][j][0]);
      float m0 = __half2float(__hmax(S[0], S[1]));
      float m1 = __half2float(__hmax(S[2], S[3]));
      row_max_new[0][0] = fmaxf(row_max_new[0][0], m0);
      row_max_new[0][1] = fmaxf(row_max_new[0][1], m1);
    }
    // Row-wise reduction: reduce across threads with same lane_id % 8 (same Q row, different columns)
    // For m16n8k16: threads with lane_id % 8 == r process the same Q row (for r = 0..7)
    // There are 4 such threads per row: r, r+8, r+16, r+24
    // Reduce using shuffle across these 4 threads - get max from all threads in row group
    int row_base = row_group;  // Base thread for this row group (0-7)
    float max0 = row_max_new[0][0];
    float max1 = row_max_new[0][1];
    // Get values from all other threads in the row group (including base)
    for (int offset = 0; offset <= 24; offset += 8) {
      int src_lane = row_base + offset;
      if (src_lane < WARP_SIZE && src_lane != lane_id) {
        float val0 = __shfl_sync(0xffffffff, row_max_new[0][0], src_lane, WARP_SIZE);
        float val1 = __shfl_sync(0xffffffff, row_max_new[0][1], src_lane, WARP_SIZE);
        max0 = fmaxf(max0, val0);
        max1 = fmaxf(max1, val1);
      }
    }
    // Broadcast the reduced value from the base thread to all threads in row group
    row_max_new[0][0] = __shfl_sync(0xffffffff, max0, row_base, WARP_SIZE);
    row_max_new[0][1] = __shfl_sync(0xffffffff, max1, row_base, WARP_SIZE);

    // Exp and sum for this block (tile-local):
    // Use m_tile = row_max_new as reference, so row_sum_new stores l_tile.
    float m_tile_0 = row_max_new[0][0];
    float m_tile_1 = row_max_new[0][1];

    #pragma unroll
    for (int j = 0; j < kWarpTileSeqLenK; ++j) {
      half *S = reinterpret_cast<half *>(&R_S[0][j][0]);
      float4 p;
      p.x = __expf(__half2float(S[0]) - m_tile_0);
      p.y = __expf(__half2float(S[1]) - m_tile_0);
      p.z = __expf(__half2float(S[2]) - m_tile_1);
      p.w = __expf(__half2float(S[3]) - m_tile_1);
      row_sum_new[0][0] += p.x + p.y;
      row_sum_new[0][1] += p.z + p.w;
      S[0] = __float2half_rn(p.x);
      S[1] = __float2half_rn(p.y);
      S[2] = __float2half_rn(p.z);
      S[3] = __float2half_rn(p.w);
    }
    // Row-wise reduction: reduce across threads with same lane_id % 8 (same Q row, different columns)
    // Sum across the 4 threads in each row group (reuse row_base from above)
    float sum0 = row_sum_new[0][0];
    float sum1 = row_sum_new[0][1];
    // Get values from all other threads in the row group (including base)
    for (int offset = 0; offset <= 24; offset += 8) {
      int src_lane = row_base + offset;
      if (src_lane < WARP_SIZE && src_lane != lane_id) {
        float val0 = __shfl_sync(0xffffffff, row_sum_new[0][0], src_lane, WARP_SIZE);
        float val1 = __shfl_sync(0xffffffff, row_sum_new[0][1], src_lane, WARP_SIZE);
        sum0 += val0;
        sum1 += val1;
      }
    }
    // Broadcast the reduced value from the base thread to all threads in row group
    row_sum_new[0][0] = __shfl_sync(0xffffffff, sum0, row_base, WARP_SIZE);
    row_sum_new[0][1] = __shfl_sync(0xffffffff, sum1, row_base, WARP_SIZE);

    // Wait for V
    if constexpr (kPrefetchKV) {
      if (kv_idx + 1 < num_kv) {
        CP_ASYNC_WAIT_GROUP(1);
      } else {
        CP_ASYNC_WAIT_GROUP(0);
      }
    } else {
      CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

    // Compute O = P @ V
    fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV, 2>(R_O, 0);
    #pragma unroll
    for (int bc = 0; bc < (Bc / kMmaAtomK); ++bc) {
      #pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) {
        int v_row = bc * kMmaAtomK + lane_id % 16;
        int v_col = j * kMmaAtomN;
        int v_buffer_offset = kPrefetchKV ? ((kv_idx & 1) * K_tile_size) : 0;
        uint32_t ptr = V_smem_ptr + v_buffer_offset + 
               (v_row * (kHeadDim + kPadV) + v_col) * sizeof(half);
        LDMATRIX_X2_T(R_V[j][0], R_V[j][1], ptr);
      }

      int w = bc * 2;
      #pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) {
        HMMA16816(R_O[0][j][0], R_O[0][j][1],
                  R_S[0][w][0], R_S[0][w][1], R_S[0][w + 1][0], R_S[0][w + 1][1],
                  R_V[j][0], R_V[j][1], R_O[0][j][0], R_O[0][j][1]);
      }
    }
    __syncthreads();

    // Rescale O with online softmax correction.
    // We have:
    //   - tile-local max m_tile_{0,1} and sums row_sum_new (l_tile)
    //   - running state row_max_old (m_old) and row_sum_old (l_old)
    // We compute:
    //   m_new = max(m_old, m_tile)
    //   alpha = exp(m_old - m_new)
    //   beta  = exp(m_tile - m_new)
    //   l_new = alpha * l_old + beta * l_tile
    //   O_new = alpha * O_old + beta * O_tile
    float l_old_0 = row_sum_old[0][0];
    float l_old_1 = row_sum_old[0][1];
    float l_tile_0 = row_sum_new[0][0];
    float l_tile_1 = row_sum_new[0][1];

    float m_old_0 = row_max_old[0][0];
    float m_old_1 = row_max_old[0][1];

    float m_new_0;
    float m_new_1;
    float alpha_0, beta_0;
    float alpha_1, beta_1;

    // Row group 0
    if (l_old_0 == 0.0f) {
      // No previous contributions
      m_new_0 = m_tile_0;
      alpha_0 = 0.0f;
      beta_0 = 1.0f;
    } else if (l_tile_0 == 0.0f) {
      // No contributions from this tile
      m_new_0 = m_old_0;
      alpha_0 = 1.0f;
      beta_0 = 0.0f;
    } else {
      m_new_0 = fmaxf(m_old_0, m_tile_0);
      alpha_0 = __expf(m_old_0 - m_new_0);
      beta_0  = __expf(m_tile_0 - m_new_0);
    }

    // Row group 1
    if (l_old_1 == 0.0f) {
      m_new_1 = m_tile_1;
      alpha_1 = 0.0f;
      beta_1 = 1.0f;
    } else if (l_tile_1 == 0.0f) {
      m_new_1 = m_old_1;
      alpha_1 = 1.0f;
      beta_1 = 0.0f;
    } else {
      m_new_1 = fmaxf(m_old_1, m_tile_1);
      alpha_1 = __expf(m_old_1 - m_new_1);
      beta_1  = __expf(m_tile_1 - m_new_1);
    }

    #pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) {
      half *O = reinterpret_cast<half *>(&R_O[0][j][0]);
      if constexpr (kOStorageAccFloat32) {
        float *D = reinterpret_cast<float *>(&R_D[0][j][0]);
        D[0] = __fmaf_rn(alpha_0, D[0], beta_0 * __half2float(O[0]));
        D[1] = __fmaf_rn(alpha_0, D[1], beta_0 * __half2float(O[1]));
        D[2] = __fmaf_rn(alpha_1, D[2], beta_1 * __half2float(O[2]));
        D[3] = __fmaf_rn(alpha_1, D[3], beta_1 * __half2float(O[3]));
      } else {
        half *D = reinterpret_cast<half *>(&R_D[0][j][0]);
        D[0] = __float2half_rn(__fmaf_rn(alpha_0, __half2float(D[0]),
                                         beta_0 * __half2float(O[0])));
        D[1] = __float2half_rn(__fmaf_rn(alpha_0, __half2float(D[1]),
                                         beta_0 * __half2float(O[1])));
        D[2] = __float2half_rn(__fmaf_rn(alpha_1, __half2float(D[2]),
                                         beta_1 * __half2float(O[2])));
        D[3] = __float2half_rn(__fmaf_rn(alpha_1, __half2float(D[3]),
                                         beta_1 * __half2float(O[3])));
      }
    }

    // Update running l and m
    row_sum_old[0][0] = alpha_0 * l_old_0 + beta_0 * l_tile_0;
    row_sum_old[0][1] = alpha_1 * l_old_1 + beta_1 * l_tile_1;
    row_max_old[0][0] = m_new_0;
    row_max_old[0][1] = m_new_1;

    if constexpr (kPrefetchKV) {
      if (kv_idx + 1 < num_kv) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
      }
    }
  }
  __syncthreads();

  // Final rescale by 1/sum (with safety check for zero sum)
  float inv_sum_0 = (row_sum_old[0][0] > 0.0f) ? __frcp_rn(row_sum_old[0][0]) : 0.0f;
  float inv_sum_1 = (row_sum_old[0][1] > 0.0f) ? __frcp_rn(row_sum_old[0][1]) : 0.0f;
  
  #pragma unroll
  for (int j = 0; j < kWarpTileHeadDimV; ++j) {
    if constexpr (kOStorageAccFloat32) {
      float *D = reinterpret_cast<float *>(&R_D[0][j][0]);
      half *H = reinterpret_cast<half *>(&R_D[0][j][0]);
      H[0] = __float2half_rn(inv_sum_0 * D[0]);
      H[1] = __float2half_rn(inv_sum_0 * D[1]);
      H[2] = __float2half_rn(inv_sum_1 * D[2]);
      H[3] = __float2half_rn(inv_sum_1 * D[3]);
    } else {
      half *D = reinterpret_cast<half *>(&R_D[0][j][0]);
      D[0] = __float2half_rn(inv_sum_0 * __half2float(D[0]));
      D[1] = __float2half_rn(inv_sum_0 * __half2float(D[1]));
      D[2] = __float2half_rn(inv_sum_1 * __half2float(D[2]));
      D[3] = __float2half_rn(inv_sum_1 * __half2float(D[3]));
    }
  }

  // Store O: for simplicity and correctness, each thread writes its own
  // four output elements directly to global memory based on the known
  // MMA m16n8k16 fragment layout.
  //
  // Layout reminder:
  // - Each thread holds 2 rows: row_0 = lane_id % 8, row_1 = row_0 + 8
  // - Each thread holds 2 cols per row:
  //     col_pair = (lane_id / 8) * 2  (0, 2, 4, or 6)
  //     cols are (col_pair, col_pair + 1)
  // - Column groups of 8 come from j * kMmaAtomN (j-th head-dim tile).
  //
  // This is not the most coalesced pattern but is much easier to reason
  // about and is sufficient for a high-performance baseline.
  #pragma unroll
  for (int j = 0; j < kWarpTileHeadDimV; ++j) {
    half *D = reinterpret_cast<half *>(&R_D[0][j][0]);

    int row_in_tile = lane_id % 8;
    int row0 = q_tile_id * Br + warp_QP * kMmaAtomM + row_in_tile;
    int row1 = row0 + 8;

    int col_pair = (lane_id / 8) * 2;  // 0, 2, 4, or 6
    int col0 = j * kMmaAtomN + col_pair;
    int col1 = col0 + 1;

    if (row0 < QKV_seqlen) {
      if (col0 < kHeadDim) {
        int addr = qkv_offset + row0 * kHeadDim + col0;
        O[addr] = D[0];
      }
      if (col1 < kHeadDim) {
        int addr = qkv_offset + row0 * kHeadDim + col1;
        O[addr] = D[1];
      }
    }

    if (row1 < QKV_seqlen) {
      if (col0 < kHeadDim) {
        int addr = qkv_offset + row1 * kHeadDim + col0;
        O[addr] = D[2];
      }
      if (col1 < kHeadDim) {
        int addr = qkv_offset + row1 * kHeadDim + col1;
        O[addr] = D[3];
      }
    }
  }
}

template <const int kHeadDim, const int kStage>
void launch_omni_attn_mma_stages_split_q_shared_kv(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor kv_num_blocks, torch::Tensor kv_indices,
    torch::Tensor block_mask_types) {
  
  constexpr int kMmaAtomM = 16, kMmaAtomN = 8, kMmaAtomK = 16;
  constexpr int kMmaTileSeqLenQ = (kHeadDim < 128) ? 8 : 8;
  constexpr int kMmaTileSeqLenK = 1;
  constexpr int kMmaTileSeqLenP = kMmaTileSeqLenQ;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kWarpTileSeqLenQ = 1;
  // For head_dim <= 64, use kWarpTileSeqLenK = 16 so that:
  //   Br = 128, Bc = 128, matching BLOCK_SIZE used by the Python block mask.
  // For larger head_dim we keep the previous settings (but we currently
  // only enable the MMA path for head_dim <= 64 in the wrapper).
  constexpr int kWarpTileSeqLenK =
      (kHeadDim <= 64 ? 16 : (kHeadDim < 128 ? 8 : 4));
  constexpr int kWarpTileSeqLenP = 1;
  constexpr int kWarpTileHeadDimV = kHeadDim / (kMmaAtomN * kMmaTileHeadDimV);
  constexpr int kPadQ = 8, kPadK = 8, kPadV = 8;
  constexpr int kOStorageAccFloat32 = (kHeadDim < 256) ? 1 : 0;

  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;

  constexpr int Q_size = Br * (kHeadDim + kPadQ);
  constexpr int KV_size = Bc * (kHeadDim + kPadK);
  const int smem_size = (Q_size + kStage * KV_size) * sizeof(half);

  const int batch = Q.size(0), heads = Q.size(1), seqlen = Q.size(2);

  dim3 grid(div_ceil(seqlen, Br), batch * heads);
  dim3 block(kNumThreads);

  cudaFuncSetAttribute(
      omni_attn_mma_stages_split_q_shared_kv_kernel<
          kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK,
          kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV,
          kWarpTileSeqLenQ, kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
          kOStorageAccFloat32, kStage, kPadQ, kPadK, kPadV>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

  omni_attn_mma_stages_split_q_shared_kv_kernel<
      kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK,
      kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV,
      kWarpTileSeqLenQ, kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
      kOStorageAccFloat32, kStage, kPadQ, kPadK, kPadV>
      <<<grid, block, smem_size>>>(
          reinterpret_cast<half *>(Q.data_ptr()),
          reinterpret_cast<half *>(K.data_ptr()),
          reinterpret_cast<half *>(V.data_ptr()),
          reinterpret_cast<half *>(O.data_ptr()),
          reinterpret_cast<int *>(kv_num_blocks.data_ptr()),
          reinterpret_cast<int *>(kv_indices.data_ptr()),
          reinterpret_cast<int *>(block_mask_types.data_ptr()),
          seqlen, heads, batch,
          div_ceil(seqlen, Br), kv_indices.size(3),
          kv_num_blocks.stride(0), kv_num_blocks.stride(1), kv_num_blocks.stride(2),
          kv_indices.stride(0), kv_indices.stride(1), kv_indices.stride(2), kv_indices.stride(3),
          block_mask_types.stride(0), block_mask_types.stride(1), 
          block_mask_types.stride(2), block_mask_types.stride(3));
}

void omni_attn_mma_stages_split_q_shared_kv(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor kv_num_blocks, torch::Tensor kv_indices,
    torch::Tensor block_mask_types, int stages) {
  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf)

  const int batch = Q.size(0);
  const int heads = Q.size(1);
  const int seqlen = Q.size(2);
  const int head_dim = Q.size(3);

  // For now, always use the simple correctness-first kernel. The optimized
  // MMA kernel above is kept for future work, but is not invoked until its
  // correctness is fully validated.
  const int num_q_blocks = kv_num_blocks.size(2);
  const int max_blocks = kv_indices.size(3);

  const int total_q = batch * heads * seqlen;
  const int threads = 128;
  const int blocks = div_ceil(total_q, threads);

  omni_attn_simple_kernel<<<blocks, threads>>>(
      reinterpret_cast<half *>(Q.data_ptr()),
      reinterpret_cast<half *>(K.data_ptr()),
      reinterpret_cast<half *>(V.data_ptr()),
      reinterpret_cast<half *>(O.data_ptr()),
      reinterpret_cast<int *>(kv_num_blocks.data_ptr()),
      reinterpret_cast<int *>(kv_indices.data_ptr()),
      reinterpret_cast<int *>(block_mask_types.data_ptr()),
      seqlen,
      heads,
      batch,
      num_q_blocks,
      max_blocks,
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
      head_dim);
}

// MMA-only entry point for debugging/benchmarking.
// This does not affect the main simple kernel used in tests.
void omni_attn_mma_stages_split_q_shared_kv_mma_only(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor kv_num_blocks, torch::Tensor kv_indices,
    torch::Tensor block_mask_types, int stages) {
  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf)

  const int head_dim = Q.size(3);

#define LAUNCH(D, S)                                                             \
  launch_omni_attn_mma_stages_split_q_shared_kv<D, S>(                          \
      Q, K, V, O, kv_num_blocks, kv_indices, block_mask_types)

  if (stages > 1) {
    switch (head_dim) {
      case 32: LAUNCH(32, 2); break;
      case 64: LAUNCH(64, 2); break;
      case 128: LAUNCH(128, 2); break;
      default:
        throw std::runtime_error("headdim not supported for MMA-only kernel");
    }
  } else {
    switch (head_dim) {
      case 32: LAUNCH(32, 1); break;
      case 64: LAUNCH(64, 1); break;
      case 96: LAUNCH(96, 1); break;
      case 128: LAUNCH(128, 1); break;
      case 256: LAUNCH(256, 1); break;
      default:
        throw std::runtime_error("headdim not supported for MMA-only kernel");
    }
  }

#undef LAUNCH
}
