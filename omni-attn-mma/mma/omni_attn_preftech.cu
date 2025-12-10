#include "utils.h"

// Block mask type definitions
#define BLOCK_MASK_MASKED 0
#define BLOCK_MASK_CAUSAL 1
#define BLOCK_MASK_FULL 2
#define BLOCK_MASK_PARTIAL 3  // Requires per-token dense mask check

// Prefetch strategy with double-buffering:
// - K uses buffer 0, V uses buffer 1 (separate smem regions)
// - Prefetch V while computing Q@K^T
// - Prefetch next K while computing P@V
// - Overlaps memory loads with computation

template <
  const int kHeadDim,          // Headdim, 32,64,128
  const int kMmaAtomM,         // MMA Atom M, 16
  const int kMmaAtomN,         // MMA Atom N, 8
  const int kMmaAtomK,         // MMA Atom K, 16
  const int kMmaTileSeqLenQ,   // 4, more MMA(warp), M=16*4=64
  const int kMmaTileSeqLenK,   // 1, more MMA(warp), N=8*1 =8
  const int kMmaTileSeqLenP,   // 4, more MMA(warp), M=16*4=64
  const int kMmaTileHeadDimV,  // 1, more MMA(warp), N=8*1 =8
  const int kWarpTileSeqLenQ,  // 1, more values, M, Br=64*1=64
  const int kWarpTileSeqLenK,  // 8, more values, N, Bc=8*8 =64
  const int kWarpTileSeqLenP,  // 1, more values, M, Br=64*1=64
  const int kWarpTileHeadDimV, // 8, more values, N
  const int kOStorageAccFloat32, // 0/1, MMA Acc always be fp16, but O storage can be fp32 or half.
  const int kPadQ,               // Pad Q/K/V 0,8
  const int kPadK, const int kPadV>
__global__ void __launch_bounds__(WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK)
omni_attn_prefetch(
    half *Q, half *K, half *V, half *O,
    int Q_BLOCK_SIZE, int KV_BLOCK_SIZE,
    const int *kv_num_blocks, const int *kv_indices, 
    const int *block_mask_types,
    const int *partial_block_mask_indices,
    const bool *partial_block_masks,
    int QKV_seqlen, int QKV_seqlen_orig, int QKV_head, int QKV_batch,
    int kv_num_blocks_stride0, int kv_num_blocks_stride1, int kv_num_blocks_stride2,
    int kv_indices_stride0, int kv_indices_stride1, int kv_indices_stride2, int kv_indices_stride3,
    int block_mask_types_stride0, int block_mask_types_stride1, int block_mask_types_stride2, int block_mask_types_stride3,
    bool has_partial_masks) {
  
  static_assert(kMmaAtomM == 16 && kMmaAtomN == 8 && kMmaAtomK == 16);                                 // m16n8k16
  static_assert(kMmaTileSeqLenQ <= 8 && kMmaTileSeqLenK == 1);    // Q@K^T
  static_assert(kMmaTileSeqLenP <= 8 && kMmaTileHeadDimV == 1);   // P@V
  static_assert(kWarpTileSeqLenQ == 1 && kWarpTileSeqLenK <= 16); // Q@K^T
  static_assert(kWarpTileSeqLenP == 1 && kWarpTileHeadDimV == (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV))); // P@V
  static_assert(kOStorageAccFloat32 == 0 || kOStorageAccFloat32 == 1);
  static_assert(kPadQ >= 0 && kPadQ % 8 == 0); // 0,8,16
  static_assert(kPadK >= 0 && kPadK % 8 == 0); // 0,8,16
  static_assert(kPadV >= 0 && kPadV % 8 == 0); // 0,8,16
  
  constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ; // 16*4*1=64
  constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; //  8*1*8=64
  static_assert(Br >= Bc); // for shared memory reuse.
  constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK; // 32*4*1=128, num threads
  const float scale = 1.0f / sqrt((float)kHeadDim);

  if (Br != Q_BLOCK_SIZE || Bc != KV_BLOCK_SIZE) {
    return;
  }

  // grid(div_ceil(QKV_seqlen, Br), QKV_batch * QKV_head), (x,y,z)
  const int QKV_batch_id = blockIdx.y / QKV_head; // Batch size
  const int QKV_head_id = blockIdx.y % QKV_head;  // Head num
  const int Q_tile_id = blockIdx.x;               // Q tile_id, range [0, Tr]
  const int O_tile_id = Q_tile_id;                // O tile_id, same as Q.
  
  const int q_block = Q_tile_id; // Only valid when Br == Q_BLOCK_SIZE
  const int tid = threadIdx.x;                    // within block
  const int warp_id = tid / WARP_SIZE;            // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE;            // 0~31
  const int warp_QP = warp_id;                    // 0,1,2,3 or 0~7
  const int warp_KV = 0;                          // 0

  const int Q_gmem_offset = ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) + (QKV_head_id * QKV_seqlen * kHeadDim)); // Q [seqlen,d]
  const int K_gmem_offset = Q_gmem_offset; // K [seqlen,d]
  const int V_gmem_offset = Q_gmem_offset; // V [seqlen,d]
  const int O_gmem_offset = Q_gmem_offset; // O [seqlen,d]

  int load_smem_Q_Br = (tid / (kNumThreads / Br)); // Br 64, tid / 2, row 0~64
  int load_smem_Q_d = (tid % (kNumThreads / Br)) * (kHeadDim / (kNumThreads / Br));
  int load_smem_K_Bc = (tid / (kNumThreads / Bc)); // Bc 64, tid / 2, row 0~64
  int load_smem_K_d = (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));
  int load_smem_V_Bc = (tid / (kNumThreads / Bc));
  int load_smem_V_d = (tid % (kNumThreads / Bc)) * (kHeadDim / (kNumThreads / Bc));
  int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;
  if (load_gmem_Q_Br >= QKV_seqlen)
    return;

  // Shared memory for Q,K,V with double-buffering
  // K uses buffer 0, V uses buffer 1 (separate smem regions)
  extern __shared__ half smem[];
  constexpr int Q_tile_size = Br * (kHeadDim + kPadQ); // 64*64=4096
  constexpr int K_tile_size = Bc * (kHeadDim + kPadK); // K[Bc,d]
  half *Q_tile_smem = smem;                           
  half *K_tile_smem = Q_tile_smem + Q_tile_size;       // Base for K buffers
  half *V_tile_smem = K_tile_smem + K_tile_size;       // Base for V buffers

  uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  // Registers/SMEM for thread block
  float lane_block_row_max_old[kWarpTileSeqLenQ][2]; // [1][2]
  float lane_block_row_sum_old[kWarpTileSeqLenQ][2]; // [1][2]
  fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_max_old, -INFINITY);
  fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_sum_old, 0.0f);

  // MMA register arrays
  constexpr int kNumPrefetchQs2r = 1; // Always load Q on-demand
  uint32_t R_Q[kNumPrefetchQs2r][kWarpTileSeqLenQ][4]; // [1][1][4]
  uint32_t R_K[kWarpTileSeqLenK][2];                   // [8][2]
  uint32_t R_V[kWarpTileHeadDimV][2];
  uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][4]; // [1][8][4], acc f32.
  uint32_t R_O[kWarpTileSeqLenP][kWarpTileHeadDimV][4]; // [1][8][4], acc f32.
  uint32_t R_D[kWarpTileSeqLenP][kWarpTileHeadDimV][(kOStorageAccFloat32) ? 4 : 2];
  fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV,((kOStorageAccFloat32) ? 4 : 2)>(R_D, 0);

  // load Q from gmem -> smem, only load once. use CP_ASYNC_CG
  {
    int load_gmem_Q_d = load_smem_Q_d;
    int load_gmem_Q_addr = (Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d);
    uint32_t load_smem_Q_ptr = (smem_Q_base_ptr + (load_smem_Q_Br * (kHeadDim + kPadQ) + load_smem_Q_d) * sizeof(half));
    #pragma unroll
    for (int i = 0; i < (kHeadDim / (kNumThreads / Br)); i += 8) {
      CP_ASYNC_CG(load_smem_Q_ptr + i * 2, &Q[load_gmem_Q_addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();
  }
  CP_ASYNC_WAIT_GROUP(0);
  __syncthreads();

  // get active KV block count
  int kv_nb = kv_num_blocks[QKV_batch_id * kv_num_blocks_stride0 +
                            QKV_head_id * kv_num_blocks_stride1 +
                            q_block * kv_num_blocks_stride2];

  // Main loop over active KV blocks
  #pragma unroll 1
  for (int kv_idx = 0; kv_idx < kv_nb; ++kv_idx) {
    // Get KV block index and mask type
    int kv_block = kv_indices[QKV_batch_id * kv_indices_stride0 +
                              QKV_head_id * kv_indices_stride1 +
                              q_block * kv_indices_stride2 +
                              kv_idx * kv_indices_stride3];
    
    int mask_type = block_mask_types[QKV_batch_id * block_mask_types_stride0 +
                                    QKV_head_id * block_mask_types_stride1 +
                                    q_block * block_mask_types_stride2 +
                                    kv_idx * block_mask_types_stride3];

    // Skip if masked
    if (mask_type == BLOCK_MASK_MASKED) {
      continue;
    }

    // Get partial block index if needed
    int partial_block_index = -1;
    if (mask_type == BLOCK_MASK_PARTIAL) {
      if (!has_partial_masks || partial_block_mask_indices == nullptr) {
        continue;
      }
      partial_block_index = partial_block_mask_indices[QKV_batch_id * block_mask_types_stride0 +
                                                       QKV_head_id * block_mask_types_stride1 +
                                                       q_block * block_mask_types_stride2 +
                                                       kv_idx * block_mask_types_stride3];
      if (partial_block_index < 0) {
        continue;
      }
    }

    // Compute KV block boundaries
    int kv_block_start = kv_block * KV_BLOCK_SIZE;
    int kv_block_end = min(kv_block_start + KV_BLOCK_SIZE, QKV_seqlen_orig);
    
    // Load K tile from gmem -> smem buffer 0
    if (kv_idx == 0) {
      // First iteration: sync load K
      {
        int load_gmem_K_Bc = kv_block_start + load_smem_K_Bc;
        if (load_smem_K_Bc < Bc) {
          int load_gmem_K_d = load_smem_K_d;
          if (load_gmem_K_Bc < kv_block_end) {
            int load_gmem_K_addr = (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
            uint32_t load_smem_K_ptr = (smem_K_base_ptr + (load_smem_K_Bc * (kHeadDim + kPadK) + load_gmem_K_d) * sizeof(half));
            #pragma unroll
            for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
              CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
            }
            CP_ASYNC_COMMIT_GROUP();
          }
        }
      }
      CP_ASYNC_WAIT_GROUP(0);
      __syncthreads();
    } else {
      // Subsequent iterations: wait for K that was prefetched in previous iteration
      CP_ASYNC_WAIT_GROUP(0);
      __syncthreads();
    }
    
    // Prefetch V tile from gmem -> smem buffer 1 (async, before Q@K^T)
    {
      int load_gmem_V_Bc = kv_block_start + load_smem_V_Bc;
      if (load_smem_V_Bc < Bc && load_gmem_V_Bc < kv_block_end) {
        int load_gmem_V_d = load_smem_V_d;
        int load_gmem_V_addr = (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
        uint32_t load_smem_V_ptr = (smem_V_base_ptr + (load_smem_V_Bc * (kHeadDim + kPadV) + load_gmem_V_d) * sizeof(half));
        #pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      }
    }

    // compute S = Q @ K^T with multiple MMA tiles(m16n8k16) using FP32 accumulation
    fill_3D_regs<uint32_t, kWarpTileSeqLenQ, kWarpTileSeqLenK, 4>(R_S, 0);
    
    #pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
      // load Q from smem to registers (m16k16)
      #pragma unroll
      for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
        int warp_smem_Q_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
        int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
        int lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8;
        uint32_t lane_smem_Q_ptr = (smem_Q_base_ptr + (lane_smem_Q_Br * (kHeadDim + kPadQ) + lane_smem_Q_d) * sizeof(half));
        LDMATRIX_X4(R_Q[0][i][0], R_Q[0][i][1], R_Q[0][i][2], R_Q[0][i][3], lane_smem_Q_ptr);
      }

      // load K from smem to registers (k16n8)
      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        int warp_smem_K_Bc = warp_KV * (kMmaAtomN * kWarpTileSeqLenK) + j * kMmaAtomN;
        int lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 8;
        int lane_smem_K_d = tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8;
        uint32_t lane_smem_K_ptr = (smem_K_base_ptr + (lane_smem_K_Bc * (kHeadDim + kPadK) + lane_smem_K_d) * sizeof(half));
        LDMATRIX_X2(R_K[j][0], R_K[j][1], lane_smem_K_ptr);
      }

      // MMA computation: R_S += R_Q @ R_K^T (m16n8k16) with FP32 accumulation
      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        HMMA16816F32(R_S[0][j][0], R_S[0][j][1], R_S[0][j][2], R_S[0][j][3],
                     R_Q[0][0][0], R_Q[0][0][1], R_Q[0][0][2], R_Q[0][0][3],
                     R_K[j][0], R_K[j][1], R_S[0][j][0], R_S[0][j][1],
                     R_S[0][j][2], R_S[0][j][3]);
      }
    }
    __syncthreads();

    // apply scale before mask, scores = (Q @ K^T) / sqrt(d)
    #pragma unroll
    for (int j = 0; j < kWarpTileSeqLenK; ++j) {
      float *S = reinterpret_cast<float *>(&(R_S[0][j][0]));
      S[0] = S[0] * scale;
      S[1] = S[1] * scale;
      S[2] = S[2] * scale;
      S[3] = S[3] * scale;
    }

    // Prefetch next K tile (if not last block)
    if ((kv_idx + 1) < kv_nb) {
      int next_kv_block = kv_indices[QKV_batch_id * kv_indices_stride0 +
                                     QKV_head_id * kv_indices_stride1 +
                                     q_block * kv_indices_stride2 +
                                     (kv_idx + 1) * kv_indices_stride3];
      int next_mask_type = block_mask_types[QKV_batch_id * block_mask_types_stride0 +
                                            QKV_head_id * block_mask_types_stride1 +
                                            q_block * block_mask_types_stride2 +
                                            (kv_idx + 1) * block_mask_types_stride3];
      
      if (next_mask_type != BLOCK_MASK_MASKED) {
        int next_kv_block_start = next_kv_block * KV_BLOCK_SIZE;
        int next_kv_block_end = min(next_kv_block_start + KV_BLOCK_SIZE, QKV_seqlen_orig);
        {
          int load_gmem_K_Bc = next_kv_block_start + load_smem_K_Bc;
          if (load_smem_K_Bc < Bc) {
            int load_gmem_K_d = load_smem_K_d;
            if (load_gmem_K_Bc < next_kv_block_end) {
              int load_gmem_K_addr = (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
              uint32_t load_smem_K_ptr = (smem_K_base_ptr + (load_smem_K_Bc * (kHeadDim + kPadK) + load_gmem_K_d) * sizeof(half));
              #pragma unroll
              for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
                CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
              }
              CP_ASYNC_COMMIT_GROUP();
            }
          }
        }
      }
    }

    // Apply mask in fp32 (AFTER scaling)
    // Use consistent indexing: q_block_start for both causal and partial masks
    int q_block_start = q_block * Q_BLOCK_SIZE;  // Same as Q_tile_id * Br when Br == Q_BLOCK_SIZE
    int kv_base = kv_block * KV_BLOCK_SIZE;
    
    if (mask_type == BLOCK_MASK_CAUSAL) {
      int row_in_warp = lane_id % 16;
      int row_pair = row_in_warp / 2;
      int q_base = Q_tile_id * Br;
      int q_row0 = q_base + warp_QP * kMmaAtomM + row_pair * 2;
      int q_row1 = q_row0 + 1;

      int col_in_tile = lane_id % 8;
      int col_pair = col_in_tile / 2;
      
      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        int kv_col_0 = kv_base + j * kMmaAtomN + col_pair * 2;
        int kv_col_1 = kv_col_0 + 1;
        
        float *S = reinterpret_cast<float *>(&(R_S[0][j][0]));
        if (q_row0 < kv_col_0)     S[0] = -INFINITY;
        if (q_row0 < kv_col_1)     S[1] = -INFINITY;
        if (q_row1 < kv_col_0)     S[2] = -INFINITY;
        if (q_row1 < kv_col_1)     S[3] = -INFINITY;
      }
    }

    // Apply partial mask if needed
    if (mask_type == BLOCK_MASK_PARTIAL && has_partial_masks && partial_block_masks != nullptr && partial_block_index >= 0) {
      int row_in_warp = lane_id % 16;
      int row_in_tile = lane_id % 8;  // 0-7 for rows within the 8-row group
      int col_pair = (lane_id / 8) * 2;  // 0, 2, 4, 6 - first column of the pair
      const int block_area = Q_BLOCK_SIZE * KV_BLOCK_SIZE;
      
      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        int kv_col_0 = kv_base + j * kMmaAtomN + col_pair;
        int kv_col_1 = kv_col_0 + 1;
        
        float *S = reinterpret_cast<float *>(&(R_S[0][j][0]));
        
        if (row_in_warp < 8) {
          int actual_row = q_block_start + warp_QP * kMmaAtomM + row_in_tile;
          
          // Check mask for S[0] (kv_col_0)
          if (actual_row < QKV_seqlen_orig && kv_col_0 < QKV_seqlen_orig) {
            int local_q = actual_row - q_block_start;
            int local_kv = kv_col_0 - kv_base;
            if (local_q >= 0 && local_q < Q_BLOCK_SIZE && local_kv >= 0 && local_kv < KV_BLOCK_SIZE) {
              int mask_offset_0 = partial_block_index * block_area + local_q * KV_BLOCK_SIZE + local_kv;
              if (!partial_block_masks[mask_offset_0]) {
                S[0] = -INFINITY;
              }
            }
          }
          
          // Check mask for S[1] (kv_col_1)
          if (actual_row < QKV_seqlen_orig && kv_col_1 < QKV_seqlen_orig) {
            int local_q = actual_row - q_block_start;
            int local_kv = kv_col_1 - kv_base;
            if (local_q >= 0 && local_q < Q_BLOCK_SIZE && local_kv >= 0 && local_kv < KV_BLOCK_SIZE) {
              int mask_offset_1 = partial_block_index * block_area + local_q * KV_BLOCK_SIZE + local_kv;
              if (!partial_block_masks[mask_offset_1]) {
                S[1] = -INFINITY;
              }
            }
          }
        }
        // S[2] and S[3] are for rows 8-15
        if (row_in_warp >= 8) {
          int actual_row = q_block_start + warp_QP * kMmaAtomM + row_in_tile + 8;
          
          // Check mask for S[2] (kv_col_0)
          if (actual_row < QKV_seqlen_orig && kv_col_0 < QKV_seqlen_orig) {
            int local_q = actual_row - q_block_start;
            int local_kv = kv_col_0 - kv_base;
            if (local_q >= 0 && local_q < Q_BLOCK_SIZE && local_kv >= 0 && local_kv < KV_BLOCK_SIZE) {
              int mask_offset_2 = partial_block_index * block_area + local_q * KV_BLOCK_SIZE + local_kv;
              if (!partial_block_masks[mask_offset_2]) {
                S[2] = -INFINITY;
              }
            }
          }
          
          // Check mask for S[3] (kv_col_1)
          if (actual_row < QKV_seqlen_orig && kv_col_1 < QKV_seqlen_orig) {
            int local_q = actual_row - q_block_start;
            int local_kv = kv_col_1 - kv_base;
            if (local_q >= 0 && local_q < Q_BLOCK_SIZE && local_kv >= 0 && local_kv < KV_BLOCK_SIZE) {
              int mask_offset_3 = partial_block_index * block_area + local_q * KV_BLOCK_SIZE + local_kv;
              if (!partial_block_masks[mask_offset_3]) {
                S[3] = -INFINITY;
              }
            }
          }
        }
      }
    }

    // Wait for V to be ready (prefetched in buffer 1) - FIX: use group 0, not 1
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();

    // compute P = softmax(S)
    // compute row max and row sum (new and global)
    float lane_row_max_new[kWarpTileSeqLenQ][2];
    float lane_row_sum_new[kWarpTileSeqLenQ][2];
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_max_new, -INFINITY);
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_sum_new, 0.0f);

    static_assert(kWarpTileSeqLenQ == 1);
    // Row max reduction - properly exclude masked values (-INFINITY)
    {
      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        float *S = reinterpret_cast<float *>(&(R_S[0][j][0]));
        float tmp_max_0 = max(S[0], S[1]);
        float tmp_max_1 = max(S[2], S[3]);
        lane_row_max_new[0][0] = max(lane_row_max_new[0][0], tmp_max_0);
        lane_row_max_new[0][1] = max(lane_row_max_new[0][1], tmp_max_1);
      }
      // Warp level reduce max, warp_size = 4
      lane_row_max_new[0][0] = warp_reduce_max<float, 4>(lane_row_max_new[0][0]);
      lane_row_max_new[0][1] = warp_reduce_max<float, 4>(lane_row_max_new[0][1]);
    }

    // Compute exp and row sum
    {
      float block_row_max_new_0 = lane_row_max_new[0][0];
      float block_row_max_new_1 = lane_row_max_new[0][1];
      float block_row_max_old_0 = lane_block_row_max_old[0][0];
      float block_row_max_old_1 = lane_block_row_max_old[0][1];
      
      const float MASKED_THRESHOLD = -1e8f;
      
      if (block_row_max_new_0 > MASKED_THRESHOLD) {
        if (block_row_max_old_0 > MASKED_THRESHOLD) {
          block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
        }
      } else {
        block_row_max_new_0 = (block_row_max_old_0 > MASKED_THRESHOLD) ? block_row_max_old_0 : block_row_max_new_0;
      }
      
      if (block_row_max_new_1 > MASKED_THRESHOLD) {
        if (block_row_max_old_1 > MASKED_THRESHOLD) {
          block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);
        }
      } else {
        block_row_max_new_1 = (block_row_max_old_1 > MASKED_THRESHOLD) ? block_row_max_old_1 : block_row_max_new_1;
      }

      #pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        float *S = reinterpret_cast<float *>(&(R_S[0][j][0]));
        half *h_S = reinterpret_cast<half *>(&(R_S[0][j][0]));
        // P = Exp(S - m_new), where S is already scaled logits
        S[0] = __expf(S[0] - block_row_max_new_0);
        S[1] = __expf(S[1] - block_row_max_new_0);
        S[2] = __expf(S[2] - block_row_max_new_1);
        S[3] = __expf(S[3] - block_row_max_new_1);
        lane_row_sum_new[0][0] += (S[0] + S[1]);
        lane_row_sum_new[0][1] += (S[2] + S[3]);
        // Update R_S for P[Br,Bc] = Exp(S-m), point wise.
        // Also convert F32 -> half for P@V MMA, reuse R_S as P.
        h_S[0] = __float2half_rn(S[0]);
        h_S[1] = __float2half_rn(S[1]);
        h_S[2] = __float2half_rn(S[2]);
        h_S[3] = __float2half_rn(S[3]);
      }
      lane_row_sum_new[0][0] = warp_reduce_sum<float, 4>(lane_row_sum_new[0][0]);
      lane_row_sum_new[0][1] = warp_reduce_sum<float, 4>(lane_row_sum_new[0][1]);
      
      // Store the global max for use in rescaling section
      lane_row_max_new[0][0] = block_row_max_new_0;
      lane_row_max_new[0][1] = block_row_max_new_1;
    }

    // compute O = P @ V with multiple MMA tiles(m16n8k16)
    fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV, 4>(R_O, 0);
    #pragma unroll
    for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {
      // load V from smem to registers (k16n8, transposed)
      #pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) {
        int warp_smem_V_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
        int lane_smem_V_Bc = tile_V_Bc * kMmaAtomK + lane_id % 16;
        int lane_smem_V_d = warp_smem_V_d;
        uint32_t lane_smem_V_ptr = (smem_V_base_ptr + (lane_smem_V_Bc * (kHeadDim + kPadV) + lane_smem_V_d) * sizeof(half));
        LDMATRIX_X2_T(R_V[j][0], R_V[j][1], lane_smem_V_ptr);
      }

      int w = tile_V_Bc * 2;
      #pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) {
        // MMA always accumulate with F32 dtype for high precision.
        HMMA16816F32(R_O[0][j][0], R_O[0][j][1], R_O[0][j][2], R_O[0][j][3],
                     R_S[0][w][0], R_S[0][w][1], R_S[0][w + 1][0],
                     R_S[0][w + 1][1], R_V[j][0], R_V[j][1], R_O[0][j][0],
                     R_O[0][j][1], R_O[0][j][2], R_O[0][j][3]);
      }
    }
    __syncthreads();

    // rescale O with online softmax correction.
    static_assert(kWarpTileSeqLenP == 1);
    {
      float block_row_max_new_0 = lane_row_max_new[0][0];
      float block_row_max_new_1 = lane_row_max_new[0][1];
      float block_row_sum_new_0 = lane_row_sum_new[0][0];
      float block_row_sum_new_1 = lane_row_sum_new[0][1];

      float block_row_max_old_0 = lane_block_row_max_old[0][0];
      float block_row_max_old_1 = lane_block_row_max_old[0][1];
      const float MASKED_THRESHOLD = -1e9f;
      
      // Avoid inf value while using m_old for rescaling O.
      block_row_max_old_0 = (kv_idx > 0 ? block_row_max_old_0 : block_row_max_new_0);
      block_row_max_old_1 = (kv_idx > 0 ? block_row_max_old_1 : block_row_max_new_1);

      // rescale factor for O and l, exp(m_old - m_global)
      float rescale_o_factor_0 = 1.0f;
      float rescale_o_factor_1 = 1.0f;
      
      if (block_row_max_old_0 > MASKED_THRESHOLD && block_row_max_new_0 > MASKED_THRESHOLD) {
        float max_diff_0 = block_row_max_old_0 - block_row_max_new_0;
        const float MIN_EXP_DIFF = 1e-5f;
        if (max_diff_0 < -MIN_EXP_DIFF) {  // m_new > m_old (max increased)
          max_diff_0 = fmaxf(-88.0f, fminf(88.0f, max_diff_0));
          rescale_o_factor_0 = __expf(max_diff_0);
        }
      }
      
      if (block_row_max_old_1 > MASKED_THRESHOLD && block_row_max_new_1 > MASKED_THRESHOLD) {
        float max_diff_1 = block_row_max_old_1 - block_row_max_new_1;
        const float MIN_EXP_DIFF = 1e-5f;
        if (max_diff_1 < -MIN_EXP_DIFF) {  // m_new > m_old
          max_diff_1 = fmaxf(-88.0f, fminf(88.0f, max_diff_1));
          rescale_o_factor_1 = __expf(max_diff_1);
        }
      }

      #pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) {
        float *O = reinterpret_cast<float *>(&(R_O[0][j][0]));
        if constexpr (kOStorageAccFloat32) {
          float *D = reinterpret_cast<float *>(&(R_D[0][j][0]));
          D[0] = __fmaf_rn(rescale_o_factor_0, D[0], O[0]);
          D[1] = __fmaf_rn(rescale_o_factor_0, D[1], O[1]);
          D[2] = __fmaf_rn(rescale_o_factor_1, D[2], O[2]);
          D[3] = __fmaf_rn(rescale_o_factor_1, D[3], O[3]);
        } else {
          half *D_half = reinterpret_cast<half *>(&(R_D[0][j][0]));
          D_half[0] = __float2half_rn(__fmaf_rn(rescale_o_factor_0, __half2float(D_half[0]), O[0]));
          D_half[1] = __float2half_rn(__fmaf_rn(rescale_o_factor_0, __half2float(D_half[1]), O[1]));
          D_half[2] = __float2half_rn(__fmaf_rn(rescale_o_factor_1, __half2float(D_half[2]), O[2]));
          D_half[3] = __float2half_rn(__fmaf_rn(rescale_o_factor_1, __half2float(D_half[3]), O[3]));
        }
      }

      // update statistics
      float block_row_sum_old_0 = lane_block_row_sum_old[0][0];
      float block_row_sum_old_1 = lane_block_row_sum_old[0][1];
      lane_block_row_sum_old[0][0] = __fmaf_rn(rescale_o_factor_0, block_row_sum_old_0, block_row_sum_new_0);
      lane_block_row_sum_old[0][1] = __fmaf_rn(rescale_o_factor_1, block_row_sum_old_1, block_row_sum_new_1);
      lane_block_row_max_old[0][0] = block_row_max_new_0;
      lane_block_row_max_old[0][1] = block_row_max_new_1;
    }

    // Wait for next K to be ready (if prefetched)
    if ((kv_idx + 1) < kv_nb) {
      CP_ASYNC_WAIT_GROUP(0);
      __syncthreads();
    }
  } // end loop over active KV blocks
  __syncthreads();

  // Final rescale and write O to gmem
  static_assert(kWarpTileSeqLenP == 1);
  {
    // Avoid division by zero: if row_sum is 0 or very small, output should be zeros (all masked)
    constexpr float kMinRowSum = 1e-6f;
    bool has_valid_sum_0 = (lane_block_row_sum_old[0][0] > kMinRowSum);
    bool has_valid_sum_1 = (lane_block_row_sum_old[0][1] > kMinRowSum);
    float rescale_factor_0 = has_valid_sum_0 ? __frcp_rn(lane_block_row_sum_old[0][0]) : 0.0f;
    float rescale_factor_1 = has_valid_sum_1 ? __frcp_rn(lane_block_row_sum_old[0][1]) : 0.0f;

    #pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) {
      if constexpr (kOStorageAccFloat32) {
        float *D = reinterpret_cast<float *>(&(R_D[0][j][0]));
        half *D_half = reinterpret_cast<half *>(&(R_D[0][j][0]));
        D_half[0] = __float2half_rn(rescale_factor_0 * D[0]);
        D_half[1] = __float2half_rn(rescale_factor_0 * D[1]);
        D_half[2] = __float2half_rn(rescale_factor_1 * D[2]);
        D_half[3] = __float2half_rn(rescale_factor_1 * D[3]);
      } else {
        half *D_half = reinterpret_cast<half *>(&(R_D[0][j][0]));
        D_half[0] = __float2half_rn(rescale_factor_0 * __half2float(D_half[0]));
        D_half[1] = __float2half_rn(rescale_factor_0 * __half2float(D_half[1]));
        D_half[2] = __float2half_rn(rescale_factor_1 * __half2float(D_half[2]));
        D_half[3] = __float2half_rn(rescale_factor_1 * __half2float(D_half[3]));
      }
    }
  }

  // Store O(D): Write O[Br,d] from regs -> gmem, collective store
  static_assert(kWarpTileSeqLenP == 1);
  { // kWarpTileSeqLenP = 1
    #pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) {
      uint32_t R_Z[2][4];
      R_Z[0][0] = R_D[0][j][0];
      R_Z[1][0] = R_D[0][j][1];
      R_Z[0][1] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 1, 4);
      R_Z[0][2] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 2, 4);
      R_Z[0][3] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 3, 4);
      R_Z[1][1] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 1, 4);
      R_Z[1][2] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 2, 4);
      R_Z[1][3] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 3, 4);
      if (lane_id % 4 == 0) {
        int store_warp_regs_O_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenP) + 0 * kMmaAtomM;
        int store_lane_gmem_O_Br = O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4;
        int store_warp_regs_O_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
        int store_lane_gmem_O_d = store_warp_regs_O_d;
        int store_gmem_O_addr_0 = (O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim + store_lane_gmem_O_d);
        int store_gmem_O_addr_1 = (O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim + store_lane_gmem_O_d);
        LDST128BITS(O[store_gmem_O_addr_0]) = LDST128BITS(R_Z[0][0]);
        LDST128BITS(O[store_gmem_O_addr_1]) = LDST128BITS(R_Z[1][0]);
      }
    }
  }
}

// ============================================================================
// Wrapper function to launch the template kernel
// ============================================================================

void omni_attn_preftech_kernel(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor kv_num_blocks, torch::Tensor kv_indices,
    torch::Tensor block_mask_types,
    int Q_BLOCK_SIZE, int KV_BLOCK_SIZE,
    int seqlen_orig,
    torch::Tensor partial_block_mask_indices, torch::Tensor partial_block_masks,
    bool has_partial) {
  
  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf)
  
  const int batch = Q.size(0);
  const int heads = Q.size(1);
  const int seqlen = Q.size(2);
  const int head_dim = Q.size(3);
  
  const int num_q_blocks = kv_num_blocks.size(2);
  const int expected_num_q_blocks = (seqlen + Q_BLOCK_SIZE - 1) / Q_BLOCK_SIZE;
  if (num_q_blocks != expected_num_q_blocks) {
    throw std::runtime_error(
        "Block mask num_q_blocks (" + std::to_string(num_q_blocks) + 
        ") does not match expected value (" + std::to_string(expected_num_q_blocks) + 
        ") based on seqlen=" + std::to_string(seqlen) + " and Q_BLOCK_SIZE=" + 
        std::to_string(Q_BLOCK_SIZE));
  }
  
  if (head_dim != 32 && head_dim != 64 && head_dim != 128) {
    throw std::runtime_error(
        "Unsupported head_dim=" + std::to_string(head_dim) +
        ". Supported values: 32, 64, 128");
  }
  
  if (Q_BLOCK_SIZE != 64 && Q_BLOCK_SIZE != 128) {
    throw std::runtime_error(
        "Unsupported Q_BLOCK_SIZE=" + std::to_string(Q_BLOCK_SIZE) +
        ". Supported values: 64, 128");
  }
  if (KV_BLOCK_SIZE != 64 && KV_BLOCK_SIZE != 128) {
    throw std::runtime_error(
        "Unsupported KV_BLOCK_SIZE=" + std::to_string(KV_BLOCK_SIZE) +
        ". Supported values: 64, 128");
  }
  
  const bool has_partial_masks = has_partial;
  if (has_partial_masks) {
    if (!partial_block_mask_indices.is_contiguous() ||
        !partial_block_masks.is_contiguous()) {
      throw std::runtime_error(
          "Partial block mask tensors must be contiguous.");
    }
    CHECK_TORCH_TENSOR_DTYPE(partial_block_mask_indices, torch::kInt32)
    CHECK_TORCH_TENSOR_DTYPE(partial_block_masks, torch::kBool)
  }
  
  const int *partial_idx_ptr = has_partial_masks
                                   ? reinterpret_cast<int *>(
                                         partial_block_mask_indices.data_ptr())
                                   : nullptr;
  const bool *partial_masks_ptr = has_partial_masks
                                      ? reinterpret_cast<bool *>(
                                            partial_block_masks.data_ptr())
                                      : nullptr;
  
  // Select template parameters based on BLOCK_SIZE
  constexpr int kMmaAtomM = 16;
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaAtomK = 16;
  constexpr int kMmaTileSeqLenK = 1;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kWarpTileSeqLenQ = 1;
  constexpr int kWarpTileSeqLenP = 1;
  constexpr int kPadQ = 0;
  constexpr int kPadK = 0;
  constexpr int kPadV = 0;
  constexpr int kOStorageAccFloat32 = 1;  // Use FP32 accumulation for better precision
  
  // Launch kernel based on head_dim and BLOCK_SIZE
  if (Q_BLOCK_SIZE == 64 && KV_BLOCK_SIZE == 64) {
    constexpr int kMmaTileSeqLenQ = 4;
    constexpr int kMmaTileSeqLenP = 4;
    constexpr int kWarpTileSeqLenK = 8;
    constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ; // 64
    constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; // 64
    constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK; // 128
    
    dim3 grid(div_ceil(seqlen, Br), batch * heads);
    dim3 block(kNumThreads);
    
    if (head_dim == 32) {
      constexpr int kHeadDim = 32;
      constexpr int kWarpTileHeadDimV = 4;
      constexpr int Q_tile_size = Br * (kHeadDim + kPadQ);
      constexpr int K_tile_size = Bc * (kHeadDim + kPadK);
      constexpr int V_tile_size = Bc * (kHeadDim + kPadV);
      // Double-buffering: need separate K and V buffers
      constexpr int smem_size = (Q_tile_size + K_tile_size + V_tile_size) * sizeof(half);
      set_kernel_max_dynamic_smem(reinterpret_cast<const void *>(
                                      omni_attn_prefetch<kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK,
                                                          kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV,
                                                          kWarpTileSeqLenQ, kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
                                                          kOStorageAccFloat32, kPadQ, kPadK, kPadV>),
                                  smem_size);
      omni_attn_prefetch<kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK,
          kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV,
          kWarpTileSeqLenQ, kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
          kOStorageAccFloat32, kPadQ, kPadK, kPadV>
          <<<grid, block, smem_size>>>(reinterpret_cast<half *>(Q.data_ptr()),
              reinterpret_cast<half *>(K.data_ptr()), reinterpret_cast<half *>(V.data_ptr()),
              reinterpret_cast<half *>(O.data_ptr()), Q_BLOCK_SIZE, KV_BLOCK_SIZE,
              reinterpret_cast<int *>(kv_num_blocks.data_ptr()),
              reinterpret_cast<int *>(kv_indices.data_ptr()),
              reinterpret_cast<int *>(block_mask_types.data_ptr()),
              partial_idx_ptr, partial_masks_ptr,
              seqlen, seqlen_orig, heads, batch,
              kv_num_blocks.stride(0), kv_num_blocks.stride(1), kv_num_blocks.stride(2),
              kv_indices.stride(0), kv_indices.stride(1), kv_indices.stride(2), kv_indices.stride(3),
              block_mask_types.stride(0), block_mask_types.stride(1),
              block_mask_types.stride(2), block_mask_types.stride(3),
              has_partial_masks);
    } else if (head_dim == 64) {
      constexpr int kHeadDim = 64;
      constexpr int kWarpTileHeadDimV = 8;
      constexpr int Q_tile_size = Br * (kHeadDim + kPadQ);
      constexpr int K_tile_size = Bc * (kHeadDim + kPadK);
      constexpr int V_tile_size = Bc * (kHeadDim + kPadV);
      constexpr int smem_size = (Q_tile_size + K_tile_size + V_tile_size) * sizeof(half);
      set_kernel_max_dynamic_smem(reinterpret_cast<const void *>(
                                      omni_attn_prefetch<kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK,
                                                          kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV,
                                                          kWarpTileSeqLenQ, kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
                                                          kOStorageAccFloat32, kPadQ, kPadK, kPadV>),
                                  smem_size);
      omni_attn_prefetch<kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK,
          kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV,
          kWarpTileSeqLenQ, kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
          kOStorageAccFloat32, kPadQ, kPadK, kPadV>
          <<<grid, block, smem_size>>>(reinterpret_cast<half *>(Q.data_ptr()),
              reinterpret_cast<half *>(K.data_ptr()), reinterpret_cast<half *>(V.data_ptr()),
              reinterpret_cast<half *>(O.data_ptr()), Q_BLOCK_SIZE, KV_BLOCK_SIZE,
              reinterpret_cast<int *>(kv_num_blocks.data_ptr()),
              reinterpret_cast<int *>(kv_indices.data_ptr()),
              reinterpret_cast<int *>(block_mask_types.data_ptr()),
              partial_idx_ptr, partial_masks_ptr,
              seqlen, seqlen_orig, heads, batch,
              kv_num_blocks.stride(0), kv_num_blocks.stride(1), kv_num_blocks.stride(2),
              kv_indices.stride(0), kv_indices.stride(1), kv_indices.stride(2), kv_indices.stride(3),
              block_mask_types.stride(0), block_mask_types.stride(1),
              block_mask_types.stride(2), block_mask_types.stride(3),
              has_partial_masks);
    } else if (head_dim == 128) {
      constexpr int kHeadDim = 128;
      constexpr int kWarpTileHeadDimV = 16;
      constexpr int Q_tile_size = Br * (kHeadDim + kPadQ);
      constexpr int K_tile_size = Bc * (kHeadDim + kPadK);
      constexpr int V_tile_size = Bc * (kHeadDim + kPadV);
      constexpr int smem_size = (Q_tile_size + K_tile_size + V_tile_size) * sizeof(half);
      set_kernel_max_dynamic_smem(reinterpret_cast<const void *>(
                                      omni_attn_prefetch<kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK,
                                                          kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV,
                                                          kWarpTileSeqLenQ, kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
                                                          kOStorageAccFloat32, kPadQ, kPadK, kPadV>),
                                  smem_size);
      omni_attn_prefetch<kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK,
          kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV,
          kWarpTileSeqLenQ, kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
          kOStorageAccFloat32, kPadQ, kPadK, kPadV>
          <<<grid, block, smem_size>>>(reinterpret_cast<half *>(Q.data_ptr()),
              reinterpret_cast<half *>(K.data_ptr()), reinterpret_cast<half *>(V.data_ptr()),
              reinterpret_cast<half *>(O.data_ptr()), Q_BLOCK_SIZE, KV_BLOCK_SIZE,
              reinterpret_cast<int *>(kv_num_blocks.data_ptr()),
              reinterpret_cast<int *>(kv_indices.data_ptr()),
              reinterpret_cast<int *>(block_mask_types.data_ptr()),
              partial_idx_ptr, partial_masks_ptr,
              seqlen, seqlen_orig, heads, batch,
              kv_num_blocks.stride(0), kv_num_blocks.stride(1), kv_num_blocks.stride(2),
              kv_indices.stride(0), kv_indices.stride(1), kv_indices.stride(2), kv_indices.stride(3),
              block_mask_types.stride(0), block_mask_types.stride(1),
              block_mask_types.stride(2), block_mask_types.stride(3),
              has_partial_masks);
    } else {
      throw std::runtime_error(
          "Unsupported configuration for BLOCK_SIZE=64: head_dim=" + std::to_string(head_dim));
    }
  } else if (Q_BLOCK_SIZE == 128 && KV_BLOCK_SIZE == 128) {
    constexpr int kMmaTileSeqLenQ = 8;
    constexpr int kMmaTileSeqLenP = 8;
    constexpr int kWarpTileSeqLenK = 16;
    constexpr int Br = kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ; // 128
    constexpr int Bc = kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; // 128
    constexpr int kNumThreads = WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK; // 256
    
    dim3 grid(div_ceil(seqlen, Br), batch * heads);
    dim3 block(kNumThreads);
    
    if (head_dim == 32) {
      constexpr int kHeadDim = 32;
      constexpr int kWarpTileHeadDimV = 4;
      constexpr int Q_tile_size = Br * (kHeadDim + kPadQ);
      constexpr int K_tile_size = Bc * (kHeadDim + kPadK);
      constexpr int V_tile_size = Bc * (kHeadDim + kPadV);
      constexpr int smem_size = (Q_tile_size + K_tile_size + V_tile_size) * sizeof(half);
      set_kernel_max_dynamic_smem(reinterpret_cast<const void *>(
                                      omni_attn_prefetch<kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK,
                                                          kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV,
                                                          kWarpTileSeqLenQ, kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
                                                          kOStorageAccFloat32, kPadQ, kPadK, kPadV>),
                                  smem_size);
      omni_attn_prefetch<kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK,
          kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV,
          kWarpTileSeqLenQ, kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
          kOStorageAccFloat32, kPadQ, kPadK, kPadV>
          <<<grid, block, smem_size>>>(reinterpret_cast<half *>(Q.data_ptr()),
              reinterpret_cast<half *>(K.data_ptr()), reinterpret_cast<half *>(V.data_ptr()),
              reinterpret_cast<half *>(O.data_ptr()), Q_BLOCK_SIZE, KV_BLOCK_SIZE,
              reinterpret_cast<int *>(kv_num_blocks.data_ptr()),
              reinterpret_cast<int *>(kv_indices.data_ptr()),
              reinterpret_cast<int *>(block_mask_types.data_ptr()),
              partial_idx_ptr, partial_masks_ptr,
              seqlen, seqlen_orig, heads, batch,
              kv_num_blocks.stride(0), kv_num_blocks.stride(1), kv_num_blocks.stride(2),
              kv_indices.stride(0), kv_indices.stride(1), kv_indices.stride(2), kv_indices.stride(3),
              block_mask_types.stride(0), block_mask_types.stride(1),
              block_mask_types.stride(2), block_mask_types.stride(3),
              has_partial_masks);
    } else if (head_dim == 64) {
      constexpr int kHeadDim = 64;
      constexpr int kWarpTileHeadDimV = 8;
      constexpr int Q_tile_size = Br * (kHeadDim + kPadQ);
      constexpr int K_tile_size = Bc * (kHeadDim + kPadK);
      constexpr int V_tile_size = Bc * (kHeadDim + kPadV);
      constexpr int smem_size = (Q_tile_size + K_tile_size + V_tile_size) * sizeof(half);
      set_kernel_max_dynamic_smem(reinterpret_cast<const void *>(
                                      omni_attn_prefetch<kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK,
                                                          kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV,
                                                          kWarpTileSeqLenQ, kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
                                                          kOStorageAccFloat32, kPadQ, kPadK, kPadV>),
                                  smem_size);
      omni_attn_prefetch<kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK,
          kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV,
          kWarpTileSeqLenQ, kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
          kOStorageAccFloat32, kPadQ, kPadK, kPadV>
          <<<grid, block, smem_size>>>(reinterpret_cast<half *>(Q.data_ptr()),
              reinterpret_cast<half *>(K.data_ptr()), reinterpret_cast<half *>(V.data_ptr()),
              reinterpret_cast<half *>(O.data_ptr()), Q_BLOCK_SIZE, KV_BLOCK_SIZE,
              reinterpret_cast<int *>(kv_num_blocks.data_ptr()),
              reinterpret_cast<int *>(kv_indices.data_ptr()),
              reinterpret_cast<int *>(block_mask_types.data_ptr()),
              partial_idx_ptr, partial_masks_ptr,
              seqlen, seqlen_orig, heads, batch,
              kv_num_blocks.stride(0), kv_num_blocks.stride(1), kv_num_blocks.stride(2),
              kv_indices.stride(0), kv_indices.stride(1), kv_indices.stride(2), kv_indices.stride(3),
              block_mask_types.stride(0), block_mask_types.stride(1),
              block_mask_types.stride(2), block_mask_types.stride(3),
              has_partial_masks);
    } else if (head_dim == 128) {
      constexpr int kHeadDim = 128;
      constexpr int kWarpTileHeadDimV = 16;
      constexpr int Q_tile_size = Br * (kHeadDim + kPadQ);
      constexpr int K_tile_size = Bc * (kHeadDim + kPadK);
      constexpr int V_tile_size = Bc * (kHeadDim + kPadV);
      constexpr int smem_size = (Q_tile_size + K_tile_size + V_tile_size) * sizeof(half);
      set_kernel_max_dynamic_smem(reinterpret_cast<const void *>(
                                      omni_attn_prefetch<kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK,
                                                          kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV,
                                                          kWarpTileSeqLenQ, kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
                                                          kOStorageAccFloat32, kPadQ, kPadK, kPadV>),
                                  smem_size);
      omni_attn_prefetch<kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK,
          kMmaTileSeqLenQ, kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV,
          kWarpTileSeqLenQ, kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
          kOStorageAccFloat32, kPadQ, kPadK, kPadV>
          <<<grid, block, smem_size>>>(reinterpret_cast<half *>(Q.data_ptr()),
              reinterpret_cast<half *>(K.data_ptr()), reinterpret_cast<half *>(V.data_ptr()),
              reinterpret_cast<half *>(O.data_ptr()), Q_BLOCK_SIZE, KV_BLOCK_SIZE,
              reinterpret_cast<int *>(kv_num_blocks.data_ptr()),
              reinterpret_cast<int *>(kv_indices.data_ptr()),
              reinterpret_cast<int *>(block_mask_types.data_ptr()),
              partial_idx_ptr, partial_masks_ptr,
              seqlen, seqlen_orig, heads, batch,
              kv_num_blocks.stride(0), kv_num_blocks.stride(1), kv_num_blocks.stride(2),
              kv_indices.stride(0), kv_indices.stride(1), kv_indices.stride(2), kv_indices.stride(3),
              block_mask_types.stride(0), block_mask_types.stride(1),
              block_mask_types.stride(2), block_mask_types.stride(3),
              has_partial_masks);
    } else {
      throw std::runtime_error(
          "Unsupported configuration for BLOCK_SIZE=128: head_dim=" + std::to_string(head_dim));
    }
  } else {
    throw std::runtime_error(
        "Unsupported BLOCK_SIZE combination: Q_BLOCK_SIZE=" + std::to_string(Q_BLOCK_SIZE) +
        ", KV_BLOCK_SIZE=" + std::to_string(KV_BLOCK_SIZE) +
        ". Supported: (64,64) or (128,128)");
  }
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
  }
}
