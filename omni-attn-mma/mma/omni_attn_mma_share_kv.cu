#include "utils.h"

// Omni-Attention MMA kernel with sparse block support
// Based on flash_attn_mma_stages_split_q_shared_kv, but supports:
// - Block-sparse attention via kv_indices and block_mask_types
// - Per-block mask types: MASKED (skip), CAUSAL (causal mask), FULL (no mask)
// - Only processes active KV blocks specified by kv_indices

// Block mask types
#define BLOCK_MASK_MASKED 0  // Skip block entirely
#define BLOCK_MASK_CAUSAL 1  // Apply causal masking
#define BLOCK_MASK_FULL 2    // No masking

template <
    const int kHeadDim,          // Headdim, 32,64,128
    const int kMmaAtomM,         // MMA Atom M, 16
    const int kMmaAtomN,         // MMA Atom N, 8
    const int kMmaAtomK,         // MMA Atom K, 16
    const int kMmaTileSeqLenQ,   // 4, more MMA(warp), M=16*4=64, Q@K^T=[Br(M),
                                 // d(K)]@[d(K),  Bc(N)]
    const int kMmaTileSeqLenK,   // 1, more MMA(warp), N=8*1 =8,  Q@K^T=[Br(M),
                                 // d(K)]@[d(K),  Bc(N)]
    const int kMmaTileSeqLenP,   // 4, more MMA(warp), M=16*4=64, P@V
                                 // =[Br(M),Bc(K)]@[Bc(K), d(N) ]
    const int kMmaTileHeadDimV,  // 1, more MMA(warp), N=8*1 =8,  P@V
                                 // =[Br(M),Bc(K)]@[Bc(K), d(N) ]
    const int kWarpTileSeqLenQ,  // 1, more values, M, Br=64*1=64, matmul M
    const int kWarpTileSeqLenK,  // 8, more values, N, Bc=8*8 =64, matmul N
    const int kWarpTileSeqLenP,  // 1, more values, M, Br=64*1=64, matmul M
    const int kWarpTileHeadDimV, // 8, more values, N,
                                 // d=8*(1|2|3|4|...)=8|...|32|64|96|128|...
    const int kOStorageAccFloat32, // 0/1, MMA Acc always be fp16, but O
                                   // storage can be fp32 or half.
    const int kStage,              // 1,2
    const int kPadQ,               // Pad Q/K/V 0,8
    const int kPadK, const int kPadV>
__global__ void __launch_bounds__(WARP_SIZE *kMmaTileSeqLenQ *kMmaTileSeqLenK)
    omni_attn_mma_stages_split_q_shared_kv_kernel(
        half *Q, half *K, half *V,
        half *O, 
        int *kv_num_blocks,      // [batch, nheads, num_q_blocks]
        int *kv_indices,         // [batch, nheads, num_q_blocks, max_blocks]
        int *block_mask_types,   // [batch, nheads, num_q_blocks, max_blocks]
        int QKV_seqlen, int QKV_head, int QKV_batch,
        int num_q_blocks, int max_blocks,
        int kv_num_blocks_stride0, int kv_num_blocks_stride1, int kv_num_blocks_stride2,
        int kv_indices_stride0, int kv_indices_stride1, int kv_indices_stride2, int kv_indices_stride3,
        int block_mask_types_stride0, int block_mask_types_stride1, int block_mask_types_stride2, int block_mask_types_stride3) {
  // Matmul Layout: Q[Br,d]@K^T[d,Bc] NT, P[Br,Bc]@V[Bc,d] NN.
  // NOTE: K[Bc,d] with row major means K^T[d,Bc] in col major.
  static_assert(kMmaAtomM == 16 && kMmaAtomN == 8 &&
                kMmaAtomK == 16);                                 // m16n8k16
  static_assert(kMmaTileSeqLenQ <= 8 && kMmaTileSeqLenK == 1);    // Q@K^T
  static_assert(kMmaTileSeqLenP <= 8 && kMmaTileHeadDimV == 1);   // P@V
  static_assert(kWarpTileSeqLenQ == 1 && kWarpTileSeqLenK <= 16); // Q@K^T
  // kWarpTileHeadDimV: d=8*(1|2|3|4|...) = 8|...|32|64|96|128|..., etc.
  // e.g, kWarpTileHeadDimV = 8 -> d = 8*8 = 64; 16 -> d = 8*16 = 128.
  static_assert(kWarpTileSeqLenP == 1 &&
                kWarpTileHeadDimV ==
                    (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV))); // P@V
  static_assert(kOStorageAccFloat32 == 0 || kOStorageAccFloat32 == 1);
  static_assert(kStage < 3 && kStage > 0);
  static_assert(kPadQ >= 0 && kPadQ % 8 == 0); // 0,8,16
  static_assert(kPadK >= 0 && kPadK % 8 == 0); // 0,8,16
  static_assert(kPadV >= 0 && kPadV % 8 == 0); // 0,8,16
  constexpr int Br =
      kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ; // 16*4*1=64
  constexpr int Bc =
      kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK; //  8*1*8=64
  static_assert(Br >= Bc); // for shared memory reuse.
  constexpr int kNumThreads =
      WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK; // 32*4*1=128, num threads
  const float scale = 1.0f / sqrt((float)kHeadDim);

  // grid(div_ceil(QKV_seqlen, Br), QKV_batch * QKV_head), (x,y,z)
  const int QKV_batch_id = blockIdx.y / QKV_head; // Batch size
  const int QKV_head_id = blockIdx.y % QKV_head;  // Head num
  const int Q_tile_id = blockIdx.x;               // Q tile_id, range [0, num_q_blocks)
  const int O_tile_id = Q_tile_id;                // O tile_id, same as Q.
  const int tid = threadIdx.x;                    // within block
  const int warp_id = tid / WARP_SIZE;            // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE;            // 0~31
  const int warp_QP = warp_id;                    // 0,1,2,3 or 0~7
  const int warp_KV = 0;                          // 0

  const int Q_gmem_offset =
      ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) +
       (QKV_head_id * QKV_seqlen * kHeadDim)); // Q [seqlen,d]
  const int K_gmem_offset =
      ((QKV_batch_id * QKV_head * QKV_seqlen * kHeadDim) +
       (QKV_head_id * QKV_seqlen * kHeadDim)); // K [seqlen,d]
  const int V_gmem_offset = Q_gmem_offset;     // V [seqlen,d]
  const int O_gmem_offset = Q_gmem_offset;     // O [seqlen,d]

  // Mapping Q gmem -> tid -> smem, Q[Br,d]=[64,64 or 128], 128 threads.
  int load_smem_Q_Br = (tid / (kNumThreads / Br)); // Br 64, tid / 2, row 0~64
  int load_smem_Q_d =
      (tid % (kNumThreads / Br)) *
      (kHeadDim / (kNumThreads / Br)); // (tid % 2) * 32, 0,32,...
  // Mapping K gmem -> tid -> smem, K[Bc,d]=[64 or 128,64], 128 threads.
  int load_smem_K_Bc = (tid / (kNumThreads / Bc)); // Bc 64, tid / 2, row 0~64
  int load_smem_K_d =
      (tid % (kNumThreads / Bc)) *
      (kHeadDim / (kNumThreads / Bc)); // (tid % 2) * 32, 0,32,...
  // Mapping V gmem -> tid -> smem, V[Bc,d]=[64,64 or 128], 128 threads.
  int load_smem_V_Bc = (tid / (kNumThreads / Bc)); // Bc 64, tid / 2, row 0~64
  int load_smem_V_d =
      (tid % (kNumThreads / Bc)) *
      (kHeadDim / (kNumThreads / Bc)); // (tid % 2) * 32, 0,32,...
  // global Q row of current head for tile [Br,d] per block.
  int load_gmem_Q_Br = Q_tile_id * Br + load_smem_Q_Br;
  if (load_gmem_Q_Br >= QKV_seqlen)
    return;

  // Shared memory for Q,K,V, we don not need additional smem for O
  // collective store which perform via registers reuse and warp shuffle.
  extern __shared__ half smem[];
  constexpr int Q_tile_size =
      Br * (kHeadDim + kPadQ); // 64*64=4096, ~8192 bytes=8M
  constexpr int K_tile_size = Bc * (kHeadDim + kPadK); // K[Bc,d]
  constexpr int V_tile_size = Bc * (kHeadDim + kPadV); // V[Bc,d]
  half *Q_tile_smem = smem;                            // 8M/16M
  half *K_tile_smem = Q_tile_smem + Q_tile_size;       // 8M/16M
  half *V_tile_smem = K_tile_smem; // KV shared the same smem

  uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  // Registers/SMEM for thread block
  // block m_old, l_old, store in lane, use float to
  // keep precision.
  float lane_block_row_max_old[kWarpTileSeqLenQ][2]; // [1][2]
  float lane_block_row_sum_old[kWarpTileSeqLenQ][2]; // [1][2]
  fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_max_old, -INFINITY);
  fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_sum_old, 0.0f);

  // Registers for S=Q@K^T/O=P@V
  constexpr bool kCanPrefetchQs2r =
      ((kHeadDim / kMmaAtomK) <= 8) && (kHeadDim < 64);
  constexpr bool kDelayPrefetchQs2r =
      (true && kCanPrefetchQs2r);                   // TODO: make it optional.
  constexpr bool kCanPrefetchKVg2s = (kStage == 2); // whether prefetch KV g2s.
  constexpr int kPrefetchKg2sSmemId = 0;            // smem id for K g2s, 0.
  constexpr int kPrefetchVg2sSmemId =
      kCanPrefetchKVg2s ? 1 : 0; // smem id for V g2s, 1.
  constexpr int kNumPrefetchQs2r =
      (kCanPrefetchQs2r) ? (kHeadDim / kMmaAtomK) : 1;
  uint32_t R_Q[kNumPrefetchQs2r][kWarpTileSeqLenQ][4]; // [4/8/1][1][4]
  uint32_t R_K[kWarpTileSeqLenK][2];                   // [8][2]
  uint32_t R_V[kWarpTileHeadDimV][2];                  // [8][2]
  // registers for current tile_K_seqlen within, [64,64] = S_tile[Br,Bc]
  // = Q_tile[Br,d] * K[Bc,d], each thread hold 2x32 bits regs.
  uint32_t R_S[kWarpTileSeqLenQ][kWarpTileSeqLenK][2]; // [1][8][2]
  // registers for tile_K_seqlen O=PV[Br,d]=P@V, [2][2/4][2], 8 or 16 regs.
  uint32_t R_O[kWarpTileSeqLenP][kWarpTileHeadDimV][2]; // [1][8][2]
  // registers final Output [D]=final rescale(R_O), [2][2/4][2], 8 or 16 regs.
  uint32_t R_D[kWarpTileSeqLenP][kWarpTileHeadDimV]
              [(kOStorageAccFloat32) ? 4 : 2];
  fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV,
               ((kOStorageAccFloat32) ? 4 : 2)>(R_D, 0);

  // load Q from gmem -> smem, only load once.
  {
    int load_gmem_Q_d = load_smem_Q_d;
    int load_gmem_Q_addr =
        (Q_gmem_offset + load_gmem_Q_Br * kHeadDim + load_gmem_Q_d);
    uint32_t load_smem_Q_ptr =
        (smem_Q_base_ptr +
         (load_smem_Q_Br * (kHeadDim + kPadQ) + load_gmem_Q_d) * sizeof(half));
#pragma unroll
    for (int i = 0; i < (kHeadDim / (kNumThreads / Br)); i += 8) {
      CP_ASYNC_CG(load_smem_Q_ptr + i * 2, &Q[load_gmem_Q_addr + i], 16);
    }
    CP_ASYNC_COMMIT_GROUP();
  }

  // Get number of active KV blocks for this Q block
  int kv_num_blocks_offset = QKV_batch_id * kv_num_blocks_stride0 +
                             QKV_head_id * kv_num_blocks_stride1 +
                             Q_tile_id * kv_num_blocks_stride2;
  int num_active_kv_blocks = kv_num_blocks[kv_num_blocks_offset];

  // <loop over active KV blocks>: Instead of looping over all KV blocks,
  // we only process blocks specified by kv_indices
#pragma unroll 1
  for (int active_kv_idx = 0; active_kv_idx < num_active_kv_blocks; ++active_kv_idx) {
    // Get KV block index and mask type
    int kv_indices_offset = QKV_batch_id * kv_indices_stride0 +
                            QKV_head_id * kv_indices_stride1 +
                            Q_tile_id * kv_indices_stride2 +
                            active_kv_idx * kv_indices_stride3;
    int kv_block_col_idx = kv_indices[kv_indices_offset];
    
    int block_mask_types_offset = QKV_batch_id * block_mask_types_stride0 +
                                  QKV_head_id * block_mask_types_stride1 +
                                  Q_tile_id * block_mask_types_stride2 +
                                  active_kv_idx * block_mask_types_stride3;
    int mask_type = block_mask_types[block_mask_types_offset];

    // Skip if fully masked
    if (mask_type == BLOCK_MASK_MASKED) {
      continue;
    }

    // Calculate KV block offset in sequence
    int load_gmem_K_Bc_offset = kv_block_col_idx * Bc;
    int load_gmem_V_Bc_offset = kv_block_col_idx * Bc;

    // Load K tile from gmem -> smem
    if constexpr (kCanPrefetchKVg2s) {
      if (active_kv_idx == 0) {
        int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
        int load_gmem_K_d = load_smem_K_d;
        int load_gmem_K_addr =
            (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
        uint32_t load_smem_K_ptr =
            (smem_K_base_ptr +
             (kPrefetchKg2sSmemId * K_tile_size +
              load_smem_K_Bc * (kHeadDim + kPadK) + load_gmem_K_d) *
                 sizeof(half));
#pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
      }
      // Prefetch V g2s
      {
        int load_gmem_V_Bc = load_gmem_V_Bc_offset + load_smem_V_Bc;
        int load_gmem_V_d = load_smem_V_d;
        int load_gmem_V_addr =
            (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
        uint32_t load_smem_V_ptr =
            (smem_V_base_ptr +
             (kPrefetchVg2sSmemId * V_tile_size +
              load_gmem_V_Bc * (kHeadDim + kPadV) + load_gmem_V_d) *
                 sizeof(half));
#pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      }
    } else {
      int load_gmem_K_Bc = load_gmem_K_Bc_offset + load_smem_K_Bc;
      int load_gmem_K_d = load_smem_K_d;
      int load_gmem_K_addr =
          (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
      uint32_t load_smem_K_ptr =
          (smem_K_base_ptr +
           (kPrefetchKg2sSmemId * K_tile_size +
            load_gmem_K_Bc * (kHeadDim + kPadK) + load_gmem_K_d) *
               sizeof(half));
#pragma unroll
      for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
        CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
      }
      CP_ASYNC_COMMIT_GROUP();
      CP_ASYNC_WAIT_GROUP(0);
      __syncthreads();
    }

    // Prefetch Q s2r
    if constexpr (kCanPrefetchQs2r && (!kDelayPrefetchQs2r)) {
      if (active_kv_idx == 0) {
        if constexpr (!kCanPrefetchKVg2s) {
          CP_ASYNC_WAIT_GROUP(0);
        } else {
          CP_ASYNC_WAIT_GROUP(1);
        }
        __syncthreads();

#pragma unroll
        for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
#pragma unroll
          for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
            int warp_smem_Q_Br =
                warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
            int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
            int lane_smem_Q_d =
                tile_K_d * kMmaAtomK + (lane_id / 16) * 8;
            uint32_t lane_smem_Q_ptr =
                (smem_Q_base_ptr +
                 (lane_smem_Q_Br * (kHeadDim + kPadQ) + lane_smem_Q_d) *
                     sizeof(half));
            LDMATRIX_X4(R_Q[tile_K_d][i][0], R_Q[tile_K_d][i][1],
                        R_Q[tile_K_d][i][2], R_Q[tile_K_d][i][3],
                        lane_smem_Q_ptr);
          }
        }
        __syncthreads();
      }
    }

    // Compute S = Q@K^T
    fill_3D_regs<uint32_t, kWarpTileSeqLenQ, kWarpTileSeqLenK, 2>(R_S, 0);
#pragma unroll
    for (int tile_K_d = 0; tile_K_d < (kHeadDim / kMmaAtomK); ++tile_K_d) {
      if constexpr (!kCanPrefetchQs2r) {
#pragma unroll
        for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
          int warp_smem_Q_Br =
              warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
          int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
          int lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8;
          uint32_t lane_smem_Q_ptr =
              (smem_Q_base_ptr +
               (lane_smem_Q_Br * (kHeadDim + kPadQ) + lane_smem_Q_d) *
                   sizeof(half));
          LDMATRIX_X4(R_Q[0][i][0], R_Q[0][i][1], R_Q[0][i][2], R_Q[0][i][3],
                      lane_smem_Q_ptr);
        }
      } else {
        if constexpr (kDelayPrefetchQs2r) {
          if (active_kv_idx == 0) {
            if (tile_K_d == 0) {
              if constexpr (!kCanPrefetchKVg2s) {
                CP_ASYNC_WAIT_GROUP(0);
              } else {
                CP_ASYNC_WAIT_GROUP(1);
              }
              __syncthreads();
            }
#pragma unroll
            for (int i = 0; i < kWarpTileSeqLenQ; ++i) {
              int warp_smem_Q_Br =
                  warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
              int lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16;
              int lane_smem_Q_d =
                  tile_K_d * kMmaAtomK + (lane_id / 16) * 8;
              uint32_t lane_smem_Q_ptr =
                  (smem_Q_base_ptr +
                   (lane_smem_Q_Br * (kHeadDim + kPadQ) + lane_smem_Q_d) *
                       sizeof(half));
              LDMATRIX_X4(R_Q[tile_K_d][i][0], R_Q[tile_K_d][i][1],
                          R_Q[tile_K_d][i][2], R_Q[tile_K_d][i][3],
                          lane_smem_Q_ptr);
            }
          }
        }
      }

      // Load K
#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        int warp_smem_K_Bc =
            warp_KV * (kMmaAtomN * kWarpTileSeqLenK) + j * kMmaAtomN;
        int lane_smem_K_Bc = warp_smem_K_Bc + lane_id % 8;
        int lane_smem_K_d =
            tile_K_d * kMmaAtomK + ((lane_id / 8) % 2) * 8;
        uint32_t lane_smem_K_ptr =
            (smem_K_base_ptr +
             (kPrefetchKg2sSmemId * K_tile_size +
              lane_smem_K_Bc * (kHeadDim + kPadK) + lane_smem_K_d) *
                 sizeof(half));
        LDMATRIX_X2(R_K[j][0], R_K[j][1], lane_smem_K_ptr);
      }

      // MMA compute
      if constexpr (kCanPrefetchQs2r) {
        static_assert(kWarpTileSeqLenQ == 1);
        {
#pragma unroll
          for (int j = 0; j < kWarpTileSeqLenK; ++j) {
            HMMA16816(R_S[0][j][0], R_S[0][j][1], R_Q[tile_K_d][0][0],
                      R_Q[tile_K_d][0][1], R_Q[tile_K_d][0][2],
                      R_Q[tile_K_d][0][3], R_K[j][0], R_K[j][1], R_S[0][j][0],
                      R_S[0][j][1]);
          }
        }
      } else {
        static_assert(kWarpTileSeqLenQ == 1);
        {
#pragma unroll
          for (int j = 0; j < kWarpTileSeqLenK; ++j) {
            HMMA16816(R_S[0][j][0], R_S[0][j][1], R_Q[0][0][0], R_Q[0][0][1],
                      R_Q[0][0][2], R_Q[0][0][3], R_K[j][0], R_K[j][1],
                      R_S[0][j][0], R_S[0][j][1]);
          }
        }
      }
    } // end loop over d, S=Q@K^T
    __syncthreads();

    // Apply causal masking if needed
    if (mask_type == BLOCK_MASK_CAUSAL) {
      // Apply causal mask: q_idx >= kv_idx (using global sequence indices)
      // The MMA fragment layout for m16n8k16 distributes elements across threads
      // We need to apply the mask based on the actual global Q and KV positions
      static_assert(kWarpTileSeqLenQ == 1);
      {
        // Calculate global Q block start
        int q_block_start = Q_tile_id * Br;
        int kv_block_start = kv_block_col_idx * Bc;
        
#pragma unroll
        for (int j = 0; j < kWarpTileSeqLenK; ++j) {
          // Calculate warp-level offsets
          int warp_q_start = warp_QP * (kMmaAtomM * kWarpTileSeqLenQ);
          int warp_kv_start = warp_KV * (kMmaAtomN * kWarpTileSeqLenK) + j * kMmaAtomN;
          
          // For m16n8k16 MMA, the fragment layout is:
          // Each thread holds elements from rows [warp_q_start + (lane_id % 16) + row_offset]
          // and columns [warp_kv_start + (lane_id % 8) + col_offset]
          // The exact mapping depends on the MMA instruction layout
          // For simplicity, we apply the mask conservatively:
          // - Check if the Q row is before the KV block start
          // - Apply mask to all elements in that row if q_global < kv_block_start
          
          int q_row_in_warp = lane_id % 16;  // 0-15
          int q_global_row = q_block_start + warp_q_start + q_row_in_warp;
          
          // If this Q row is before the KV block, mask all elements
          if (q_global_row < kv_block_start) {
            half *t_hptr_S = reinterpret_cast<half *>(&(R_S[0][j][0]));
            t_hptr_S[0] = __float2half(-INFINITY);
            t_hptr_S[1] = __float2half(-INFINITY);
            t_hptr_S[2] = __float2half(-INFINITY);
            t_hptr_S[3] = __float2half(-INFINITY);
          } else {
            // For elements within the block, check element-wise
            // The MMA fragment has 4 half values per thread
            // Approximate mapping: elements correspond to different columns
            int kv_col_in_warp = lane_id % 8;  // 0-7
            int kv_global_col_base = kv_block_start + warp_kv_start + kv_col_in_warp;
            
            half *t_hptr_S = reinterpret_cast<half *>(&(R_S[0][j][0]));
            // Apply mask element-wise: q_global_row >= kv_global_col
            // For the 4 elements, they roughly correspond to columns at offsets
            for (int elem = 0; elem < 4; ++elem) {
              // Approximate column mapping for m16n8k16 fragment
              int kv_col_offset = (elem / 2) * 8 + (elem % 2) * 4;
              int kv_global_col = kv_global_col_base + kv_col_offset;
              
              if (q_global_row < kv_global_col) {
                t_hptr_S[elem] = __float2half(-INFINITY);
              }
            }
          }
        }
      }
    }

    // Load V if not prefetched
    if constexpr (!kCanPrefetchKVg2s) {
      int load_gmem_V_Bc = load_gmem_V_Bc_offset + load_smem_V_Bc;
      int load_gmem_V_d = load_smem_V_d;
      int load_gmem_V_addr =
          (V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d);
      uint32_t load_smem_V_ptr =
          (smem_V_base_ptr +
           (kPrefetchVg2sSmemId * V_tile_size +
            load_gmem_V_Bc * (kHeadDim + kPadV) + load_gmem_V_d) *
               sizeof(half));
#pragma unroll
      for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
        CP_ASYNC_CG(load_smem_V_ptr + i * 2, &V[load_gmem_V_addr + i], 16);
      }
      CP_ASYNC_COMMIT_GROUP();
    }

    // Prefetch next K if applicable
    if constexpr (kCanPrefetchKVg2s) {
      if ((active_kv_idx + 1) < num_active_kv_blocks) {
        // Get next KV block index
        int next_kv_indices_offset = QKV_batch_id * kv_indices_stride0 +
                                      QKV_head_id * kv_indices_stride1 +
                                      Q_tile_id * kv_indices_stride2 +
                                      (active_kv_idx + 1) * kv_indices_stride3;
        int next_kv_block_col_idx = kv_indices[next_kv_indices_offset];
        int next_load_gmem_K_Bc_offset = next_kv_block_col_idx * Bc;
        
        int load_gmem_K_Bc = next_load_gmem_K_Bc_offset + load_smem_K_Bc;
        int load_gmem_K_d = load_smem_K_d;
        int load_gmem_K_addr =
            (K_gmem_offset + load_gmem_K_Bc * kHeadDim + load_gmem_K_d);
        uint32_t load_smem_K_ptr =
            (smem_K_base_ptr +
             (kPrefetchKg2sSmemId * K_tile_size +
              load_gmem_K_Bc * (kHeadDim + kPadK) + load_gmem_K_d) *
                 sizeof(half));
#pragma unroll
        for (int i = 0; i < (kHeadDim / (kNumThreads / Bc)); i += 8) {
          CP_ASYNC_CG(load_smem_K_ptr + i * 2, &K[load_gmem_K_addr + i], 16);
        }
        CP_ASYNC_COMMIT_GROUP();
      }
    }

    // Online safe softmax, warp/block reduce max/sum, row wise
    float lane_row_max_new[kWarpTileSeqLenQ][2];
    float lane_row_sum_new[kWarpTileSeqLenQ][2];
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_max_new, -INFINITY);
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_sum_new, 0.0f);

    static_assert(kWarpTileSeqLenQ == 1);
    // Row max for [Br,Bc] tile
    {
#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        half *t_hptr_S_0_1 = reinterpret_cast<half *>(&(R_S[0][j][0]));
        float tmp_max_0 =
            __half2float(__hmax(t_hptr_S_0_1[0], t_hptr_S_0_1[1])) * scale;
        float tmp_max_1 =
            __half2float(__hmax(t_hptr_S_0_1[2], t_hptr_S_0_1[3])) * scale;
        lane_row_max_new[0][0] = max(lane_row_max_new[0][0], tmp_max_0);
        lane_row_max_new[0][1] = max(lane_row_max_new[0][1], tmp_max_1);
      }

      lane_row_max_new[0][0] =
          warp_reduce_max<float, 4>(lane_row_max_new[0][0]);
      lane_row_max_new[0][1] =
          warp_reduce_max<float, 4>(lane_row_max_new[0][1]);
    }

    static_assert(kWarpTileSeqLenQ == 1);
    // Exp sum and mul scale_factor
    {
      float block_row_max_new_0 = lane_row_max_new[0][0];
      float block_row_max_new_1 = lane_row_max_new[0][1];
      float block_row_max_old_0 = lane_block_row_max_old[0][0];
      float block_row_max_old_1 = lane_block_row_max_old[0][1];
      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);

#pragma unroll
      for (int j = 0; j < kWarpTileSeqLenK; ++j) {
        half *t_hptr_S_0_1 = reinterpret_cast<half *>(&(R_S[0][j][0]));
        float4 t_reg_S_0_1;
        t_reg_S_0_1.x = __expf(__fmaf_rn(__half2float(t_hptr_S_0_1[0]), scale,
                                         -block_row_max_new_0));
        t_reg_S_0_1.y = __expf(__fmaf_rn(__half2float(t_hptr_S_0_1[1]), scale,
                                         -block_row_max_new_0));
        t_reg_S_0_1.z = __expf(__fmaf_rn(__half2float(t_hptr_S_0_1[2]), scale,
                                         -block_row_max_new_1));
        t_reg_S_0_1.w = __expf(__fmaf_rn(__half2float(t_hptr_S_0_1[3]), scale,
                                         -block_row_max_new_1));
        lane_row_sum_new[0][0] += (t_reg_S_0_1.x + t_reg_S_0_1.y);
        lane_row_sum_new[0][1] += (t_reg_S_0_1.z + t_reg_S_0_1.w);
        t_hptr_S_0_1[0] = __float2half_rn(t_reg_S_0_1.x);
        t_hptr_S_0_1[1] = __float2half_rn(t_reg_S_0_1.y);
        t_hptr_S_0_1[2] = __float2half_rn(t_reg_S_0_1.z);
        t_hptr_S_0_1[3] = __float2half_rn(t_reg_S_0_1.w);
      }

      lane_row_sum_new[0][0] =
          warp_reduce_sum<float, 4>(lane_row_sum_new[0][0]);
      lane_row_sum_new[0][1] =
          warp_reduce_sum<float, 4>(lane_row_sum_new[0][1]);
    }

    // Wait for V ready
    if constexpr (kCanPrefetchKVg2s) {
      if ((active_kv_idx + 1) < num_active_kv_blocks) {
        CP_ASYNC_WAIT_GROUP(1);
      } else {
        CP_ASYNC_WAIT_GROUP(0);
      }
    } else {
      CP_ASYNC_WAIT_GROUP(0);
    }
    __syncthreads();

    // Compute O = P@V
    fill_3D_regs<uint32_t, kWarpTileSeqLenP, kWarpTileHeadDimV, 2>(R_O, 0);
#pragma unroll
    for (int tile_V_Bc = 0; tile_V_Bc < (Bc / kMmaAtomK); ++tile_V_Bc) {
#pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) {
        int warp_smem_V_d = warp_KV * (kMmaAtomN * kWarpTileHeadDimV) +
                            j * kMmaAtomN;
        int lane_smem_V_Bc =
            tile_V_Bc * kMmaAtomK + lane_id % 16;
        int lane_smem_V_d = warp_smem_V_d;
        uint32_t lane_smem_V_ptr =
            (smem_V_base_ptr +
             (kPrefetchVg2sSmemId * V_tile_size +
              lane_smem_V_Bc * (kHeadDim + kPadV) + lane_smem_V_d) *
                 sizeof(half));
        LDMATRIX_X2_T(R_V[j][0], R_V[j][1], lane_smem_V_ptr);
      }

      int w = tile_V_Bc * 2;
      static_assert(kWarpTileSeqLenP == 1);
      {
#pragma unroll
        for (int j = 0; j < kWarpTileHeadDimV; ++j) {
          HMMA16816(R_O[0][j][0], R_O[0][j][1], R_S[0][w][0], R_S[0][w][1],
                    R_S[0][w + 1][0], R_S[0][w + 1][1], R_V[j][0], R_V[j][1],
                    R_O[0][j][0], R_O[0][j][1]);
        }
      }
    }
    __syncthreads();

    // Rescale O
    static_assert(kWarpTileSeqLenP == 1);
    {
      float block_row_max_new_0 = lane_row_max_new[0][0];
      float block_row_max_new_1 = lane_row_max_new[0][1];
      float block_row_sum_new_0 = lane_row_sum_new[0][0];
      float block_row_sum_new_1 = lane_row_sum_new[0][1];
      float block_row_max_old_0 = lane_block_row_max_old[0][0];
      float block_row_max_old_1 = lane_block_row_max_old[0][1];
      block_row_max_new_0 = max(block_row_max_old_0, block_row_max_new_0);
      block_row_max_new_1 = max(block_row_max_old_1, block_row_max_new_1);
      block_row_max_old_0 =
          (active_kv_idx > 0 ? block_row_max_old_0 : block_row_max_new_0);
      block_row_max_old_1 =
          (active_kv_idx > 0 ? block_row_max_old_1 : block_row_max_new_1);

      float rescale_o_factor_0 =
          __expf(block_row_max_old_0 - block_row_max_new_0);
      float rescale_o_factor_1 =
          __expf(block_row_max_old_1 - block_row_max_new_1);

#pragma unroll
      for (int j = 0; j < kWarpTileHeadDimV; ++j) {
        half *t_hptr_O_0_1 = reinterpret_cast<half *>(&(R_O[0][j][0]));
        if constexpr (kOStorageAccFloat32) {
          float *t_fptr_D_0_1 = reinterpret_cast<float *>(&(R_D[0][j][0]));
          t_fptr_D_0_1[0] = __fmaf_rn(rescale_o_factor_0, t_fptr_D_0_1[0],
                                      __half2float(t_hptr_O_0_1[0]));
          t_fptr_D_0_1[1] = __fmaf_rn(rescale_o_factor_0, t_fptr_D_0_1[1],
                                      __half2float(t_hptr_O_0_1[1]));
          t_fptr_D_0_1[2] = __fmaf_rn(rescale_o_factor_1, t_fptr_D_0_1[2],
                                      __half2float(t_hptr_O_0_1[2]));
          t_fptr_D_0_1[3] = __fmaf_rn(rescale_o_factor_1, t_fptr_D_0_1[3],
                                      __half2float(t_hptr_O_0_1[3]));
        } else {
          half *t_hptr_D_0_1 = reinterpret_cast<half *>(&(R_D[0][j][0]));
          t_hptr_D_0_1[0] = __float2half_rn(
              __fmaf_rn(rescale_o_factor_0, __half2float(t_hptr_D_0_1[0]),
                        __half2float(t_hptr_O_0_1[0])));
          t_hptr_D_0_1[1] = __float2half_rn(
              __fmaf_rn(rescale_o_factor_0, __half2float(t_hptr_D_0_1[1]),
                        __half2float(t_hptr_O_0_1[1])));
          t_hptr_D_0_1[2] = __float2half_rn(
              __fmaf_rn(rescale_o_factor_1, __half2float(t_hptr_D_0_1[2]),
                        __half2float(t_hptr_O_0_1[2])));
          t_hptr_D_0_1[3] = __float2half_rn(
              __fmaf_rn(rescale_o_factor_1, __half2float(t_hptr_D_0_1[3]),
                        __half2float(t_hptr_O_0_1[3])));
        }
      }

      float block_row_sum_old_0 = lane_block_row_sum_old[0][0];
      float block_row_sum_old_1 = lane_block_row_sum_old[0][1];
      lane_block_row_sum_old[0][0] = (__fmaf_rn(
          rescale_o_factor_0, block_row_sum_old_0, block_row_sum_new_0));
      lane_block_row_sum_old[0][1] = (__fmaf_rn(
          rescale_o_factor_1, block_row_sum_old_1, block_row_sum_new_1));
      lane_block_row_max_old[0][0] = block_row_max_new_0;
      lane_block_row_max_old[0][1] = block_row_max_new_1;
    }

    if constexpr (kCanPrefetchKVg2s) {
      if ((active_kv_idx + 1) < num_active_kv_blocks) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
      }
    }

  } // end loop over active KV blocks
  __syncthreads();

  // Final rescale
  static_assert(kWarpTileSeqLenP == 1);
  {
    float rescale_factor_0 = __frcp_rn(lane_block_row_sum_old[0][0]);
    float rescale_factor_1 = __frcp_rn(lane_block_row_sum_old[0][1]);
#pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) {
      if constexpr (kOStorageAccFloat32) {
        float *t_fptr_D_0_1 = reinterpret_cast<float *>(&(R_D[0][j][0]));
        half *t_hptr_D_0_1 = reinterpret_cast<half *>(&(R_D[0][j][0]));
        t_hptr_D_0_1[0] = __float2half_rn(rescale_factor_0 * t_fptr_D_0_1[0]);
        t_hptr_D_0_1[1] = __float2half_rn(rescale_factor_0 * t_fptr_D_0_1[1]);
        t_hptr_D_0_1[2] = __float2half_rn(rescale_factor_1 * t_fptr_D_0_1[2]);
        t_hptr_D_0_1[3] = __float2half_rn(rescale_factor_1 * t_fptr_D_0_1[3]);
      } else {
        half *t_hptr_D_0_1 = reinterpret_cast<half *>(&(R_D[0][j][0]));
        t_hptr_D_0_1[0] =
            __float2half_rn(rescale_factor_0 * __half2float(t_hptr_D_0_1[0]));
        t_hptr_D_0_1[1] =
            __float2half_rn(rescale_factor_0 * __half2float(t_hptr_D_0_1[1]));
        t_hptr_D_0_1[2] =
            __float2half_rn(rescale_factor_1 * __half2float(t_hptr_D_0_1[2]));
        t_hptr_D_0_1[3] =
            __float2half_rn(rescale_factor_1 * __half2float(t_hptr_D_0_1[3]));
      }
    }
  }

  // Store O
  static_assert(kWarpTileSeqLenP == 1);
  {
#pragma unroll
    for (int j = 0; j < kWarpTileHeadDimV; ++j) {
      if constexpr (kCanPrefetchQs2r && kNumPrefetchQs2r > 1) {
        R_Q[0][0][0] = R_D[0][j][0];
        R_Q[1][0][0] = R_D[0][j][1];
        R_Q[0][0][1] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 1, 4);
        R_Q[0][0][2] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 2, 4);
        R_Q[0][0][3] = __shfl_sync((0xffffffff), R_D[0][j][0], lane_id + 3, 4);
        R_Q[1][0][1] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 1, 4);
        R_Q[1][0][2] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 2, 4);
        R_Q[1][0][3] = __shfl_sync((0xffffffff), R_D[0][j][1], lane_id + 3, 4);
        if (lane_id % 4 == 0) {
          int store_warp_regs_O_Br =
              warp_QP * (kMmaAtomM * kWarpTileSeqLenP) + 0 * kMmaAtomM;
          int store_lane_gmem_O_Br =
              O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4;
          int store_warp_regs_O_d =
              warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
          int store_lane_gmem_O_d = store_warp_regs_O_d;
          int store_gmem_O_addr_0 =
              (O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim +
               store_lane_gmem_O_d);
          int store_gmem_O_addr_1 =
              (O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim +
               store_lane_gmem_O_d);
          LDST128BITS(O[store_gmem_O_addr_0]) = LDST128BITS(R_Q[0][0][0]);
          LDST128BITS(O[store_gmem_O_addr_1]) = LDST128BITS(R_Q[1][0][0]);
        }
      } else {
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
          int store_warp_regs_O_Br =
              warp_QP * (kMmaAtomM * kWarpTileSeqLenP) + 0 * kMmaAtomM;
          int store_lane_gmem_O_Br =
              O_tile_id * Br + store_warp_regs_O_Br + lane_id / 4;
          int store_warp_regs_O_d =
              warp_KV * (kMmaAtomN * kWarpTileHeadDimV) + j * kMmaAtomN;
          int store_lane_gmem_O_d = store_warp_regs_O_d;
          int store_gmem_O_addr_0 =
              (O_gmem_offset + (store_lane_gmem_O_Br + 0) * kHeadDim +
               store_lane_gmem_O_d);
          int store_gmem_O_addr_1 =
              (O_gmem_offset + (store_lane_gmem_O_Br + 8) * kHeadDim +
               store_lane_gmem_O_d);
          LDST128BITS(O[store_gmem_O_addr_0]) = LDST128BITS(R_Z[0][0]);
          LDST128BITS(O[store_gmem_O_addr_1]) = LDST128BITS(R_Z[1][0]);
        }
      }
    }
  }
}

template <const int kHeadDim, const int kStage>
void launch_omni_attn_mma_stages_split_q_shared_kv(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor kv_num_blocks, torch::Tensor kv_indices,
    torch::Tensor block_mask_types) {
  constexpr int kMmaAtomM = 16;
  constexpr int kMmaAtomN = 8;
  constexpr int kMmaAtomK = 16;
#ifdef BUILD_FLASH_ATTN_MMA_L20
  constexpr int kMmaTileSeqLenQ = 4;
  constexpr int kMmaTileSeqLenK = 1;
  constexpr int kMmaTileSeqLenP = 4;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kWarpTileSeqLenQ = 1;
  constexpr int kWarpTileSeqLenK = (kStage > 1) ? 4 : 8;
  constexpr int kWarpTileSeqLenP = 1;
#else
  constexpr int kMmaTileSeqLenQ = (kHeadDim < 128) ? 8 : 8;
  constexpr int kMmaTileSeqLenK = 1;
  constexpr int kMmaTileSeqLenP = (kHeadDim < 128) ? 8 : 8;
  constexpr int kMmaTileHeadDimV = 1;
  constexpr int kWarpTileSeqLenQ = 1;
  constexpr int kWarpTileSeqLenK = (kHeadDim < 128) ? 8 : 4;
  constexpr int kWarpTileSeqLenP = 1;
#endif
  constexpr int kWarpTileHeadDimV =
      (kHeadDim / (kMmaAtomN * kMmaTileHeadDimV));
  constexpr int Br =
      kMmaAtomM * kMmaTileSeqLenQ * kWarpTileSeqLenQ;
  constexpr int Bc =
      kMmaAtomN * kMmaTileSeqLenK * kWarpTileSeqLenK;
  constexpr int kNumThreads =
      WARP_SIZE * kMmaTileSeqLenQ * kMmaTileSeqLenK;
  constexpr int kPadQ = 8;
  constexpr int kPadK = 8;
  constexpr int kPadV = 8;
  constexpr int kOStorageAccFloat32 = (kHeadDim < 256) ? 1 : 0;

  constexpr int Q_tile_size = (Br * (kHeadDim + kPadQ));
  constexpr int K_tile_size = (Bc * (kHeadDim + kPadK));
  constexpr int V_tile_size = (Bc * (kHeadDim + kPadV));
  const int smem_max_size =
      (Q_tile_size + kStage * max(K_tile_size, V_tile_size)) * sizeof(half);

  const int QKV_batch = Q.size(0);
  const int QKV_head = Q.size(1);
  const int QKV_seqlen = Q.size(2);
  const int num_q_blocks = (QKV_seqlen + Br - 1) / Br;
  const int max_blocks = kv_indices.size(3);

  dim3 grid(div_ceil(QKV_seqlen, Br), QKV_batch * QKV_head);
  dim3 block(kNumThreads);

  cudaFuncSetAttribute(
      omni_attn_mma_stages_split_q_shared_kv_kernel<
          kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
          kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ,
          kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
          kOStorageAccFloat32, kStage, kPadQ, kPadK, kPadV>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      98304);

  omni_attn_mma_stages_split_q_shared_kv_kernel<
      kHeadDim, kMmaAtomM, kMmaAtomN, kMmaAtomK, kMmaTileSeqLenQ,
      kMmaTileSeqLenK, kMmaTileSeqLenP, kMmaTileHeadDimV, kWarpTileSeqLenQ,
      kWarpTileSeqLenK, kWarpTileSeqLenP, kWarpTileHeadDimV,
      kOStorageAccFloat32, kStage, kPadQ, kPadK, kPadV>
      <<<grid, block, smem_max_size>>>(
          reinterpret_cast<half *>(Q.data_ptr()),
          reinterpret_cast<half *>(K.data_ptr()),
          reinterpret_cast<half *>(V.data_ptr()),
          reinterpret_cast<half *>(O.data_ptr()),
          reinterpret_cast<int *>(kv_num_blocks.data_ptr()),
          reinterpret_cast<int *>(kv_indices.data_ptr()),
          reinterpret_cast<int *>(block_mask_types.data_ptr()),
          QKV_seqlen, QKV_head, QKV_batch, num_q_blocks, max_blocks,
          kv_num_blocks.stride(0), kv_num_blocks.stride(1),
          kv_num_blocks.stride(2), kv_indices.stride(0), kv_indices.stride(1),
          kv_indices.stride(2), kv_indices.stride(3),
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
  const int d = Q.size(3);

  if (stages > 1) {
    switch (d) {
    case 32:
      launch_omni_attn_mma_stages_split_q_shared_kv<32, 2>(
          Q, K, V, O, kv_num_blocks, kv_indices, block_mask_types);
      break;
    case 64:
      launch_omni_attn_mma_stages_split_q_shared_kv<64, 2>(
          Q, K, V, O, kv_num_blocks, kv_indices, block_mask_types);
      break;
    case 96:
      launch_omni_attn_mma_stages_split_q_shared_kv<96, 2>(
          Q, K, V, O, kv_num_blocks, kv_indices, block_mask_types);
      break;
    case 128:
      launch_omni_attn_mma_stages_split_q_shared_kv<128, 2>(
          Q, K, V, O, kv_num_blocks, kv_indices, block_mask_types);
      break;
    default:
      throw std::runtime_error("headdim not support!");
      break;
    }
  } else {
    switch (d) {
    case 32:
      launch_omni_attn_mma_stages_split_q_shared_kv<32, 1>(
          Q, K, V, O, kv_num_blocks, kv_indices, block_mask_types);
      break;
    case 64:
      launch_omni_attn_mma_stages_split_q_shared_kv<64, 1>(
          Q, K, V, O, kv_num_blocks, kv_indices, block_mask_types);
      break;
    case 96:
      launch_omni_attn_mma_stages_split_q_shared_kv<96, 1>(
          Q, K, V, O, kv_num_blocks, kv_indices, block_mask_types);
      break;
    case 128:
      launch_omni_attn_mma_stages_split_q_shared_kv<128, 1>(
          Q, K, V, O, kv_num_blocks, kv_indices, block_mask_types);
      break;
    case 256:
      launch_omni_attn_mma_stages_split_q_shared_kv<256, 1>(
          Q, K, V, O, kv_num_blocks, kv_indices, block_mask_types);
      break;
    default:
      throw std::runtime_error("headdim not support!");
      break;
    }
  }
}

