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
  constexpr int V_tile_size = Bc * (kHeadDim + kPadV); // V[Bc,d]
  half *Q_tile_smem = smem;                           
  half *K_tile_smem = Q_tile_smem + Q_tile_size;       // Base for K buffers
  half *V_tile_smem = K_tile_smem + K_tile_size;       // Base for V buffers

  uint32_t smem_Q_base_ptr = __cvta_generic_to_shared(Q_tile_smem);
  uint32_t smem_K_base_ptr = __cvta_generic_to_shared(K_tile_smem);
  uint32_t smem_V_base_ptr = __cvta_generic_to_shared(V_tile_smem);

  // Double-buffering constants
  constexpr int kPrefetchKg2sSmemId = 0;            // smem id for K g2s, 0.
  constexpr int kPrefetchVg2sSmemId = 0;            // smem id for V g2s, 0 (V has separate base pointer).

  // Registers/SMEM for thread block
  float lane_block_row_max_old[kWarpTileSeqLenQ][2]; // [1][2]
  float lane_block_row_sum_old[kWarpTileSeqLenQ][2]; // [1][2]
  fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_max_old, -INFINITY);
  fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_block_row_sum_old, 0.0f);

  // MODIFIED: Simple computation arrays instead of MMA registers
  float R_O[Br][kHeadDim];  // Output accumulator
  int q_thread = tid;
  if (q_thread < Br) {
    for (int d = 0; d < kHeadDim; d++) {
      R_O[q_thread][d] = 0.0f;
    }
  }

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

    // MODIFIED: compute S = Q @ K^T - simple version
    float R_S[Bc];  // Scores for this thread's Q row
    if (q_thread < Br) {
      int q_pos = Q_tile_id * Br + q_thread;
      if (q_pos < QKV_seqlen) {
        for (int kv = 0; kv < Bc; kv++) {
          int kv_pos = kv_block_start + kv;
          if (kv_pos >= kv_block_end) {
            R_S[kv] = -INFINITY;
            continue;
          }
          
          float score = 0.0f;
          #pragma unroll
          for (int d = 0; d < kHeadDim; d++) {
            float q_val = __half2float(Q_tile_smem[q_thread * (kHeadDim + kPadQ) + d]);
            float k_val = __half2float(K_tile_smem[kv * (kHeadDim + kPadK) + d]);
            score += q_val * k_val;
          }
          R_S[kv] = score * scale;
        }
      }
    }
    __syncthreads();

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

    // Apply causal mask if needed
    if (mask_type == BLOCK_MASK_CAUSAL && q_thread < Br) {
      int q_pos = Q_tile_id * Br + q_thread;
      int kv_base = kv_block * KV_BLOCK_SIZE;
      
      for (int kv = 0; kv < Bc; kv++) {
        int kv_pos = kv_base + kv;
        if (q_pos < kv_pos) {
          R_S[kv] = -INFINITY;
        }
      }
    }

    // Apply partial mask if needed
    if (mask_type == BLOCK_MASK_PARTIAL && has_partial_masks && partial_block_masks != nullptr && q_thread < Br) {
      int q_pos = Q_tile_id * Br + q_thread;
      int kv_base = kv_block * KV_BLOCK_SIZE;
      int q_block_start = q_block * Q_BLOCK_SIZE;
      const int block_area = Q_BLOCK_SIZE * KV_BLOCK_SIZE;
      
      for (int kv = 0; kv < Bc; kv++) {
        int kv_pos = kv_base + kv;
        if (q_pos < QKV_seqlen_orig && kv_pos < QKV_seqlen_orig) {
          int local_q = q_pos - q_block_start;
          int local_kv = kv_pos - kv_base;
          if (local_q >= 0 && local_q < Q_BLOCK_SIZE && local_kv >= 0 && local_kv < KV_BLOCK_SIZE) {
            int mask_offset = partial_block_index * block_area + local_q * KV_BLOCK_SIZE + local_kv;
            if (!partial_block_masks[mask_offset]) {
              R_S[kv] = -INFINITY;
            }
          }
        }
      }
    }

    // Wait for V to be ready (prefetched in buffer 1)
    CP_ASYNC_WAIT_GROUP(1);
    __syncthreads();

    // MODIFIED: compute P = softmax(S) - simple version
    float lane_row_max_new[kWarpTileSeqLenQ][2];
    float lane_row_sum_new[kWarpTileSeqLenQ][2];
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_max_new, -INFINITY);
    fill_2D_regs<float, kWarpTileSeqLenQ, 2>(lane_row_sum_new, 0.0f);

    static_assert(kWarpTileSeqLenQ == 1);
    // Row max reduction
    if (q_thread < Br) {
      float block_max = -INFINITY;
      for (int kv = 0; kv < Bc; kv++) {
        block_max = max(block_max, R_S[kv]);
      }
      lane_row_max_new[0][q_thread & 1] = block_max;
    }

    // Compute exp and row sum
    if (q_thread < Br) {
      float block_row_max_new = lane_row_max_new[0][q_thread & 1];
      float block_row_max_old = lane_block_row_max_old[0][q_thread & 1];
      block_row_max_new = max(block_row_max_old, block_row_max_new);
      float block_row_max_old_safe = (kv_idx > 0 ? block_row_max_old : block_row_max_new);

      float new_sum = 0.0f;
      for (int kv = 0; kv < Bc; kv++) {
        float exp_score = __expf(R_S[kv] - block_row_max_new);
        R_S[kv] = exp_score;
        new_sum += exp_score;
      }
      lane_row_sum_new[0][q_thread & 1] = new_sum;
    }

    // MODIFIED: compute O = P @ V (using V from buffer 1) - simple version
    if (q_thread < Br) {
      float block_row_max_new = lane_row_max_new[0][q_thread & 1];
      float block_row_max_old = lane_block_row_max_old[0][q_thread & 1];
      block_row_max_new = max(block_row_max_old, block_row_max_new);
      float block_row_max_old_safe = (kv_idx > 0 ? block_row_max_old : block_row_max_new);
      float block_row_sum_new = lane_row_sum_new[0][q_thread & 1];
      float block_row_sum_old = lane_block_row_sum_old[0][q_thread & 1];

      float rescale_o_factor = __expf(block_row_max_old_safe - block_row_max_new);
      
      for (int d = 0; d < kHeadDim; d++) {
        float acc = 0.0f;
        for (int kv = 0; kv < Bc; kv++) {
          float p_val = R_S[kv];
          float v_val = __half2float(V_tile_smem[kv * (kHeadDim + kPadV) + d]);
          acc += p_val * v_val;
        }
        R_O[q_thread][d] = rescale_o_factor * R_O[q_thread][d] + acc;
      }
      
      lane_block_row_sum_old[0][q_thread & 1] = rescale_o_factor * block_row_sum_old + block_row_sum_new;
      lane_block_row_max_old[0][q_thread & 1] = block_row_max_new;
    }
    __syncthreads();

    // Wait for next K to be ready (if prefetched)
    if ((kv_idx + 1) < kv_nb) {
      CP_ASYNC_WAIT_GROUP(0);
      __syncthreads();
    }
  } // end loop over active KV blocks
  __syncthreads();

  // Final rescale and write O to gmem
  static_assert(kWarpTileSeqLenP == 1);
  if (q_thread < Br) {
    int q_pos = Q_tile_id * Br + q_thread;
    if (q_pos < QKV_seqlen) {
      float rescale_factor = __frcp_rn(lane_block_row_sum_old[0][q_thread & 1]);
      
      for (int d = 0; d < kHeadDim; d++) {
        float out_val = R_O[q_thread][d] * rescale_factor;
        O[O_gmem_offset + q_pos * kHeadDim + d] = __float2half(out_val);
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
  constexpr int kOStorageAccFloat32 = 0;
  
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
