#include "utils.h"

// Apply cp.async and Q tiling strategy

/**
  Q: [B, H, seqlen, head_dim]
  K: [B, H, seqlen, head_dim]
  V: [B, H, seqlen, head_dim]
  O: [B, H, seqlen, head_dim]
  kv_num_blocks: [B, H, num_q_blocks] number of kv blocks in each q block row tile
  kv_indices: [B, H, num_q_blocks, max_kv_blocks] indices of kv blocks in each q block row tile
  block_mask_types: [B, H, num_q_blocks, max_kv_blocks] mask type for each kv block in each q block row tile

  FA2 split-Q tiling strategy(with online softmax) + sparse block mask + separate buffer for Q, K, V, O
*/

// Block mask type definitions
#define BLOCK_MASK_MASKED 0
#define BLOCK_MASK_CAUSAL 1
#define BLOCK_MASK_FULL 2
#define BLOCK_MASK_PARTIAL 3

// Simple kernel using cp.async and Q tiling
// Each thread block processes one Q block (Q_BLOCK_SIZE rows)
template<int Q_BLOCK_SIZE, int KV_BLOCK_SIZE, int HEAD_DIM>
__global__ void omni_attn_cp_async_kernel(
    half *__restrict__ Q,
    half *__restrict__ K,
    half *__restrict__ V,
    half *__restrict__ O,
    const int *__restrict__ kv_num_blocks,
    const int *__restrict__ kv_indices,
    const int *__restrict__ block_mask_types,
    const int *__restrict__ partial_block_mask_indices,
    const bool *__restrict__ partial_block_masks,
    int seqlen,
    int seqlen_orig,
    int heads,
    int batch,
    int num_q_blocks,
    int max_kv_blocks,
    bool has_partial_masks) {
  
  // Each block processes one Q block
  // grid(div_ceil(seqlen, Q_BLOCK_SIZE), batch * heads)
  const int batch_id = blockIdx.y / heads;
  const int head_id = blockIdx.y % heads;
  const int q_block = blockIdx.x;
  const int tid = threadIdx.x;
  
  const int q_block_start = q_block * Q_BLOCK_SIZE;
  const int q_block_end = min(q_block_start + Q_BLOCK_SIZE, seqlen);
  const int q_block_size = q_block_end - q_block_start;
  
  // If this Q block is entirely padding, output zeros and return
  if (q_block_start >= seqlen_orig) {
    const int qkv_offset = (batch_id * heads + head_id) * seqlen * HEAD_DIM;
    for (int q = 0; q < q_block_size; q++) {
      int q_pos = q_block_start + q;
      if (q_pos < seqlen) {
        int q_row_offset = qkv_offset + q_pos * HEAD_DIM;
        for (int d = 0; d < HEAD_DIM; d++) {
          O[q_row_offset + d] = __float2half(0.0f);
        }
      }
    }
    return;
  }
  
  // Shared memory for Q, K, V tiles
  extern __shared__ half smem[];
  half *Q_tile = smem;
  half *K_tile = Q_tile + Q_BLOCK_SIZE * HEAD_DIM;
  half *V_tile = K_tile + KV_BLOCK_SIZE * HEAD_DIM;
  
  const int qkv_offset = (batch_id * heads + head_id) * seqlen * HEAD_DIM;
  
  // Compute base offsets for block mask tensors (assuming C-contiguous layout)
  // kv_num_blocks: [batch, heads, num_q_blocks]
  const int kv_num_blocks_base = (batch_id * heads + head_id) * num_q_blocks + q_block;
  
  // kv_indices and block_mask_types: [batch, heads, num_q_blocks, max_kv_blocks]
  const int mask_base = ((batch_id * heads + head_id) * num_q_blocks + q_block) * max_kv_blocks;
  
  // Load Q block into shared memory using cp.async
  {
    const int num_threads = blockDim.x;
    const int elements_per_thread = (Q_BLOCK_SIZE * HEAD_DIM + num_threads - 1) / num_threads;
    
    for (int i = 0; i < elements_per_thread; i += 8) {
      int idx = tid * elements_per_thread + i;
      if (idx < Q_BLOCK_SIZE * HEAD_DIM) {
        int q_row = idx / HEAD_DIM;
        int q_col = idx % HEAD_DIM;
        int q_pos = q_block_start + q_row;
        if (q_pos < q_block_end) {
          uint32_t smem_ptr = __cvta_generic_to_shared(Q_tile + idx * sizeof(half));
          int gmem_idx = qkv_offset + q_pos * HEAD_DIM + q_col;
          CP_ASYNC_CG(smem_ptr, &Q[gmem_idx], 16);
        }
      }
    }
    CP_ASYNC_COMMIT_GROUP();
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }
  
  // Get number of active KV blocks for this Q block
  int kv_nb = kv_num_blocks[kv_num_blocks_base];
  
  // Each thread processes one Q row (for correctness and simplicity)
  int q = tid;
  bool valid_q = (q < q_block_size);
  int q_pos = valid_q ? (q_block_start + q) : 0;
  
  // Online softmax state (per thread, in registers)
  float row_max = -INFINITY;
  float row_sum = 0.0f;
  float row_out[HEAD_DIM];
  for (int d = 0; d < HEAD_DIM; d++) {
    row_out[d] = 0.0f;
  }
  
  const float scale = 1.0f / sqrtf((float)HEAD_DIM);
  
  if (!valid_q) {
    return;
  }
  
  // Process each active KV block
  for (int kv_idx = 0; kv_idx < kv_nb && kv_idx < max_kv_blocks; kv_idx++) {
    int idx = mask_base + kv_idx;
    int kv_block = kv_indices[idx];
    int mask_type = block_mask_types[idx];
    
    if (mask_type == BLOCK_MASK_MASKED) {
      continue;
    }
    
    int partial_block_index = -1;
    if (mask_type == BLOCK_MASK_PARTIAL) {
      if (!has_partial_masks || partial_block_mask_indices == nullptr) {
        continue;
      }
      partial_block_index = partial_block_mask_indices[idx];
      if (partial_block_index < 0) {
        continue;
      }
    }
    
    int kv_block_start = kv_block * KV_BLOCK_SIZE;
    int kv_block_end = min(kv_block_start + KV_BLOCK_SIZE, seqlen_orig);
    int kv_block_size = kv_block_end - kv_block_start;
    
    // Load K block into shared memory using cp.async
    {
      const int num_threads = blockDim.x;
      const int elements_per_thread = (KV_BLOCK_SIZE * HEAD_DIM + num_threads - 1) / num_threads;
      
      for (int i = 0; i < elements_per_thread; i += 8) {
        int idx = tid * elements_per_thread + i;
        if (idx < KV_BLOCK_SIZE * HEAD_DIM) {
          int kv_row = idx / HEAD_DIM;
          int kv_col = idx % HEAD_DIM;
          int kv_pos = kv_block_start + kv_row;
          if (kv_pos < kv_block_end) {
            uint32_t smem_ptr = __cvta_generic_to_shared(K_tile + idx * sizeof(half));
            int gmem_idx = qkv_offset + kv_pos * HEAD_DIM + kv_col;
            CP_ASYNC_CG(smem_ptr, &K[gmem_idx], 16);
          }
        }
      }
      CP_ASYNC_COMMIT_GROUP();
      CP_ASYNC_WAIT_GROUP(0);
      __syncthreads();
    }
    
    float block_max = -INFINITY;
    for (int kv = 0; kv < kv_block_size; kv++) {
      int kv_pos = kv_block_start + kv;
      
      if (mask_type == BLOCK_MASK_CAUSAL && q_pos < kv_pos) {
        continue;
      }
      
      if (mask_type == BLOCK_MASK_PARTIAL) {
        int local_q = q_pos - q_block_start;
        int local_kv = kv;
        if (local_q < 0 || local_q >= Q_BLOCK_SIZE || local_kv < 0 || local_kv >= KV_BLOCK_SIZE) {
          continue;
        }
        const int block_area = Q_BLOCK_SIZE * KV_BLOCK_SIZE;
        int mask_offset = partial_block_index * block_area + local_q * KV_BLOCK_SIZE + local_kv;
        if (partial_block_masks == nullptr || !partial_block_masks[mask_offset]) {
          continue;
        }
      }
      
      float score = 0.0f;
      for (int d = 0; d < HEAD_DIM; d++) {
        float q_val = __half2float(Q_tile[q * HEAD_DIM + d]);
        float k_val = __half2float(K_tile[kv * HEAD_DIM + d]);
        score += q_val * k_val;
      }
      score *= scale;
      block_max = max(block_max, score);
    }
    
    // Update global max and rescale previous output
    float old_max = row_max;
    float new_max = max(old_max, block_max);
    row_max = new_max;
    
    // Rescale previous output if max increased
    if (new_max > old_max) {
      float rescale_factor = expf(old_max - new_max);
      row_sum *= rescale_factor;
      for (int d = 0; d < HEAD_DIM; d++) {
        row_out[d] *= rescale_factor;
      }
    }
    
    // Load V block
    {
      const int num_threads = blockDim.x;
      const int elements_per_thread = (KV_BLOCK_SIZE * HEAD_DIM + num_threads - 1) / num_threads;
      
      for (int i = 0; i < elements_per_thread; i += 8) {
        int idx = tid * elements_per_thread + i;
        if (idx < KV_BLOCK_SIZE * HEAD_DIM) {
          int kv_row = idx / HEAD_DIM;
          int kv_col = idx % HEAD_DIM;
          int kv_pos = kv_block_start + kv_row;
          if (kv_pos < kv_block_end) {
            uint32_t smem_ptr = __cvta_generic_to_shared(V_tile + idx * sizeof(half));
            int gmem_idx = qkv_offset + kv_pos * HEAD_DIM + kv_col;
            CP_ASYNC_CG(smem_ptr, &V[gmem_idx], 16);
          }
        }
      }
      CP_ASYNC_COMMIT_GROUP();
      CP_ASYNC_WAIT_GROUP(0);
      __syncthreads();
    }
    
    float new_sum = 0.0f;
    for (int kv = 0; kv < kv_block_size; kv++) {
      int kv_pos = kv_block_start + kv;
      
      if (mask_type == BLOCK_MASK_CAUSAL && q_pos < kv_pos) {
        continue;
      }
      
      if (mask_type == BLOCK_MASK_PARTIAL) {
        int local_q = q_pos - q_block_start;
        int local_kv = kv;
        if (local_q < 0 || local_q >= Q_BLOCK_SIZE || local_kv < 0 || local_kv >= KV_BLOCK_SIZE) {
          continue;
        }
        const int block_area = Q_BLOCK_SIZE * KV_BLOCK_SIZE;
        int mask_offset = partial_block_index * block_area + local_q * KV_BLOCK_SIZE + local_kv;
        if (partial_block_masks == nullptr || !partial_block_masks[mask_offset]) {
          continue;
        }
      }
      
      float score = 0.0f;
      for (int d = 0; d < HEAD_DIM; d++) {
        float q_val = __half2float(Q_tile[q * HEAD_DIM + d]);
        float k_val = __half2float(K_tile[kv * HEAD_DIM + d]);
        score += q_val * k_val;
      }
      score *= scale;
      
      float exp_score = expf(score - row_max);
      new_sum += exp_score;
      
      for (int d = 0; d < HEAD_DIM; d++) {
        float v_val = __half2float(V_tile[kv * HEAD_DIM + d]);
        row_out[d] += exp_score * v_val;
      }
    }
    
    // Update row_sum
    row_sum += new_sum;
  }
  
  // Final normalization and write output
  __syncthreads();
  {
    // Each thread writes its Q row
    float inv_sum = (row_sum > 0.0f) ? (1.0f / row_sum) : 0.0f;
    for (int d = 0; d < HEAD_DIM; d++) {
      float out_val = row_out[d] * inv_sum;
      O[qkv_offset + q_pos * HEAD_DIM + d] = __float2half(out_val);
    }
  }
}

void omni_attn_cp_async(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor kv_num_blocks, torch::Tensor kv_indices,
    torch::Tensor block_mask_types, int BLOCK_SIZE, int seqlen_orig,
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
  
  // Use BLOCK_SIZE from block mask (must match the BLOCK_SIZE used to create the mask)
  const int num_q_blocks = kv_num_blocks.size(2);
  const int max_kv_blocks = kv_indices.size(3);
  
  // Validate BLOCK_SIZE matches expected value
  const int expected_num_q_blocks = (seqlen + BLOCK_SIZE - 1) / BLOCK_SIZE;
  if (num_q_blocks != expected_num_q_blocks) {
    throw std::runtime_error(
        "Block mask num_q_blocks (" + std::to_string(num_q_blocks) + 
        ") does not match expected value (" + std::to_string(expected_num_q_blocks) + 
        ") based on seqlen=" + std::to_string(seqlen) + " and BLOCK_SIZE=" + 
        std::to_string(BLOCK_SIZE));
  }
  
  const int Q_BLOCK_SIZE = BLOCK_SIZE;
  const int KV_BLOCK_SIZE = BLOCK_SIZE; // Assume same for now (as per user's assumption)
  
  // Ensure tensors are contiguous for simpler indexing
  // (This is a common requirement and simplifies the kernel significantly)
  if (!kv_num_blocks.is_contiguous() || !kv_indices.is_contiguous() || 
      !block_mask_types.is_contiguous()) {
    throw std::runtime_error(
        "Block mask tensors must be contiguous. Call .contiguous() before calling this function.");
  }
  
  const bool has_partial_masks = has_partial;
  if (has_partial_masks) {
    if (!partial_block_mask_indices.is_contiguous() || !partial_block_masks.is_contiguous()) {
      throw std::runtime_error("Partial block mask tensors must be contiguous.");
    }
    CHECK_TORCH_TENSOR_DTYPE(partial_block_mask_indices, torch::kInt32)
    CHECK_TORCH_TENSOR_DTYPE(partial_block_masks, torch::kBool)
  }
  
  const int *partial_idx_ptr = has_partial_masks
                                   ? reinterpret_cast<int *>(partial_block_mask_indices.data_ptr())
                                   : nullptr;
  const bool *partial_masks_ptr = has_partial_masks
                                      ? reinterpret_cast<bool *>(partial_block_masks.data_ptr())
                                      : nullptr;
  
  // Validate head_dim
  if (head_dim != 32 && head_dim != 64 && head_dim != 128) {
    throw std::runtime_error(
        "Unsupported head_dim=" + std::to_string(head_dim) +
        ". Supported values: 32, 64, 128");
  }
  
  // Validate BLOCK_SIZE
  if (Q_BLOCK_SIZE != 64 && Q_BLOCK_SIZE != 128) {
    throw std::runtime_error(
        "Unsupported Q_BLOCK_SIZE=" + std::to_string(Q_BLOCK_SIZE) +
        ". Supported values: 64, 128");
  }
  
  // Launch configuration
  // Need at least Q_BLOCK_SIZE threads (one per Q row), round up to warp size
  const int num_threads = ((Q_BLOCK_SIZE + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE;
  dim3 grid(div_ceil(seqlen, Q_BLOCK_SIZE), batch * heads);
  dim3 block(num_threads);
  
  // Shared memory size: Q_tile + K_tile + V_tile
  const int smem_size = (Q_BLOCK_SIZE * head_dim + 
                         KV_BLOCK_SIZE * head_dim + 
                         KV_BLOCK_SIZE * head_dim) * sizeof(half);
  
  // Launch kernel based on template parameters
  if (Q_BLOCK_SIZE == 64 && KV_BLOCK_SIZE == 64) {
    if (head_dim == 32) {
      omni_attn_cp_async_kernel<64, 64, 32><<<grid, block, smem_size>>>(
          reinterpret_cast<half *>(Q.data_ptr()),
          reinterpret_cast<half *>(K.data_ptr()),
          reinterpret_cast<half *>(V.data_ptr()),
          reinterpret_cast<half *>(O.data_ptr()),
          reinterpret_cast<int *>(kv_num_blocks.data_ptr()),
          reinterpret_cast<int *>(kv_indices.data_ptr()),
          reinterpret_cast<int *>(block_mask_types.data_ptr()),
          partial_idx_ptr, partial_masks_ptr,
          seqlen, seqlen_orig, heads, batch, num_q_blocks, max_kv_blocks, has_partial_masks);
    } else if (head_dim == 64) {
      omni_attn_cp_async_kernel<64, 64, 64><<<grid, block, smem_size>>>(
          reinterpret_cast<half *>(Q.data_ptr()),
          reinterpret_cast<half *>(K.data_ptr()),
          reinterpret_cast<half *>(V.data_ptr()),
          reinterpret_cast<half *>(O.data_ptr()),
          reinterpret_cast<int *>(kv_num_blocks.data_ptr()),
          reinterpret_cast<int *>(kv_indices.data_ptr()),
          reinterpret_cast<int *>(block_mask_types.data_ptr()),
          partial_idx_ptr, partial_masks_ptr,
          seqlen, seqlen_orig, heads, batch, num_q_blocks, max_kv_blocks, has_partial_masks);
    } else if (head_dim == 128) {
      omni_attn_cp_async_kernel<64, 64, 128><<<grid, block, smem_size>>>(
          reinterpret_cast<half *>(Q.data_ptr()),
          reinterpret_cast<half *>(K.data_ptr()),
          reinterpret_cast<half *>(V.data_ptr()),
          reinterpret_cast<half *>(O.data_ptr()),
          reinterpret_cast<int *>(kv_num_blocks.data_ptr()),
          reinterpret_cast<int *>(kv_indices.data_ptr()),
          reinterpret_cast<int *>(block_mask_types.data_ptr()),
          partial_idx_ptr, partial_masks_ptr,
          seqlen, seqlen_orig, heads, batch, num_q_blocks, max_kv_blocks, has_partial_masks);
    }
  } else if (Q_BLOCK_SIZE == 128 && KV_BLOCK_SIZE == 128) {
    if (head_dim == 32) {
      omni_attn_cp_async_kernel<128, 128, 32><<<grid, block, smem_size>>>(
          reinterpret_cast<half *>(Q.data_ptr()),
          reinterpret_cast<half *>(K.data_ptr()),
          reinterpret_cast<half *>(V.data_ptr()),
          reinterpret_cast<half *>(O.data_ptr()),
          reinterpret_cast<int *>(kv_num_blocks.data_ptr()),
          reinterpret_cast<int *>(kv_indices.data_ptr()),
          reinterpret_cast<int *>(block_mask_types.data_ptr()),
          partial_idx_ptr, partial_masks_ptr,
          seqlen, seqlen_orig, heads, batch, num_q_blocks, max_kv_blocks, has_partial_masks);
    } else if (head_dim == 64) {
      omni_attn_cp_async_kernel<128, 128, 64><<<grid, block, smem_size>>>(
          reinterpret_cast<half *>(Q.data_ptr()),
          reinterpret_cast<half *>(K.data_ptr()),
          reinterpret_cast<half *>(V.data_ptr()),
          reinterpret_cast<half *>(O.data_ptr()),
          reinterpret_cast<int *>(kv_num_blocks.data_ptr()),
          reinterpret_cast<int *>(kv_indices.data_ptr()),
          reinterpret_cast<int *>(block_mask_types.data_ptr()),
          partial_idx_ptr, partial_masks_ptr,
          seqlen, seqlen_orig, heads, batch, num_q_blocks, max_kv_blocks, has_partial_masks);
    } else if (head_dim == 128) {
      omni_attn_cp_async_kernel<128, 128, 128><<<grid, block, smem_size>>>(
          reinterpret_cast<half *>(Q.data_ptr()),
          reinterpret_cast<half *>(K.data_ptr()),
          reinterpret_cast<half *>(V.data_ptr()),
          reinterpret_cast<half *>(O.data_ptr()),
          reinterpret_cast<int *>(kv_num_blocks.data_ptr()),
          reinterpret_cast<int *>(kv_indices.data_ptr()),
          reinterpret_cast<int *>(block_mask_types.data_ptr()),
          partial_idx_ptr, partial_masks_ptr,
          seqlen, seqlen_orig, heads, batch, num_q_blocks, max_kv_blocks, has_partial_masks);
    }
  } else {
    throw std::runtime_error(
        "Unsupported BLOCK_SIZE combination: Q_BLOCK_SIZE=" + std::to_string(Q_BLOCK_SIZE) +
        ", KV_BLOCK_SIZE=" + std::to_string(KV_BLOCK_SIZE));
  }
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
  }
}