#include "utils.h"

// Block mask type definitions
#define BLOCK_MASK_MASKED 0
#define BLOCK_MASK_CAUSAL 1
#define BLOCK_MASK_FULL 2
#define BLOCK_MASK_PARTIAL 3  // Requires per-token dense mask check

// -----------------------------------------------------------------------------
// Simple, correctness-first CUDA kernel (no MMA, no cp.async)
//
// This kernel is intentionally straightforward:
// - One thread computes a full output row [head_dim] for a single (batch, head, q)
// - It walks the block mask (kv_num_blocks / kv_indices / block_mask_types)
// - It recomputes QK^T twice (for max and for sum/weighted value) in fp32
// - It applies FULL / CAUSAL / MASKED semantics to match OmniBlockMask.to_score_mask()
//
// This is NOT performance-oriented, but is much easier to reason about and
// serves as a correctness baseline before enabling the optimized MMA kernel.
// -----------------------------------------------------------------------------
__global__ void omni_attn_simple_kernel_impl(
    half *__restrict__ Q,
    half *__restrict__ K,
    half *__restrict__ V,
    half *__restrict__ O,
    const int *__restrict__ kv_num_blocks,
    const int *__restrict__ kv_indices,
    const int *__restrict__ block_mask_types,
    const bool *__restrict__ dense_mask,  // Optional: per-token mask for PARTIAL blocks [B, H, Q, KV]
    int QKV_seqlen,
    int QKV_seqlen_orig,  // Original sequence length (before padding)
    int QKV_head,
    int QKV_batch,
    int num_q_blocks,
    int max_blocks,
    int BLOCK_SIZE,
    int kv_num_blocks_stride0,
    int kv_num_blocks_stride1,
    int kv_num_blocks_stride2,
    int kv_indices_stride0,
    int kv_indices_stride1,
    int kv_indices_stride2,
    int kv_indices_stride3,
    int block_mask_types_stride0,
    int block_mask_types_stride1,
    int block_mask_types_stride2,
    int block_mask_types_stride3,
    int dense_mask_stride0,  // Strides for dense_mask
    int dense_mask_stride1,
    int dense_mask_stride2,
    int dense_mask_stride3,
    bool has_dense_mask,  // Whether dense_mask is provided (non-null)
    int head_dim) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int total_q = QKV_batch * QKV_head * QKV_seqlen;
  if (tid >= total_q) {
    return;
  }

  // Map linear thread id -> (batch_id, head_id, q_idx)
  int q_idx = tid % QKV_seqlen;
  int tmp = tid / QKV_seqlen;
  int head_id = tmp % QKV_head;
  int batch_id = tmp / QKV_head;
  
  // If this is a padding position, output zeros and return
  if (q_idx >= QKV_seqlen_orig) {
    const int qkv_offset = (batch_id * QKV_head + head_id) * QKV_seqlen * head_dim;
    const int q_row_offset = qkv_offset + q_idx * head_dim;
    for (int d = 0; d < head_dim; ++d) {
      O[q_row_offset + d] = __float2half(0.0f);
    }
    return;
  }

  // Use BLOCK_SIZE directly (must match the BLOCK_SIZE used to create the mask)
  int q_block = q_idx / BLOCK_SIZE;
  if (q_block >= num_q_blocks) {
    q_block = num_q_blocks - 1;
  }

  int kv_nb = kv_num_blocks[batch_id * kv_num_blocks_stride0 +
                            head_id * kv_num_blocks_stride1 +
                            q_block * kv_num_blocks_stride2];

  const int qkv_offset =
      (batch_id * QKV_head + head_id) * QKV_seqlen * head_dim;
  const int q_row_offset = qkv_offset + q_idx * head_dim;

  const float scale = 1.0f / sqrtf((float)head_dim);

  // ---------------------------------------------------------------------------
  // Pass 1: find row max over all valid KV positions (respecting block mask)
  // ---------------------------------------------------------------------------
  float max_score = -INFINITY;

  for (int kv_idx = 0; kv_idx < kv_nb; ++kv_idx) {
    int indices_off = batch_id * kv_indices_stride0 +
                      head_id * kv_indices_stride1 +
                      q_block * kv_indices_stride2 +
                      kv_idx * kv_indices_stride3;
    int kv_block = kv_indices[indices_off];

    int mask_type = block_mask_types[batch_id * block_mask_types_stride0 +
                                     head_id * block_mask_types_stride1 +
                                     q_block * block_mask_types_stride2 +
                                     kv_idx * block_mask_types_stride3];


    if (mask_type == BLOCK_MASK_MASKED) {
      continue;
    }

    int kv_block_start = kv_block * BLOCK_SIZE;
    int kv_block_end = min(kv_block_start + BLOCK_SIZE, QKV_seqlen_orig);  // Bound by original length

    int kv_start = kv_block_start;
    int kv_end = kv_block_end;

    // CAUSAL semantics: within this block, only attend to positions <= q_idx.
    if (mask_type == BLOCK_MASK_CAUSAL) {
      if (q_idx < kv_block_start) {
        // For this row, the whole block is in the future.
        continue;
      }
      kv_end = min(kv_end, q_idx + 1);
    }
    
    // PARTIAL blocks require per-token dense mask check
    // If dense_mask is not provided, skip PARTIAL blocks (error case)
    if (mask_type == BLOCK_MASK_PARTIAL && !has_dense_mask) {
      continue;  // Skip if no dense mask provided
    }

    for (int kv = kv_start; kv < kv_end; ++kv) {
      // For PARTIAL blocks, check dense mask per-token
      if (mask_type == BLOCK_MASK_PARTIAL) {
        int dense_mask_idx = batch_id * dense_mask_stride0 +
                            head_id * dense_mask_stride1 +
                            q_idx * dense_mask_stride2 +
                            kv * dense_mask_stride3;
        if (!dense_mask[dense_mask_idx]) {
          continue;  // Skip masked positions in PARTIAL blocks
        }
      }
      
      // Explicit per-position causal check (defensive programming)
      // For CAUSAL blocks, ensure we only process kv <= q_idx
      if (mask_type == BLOCK_MASK_CAUSAL && kv > q_idx) {
        continue;
      }
      
      int k_row_offset = qkv_offset + kv * head_dim;
      float score = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        float qv = __half2float(Q[q_row_offset + d]);
        float kvv = __half2float(K[k_row_offset + d]);
        score += qv * kvv;
      }
      score *= scale;
      
      if (score > max_score) {
        max_score = score;
      }
    }
  }

  // If everything is masked, emit zeros.
  if (max_score == -INFINITY) {
    for (int d = 0; d < head_dim; ++d) {
      O[q_row_offset + d] = __float2half(0.0f);
    }
    return;
  }

  // ---------------------------------------------------------------------------
  // Pass 2: compute softmax and P@V in fp32, then normalize
  // ---------------------------------------------------------------------------
  float sum_exp = 0.0f;
  constexpr int kMaxHeadDim = 256;
  float out[kMaxHeadDim];
  for (int d = 0; d < head_dim; ++d) {
    out[d] = 0.0f;
  }

  for (int kv_idx = 0; kv_idx < kv_nb; ++kv_idx) {
    int indices_off = batch_id * kv_indices_stride0 +
                      head_id * kv_indices_stride1 +
                      q_block * kv_indices_stride2 +
                      kv_idx * kv_indices_stride3;
    int kv_block = kv_indices[indices_off];

    int mask_type = block_mask_types[batch_id * block_mask_types_stride0 +
                                     head_id * block_mask_types_stride1 +
                                     q_block * block_mask_types_stride2 +
                                     kv_idx * block_mask_types_stride3];



    if (mask_type == BLOCK_MASK_MASKED) {
      continue;
    }

    int kv_block_start = kv_block * BLOCK_SIZE;
    int kv_block_end = min(kv_block_start + BLOCK_SIZE, QKV_seqlen_orig);  // Bound by original length

    int kv_start = kv_block_start;
    int kv_end = kv_block_end;

    if (mask_type == BLOCK_MASK_CAUSAL) {
      if (q_idx < kv_block_start) {
        continue;
      }
      kv_end = min(kv_end, q_idx + 1);
    }
    
    // PARTIAL blocks require per-token dense mask check
    if (mask_type == BLOCK_MASK_PARTIAL && !has_dense_mask) {
      continue;  // Skip if no dense mask provided
    }

    for (int kv = kv_start; kv < kv_end; ++kv) {
      // For PARTIAL blocks, check dense mask per-token
      if (mask_type == BLOCK_MASK_PARTIAL) {
        int dense_mask_idx = batch_id * dense_mask_stride0 +
                            head_id * dense_mask_stride1 +
                            q_idx * dense_mask_stride2 +
                            kv * dense_mask_stride3;
        if (!dense_mask[dense_mask_idx]) {
          continue;  // Skip masked positions in PARTIAL blocks
        }
      }
      
      // Explicit per-position causal check (defensive programming)
      // For CAUSAL blocks, ensure we only process kv <= q_idx
      if (mask_type == BLOCK_MASK_CAUSAL && kv > q_idx) {
        continue;
      }
      
      int k_row_offset = qkv_offset + kv * head_dim;
      float score = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        float qv = __half2float(Q[q_row_offset + d]);
        float kvv = __half2float(K[k_row_offset + d]);
        score += qv * kvv;
      }
      score *= scale;
      float score_diff = score - max_score;
      float w = __expf(score_diff);
      sum_exp += w;
      

      for (int d = 0; d < head_dim; ++d) {
        float vv = __half2float(V[k_row_offset + d]);
        out[d] += w * vv;
      }
      
    }
  }

  if (sum_exp <= 0.0f) {
    for (int d = 0; d < head_dim; ++d) {
      O[q_row_offset + d] = __float2half(0.0f);
    }
    return;
  }

  float inv_sum = 1.0f / sum_exp;

  for (int d = 0; d < head_dim; ++d) {
    float final_val = out[d] * inv_sum;
    O[q_row_offset + d] = __float2half(final_val);
  }
}

// Wrapper function to launch the simple kernel
void omni_attn_simple_kernel(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor kv_num_blocks, torch::Tensor kv_indices,
    torch::Tensor block_mask_types, int BLOCK_SIZE, int seqlen_orig,
    torch::Tensor dense_mask = torch::Tensor()) {  // Optional dense mask for PARTIAL blocks
  
  CHECK_TORCH_TENSOR_DTYPE(Q, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(K, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(V, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(O, torch::kHalf)
  
  const int batch = Q.size(0);
  const int heads = Q.size(1);
  const int seqlen = Q.size(2);
  const int head_dim = Q.size(3);
  
  const int num_q_blocks = kv_num_blocks.size(2);
  const int max_blocks = kv_indices.size(3);
  
  // Validate BLOCK_SIZE matches expected value
  const int expected_num_q_blocks = (seqlen + BLOCK_SIZE - 1) / BLOCK_SIZE;
  if (num_q_blocks != expected_num_q_blocks) {
    throw std::runtime_error(
        "Block mask num_q_blocks (" + std::to_string(num_q_blocks) + 
        ") does not match expected value (" + std::to_string(expected_num_q_blocks) + 
        ") based on seqlen=" + std::to_string(seqlen) + " and BLOCK_SIZE=" + 
        std::to_string(BLOCK_SIZE));
  }
  
  // Ensure tensors are contiguous
  if (!Q.is_contiguous() || !K.is_contiguous() || !V.is_contiguous() || !O.is_contiguous()) {
    throw std::runtime_error(
        "Q, K, V, O tensors must be contiguous. Call .contiguous() before calling this function.");
  }
  if (!kv_num_blocks.is_contiguous() || !kv_indices.is_contiguous() || 
      !block_mask_types.is_contiguous()) {
    throw std::runtime_error(
        "Block mask tensors must be contiguous. Call .contiguous() before calling this function.");
  }
  
  // Check if dense_mask is provided and valid
  bool has_dense_mask = dense_mask.defined() && dense_mask.numel() > 0;
  if (has_dense_mask) {
    CHECK_TORCH_TENSOR_DTYPE(dense_mask, torch::kBool)
    if (dense_mask.dim() != 4 || dense_mask.size(0) != batch || 
        dense_mask.size(1) != heads || dense_mask.size(2) != seqlen ||
        dense_mask.size(3) != seqlen) {
      throw std::runtime_error(
          "dense_mask must have shape [batch, heads, seqlen, seqlen] matching Q/K/V");
    }
    if (!dense_mask.is_contiguous()) {
      throw std::runtime_error("dense_mask must be contiguous");
    }
  }
  
  const int total_q = batch * heads * seqlen;
  const int threads = 128;
  const int blocks = div_ceil(total_q, threads);
  
  // Get dense_mask data pointer and strides (or nullptr if not provided)
  const bool *dense_mask_ptr = has_dense_mask ? 
      reinterpret_cast<const bool *>(dense_mask.data_ptr()) : nullptr;
  int dense_mask_stride0 = has_dense_mask ? dense_mask.stride(0) : 0;
  int dense_mask_stride1 = has_dense_mask ? dense_mask.stride(1) : 0;
  int dense_mask_stride2 = has_dense_mask ? dense_mask.stride(2) : 0;
  int dense_mask_stride3 = has_dense_mask ? dense_mask.stride(3) : 0;
  
  omni_attn_simple_kernel_impl<<<blocks, threads>>>(
      reinterpret_cast<half *>(Q.data_ptr()),
      reinterpret_cast<half *>(K.data_ptr()),
      reinterpret_cast<half *>(V.data_ptr()),
      reinterpret_cast<half *>(O.data_ptr()),
      reinterpret_cast<int *>(kv_num_blocks.data_ptr()),
      reinterpret_cast<int *>(kv_indices.data_ptr()),
      reinterpret_cast<int *>(block_mask_types.data_ptr()),
      dense_mask_ptr,
      seqlen,
      seqlen_orig,
      heads,
      batch,
      num_q_blocks,
      max_blocks,
      BLOCK_SIZE,
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
      dense_mask_stride0,
      dense_mask_stride1,
      dense_mask_stride2,
      dense_mask_stride3,
      has_dense_mask,
      head_dim);
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA kernel launch failed: " + std::string(cudaGetErrorString(err)));
  }
}
