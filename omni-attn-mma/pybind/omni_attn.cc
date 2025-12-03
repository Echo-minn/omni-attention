#include <torch/extension.h>
#include <torch/types.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

// Basic

void omni_attn_simple_kernel(torch::Tensor Q, torch::Tensor K,
                                             torch::Tensor V, torch::Tensor O,
                                             torch::Tensor kv_num_blocks,
                                             torch::Tensor kv_indices,
                                             torch::Tensor block_mask_types,
                                             int BLOCK_SIZE,
                                             int seqlen_orig,
                                             torch::Tensor dense_mask = torch::Tensor());

void omni_attn_cp_async(torch::Tensor Q, torch::Tensor K,
                                             torch::Tensor V, torch::Tensor O,
                                             torch::Tensor kv_num_blocks,
                                             torch::Tensor kv_indices,
                                             torch::Tensor block_mask_types,
                                             int BLOCK_SIZE,
                                             int seqlen_orig);

void omni_attn_preftech_kernel(torch::Tensor Q, torch::Tensor K,
                                             torch::Tensor V, torch::Tensor O,
                                             torch::Tensor kv_num_blocks,
                                             torch::Tensor kv_indices,
                                             torch::Tensor block_mask_types);

void omni_attn_mma_stages_split_q_shared_kv(torch::Tensor Q, torch::Tensor K,
                                             torch::Tensor V, torch::Tensor O,
                                             torch::Tensor kv_num_blocks,
                                             torch::Tensor kv_indices,
                                             torch::Tensor block_mask_types,
                                             int stages,
                                             int BLOCK_SIZE);

void omni_attn_swizzle_kernel(torch::Tensor Q, torch::Tensor K,
                                             torch::Tensor V, torch::Tensor O,
                                             torch::Tensor kv_num_blocks,
                                             torch::Tensor kv_indices,
                                             torch::Tensor block_mask_types);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Basic
  TORCH_BINDING_COMMON_EXTENSION(omni_attn_mma_stages_split_q_shared_kv)
  TORCH_BINDING_COMMON_EXTENSION(omni_attn_simple_kernel)
  TORCH_BINDING_COMMON_EXTENSION(omni_attn_cp_async)
  TORCH_BINDING_COMMON_EXTENSION(omni_attn_preftech_kernel)
  TORCH_BINDING_COMMON_EXTENSION(omni_attn_swizzle_kernel)
}
