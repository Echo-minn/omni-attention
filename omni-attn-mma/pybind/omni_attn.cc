#include <torch/extension.h>
#include <torch/types.h>

#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

// Basic

void omni_attn_mma_stages_split_q_shared_kv(torch::Tensor Q, torch::Tensor K,
                                             torch::Tensor V, torch::Tensor O,
                                             torch::Tensor kv_num_blocks,
                                             torch::Tensor kv_indices,
                                             torch::Tensor block_mask_types,
                                             int stages);

void omni_attn_mma_stages_split_q_shared_kv_mma_only(
    torch::Tensor Q, torch::Tensor K, torch::Tensor V, torch::Tensor O,
    torch::Tensor kv_num_blocks, torch::Tensor kv_indices,
    torch::Tensor block_mask_types, int stages);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Basic
  TORCH_BINDING_COMMON_EXTENSION(omni_attn_mma_stages_split_q_shared_kv)
  TORCH_BINDING_COMMON_EXTENSION(omni_attn_mma_stages_split_q_shared_kv_mma_only)
}
