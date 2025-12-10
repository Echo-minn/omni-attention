# Omni-Attention Kernel Design: Architecture and Implementation

## 1. Debug Data Preparation

The debug data generation pipeline serves as the foundation for correctness verification and performance benchmarking. Our approach systematically generates deterministic test cases with varying sparsity patterns to validate kernel behavior across different attention mask configurations.

### Data Generation Strategy

The debug data preparation process (`generate_debug_data.py`) employs a multi-stage pipeline that produces self-contained test artifacts. For each test pattern, we generate:

1. **Input Tensors**: Random Q, K, V tensors with fixed seeds for reproducibility, stored in both FP16 (kernel input) and FP32 (reference computation) formats.

2. **Mask Representations**: We construct masks at two levels of granularity:
   - **Dense masks**: Full `[B, H, seq_len, seq_len]` boolean tensors for reference computation
   - **Block-sparse masks**: Compressed `OmniBlockMask` structures encoding sparsity patterns

3. **Reference Outputs**: Ground-truth attention outputs computed via naive PyTorch implementation using dense masks, enabling numerical verification of kernel correctness.

4. **Metadata**: Configuration parameters (batch size, sequence length, block sizes, seed values) are embedded within each debug artifact to ensure test reproducibility.

### Pattern Variants

The pipeline generates three distinct mask pattern classes:

- **Causal patterns**: Standard autoregressive attention where each query attends only to preceding keys
- **Full/Causal mixtures**: Random combinations of FULL and CAUSAL block types to simulate heterogeneous attention patterns
- **Full/Causal/Partial mixtures**: Complex patterns including PARTIAL blocks that require per-token mask lookups

Each pattern variant is serialized to disk as a PyTorch checkpoint, enabling offline correctness testing and performance profiling without regenerating inputs.

## 2. Mask Presentation and Overhead Reduction

### Block-Sparse Mask Format

To address the computational overhead of irregular attention patterns, we adopt a block-sparse mask representation that aligns with GPU memory hierarchy and tiling strategies. The `OmniBlockMask` data structure encodes sparsity at block granularity rather than per-token, dramatically reducing metadata size and enabling efficient kernel-level skipping of masked regions.

The mask format consists of three core tensors:

- **`kv_num_blocks`**: `[B, H, num_q_blocks]` - Number of active KV blocks per Q block
- **`kv_indices`**: `[B, H, num_q_blocks, max_blocks]` - Indices of active KV blocks for each Q block
- **`block_mask_types`**: `[B, H, num_q_blocks, max_blocks]` - Mask type per block (FULL, CAUSAL, PARTIAL, MASKED)

### Mask Type Semantics

Each block is classified into one of four categories:

1. **FULL**: No masking applied within the block; all Q-K pairs are computed
2. **CAUSAL**: Causal masking applied; only `q_idx >= kv_idx` pairs are valid
3. **PARTIAL**: Per-token masking required; uses auxiliary dense bitmasks stored separately
4. **MASKED**: Block entirely skipped; no computation or memory access

This classification enables kernel specialization: FULL blocks use optimized dense matrix multiplication paths, CAUSAL blocks apply efficient triangular masking, and MASKED blocks are completely elided from computation.

### Overhead Reduction Techniques

**Block Alignment**: Sequences are padded to multiples of `Q_BLOCK_SIZE` and `KV_BLOCK_SIZE` (typically 64 or 128), ensuring that mask boundaries align with kernel tile boundaries. This eliminates the need for per-token boundary checks within the inner computation loop.

**Compressed Partial Masks**: For PARTIAL blocks, we employ a two-level indirection scheme. Rather than storing dense masks for all blocks, we maintain:
- `partial_block_mask_indices`: Sparse index mapping from block coordinates to a compact mask pool
- `partial_block_masks`: A deduplicated tensor of `[num_partial_blocks, Q_BLOCK_SIZE, KV_BLOCK_SIZE]` boolean masks

This design reduces memory overhead when multiple blocks share identical mask patterns (common in interleaved text-image sequences).

**Early Block Skipping**: The kernel iterates only over active blocks as specified by `kv_indices`, completely bypassing masked regions. This reduces both computation and memory bandwidth, with effectiveness proportional to mask sparsity.

**Shared KV Buffers**: For Q blocks that attend to overlapping KV regions, we reuse shared memory buffers across multiple Q tiles, reducing redundant global memory loads. This optimization is particularly effective for causal attention patterns where adjacent Q blocks share significant KV overlap.

## 3. Technology Stack and Framework

### High-Level Architecture

The implementation follows a hybrid Python-CUDA architecture that separates interface logic from performance-critical computation:

**Python Layer** (`omni_attn_torch.py`):
- PyTorch tensor management and shape validation
- Block mask construction and conversion utilities
- Reference implementations for correctness verification
- Kernel dispatch logic with automatic fallback mechanisms

**CUDA Kernel Layer** (`mma/omni_attn_*.cu`):
- Hand-tuned attention kernels using CUDA C++
- Direct hardware-level optimizations (MMA instructions, shared memory management)
- Multiple kernel variants targeting different performance/feature tradeoffs

**Binding Layer** (`setup.py`):
- PyBind11 integration for seamless Python-CUDA interop
- Automatic compilation and module loading
- Type conversion and memory layout management

### Parallel Computing Techniques

**FlashAttention-2 Split-Q Strategy**: We adopt FlashAttention-2's tiling approach where Q is split into row tiles (`Br = 64` or `128`) while K/V are processed in column tiles (`Bc = 64` or `128`). This enables online softmax computation without materializing the full attention matrix, reducing memory complexity from O(N²) to O(N).

**Tensor Core Matrix Multiply-Accumulate (MMA)**: The kernels leverage NVIDIA's Tensor Cores via `HMMA16816` instructions for FP16 matrix multiplication. Each warp computes `16×8` output tiles using `16×16×8` MMA operations, achieving high arithmetic intensity.

**Asynchronous Memory Operations**: We employ `cp.async` instructions to pipeline global memory loads with computation. K and V tiles are prefetched into shared memory while previous tiles are being processed, hiding memory latency.

**Multi-Stage Pipeline**: The kernel supports 1-stage and 2-stage execution modes. In 2-stage mode, K/V tiles are double-buffered, allowing computation and memory transfer to overlap more effectively at the cost of increased shared memory usage.

**Shared Memory Optimization**: 
- Padding (`kPadQ`, `kPadK`, `kPadV`) is applied to avoid bank conflicts in shared memory
- Layout swizzling is used in advanced kernels to further reduce bank conflicts
- Shared memory is reused across Q tiles when KV blocks overlap

**Warp-Level Parallelism**: Each thread block contains multiple warps (typically 4-8) that cooperatively process Q tiles. Warps are assigned to different Q rows, enabling fine-grained parallelism while maintaining data locality.

### Framework Integration

The kernels integrate with PyTorch's autograd system through custom autograd functions, enabling end-to-end training workflows. For correctness validation, we provide compatibility with PyTorch's FlexAttention API, allowing direct comparison against reference implementations.

The build system uses standard CUDA compilation toolchains (nvcc) with compile-time template specialization for different head dimensions (32, 64, 128) and block sizes, generating optimized kernel variants without runtime dispatch overhead.

