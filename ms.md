# Project Milestone Report: Attention Variant with Interleaved Masking for Cross-Modal VLMs

> Mira Xiao(minxiao@andrew.cmu.edu)
>
> GuanWei Wu(guanweiwu@andrew.cmu.edu)

## Summary of the project progress so far

For this project, our goal is to implement a flexible and high-performance omni attention kernel which support irregular interleaved masking patterns for unified vision language models. To evaluate its performance, I setup the correctness and proformance verification scripts which compared to naive attention and FlexAttention with different `Q,K,V,O` shapes.

First I setup the cuda kernel building environment and make sure it can be bind and called from python via pybind11. Then I try to start with write the standart MHA attention module in pytorch as baseline for the omni attention kernel. Then after reading papers and refer to the implementatio strategy in FlexAttention and FlashAttention, I got an idea from the LLM MoE model structure which take advantage of sparse block to skip masked parts and thus reducing the memory usage and computation overhead. So, with that in mind, I designed a Sparse Block Mask class which present the non-contiguous mask sparsity pattern and generate the kv-block indices and mask types for each q-block. Since the original mask presentaion of different modalities are of various lengths, I convert the mask data into same size blocks to align with the kernel tiling split strategy. After conversion, we now have same size mask chunks and each can be `FULL`, `CAUSAL` or `PARTIAL`. 

For the time being, with the designed input format, I have finished the attention cuda kernel with randomly generated interleaved masking and applied the q-tiling strategy/copy async/MMA to the omni attention kernel, and working on debugging with the shared KV memory buffer optimization. For the last mile of the project, I will apply the double buffering to prefetch kv and overlap compute and memory, which may have more memory overhead but can improve the throughput, and layout swizzling to the omni attention kernel.

## Preliminary results
For the time being, We have finished the correctness and performance verification scripts for the omni attention kernel, and implemented the shared KV memory buffer optimization, which gets 1.05x speedup over the flex attention on sequence length 1024. The results are as follows:
```shell
============================================================
Testing with data: data/1024/debug_data_F_C.pt ...
============================================================
Simple PASSED, time: 22.43ms, TFLOPS: 0.10, speedup: 0.01x
Shared_kv PASSED, time: 0.25ms, TFLOPS: 8.73, speedup: 1.05x
FlexAttention PASSED, time: 0.26ms, TFLOPS: 8.34
```

## Issues & Concerns

- The shared KV memory buffer optimization is not working as expected, still need to debug and fix it.
- The kernel building is time-consuming, and the debugging is not efficient.
- The computation performance is highly related to the sparsity of the inputs, the attention benefits from the sparse mask presentation deisgn differently. We need at least 30% sparsity to see the performance improvement, otherwise fallback to the naive attention kernel.
- Cross modality mask chunks are having more wrap divergence issue, currently we're using fixed mask block size(128), the best chunk size and splitting strategy may also have impact on the performance.
- Also need hyperparameter tuning for the kernel to find the best configuration for different input shapes, and dynamically decide the block size and others.
- Not sure whether to support big D-dim or not, if so, we may need to apply d-tiling loops inside the Q tiles to make them fit into the on-ship memory.

## Current and Future Goals/Deliverables
### Completed Deliverables
- [x] Setup correctness and proformance test scripts, with different input settings
- [x] Setup cuda + pybind11 environment 
- [x] Finish random mask pattern generation and print
- [x] FInish Sparse block mask class and generator
- [x] Finish naive attention cuda kernel with randomly generated interleaved masking
- [x] FInish applying the q-tiling strategy/copy async to the omni attention kernel
- [x] Finish applying the MMA computation units
- [x] Working on applying the shared kv strategy to the omni attention kernel

### Future Deliverables
- [ ] Improve the performance by implementing double buffering to prefetch kv and overlap compute and memory
- [ ] Improve the performance by applying layout swizzling for the omni attention kernel
- [ ] Profiling and summarization for the final report

Overall, we have completed about 50 - 60% of our planned deliverables, and are on track to complete 100% by the end of next week

## Schedule for the Coming Weeks

-  **half week 1, Dec 1 - Dec 4** implementing double buffering to prefetch kv tiles
    - Given this strategy will have more memory overhead, we mark it stage2 and dynamically decide to use it or not based on the input shape.
    - Prefetch next K tile while computing `S = Q @ K^T`, use `CP_ASYNC_CG` to async copy K tile to smem without waiting for the computation to complete.
    - Prefetch next V tile and overlap its loading with the computation of `O_partial = P @ V`
-  **half week 2, Dec 5 - Dec 7** applying layout swizzling to the tiles
    - Manually apply SMEM swizzling instead of padding to reduce bank conflicts
    - Apply layout swizzling to the Q, K, V tiles
-  **last mile, Dec 8** profiling and summarization for the final report
    - print out the performance comparison table with different input shapes
    - sensitivity analysis for the kernel hyperparameters, like `Br`, `Bc`, `kMmaAtomK`, `kMmaAtomN`, `kMmaAtomK`
    - sensitivity analysis for the sparsity of the inputs
