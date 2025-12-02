# **Fine-grained FlashAttention with interleaved masking for Cross-Modal Data Training Efficiency**

> Mira Xiao(minxiao@andrew.cmu.edu)
>
> GuanWei Wu(guanweiwu@andrew.cmu.edu)

I'm thinking about a way to support interleaved masking technique in FlashAttention to accerlate models training on arbitrary interleavings of text and images. They are engineering/research intensive and promsing.

## Background and Motivation

The well-known Transformer language models are auto-regressive and tokens only attend previous ones, so we apply causal mask to hide future tokens behind current step in standard attention.
Large multimodal models increasingly train on **interleaved text and image sequences**, enabling natural conversational interfaces that combine language with visual understanding and generation. Current attention mechanisms, however, struggle with the **heterogeneous structure** of such data. Text requires **causal masking** (to preserve autoregressive behavior), while images often benefit from **non-causal or intra-block full attention** to capture spatial coherence. Naively combining these modalities leads to inefficient kernels, excessive memory use, and degraded training throughput.

<img src="./imgs/causal_mask.png" alt="Causal Mask: Text-only Attention" width="380" height="350" style="display: block; margin: 0 auto" />

<img src="./imgs/interleaved_mask.png" alt="Interleaved Mask: Fine-Grained Cross-Modal Masking" width="450" height="400" style="display: block; margin: 0 auto" />

**FlashAttention** has emerged as a breakthrough in training efficiency by reordering attention computation to minimize memory I/O, but it is primarily optimized for homogeneous masks (causal or full). Extending FlashAttention to handle **fine-grained interleaved masking** could unlock significant efficiency gains for multimodal training at scale.

## Project Objectives

This project proposes to design and evaluate **Fine-grained FlashAttention with Interleaved Masking**, focusing on:

1. **Mask generalization**: Support text–image interleaving with different masking rules (causal text, full intra-image, controlled cross-modal).
2. **Tile-aware optimization**: Align FlashAttention tiles with modality boundaries to reduce irregular per-element masks.
3. **Kernel optimization**: Write custom high-performance kernels for the interleaved masking.
4. **Evaluation**: Benchmark against standard FlashAttention/FlexAttention and dense attention on multimodal pretraining workloads.

## Anticipated Challenges

1. **Irregular masking overhead**: Interleaved data produce heterogeneous masks that reduce GPU warp efficiency if handled naively.
2. **Tile misalignment**: Tiles overlapping modality boundaries may require per-element masking, undermining FlashAttention’s efficiency.
3. **Sequence length**: Discretized images greatly increase sequence length, stressing memory bandwidth despite IO-aware design.
4. **Cross-modal balance**: Different loss types (cross-entropy for text, reconstruction for images) may complicate optimization and gradient stability.

## Project Goals Milestones

Baseline: Naive version attention variant with intervleaved masking

Target: FlashAttention with causal mask

- 50%: 50% performance from the baseline to the target
- 75%: 70%
- 100%: 85%
- 125%: 90%
- 150%: 90%+

## How to achieve the objectives

Stack: Python/PyTorch for model part, C++/CUDA for kernel part; PYBIND11 for binding the kernel to Python.

1. Apply the interleaved masking technique in FlashAttention.
2. Align FlashAttention tiles with modality boundaries to reduce irregular per-element masks.
3. Optimize the kernel for the interleaved masking(tile alignment, precomputed mask metadata, hybrid causal/full kernels, vectorization, synchronization, etc.).
4. Benchmark against standard FlashAttention/FlexAttention and dense attention on multimodal pretraining workloads.

## Dataset preparation

1. Construct text/image interleaved sequences with some tags in between.
2. Multimodal training dataset(subset of the pre-training dataset)

## Expected Contributions

* A novel high-performance attention variant supporting **fine-grained interleaved masks**.
* Implementation strategies (tile alignment, precomputed mask metadata, hybrid causal/full kernels).
* Empirical benchmarks on multimodal datasets demonstrating improved **training throughput, memory use, and stability**.

## References

[1] [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](http://arxiv.org/abs/2408.11039)\
[2] [Flex Attention: a Programming Model for Generating  Optimized Attention Kernels](https://arxiv.org/pdf/2412.05496)\
[3] [Blog: FlexAttention - The Flexibility of PyTorch with the Performance of FlashAttention](https://pytorch.org/blog/flexattention/)\
[4] [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/pdf/2205.14135)
