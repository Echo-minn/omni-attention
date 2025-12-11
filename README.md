# Quick Start

> in the `/omni-attn-mma` directory

## 1. Install dependencies

```bash
CUDA + PyTorch + Pybind11 + Pybind11 CUDA Extension
torch: 2.9.0+cu128
cuda: 12.8
pybind11: 3.0.1
Python 3.12.3
```

## 2. Build the kernels

```bash
python setup.py build_ext --inplace
```

## 3. Generate debug data or attention data

```bash
python generate_debug_data.py --B 1 --H 8 --seq_len 512 --head_dim 64 --BLOCK_SIZE 128 --output data/512/debug_data_partial.pt
python generate_attn_data.py --B 1 --H 8 --seq_len 512 --head_dim 64 --Q_BLOCK_SIZE 128 --KV_BLOCK_SIZE 128 --output data/512/attn_data_causal.pt
```

## 4. Run the tests

```bash
python test_with_debug_data.py --input data/512/debug_data_partial.pt
```