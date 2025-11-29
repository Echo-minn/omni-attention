"""
Setup script for building the Omni-Attention CUDA MMA extension.

Usage:
    python setup.py install        # Install the extension
    python setup.py build_ext      # Build without installing
    pip install -e .               # Install in editable mode
"""

import os
import subprocess
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_cuda_arch_flags():
    """Get CUDA architecture flags based on available GPUs or defaults."""
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            arch = f"{major}{minor}"
            return [f"-gencode=arch=compute_{arch},code=sm_{arch}"]
    except:
        pass
    
    # Default to common architectures
    return [
        "-gencode=arch=compute_80,code=sm_80",  # A100
        "-gencode=arch=compute_86,code=sm_86",  # RTX 3090
        "-gencode=arch=compute_89,code=sm_89",  # RTX 4090
        "-gencode=arch=compute_90,code=sm_90",  # H100
    ]


# Get the directory containing this setup.py
this_dir = os.path.dirname(os.path.abspath(__file__))

# Source files
sources = [
    os.path.join(this_dir, "pybind", "omni_attn.cc"),
    os.path.join(this_dir, "mma", "omni_attn_mma_share_kv.cu"),
]

# Include directories
include_dirs = [
    os.path.join(this_dir, "utils"),
]

# CUDA flags
cuda_flags = [
    "-O3",
    "-std=c++17",
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    "-use_fast_math",
    "--ptxas-options=-v",
    "-lineinfo",
] + get_cuda_arch_flags()

# C++ flags
cxx_flags = [
    "-O3",
    "-std=c++17",
]

# Extra compile args
extra_compile_args = {
    "cxx": cxx_flags,
    "nvcc": cuda_flags,
}

setup(
    name="omni_attn_mma_cuda",
    version="0.1.0",
    description="Omni-Attention with CUDA MMA kernels",
    author="Omni-Attention Team",
    ext_modules=[
        CUDAExtension(
            name="omni_attn_mma_cuda",
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={
        "build_ext": BuildExtension,
    },
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0",
    ],
)

