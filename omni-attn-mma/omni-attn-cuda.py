import argparse
import math
import os
import random
import time
from functools import partial
from typing import Optional

import numpy as np
import torch
from flash_attn import flash_attn_func
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)
torch.set_printoptions(
    precision=6, threshold=8, edgeitems=3, linewidth=120, sci_mode=False
)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rand-q", "--no-rq", action="store_true")
    parser.add_argument("--no-rand-k", "--no-rk", action="store_true")
    parser.add_argument("--no-rand-v", "--no-rv", action="store_true")
    parser.add_argument("--no-rand-qkv", "--no-rqkv", action="store_true")
    parser.add_argument("--run-torch-unfused", "--torch", action="store_true")
    parser.add_argument("--run-torch-sdpa", "--sdpa", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-all", action="store_true")
    parser.add_argument("--show-all", "--show", action="store_true")
    parser.add_argument("--show-matrix", action="store_true")
    parser.add_argument(
        "--only-flops-matmul", "--flops-mm", action="store_true"
    )
    parser.add_argument(
        "--run-acc-f32", "--acc-f32", "--f32", action="store_true"
    )
    parser.add_argument("--B", type=int, default=None)
    parser.add_argument("--H", type=int, default=None)
    parser.add_argument("--N", type=int, default=None)
    parser.add_argument("--D", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", "--v", action="store_true")
    parser.add_argument("--warmup", "--w", type=int, default=1)
    parser.add_argument("--iters", "--i", type=int, default=5)
    parser.add_argument("--range-k", "--gk", action="store_true")
    parser.add_argument("--build-others", "--others", action="store_true")
    parser.add_argument(
        "--tag-hints", "--tags", "--hints", type=str, default=None
    )
    return parser.parse_args()


args = get_args()


def set_rand_seed(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device_name():
    device_name = torch.cuda.get_device_name(torch.cuda.current_device())
    # since we will run GPU on WSL2, so add WSL2 tag.
    if "Laptop" in device_name:
        device_name += " WSL2"
    return device_name


def get_device_capability():
    return torch.cuda.get_device_capability(torch.cuda.current_device())


def get_build_sources():
    build_sources = []
    # Basic
    build_sources.append("./mma/basic/omni_attn_mma_share_kv.cu")
    # Pybind
    build_sources.append("./pybind/omni_attn.cc")
    return build_sources


def get_project_dir():
    return os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )


project_dir = get_project_dir()


def get_build_cuda_cflags(build_pkg: bool = False):
    device_name = get_device_name()
    project_dir = get_project_dir()
    extra_cuda_cflags = []
    extra_cuda_cflags.append("-O3")
    extra_cuda_cflags.append("-std=c++17")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF_OPERATORS__")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF_CONVERSIONS__")
    extra_cuda_cflags.append("-U__CUDA_NO_HALF2_OPERATORS__")
    extra_cuda_cflags.append("-U__CUDA_NO_BFLOAT16_CONVERSIONS__")
    extra_cuda_cflags.append("--expt-relaxed-constexpr")
    extra_cuda_cflags.append("--expt-extended-lambda")
    extra_cuda_cflags.append("--use_fast_math")
    extra_cuda_cflags.append("-DFLASH_ATTN_MMA_DEBUG" if args.debug else "")
    extra_cuda_cflags.append(
        "-DBUILD_FLASH_ATTN_MMA_OTHERS" if args.build_others else ""
    )
    extra_cuda_cflags.append(
        "-DBUILD_FLASH_ATTN_MMA_L20" if "L20" in device_name else ""
    )
    extra_cuda_cflags.append(
        "-DBUILD_FLASH_ATTN_MMA_4090" if "4090" in device_name else ""
    )
    extra_cuda_cflags.append(
        "-DBUILD_FLASH_ATTN_MMA_3080" if "3080" in device_name else ""
    )
    extra_cuda_cflags.append(
        "-diag-suppress 177" if not build_pkg else "--ptxas-options=-v"
    )
    extra_cuda_cflags.append(
        "-Xptxas -v" if not build_pkg else "--ptxas-options=-O3"
    )
    # Point include paths to the actual omni-attn source tree
    extra_cuda_cflags.append(f"-I {project_dir}/attention/omni-attn")
    extra_cuda_cflags.append(f"-I {project_dir}/attention/omni-attn/utils")
    extra_cuda_cflags.append(f"-I {project_dir}/attention/omni-attn/mma")
    extra_cuda_cflags.append(f"-I {project_dir}/attention/omni-attn/pybind")
    # If CUTLASS is installed separately, uncomment and set the proper include paths below:
    # extra_cuda_cflags.append(f"-I {project_dir}/third-party/cutlass/include")
    # extra_cuda_cflags.append(f"-I {project_dir}/third-party/cutlass/tools/util/include")
    return extra_cuda_cflags


def get_build_cflags():
    extra_cflags = []
    extra_cflags.append("-std=c++17")
    extra_cflags.append(
        "-DBUILD_FLASH_ATTN_MMA_OTHERS" if args.build_others else ""
    )
    return extra_cflags


def pretty_print_line(m: str = "", sep: str = "-", width: int = 150):
    res_len = width - len(m)
    left_len = int(res_len / 2)
    right_len = res_len - left_len
    pretty_line = sep * left_len + m + sep * right_len
    print(pretty_line)


if args.D and args.D > 256:
    args.run_torch_sdpa = True
pretty_print_line()
print(args)
pretty_print_line()

# Load the CUDA kernel as a python module
lib = load(
    name="omni_attn_lib",
    sources=get_build_sources(),
    extra_cuda_cflags=get_build_cuda_cflags(),
    extra_cflags=get_build_cflags(),
    verbose=args.verbose,
)


def get_mha_tflops(
    B: int, H: int, N: int, D: int, secs: float = 1.0, only_matmul: bool = False
):
    # Q @ K^T FLOPs
    flops_qk = B * H * N * N * (2 * D - 1)

    # Scaling FLOPs
    flops_scaling = B * H * N * N

    # Safe_Softmax FLOPs
    flops_row_max = B * H * N * (N - 1)  # row max
    flops_subtract_max = B * H * N * N  # sub max
    flops_exp = B * H * N * N  # pointwise exp
    flops_row_sum = B * H * N * (N - 1)  # row sum
    flops_normalization = B * H * N * N  # normalization

    flops_safe_softmax = (
        flops_row_max
        + flops_subtract_max
        + flops_exp
        + flops_row_sum
        + flops_normalization
    )

    # P @ V FLOPs
    flops_pv = B * H * N * D * (2 * N - 1)

    # Total FLOPs
    total_flops = flops_qk + flops_scaling + flops_safe_softmax + flops_pv
    if only_matmul:
        total_flops = flops_qk + flops_pv

    # Convert to TFLOPS
    # 1 TFLOPS = 10^12 FLOPS
    # ref: https://imgtec.eetrend.com/blog/2021/100062210.html.
    tflops = total_flops * 1e-12 / (secs)

    return tflops


MAX_TFLOPS = -1
STATIS_INFO: dict[str, list[float]] = {}
TOATL_TFLOPS: dict[str, float] = {}


def run_benchmark(
    perf_func: callable,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    s: Optional[torch.Tensor] = None,  # DEBUG
    stages: int = -1,
    warmup: int = args.warmup,
    iters: int = args.iters,
    show_matrix: bool = args.show_matrix,
    only_show_improved: bool = not args.show_all,
):

    global MAX_TFLOPS
    global MAX_HEADDIM_CFG

    tag_hints: str = args.tag_hints  # e.g "share-qkv,tiling-kv,swizzle"
    if tag_hints:
        tag_hints: list = tag_hints.strip().split(",")
        tag_hints.append("omni")
        tag_hints.append("sdpa")
        tag_hints.append("unfused")
        hit_hints = False
        for hint in tag_hints:
            if hint in tag:
                hit_hints = True
        if not hit_hints:
            return None, None

    if not args.build_others:
        others_tags = ["s2g", "rr"]
        for o_tag in others_tags:
            if o_tag in tag:
                return None, None

    if "sdpa" in tag and (not args.run_torch_sdpa):
        return None, None
    if "unfused" in tag and (not args.run_torch_unfused):
        return None, None
    if "acc-f32" in tag and (not args.run_acc_f32):
        return None, None

    B, H, N, D = q.size()

    max_supported_D = MAX_HEADDIM_CFG.get(tag, None)
    # skip if headdim not supported.
    if max_supported_D is not None:
        if D > max_supported_D:
            return None, None

    if out is not None:
        out.fill_(0)
    if s is not None:
        s.fill_(0)
    if out is not None:
        for i in range(warmup):
            if stages >= 1:
                if s is not None:
                    perf_func(q, k, v, out, s, stages)
                else:
                    perf_func(q, k, v, out, stages)
            else:
                perf_func(q, k, v, out)
    else:
        for i in range(warmup):
            _ = perf_func(q, k, v)

    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            if stages >= 1:
                if s is not None:
                    perf_func(q, k, v, out, s, stages)
                else:
                    perf_func(q, k, v, out, stages)
            else:
                perf_func(q, k, v, out)
    else:
        for i in range(iters):
            out = perf_func(q, k, v)
    torch.cuda.synchronize()
    end = time.time()
    total_secs = end - start
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    mean_secs = total_secs / iters

    TFLOPS = get_mha_tflops(
        B, H, N, D, mean_secs, only_matmul=args.only_flops_matmul
    )
    out_info = f"{tag}"
    out_val_first = out.flatten()[:3].detach().cpu().numpy().tolist()
    out_val_last = out.flatten()[-3:].detach().cpu().numpy().tolist()
    out_val_first = [round(v, 8) for v in out_val_first]
    out_val_last = [round(v, 8) for v in out_val_last]
    out_val = out_val_first[:2]
    out_val.append(out_val_last[-1])
    out_val = [f"{v:<12}" for v in out_val]

    # caculate TFLOPS improved.
    if TFLOPS > MAX_TFLOPS:
        if MAX_TFLOPS > 0:
            improve = ((TFLOPS - MAX_TFLOPS) / MAX_TFLOPS) * 100
            improve = round(improve, 2)
        else:
            improve = 0
        MAX_TFLOPS = TFLOPS
        print(
            f"{out_info:>50}: {out_val}, time:{str(mean_time)[:8]}ms, "
            f"TFLOPS:{TFLOPS:<6.2f}(+{improve:.2f}%)"
        )
    else:
        if (not only_show_improved) or (("flash" in tag) or ("sdpa" in tag)):
            print(
                f"{out_info:>50}: {out_val}, time:{str(mean_time)[:8]}ms, "
                f"TFLOPS:{TFLOPS:<6.2f}"
            )

    if show_matrix:
        print(out)
    time.sleep(args.sleep)
    torch.cuda.synchronize()
    return out.clone(), mean_time


def get_qkvo(B, H, N, D):
    if not (args.no_rand_q or args.no_rand_qkv):
        q = torch.randn((B, H, N, D), dtype=torch.half, device="cuda")
    else:
        q = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    if not (args.no_rand_k or args.no_rand_qkv):
        k = torch.randn((B, H, N, D), dtype=torch.half, device="cuda")
    else:
        k = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
        if args.range_k:
            for i in range(N):
                k[:, :, i, :] = (i + 1) / N
            k = k.cuda().half().contiguous()
    if not (args.no_rand_v or args.no_rand_qkv):
        v = torch.randn((B, H, N, D), dtype=torch.half, device="cuda")
    else:
        v = torch.ones(B, H, N, D, device="cuda", dtype=torch.half).contiguous()

    o = torch.zeros(B, H, N, D, device="cuda", dtype=torch.half).contiguous()
    # transpose (H,N) -> (N,H) for FA2.
    fq = q.transpose(1, 2).contiguous()
    fk = k.transpose(1, 2).contiguous()
    fv = v.transpose(1, 2).contiguous()
    # transpose (N,D) -> (D,N) for V smem swizzle.
    tk = k.transpose(-2, -1).contiguous()  # [B,H,N,D] -> [B,H,D,N]
    tv = v.transpose(-2, -1).contiguous()  # [B,H,N,D] -> [B,H,D,N]

    return q, k, v, o, fq, fk, fv, tk, tv


# un-fused naive attn
def unfused_standard_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    att = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


def sdpa(q: Tensor, k: Tensor, v: Tensor, use_flash: bool = False):
    if not use_flash:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            out: Tensor = F.scaled_dot_product_attention(q, k, v)
    else:
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            out: Tensor = F.scaled_dot_product_attention(q, k, v)
    return out

Bs = [1, 4, 8] if not args.B else [args.B]
Hs = [1, 4, 8] if not args.H else [args.H]
Ns = [1024, 2048, 4096, 8192] if not args.N else [args.N]
Ds = [64, 128, 256, 512] if not args.D else [args.D]
# batch_size, n_head, seq_len, head_dim (B,H,N,D)
BHNDs = [(B, H, N, D) for B in Bs for H in Hs for N in Ns for D in Ds]
# max headdim supported for different methods. skip if D > max_D.
MAX_HEADDIM_CFG: dict[str, int] = {
    # FA2, SDPA, Naive MHA.
    "(flash)": 256,
    "(sdpa)": 4096,  # may no limit
    "(unfused)": 4096,  # may no limit
    
}

seed = args.seed if args.seed else random.choice(range(10000))
set_rand_seed(seed)
pretty_print_line()
pretty_print_line(
    f"B: batch_size, H: n_head, N: seq_len, D: head_dim, "
    f"seed: {seed}, Warmup: {args.warmup}, Iters: {args.iters}"
)

run_torch_sdpa = args.run_torch_sdpa
for B, H, N, D in BHNDs:
    MAX_TFLOPS = -1
    q, k, v, o, fq, fk, fv, tk, tv = get_qkvo(B, H, N, D)
    if D > 256:
        args.run_torch_sdpa = True
    else:
        args.run_torch_sdpa = run_torch_sdpa
    torch.cuda.synchronize()
    pretty_print_line()
    pretty_print_line(
        f"B={B}, H={H}, N={N}, D={D}, Warmup: {args.warmup}, Iters: {args.iters}"
    )
    # Naive MHA.
    out_unfused, _ = run_benchmark(unfused_standard_attn, q, k, v, "(unfused)")
    
    
    # Omni-Attention MMA with sparse blocks (full attention pattern)
    # Create block mask for full attention (all blocks are FULL type)
    # The kernel uses Br and Bc which depend on headdim and stage
    # For simplicity, use fixed block sizes that match common configurations
    import math
    # Br and Bc are typically 64 or 128 depending on headdim
    # For headdim 64: Br=64, Bc=64 (stage 1) or Br=64, Bc=32 (stage 2)
    # For headdim 128: Br=128, Bc=64 (stage 1) or Br=128, Bc=32 (stage 2)
    # Use Br=64, Bc=64 as default (matches stage 1 for headdim 64)
    BLOCK_M = 64  # Br tile size
    BLOCK_N = 64  # Bc tile size
    num_q_blocks = (args.N + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (args.N + BLOCK_N - 1) // BLOCK_N
    
    # Create full attention pattern (all blocks are FULL = 2)
    block_mask_pattern = torch.full(
        (num_q_blocks, num_kv_blocks), 2, dtype=torch.int32, device=q.device
    )
    
    # Create kv_num_blocks, kv_indices, block_mask_types
    kv_num_blocks = torch.full(
        (args.B, args.H, num_q_blocks), num_kv_blocks, dtype=torch.int32, device=q.device
    )
    kv_indices = torch.zeros(
        (args.B, args.H, num_q_blocks, num_kv_blocks), dtype=torch.int32, device=q.device
    )
    block_mask_types = torch.zeros(
        (args.B, args.H, num_q_blocks, num_kv_blocks), dtype=torch.int32, device=q.device
    )
    
    # Fill in indices and types
    for q_idx in range(num_q_blocks):
        for kv_idx in range(num_kv_blocks):
            for b in range(args.B):
                for h in range(args.H):
                    kv_indices[b, h, q_idx, kv_idx] = kv_idx
                    block_mask_types[b, h, q_idx, kv_idx] = 2  # FULL
    
    # Benchmark omni-attn-mma kernel
    def omni_attn_benchmark_func(q, k, v, o, stages):
        lib.omni_attn_mma_stages_split_q_shared_kv(
            q, k, v, o, kv_num_blocks, kv_indices, block_mask_types, stages
        )
        return o
    
    out_omni_mma_share_kv1, _ = run_benchmark(
        lambda q, k, v, o, s: omni_attn_benchmark_func(q, k, v, o, 1),
        q,
        k,
        v,
        "omni-mma(split-q+share-kv+stage1)",
        o,
        stages=1,
    )
    out_omni_mma_share_kv2, _ = run_benchmark(
        lambda q, k, v, o, s: omni_attn_benchmark_func(q, k, v, o, 2),
        q,
        k,
        v,
        "omni-mma(split-q+share-kv+stage2)",
        o,
        stages=2,
    )
    
    pretty_print_line()

    torch.cuda.synchronize()
    if args.check:
        if D <= 256:
            
            pretty_print_line()
