"""
model/cuda/ops.py

Host API wrapping the PyCUDA SourceModule for GT 730 (sm_35).
Handles environment bootstrap, compilation, grid/block sizing, and
host<->device transfers. Import this module (not pycuda directly)
from model code so the CUDA env is always configured first.
"""

import numpy as np

from model.cuda import env as _env

_env.configure()

import pycuda.autoinit  # noqa: E402  (must follow env.configure())
import pycuda.driver as cuda  # noqa: E402
import pycuda.gpuarray as gpuarray  # noqa: E402
from pycuda.compiler import SourceModule  # noqa: E402

from logging_config import logger  # noqa: E402
from model.cuda.kernels import CUDA_SOURCE  # noqa: E402

_mod = SourceModule(CUDA_SOURCE, options=_env.NVCC_OPTIONS)

_gemm_kernel = _mod.get_function("gemm_fp32")
_add_bias_kernel = _mod.get_function("add_bias_fp32")
_layernorm_kernel = _mod.get_function("layernorm_fp32")
_gelu_kernel = _mod.get_function("gelu_fp32")
_softmax_kernel = _mod.get_function("softmax_fp32")
_add_kernel = _mod.get_function("add_fp32")
_causal_mha_kernel = _mod.get_function("causal_mha_fp32")
_split_qkv_kernel = _mod.get_function("split_qkv_fp32")
_cross_entropy_kernel = _mod.get_function("cross_entropy_fp32")
_gelu_backward_kernel = _mod.get_function("gelu_backward_fp32")
_layernorm_cache_kernel = _mod.get_function("layernorm_cache_fp32")
_layernorm_backward_kernel = _mod.get_function("layernorm_backward_fp32")
_embed_backward_kernel = _mod.get_function("embed_backward_fp32")
_pos_embed_backward_kernel = _mod.get_function("pos_embed_backward_fp32")
_embed_forward_kernel = _mod.get_function("embed_forward_fp32")
_scal_mul_kernel = _mod.get_function("scal_mul_fp32")
_adamw_update_kernel = _mod.get_function("adamw_update_fp32")
_softmax_backward_kernel = _mod.get_function("softmax_backward_fp32")
_add_block_kernel = _mod.get_function("add_block_fp32")
_grad_norm_contrib_kernel = _mod.get_function("grad_norm_contrib_fp32")
_add_inplace_kernel = _mod.get_function("add_inplace_fp32")
_interleaved_to_heads_kernel = _mod.get_function("interleaved_to_heads")
_merge_heads_kernel = _mod.get_function("merge_heads_kernel")
_pack_qkv_kernel = _mod.get_function("pack_qkv_fp32")
_matmul_score_kernel = _mod.get_function("matmul_score_kernel")
_softmax_fused_backward_kernel = _mod.get_function("softmax_fused_backward")
_matmul_grad_q_kernel = _mod.get_function("matmul_grad_q_kernel")
_matmul_grad_k_kernel = _mod.get_function("matmul_grad_k_kernel")
_matmul_grad_v_kernel = _mod.get_function("matmul_grad_v")
_gemm_batched_kernel = _mod.get_function("gemm_batched_fp32")
_reduce_sum_axis0_kernel = _mod.get_function("reduce_sum_axis0_fp32")
_transpose_2d_kernel = _mod.get_function("transpose_2d_fp32")
_gemm_bias_kernel = _mod.get_function("gemm_bias_fp32")
_split_heads_kernel = _mod.get_function("split_heads_kernel")
_merge_heads_qkv_kernel = _mod.get_function("merge_heads_qkv_kernel")
_fused_attn_fwd_kernel = _mod.get_function("fused_attention_forward_kernel")

logger.info("PyCUDA kernels compiled for sm_35 (Kepler GT 730)")

MAX_THREADS_PER_BLOCK = 1024


def _launch_1d(n: int, threads: int = 256):
    return (int(np.ceil(n / threads)), 1, 1), (threads, 1, 1)


def get_memory_info():
    # type: () -> tuple
    """Return (free_bytes, total_bytes) of GPU VRAM, straight from the driver."""
    free_bytes, total_bytes = cuda.mem_get_info()
    return free_bytes, total_bytes


def next_pow2(n: int, cap: int = MAX_THREADS_PER_BLOCK) -> int:
    """Smallest power of two >= n, capped at `cap`. Minimum 32 (one warp)."""
    p = 32
    while p < n and p < cap:
        p *= 2
    return min(p, cap)


def to_device(arr: np.ndarray) -> gpuarray.GPUArray:
    """Upload a NumPy float32 array to the device."""
    return gpuarray.to_gpu(np.ascontiguousarray(arr, dtype=np.float32))


def to_host(arr: gpuarray.GPUArray) -> np.ndarray:
    """Download a device array to NumPy."""
    return arr.get()


def add_arrays(a: gpuarray.GPUArray, b: gpuarray.GPUArray) -> gpuarray.GPUArray:
    """Elementwise add of two same-shaped device arrays."""
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} vs {b.shape}"
    n_elements = np.int32(a.size)
    out = gpuarray.empty_like(a)
    threads = 256
    blocks = int(np.ceil(int(n_elements) / threads))
    _add_kernel(a, b, out, n_elements, block=(threads, 1, 1), grid=(blocks, 1, 1))
    return out


def matmul(A: gpuarray.GPUArray, B: gpuarray.GPUArray, tracer=None, name: str = "gemm") -> gpuarray.GPUArray:
    """C = A @ B for 2D float32 device arrays.

    Non-contiguous ``.T`` views are materialized with a transpose kernel, then
    multiplied with the contiguous gemm (strided gemm is TDR-slow on Kepler).
    """
    assert A.ndim == 2 and B.ndim == 2, f"matmul expects 2D, got {A.shape} @ {B.shape}"
    M, Ka = int(A.shape[0]), int(A.shape[1])
    Kb, N = int(B.shape[0]), int(B.shape[1])
    assert Ka == Kb, f"Shape mismatch: {A.shape} vs {B.shape}"

    if not A.flags.c_contiguous:
        # A is a .T view of shape (M, K) over physical (K, M) storage.
        base = gpuarray.GPUArray((Ka, M), np.float32, gpudata=A.gpudata)
        A = transpose_2d(base)  # contiguous (M, K)
    if not B.flags.c_contiguous:
        base = gpuarray.GPUArray((N, Ka), np.float32, gpudata=B.gpudata)
        B = transpose_2d(base)  # contiguous (K, N)

    C = gpuarray.empty((M, N), dtype=np.float32)
    block_dim = (16, 16, 1)
    grid_dim = (int(np.ceil(N / 16)), int(np.ceil(M / 16)), 1)

    if tracer is not None and getattr(tracer, "trace_vectorization", False):
        tracer.log_vectorization(name, (M, Ka), (Kb, N), (M, N), grid_dim, block_dim)

    _gemm_kernel(A, B, C, np.int32(M), np.int32(N), np.int32(Ka), block=block_dim, grid=grid_dim)
    return C


def transpose_2d(x: gpuarray.GPUArray) -> gpuarray.GPUArray:
    """Return contiguous transpose of a C-contiguous 2D array."""
    assert x.ndim == 2 and x.flags.c_contiguous
    rows, cols = int(x.shape[0]), int(x.shape[1])
    out = gpuarray.empty((cols, rows), dtype=np.float32)
    n = rows * cols
    grid, block = _launch_1d(n)
    _transpose_2d_kernel(x, out, np.int32(rows), np.int32(cols), block=block, grid=grid)
    return out


def matmul_bias(
    A: gpuarray.GPUArray, B: gpuarray.GPUArray, bias: gpuarray.GPUArray,
    tracer=None, name: str = "gemm_bias",
) -> gpuarray.GPUArray:
    """C = A @ B + bias (contiguous A, B only)."""
    assert A.flags.c_contiguous and B.flags.c_contiguous
    M, K = int(A.shape[0]), int(A.shape[1])
    assert B.shape[0] == K and int(bias.size) == int(B.shape[1])
    N = int(B.shape[1])
    C = gpuarray.empty((M, N), dtype=np.float32)
    block_dim = (16, 16, 1)
    grid_dim = (int(np.ceil(N / 16)), int(np.ceil(M / 16)), 1)
    if tracer is not None and getattr(tracer, "trace_vectorization", False):
        tracer.log_vectorization(name, (M, K), (K, N), (M, N), grid_dim, block_dim)
    _gemm_bias_kernel(
        A, B, bias, C, np.int32(M), np.int32(N), np.int32(K),
        block=block_dim, grid=grid_dim,
    )
    return C


def reduce_sum_axis0(x: gpuarray.GPUArray) -> gpuarray.GPUArray:
    """Sum over axis 0 of a 2D [rows, channels] array. Returns [channels]."""
    assert x.ndim == 2 and x.flags.c_contiguous
    rows, channels = int(x.shape[0]), int(x.shape[1])
    out = gpuarray.empty((channels,), dtype=np.float32)
    threads = next_pow2(min(rows, 256))
    shared = threads * np.dtype(np.float32).itemsize
    _reduce_sum_axis0_kernel(
        x, out, np.int32(rows), np.int32(channels),
        block=(threads, 1, 1), grid=(channels, 1, 1), shared=shared,
    )
    return out


def linear_backward(
    dout: gpuarray.GPUArray,
    x: gpuarray.GPUArray,
    weight: gpuarray.GPUArray,
) -> tuple:
    """Backward for y = x @ weight + b. weight is [in, out]. Returns (d_x, d_weight, d_bias)."""
    d_weight = matmul(x.T, dout)
    d_bias = reduce_sum_axis0(dout)
    d_x = matmul(dout, weight.T)
    return d_x, d_weight, d_bias


def add_bias(a: gpuarray.GPUArray, bias: gpuarray.GPUArray) -> gpuarray.GPUArray:
    """out = a + bias, broadcasting bias over the leading dimension(s) of `a`."""
    n_elements = np.int32(a.size)
    bias_len = np.int32(bias.size)
    out = gpuarray.empty_like(a)

    threads = 256
    blocks = int(np.ceil(int(n_elements) / threads))
    _add_bias_kernel(a, bias, out, n_elements, bias_len, block=(threads, 1, 1), grid=(blocks, 1, 1))
    return out


def layernorm(x: gpuarray.GPUArray, gamma: gpuarray.GPUArray, beta: gpuarray.GPUArray, eps: float = 1e-5) -> gpuarray.GPUArray:
    """Layernorm over the last dimension. `x` is treated as [total_rows, hidden_dim]."""
    hidden_dim = int(x.shape[-1])
    total_rows = int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1

    out = gpuarray.empty_like(x)
    threads = next_pow2(hidden_dim)
    shared_bytes = threads * np.dtype(np.float32).itemsize

    _layernorm_kernel(
        x, out, gamma, beta, np.int32(hidden_dim), np.float32(eps), np.int32(total_rows),
        block=(threads, 1, 1), grid=(total_rows, 1, 1), shared=shared_bytes,
    )
    return out


def gelu(x: gpuarray.GPUArray) -> gpuarray.GPUArray:
    """Elementwise GeLU (tanh approximation)."""
    n_elements = np.int32(x.size)
    out = gpuarray.empty_like(x)

    threads = 256
    blocks = int(np.ceil(int(n_elements) / threads))
    _gelu_kernel(x, out, n_elements, block=(threads, 1, 1), grid=(blocks, 1, 1))
    return out


def softmax(logits: gpuarray.GPUArray) -> gpuarray.GPUArray:
    """Softmax over the last dimension. `logits` is treated as [total_rows, vocab_size]."""
    vocab_size = int(logits.shape[-1])
    total_rows = int(np.prod(logits.shape[:-1])) if logits.ndim > 1 else 1

    probs = gpuarray.empty_like(logits)
    threads = next_pow2(vocab_size)
    shared_bytes = threads * np.dtype(np.float32).itemsize

    _softmax_kernel(
        logits, probs, np.int32(vocab_size), np.int32(total_rows),
        block=(threads, 1, 1), grid=(total_rows, 1, 1), shared=shared_bytes,
    )
    return probs


def split_qkv(qkv: gpuarray.GPUArray, hidden_dim: int):
    """Split [rows, 3*C] qkv into contiguous Q,K,V [rows, C] on device."""
    rows = int(qkv.shape[0])
    c = int(hidden_dim)
    n = rows * c
    q = gpuarray.empty((rows, c), dtype=np.float32)
    k = gpuarray.empty((rows, c), dtype=np.float32)
    v = gpuarray.empty((rows, c), dtype=np.float32)
    threads = 256
    blocks = int(np.ceil(n / threads))
    _split_qkv_kernel(qkv, q, k, v, np.int32(rows), np.int32(c), block=(threads, 1, 1), grid=(blocks, 1, 1))
    return q, k, v


def causal_self_attention(
    q: gpuarray.GPUArray,
    k: gpuarray.GPUArray,
    v: gpuarray.GPUArray,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    scale: float,
) -> tuple:
    """Causal MHA on interleaved Q/K/V [B*T, C]. Returns (attn_concat, probs)."""
    B, T, H, hd = int(batch_size), int(seq_len), int(num_heads), int(head_dim)
    C = H * hd
    out = gpuarray.empty((B * T, C), dtype=np.float32)
    probs = gpuarray.empty((B * H * T * T,), dtype=np.float32)

    num_warps = (hd + 31) // 32
    shared_bytes = (2 * T + hd + num_warps) * np.dtype(np.float32).itemsize
    grid = (T, H, B)
    block = (int(hd), 1, 1)

    _causal_mha_kernel(
        q, k, v, out, probs,
        np.int32(B), np.int32(T), np.int32(H), np.int32(hd), np.float32(scale),
        block=block, grid=grid, shared=shared_bytes,
    )
    return out, probs


# Phase 2C fused forward is correct but ~2x slower than causal_mha on sm_35 at T=256.
# Default uses corrected causal_mha; set True to force the GPU5 fused kernel.
_USE_FUSED_ATTENTION_FORWARD = False


def split_heads_from_qkv(
    qkv: gpuarray.GPUArray, batch_size: int, seq_len: int, num_heads: int, head_dim: int,
) -> tuple:
    """QKV [B*T, 3*C] -> Q,K,V each [B*NH, T, HD]."""
    B, T, NH, HD = int(batch_size), int(seq_len), int(num_heads), int(head_dim)
    q = gpuarray.empty((B * NH, T, HD), dtype=np.float32)
    k = gpuarray.empty((B * NH, T, HD), dtype=np.float32)
    v = gpuarray.empty((B * NH, T, HD), dtype=np.float32)
    n = B * T * NH * HD
    grid, block = _launch_1d(n)
    _split_heads_kernel(
        qkv, q, k, v, np.int32(B), np.int32(T), np.int32(NH), np.int32(HD),
        block=block, grid=grid,
    )
    return q, k, v


def pack_qkv_from_heads(
    d_q: gpuarray.GPUArray, d_k: gpuarray.GPUArray, d_v: gpuarray.GPUArray,
    batch_size: int, seq_len: int, num_heads: int, head_dim: int,
) -> gpuarray.GPUArray:
    """Pack dQ/dK/dV [B*NH, T, HD] into dQKV [B*T, 3*C]."""
    B, T, NH, HD = int(batch_size), int(seq_len), int(num_heads), int(head_dim)
    out = gpuarray.empty((B * T, 3 * NH * HD), dtype=np.float32)
    n = B * T * NH * HD
    grid, block = _launch_1d(n)
    _merge_heads_qkv_kernel(
        d_q, d_k, d_v, out, np.int32(B), np.int32(T), np.int32(NH), np.int32(HD),
        block=block, grid=grid,
    )
    return out


def fused_causal_attention_from_qkv(
    qkv: gpuarray.GPUArray,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    scale: float,
) -> tuple:
    """Attention from QKV [B*T, 3*C].

    Returns (attn_concat [B*T, C], probs flat, q_h, k_h, v_h) with heads
    in [B*NH, T, HD] for the backward path (avoids re-split).
    """
    B, T, NH, HD = int(batch_size), int(seq_len), int(num_heads), int(head_dim)
    C = NH * HD
    H = B * NH
    M, D = T, HD

    if _USE_FUSED_ATTENTION_FORWARD:
        q_h, k_h, v_h = split_heads_from_qkv(qkv, B, T, NH, HD)
        probs = gpuarray.empty((H, M, M), dtype=np.float32)
        out_h = gpuarray.empty((H, M, D), dtype=np.float32)
        row_max = gpuarray.empty((H, M), dtype=np.float32)
        row_sum = gpuarray.empty((H, M), dtype=np.float32)
        threads = next_pow2(max(D, min(M, 256)))
        shared_bytes = (D + M + threads) * np.dtype(np.float32).itemsize
        _fused_attn_fwd_kernel(
            q_h, k_h, v_h, probs, out_h, row_max, row_sum,
            np.int32(H), np.int32(M), np.int32(D), np.float32(scale),
            block=(threads, 1, 1), grid=(H, M, 1), shared=shared_bytes,
        )
        attn_concat = merge_heads(out_h, B, T, NH, HD)
        return attn_concat, probs.reshape(H * M * M), q_h, k_h, v_h

    # Default: corrected causal_mha (faster on Kepler) + head-layout cache.
    q, k, v = split_qkv(qkv, C)
    attn_concat, probs = causal_self_attention(q, k, v, B, T, NH, HD, scale)
    q_h = interleaved_to_heads(q, B, T, NH, HD)
    k_h = interleaved_to_heads(k, B, T, NH, HD)
    v_h = interleaved_to_heads(v, B, T, NH, HD)
    return attn_concat, probs, q_h, k_h, v_h


def softmax_backward(
    probs: gpuarray.GPUArray,
    d_probs: gpuarray.GPUArray,
    scale: float = 1.0,
) -> gpuarray.GPUArray:
    """Softmax backward for attention matrices [rows, T]. Returns d_scores on device."""
    assert probs.shape == d_probs.shape
    T = int(probs.shape[-1])
    total_rows = int(np.prod(probs.shape[:-1])) if probs.ndim > 1 else 1
    d_scores = gpuarray.empty_like(probs)
    threads = next_pow2(T)
    shared_bytes = threads * np.dtype(np.float32).itemsize
    _softmax_backward_kernel(
        probs, d_probs, d_scores,
        np.int32(T), np.int32(total_rows), np.float32(scale),
        block=(threads, 1, 1), grid=(total_rows, 1, 1), shared=shared_bytes,
    )
    return d_scores


def interleaved_to_heads(
    x: gpuarray.GPUArray, batch_size: int, seq_len: int, num_heads: int, head_dim: int,
) -> gpuarray.GPUArray:
    """[B*T, NH*HD] -> [B*NH, T, HD] contiguous."""
    B, T, NH, HD = int(batch_size), int(seq_len), int(num_heads), int(head_dim)
    out = gpuarray.empty((B * NH, T, HD), dtype=np.float32)
    n = B * T * NH * HD
    grid, block = _launch_1d(n)
    _interleaved_to_heads_kernel(
        x, out, np.int32(B), np.int32(T), np.int32(NH), np.int32(HD),
        block=block, grid=grid,
    )
    return out


def merge_heads(
    heads: gpuarray.GPUArray, batch_size: int, seq_len: int, num_heads: int, head_dim: int,
) -> gpuarray.GPUArray:
    """[B*NH, T, HD] -> [B*T, NH*HD]."""
    B, T, NH, HD = int(batch_size), int(seq_len), int(num_heads), int(head_dim)
    out = gpuarray.empty((B * T, NH * HD), dtype=np.float32)
    n = B * T * NH * HD
    grid, block = _launch_1d(n)
    _merge_heads_kernel(
        heads, out, np.int32(B), np.int32(T), np.int32(NH), np.int32(HD),
        block=block, grid=grid,
    )
    return out


def pack_qkv(q: gpuarray.GPUArray, k: gpuarray.GPUArray, v: gpuarray.GPUArray) -> gpuarray.GPUArray:
    """Pack [rows, C] Q/K/V into [rows, 3*C]."""
    rows, c = int(q.shape[0]), int(q.shape[1])
    out = gpuarray.empty((rows, 3 * c), dtype=np.float32)
    n = rows * c
    grid, block = _launch_1d(n)
    _pack_qkv_kernel(q, k, v, out, np.int32(rows), np.int32(c), block=block, grid=grid)
    return out


def attention_backward_heads(
    d_attn_concat: gpuarray.GPUArray,
    q: gpuarray.GPUArray,
    k: gpuarray.GPUArray,
    v: gpuarray.GPUArray,
    probs: gpuarray.GPUArray,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    scale: float,
    heads_layout: bool = False,
) -> tuple:
    """Attention backward over all heads in a few batched launches.

    If ``heads_layout`` is True, q/k/v are [B*NH, T, HD]; else [B*T, C].
    d_attn_concat is always [B*T, C].
    Returns (d_q, d_k, d_v) in the same layout as q/k/v.
    """
    B, T, NH, HD = int(batch_size), int(seq_len), int(num_heads), int(head_dim)
    H = B * NH
    M, D = T, HD

    if heads_layout:
        q_h, k_h, v_h = q, k, v
    else:
        q_h = interleaved_to_heads(q, B, T, NH, HD)
        k_h = interleaved_to_heads(k, B, T, NH, HD)
        v_h = interleaved_to_heads(v, B, T, NH, HD)
    d_out_h = interleaved_to_heads(d_attn_concat, B, T, NH, HD)
    probs_h = probs.reshape(H, M, M)

    d_probs = gpuarray.empty((H, M, M), dtype=np.float32)
    d_raw = gpuarray.empty((H, M, M), dtype=np.float32)
    row_sum = gpuarray.empty((H, M), dtype=np.float32)
    d_q_h = gpuarray.empty((H, M, D), dtype=np.float32)
    d_k_h = gpuarray.empty((H, M, D), dtype=np.float32)
    d_v_h = gpuarray.empty((H, M, D), dtype=np.float32)

    _gemm_batched_kernel(
        probs_h, d_out_h, d_v_h,
        np.int32(M), np.int32(D), np.int32(M), np.int32(H),
        np.int32(1), np.int32(0),
        block=(16, 16, 1),
        grid=(int(np.ceil(D / 16)), int(np.ceil(M / 16)), H),
    )
    _gemm_batched_kernel(
        d_out_h, v_h, d_probs,
        np.int32(M), np.int32(M), np.int32(D), np.int32(H),
        np.int32(0), np.int32(1),
        block=(16, 16, 1),
        grid=(int(np.ceil(M / 16)), int(np.ceil(M / 16)), H),
    )

    sm_threads = next_pow2(min(M, 256))
    sm_shared = sm_threads * np.dtype(np.float32).itemsize
    _softmax_fused_backward_kernel(
        d_probs, probs_h, row_sum, d_raw,
        np.int32(H), np.int32(M),
        block=(sm_threads, 1, 1), grid=(H, M, 1), shared=sm_shared,
    )
    scal_mul(d_raw, scale)

    _gemm_batched_kernel(
        d_raw, k_h, d_q_h,
        np.int32(M), np.int32(D), np.int32(M), np.int32(H),
        np.int32(0), np.int32(0),
        block=(16, 16, 1),
        grid=(int(np.ceil(D / 16)), int(np.ceil(M / 16)), H),
    )
    _gemm_batched_kernel(
        d_raw, q_h, d_k_h,
        np.int32(M), np.int32(D), np.int32(M), np.int32(H),
        np.int32(1), np.int32(0),
        block=(16, 16, 1),
        grid=(int(np.ceil(D / 16)), int(np.ceil(M / 16)), H),
    )

    if heads_layout:
        return d_q_h, d_k_h, d_v_h
    return (
        merge_heads(d_q_h, B, T, NH, HD),
        merge_heads(d_k_h, B, T, NH, HD),
        merge_heads(d_v_h, B, T, NH, HD),
    )


def add_block(
    acc: gpuarray.GPUArray,
    block: gpuarray.GPUArray,
    row0: int,
    col_start: int,
    C: int,
    hd: int,
) -> None:
    """Accumulate [block_rows, hd] block into acc[row0:row0+block_rows, col_start:col_start+hd]."""
    block_rows = int(block.shape[0])
    n = block_rows * hd
    threads = 256
    blocks = int(np.ceil(n / threads))
    _add_block_kernel(
        acc, block, np.int32(row0), np.int32(C), np.int32(col_start), np.int32(hd), np.int32(block_rows),
        block=(threads, 1, 1), grid=(blocks, 1, 1),
    )


def add_inplace(acc: gpuarray.GPUArray, block: gpuarray.GPUArray) -> None:
    """acc += block elementwise (same shape)."""
    assert acc.shape == block.shape
    n = np.int32(acc.size)
    threads = 256
    blocks = int(np.ceil(int(n) / threads))
    _add_inplace_kernel(acc, block, n, block=(threads, 1, 1), grid=(blocks, 1, 1))


def grad_global_norm_sq(grads) -> float:
    """Sum of squares of all gradient tensors on device (single scalar D2H)."""
    buf = _zeros_gpu((1,))
    threads = 256
    for g in grads.values():
        n = np.int32(g.size)
        blocks = int(np.ceil(int(n) / threads))
        _grad_norm_contrib_kernel(g, buf, n, block=(threads, 1, 1), grid=(blocks, 1, 1))
    return float(buf.get()[0])


def layernorm_with_cache(
    x: gpuarray.GPUArray, gamma: gpuarray.GPUArray, beta: gpuarray.GPUArray, eps: float = 1e-5,
) -> tuple:
    """Layernorm on device; returns (y, xhat, invstd_row) all on GPU."""
    hidden_dim = int(x.shape[-1])
    total_rows = int(np.prod(x.shape[:-1])) if x.ndim > 1 else 1
    y = gpuarray.empty_like(x)
    xhat = gpuarray.empty_like(x)
    invstd_row = gpuarray.empty((total_rows,), dtype=np.float32)
    threads = next_pow2(hidden_dim)
    shared_bytes = threads * np.dtype(np.float32).itemsize
    _layernorm_cache_kernel(
        x, y, xhat, invstd_row, gamma, beta,
        np.int32(hidden_dim), np.float32(eps), np.int32(total_rows),
        block=(threads, 1, 1), grid=(total_rows, 1, 1), shared=shared_bytes,
    )
    return y, xhat, invstd_row


def _zeros_gpu(shape, dtype=np.float32) -> gpuarray.GPUArray:
    return to_device(np.zeros(shape, dtype=dtype))


def layernorm_backward(
    dout: gpuarray.GPUArray,
    xhat: gpuarray.GPUArray,
    invstd_row: gpuarray.GPUArray,
    gamma: gpuarray.GPUArray,
) -> tuple:
    """Backward pass for layernorm_with_cache. Returns (dx, dgamma, dbeta) on device."""
    hidden_dim = int(xhat.shape[-1])
    total_rows = int(np.prod(xhat.shape[:-1])) if xhat.ndim > 1 else 1
    dx = gpuarray.empty_like(xhat)
    dgamma = _zeros_gpu((hidden_dim,))
    dbeta = _zeros_gpu((hidden_dim,))
    threads = next_pow2(hidden_dim)
    shared_bytes = threads * np.dtype(np.float32).itemsize
    _layernorm_backward_kernel(
        dout, xhat, invstd_row, gamma, dx, dgamma, dbeta,
        np.int32(hidden_dim), np.int32(total_rows),
        block=(threads, 1, 1), grid=(total_rows, 1, 1), shared=shared_bytes,
    )
    return dx, dgamma, dbeta


def gelu_backward(x: gpuarray.GPUArray, d_out: gpuarray.GPUArray) -> gpuarray.GPUArray:
    """GeLU backward on device."""
    n_elements = np.int32(x.size)
    d_x = gpuarray.empty_like(x)
    threads = 256
    blocks = int(np.ceil(int(n_elements) / threads))
    _gelu_backward_kernel(x, d_out, d_x, n_elements, block=(threads, 1, 1), grid=(blocks, 1, 1))
    return d_x


def cross_entropy(logits: gpuarray.GPUArray, targets: np.ndarray) -> tuple:
    """Mean cross-entropy on device. logits [rows, V], targets [rows] int.

    Returns (loss float, dlogits [rows, V] on device).
    """
    rows, vocab_size = int(logits.shape[0]), int(logits.shape[1])
    targets_d = gpuarray.to_gpu(np.ascontiguousarray(targets, dtype=np.int32))
    d_logits = gpuarray.empty_like(logits)
    loss_buf = _zeros_gpu((1,))
    threads = next_pow2(vocab_size)
    shared_bytes = threads * np.dtype(np.float32).itemsize
    _cross_entropy_kernel(
        logits, targets_d, d_logits, loss_buf,
        np.int32(vocab_size), np.int32(rows),
        block=(threads, 1, 1), grid=(rows, 1, 1), shared=shared_bytes,
    )
    return float(loss_buf.get()[0]) / rows, d_logits


def scal_mul(arr: gpuarray.GPUArray, scale: float) -> gpuarray.GPUArray:
    """Multiply device array by scalar in-place."""
    n = np.int32(arr.size)
    threads = 256
    blocks = int(np.ceil(int(n) / threads))
    _scal_mul_kernel(arr, np.float32(scale), n, block=(threads, 1, 1), grid=(blocks, 1, 1))
    return arr


def adamw_update(
    w: gpuarray.GPUArray,
    g: gpuarray.GPUArray,
    m: gpuarray.GPUArray,
    v: gpuarray.GPUArray,
    lr: float,
    wd: float,
    b1: float,
    b2: float,
    eps: float,
    bc1: float,
    bc2: float,
) -> None:
    n = np.int32(w.size)
    threads = 256
    blocks = int(np.ceil(int(n) / threads))
    _adamw_update_kernel(
        w, g, m, v,
        np.float32(lr), np.float32(wd), np.float32(b1), np.float32(b2), np.float32(eps),
        np.float32(bc1), np.float32(bc2), n,
        block=(threads, 1, 1), grid=(blocks, 1, 1),
    )


def embedding_lookup(ids: np.ndarray, emb: gpuarray.GPUArray, pos_emb: gpuarray.GPUArray, T: int) -> gpuarray.GPUArray:
    """Build [B*T, C] input embeddings on device from token ids [B, T]."""
    ids = np.asarray(ids, dtype=np.int32)
    B = int(ids.shape[0])
    C = int(emb.shape[1])
    out = gpuarray.empty((B * T, C), dtype=np.float32)
    ids_d = gpuarray.to_gpu(np.ascontiguousarray(ids.reshape(-1), dtype=np.int32))
    n = B * T * C
    threads = 256
    blocks = int(np.ceil(n / threads))
    _embed_forward_kernel(
        emb, pos_emb, ids_d, out, np.int32(B), np.int32(T), np.int32(C),
        block=(threads, 1, 1), grid=(blocks, 1, 1),
    )
    return out


def embed_backward(
    ids: np.ndarray, d_h: gpuarray.GPUArray, vocab_size: int, embed_dim: int,
) -> tuple:
    """Returns (d_token_embedding, d_position_embedding) on device."""
    B, T = ids.shape
    C = embed_dim
    d_tok = _zeros_gpu((vocab_size, C))
    d_pos = _zeros_gpu((T, C))
    ids_d = gpuarray.to_gpu(np.ascontiguousarray(ids, dtype=np.int32))
    threads = min(next_pow2(C), MAX_THREADS_PER_BLOCK)
    _embed_backward_kernel(
        d_tok, ids_d, d_h, np.int32(B), np.int32(T), np.int32(C),
        block=(threads, 1, 1), grid=(B, T, 1),
    )
    _pos_embed_backward_kernel(
        d_pos, d_h, np.int32(B), np.int32(T), np.int32(C),
        block=(threads, 1, 1), grid=(T, 1, 1),
    )
    return d_tok, d_pos


def sync_to_host(device_arr: gpuarray.GPUArray, host_arr: np.ndarray) -> None:
    """Copy device array into existing host buffer in-place (no reallocation)."""
    host_arr[:] = device_arr.get().reshape(host_arr.shape)
