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

logger.info("PyCUDA kernels compiled for sm_35 (Kepler GT 730)")

MAX_THREADS_PER_BLOCK = 1024


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
    """C = A @ B for 2D row-major float32 device arrays."""
    assert A.shape[1] == B.shape[0], f"Shape mismatch: {A.shape} vs {B.shape}"

    M, K = np.int32(A.shape[0]), np.int32(A.shape[1])
    N = np.int32(B.shape[1])

    C = gpuarray.empty((int(M), int(N)), dtype=np.float32)

    block_dim = (16, 16, 1)
    grid_dim = (int(np.ceil(int(N) / 16)), int(np.ceil(int(M) / 16)), 1)

    if tracer is not None and getattr(tracer, "trace_vectorization", False):
        tracer.log_vectorization(name, (int(M), int(K)), (int(K), int(N)), (int(M), int(N)), grid_dim, block_dim)

    _gemm_kernel(A, B, C, M, N, K, block=block_dim, grid=grid_dim)
    return C


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
    """Fused causal MHA on device. Q/K/V are [B*T, C] with C = H*hd.

    Returns (attn_concat [B*T, C], probs [B, H, T, T] on device).
    """
    B, T, H, hd = int(batch_size), int(seq_len), int(num_heads), int(head_dim)
    C = H * hd
    out = gpuarray.empty((B * T, C), dtype=np.float32)
    probs = gpuarray.empty((B * H * T * T,), dtype=np.float32)

    shared_bytes = (2 * T + hd) * np.dtype(np.float32).itemsize
    grid = (T, H, B)
    block = (hd, 1, 1)

    _causal_mha_kernel(
        q, k, v, out, probs,
        np.int32(B), np.int32(T), np.int32(H), np.int32(hd), np.float32(scale),
        block=block, grid=grid, shared=shared_bytes,
    )
    return out, probs


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


def linear_backward(
    dout: gpuarray.GPUArray,
    x: gpuarray.GPUArray,
    weight: gpuarray.GPUArray,
) -> tuple:
    """Backward for y = x @ weight + b. weight is [in, out]. Returns (d_x, d_weight, d_bias)."""
    d_weight = matmul(x.T, dout)
    d_bias = to_device(to_host(dout).sum(axis=0).astype(np.float32))
    d_x = matmul(dout, weight.T)
    return d_x, d_weight, d_bias


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
