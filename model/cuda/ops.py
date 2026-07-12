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
