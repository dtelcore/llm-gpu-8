"""
model/cuda/fp16_storage.py

Stage 3.5 / 3.10: store selected forward-cache activations as FP16 on device;
cast back to FP32 before existing Kepler *_fp32 kernels / backward use.

Casts use device kernels (float_to_half / half_to_float) — no host round-trip.
Compute math remains FP32 (sm_35 has no useful AMP GEMM path).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
from pycuda import gpuarray

from model.cuda import ops as cuda_ops

# Keys commonly large in BiggerTest activation accounting.
DEFAULT_FP16_KEYS = (
    "ln1_out_d", "ln1_xhat_d", "ln2_out_d", "ln2_xhat_d",
    "attn_concat_d", "probs_d", "q_d", "k_d", "v_d",
    "hidden_d", "act_d", "h_final_d",
    # logits_d excluded: train reads it for loss before backward expand
)

_ENABLED = False


def set_fp16_activation_storage(enabled: bool) -> None:
    global _ENABLED
    _ENABLED = bool(enabled)


def fp16_storage_enabled() -> bool:
    return _ENABLED


def _is_gpuarray(x: Any) -> bool:
    return hasattr(x, "gpudata") and hasattr(x, "dtype")


def to_fp16_storage(arr: gpuarray.GPUArray) -> gpuarray.GPUArray:
    """Device float32 → float16 for activation storage."""
    return cuda_ops.float_to_half(arr)


def to_fp32_compute(arr: gpuarray.GPUArray) -> gpuarray.GPUArray:
    """Device float16 → float32 for existing compute / backward kernels."""
    return cuda_ops.half_to_float(arr)


def compress_cache_fp16(cache: Dict, keys: Iterable[str] = DEFAULT_FP16_KEYS) -> Dict[str, int]:
    """In-place: convert matching GPU arrays in forward cache to FP16 storage.

    Returns mapping of key→saved_bytes (fp32_bytes - fp16_bytes).
    """
    if not _ENABLED:
        return {}
    keyset = set(keys)
    saved: Dict[str, int] = {}

    def visit(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in list(node.items()):
                child = f"{path}.{k}" if path else str(k)
                if k in keyset and _is_gpuarray(v) and v.dtype == np.float32:
                    before = int(v.nbytes)
                    node[k] = to_fp16_storage(v)
                    saved[child] = before - int(node[k].nbytes)
                else:
                    visit(v, child)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                visit(v, f"{path}[{i}]")

    visit(cache, "")
    cache["_fp16_storage"] = True
    return saved


def expand_cache_fp32(cache: Dict, keys: Iterable[str] = DEFAULT_FP16_KEYS) -> None:
    """In-place: cast FP16-stored cache tensors back to FP32 for backward."""
    if not cache.get("_fp16_storage"):
        return
    keyset = set(keys)

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for k, v in list(node.items()):
                if k in keyset and _is_gpuarray(v) and v.dtype == np.float16:
                    node[k] = to_fp32_compute(v)
                else:
                    visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)

    visit(cache)
    cache["_fp16_storage"] = False


def estimate_savings_bytes(cache: Dict, keys: Iterable[str] = DEFAULT_FP16_KEYS) -> int:
    """Half of FP32 nbytes for matching tensors (theoretical)."""
    keyset = set(keys)
    total = 0

    def visit(node: Any) -> None:
        nonlocal total
        if isinstance(node, dict):
            for k, v in node.items():
                if k in keyset and _is_gpuarray(v) and v.dtype == np.float32:
                    total += int(v.nbytes) // 2
                else:
                    visit(v)
        elif isinstance(node, list):
            for v in node:
                visit(v)

    visit(cache)
    return total
