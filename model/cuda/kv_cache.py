"""
model/cuda/kv_cache.py

Stage 4.1: static device KV arenas sized from GPTConfig.max_len.
Prefill packs GPU K/V into fixed [B*H, max_len, hd] buffers; decode appends
rows in-place without host concatenate / realloc.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pycuda.gpuarray as gpuarray

from model.cuda import ops as cuda_ops


def alloc_layer_arena(batch_heads: int, max_len: int, head_dim: int) -> Dict[str, gpuarray.GPUArray]:
    """Allocate one layer's K/V arenas (FP32)."""
    shape = (int(batch_heads), int(max_len), int(head_dim))
    return {
        "k_d": gpuarray.empty(shape, dtype=np.float32),
        "v_d": gpuarray.empty(shape, dtype=np.float32),
    }


def pack_prefill_into_arena(
    k_src: gpuarray.GPUArray,
    v_src: gpuarray.GPUArray,
    arena: Dict[str, gpuarray.GPUArray],
    *,
    batch_heads: int,
    seq_len: int,
    max_len: int,
    head_dim: int,
) -> None:
    """Copy prefill heads [BH, T, hd] into arena slots [0 .. T)."""
    cuda_ops.kv_pack_prefill(
        k_src, arena["k_d"],
        batch_heads=batch_heads, seq_len=seq_len, max_len=max_len, head_dim=head_dim,
    )
    cuda_ops.kv_pack_prefill(
        v_src, arena["v_d"],
        batch_heads=batch_heads, seq_len=seq_len, max_len=max_len, head_dim=head_dim,
    )


def append_kv_row(
    k_new: gpuarray.GPUArray,
    v_new: gpuarray.GPUArray,
    arena: Dict[str, gpuarray.GPUArray],
    *,
    batch_heads: int,
    t: int,
    max_len: int,
    head_dim: int,
) -> None:
    """Write new K/V rows at index ``t`` (host knows T; no cache readback)."""
    cuda_ops.kv_append_row(
        k_new, arena["k_d"],
        batch_heads=batch_heads, t=t, max_len=max_len, head_dim=head_dim,
    )
    cuda_ops.kv_append_row(
        v_new, arena["v_d"],
        batch_heads=batch_heads, t=t, max_len=max_len, head_dim=head_dim,
    )


def arena_nbytes(kv_state: Dict) -> int:
    """Byte size of device arenas (or host arrays for legacy state)."""
    total = 0
    for layer in kv_state.get("layers", []):
        if "k_d" in layer:
            total += int(layer["k_d"].nbytes) + int(layer["v_d"].nbytes)
        else:
            total += int(layer["k"].nbytes) + int(layer["v"].nbytes)
    return total


def build_device_kv_state(
    cache: Dict,
    *,
    max_len: int,
    num_heads: int,
    head_dim: int,
) -> Dict:
    """Build generate-only device KV state from a GPU forward cache."""
    B = int(cache["B"])
    T = int(cache["T"])
    H = int(num_heads)
    hd = int(head_dim)
    BH = B * H
    layers: List[Dict] = []
    for layer_cache in cache["layers"]:
        attn = layer_cache["attn"]
        arena = alloc_layer_arena(BH, max_len, hd)
        if attn.get("gpu") and "k_d" in attn:
            pack_prefill_into_arena(
                attn["k_d"], attn["v_d"], arena,
                batch_heads=BH, seq_len=T, max_len=max_len, head_dim=hd,
            )
        else:
            # Host fallback: upload then pack via temporary device buffers.
            k_h = attn["k_h"].reshape(BH, T, hd).astype(np.float32, copy=False)
            v_h = attn["v_h"].reshape(BH, T, hd).astype(np.float32, copy=False)
            k_tmp = cuda_ops.to_device(np.ascontiguousarray(k_h))
            v_tmp = cuda_ops.to_device(np.ascontiguousarray(v_h))
            pack_prefill_into_arena(
                k_tmp, v_tmp, arena,
                batch_heads=BH, seq_len=T, max_len=max_len, head_dim=hd,
            )
        layers.append(arena)
    return {
        "layers": layers,
        "T": T,
        "B": B,
        "device": True,
        "max_len": int(max_len),
        "num_heads": H,
        "head_dim": hd,
    }


def clone_device_kv_state(kv_state: Dict) -> Dict:
    """Deep-copy device arenas (for graph probe without mutating live state)."""
    if not kv_state.get("device"):
        return {
            "B": kv_state["B"],
            "T": kv_state["T"],
            "layers": [
                {"k": ly["k"].copy(), "v": ly["v"].copy()}
                for ly in kv_state["layers"]
            ],
        }
    import pycuda.driver as cuda

    layers = []
    for ly in kv_state["layers"]:
        k_new = gpuarray.empty_like(ly["k_d"])
        v_new = gpuarray.empty_like(ly["v_d"])
        cuda.memcpy_dtod(k_new.gpudata, ly["k_d"].gpudata, ly["k_d"].nbytes)
        cuda.memcpy_dtod(v_new.gpudata, ly["v_d"].gpudata, ly["v_d"].nbytes)
        layers.append({"k_d": k_new, "v_d": v_new})
    out = dict(kv_state)
    out["layers"] = layers
    return out
