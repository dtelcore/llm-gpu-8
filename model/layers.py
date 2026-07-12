"""
model/layers.py

Attention and MLP block ops, built on top of model/cuda/ops.py primitives.
Each function accepts a NumPy [rows, C] activation, uploads *that* to the
device, runs the relevant kernels, and returns a NumPy array -- keeping
model/gpt.py free of any direct PyCUDA calls.

V2: `weight`/`bias` arguments are now GPU-resident gpuarrays (from
model.weights.ModelParameters.device_weights/device_biases), not NumPy
arrays. Previously every call re-uploaded the full weight matrix from host
to device -- on the GT 730's slow PCIe link that dwarfed the actual matmul
cost. Weights now live on the device permanently and are only re-synced
once per optimizer step (see ModelParameters.sync_device), not once per op.
Only the activation `x` still crosses the host<->device boundary here,
since the analytic NumPy backward pass needs it cached on the host anyway.
"""

from typing import Optional, Tuple

import numpy as np

from model.cuda import ops
from model.trace import TraceContext


def linear(x: np.ndarray, weight, bias=None, tracer: TraceContext = None, name: str = "linear") -> np.ndarray:
    """y = x @ weight + bias, executed on the GPU. `weight`/`bias` must already be
    device-resident gpuarrays (see ModelParameters.device_weights/device_biases)."""
    xd = ops.to_device(x)
    out = ops.matmul(xd, weight, tracer=tracer, name=name)
    if bias is not None:
        out = ops.add_bias(out, bias)
    return ops.to_host(out)


def layernorm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    xd = ops.to_device(x)
    gd = ops.to_device(gamma)
    bd = ops.to_device(beta)
    out = ops.layernorm(xd, gd, bd, eps=eps)
    return ops.to_host(out)


def gelu(x: np.ndarray) -> np.ndarray:
    xd = ops.to_device(x)
    out = ops.gelu(xd)
    return ops.to_host(out)


def softmax(x: np.ndarray) -> np.ndarray:
    xd = ops.to_device(x)
    out = ops.softmax(xd)
    return ops.to_host(out)


def causal_self_attention(
    x: np.ndarray,
    qkv_weight: np.ndarray,
    qkv_bias: np.ndarray,
    out_weight: np.ndarray,
    out_bias: np.ndarray,
    num_heads: int,
    tracer: TraceContext = None,
) -> np.ndarray:
    """Single-sequence causal self-attention. x: [T, C]."""
    T, C = x.shape
    head_dim = C // num_heads

    qkv = linear(x, qkv_weight, qkv_bias, tracer=tracer, name="qkv_proj")  # [T, 3C]
    q, k, v = np.split(qkv, 3, axis=-1)  # each [T, C]

    q = q.reshape(T, num_heads, head_dim).transpose(1, 0, 2)  # [H, T, hd]
    k = k.reshape(T, num_heads, head_dim).transpose(1, 0, 2)
    v = v.reshape(T, num_heads, head_dim).transpose(1, 0, 2)

    causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scale = 1.0 / np.sqrt(head_dim)

    head_outputs = np.empty((num_heads, T, head_dim), dtype=np.float32)
    for h in range(num_heads):
        scores = (q[h] @ k[h].T) * scale  # [T, T], small enough for host matmul
        scores = np.where(causal_mask, -1e9, scores).astype(np.float32)
        probs = softmax(scores)  # GPU softmax kernel
        head_outputs[h] = probs @ v[h]

    attn_out = head_outputs.transpose(1, 0, 2).reshape(T, C)
    return linear(attn_out, out_weight, out_bias, tracer=tracer, name="attn_out_proj")


def mlp_block(
    x: np.ndarray,
    expand_weight: np.ndarray,
    expand_bias: np.ndarray,
    contract_weight: np.ndarray,
    contract_bias: np.ndarray,
    tracer: TraceContext = None,
) -> np.ndarray:
    hidden = linear(x, expand_weight, expand_bias, tracer=tracer, name="mlp_expand")
    hidden = gelu(hidden)
    return linear(hidden, contract_weight, contract_bias, tracer=tracer, name="mlp_contract")
