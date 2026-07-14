"""
model/layers.py

Device-native layer primitives built on model/cuda/ops.py.

V3: activations stay on the GPU between ops. Each function accepts
device-resident gpuarrays and returns device arrays — no host round-trips.
Callers (model/gpt.py) upload once at the forward entry and download only
what backward / logging needs, not after every kernel.
"""

from typing import Optional

import numpy as np

from model.cuda import ops
from model.trace import TraceContext


def linear(xd, weight, bias=None, tracer: TraceContext = None, name: str = "linear"):
    """y = x @ weight + bias on device. `weight`/`bias` must be device-resident."""
    out = ops.matmul(xd, weight, tracer=tracer, name=name)
    if bias is not None:
        out = ops.add_bias(out, bias)
    return out


def layernorm(xd, gamma, beta, eps: float = 1e-5):
    """Layernorm on device. gamma/beta must be device-resident 1-D arrays."""
    return ops.layernorm(xd, gamma, beta, eps=eps)


def gelu(xd):
    """GeLU on device."""
    return ops.gelu(xd)


def softmax(xd):
    """Softmax over the last dimension on device."""
    return ops.softmax(xd)


def add_residual(a, b):
    """Elementwise residual add on device."""
    return ops.add_arrays(a, b)


def mlp_block_device(
    xd,
    expand_weight,
    expand_bias,
    contract_weight,
    contract_bias,
    tracer: TraceContext = None,
):
    """FFN block entirely on device: linear -> gelu -> linear."""
    hidden = linear(xd, expand_weight, expand_bias, tracer=tracer, name="mlp_expand")
    hidden = gelu(hidden)
    return linear(hidden, contract_weight, contract_bias, tracer=tracer, name="mlp_contract")


def causal_self_attention_device(
    qkv_d,
    batch_size: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    scale: float,
):
    """Fused GPU causal attention from a [B*T, 3*C] qkv projection."""
    c = num_heads * head_dim
    q_d, k_d, v_d = ops.split_qkv(qkv_d, c)
    return ops.causal_self_attention(
        q_d, k_d, v_d, batch_size, seq_len, num_heads, head_dim, scale,
    )
