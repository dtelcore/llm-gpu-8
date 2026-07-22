"""Parity: causal attention forward (host reference vs GPU)."""

from __future__ import annotations

import numpy as np

from model.gpt import _batched_attention_host
from tests.parity._common import B, C, H, HD, CudaTestCase, T, assert_close


class TestAttentionParity(CudaTestCase):
    def test_causal_attention_forward(self) -> None:
        ops = self.cuda_ops
        rng = np.random.default_rng(4)
        scale = 1.0 / np.sqrt(HD)
        # causal_self_attention expects interleaved Q/K/V as [B*T, C]
        q = rng.standard_normal((B * T, C), dtype=np.float32) * 0.1
        k = rng.standard_normal((B * T, C), dtype=np.float32) * 0.1
        v = rng.standard_normal((B * T, C), dtype=np.float32) * 0.1
        qkv = np.concatenate([q, k, v], axis=-1)

        attn_ref, probs_ref, _, _, _ = _batched_attention_host(qkv, B, T, H, HD, scale)

        qd = ops.to_device(q)
        kd = ops.to_device(k)
        vd = ops.to_device(v)
        attn_d, probs_d = ops.causal_self_attention(qd, kd, vd, B, T, H, HD, scale)
        assert_close("attention.out", ops.to_host(attn_d), attn_ref, rtol=2e-4, atol=2e-5)

        probs_h = ops.to_host(probs_d).reshape(B, H, T, T)
        assert_close("attention.probs", probs_h, probs_ref, rtol=2e-4, atol=2e-5)
