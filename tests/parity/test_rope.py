"""Parity: RoPE apply + backward vs NumPy."""

from __future__ import annotations

import numpy as np

from model.gpt import _rope_np
from tests.parity._common import CudaTestCase, HD, assert_close

BH = 4
T = 8
BASE = 10000.0


class TestRoPEParity(CudaTestCase):
    def test_rope_apply_forward_backward(self) -> None:
        ops = self.cuda_ops
        rng = np.random.default_rng(5)
        X = rng.standard_normal((BH, T, HD), dtype=np.float32)

        ref = _rope_np(X, base=BASE, pos_offset=0, backward=False)
        Xd = ops.to_device(X.copy())
        ops.rope_apply_inplace(
            Xd, batch_heads=BH, seq_len=T, head_dim=HD, base=BASE, pos_offset=0, backward=False,
        )
        assert_close("rope.forward", ops.to_host(Xd), ref)

        # Adjoint: rotate by -theta; applying forward then backward recovers X.
        Y = ref.copy()
        back = _rope_np(Y, base=BASE, pos_offset=0, backward=True)
        assert_close("rope.roundtrip_np", back, X, rtol=1e-5, atol=1e-6)

        Yd = ops.to_device(ref.copy())
        ops.rope_apply_inplace(
            Yd, batch_heads=BH, seq_len=T, head_dim=HD, base=BASE, pos_offset=0, backward=True,
        )
        assert_close("rope.roundtrip_cuda", ops.to_host(Yd), X, rtol=1e-5, atol=1e-6)

    def test_rope_pos_offset(self) -> None:
        ops = self.cuda_ops
        rng = np.random.default_rng(6)
        X = rng.standard_normal((BH, 1, HD), dtype=np.float32)
        offset = 3
        ref = _rope_np(X, base=BASE, pos_offset=offset)
        Xd = ops.to_device(X.copy())
        ops.rope_apply_inplace(
            Xd, batch_heads=BH, seq_len=1, head_dim=HD, base=BASE, pos_offset=offset,
        )
        assert_close("rope.offset", ops.to_host(Xd), ref)
