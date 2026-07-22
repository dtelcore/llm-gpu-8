"""Parity: LayerNorm forward + backward."""

from __future__ import annotations

import numpy as np

from model.gpt import _layernorm_backward
from tests.parity._common import B, C, CudaTestCase, T, assert_close, layernorm_np


class TestLayerNormParity(CudaTestCase):
    def test_layernorm_forward_backward(self) -> None:
        ops = self.cuda_ops
        rng = np.random.default_rng(2)
        X = rng.standard_normal((B * T, C), dtype=np.float32)
        gamma = rng.standard_normal((C,), dtype=np.float32) * 0.1 + 1.0
        beta = rng.standard_normal((C,), dtype=np.float32) * 0.01
        dout = rng.standard_normal((B * T, C), dtype=np.float32)

        out_ref, xhat_ref, invstd_ref = layernorm_np(X, gamma, beta)
        dx_ref, dg_ref, db_ref = _layernorm_backward(dout, xhat_ref, invstd_ref.reshape(-1, 1), gamma)

        Xd = ops.to_device(X)
        gd = ops.to_device(gamma)
        bd = ops.to_device(beta)
        out_d, xhat_d, invstd_d = ops.layernorm_with_cache(Xd, gd, bd)
        assert_close("layernorm.out", ops.to_host(out_d), out_ref)

        dout_d = ops.to_device(dout)
        dx_d, dg_d, db_d = ops.layernorm_backward(dout_d, xhat_d, invstd_d, gd)
        assert_close("layernorm.dx", ops.to_host(dx_d), dx_ref.astype(np.float32))
        assert_close("layernorm.dgamma", ops.to_host(dg_d), dg_ref.astype(np.float32))
        assert_close("layernorm.dbeta", ops.to_host(db_d), db_ref.astype(np.float32))
