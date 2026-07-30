"""Parity: LayerNorm / RMSNorm forward + backward (+ residual fuse)."""

from __future__ import annotations

import numpy as np

from model.gpt import _layernorm_backward, _rmsnorm_backward, _rmsnorm_cache
from tests.parity._common import B, C, CudaTestCase, T, assert_close, layernorm_np


def rmsnorm_np(x: np.ndarray, gamma: np.ndarray, eps: float = 1e-5):
    y, xhat, inv = _rmsnorm_cache(x, gamma, eps=eps)
    return y, xhat, inv.reshape(-1)


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

    def test_rmsnorm_forward_backward(self) -> None:
        ops = self.cuda_ops
        rng = np.random.default_rng(3)
        X = rng.standard_normal((B * T, C), dtype=np.float32)
        gamma = rng.standard_normal((C,), dtype=np.float32) * 0.1 + 1.0
        dout = rng.standard_normal((B * T, C), dtype=np.float32)

        out_ref, xhat_ref, inv_ref = rmsnorm_np(X, gamma)
        dx_ref, dg_ref = _rmsnorm_backward(dout, xhat_ref, inv_ref.reshape(-1, 1), gamma)

        Xd = ops.to_device(X)
        gd = ops.to_device(gamma)
        out_d, xhat_d, inv_d = ops.rmsnorm_with_cache(Xd, gd)
        assert_close("rmsnorm.out", ops.to_host(out_d), out_ref)
        assert_close("rmsnorm.xhat", ops.to_host(xhat_d), xhat_ref)

        dout_d = ops.to_device(dout)
        dx_d, dg_d = ops.rmsnorm_backward(dout_d, xhat_d, inv_d, gd)
        assert_close("rmsnorm.dx", ops.to_host(dx_d), dx_ref.astype(np.float32))
        assert_close("rmsnorm.dgamma", ops.to_host(dg_d), dg_ref.astype(np.float32))

    def test_residual_rmsnorm_fuse(self) -> None:
        ops = self.cuda_ops
        rng = np.random.default_rng(4)
        X = rng.standard_normal((B * T, C), dtype=np.float32)
        R = rng.standard_normal((B * T, C), dtype=np.float32)
        gamma = rng.standard_normal((C,), dtype=np.float32) * 0.1 + 1.0

        x_out_ref = X + R
        y_ref, xhat_ref, inv_ref = rmsnorm_np(x_out_ref, gamma)

        Xd, Rd, gd = ops.to_device(X), ops.to_device(R), ops.to_device(gamma)
        x_out_d, y_d, xhat_d, inv_d = ops.residual_rmsnorm_with_cache(Xd, Rd, gd)
        assert_close("residual_rmsnorm.x_out", ops.to_host(x_out_d), x_out_ref.astype(np.float32))
        assert_close("residual_rmsnorm.y", ops.to_host(y_d), y_ref)
        assert_close("residual_rmsnorm.xhat", ops.to_host(xhat_d), xhat_ref)
