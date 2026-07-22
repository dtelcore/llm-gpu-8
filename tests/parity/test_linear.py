"""Parity: linear / matmul_bias + linear_backward."""

from __future__ import annotations

import numpy as np

from tests.parity._common import B, C, CudaTestCase, T, assert_close


class TestLinearParity(CudaTestCase):
    def test_matmul_bias_forward(self) -> None:
        ops = self.cuda_ops
        rng = np.random.default_rng(0)
        M, K, N = B * T, C, C
        A = rng.standard_normal((M, K), dtype=np.float32)
        W = rng.standard_normal((K, N), dtype=np.float32) * 0.1
        bias = rng.standard_normal((N,), dtype=np.float32) * 0.01
        expected = A @ W + bias

        Ad = ops.to_device(A)
        Wd = ops.to_device(W)
        bd = ops.to_device(bias)
        out = ops.to_host(ops.matmul_bias(Ad, Wd, bd))
        assert_close("matmul_bias", out, expected)

    def test_linear_backward(self) -> None:
        ops = self.cuda_ops
        rng = np.random.default_rng(1)
        M, K, N = B * T, C, C
        X = rng.standard_normal((M, K), dtype=np.float32)
        W = rng.standard_normal((K, N), dtype=np.float32) * 0.1
        dY = rng.standard_normal((M, N), dtype=np.float32)

        dX_ref = dY @ W.T
        dW_ref = X.T @ dY
        db_ref = dY.sum(axis=0)

        dYd = ops.to_device(dY)
        Xd = ops.to_device(X)
        Wd = ops.to_device(W)
        dX_d, dW_d, db_d = ops.linear_backward(dYd, Xd, Wd)
        assert_close("linear_backward.dX", ops.to_host(dX_d), dX_ref)
        assert_close("linear_backward.dW", ops.to_host(dW_d), dW_ref)
        assert_close("linear_backward.db", ops.to_host(db_d), db_ref)
