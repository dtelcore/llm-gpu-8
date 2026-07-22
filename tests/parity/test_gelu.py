"""Parity: GELU forward + backward."""

from __future__ import annotations

import numpy as np

from tests.parity._common import CudaTestCase, assert_close, gelu_grad_np, gelu_np


class TestGeluParity(CudaTestCase):
    def test_gelu_forward_backward(self) -> None:
        ops = self.cuda_ops
        rng = np.random.default_rng(3)
        X = rng.standard_normal((64, 32), dtype=np.float32)
        d_out = rng.standard_normal(X.shape, dtype=np.float32)

        y_ref = gelu_np(X)
        dx_ref = gelu_grad_np(X) * d_out

        Xd = ops.to_device(X)
        y = ops.to_host(ops.gelu(Xd))
        assert_close("gelu.forward", y, y_ref)

        d_out_d = ops.to_device(d_out)
        dx = ops.to_host(ops.gelu_backward(Xd, d_out_d))
        assert_close("gelu.backward", dx, dx_ref.astype(np.float32))
