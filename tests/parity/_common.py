"""
tests/parity/_common.py

Shared helpers for NumPy (reference) vs CUDA (DUT) comparisons.
"""

from __future__ import annotations

import unittest
from typing import Optional

import numpy as np

RTOL = 1e-4
ATOL = 1e-5

# Tiny shapes safe for GT 730 + shared display
B = 2
T = 8
C = 32
H = 4
HD = C // H
V = 20
SEED = 0


def assert_finite(name: str, arr: np.ndarray) -> None:
    if not np.isfinite(arr).all():
        raise AssertionError(f"{name} has NaN/Inf (min={np.nanmin(arr)}, max={np.nanmax(arr)})")


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray, rtol: float = RTOL, atol: float = ATOL) -> None:
    assert_finite(f"{name}.actual", actual)
    assert_finite(f"{name}.expected", expected)
    if actual.shape != expected.shape:
        raise AssertionError(f"{name} shape {actual.shape} != {expected.shape}")
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        raise AssertionError(
            f"{name} mismatch: max_abs={diff.max():.6g} mean_abs={diff.mean():.6g} "
            f"rtol={rtol} atol={atol}"
        )


def gelu_np(x: np.ndarray) -> np.ndarray:
    k = 0.79788456
    c = 0.044715
    return 0.5 * x * (1.0 + np.tanh(k * (x + c * x ** 3)))


def gelu_grad_np(x: np.ndarray) -> np.ndarray:
    k = 0.79788456
    c = 0.044715
    inner = k * (x + c * x ** 3)
    tanh_val = np.tanh(inner)
    sech2 = 1.0 - tanh_val ** 2
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * k * (1.0 + 3.0 * c * x ** 2)


def layernorm_np(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    invstd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mean) * invstd
    out = xhat * gamma + beta
    return out.astype(np.float32), xhat.astype(np.float32), invstd.astype(np.float32).reshape(-1)


class CudaTestCase(unittest.TestCase):
    """Skip suite cleanly when PyCUDA / GT 730 path is unavailable."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            from model.cuda import ops as cuda_ops  # noqa: F401
            cls.cuda_ops = cuda_ops
        except Exception as exc:  # pragma: no cover
            raise unittest.SkipTest(f"CUDA unavailable: {exc}") from exc
