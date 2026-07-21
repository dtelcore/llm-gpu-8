#!/usr/bin/env python3
"""
collab/smoke_cuda.py

CUDA smoke for Colab T4. Run from repo root:

    python collab/smoke_cuda.py

Applies collab.bootstrap patches, then imports ops / runs vector-add.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Bootstrap before any model.cuda.ops import.
_COLLAB = Path(__file__).resolve().parent
if str(_COLLAB.parent) not in sys.path:
    sys.path.insert(0, str(_COLLAB.parent))

from collab.bootstrap import apply, repo_root  # noqa: E402

apply()


def main() -> int:
    print("=" * 60)
    print("COLLAB CUDA SMOKE")
    print("=" * 60)
    print(f"repo_root={repo_root()}")
    print(f"platform={sys.platform}")

    import model.cuda.env as cuda_env

    print(f"NVCC_OPTIONS={cuda_env.NVCC_OPTIONS}")

    import numpy as np
    import pycuda.driver as cuda
    from pycuda.compiler import SourceModule

    # Full project path (same JIT as training).
    from model.cuda import ops as cuda_ops  # noqa: F401

    device = cuda.Device(0)
    name = device.name()
    cc = device.compute_capability()
    free_b, total_b = cuda.mem_get_info()
    print(f"device={name}")
    print(f"compute_capability={cc[0]}.{cc[1]}")
    print(f"vram_free_mb={free_b / (1024 ** 2):.0f}  vram_total_mb={total_b / (1024 ** 2):.0f}")
    free2, total2 = cuda_ops.get_memory_info()
    print(f"ops.get_memory_info free_mb={free2 / (1024 ** 2):.0f} total_mb={total2 / (1024 ** 2):.0f}")
    print("[OK] model.cuda.ops imported (bootstrap patches applied)")

    mod = SourceModule(
        """
        __global__ void add_vec(float *a, float *b, float *c, int n) {
            int i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i < n) c[i] = a[i] + b[i];
        }
        """,
        options=list(cuda_env.NVCC_OPTIONS),
    )
    add_vec = mod.get_function("add_vec")
    a = np.array([1, 2, 3, 4], dtype=np.float32)
    b = np.array([10, 20, 30, 40], dtype=np.float32)
    c = np.zeros_like(a)
    add_vec(
        cuda.In(a), cuda.In(b), cuda.Out(c), np.int32(len(a)),
        block=(4, 1, 1), grid=(1, 1),
    )
    expected = [11.0, 22.0, 33.0, 44.0]
    assert list(c) == expected, f"vector-add mismatch: {c} != {expected}"
    print(f"[OK] vector-add {list(c)}")
    print("=" * 60)
    print("SMOKE PASSED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        raise SystemExit(1)
