"""One-off benchmark: matmul_bias_gelu + linear (2a) vs. fused_mlp_row (2b).

Used to decide the default of `model.cuda.ops._USE_FUSED_MLP_ROW_KERNEL`.
Run at bigwide dims: C=512, hidden=2048 (4x), rows=B*T=4*256=1024.
"""
import time

import numpy as np

from model.cuda import ops as cuda_ops

C = 512
HD = 2048
ROWS = 4 * 256
N_WARMUP = 3
N_ITERS = 20


def bench(fn, *args) -> float:
    for _ in range(N_WARMUP):
        fn(*args)
    cuda_ops.cuda.Context.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_ITERS):
        fn(*args)
    cuda_ops.cuda.Context.synchronize()
    return (time.perf_counter() - t0) / N_ITERS * 1000


def main() -> None:
    rng = np.random.default_rng(0)
    x = cuda_ops.to_device(rng.standard_normal((ROWS, C)).astype(np.float32))
    w1 = cuda_ops.to_device(rng.standard_normal((C, HD)).astype(np.float32) * 0.02)
    b1 = cuda_ops.to_device(np.zeros((HD,), dtype=np.float32))
    w2 = cuda_ops.to_device(rng.standard_normal((HD, C)).astype(np.float32) * 0.02)
    b2 = cuda_ops.to_device(np.zeros((C,), dtype=np.float32))

    def two_kernel_path():
        hidden, act = cuda_ops.matmul_bias_gelu(x, w1, b1, name="bench_expand")
        return cuda_ops.matmul_bias(act, w2, b2, name="bench_contract")

    def row_fused_path():
        return cuda_ops.fused_mlp_row(x, w1, b1, w2, b2)

    out_a = two_kernel_path()
    out_b = row_fused_path()
    max_diff = float(np.max(np.abs(out_a.get() - out_b.get())))
    print(f"max abs diff (2a vs 2b): {max_diff:.6e}")

    t_a = bench(two_kernel_path)
    t_b = bench(row_fused_path)
    print(f"2a matmul_bias_gelu + matmul_bias : {t_a:.3f} ms/iter")
    print(f"2b fused_mlp_row                  : {t_b:.3f} ms/iter")
    print(f"2b is {'FASTER' if t_b < t_a else 'SLOWER'} than 2a by {abs(t_a - t_b):.3f} ms ({abs(t_a - t_b) / t_a * 100:.1f}%)")


if __name__ == "__main__":
    main()
