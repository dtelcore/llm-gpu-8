"""
collab/bootstrap.py

Runtime patches for Google Colab / Linux T4 WITHOUT editing model/cuda/*.

Call `apply()` once before any `import model.cuda.ops` or `train`/`generate`
path that pulls in ops.

Patches:
  1. model.cuda.env — on non-Windows, clear MSVC NVCC_OPTIONS and no-op configure()
  2. model.cuda.kernels.CUDA_SOURCE — arch-gated __shfl_down_sync for sm_70+
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_applied = False

_OLD_WARP = """__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}"""

_NEW_WARP = """__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
#if __CUDA_ARCH__ >= 700
        val += __shfl_down_sync(0xffffffff, val, offset);
#else
        val += __shfl_down(val, offset);
#endif
    }
    return val;
}"""


def ensure_repo_root_on_path() -> Path:
    root = str(_REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return _REPO_ROOT


def apply() -> None:
    """Idempotent: patch env + kernels before ops JIT-compiles."""
    global _applied
    ensure_repo_root_on_path()
    if _applied:
        return

    # Refuse if ops already loaded — patches would be too late.
    if "model.cuda.ops" in sys.modules:
        raise RuntimeError(
            "model.cuda.ops is already imported; collab.bootstrap.apply() must run first. "
            "Use `python collab/smoke_cuda.py` or `python collab/run_train.py` "
            "(or `from collab.bootstrap import apply; apply()` at the top of the notebook)."
        )

    import model.cuda.env as cuda_env

    if sys.platform != "win32":
        cuda_env.NVCC_OPTIONS = ["-O2"]

        def _configure_linux() -> None:
            if getattr(cuda_env, "_configured", False):
                return
            cuda_env._configured = True

        cuda_env.configure = _configure_linux  # type: ignore[method-assign]

    import model.cuda.kernels as kernels

    if _OLD_WARP not in kernels.CUDA_SOURCE:
        # Already patched or kernel text changed — try a looser replace.
        if "__shfl_down_sync" not in kernels.CUDA_SOURCE:
            kernels.CUDA_SOURCE = kernels.CUDA_SOURCE.replace(
                "val += __shfl_down(val, offset);",
                "#if __CUDA_ARCH__ >= 700\n"
                "        val += __shfl_down_sync(0xffffffff, val, offset);\n"
                "#else\n"
                "        val += __shfl_down(val, offset);\n"
                "#endif",
            )
    else:
        kernels.CUDA_SOURCE = kernels.CUDA_SOURCE.replace(_OLD_WARP, _NEW_WARP)

    _applied = True


def repo_root() -> Path:
    return _REPO_ROOT
