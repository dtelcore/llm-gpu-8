"""
model/cuda/graph.py

Stage 3.11 / Stage 4: CUDA Graph capture with probe + fallback (Kepler / CUDA 10.1).

GPU-only callables and the Stage 4 KV kernel chain (append → decode attn → argmax)
can be captured on a dedicated stream. Full transformer decode still uses the
PyCUDA default stream for GEMM/norm, so generate stays eager-device for the body.
"""
from __future__ import annotations

import ctypes
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from logging_config import logger

# CUDA driver / capture constants (CUDA 10.1+)
CUDA_SUCCESS = 0
CUDA_STREAM_CAPTURE_MODE_GLOBAL = 0
CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900
CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901


@dataclass
class GraphStatus:
    supported: bool = False
    captured: bool = False
    mode: str = "eager"  # eager | graph | fallback
    reason: str = ""
    capture_ms: float = 0.0
    replay_ms: float = 0.0
    eager_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supported": self.supported,
            "captured": self.captured,
            "mode": self.mode,
            "reason": self.reason,
            "capture_ms": self.capture_ms,
            "replay_ms": self.replay_ms,
            "eager_ms": self.eager_ms,
            "details": self.details,
        }


class _Driver:
    """Thin ctypes wrapper around nvcuda.dll graph APIs."""

    def __init__(self):
        self.lib = None
        self.error = ""
        try:
            self.lib = ctypes.WinDLL("nvcuda.dll")
        except OSError as exc:
            self.error = f"nvcuda.dll load failed: {exc}"
            return
        # Resolve symbols; missing symbols → unsupported on this toolkit.
        needed = (
            "cuStreamBeginCapture",
            "cuStreamEndCapture",
            "cuGraphInstantiate",
            "cuGraphLaunch",
            "cuGraphDestroy",
            "cuGraphExecDestroy",
            "cuCtxSynchronize",
        )
        for name in needed:
            if not hasattr(self.lib, name):
                self.error = f"missing driver symbol {name}"
                self.lib = None
                return
        self.lib.cuStreamBeginCapture.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.cuStreamBeginCapture.restype = ctypes.c_int
        self.lib.cuStreamEndCapture.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.cuStreamEndCapture.restype = ctypes.c_int
        # CUDA 10.1: cuGraphInstantiate(phGraphExec, hGraph, phErrorNode, logBuffer, bufferSize)
        self.lib.cuGraphInstantiate.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_char_p,
            ctypes.c_size_t,
        ]
        self.lib.cuGraphInstantiate.restype = ctypes.c_int
        self.lib.cuGraphLaunch.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        self.lib.cuGraphLaunch.restype = ctypes.c_int
        self.lib.cuGraphDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cuGraphDestroy.restype = ctypes.c_int
        self.lib.cuGraphExecDestroy.argtypes = [ctypes.c_void_p]
        self.lib.cuGraphExecDestroy.restype = ctypes.c_int
        self.lib.cuCtxSynchronize.argtypes = []
        self.lib.cuCtxSynchronize.restype = ctypes.c_int

    @property
    def ok(self) -> bool:
        return self.lib is not None


_DRIVER: Optional[_Driver] = None


def _driver() -> _Driver:
    global _DRIVER
    if _DRIVER is None:
        _DRIVER = _Driver()
    return _DRIVER


def probe_cuda_graphs() -> GraphStatus:
    """Check whether CUDA Graph driver symbols are available."""
    d = _driver()
    st = GraphStatus(supported=d.ok, mode="eager")
    if not d.ok:
        st.reason = d.error or "CUDA Graph APIs unavailable"
    else:
        st.reason = "driver symbols present (CUDA 10+)"
        st.details["api"] = "cuStreamBeginCapture/cuGraphLaunch"
    return st


class CudaGraph:
    """Capture a GPU-only callable on a dedicated stream and replay it."""

    def __init__(self):
        self._graph = ctypes.c_void_p()
        self._exec = ctypes.c_void_p()
        self._stream = None
        self.status = GraphStatus()
        self._alive = False

    def capture(self, fn: Callable[[Any], None], stream=None) -> GraphStatus:
        """Capture ``fn(stream)`` which must only enqueue GPU work on ``stream``.

        Host sync / malloc / CPU work inside ``fn`` invalidates capture → fallback.
        """
        from model.cuda import ops as cuda_ops  # ensures context
        import pycuda.driver as cuda

        d = _driver()
        self.status = probe_cuda_graphs()
        if not d.ok:
            self.status.mode = "fallback"
            return self.status

        self._stream = stream or cuda.Stream()
        handle = int(self._stream.handle)

        # Warm-up outside capture (populate caches / JIT).
        try:
            fn(self._stream)
            cuda.Context.synchronize()
        except Exception as exc:
            self.status.mode = "fallback"
            self.status.reason = f"warmup failed: {exc}"
            return self.status

        t0 = time.perf_counter()
        rc = d.lib.cuStreamBeginCapture(handle, CUDA_STREAM_CAPTURE_MODE_GLOBAL)
        if rc != CUDA_SUCCESS:
            self.status.mode = "fallback"
            self.status.reason = f"cuStreamBeginCapture rc={rc}"
            return self.status

        try:
            fn(self._stream)
        except Exception as exc:
            # Best-effort end to leave stream usable.
            graph_ptr = ctypes.c_void_p()
            d.lib.cuStreamEndCapture(handle, ctypes.byref(graph_ptr))
            self.status.mode = "fallback"
            self.status.reason = f"capture body failed: {exc}"
            return self.status

        graph_ptr = ctypes.c_void_p()
        rc = d.lib.cuStreamEndCapture(handle, ctypes.byref(graph_ptr))
        self.status.capture_ms = (time.perf_counter() - t0) * 1000.0
        if rc != CUDA_SUCCESS or not graph_ptr.value:
            self.status.mode = "fallback"
            self.status.reason = f"cuStreamEndCapture rc={rc}"
            return self.status

        self._graph = graph_ptr
        exec_ptr = ctypes.c_void_p()
        err_node = ctypes.c_void_p()
        log_buf = ctypes.create_string_buffer(512)
        rc = d.lib.cuGraphInstantiate(
            ctypes.byref(exec_ptr),
            self._graph,
            ctypes.byref(err_node),
            log_buf,
            512,
        )
        if rc != CUDA_SUCCESS or not exec_ptr.value:
            d.lib.cuGraphDestroy(self._graph)
            self._graph = ctypes.c_void_p()
            self.status.mode = "fallback"
            self.status.reason = f"cuGraphInstantiate rc={rc} log={log_buf.value!r}"
            return self.status

        self._exec = exec_ptr
        self._alive = True
        self.status.supported = True
        self.status.captured = True
        self.status.mode = "graph"
        self.status.reason = "captured"
        return self.status

    def launch(self) -> None:
        if not self._alive:
            raise RuntimeError("CudaGraph has no instantiated exec (capture failed?)")
        d = _driver()
        handle = int(self._stream.handle)
        rc = d.lib.cuGraphLaunch(self._exec, handle)
        if rc != CUDA_SUCCESS:
            raise RuntimeError(f"cuGraphLaunch rc={rc}")
        self._stream.synchronize()

    def destroy(self) -> None:
        d = _driver()
        if not d.ok:
            return
        if self._exec and self._exec.value:
            d.lib.cuGraphExecDestroy(self._exec)
            self._exec = ctypes.c_void_p()
        if self._graph and self._graph.value:
            d.lib.cuGraphDestroy(self._graph)
            self._graph = ctypes.c_void_p()
        self._alive = False

    def __del__(self):
        try:
            self.destroy()
        except Exception:
            pass


def capture_gpu_callable(fn: Callable[[Any], None], repeats: int = 20) -> GraphStatus:
    """Capture ``fn(stream)``, time eager vs replay, return status dict fields."""
    import pycuda.driver as cuda

    st = probe_cuda_graphs()
    if not st.supported:
        st.mode = "fallback"
        return st

    # Eager timing
    stream = cuda.Stream()
    fn(stream)
    cuda.Context.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(stream)
        stream.synchronize()
    st.eager_ms = (time.perf_counter() - t0) * 1000.0 / repeats

    g = CudaGraph()
    cap = g.capture(fn, stream=stream)
    if not cap.captured:
        return cap

    t0 = time.perf_counter()
    for _ in range(repeats):
        g.launch()
    cap.replay_ms = (time.perf_counter() - t0) * 1000.0 / repeats
    cap.eager_ms = st.eager_ms
    g.destroy()
    return cap


def try_capture_decode(decode_fn: Callable[[], None]) -> GraphStatus:
    """Attempt to capture a generate decode step (status-only; see replayable)."""
    st, _ = try_capture_decode_replayable(decode_fn)
    return st


def try_capture_decode_replayable(
    decode_fn: Callable[[], None],
) -> tuple:
    """Capture Stage 4 decode kernels on a dedicated stream; return (status, graph|None).

    Full transformer decode still uses the PyCUDA default stream (GEMM/norm), so a
    whole-step capture of ``decode_fn`` is attempted only as a diagnostic. The
    graph we keep for baselines/replay is the sync-free KV kernel chain:
    ``kv_append_row`` → ``causal_mha_decode`` → ``argmax_1d``.
    """
    from model.cuda import ops as cuda_ops
    import numpy as np
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray

    st = probe_cuda_graphs()
    if not st.supported:
        st.mode = "fallback"
        return st, None

    # --- Diagnostic: full decode_fn on dedicated stream (usually fails; recorded) ---
    full_reason = ""
    try:
        decode_fn()
        cuda.Context.synchronize()
    except Exception as exc:
        full_reason = f"decode warmup failed: {exc}"

    # --- Capture Stage 4 kernel chain (stream-explicit, no D2H) ---
    BH, max_len, hd, V = 4, 64, 8, 128
    q = gpuarray.empty((BH, hd), dtype=np.float32)
    k_new = gpuarray.empty((BH, hd), dtype=np.float32)
    v_new = gpuarray.empty((BH, hd), dtype=np.float32)
    k_arena = gpuarray.empty((BH, max_len, hd), dtype=np.float32)
    v_arena = gpuarray.empty((BH, max_len, hd), dtype=np.float32)
    out = gpuarray.empty((BH, hd), dtype=np.float32)
    logits = gpuarray.empty((V,), dtype=np.float32)
    idx = gpuarray.empty((1,), dtype=np.int32)
    for buf in (q, k_new, v_new, k_arena, v_arena, out, logits):
        cuda.memset_d8(buf.gpudata, 0, buf.nbytes)
    cuda.memset_d8(idx.gpudata, 0, idx.nbytes)
    scale = np.float32(1.0 / np.sqrt(hd))
    t_pos = 3
    valid = t_pos + 1

    def _kernel_chain(stream):
        cuda_ops.kv_append_row(
            k_new, k_arena, batch_heads=BH, t=t_pos, max_len=max_len, head_dim=hd,
            stream=stream,
        )
        cuda_ops.kv_append_row(
            v_new, v_arena, batch_heads=BH, t=t_pos, max_len=max_len, head_dim=hd,
            stream=stream,
        )
        cuda_ops.causal_mha_decode(
            q, k_arena, v_arena,
            batch_heads=BH, max_len=max_len, valid_len=valid, head_dim=hd,
            scale=float(scale), out=out, stream=stream,
        )
        cuda_ops.argmax_1d(logits, out_idx=idx, stream=stream)

    # Warmup
    stream = cuda.Stream()
    try:
        _kernel_chain(stream)
        stream.synchronize()
    except Exception as exc:
        st.mode = "fallback"
        st.reason = f"kernel-chain warmup failed: {exc}"
        if full_reason:
            st.details["full_decode"] = full_reason
        return st, None

    g = CudaGraph()
    cap = g.capture(_kernel_chain, stream=stream)
    if not cap.captured:
        st.mode = "fallback"
        st.reason = cap.reason or "kernel-chain capture failed"
        st.details["full_decode"] = full_reason or "not captured (default-stream GEMM)"
        return st, None

    try:
        t1 = time.perf_counter()
        g.launch()
        cap.replay_ms = (time.perf_counter() - t1) * 1000.0
    except Exception as exc:
        g.destroy()
        st.mode = "fallback"
        st.reason = f"kernel-chain launch failed: {exc}"
        return st, None

    cap.details = {
        "api": "cuStreamBeginCapture/cuGraphLaunch",
        "scope": "kv_append+causal_mha_decode+argmax",
        "full_decode": full_reason or (
            "eager GPU decode (PyCUDA default-stream GEMM/norm not in graph)"
        ),
        "BH": BH, "max_len": max_len, "hd": hd,
    }
    logger.info(
        "CUDA Graph Stage 4 kernel-chain captured (%.3f ms capture, %.3f ms replay)",
        cap.capture_ms, cap.replay_ms,
    )
    return cap, g
