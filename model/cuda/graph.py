"""
model/cuda/graph.py

Stage 3.11: CUDA Graph capture with probe + fallback (Kepler / CUDA 10.1).

Training and KV decode do host syncs (to_host / sampling), so full-step capture
often falls back to eager. GPU-only callables can still be captured and replayed.
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
    """Attempt to capture a generate decode step.

    Host KV decode uses ``to_host`` / NumPy attention, which invalidates stream
    capture — expected fallback on the current Kepler generate path.
    """
    st = probe_cuda_graphs()
    if not st.supported:
        st.mode = "fallback"
        return st

    import pycuda.driver as cuda

    stream = cuda.Stream()
    # Warm-up
    try:
        decode_fn()
        cuda.Context.synchronize()
    except Exception as exc:
        st.mode = "fallback"
        st.reason = f"decode warmup failed: {exc}"
        return st

    d = _driver()
    handle = int(stream.handle)
    rc = d.lib.cuStreamBeginCapture(handle, CUDA_STREAM_CAPTURE_MODE_GLOBAL)
    if rc != CUDA_SUCCESS:
        st.mode = "fallback"
        st.reason = f"cuStreamBeginCapture rc={rc}"
        return st

    # Decode uses default stream + host sync — capture will invalidate.
    try:
        decode_fn()
    except Exception as exc:
        graph_ptr = ctypes.c_void_p()
        d.lib.cuStreamEndCapture(handle, ctypes.byref(graph_ptr))
        st.mode = "fallback"
        st.reason = (
            f"decode capture body failed (host sync/KV expected): {exc}"
        )
        logger.info("CUDA Graph decode fallback: %s", st.reason)
        return st

    graph_ptr = ctypes.c_void_p()
    rc = d.lib.cuStreamEndCapture(handle, ctypes.byref(graph_ptr))
    if rc != CUDA_SUCCESS:
        st.mode = "fallback"
        st.reason = (
            f"decode cuStreamEndCapture rc={rc} "
            "(host KV decode not capture-compatible)"
        )
        logger.info("CUDA Graph decode fallback: %s", st.reason)
        return st

    # Unexpected success path — instantiate for completeness.
    if graph_ptr.value:
        d.lib.cuGraphDestroy(graph_ptr)
    st.mode = "fallback"
    st.reason = "decode ended capture but path is not wired for replay (host KV)"
    return st
