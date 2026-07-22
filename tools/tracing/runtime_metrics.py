"""
tools/tracing/runtime_metrics.py

Opt-in SyncMeter and MemoryTimeline recorders. Completely disabled by default
so BiggerTest / bench_step baselines are unchanged when flags are off.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class SyncMeter:
    """Counts host↔device transfer stalls when metrics are enabled."""

    def __init__(self) -> None:
        self.enabled: bool = False
        self.sync_count: int = 0
        self.sync_ms: float = 0.0
        self.categories: Dict[str, int] = {}

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def reset_window(self) -> None:
        self.sync_count = 0
        self.sync_ms = 0.0
        self.categories = {}

    def record(self, category: str, elapsed_ms: float) -> None:
        if not self.enabled:
            return
        self.sync_count += 1
        self.sync_ms += float(elapsed_ms)
        self.categories[category] = self.categories.get(category, 0) + 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "sync_count": int(self.sync_count),
            "sync_ms": float(self.sync_ms),
            "categories": dict(self.categories),
        }

    @contextmanager
    def measure(self, category: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(category, (time.perf_counter() - t0) * 1000.0)


class MemoryTimeline:
    """ScratchPool alloc/reuse/clear event log (pool lifetime, not tensor lifetime)."""

    def __init__(self) -> None:
        self.enabled: bool = False
        self.step: int = 0
        self.events: List[Dict[str, Any]] = []
        self.pool_resident_bytes: int = 0
        self.peak_pool_bytes: int = 0
        self._log_path: Optional[Path] = None
        self._buffer_nbytes: Dict[Any, int] = {}

    def enable(self, log_path: Optional[str] = None) -> None:
        self.enabled = True
        self.events = []
        self.pool_resident_bytes = 0
        self.peak_pool_bytes = 0
        self._buffer_nbytes = {}
        self._log_path = Path(log_path) if log_path else None
        if self._log_path is not None:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            # Truncate for a fresh run
            self._log_path.write_text("", encoding="utf-8")

    def disable(self) -> None:
        self.enabled = False

    def set_step(self, step: int) -> None:
        self.step = int(step)

    def scratch_peak_mb(self) -> float:
        return self.peak_pool_bytes / (1024.0 ** 2)

    def _append(self, event: Dict[str, Any]) -> None:
        self.events.append(event)
        if self._log_path is not None:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")

    def record_alloc(
        self,
        name: Optional[str],
        shape: tuple,
        nbytes: int,
        key: Any,
        driver_free: Optional[int] = None,
        driver_total: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return
        prev = self._buffer_nbytes.get(key, 0)
        self.pool_resident_bytes += int(nbytes) - int(prev)
        self._buffer_nbytes[key] = int(nbytes)
        if self.pool_resident_bytes > self.peak_pool_bytes:
            self.peak_pool_bytes = self.pool_resident_bytes
        ev: Dict[str, Any] = {
            "event": "alloc",
            "step": self.step,
            "name": name,
            "shape": list(shape),
            "nbytes": int(nbytes),
            "pool_resident_bytes": int(self.pool_resident_bytes),
            "peak_pool_bytes": int(self.peak_pool_bytes),
        }
        if driver_free is not None:
            ev["driver_free_bytes"] = int(driver_free)
        if driver_total is not None:
            ev["driver_total_bytes"] = int(driver_total)
            ev["driver_used_bytes"] = int(driver_total) - int(driver_free or 0)
        self._append(ev)

    def record_reuse(self, name: Optional[str], shape: tuple, nbytes: int) -> None:
        if not self.enabled:
            return
        self._append(
            {
                "event": "reuse",
                "step": self.step,
                "name": name,
                "shape": list(shape),
                "nbytes": int(nbytes),
                "pool_resident_bytes": int(self.pool_resident_bytes),
                "peak_pool_bytes": int(self.peak_pool_bytes),
            }
        )

    def record_clear(self) -> None:
        if not self.enabled:
            return
        self.pool_resident_bytes = 0
        self._buffer_nbytes = {}
        self._append({"event": "clear", "step": self.step})

    def export_jsonl(self, path: Optional[str] = None) -> Path:
        out = Path(path) if path else self._log_path
        if out is None:
            raise ValueError("No log path set; pass path= or enable(log_path=...)")
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for ev in self.events:
                f.write(json.dumps(ev) + "\n")
        return out

    def summary(self) -> Dict[str, Any]:
        allocs = sum(1 for e in self.events if e.get("event") == "alloc")
        reuses = sum(1 for e in self.events if e.get("event") == "reuse")
        by_name: Dict[str, int] = {}
        for e in self.events:
            if e.get("event") == "alloc":
                n = e.get("name") or "<unnamed>"
                by_name[n] = max(by_name.get(n, 0), int(e.get("nbytes", 0)))
        return {
            "allocations": allocs,
            "reuses": reuses,
            "peak_pool_bytes": self.peak_pool_bytes,
            "peak_pool_mb": self.scratch_peak_mb(),
            "largest_buffers": sorted(by_name.items(), key=lambda kv: -kv[1]),
        }


class KernelTimeline:
    """Software kernel / op timeline (Stage 3.7). Opt-in; not CUDA Graph capture."""

    def __init__(self) -> None:
        self.enabled: bool = False
        self.events: List[Dict[str, Any]] = []
        self.step: int = 0

    def enable(self) -> None:
        self.enabled = True
        self.events = []

    def disable(self) -> None:
        self.enabled = False

    def set_step(self, step: int) -> None:
        self.step = int(step)

    def reset(self) -> None:
        self.events = []

    @contextmanager
    def measure(self, name: str, category: str = "op") -> Iterator[None]:
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.events.append(
                {
                    "name": name,
                    "category": category,
                    "step": self.step,
                    "ms": (time.perf_counter() - t0) * 1000.0,
                }
            )

    def summary(self) -> Dict[str, Any]:
        by_name: Dict[str, float] = {}
        for e in self.events:
            by_name[e["name"]] = by_name.get(e["name"], 0.0) + float(e["ms"])
        total = sum(by_name.values())
        return {
            "event_count": len(self.events),
            "total_ms": total,
            "by_name_ms": dict(sorted(by_name.items(), key=lambda kv: -kv[1])),
        }

    def export_json(self, path: str) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"step": self.step, "events": self.events, "summary": self.summary()}
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out


class RuntimeMetrics:
    """Facade: SyncMeter + enable helpers used by train/ops."""

    def __init__(self) -> None:
        self.sync = SyncMeter()
        self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True
        self.sync.enable()

    def disable(self) -> None:
        self._enabled = False
        self.sync.disable()

    def reset_window(self) -> None:
        self.sync.reset_window()

    def snapshot(self) -> Dict[str, Any]:
        return self.sync.snapshot()

    @contextmanager
    def measure(self, category: str) -> Iterator[None]:
        with self.sync.measure(category):
            yield


runtime_metrics = RuntimeMetrics()
memory_timeline = MemoryTimeline()
kernel_timeline = KernelTimeline()
