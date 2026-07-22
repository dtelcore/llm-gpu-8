"""
model/cuda/allocator.py

Stage 3.6: lifetime-tagged freelist for ephemeral GPU activations.
Not a ScratchPool redesign — complementary reuse for fresh empty() sites.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from pycuda import gpuarray


Key = Tuple[Tuple[int, ...], str, str]  # shape, dtype, lifetime


class LifetimeAllocator:
    """Freelist keyed by (shape, dtype, lifetime_tag)."""

    def __init__(self) -> None:
        self._free: Dict[Key, List[gpuarray.GPUArray]] = defaultdict(list)
        self._live: Dict[int, Tuple[Key, gpuarray.GPUArray]] = {}
        self.alloc_count = 0
        self.reuse_count = 0
        self.release_count = 0
        self.peak_live_bytes = 0
        self._live_bytes = 0

    def empty(self, shape, dtype=np.float32, lifetime: str = "ephemeral") -> gpuarray.GPUArray:
        shape_t = tuple(int(s) for s in shape)
        dt = np.dtype(dtype).str
        key: Key = (shape_t, dt, str(lifetime))
        bucket = self._free[key]
        if bucket:
            buf = bucket.pop()
            self.reuse_count += 1
        else:
            buf = gpuarray.empty(shape_t, dtype=np.dtype(dtype))
            self.alloc_count += 1
        self._live[id(buf)] = (key, buf)
        self._live_bytes += int(buf.nbytes)
        self.peak_live_bytes = max(self.peak_live_bytes, self._live_bytes)
        return buf

    def release(self, buf: gpuarray.GPUArray) -> None:
        entry = self._live.pop(id(buf), None)
        if entry is None:
            return
        key, _owned = entry
        self._live_bytes = max(0, self._live_bytes - int(buf.nbytes))
        self._free[key].append(buf)
        self.release_count += 1

    def recycle_lifetime(self, lifetime: str) -> int:
        """Return all live buffers with this lifetime tag to the freelist."""
        n = 0
        for bid, (key, buf) in list(self._live.items()):
            if key[2] == lifetime:
                self._live.pop(bid)
                self._live_bytes = max(0, self._live_bytes - int(buf.nbytes))
                self._free[key].append(buf)
                self.release_count += 1
                n += 1
        return n

    def clear(self) -> None:
        self._free.clear()
        self._live.clear()
        self._live_bytes = 0

    def stats(self) -> Dict[str, int]:
        free_bufs = sum(len(v) for v in self._free.values())
        return {
            "alloc_count": int(self.alloc_count),
            "reuse_count": int(self.reuse_count),
            "release_count": int(self.release_count),
            "peak_live_bytes": int(self.peak_live_bytes),
            "freelist_buffers": int(free_bufs),
            "live_buffers": len(self._live),
        }


lifetime_allocator = LifetimeAllocator()
