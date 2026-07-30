"""
training/gpu_optimizer.py

AdamW optimizer that updates GPU-resident weight mirrors directly.
Host NumPy copies are synced only when saving checkpoints.
"""

import math
from typing import Dict, Iterable, Optional

import numpy as np
import pycuda.gpuarray as gpuarray

from model.cuda import ops as cuda_ops
from model.weights import ModelParameters


class AdamWGPU:
    """AdamW on device weight mirrors (ModelParameters.device_weights/biases)."""

    def __init__(
        self,
        params: ModelParameters,
        learning_rate: float,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        warmup_steps: int = 0,
        gradient_clip: float = 1.0,
        total_steps: int = 0,
        min_lr_ratio: float = 0.1,
    ) -> None:
        self.params = params
        self.base_lr = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.warmup_steps = max(0, warmup_steps)
        self.gradient_clip = gradient_clip
        self.total_steps = max(0, int(total_steps))
        self.min_lr_ratio = float(min_lr_ratio)
        self.t = 0

        # Exclude tied lm_head view — it aliases token_embedding storage.
        all_keys = list(params.trainable_param_names())
        self.m: Dict[str, gpuarray.GPUArray] = {}
        self.v: Dict[str, gpuarray.GPUArray] = {}
        for key in all_keys:
            if key in params.device_weights:
                arr = params.device_weights[key]
            elif key in params.device_biases:
                arr = params.device_biases[key]
            else:
                continue
            z = np.zeros(arr.shape, dtype=np.float32)
            self.m[key] = cuda_ops.to_device(z)
            self.v[key] = cuda_ops.to_device(z)

        # Pointer-table indirection for the batched single-launch update (item 3
        # of the fusion plan). w/m/v gpuarrays are allocated once for the run
        # (see model/weights.py) so their pointers + offsets are built once here;
        # only grad pointers are gathered fresh each step() since grad buffers
        # are freshly allocated every backward pass.
        self._batch_keys = [k for k in all_keys if k in self.m]
        sizes = [int(self._get_weight(k).size) for k in self._batch_keys]
        offsets = np.zeros(len(sizes) + 1, dtype=np.int64)
        np.cumsum(sizes, out=offsets[1:])
        self._total_n = int(offsets[-1])
        self._offsets_d = cuda_ops.to_device_int64(offsets)
        self._w_ptrs_d = cuda_ops.to_device_ptrs(
            [self._get_weight(k).gpudata for k in self._batch_keys]
        )
        self._m_ptrs_d = cuda_ops.to_device_ptrs([self.m[k].gpudata for k in self._batch_keys])
        self._v_ptrs_d = cuda_ops.to_device_ptrs([self.v[k].gpudata for k in self._batch_keys])

    def current_lr(self) -> float:
        if self.warmup_steps > 0 and self.t < self.warmup_steps:
            return self.base_lr * (self.t + 1) / self.warmup_steps
        if self.total_steps <= self.warmup_steps:
            return self.base_lr
        min_lr = self.base_lr * self.min_lr_ratio
        denom = max(1, self.total_steps - self.warmup_steps)
        progress = min(1.0, max(0.0, (self.t - self.warmup_steps) / denom))
        return min_lr + 0.5 * (self.base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

    def clip_grads_(self, grads: Dict[str, gpuarray.GPUArray]) -> float:
        total_sq = cuda_ops.grad_global_norm_sq(grads)
        global_norm = float(np.sqrt(total_sq))
        if self.gradient_clip and global_norm > self.gradient_clip:
            scale = self.gradient_clip / (global_norm + 1e-6)
            for key in grads:
                cuda_ops.scal_mul(grads[key], scale)
        return global_norm

    def _get_weight(self, key: str) -> gpuarray.GPUArray:
        if key in self.params.device_weights:
            return self.params.device_weights[key]
        return self.params.device_biases[key]

    def step(self, grads: Dict[str, gpuarray.GPUArray]) -> None:
        self.t += 1
        lr = self.current_lr()
        b1, b2, eps = self.beta1, self.beta2, self.epsilon
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t

        # Any tensor without a fresh grad this step keeps updating against a
        # stale pointer otherwise (undefined old data) -- fall back to the
        # per-tensor kernel for a missing subset, batched path for the rest.
        missing = [k for k in self._batch_keys if k not in grads]
        if missing:
            present_keys = [k for k in self._batch_keys if k in grads]
            for key in present_keys:
                w = self._get_weight(key)
                cuda_ops.adamw_update(
                    w, grads[key], self.m[key], self.v[key],
                    lr, self.weight_decay, b1, b2, eps, bc1, bc2,
                )
            return

        g_ptrs_d = cuda_ops.to_device_ptrs([grads[k].gpudata for k in self._batch_keys])
        cuda_ops.adamw_update_batched(
            self._offsets_d, self._w_ptrs_d, g_ptrs_d, self._m_ptrs_d, self._v_ptrs_d,
            len(self._batch_keys), self._total_n,
            lr, self.weight_decay, b1, b2, eps, bc1, bc2,
        )

    def sync_host_weights(self, names: Optional[Iterable[str]] = None) -> None:
        """Pull GPU mirrors back to host NumPy dicts (checkpoint save only)."""
        if names is not None:
            keys = list(names)
        else:
            keys = list(self.params.trainable_param_names())
        for key in keys:
            if self.params.tie_embeddings and key == "lm_head":
                continue
            if key in self.params.device_weights:
                host = self.params.weights[key]
                if not host.flags.c_contiguous:
                    host = np.ascontiguousarray(host)
                    self.params.weights[key] = host
                cuda_ops.sync_to_host(self.params.device_weights[key], host)
            elif key in self.params.device_biases:
                cuda_ops.sync_to_host(self.params.device_biases[key], self.params.biases[key])
        if self.params.tie_embeddings:
            self.params.weights["lm_head"] = self.params.weights["token_embedding"].T
            self.params.device_weights["lm_head"] = self.params.device_weights["token_embedding"].T
