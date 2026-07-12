"""
training/optimizer.py

AdamW optimizer operating on the NumPy parameter dict from
model.weights.ModelParameters. Gradients are host-side NumPy arrays
(produced by model.gpt.GPTModel.backward), so this stays plain NumPy --
no GPU state needed for the update step.
"""

from typing import Dict

import numpy as np


class AdamW:
    def __init__(
        self,
        params: Dict[str, np.ndarray],
        learning_rate: float,
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        warmup_steps: int = 0,
        gradient_clip: float = 1.0,
    ) -> None:
        self.params = params
        self.base_lr = learning_rate
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.warmup_steps = max(0, warmup_steps)
        self.gradient_clip = gradient_clip

        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def current_lr(self) -> float:
        if self.warmup_steps > 0 and self.t < self.warmup_steps:
            return self.base_lr * (self.t + 1) / self.warmup_steps
        return self.base_lr

    def clip_grads_(self, grads: Dict[str, np.ndarray]) -> float:
        """In-place global-norm clipping. Returns the pre-clip global norm."""
        total_sq = sum(float(np.sum(g.astype(np.float64) ** 2)) for g in grads.values())
        global_norm = float(np.sqrt(total_sq))
        if self.gradient_clip and global_norm > self.gradient_clip:
            scale = self.gradient_clip / (global_norm + 1e-6)
            for g in grads.values():
                g *= scale
        return global_norm

    def step(self, grads: Dict[str, np.ndarray]) -> None:
        self.t += 1
        lr = self.current_lr()

        for key, grad in grads.items():
            if key not in self.params:
                continue
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grad
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grad**2)

            m_hat = self.m[key] / (1 - self.beta1**self.t)
            v_hat = self.v[key] / (1 - self.beta2**self.t)

            update = lr * (m_hat / (np.sqrt(v_hat) + self.epsilon))
            self.params[key] -= update
            if self.weight_decay > 0.0:
                self.params[key] -= lr * self.weight_decay * self.params[key]
