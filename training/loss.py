"""
training/loss.py

Cross-entropy loss + gradient for next-token prediction, plus a compact
top-k logit dump hook for --trace-logits.
"""

from typing import Tuple

import numpy as np

from model.trace import TraceContext


def softmax_cross_entropy_batch_gpu(logits_d, targets: np.ndarray) -> Tuple[float, "object"]:
    """GPU logits [B, T, V] or [B*T, V]. Returns (mean loss, dlogits on device)."""
    from model.cuda import ops as cuda_ops

    targets = np.asarray(targets, dtype=np.int32)
    if hasattr(logits_d, "shape") and len(logits_d.shape) == 3:
        B, T, V = logits_d.shape
        flat = logits_d.reshape(B * T, V)
        flat_targets = targets.reshape(B * T)
    else:
        flat = logits_d
        flat_targets = targets.reshape(-1)
    loss, dflat = cuda_ops.cross_entropy(flat, flat_targets)
    if hasattr(logits_d, "shape") and len(logits_d.shape) == 3:
        return loss, dflat.reshape(B, T, V)
    return loss, dflat


def softmax_cross_entropy_batch(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """logits: [B, T, V], targets: [B, T]. Returns (mean loss, dlogits [B, T, V])."""
    B, T, V = logits.shape
    flat_logits = logits.reshape(B * T, V)
    flat_targets = targets.reshape(B * T)
    loss, dflat = softmax_cross_entropy(flat_logits, flat_targets)
    return loss, dflat.reshape(B, T, V)


def softmax_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> Tuple[float, np.ndarray]:
    """logits: [T, V], targets: [T] int ids. Returns (mean loss, dlogits [T, V])."""
    T = logits.shape[0]
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / np.sum(exp, axis=-1, keepdims=True)

    correct_probs = probs[np.arange(T), targets]
    loss = -np.mean(np.log(np.clip(correct_probs, 1e-12, None)))

    dlogits = probs.copy()
    dlogits[np.arange(T), targets] -= 1.0
    dlogits /= T
    return float(loss), dlogits.astype(np.float32)


def trace_predictions(logits: np.ndarray, targets: np.ndarray, ids: np.ndarray, tokenizer, tracer: TraceContext, label: str = "") -> None:
    """Dump last-position top-k logits + the token that was actually predicted vs the target."""
    if not (tracer.trace_logits and tracer.active_step):
        return
    tracer.dump_logits(logits[-1], tokenizer, label=f"{label} (target={tokenizer.id_to_token(int(targets[-1]))!r})")
