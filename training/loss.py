"""
training/loss.py

Cross-entropy loss + gradient for next-token prediction, plus a compact
top-k logit dump hook for --trace-logits.
"""

from typing import Tuple

import numpy as np

from model.trace import TraceContext


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
