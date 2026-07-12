"""
model/trace.py

CLI-gated diagnostics: compact tables for token<->id mapping, top-k logits,
neuron/activation statistics, and GEMM shape/grid audits.

Nothing here is verbose by default -- every dump is gated by a boolean flag
on TraceContext, and the training loop only sets `active_step = True` on a
sparse subset of steps (see --trace-every), so normal runs stay quiet.
"""

from typing import Optional, Sequence, Tuple

import numpy as np


class TraceContext:
    """Holds CLI trace flags and exposes gated dump helpers."""

    def __init__(
        self,
        verbose: bool = False,
        trace_logits: bool = False,
        trace_tokens: bool = False,
        trace_neurons: bool = False,
        trace_vectorization: bool = False,
        trace_every: int = 100,
    ) -> None:
        self.verbose = verbose
        self.trace_logits = trace_logits or verbose
        self.trace_tokens = trace_tokens or verbose
        self.trace_neurons = trace_neurons
        self.trace_vectorization = trace_vectorization
        self.trace_every = max(1, trace_every)
        self.active_step = False

    @classmethod
    def from_args(cls, args) -> "TraceContext":
        return cls(
            verbose=args.verbose,
            trace_logits=args.trace_logits,
            trace_tokens=args.trace_tokens,
            trace_neurons=args.trace_neurons,
            trace_vectorization=args.trace_vectorization,
            trace_every=getattr(args, "trace_every", 100),
        )

    @property
    def any_enabled(self) -> bool:
        return any(
            [self.trace_logits, self.trace_tokens, self.trace_neurons, self.trace_vectorization]
        )

    def update_step(self, global_step: int) -> None:
        """Mark whether this step should emit deep traces."""
        self.active_step = self.any_enabled and (global_step % self.trace_every == 0)

    # ------------------------------------------------------------------
    # Dump helpers -- all gated by the caller checking the relevant flag
    # and `self.active_step` before invoking.
    # ------------------------------------------------------------------

    def dump_tokens(self, ids: Sequence[int], tokenizer, label: str = "") -> None:
        if not (self.trace_tokens and self.active_step):
            return
        print(f"\n[TraceTokens] {label}")
        print(f"{'Idx':<5} | {'ID':<5} | Token")
        print("-" * 28)
        for i, tid in enumerate(ids):
            print(f"{i:<5} | {int(tid):<5} | {tokenizer.id_to_token(int(tid))!r}")

    def dump_logits(self, logits_row: np.ndarray, tokenizer, top_k: int = 10, label: str = "") -> None:
        if not (self.trace_logits and self.active_step):
            return
        shifted = logits_row - np.max(logits_row)
        probs = np.exp(shifted) / np.sum(np.exp(shifted))
        top_indices = np.argsort(probs)[-top_k:][::-1]

        print(f"\n[TraceLogits] {label}")
        print(f"{'Rank':<5} | {'ID':<5} | {'Token':<8} | {'Prob':<8} | Logit")
        print("-" * 50)
        for rank, idx in enumerate(top_indices):
            token = tokenizer.id_to_token(int(idx))
            print(f"{rank + 1:<5} | {idx:<5} | {token!r:<8} | {probs[idx] * 100:6.2f}% | {logits_row[idx]:+.4f}")

    def dump_neurons(self, name: str, array: np.ndarray) -> None:
        if not (self.trace_neurons and self.active_step):
            return
        flat = array.reshape(-1)
        print(
            f"[TraceNeurons] {name:<24} shape={tuple(array.shape)!s:<16} "
            f"mean={flat.mean():+.4f} std={flat.std():.4f} "
            f"min={flat.min():+.4f} max={flat.max():+.4f} l2={np.linalg.norm(flat):.4f}"
        )

    def log_vectorization(
        self,
        name: str,
        a_shape: Tuple[int, int],
        b_shape: Tuple[int, int],
        c_shape: Tuple[int, int],
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
    ) -> None:
        if not (self.trace_vectorization and self.active_step):
            return
        print(
            f"[Vectorization] {name}: A{a_shape} x B{b_shape} -> C{c_shape} "
            f"| grid={grid} block={block}"
        )
