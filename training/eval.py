"""
training/eval.py

Validation holdout helpers: 90/10 corpus split and CE val_loss / val_ppl eval.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

from logging_config import logger
from training.dataset import WindowedDataset
from training.loss import softmax_cross_entropy_batch, softmax_cross_entropy_batch_gpu

if TYPE_CHECKING:
    from model.gpt import GPTModel


VAL_FRACTION = 0.10
DEFAULT_VAL_MAX_BATCHES = 8


def split_train_val_corpus(
    corpus: List[str],
    *,
    seed: int = 42,
    val_fraction: float = VAL_FRACTION,
) -> Tuple[List[str], List[str]]:
    """Seeded sentence-level holdout. Returns (train_corpus, val_corpus)."""
    if not corpus:
        return [], []
    if len(corpus) == 1:
        return list(corpus), list(corpus)

    rng = np.random.default_rng(seed)
    indices = np.arange(len(corpus))
    rng.shuffle(indices)
    n_val = max(1, int(round(len(corpus) * val_fraction)))
    n_val = min(n_val, len(corpus) - 1)
    val_idx = set(int(i) for i in indices[:n_val])
    train = [corpus[i] for i in range(len(corpus)) if i not in val_idx]
    val = [corpus[i] for i in range(len(corpus)) if i in val_idx]
    logger.info(
        "Corpus split: train=%d sentences, val=%d sentences (%.0f%% holdout, seed=%d)",
        len(train), len(val), 100.0 * len(val) / len(corpus), seed,
    )
    return train, val


def ensure_train_val_split(config: Dict, *, seed: int = 42) -> Tuple[List[str], List[str]]:
    """Ensure config['dataset'] has corpus + val_corpus; split if needed. Mutates config."""
    dataset = config.setdefault("dataset", {})
    corpus = list(dataset.get("corpus") or [])
    val_corpus = dataset.get("val_corpus")

    if val_corpus is not None and len(val_corpus) > 0 and corpus:
        return list(corpus), list(val_corpus)

    if not corpus:
        return [], []

    # If corpus looks like a full unsplit corpus, split it.
    train, val = split_train_val_corpus(corpus, seed=seed)
    dataset["corpus"] = train
    dataset["val_corpus"] = val
    return train, val


def perplexity_from_loss(loss: float) -> float:
    return float(np.exp(min(float(loss), 50.0)))


def evaluate_val_loss(
    model: "GPTModel",
    val_dataset: Optional[WindowedDataset],
    *,
    max_batches: int = DEFAULT_VAL_MAX_BATCHES,
    seed: int = 42,
) -> Tuple[Optional[float], Optional[float]]:
    """Mean CE over up to max_batches from val_dataset. Returns (val_loss, val_ppl)."""
    if val_dataset is None or val_dataset.num_windows() < 1:
        return None, None

    rng = np.random.default_rng(seed)
    losses: List[float] = []
    for i, batch in enumerate(val_dataset.iter_batches(shuffle=True, rng=rng)):
        if i >= max_batches:
            break
        xs = np.stack([x for x, _ in batch])
        ys = np.stack([y for _, y in batch])
        logits, cache = model.forward_batch(xs)
        if cache.get("gpu"):
            loss, _ = softmax_cross_entropy_batch_gpu(cache["logits_d"], ys)
        else:
            loss, _ = softmax_cross_entropy_batch(logits, ys)
        losses.append(float(loss))

    if not losses:
        return None, None
    val_loss = float(np.mean(losses))
    return val_loss, perplexity_from_loss(val_loss)
