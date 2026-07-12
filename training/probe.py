"""
training/probe.py

Post-checkpoint sanity check: reload the just-saved checkpoint from disk,
run one forward pass + loss computation, and verify everything is finite.
Catches silent corruption in save/load or NaN/Inf blowups early.
"""

from typing import List

import numpy as np

from logging_config import logger
from model.gpt import GPTModel
from training.checkpoint import load_checkpoint
from training.loss import softmax_cross_entropy


def run_probe(checkpoint_dir: str, sample_text: str = "the quick brown fox") -> bool:
    """Returns True if the checkpoint reloads cleanly and produces finite loss/logits."""
    try:
        gpt_config, params, tokenizer, _, _ = load_checkpoint(checkpoint_dir)
    except Exception as exc:
        logger.error(f"[Probe] Failed to reload checkpoint {checkpoint_dir}: {exc}")
        return False

    model = GPTModel(gpt_config, params)

    ids: List[int] = tokenizer.encode(sample_text)[: gpt_config.max_len]
    if len(ids) < 2:
        logger.warning("[Probe] Sample text too short after encoding; skipping numeric check")
        return True

    ids_arr = np.asarray(ids[:-1])
    targets_arr = np.asarray(ids[1:])

    logits, _ = model.forward(ids_arr)
    loss, dlogits = softmax_cross_entropy(logits, targets_arr)

    finite = np.isfinite(logits).all() and np.isfinite(loss) and np.isfinite(dlogits).all()
    if not finite:
        logger.error(f"[Probe] Non-finite values detected (loss={loss})")
        return False

    logger.info(f"[Probe] OK -- reload + forward pass succeeded, loss={loss:.4f}")
    print(f"[Probe] Checkpoint '{checkpoint_dir}' OK -- forward loss={loss:.4f}, logits finite={finite}")
    return True
