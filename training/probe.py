"""
training/probe.py

Post-checkpoint sanity checks and mid-training generation probes.

- run_probe: reload checkpoint from disk, forward pass + loss, verify finite.
- run_generate_probe: sample text at training milestones (25/50/75/100% of total steps).
"""

from typing import List, Optional, Set

import numpy as np

from logging_config import logger
from model.gpt import GPTModel
from training.checkpoint import load_checkpoint
from training.loss import softmax_cross_entropy

DEFAULT_GENERATE_PROBE_PROMPT = "once upon a"
DEFAULT_GENERATE_PROBE_MAX_NEW_TOKENS = 256
DEFAULT_GENERATE_PROBE_TEMPERATURE = 0.8
GENERATE_PROBE_FRACTIONS = (0.25, 0.50, 0.75, 1.0)


def generate_probe_milestones(total_steps: int) -> List[int]:
    """Return sorted unique step numbers at 25%, 50%, 75%, and 100% of total_steps."""
    if total_steps < 1:
        return []
    milestones: Set[int] = set()
    for fraction in GENERATE_PROBE_FRACTIONS:
        if fraction == 1.0:
            milestones.add(total_steps)
        else:
            milestones.add(max(1, int(total_steps * fraction)))
    return sorted(milestones)


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


def run_generate_probe(
    model: GPTModel,
    tokenizer,
    *,
    step: int,
    total_steps: int,
    prompt: str = DEFAULT_GENERATE_PROBE_PROMPT,
    max_new_tokens: int = DEFAULT_GENERATE_PROBE_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_GENERATE_PROBE_TEMPERATURE,
    seed: int = 42,
    checkpoint_dir: Optional[str] = None,
) -> str:
    """Sample text from the in-memory model at a training milestone and log/print it."""
    fraction = step / max(total_steps, 1)
    header = (
        f"[GenerateProbe] step={step:,}/{total_steps:,} "
        f"({fraction * 100:.0f}% of total) | prompt={prompt!r}"
    )
    if checkpoint_dir:
        header += f" | checkpoint={checkpoint_dir}"

    logger.info(header)
    print("\n" + "=" * 70)
    print(header)
    print("=" * 70)

    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        msg = f"[GenerateProbe] Prompt {prompt!r} encodes to zero tokens; skipping"
        logger.warning(msg)
        print(msg)
        return ""

    rng = np.random.default_rng(seed)
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        rng=rng,
    )
    text = tokenizer.decode(generated_ids)

    print(text)
    print("=" * 70 + "\n")
    logger.info("[GenerateProbe] output=%r", text)
    return text
