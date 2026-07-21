"""
training/probe.py

Post-checkpoint sanity checks and mid-training generation probes.

- run_probe: reload checkpoint from disk, forward pass + loss, verify finite.
- run_generate_probe: sample text at training milestones (25/50/75/100% of total steps).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from logging_config import logger
from model.trace import TraceContext
from paths import QUARTER_FRACTIONS, quarter_name_for_fraction
from training.checkpoint import load_checkpoint
from training.loss import softmax_cross_entropy

DEFAULT_GENERATE_PROBE_PROMPT = "once upon a"
DEFAULT_GENERATE_PROBE_MAX_NEW_TOKENS = 256
DEFAULT_GENERATE_PROBE_TEMPERATURE = 0.6
DEFAULT_GENERATE_PROBE_TOP_K = 10
DEFAULT_GENERATE_PROBE_TOP_P = 0.9
GENERATE_PROBE_FRACTIONS = QUARTER_FRACTIONS


def generate_probe_milestones(total_steps: int) -> List[int]:
    """Return sorted unique step numbers at 25%, 50%, 75%, and 100% of total_steps."""
    return [step for step, _ in milestone_steps_with_fractions(total_steps)]


def milestone_steps_with_fractions(total_steps: int) -> List[Tuple[int, float]]:
    """Return sorted (step, fraction) pairs for quarterly milestones."""
    if total_steps < 1:
        return []
    by_step: Dict[int, float] = {}
    for fraction in GENERATE_PROBE_FRACTIONS:
        if fraction == 1.0:
            step = total_steps
        else:
            step = max(1, int(total_steps * fraction))
        # Prefer the larger fraction if two map to the same step (tiny runs).
        by_step[step] = max(by_step.get(step, 0.0), fraction)
    return sorted(by_step.items(), key=lambda x: x[0])


def milestone_fraction_map(total_steps: int) -> Dict[int, float]:
    return {step: frac for step, frac in milestone_steps_with_fractions(total_steps)}


def quarter_dir_name_for_step(step: int, total_steps: int) -> Optional[str]:
    frac = milestone_fraction_map(total_steps).get(step)
    if frac is None:
        return None
    return quarter_name_for_fraction(frac)


def make_full_tracer() -> TraceContext:
    """Force all trace channels on for quarterly diagnostics."""
    tracer = TraceContext(
        verbose=True,
        trace_logits=True,
        trace_tokens=True,
        trace_neurons=True,
        trace_vectorization=True,
        trace_every=1,
    )
    tracer.active_step = True
    return tracer


def run_probe(checkpoint_dir: str, sample_text: str = "the quick brown fox") -> bool:
    """Returns True if the checkpoint reloads cleanly and produces finite loss/logits."""
    from model.gpt import GPTModel

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
    model,
    tokenizer,
    *,
    step: int,
    total_steps: int,
    prompt: str = DEFAULT_GENERATE_PROBE_PROMPT,
    max_new_tokens: int = DEFAULT_GENERATE_PROBE_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_GENERATE_PROBE_TEMPERATURE,
    top_k: Optional[int] = DEFAULT_GENERATE_PROBE_TOP_K,
    top_p: Optional[float] = DEFAULT_GENERATE_PROBE_TOP_P,
    seed: int = 42,
    checkpoint_dir: Optional[str] = None,
    tracer: Optional[TraceContext] = None,
) -> str:
    """Sample text from the in-memory model at a training milestone and log/print it."""
    fraction = step / max(total_steps, 1)
    header = (
        f"[GenerateProbe] step={step:,}/{total_steps:,} "
        f"({fraction * 100:.0f}% of total) | prompt={prompt!r} "
        f"| temp={temperature} | top_k={top_k} | top_p={top_p}"
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
        top_k=top_k,
        top_p=top_p,
        rng=rng,
        tracer=tracer,
        tokenizer=tokenizer if tracer is not None else None,
    )
    text = tokenizer.decode(generated_ids)

    print(text)
    print("=" * 70 + "\n")
    logger.info("[GenerateProbe] output=%r", text)
    return text
