"""
tools/bench_generate.py

Stage 3.2 generate latency bench: full-recompute vs KV cache.

Protocol: fixed prompt/seed, generate N new tokens, report wall time,
tokens/sec, and KV memory overhead.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from model.config import GPTConfig
from model.gpt import GPTModel, _kv_state_nbytes
from model.weights import ModelParameters
from tokenizer.tokenizer import CharacterGPTTokenizer
from training.checkpoint import load_checkpoint

DEFAULT_PROMPT = "once upon a time"


def _tiny_model(seed: int = 42):
    corpus = [
        "once upon a time there was a little girl",
        "the cat sat on the mat and smiled",
        "a dog ran through the park today",
    ]
    tok = CharacterGPTTokenizer.from_corpus(corpus)
    cfg = GPTConfig({
        "vocab_size": tok.vocab_size,
        "max_len": 64,
        "embedding_dim": 64,
        "num_heads": 4,
        "num_layers": 2,
        "dropout_prob": 0.0,
        "name": "bench_generate",
    })
    params = ModelParameters(cfg, seed=seed)
    return GPTModel(cfg, params), tok


def _load_model(checkpoint: str | None, seed: int):
    if checkpoint:
        cfg, params, tok, _, _ = load_checkpoint(checkpoint)
        return GPTModel(cfg, params), tok
    return _tiny_model(seed=seed)


def _run_once(model, prompt_ids, n_tokens, seed, use_kv: bool):
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    out = model.generate(
        list(prompt_ids),
        max_new_tokens=n_tokens,
        temperature=0.8,
        top_k=10,
        top_p=0.9,
        rng=rng,
        use_kv_cache=use_kv,
    )
    dt = time.perf_counter() - t0
    return out, dt


def _measure_kv_bytes(model, prompt_ids, n_tokens, seed):
    """Prefill + decode loop to report peak KV host bytes."""
    rng = np.random.default_rng(seed)
    ids = list(prompt_ids)
    logits, kv = model._prefill_kv(ids[-model.config.max_len :])
    peak = _kv_state_nbytes(kv)
    for _ in range(n_tokens):
        next_id = int(np.argmax(logits[-1]))  # deterministic probe for memory path
        # still advance with sampling for shape parity
        next_id = int(rng.integers(0, model.config.vocab_size))
        ids.append(next_id)
        if kv["T"] >= model.config.max_len:
            logits, kv = model._prefill_kv(ids[-model.config.max_len :])
        else:
            logits, kv = model._decode_kv(next_id, kv)
        peak = max(peak, _kv_state_nbytes(kv))
    return peak


def main():
    ap = argparse.ArgumentParser(description="Bench generate with/without KV cache")
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="output/baselines/stage32_kv_generate.json")
    args = ap.parse_args()

    model, tok = _load_model(args.checkpoint, args.seed)
    prompt_ids = tok.encode(args.prompt)
    if not prompt_ids:
        raise SystemExit(f"prompt {args.prompt!r} encodes empty")

    # Determinism / match check (short)
    a, _ = _run_once(model, prompt_ids, 32, args.seed, use_kv=True)
    b, _ = _run_once(model, prompt_ids, 32, args.seed, use_kv=True)
    c, _ = _run_once(model, prompt_ids, 32, args.seed, use_kv=False)
    self_match = a == b
    cross_match = a == c

    # Warmup
    _run_once(model, prompt_ids, min(8, args.max_new_tokens), args.seed, use_kv=True)

    out_kv, dt_kv = _run_once(model, prompt_ids, args.max_new_tokens, args.seed, use_kv=True)
    out_no, dt_no = _run_once(model, prompt_ids, args.max_new_tokens, args.seed, use_kv=False)
    kv_bytes = _measure_kv_bytes(model, prompt_ids, args.max_new_tokens, args.seed)

    n = args.max_new_tokens
    result = {
        "version": "0.1.1-dev",
        "milestone": "stage_3_2",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint": args.checkpoint or "tiny_random",
        "prompt": args.prompt,
        "max_new_tokens": n,
        "seed": args.seed,
        "model": {
            "embedding_dim": model.config.embedding_dim,
            "layers": model.config.num_layers,
            "heads": model.config.num_heads,
            "context": model.config.max_len,
            "vocab_size": model.config.vocab_size,
        },
        "before_no_kv": {
            "wall_s": round(dt_no, 4),
            "tokens_per_sec": round(n / dt_no, 2) if dt_no > 0 else None,
        },
        "after_kv": {
            "wall_s": round(dt_kv, 4),
            "tokens_per_sec": round(n / dt_kv, 2) if dt_kv > 0 else None,
            "kv_peak_bytes": int(kv_bytes),
            "kv_peak_mb": round(kv_bytes / (1024 * 1024), 4),
        },
        "speedup": round(dt_no / dt_kv, 3) if dt_kv > 0 else None,
        "determinism": {
            "kv_self_match_32": self_match,
            "kv_vs_nokv_match_32": cross_match,
        },
        "sample_tail_kv": tok.decode(out_kv[-40:]),
        "sample_tail_nokv": tok.decode(out_no[-40:]),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
