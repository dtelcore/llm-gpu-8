"""
tools/stage4_baseline.py

Freeze Stage 4 GPU-resident KV decode + CUDA Graph kernel-chain baseline.
"""
from __future__ import annotations

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
from version import __version__


def main() -> Path:
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
        "name": "stage4_baseline",
        "norm_type": "rmsnorm",
        "pos_encoding": "rope",
        "tie_embeddings": True,
    })
    model = GPTModel(cfg, ModelParameters(cfg, seed=42))
    prompt = tok.encode("once upon a time")

    # Determinism: greedy KV vs no-KV
    kw = dict(max_new_tokens=32, temperature=1e-6, top_k=None, top_p=None)
    a = model.generate(prompt, rng=np.random.default_rng(0), use_kv_cache=True, **kw)
    b = model.generate(prompt, rng=np.random.default_rng(0), use_kv_cache=False, **kw)

    # Device arena sizing
    _, kv = model._prefill_kv(prompt)
    kv_bytes = _kv_state_nbytes(kv)

    # Eager device decode timing
    t0 = time.perf_counter()
    model.generate(
        prompt, max_new_tokens=64, temperature=1e-6,
        use_kv_cache=True, use_cuda_graph=False,
        rng=np.random.default_rng(1),
    )
    eager_s = time.perf_counter() - t0

    # Graph probe (kernel-chain capture)
    t0 = time.perf_counter()
    model.generate(
        prompt, max_new_tokens=64, temperature=1e-6,
        use_kv_cache=True, use_cuda_graph=True,
        rng=np.random.default_rng(1),
    )
    graph_s = time.perf_counter() - t0
    status = getattr(model, "_cuda_graph_status", {}) or {}

    out = {
        "version": __version__,
        "milestone": "stage_4",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "model": {
            "embedding_dim": cfg.embedding_dim,
            "layers": cfg.num_layers,
            "heads": cfg.num_heads,
            "context": cfg.max_len,
            "vocab_size": cfg.vocab_size,
            "norm_type": cfg.norm_type,
            "pos_encoding": cfg.pos_encoding,
        },
        "device_kv": {
            "enabled": bool(kv.get("device")),
            "arena_shape": list(kv["layers"][0]["k_d"].shape),
            "kv_bytes": kv_bytes,
            "kv_mb": round(kv_bytes / (1024 * 1024), 4),
        },
        "determinism": {
            "kv_vs_nokv_greedy_32": a == b,
        },
        "eager_device_decode": {
            "wall_s": round(eager_s, 4),
            "tokens": 64,
            "tokens_per_sec": round(64 / eager_s, 2) if eager_s > 0 else None,
        },
        "decode_capture": {
            "supported": status.get("supported"),
            "captured": status.get("captured"),
            "mode": status.get("mode"),
            "reason": status.get("reason"),
            "capture_ms": status.get("capture_ms"),
            "replay_ms": status.get("replay_ms"),
            "details": status.get("details"),
            "generate_wall_s": round(graph_s, 4),
        },
        "note": (
            "Stage 4: static device KV arenas + causal_mha_decode + device argmax. "
            "CUDA Graph captures the KV kernel chain (append/decode/argmax); full "
            "transformer decode stays eager on the PyCUDA default stream."
        ),
        "compare_to": "output/baselines/stage32_kv_generate.json",
    }

    dest = ROOT / "output" / "baselines" / "stage4_gpu_kv_decode.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"Wrote {dest}")
    return dest


if __name__ == "__main__":
    main()
