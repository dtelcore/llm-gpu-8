"""
tools/tracing/activation_account.py

Stage 3.4: attribute device / host activation bytes after a forward step.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from model.config import GPTConfig
from model.cuda import ops as cuda_ops
from model.gpt import GPTModel
from model.weights import ModelParameters
from tools.tracing.runtime_metrics import runtime_metrics, memory_timeline


def _nbytes(obj: Any) -> int:
    if obj is None:
        return 0
    if hasattr(obj, "nbytes"):
        try:
            return int(obj.nbytes)
        except Exception:
            pass
    if hasattr(obj, "size") and hasattr(obj, "dtype"):
        try:
            return int(obj.size) * int(np.dtype(obj.dtype).itemsize)
        except Exception:
            return 0
    return 0


def _walk(obj: Any, bucket: str, acc: Dict[str, int], seen: set) -> None:
    oid = id(obj)
    if oid in seen:
        return
    if hasattr(obj, "gpudata") or hasattr(obj, "nbytes") or isinstance(obj, np.ndarray):
        seen.add(oid)
        acc[bucket] = acc.get(bucket, 0) + _nbytes(obj)
        return
    if isinstance(obj, dict):
        seen.add(oid)
        for k, v in obj.items():
            key = str(k).lower()
            if any(s in key for s in ("ln1", "ln2", "final_xhat", "final_inv", "xhat", "invstd")):
                sub = "ln_cache"
            elif any(s in key for s in ("q_", "k_", "v_", "probs", "attn", "scale")):
                sub = "attention_cache"
            elif any(s in key for s in ("mlp", "hidden", "act")):
                sub = "mlp_cache"
            elif any(s in key for s in ("logits", "h_final", "h_pre", "x0", "ids")):
                sub = "other_activations"
            else:
                sub = bucket
            _walk(v, sub, acc, seen)
        return
    if isinstance(obj, (list, tuple)):
        seen.add(oid)
        for v in obj:
            _walk(v, bucket, acc, seen)


def account_params(params: ModelParameters) -> Dict[str, int]:
    w_bytes = sum(_nbytes(v) for v in params.weights.values())
    b_bytes = sum(_nbytes(v) for v in params.biases.values())
    dw = sum(_nbytes(v) for v in getattr(params, "device_weights", {}).values())
    db = sum(_nbytes(v) for v in getattr(params, "device_biases", {}).values())
    return {
        "parameter_host_bytes": w_bytes + b_bytes,
        "parameter_device_bytes": dw + db,
    }


def account_forward_cache(cache: Dict) -> Dict[str, int]:
    acc: Dict[str, int] = {}
    _walk(cache, "other_activations", acc, set())
    return acc


def device_used_mb() -> float:
    import pycuda.driver as cuda
    free, total = cuda.mem_get_info()
    return (total - free) / (1024 * 1024)


def run_account(B: int, T: int, embed: int, layers: int, heads: int, vocab: int) -> Dict:
    assert not runtime_metrics.enabled
    cfg = GPTConfig({
        "vocab_size": vocab,
        "max_len": T,
        "embedding_dim": embed,
        "num_heads": heads,
        "num_layers": layers,
        "dropout_prob": 0.0,
        "name": "activation_account",
    })
    params = ModelParameters(cfg, seed=0)
    model = GPTModel(cfg, params)
    xs = np.random.randint(0, vocab, size=(B, T), dtype=np.int64)
    before = device_used_mb()
    _logits, cache = model.forward_batch(xs)
    after = device_used_mb()
    buckets = account_forward_cache(cache)
    p = account_params(params)
    scratch = int(cuda_ops.scratch_pool.resident_bytes()) if hasattr(cuda_ops, "scratch_pool") else 0

    # Adam m/v estimate (not allocated here): 2 * param device
    adam_est = 2 * p["parameter_device_bytes"]

    mb = {k: round(v / (1024 * 1024), 4) for k, v in buckets.items()}
    total_cache = sum(buckets.values())
    largest = max(buckets.items(), key=lambda kv: kv[1])[0] if buckets else "none"

    return {
        "version": "0.1.1-dev",
        "milestone": "stage_3_4",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "shapes": {"B": B, "T": T, "embed": embed, "layers": layers, "heads": heads, "vocab": vocab},
        "device_used_mb_before_forward": round(before, 3),
        "device_used_mb_after_forward": round(after, 3),
        "parameter_mb": round(p["parameter_device_bytes"] / (1024 * 1024), 4),
        "adam_m_v_estimated_mb": round(adam_est / (1024 * 1024), 4),
        "scratch_pool_mb": round(scratch / (1024 * 1024), 4),
        "activation_cache_mb": round(total_cache / (1024 * 1024), 4),
        "buckets_mb": mb,
        "buckets_bytes": buckets,
        "largest_activation_bucket": largest,
        "note": "Targets 3.5/3.6 should prioritize the largest_activation_bucket.",
    }


def main():
    ap = argparse.ArgumentParser(description="Stage 3.4 activation memory accounting")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--context", type=int, default=256)
    ap.add_argument("--embed", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=110)
    ap.add_argument("--out", type=str, default="output/baselines/stage34_activation_account.json")
    args = ap.parse_args()
    result = run_account(args.batch_size, args.context, args.embed, args.layers, args.heads, args.vocab)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
