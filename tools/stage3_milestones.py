"""
tools/stage3_milestones.py

Run Stage 3.4–3.7 and 3.11 measurement artifacts.
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
from model.cuda import ops as cuda_ops
from model.cuda.allocator import lifetime_allocator
from model.cuda.fp16_storage import (
    compress_cache_fp16,
    estimate_savings_bytes,
    expand_cache_fp32,
    set_fp16_activation_storage,
)
from model.gpt import GPTModel
from model.weights import ModelParameters
from tools.tracing.activation_account import run_account
from tools.tracing.runtime_metrics import kernel_timeline
from training.gpu_optimizer import AdamWGPU
from training.loss import softmax_cross_entropy_batch_gpu
from version import __version__


def _cfg(B=4, T=64, embed=64, layers=2, heads=4, vocab=110):
    return GPTConfig({
        "vocab_size": vocab, "max_len": T, "embedding_dim": embed,
        "num_heads": heads, "num_layers": layers, "dropout_prob": 0.0, "name": "stage3",
    })


def run_34(out: Path):
    result = run_account(B=4, T=256, embed=256, layers=4, heads=8, vocab=110)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def run_35(out: Path):
    cfg = _cfg(B=2, T=64, embed=64, layers=2, heads=4)
    params = ModelParameters(cfg, seed=1)
    model = GPTModel(cfg, params)
    xs = np.random.randint(0, cfg.vocab_size, size=(2, 64), dtype=np.int64)
    ys = np.random.randint(0, cfg.vocab_size, size=(2, 64), dtype=np.int64)

    set_fp16_activation_storage(False)
    logits0, cache0 = model.forward_batch(xs)
    est = estimate_savings_bytes(cache0)
    loss0, d0 = softmax_cross_entropy_batch_gpu(cache0["logits_d"], ys)
    g0 = model.backward_batch_gpu(cache0, d0.reshape(-1, cfg.vocab_size))

    set_fp16_activation_storage(True)
    logits1, cache1 = model.forward_batch(xs)
    saved = compress_cache_fp16(cache1)  # already compressed in forward; ok if empty
    # forward already compressed; measure host nbytes of fp16 fields
    expand_cache_fp32(cache1)
    loss1, d1 = softmax_cross_entropy_batch_gpu(cache1["logits_d"], ys)
    g1 = model.backward_batch_gpu(cache1, d1.reshape(-1, cfg.vocab_size))

    # Compare a couple grads
    def _gmax(g):
        arr = g.get() if hasattr(g, "get") else np.asarray(g)
        return float(np.max(np.abs(arr)))

    key = "lm_head"
    diff = float(np.max(np.abs(g0[key].get() - g1[key].get())))
    set_fp16_activation_storage(False)

    result = {
        "version": __version__,
        "milestone": "stage_3_5",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "estimated_activation_savings_bytes": int(est),
        "estimated_activation_savings_mb": round(est / (1024 * 1024), 4),
        "loss_fp32": float(loss0),
        "loss_fp16_storage_path": float(loss1),
        "lm_head_grad_maxdiff": diff,
        "parity_ok": diff < 1e-2,
        "note": "Activations stored FP16 via device cast kernels; compute remains FP32 for sm_35.",
        "cast_path": "device_float_to_half / half_to_float",
        "saved_keys_sample": list(saved.keys())[:8],
    }
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def run_36(out: Path):
    lifetime_allocator.clear()
    lifetime_allocator.alloc_count = 0
    lifetime_allocator.reuse_count = 0
    lifetime_allocator.release_count = 0
    lifetime_allocator.peak_live_bytes = 0

    cfg = _cfg()
    params = ModelParameters(cfg, seed=2)
    model = GPTModel(cfg, params)
    opt = AdamWGPU(params, learning_rate=1e-4, gradient_clip=1.0)
    xs = np.random.randint(0, cfg.vocab_size, size=(2, 64), dtype=np.int64)
    ys = np.random.randint(0, cfg.vocab_size, size=(2, 64), dtype=np.int64)

    for _ in range(3):
        logits, cache = model.forward_batch(xs)
        loss, d = softmax_cross_entropy_batch_gpu(cache["logits_d"], ys)
        grads = model.backward_batch_gpu(cache, d.reshape(-1, cfg.vocab_size))
        opt.clip_grads_(grads)
        opt.step(grads)

    # Explicit allocate/release microbench
    for _ in range(20):
        buf = lifetime_allocator.empty((1024, 64), lifetime="micro")
        lifetime_allocator.release(buf)

    stats = lifetime_allocator.stats()
    result = {
        "version": __version__,
        "milestone": "stage_3_6",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "allocator_stats": stats,
        "note": "qkv_split temps use LifetimeAllocator; ScratchPool redesign still demoted.",
        "reuse_observed": stats["reuse_count"] > 0,
    }
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def run_37(out: Path):
    cfg = _cfg(T=64, embed=64, layers=2, heads=4)
    params = ModelParameters(cfg, seed=3)
    model = GPTModel(cfg, params)
    xs = np.random.randint(0, cfg.vocab_size, size=(2, 64), dtype=np.int64)
    ys = np.random.randint(0, cfg.vocab_size, size=(2, 64), dtype=np.int64)

    kernel_timeline.enable()
    kernel_timeline.reset()
    kernel_timeline.set_step(0)
    logits, cache = model.forward_batch(xs)
    loss, d = softmax_cross_entropy_batch_gpu(cache["logits_d"], ys)
    model.backward_batch_gpu(cache, d.reshape(-1, cfg.vocab_size))
    summary = kernel_timeline.summary()
    sample_path = Path("output/baselines/stage37_timeline.json")
    kernel_timeline.export_json(str(sample_path))
    kernel_timeline.disable()

    result = {
        "version": __version__,
        "milestone": "stage_3_7",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "timeline_path": str(sample_path),
        "summary": summary,
        "note": "Software kernel timeline; CUDA Graph API is Stage 3.11 (graph.py).",
    }
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def run_311(out: Path):
    """Stage 3.11: CUDA Graph probe + GPU-only capture + decode fallback."""
    import numpy as np
    from pycuda import gpuarray
    from model.cuda import ops as cuda_ops
    from model.cuda.graph import capture_gpu_callable, probe_cuda_graphs, try_capture_decode
    from model.config import GPTConfig
    from model.gpt import GPTModel
    from model.weights import ModelParameters

    probe = probe_cuda_graphs()

    # GPU-only callable: device float↔half round-trip (no host sync inside).
    x = gpuarray.to_gpu(np.random.randn(4096).astype(np.float32))
    out_h = gpuarray.empty(x.shape, dtype=np.float16)
    out_f = gpuarray.empty(x.shape, dtype=np.float32)

    def _cast_pair(stream):
        n = np.int32(x.size)
        grid, block = cuda_ops._launch_1d(int(x.size))
        cuda_ops._float_to_half_kernel(x, out_h, n, block=block, grid=grid, stream=stream)
        cuda_ops._half_to_float_kernel(out_h, out_f, n, block=block, grid=grid, stream=stream)

    gpu_status = capture_gpu_callable(_cast_pair, repeats=30)

    # Tiny model decode capture (expected fallback — host KV).
    cfg = GPTConfig({
        "vocab_size": 64,
        "max_len": 32,
        "embedding_dim": 32,
        "num_heads": 4,
        "num_layers": 1,
        "dropout_prob": 0.0,
        "name": "graph_probe",
    })
    params = ModelParameters(cfg, seed=0)
    model = GPTModel(cfg, params)
    prompt = [1, 2, 3, 4]
    _logits, kv = model._prefill_kv(prompt)
    tok = 5
    kv_snap = {
        "B": kv["B"],
        "T": kv["T"],
        "layers": [{"k": ly["k"].copy(), "v": ly["v"].copy()} for ly in kv["layers"]],
    }

    def _decode():
        model._decode_kv(tok, kv_snap, tracer=None)

    decode_status = try_capture_decode(_decode)

    # Also exercise generate wiring.
    model.generate(prompt, max_new_tokens=2, use_kv_cache=True, use_cuda_graph=True)
    gen_status = getattr(model, "_cuda_graph_status", None)

    result = {
        "version": __version__,
        "milestone": "stage_3_11",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "probe": probe.to_dict(),
        "gpu_only_cast_graph": gpu_status.to_dict(),
        "decode_capture": decode_status.to_dict(),
        "generate_wiring": gen_status,
        "note": (
            "GPU-only callables may capture on CUDA 10+; host KV decode falls back "
            "to eager (to_host / NumPy attention). Use --cuda-graph on generate.py to probe."
        ),
    }
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", type=str, default="34,35,36,37,311")
    args = ap.parse_args()
    Path("output/baselines").mkdir(parents=True, exist_ok=True)
    for s in args.stages.split(","):
        s = s.strip()
        if s == "34":
            run_34(Path("output/baselines/stage34_activation_account.json"))
        elif s == "35":
            run_35(Path("output/baselines/stage35_fp16_storage.json"))
        elif s == "36":
            run_36(Path("output/baselines/stage36_allocator.json"))
        elif s == "37":
            run_37(Path("output/baselines/stage37_timeline_meta.json"))
        elif s in ("311", "3.11"):
            run_311(Path("output/baselines/stage311_cuda_graph.json"))


if __name__ == "__main__":
    main()
