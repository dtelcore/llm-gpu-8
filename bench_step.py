"""
bench_step.py

GPU training-step microbench using the stable batched cache contract
(forward_batch / backward_batch_gpu / AdamWGPU).

Also smoke-tests forward() → backward() (B=1 cache must retain B/T).
"""
import time
from typing import List

import numpy as np

from model.config import GPTConfig
from model.cuda import ops as cuda_ops
from model.gpt import GPTModel
from model.weights import ModelParameters
from model.trace import TraceContext
from tokenizer.tokenizer import CharacterGPTTokenizer
from training.dataset import WindowedDataset
from training.loss import softmax_cross_entropy, softmax_cross_entropy_batch_gpu
from training.gpu_optimizer import AdamWGPU
from setup.dataset_setup import DatasetLoader
from tools.tracing.runtime_metrics import runtime_metrics, memory_timeline

# Modest Kepler-friendly default (display-shared VRAM)
MODEL = dict(
    vocab_size=110, max_len=64, embedding_dim=64,
    num_heads=4, num_layers=2, dropout_prob=0.0, name="bench",
)
BATCH_SIZE = 4
SEED = 42
N_WARMUP = 2
N_STEPS = 8


def load_corpus(name: str) -> List[str]:
    loader = DatasetLoader(data_dir="data")
    if name in ("minimal", "tiny_code", "tiny_english"):
        return loader.load_builtin(name)
    path = name if name.endswith(".txt") else f"data/{name}.txt"
    return loader.load_from_file(path)


def smoke_forward_backward(model: GPTModel, tokenizer: CharacterGPTTokenizer) -> None:
    """Ensure single-seq forward() cache keeps B/T and backward succeeds on GPU."""
    text = "once upon a time"
    ids = np.asarray(tokenizer.encode(text)[: model.config.max_len], dtype=np.int32)
    if ids.size < 2:
        ids = np.arange(8, dtype=np.int32) % max(1, tokenizer.vocab_size)
    xs = ids[:-1]
    ys = ids[1:]
    logits, cache = model.forward(xs)
    assert cache.get("B") == 1 and cache.get("T") == len(xs), (
        f"cache contract broken: B={cache.get('B')} T={cache.get('T')}"
    )
    assert cache.get("batched") is True
    loss, dlogits = softmax_cross_entropy(logits, ys)
    grads = model.backward(cache, dlogits)
    assert grads, "backward returned empty grads"
    print(f"  smoke forward→backward OK (B=1, T={cache['T']}, loss={loss:.4f})")


def bench_dataset(label: str, corpus: List[str]) -> dict:
    assert not runtime_metrics.enabled and not memory_timeline.enabled, (
        "bench_step requires metrics disabled (baseline regression)"
    )
    tokenizer = CharacterGPTTokenizer.from_corpus(corpus)
    cfg = GPTConfig({**MODEL, "vocab_size": tokenizer.vocab_size})
    params = ModelParameters(cfg, seed=SEED)
    model = GPTModel(cfg, params)
    tracer = TraceContext()
    dataset = WindowedDataset(corpus, tokenizer, cfg.max_len, BATCH_SIZE)
    rng = np.random.default_rng(SEED)
    optimizer = AdamWGPU(params, learning_rate=1e-6, gradient_clip=1.0)

    print(f"\n{'='*60}")
    print(f"Dataset: {label} | tokens={dataset.total_tokens:,} | windows={dataset.num_windows():,}")
    print(f"Model: embed={cfg.embedding_dim} layers={cfg.num_layers} heads={cfg.num_heads} T={cfg.max_len} B={BATCH_SIZE}")
    print(f"{'='*60}")

    smoke_forward_backward(model, tokenizer)

    batch_iter = dataset.iter_batches(shuffle=True, rng=rng)
    times = []

    for step in range(N_WARMUP + N_STEPS):
        batch = next(batch_iter)
        xs = np.stack([x for x, _ in batch])
        ys = np.stack([y for _, y in batch])
        t0 = time.perf_counter()
        logits, cache = model.forward_batch(xs, tracer=tracer)
        loss, dlogits_d = softmax_cross_entropy_batch_gpu(cache["logits_d"], ys)
        grads = model.backward_batch_gpu(cache, dlogits_d.reshape(-1, model.config.vocab_size))
        optimizer.clip_grads_(grads)
        optimizer.step(grads)
        dt_ms = (time.perf_counter() - t0) * 1000
        if step >= N_WARMUP:
            times.append(dt_ms)
            print(f"  step {step - N_WARMUP + 1}: {dt_ms:.1f} ms  loss={float(loss):.4f}")

    avg_ms = float(np.mean(times))
    tok_s = (BATCH_SIZE * cfg.max_len) / (avg_ms / 1000.0)
    free_b, total_b = cuda_ops.get_memory_info()
    used_mb = (total_b - free_b) / (1024 ** 2)
    param_mb = params.param_count() * 4 / (1024 ** 2)
    adam_mb = 2 * param_mb  # FP32 m + v
    print(f"  avg: {avg_ms:.1f} ms  tok/s={tok_s:.0f}  vram_used={used_mb:.0f}MB  "
          f"params≈{param_mb:.1f}MB adam≈{adam_mb:.1f}MB")
    return {
        "label": label,
        "avg_step_ms": avg_ms,
        "tokens_per_sec": tok_s,
        "device_used_mb": used_mb,
        "parameter_mb": param_mb,
        "adam_estimated_mb": adam_mb,
        "batch_size": BATCH_SIZE,
        "max_len": cfg.max_len,
        "embedding_dim": cfg.embedding_dim,
        "num_layers": cfg.num_layers,
        "num_heads": cfg.num_heads,
        "param_count": params.param_count(),
    }


if __name__ == "__main__":
    results = []
    for name in ("minimal", "tiny_english"):
        corpus = load_corpus(name)
        results.append(bench_dataset(name, corpus))
    if results:
        best = max(results, key=lambda r: r["tokens_per_sec"])
        print(f"\nBest: {best['label']}  {best['tokens_per_sec']:.0f} tok/s  {best['avg_step_ms']:.1f} ms/step")
