"""One-off diagnostic: profile where a single training step spends time."""
import time
from typing import List

import numpy as np

from model.config import GPTConfig
from model.gpt import GPTModel
from model.weights import ModelParameters
from model.trace import TraceContext
from tokenizer.tokenizer import CharacterGPTTokenizer
from training.dataset import WindowedDataset
from training.loss import softmax_cross_entropy
from training.optimizer import AdamW
from setup.dataset_setup import DatasetLoader

# Match the user's active run
MODEL = dict(
    vocab_size=110, max_len=128, embedding_dim=128,
    num_heads=8, num_layers=6, dropout_prob=0.1, name="bench",
)
BATCH_SIZE = 8
SEED = 42


def load_corpus(name: str) -> List[str]:
    loader = DatasetLoader(data_dir="data")
    if name in ("minimal", "tiny_code", "tiny_english"):
        return loader.load_builtin(name)
    path = name if name.endswith(".txt") else f"data/{name}.txt"
    return loader.load_from_file(path)


def bench_dataset(label: str, corpus: List[str], n_steps: int = 5) -> None:
    tokenizer = CharacterGPTTokenizer.from_corpus(corpus)
    cfg = GPTConfig({**MODEL, "vocab_size": tokenizer.vocab_size})
    params = ModelParameters(cfg, seed=SEED)
    model = GPTModel(cfg, params)
    tracer = TraceContext()
    dataset = WindowedDataset(corpus, tokenizer, cfg.max_len, BATCH_SIZE)
    rng = np.random.default_rng(SEED)
    optimizer = AdamW(params.all_params(), learning_rate=1e-6)

    batch_iter = dataset.iter_batches(shuffle=True, rng=rng)
    times = []

    print(f"\n{'='*60}")
    print(f"Dataset: {label} | tokens={dataset.total_tokens:,} | windows={dataset.num_windows():,}")
    print(f"{'='*60}")

    for step in range(n_steps):
        batch = next(batch_iter)
        t0 = time.perf_counter()

        batch_grads = None
        batch_loss = 0.0
        for x, y in batch:
            logits, cache = model.forward(x, tracer=tracer)
            loss, dlogits = softmax_cross_entropy(logits, y)
            grads = model.backward(cache, dlogits)
            batch_loss += loss
            if batch_grads is None:
                batch_grads = {k: v.copy() for k, v in grads.items()}
            else:
                for k, v in grads.items():
                    batch_grads[k] += v

        for k in batch_grads:
            batch_grads[k] /= len(batch)
        optimizer.clip_grads_(batch_grads)
        optimizer.step(batch_grads)
        params.sync_device(names=batch_grads.keys())

        dt_ms = (time.perf_counter() - t0) * 1000
        times.append(dt_ms)
        print(f"  step {step+1}: {dt_ms:.1f} ms  loss={batch_loss/len(batch):.4f}")

    print(f"  avg: {np.mean(times):.1f} ms  (min={min(times):.1f}, max={max(times):.1f})")


def count_gpu_roundtrips() -> None:
    """Estimate host<->device sync points per forward pass (post V3 on-device chaining)."""
    layers_n, heads = MODEL["num_layers"], MODEL["num_heads"]
    # Per layer: ln1_in, resid1, ln1_out, qkv, softmax batch, attn_concat, ln2_out, hidden, act, resid2
    per_layer_d2h = 10
    final_d2h = 3  # h_pre_final, h_final, logits
    per_forward = layers_n * per_layer_d2h + final_d2h
    per_step = per_forward * BATCH_SIZE
    print(f"\nEstimated D2H sync points per step (V3 on-device forward):")
    print(f"  per sequence forward: ~{per_forward}  (batch={BATCH_SIZE} -> ~{per_step} total)")
    print(f"  + sync_device re-upload: all weight tensors (~4.7 MB)")


if __name__ == "__main__":
    count_gpu_roundtrips()
    for name in ("minimal", "tiny_code", "tiny_stories"):
        corpus = load_corpus(name if name != "tiny_stories" else "data/tiny_stories.txt")
        bench_dataset(name, corpus)
