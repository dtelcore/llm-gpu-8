"""Profile forward / backward / optimizer / sync_device split (batched)."""
import time

import numpy as np

from model.config import GPTConfig
from model.gpt import GPTModel
from model.weights import ModelParameters
from model.trace import TraceContext
from tokenizer.tokenizer import CharacterGPTTokenizer
from training.dataset import WindowedDataset
from training.loss import softmax_cross_entropy_batch
from training.optimizer import AdamW
from setup.dataset_setup import DatasetLoader

MODEL = dict(vocab_size=110, max_len=128, embedding_dim=128, num_heads=8, num_layers=6, dropout_prob=0.1, name="prof")
BATCH_SIZE = 8
SEED = 42


def main():
    corpus = DatasetLoader("data").load_builtin("minimal")
    tokenizer = CharacterGPTTokenizer.from_corpus(corpus)
    cfg = GPTConfig({**MODEL, "vocab_size": tokenizer.vocab_size})
    params = ModelParameters(cfg, seed=SEED)
    model = GPTModel(cfg, params)
    dataset = WindowedDataset(corpus, tokenizer, cfg.max_len, BATCH_SIZE)
    optimizer = AdamW(params.all_params(), learning_rate=1e-6)
    batch = next(dataset.iter_batches(shuffle=False))
    xs = np.stack([x for x, _ in batch])
    ys = np.stack([y for _, y in batch])

    t0 = time.perf_counter()
    logits, cache = model.forward_batch(xs, tracer=TraceContext())
    fwd = time.perf_counter() - t0

    t0 = time.perf_counter()
    loss, dlogits = softmax_cross_entropy_batch(logits, ys)
    grads = model.backward_batch(cache, dlogits)
    bwd = time.perf_counter() - t0

    t0 = time.perf_counter()
    optimizer.clip_grads_(grads)
    optimizer.step(grads)
    opt = time.perf_counter() - t0

    t0 = time.perf_counter()
    params.sync_device(names=grads.keys())
    sync = time.perf_counter() - t0

    total = (fwd + bwd + opt + sync) * 1000
    print(f"forward_batch: {fwd*1000:7.1f} ms  ({fwd/(fwd+bwd+opt+sync)*100:.0f}%)")
    print(f"backward_batch:{bwd*1000:7.1f} ms  ({bwd/(fwd+bwd+opt+sync)*100:.0f}%)")
    print(f"optimizer:     {opt*1000:7.1f} ms  ({opt/(fwd+bwd+opt+sync)*100:.0f}%)")
    print(f"sync_device:   {sync*1000:7.1f} ms  ({sync/(fwd+bwd+opt+sync)*100:.0f}%)")
    print(f"total:         {total:7.1f} ms")


if __name__ == "__main__":
    main()
