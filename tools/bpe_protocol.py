"""
tools/bpe_protocol.py

Stage 3.3 char vs BPE measurement protocol (does not change BiggerTest default).
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
from model.gpt import GPTModel
from model.weights import ModelParameters
from setup.dataset_setup import DatasetLoader
from tokenizer.bpe import BPETokenizer
from tokenizer.tokenizer import CharacterGPTTokenizer
from training.dataset import WindowedDataset
from training.gpu_optimizer import AdamWGPU
from training.loss import softmax_cross_entropy_batch_gpu
from tools.tracing.runtime_metrics import runtime_metrics, memory_timeline


def _device_used_mb() -> float:
    try:
        import pycuda.driver as cuda
        free, total = cuda.mem_get_info()
        return (total - free) / (1024 * 1024)
    except Exception:
        return float("nan")


def _short_train(model, dataset, steps: int = 3, batch_size: int = 2):
    rng = np.random.default_rng(0)
    opt = AdamWGPU(model.params, learning_rate=1e-4, gradient_clip=1.0)
    it = dataset.iter_batches(shuffle=True, rng=rng)
    times = []
    last_loss = None
    for i in range(steps):
        batch = next(it)
        xs = np.stack([x for x, _ in batch])
        ys = np.stack([y for _, y in batch])
        t0 = time.perf_counter()
        logits, cache = model.forward_batch(xs)
        loss, dlogits = softmax_cross_entropy_batch_gpu(cache["logits_d"], ys)
        grads = model.backward_batch_gpu(cache, dlogits.reshape(-1, model.config.vocab_size))
        opt.clip_grads_(grads)
        opt.step(grads)
        times.append((time.perf_counter() - t0) * 1000)
        last_loss = float(loss)
    return {
        "step_ms_mean": float(np.mean(times)),
        "train_loss_last": last_loss,
        "tokens_per_sec": float(batch_size * model.config.max_len / (np.mean(times) / 1000.0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="tiny_english")
    ap.add_argument("--num-merges", type=int, default=150)
    ap.add_argument("--context", type=int, default=64)
    ap.add_argument("--embed", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--out", type=str, default="output/baselines/stage33_bpe_protocol.json")
    args = ap.parse_args()

    assert not runtime_metrics.enabled and not memory_timeline.enabled

    loader = DatasetLoader(data_dir="data")
    if args.dataset in ("minimal", "tiny_code", "tiny_english"):
        corpus = loader.load_builtin(args.dataset)
    else:
        corpus = loader.load_from_file(args.dataset if args.dataset.endswith(".txt") else f"data/{args.dataset}.txt")

    sample_text = " ".join(corpus)[:4000]
    char_tok = CharacterGPTTokenizer.from_corpus(corpus)
    bpe_tok = BPETokenizer.from_corpus(corpus, num_merges=args.num_merges)

    char_ids = char_tok.encode(sample_text)
    bpe_cov = bpe_tok.coverage_stats(sample_text, context_tokens=args.context)

    def build_and_train(tok, label):
        cfg = GPTConfig({
            "vocab_size": tok.vocab_size,
            "max_len": args.context,
            "embedding_dim": args.embed,
            "num_heads": args.heads,
            "num_layers": args.layers,
            "dropout_prob": 0.0,
            "name": f"bpe_proto_{label}",
        })
        params = ModelParameters(cfg, seed=42)
        model = GPTModel(cfg, params)
        # WindowedDataset expects CharacterGPTTokenizer API (encode/decode/vocab_size)
        ds = WindowedDataset(corpus, tok, cfg.max_len, batch_size=2)
        vram = _device_used_mb()
        metrics = _short_train(model, ds, steps=args.steps, batch_size=2)
        metrics["device_used_mb"] = _device_used_mb()
        metrics["device_used_mb_before"] = vram
        metrics["vocab_size"] = tok.vocab_size
        metrics["param_count"] = int(sum(w.size for w in params.weights.values()))
        # coherence probe: greedy generate a few tokens
        prompt = tok.encode("the")[:8] or [0]
        out = model.generate(prompt, max_new_tokens=20, temperature=1e-6, use_kv_cache=True,
                             rng=np.random.default_rng(0))
        metrics["probe_text"] = tok.decode(out)
        return metrics

    char_m = build_and_train(char_tok, "char")
    bpe_m = build_and_train(bpe_tok, "bpe")

    # Save BPE artifact for reuse
    bpe_path = Path("output/tokenizer/stage33_bpe_vocab.json")
    bpe_tok.save(bpe_path)

    result = {
        "version": "0.1.1-dev",
        "milestone": "stage_3_3",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "note": "Char remains BiggerTest default. BPE is experiment-only.",
        "dataset": args.dataset,
        "num_merges": args.num_merges,
        "sample_chars": len(sample_text),
        "char": {
            "vocab_size": char_tok.vocab_size,
            "tokens_for_sample": len(char_ids),
            "chars_per_token": 1.0,
            "chars_covered_in_window": float(args.context),
            **char_m,
        },
        "bpe": {
            **bpe_cov,
            "vocab_size": bpe_tok.vocab_size,
            "merges": len(bpe_tok.merges),
            "vocab_path": str(bpe_path),
            **bpe_m,
        },
        "conclusions": [
            "Do not replace char BiggerTest without a longer measured run",
            "BPE increases vocab (embed/LM head cost) while compressing span per token",
            "Use this table as the Stage 3.3 scientific control",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
