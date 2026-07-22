"""Parity: single-seq forward() cache retains B/T for GPU backward."""

from __future__ import annotations

import numpy as np

from model.config import GPTConfig
from model.gpt import GPTModel
from model.weights import ModelParameters
from tests.parity._common import C, H, T, V, CudaTestCase
from training.loss import softmax_cross_entropy


class TestForwardBackwardContract(CudaTestCase):
    def test_forward_cache_keeps_batch_dims(self) -> None:
        cfg = GPTConfig(
            {
                "name": "contract",
                "vocab_size": V,
                "max_len": T,
                "embedding_dim": C,
                "num_heads": H,
                "num_layers": 2,
                "dropout_prob": 0.0,
            }
        )
        params = ModelParameters(cfg, seed=1)
        model = GPTModel(cfg, params)
        ids = np.arange(T, dtype=np.int32) % V
        xs, ys = ids[:-1], ids[1:]
        logits, cache = model.forward(xs)
        self.assertEqual(cache.get("B"), 1)
        self.assertEqual(cache.get("T"), len(xs))
        self.assertTrue(cache.get("batched"))
        self.assertTrue(cache.get("gpu"))
        loss, dlogits = softmax_cross_entropy(logits, ys)
        grads = model.backward(cache, dlogits)
        self.assertIn("lm_head", grads)
        self.assertTrue(np.isfinite(loss))
