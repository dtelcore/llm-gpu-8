"""Parity: one full GPT train step — GPU grads vs NumPy reference."""

from __future__ import annotations

import numpy as np

import model.gpt as gpt_mod
from model.config import GPTConfig
from model.gpt import GPTModel
from model.weights import ModelParameters
from tests.parity._common import B, C, H, T, V, CudaTestCase, assert_close
from training.loss import softmax_cross_entropy_batch, softmax_cross_entropy_batch_gpu
from tokenizer.tokenizer import CharacterGPTTokenizer


class TestStepParity(CudaTestCase):
    def test_forward_backward_grads(self) -> None:
        cfg = GPTConfig(
            {
                "name": "parity",
                "vocab_size": V,
                "max_len": T,
                "embedding_dim": C,
                "num_heads": H,
                "num_layers": 2,
                "dropout_prob": 0.0,
            }
        )
        # Fixed tiny vocab of V printable chars
        vocab_chars = [chr(97 + (i % 26)) for i in range(V)]
        tok = CharacterGPTTokenizer.__new__(CharacterGPTTokenizer)
        tok.vocab = vocab_chars
        tok.vocab_size = V
        tok.char_to_id = {c: i for i, c in enumerate(vocab_chars)}
        tok.id_to_char = {i: c for i, c in enumerate(vocab_chars)}

        params = ModelParameters(cfg, seed=7)
        # Snapshot host weights for both paths
        weight_snap = {k: v.copy() for k, v in params.weights.items()}
        bias_snap = {k: v.copy() for k, v in params.biases.items()}

        rng = np.random.default_rng(8)
        xs = rng.integers(0, V, size=(B, T), dtype=np.int32)
        ys = rng.integers(0, V, size=(B, T), dtype=np.int32)

        # --- GPU path ---
        gpt_mod._GPU_TRAINING = True
        gpt_mod._USE_GPU_ATTENTION = True
        params_gpu = ModelParameters(cfg, seed=7)
        for k in weight_snap:
            params_gpu.weights[k][:] = weight_snap[k]
        for k in bias_snap:
            params_gpu.biases[k][:] = bias_snap[k]
        params_gpu.upload_to_device()
        model_gpu = GPTModel(cfg, params_gpu)
        logits_g, cache_g = model_gpu.forward_batch(xs)
        self.assertTrue(cache_g.get("gpu"))
        loss_g, dlogits_g = softmax_cross_entropy_batch_gpu(cache_g["logits_d"], ys)
        grads_g = model_gpu.backward_batch_gpu(cache_g, dlogits_g.reshape(-1, V))

        # --- NumPy path (host attention caches for analytic backward) ---
        gpt_mod._GPU_TRAINING = False
        gpt_mod._USE_GPU_ATTENTION = False
        params_np = ModelParameters(cfg, seed=7)
        for k in weight_snap:
            params_np.weights[k][:] = weight_snap[k]
        for k in bias_snap:
            params_np.biases[k][:] = bias_snap[k]
        params_np.upload_to_device()
        model_np = GPTModel(cfg, params_np)
        logits_n, cache_n = model_np.forward_batch(xs)
        self.assertFalse(cache_n.get("gpu"))
        loss_n, dlogits_n = softmax_cross_entropy_batch(logits_n, ys)
        grads_n = model_np.backward_batch(cache_n, dlogits_n)

        # Restore production flags
        gpt_mod._GPU_TRAINING = True
        gpt_mod._USE_GPU_ATTENTION = True

        self.assertTrue(np.isfinite(loss_g))
        self.assertTrue(np.isfinite(loss_n))
        self.assertLess(abs(float(loss_g) - float(loss_n)), 5e-3)

        # Compare a representative set of grads
        keys = [
            "lm_head",
            "lm_head_bias",
            "token_embedding",
            "layer_0.qkv_proj",
            "layer_0.mlp_expand",
            "layer_1.attn_out_proj",
        ]
        ops = self.cuda_ops
        for key in keys:
            self.assertIn(key, grads_g)
            self.assertIn(key, grads_n)
            g = ops.to_host(grads_g[key]) if hasattr(grads_g[key], "get") else grads_g[key]
            n = grads_n[key]
            assert_close(f"grad.{key}", g.astype(np.float32), n.astype(np.float32), rtol=5e-4, atol=5e-5)
