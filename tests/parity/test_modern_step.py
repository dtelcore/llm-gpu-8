"""Parity: full train step with RMSNorm + RoPE (+ optional grad checkpoint)."""

from __future__ import annotations

import numpy as np

import model.gpt as gpt_mod
from model.config import GPTConfig
from model.gpt import GPTModel
from model.weights import ModelParameters
from tests.parity._common import B, C, H, T, V, CudaTestCase, assert_close
from training.loss import softmax_cross_entropy_batch, softmax_cross_entropy_batch_gpu
from tokenizer.tokenizer import CharacterGPTTokenizer


def _cfg(**overrides):
    base = {
        "name": "modern",
        "vocab_size": V,
        "max_len": T,
        "embedding_dim": C,
        "num_heads": H,
        "num_layers": 2,
        "dropout_prob": 0.0,
        "tie_embeddings": True,
        "norm_type": "rmsnorm",
        "pos_encoding": "rope",
        "gradient_checkpointing": False,
    }
    base.update(overrides)
    return GPTConfig(base)


class TestModernStepParity(CudaTestCase):
    def _run_gpu_step(self, cfg: GPTConfig) -> float:
        gpt_mod._GPU_TRAINING = True
        gpt_mod._USE_GPU_ATTENTION = True
        params = ModelParameters(cfg, seed=7)
        model = GPTModel(cfg, params)
        rng = np.random.default_rng(8)
        xs = rng.integers(0, V, size=(B, T), dtype=np.int32)
        ys = rng.integers(0, V, size=(B, T), dtype=np.int32)
        logits, cache = model.forward_batch(xs)
        loss, dlogits_d = softmax_cross_entropy_batch_gpu(cache["logits_d"], ys)
        grads = model.backward_batch_gpu(cache, dlogits_d.reshape(-1, V))
        self.assertIn("token_embedding", grads)
        self.assertNotIn("lm_head", grads)  # tied
        self.assertNotIn("position_embedding", grads)  # rope
        self.assertNotIn("layer_0.ln1_beta", grads)  # rmsnorm
        self.assertTrue(np.isfinite(loss))
        # Spot-check a few grads finite on device
        for key in ("token_embedding", "layer_0.qkv_proj", "final_ln_gamma"):
            g = grads[key]
            arr = self.cuda_ops.to_host(g) if hasattr(g, "gpudata") else g
            self.assertTrue(np.isfinite(arr).all(), msg=key)
        return float(loss)

    def test_rmsnorm_rope_step(self) -> None:
        loss = self._run_gpu_step(_cfg())
        self.assertGreater(loss, 0.0)

    def test_grad_checkpoint_step(self) -> None:
        loss = self._run_gpu_step(_cfg(gradient_checkpointing=True))
        self.assertGreater(loss, 0.0)

    def test_checkpoint_matches_full_cache_grads(self) -> None:
        """Same weights: grads with/without checkpointing should match."""
        gpt_mod._GPU_TRAINING = True
        gpt_mod._USE_GPU_ATTENTION = True
        cfg_full = _cfg(gradient_checkpointing=False)
        cfg_ckpt = _cfg(gradient_checkpointing=True)
        params_full = ModelParameters(cfg_full, seed=11)
        params_ckpt = ModelParameters(cfg_ckpt, seed=11)
        # Copy weights/biases
        for k, v in params_full.weights.items():
            if k in params_ckpt.weights:
                params_ckpt.weights[k] = v.copy() if not (k == "lm_head" and params_ckpt.tie_embeddings) else params_ckpt.weights[k]
        if params_ckpt.tie_embeddings:
            params_ckpt.weights["lm_head"] = params_ckpt.weights["token_embedding"].T
        for k, v in params_full.biases.items():
            if k in params_ckpt.biases:
                params_ckpt.biases[k] = v.copy()
        params_ckpt.sync_device()
        params_full.sync_device()

        rng = np.random.default_rng(9)
        xs = rng.integers(0, V, size=(B, T), dtype=np.int32)
        ys = rng.integers(0, V, size=(B, T), dtype=np.int32)

        m_full = GPTModel(cfg_full, params_full)
        m_ckpt = GPTModel(cfg_ckpt, params_ckpt)
        _, cache_f = m_full.forward_batch(xs)
        _, cache_c = m_ckpt.forward_batch(xs)
        _, d_f = softmax_cross_entropy_batch_gpu(cache_f["logits_d"], ys)
        _, d_c = softmax_cross_entropy_batch_gpu(cache_c["logits_d"], ys)
        g_f = m_full.backward_batch_gpu(cache_f, d_f.reshape(-1, V))
        g_c = m_ckpt.backward_batch_gpu(cache_c, d_c.reshape(-1, V))
        for key in g_f:
            if key not in g_c:
                continue
            a = self.cuda_ops.to_host(g_f[key]) if hasattr(g_f[key], "gpudata") else g_f[key]
            b = self.cuda_ops.to_host(g_c[key]) if hasattr(g_c[key], "gpudata") else g_c[key]
            assert_close(f"ckpt.{key}", b, a, rtol=1e-3, atol=1e-4)
