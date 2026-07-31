"""Stage 3.2: KV cache generate path determinism and isolation smoke."""
import unittest

import numpy as np

from model.config import GPTConfig
from model.gpt import GPTModel, _kv_state_nbytes
from model.weights import ModelParameters
from tokenizer.tokenizer import CharacterGPTTokenizer


def _model(**cfg_extra):
    corpus = ["once upon a time", "the cat sat", "a dog ran"]
    tok = CharacterGPTTokenizer.from_corpus(corpus)
    cfg = GPTConfig({
        "vocab_size": tok.vocab_size,
        "max_len": 32,
        "embedding_dim": 32,
        "num_heads": 4,
        "num_layers": 2,
        "dropout_prob": 0.0,
        "name": "kv_test",
        **cfg_extra,
    })
    return GPTModel(cfg, ModelParameters(cfg, seed=0)), tok


class TestKVCacheGenerate(unittest.TestCase):
    def test_kv_self_deterministic(self):
        model, tok = _model()
        prompt = tok.encode("once upon")
        a = model.generate(prompt, 16, temperature=0.7, top_k=5, rng=np.random.default_rng(1), use_kv_cache=True)
        b = model.generate(prompt, 16, temperature=0.7, top_k=5, rng=np.random.default_rng(1), use_kv_cache=True)
        self.assertEqual(a, b)

    def test_kv_matches_nokv_greedy(self):
        """Argmax decode: KV vs full recompute should agree on token ids."""
        model, tok = _model()
        prompt = tok.encode("once upon a")
        # temperature -> near-greedy via very low temp
        kw = dict(max_new_tokens=12, temperature=1e-6, top_k=None, top_p=None)
        a = model.generate(prompt, rng=np.random.default_rng(0), use_kv_cache=True, **kw)
        b = model.generate(prompt, rng=np.random.default_rng(0), use_kv_cache=False, **kw)
        self.assertEqual(a, b)

    def test_kv_grows(self):
        model, tok = _model()
        prompt = tok.encode("once")
        _, kv = model._prefill_kv(prompt)
        n0 = _kv_state_nbytes(kv)
        t0 = kv["T"]
        _, kv2 = model._decode_kv(prompt[-1], kv)
        self.assertEqual(kv2["T"], t0 + 1)
        self.assertGreater(_kv_state_nbytes(kv2), n0)

    def test_kv_rope_rmsnorm_matches_nokv_greedy(self):
        """Modern stack (RoPE + RMSNorm): KV decode must match full recompute."""
        model, tok = _model(norm_type="rmsnorm", pos_encoding="rope", tie_embeddings=True)
        prompt = tok.encode("once upon a")
        kw = dict(max_new_tokens=12, temperature=1e-6, top_k=None, top_p=None)
        a = model.generate(prompt, rng=np.random.default_rng(0), use_kv_cache=True, **kw)
        b = model.generate(prompt, rng=np.random.default_rng(0), use_kv_cache=False, **kw)
        self.assertEqual(a, b)

    def test_kv_rope_decode_shapes(self):
        """Prefill K/V stay [H, T, hd]; decode append must not introduce a batch dim."""
        model, tok = _model(norm_type="rmsnorm", pos_encoding="rope")
        prompt = tok.encode("once")
        _, kv = model._prefill_kv(prompt)
        H, hd = model.config.num_heads, model.config.head_dim
        self.assertEqual(kv["layers"][0]["k"].shape, (H, len(prompt), hd))
        _, kv2 = model._decode_kv(int(prompt[-1]), kv)
        self.assertEqual(kv2["layers"][0]["k"].shape, (H, len(prompt) + 1, hd))
        self.assertEqual(kv2["layers"][0]["v"].ndim, 3)


if __name__ == "__main__":
    unittest.main()
