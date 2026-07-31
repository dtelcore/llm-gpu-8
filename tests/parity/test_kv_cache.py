"""Stage 3.2 / Stage 4: KV cache generate path — device arenas + determinism."""
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
        self.assertTrue(kv.get("device"))
        _, kv2 = model._decode_kv(prompt[-1], kv)
        self.assertEqual(kv2["T"], t0 + 1)
        # Static arenas: nbytes stays constant; T grows.
        self.assertEqual(_kv_state_nbytes(kv2), n0)
        self.assertEqual(kv2["layers"][0]["k_d"].shape[1], model.config.max_len)

    def test_kv_rope_rmsnorm_matches_nokv_greedy(self):
        """Modern stack (RoPE + RMSNorm): KV decode must match full recompute."""
        model, tok = _model(norm_type="rmsnorm", pos_encoding="rope", tie_embeddings=True)
        prompt = tok.encode("once upon a")
        kw = dict(max_new_tokens=12, temperature=1e-6, top_k=None, top_p=None)
        a = model.generate(prompt, rng=np.random.default_rng(0), use_kv_cache=True, **kw)
        b = model.generate(prompt, rng=np.random.default_rng(0), use_kv_cache=False, **kw)
        self.assertEqual(a, b)

    def test_kv_rope_decode_shapes(self):
        """Prefill packs into [H, max_len, hd]; decode appends without realloc."""
        model, tok = _model(norm_type="rmsnorm", pos_encoding="rope")
        prompt = tok.encode("once")
        _, kv = model._prefill_kv(prompt)
        H, hd = model.config.num_heads, model.config.head_dim
        self.assertTrue(kv.get("device"))
        self.assertEqual(kv["layers"][0]["k_d"].shape, (H, model.config.max_len, hd))
        self.assertEqual(kv["T"], len(prompt))
        _, kv2 = model._decode_kv(int(prompt[-1]), kv)
        self.assertEqual(kv2["T"], len(prompt) + 1)
        self.assertEqual(kv2["layers"][0]["k_d"].shape, (H, model.config.max_len, hd))
        self.assertEqual(kv2["layers"][0]["v_d"].ndim, 3)

    def test_device_argmax_sample(self):
        model, tok = _model()
        prompt = tok.encode("once")
        logits, _ = model._prefill_kv(prompt)
        self.assertIsNotNone(model._last_logits_d)
        idx = model._sample_next_id_device(model._last_logits_d, temperature=1e-6)
        self.assertIsInstance(idx, int)
        self.assertGreaterEqual(idx, 0)
        self.assertLess(idx, model.config.vocab_size)


if __name__ == "__main__":
    unittest.main()
