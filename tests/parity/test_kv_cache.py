"""Stage 3.2: KV cache generate path determinism and isolation smoke."""
import unittest

import numpy as np

from model.config import GPTConfig
from model.gpt import GPTModel, _kv_state_nbytes
from model.weights import ModelParameters
from tokenizer.tokenizer import CharacterGPTTokenizer


def _model():
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


if __name__ == "__main__":
    unittest.main()
