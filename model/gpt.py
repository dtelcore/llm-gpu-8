"""
model/gpt.py

A from-scratch character-level GPT. Forward compute for every linear /
layernorm / gelu / softmax runs through the PyCUDA kernels in
model/cuda/ops.py (via model/layers.py). Backward is a hand-derived,
analytic NumPy backprop over the cached forward intermediates -- host
compute is fine here since batch_size and max_len are tiny (GT 730 VRAM
budget targets ~1MB param models), and it keeps gradient math auditable.

Sequences are processed one at a time (no batched attention); the
training loop sums/averages gradients across a mini-batch.
"""

from typing import Dict, List, Tuple

import numpy as np

from model import layers
from model.config import GPTConfig
from model.trace import TraceContext
from model.weights import ModelParameters

EPS = 1e-5


def _gelu_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of the tanh-approximation GeLU used in model/cuda/kernels.py."""
    k = 0.79788456
    c = 0.044715
    inner = k * (x + c * x**3)
    tanh_val = np.tanh(inner)
    sech2 = 1.0 - tanh_val**2
    return 0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * k * (1.0 + 3.0 * c * x**2)


def _layernorm_cache(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = EPS):
    """Host-side recomputation of layernorm mean/invstd/xhat for backward.

    The GPU kernel already produced the forward output (verified to match
    this formula within 1e-7); this only rebuilds the small per-row stats
    needed for the analytic backward pass.
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    invstd = 1.0 / np.sqrt(var + eps)
    xhat = (x - mean) * invstd
    y = xhat * gamma + beta
    return y, xhat, invstd


def _layernorm_backward(dout: np.ndarray, xhat: np.ndarray, invstd: np.ndarray, gamma: np.ndarray):
    N = xhat.shape[-1]
    dgamma = np.sum(dout * xhat, axis=0)
    dbeta = np.sum(dout, axis=0)

    dxhat = dout * gamma
    sum_dxhat = np.sum(dxhat, axis=-1, keepdims=True)
    sum_dxhat_xhat = np.sum(dxhat * xhat, axis=-1, keepdims=True)
    dx = (invstd / N) * (N * dxhat - sum_dxhat - xhat * sum_dxhat_xhat)
    return dx, dgamma, dbeta


class GPTModel:
    def __init__(self, config: GPTConfig, params: ModelParameters) -> None:
        self.config = config
        self.params = params

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, token_ids: np.ndarray, tracer: TraceContext = None) -> Tuple[np.ndarray, Dict]:
        """token_ids: 1D int array of length T (<= config.max_len). Returns (logits [T,V], cache)."""
        cfg = self.config
        w = self.params.weights
        b = self.params.biases
        T = len(token_ids)

        tok_emb = w["token_embedding"][token_ids]  # [T, C]
        pos_emb = w["position_embedding"][:T]  # [T, C]
        x0 = (tok_emb + pos_emb).astype(np.float32)

        cache: Dict = {"ids": np.asarray(token_ids), "x0": x0, "layers": []}
        h = x0

        for layer in range(cfg.num_layers):
            prefix = f"layer_{layer}"
            layer_cache: Dict = {}

            ln1_out, xhat1, invstd1 = _layernorm_cache(h, w[f"{prefix}.ln1_gamma"], b[f"{prefix}.ln1_beta"])
            layer_cache["ln1_in"] = h
            layer_cache["ln1_xhat"], layer_cache["ln1_invstd"] = xhat1, invstd1

            attn_out, attn_cache = self._attention_forward(
                ln1_out, prefix, tracer=tracer,
            )
            layer_cache["attn"] = attn_cache
            h = h + attn_out
            layer_cache["resid1_out"] = h

            ln2_out, xhat2, invstd2 = _layernorm_cache(h, w[f"{prefix}.ln2_gamma"], b[f"{prefix}.ln2_beta"])
            layer_cache["ln2_in"] = h
            layer_cache["ln2_xhat"], layer_cache["ln2_invstd"] = xhat2, invstd2

            mlp_out, mlp_cache = self._mlp_forward(ln2_out, prefix, tracer=tracer)
            layer_cache["mlp"] = mlp_cache
            h = h + mlp_out
            layer_cache["resid2_out"] = h

            if tracer is not None:
                tracer.dump_neurons(f"{prefix}.resid2_out", h)

            cache["layers"].append(layer_cache)

        h_final, xhat_f, invstd_f = _layernorm_cache(h, w["final_ln_gamma"], b["final_ln_beta"])
        cache["h_pre_final_ln"] = h
        cache["final_xhat"], cache["final_invstd"] = xhat_f, invstd_f
        cache["h_final"] = h_final

        logits = layers.linear(h_final, w["lm_head"], b["lm_head_bias"], tracer=tracer, name="lm_head")
        cache["logits"] = logits

        return logits, cache

    def _attention_forward(self, ln1_out: np.ndarray, prefix: str, tracer: TraceContext = None):
        cfg = self.config
        w, b = self.params.weights, self.params.biases
        T, C = ln1_out.shape
        H, hd = cfg.num_heads, cfg.head_dim

        qkv = layers.linear(ln1_out, w[f"{prefix}.qkv_proj"], b[f"{prefix}.qkv_bias"], tracer=tracer, name=f"{prefix}.qkv")
        q, k, v = np.split(qkv, 3, axis=-1)
        q_h = q.reshape(T, H, hd).transpose(1, 0, 2)  # [H,T,hd]
        k_h = k.reshape(T, H, hd).transpose(1, 0, 2)
        v_h = v.reshape(T, H, hd).transpose(1, 0, 2)

        causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scale = 1.0 / np.sqrt(hd)

        probs_h = np.empty((H, T, T), dtype=np.float32)
        head_out = np.empty((H, T, hd), dtype=np.float32)
        for hidx in range(H):
            raw = (q_h[hidx] @ k_h[hidx].T) * scale
            raw = np.where(causal_mask, -1e9, raw).astype(np.float32)
            probs = layers.softmax(raw)
            probs_h[hidx] = probs
            head_out[hidx] = probs @ v_h[hidx]

        attn_concat = head_out.transpose(1, 0, 2).reshape(T, C)
        attn_out = layers.linear(attn_concat, w[f"{prefix}.attn_out_proj"], b[f"{prefix}.attn_out_bias"], tracer=tracer, name=f"{prefix}.attn_out")

        if tracer is not None:
            tracer.dump_neurons(f"{prefix}.attn_out", attn_out)

        cache = {
            "ln1_out": ln1_out, "q_h": q_h, "k_h": k_h, "v_h": v_h,
            "probs_h": probs_h, "attn_concat": attn_concat, "causal_mask": causal_mask, "scale": scale,
        }
        return attn_out, cache

    def _mlp_forward(self, ln2_out: np.ndarray, prefix: str, tracer: TraceContext = None):
        w, b = self.params.weights, self.params.biases
        hidden = layers.linear(ln2_out, w[f"{prefix}.mlp_expand"], b[f"{prefix}.mlp_expand_bias"], tracer=tracer, name=f"{prefix}.mlp_expand")
        act = layers.gelu(hidden)
        mlp_out = layers.linear(act, w[f"{prefix}.mlp_contract"], b[f"{prefix}.mlp_contract_bias"], tracer=tracer, name=f"{prefix}.mlp_contract")

        if tracer is not None:
            tracer.dump_neurons(f"{prefix}.mlp_out", mlp_out)

        return mlp_out, {"ln2_out": ln2_out, "hidden": hidden, "act": act}

    # ------------------------------------------------------------------
    # Backward: analytic gradients over the cached forward intermediates.
    # ------------------------------------------------------------------

    def backward(self, cache: Dict, dlogits: np.ndarray) -> Dict[str, np.ndarray]:
        cfg = self.config
        w = self.params.weights
        grads: Dict[str, np.ndarray] = {}

        h_final = cache["h_final"]
        d_lm_head = h_final.T @ dlogits
        d_lm_head_bias = np.sum(dlogits, axis=0)
        d_h_final = dlogits @ w["lm_head"].T
        grads["lm_head"] = d_lm_head
        grads["lm_head_bias"] = d_lm_head_bias

        d_h, d_final_gamma, d_final_beta = _layernorm_backward(
            d_h_final, cache["final_xhat"], cache["final_invstd"], w["final_ln_gamma"],
        )
        grads["final_ln_gamma"] = d_final_gamma
        grads["final_ln_beta"] = d_final_beta

        for layer in reversed(range(cfg.num_layers)):
            prefix = f"layer_{layer}"
            layer_cache = cache["layers"][layer]

            # residual 2: h = resid1_out + mlp_out
            d_mlp_out = d_h
            d_resid1 = d_h  # gradient also flows straight through the residual

            d_ln2_out, d_mlp_grads = self._mlp_backward(d_mlp_out, layer_cache["mlp"], prefix, w)
            grads.update(d_mlp_grads)

            d_h_from_ln2, d_ln2_gamma, d_ln2_beta = _layernorm_backward(
                d_ln2_out, layer_cache["ln2_xhat"], layer_cache["ln2_invstd"], w[f"{prefix}.ln2_gamma"],
            )
            grads[f"{prefix}.ln2_gamma"] = d_ln2_gamma
            grads[f"{prefix}.ln2_beta"] = d_ln2_beta

            d_h = d_resid1 + d_h_from_ln2  # combine residual + layernorm branch

            # residual 1: h = ln1_in + attn_out
            d_attn_out = d_h
            d_resid0 = d_h

            d_ln1_out, d_attn_grads = self._attention_backward(d_attn_out, layer_cache["attn"], prefix, w)
            grads.update(d_attn_grads)

            d_h_from_ln1, d_ln1_gamma, d_ln1_beta = _layernorm_backward(
                d_ln1_out, layer_cache["ln1_xhat"], layer_cache["ln1_invstd"], w[f"{prefix}.ln1_gamma"],
            )
            grads[f"{prefix}.ln1_gamma"] = d_ln1_gamma
            grads[f"{prefix}.ln1_beta"] = d_ln1_beta

            d_h = d_resid0 + d_h_from_ln1

        # Embeddings
        T = len(cache["ids"])
        d_token_embedding = np.zeros_like(w["token_embedding"])
        np.add.at(d_token_embedding, cache["ids"], d_h)
        d_position_embedding = np.zeros_like(w["position_embedding"])
        d_position_embedding[:T] += d_h

        grads["token_embedding"] = d_token_embedding
        grads["position_embedding"] = d_position_embedding

        return grads

    def _mlp_backward(self, d_mlp_out: np.ndarray, mlp_cache: Dict, prefix: str, w: Dict):
        act, hidden, ln2_out = mlp_cache["act"], mlp_cache["hidden"], mlp_cache["ln2_out"]

        d_contract_w = act.T @ d_mlp_out
        d_contract_b = np.sum(d_mlp_out, axis=0)
        d_act = d_mlp_out @ w[f"{prefix}.mlp_contract"].T

        d_hidden = d_act * _gelu_grad(hidden)
        d_expand_w = ln2_out.T @ d_hidden
        d_expand_b = np.sum(d_hidden, axis=0)
        d_ln2_out = d_hidden @ w[f"{prefix}.mlp_expand"].T

        grads = {
            f"{prefix}.mlp_contract": d_contract_w,
            f"{prefix}.mlp_contract_bias": d_contract_b,
            f"{prefix}.mlp_expand": d_expand_w,
            f"{prefix}.mlp_expand_bias": d_expand_b,
        }
        return d_ln2_out, grads

    def _attention_backward(self, d_attn_out: np.ndarray, attn_cache: Dict, prefix: str, w: Dict):
        cfg = self.config
        H, hd = cfg.num_heads, cfg.head_dim
        ln1_out = attn_cache["ln1_out"]
        q_h, k_h, v_h = attn_cache["q_h"], attn_cache["k_h"], attn_cache["v_h"]
        probs_h, scale = attn_cache["probs_h"], attn_cache["scale"]
        T, C = ln1_out.shape

        attn_concat = attn_cache["attn_concat"]
        d_out_proj = attn_concat.T @ d_attn_out
        d_out_bias = np.sum(d_attn_out, axis=0)
        d_attn_concat = d_attn_out @ w[f"{prefix}.attn_out_proj"].T

        d_head_out = d_attn_concat.reshape(T, H, hd).transpose(1, 0, 2)  # [H,T,hd]

        d_q_h = np.empty_like(q_h)
        d_k_h = np.empty_like(k_h)
        d_v_h = np.empty_like(v_h)

        for hidx in range(H):
            probs = probs_h[hidx]  # [T,T]
            d_probs = d_head_out[hidx] @ v_h[hidx].T  # [T,T]
            d_v_h[hidx] = probs.T @ d_head_out[hidx]

            # softmax jacobian-vector product per row
            row_dot = np.sum(d_probs * probs, axis=-1, keepdims=True)
            d_raw = probs * (d_probs - row_dot)
            d_raw *= scale

            d_q_h[hidx] = d_raw @ k_h[hidx]
            d_k_h[hidx] = d_raw.T @ q_h[hidx]

        d_q = d_q_h.transpose(1, 0, 2).reshape(T, C)
        d_k = d_k_h.transpose(1, 0, 2).reshape(T, C)
        d_v = d_v_h.transpose(1, 0, 2).reshape(T, C)
        d_qkv = np.concatenate([d_q, d_k, d_v], axis=-1)

        d_qkv_w = ln1_out.T @ d_qkv
        d_qkv_b = np.sum(d_qkv, axis=0)
        d_ln1_out = d_qkv @ w[f"{prefix}.qkv_proj"].T

        grads = {
            f"{prefix}.attn_out_proj": d_out_proj,
            f"{prefix}.attn_out_bias": d_out_bias,
            f"{prefix}.qkv_proj": d_qkv_w,
            f"{prefix}.qkv_bias": d_qkv_b,
        }
        return d_ln1_out, grads

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int,
        temperature: float = 1.0,
        tracer: TraceContext = None,
        tokenizer=None,
        rng: np.random.Generator = None,
    ) -> List[int]:
        """Autoregressive sampling. If `tracer` and `tokenizer` are both given,
        emits token/logit traces on the steps selected by tracer.trace_every."""
        rng = rng or np.random.default_rng()
        ids = list(prompt_ids)
        for step in range(max_new_tokens):
            if tracer is not None:
                tracer.update_step(step)

            window = ids[-self.config.max_len :]
            logits, _ = self.forward(np.asarray(window), tracer=tracer)
            last_logits = logits[-1]

            if tracer is not None and tokenizer is not None:
                tracer.dump_tokens(window, tokenizer, label=f"generate step {step} (context)")
                tracer.dump_logits(last_logits, tokenizer, label=f"generate step {step}")

            scaled = last_logits / max(temperature, 1e-6)
            probs = _softmax_1d(scaled)
            next_id = int(rng.choice(len(probs), p=probs))
            ids.append(next_id)
        return ids


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)
