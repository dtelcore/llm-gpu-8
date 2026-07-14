"""
model/gpt.py

A from-scratch character-level GPT. Forward compute runs through PyCUDA
kernels (model/cuda/ops.py via model/layers.py) with batched sequences
stacked as [B*T, C]. Batched CPU attention runs one qkv download per layer for
the whole mini-batch. A fused GPU attention kernel (causal_mha_fp32) is also
compiled but disabled by default on GT 730. Backward is analytic NumPy.
"""

from typing import Dict, List, Tuple

import numpy as np

from model import layers
from model.config import GPTConfig
from model.cuda import ops as cuda_ops
from model.trace import TraceContext
from model.weights import ModelParameters

EPS = 1e-5
# Fused GPU attention + full GPU training path (forward, backward, optimizer on device).
_USE_GPU_ATTENTION = True
_GPU_TRAINING = True


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


def _qkv_to_heads(qkv: np.ndarray, batch_size: int, seq_len: int, num_heads: int, head_dim: int):
    """qkv [B*T, 3C] -> q_h,k_h,v_h each [B, H, T, hd]."""
    c = num_heads * head_dim
    q, k, v = np.split(qkv, 3, axis=-1)
    q_h = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    k_h = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    v_h = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(0, 2, 1, 3)
    return q_h, k_h, v_h


def _batched_attention_host(
    qkv: np.ndarray, batch_size: int, seq_len: int, num_heads: int, head_dim: int, scale: float,
):
    """CPU causal attention for a stacked batch. qkv is [B*T, 3C]."""
    B, T, H, hd = batch_size, seq_len, num_heads, head_dim
    C = H * hd
    causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    q_h, k_h, v_h = _qkv_to_heads(qkv, B, T, H, hd)
    probs_h = np.empty((B, H, T, T), dtype=np.float32)
    attn_concat = np.empty((B * T, C), dtype=np.float32)

    for b in range(B):
        head_out = np.empty((H, T, hd), dtype=np.float32)
        for hidx in range(H):
            raw = (q_h[b, hidx] @ k_h[b, hidx].T) * scale
            raw = np.where(causal_mask, -1e9, raw).astype(np.float32)
            shifted = raw - np.max(raw, axis=-1, keepdims=True)
            exp = np.exp(shifted)
            probs = exp / np.sum(exp, axis=-1, keepdims=True)
            probs_h[b, hidx] = probs
            head_out[hidx] = probs @ v_h[b, hidx]
        attn_concat[b * T:(b + 1) * T] = head_out.transpose(1, 0, 2).reshape(T, C)

    return attn_concat, probs_h, q_h, k_h, v_h


class GPTModel:
    def __init__(self, config: GPTConfig, params: ModelParameters) -> None:
        self.config = config
        self.params = params

    # ------------------------------------------------------------------
    # Forward (batched)
    # ------------------------------------------------------------------

    def forward_batch(
        self, token_ids_batch: np.ndarray, tracer: TraceContext = None,
    ) -> Tuple[np.ndarray, Dict]:
        """token_ids_batch: [B, T] int. Returns (logits [B, T, V], cache)."""
        cfg = self.config
        w = self.params.weights
        b = self.params.biases
        dw = self.params.device_weights
        db = self.params.device_biases

        token_ids_batch = np.asarray(token_ids_batch, dtype=np.int64)
        if token_ids_batch.ndim == 1:
            token_ids_batch = token_ids_batch.reshape(1, -1)
        B, T = token_ids_batch.shape
        H, hd = cfg.num_heads, cfg.head_dim
        scale = 1.0 / np.sqrt(hd)

        cache: Dict = {
            "ids": token_ids_batch, "B": B, "T": T, "batched": True, "gpu": _GPU_TRAINING,
        }

        if _GPU_TRAINING:
            h_d = cuda_ops.embedding_lookup(
                token_ids_batch.astype(np.int32),
                dw["token_embedding"], dw["position_embedding"], T,
            )
            cache["layers"] = []
            pending_ln1 = None

            for layer in range(cfg.num_layers):
                prefix = f"layer_{layer}"
                layer_cache: Dict = {"B": B, "T": T, "gpu": True}

                if pending_ln1 is None:
                    ln1_out_d, ln1_xhat_d, ln1_invstd_d = cuda_ops.layernorm_with_cache(
                        h_d, dw[f"{prefix}.ln1_gamma"], db[f"{prefix}.ln1_beta"],
                    )
                else:
                    ln1_out_d, ln1_xhat_d, ln1_invstd_d = pending_ln1
                    pending_ln1 = None
                layer_cache["ln1_out_d"] = ln1_out_d
                layer_cache["ln1_xhat_d"] = ln1_xhat_d
                layer_cache["ln1_invstd_d"] = ln1_invstd_d

                attn_out_d, attn_cache = self._attention_forward_batch(
                    ln1_out_d, None, prefix, B, T, H, hd, scale, tracer=tracer,
                )
                layer_cache["attn"] = attn_cache

                h_d, ln2_out_d, ln2_xhat_d, ln2_invstd_d = cuda_ops.residual_layernorm_with_cache(
                    h_d, attn_out_d, dw[f"{prefix}.ln2_gamma"], db[f"{prefix}.ln2_beta"],
                )
                layer_cache["ln2_out_d"] = ln2_out_d
                layer_cache["ln2_xhat_d"] = ln2_xhat_d
                layer_cache["ln2_invstd_d"] = ln2_invstd_d

                mlp_out_d, mlp_cache = self._mlp_forward_batch(
                    ln2_out_d, None, prefix, tracer=tracer,
                )
                layer_cache["mlp"] = mlp_cache

                if layer + 1 < cfg.num_layers:
                    next_prefix = f"layer_{layer + 1}"
                    h_d, ln1_n, xhat_n, inv_n = cuda_ops.residual_layernorm_with_cache(
                        h_d, mlp_out_d,
                        dw[f"{next_prefix}.ln1_gamma"], db[f"{next_prefix}.ln1_beta"],
                    )
                    pending_ln1 = (ln1_n, xhat_n, inv_n)
                else:
                    h_d, h_final_d, final_xhat_d, final_invstd_d = cuda_ops.residual_layernorm_with_cache(
                        h_d, mlp_out_d, dw["final_ln_gamma"], db["final_ln_beta"],
                    )

                if tracer is not None and tracer.trace_neurons and tracer.active_step:
                    tracer.dump_neurons(f"{prefix}.resid2_out", cuda_ops.to_host(h_d))

                cache["layers"].append(layer_cache)

            cache["h_final_d"] = h_final_d
            cache["final_xhat_d"] = final_xhat_d
            cache["final_invstd_d"] = final_invstd_d

            logits_d = layers.linear(
                h_final_d, dw["lm_head"], db["lm_head_bias"], tracer=tracer, name="lm_head",
            )
            cache["logits_d"] = logits_d
            logits = cuda_ops.to_host(logits_d).reshape(B, T, cfg.vocab_size)
            cache["logits"] = logits
            return logits, cache

        tok_emb = w["token_embedding"][token_ids_batch]  # [B, T, C]
        pos_emb = w["position_embedding"][:T]  # [T, C]
        x0 = (tok_emb + pos_emb).astype(np.float32).reshape(B * T, cfg.embedding_dim)

        cache["x0"] = x0
        cache["layers"] = []
        h_d = cuda_ops.to_device(x0)

        for layer in range(cfg.num_layers):
            prefix = f"layer_{layer}"
            layer_cache: Dict = {"B": B, "T": T}

            ln1_in = cuda_ops.to_host(h_d)
            ln1_out, layer_cache["ln1_xhat"], layer_cache["ln1_invstd"] = _layernorm_cache(
                ln1_in, w[f"{prefix}.ln1_gamma"], b[f"{prefix}.ln1_beta"],
            )
            layer_cache["ln1_in"] = ln1_in

            ln1_out_d = layers.layernorm(h_d, dw[f"{prefix}.ln1_gamma"], db[f"{prefix}.ln1_beta"])
            attn_out_d, attn_cache = self._attention_forward_batch(
                ln1_out_d, ln1_out, prefix, B, T, H, hd, scale, tracer=tracer,
            )
            layer_cache["attn"] = attn_cache
            h_d = layers.add_residual(h_d, attn_out_d)

            ln2_in = cuda_ops.to_host(h_d)
            ln2_out, layer_cache["ln2_xhat"], layer_cache["ln2_invstd"] = _layernorm_cache(
                ln2_in, w[f"{prefix}.ln2_gamma"], b[f"{prefix}.ln2_beta"],
            )
            layer_cache["ln2_in"] = ln2_in

            ln2_out_d = layers.layernorm(h_d, dw[f"{prefix}.ln2_gamma"], db[f"{prefix}.ln2_beta"])
            mlp_out_d, mlp_cache = self._mlp_forward_batch(ln2_out_d, ln2_out, prefix, tracer=tracer)
            layer_cache["mlp"] = mlp_cache
            h_d = layers.add_residual(h_d, mlp_out_d)

            if tracer is not None and tracer.trace_neurons and tracer.active_step:
                tracer.dump_neurons(f"{prefix}.resid2_out", cuda_ops.to_host(h_d))

            cache["layers"].append(layer_cache)

        cache["h_pre_final_ln"] = cuda_ops.to_host(h_d)
        cache["h_final"], cache["final_xhat"], cache["final_invstd"] = _layernorm_cache(
            cache["h_pre_final_ln"], w["final_ln_gamma"], b["final_ln_beta"],
        )

        h_final_d = layers.layernorm(h_d, dw["final_ln_gamma"], db["final_ln_beta"])
        logits_d = layers.linear(h_final_d, dw["lm_head"], db["lm_head_bias"], tracer=tracer, name="lm_head")
        logits = cuda_ops.to_host(logits_d).reshape(B, T, cfg.vocab_size)
        cache["logits"] = logits

        return logits, cache

    def forward(self, token_ids: np.ndarray, tracer: TraceContext = None) -> Tuple[np.ndarray, Dict]:
        """Single-sequence forward (wraps forward_batch with B=1)."""
        logits, cache = self.forward_batch(np.asarray(token_ids).reshape(1, -1), tracer=tracer)
        return logits[0], _squeeze_batch_cache(cache)

    def _attention_forward_batch(
        self, ln1_out_d, ln1_out_host: np.ndarray, prefix: str,
        B: int, T: int, H: int, hd: int, scale: float, tracer: TraceContext = None,
    ):
        dw, db = self.params.device_weights, self.params.device_biases

        qkv_d = layers.linear(
            ln1_out_d, dw[f"{prefix}.qkv_proj"], db[f"{prefix}.qkv_bias"],
            tracer=tracer, name=f"{prefix}.qkv",
        )

        if _USE_GPU_ATTENTION:
            attn_concat_d, probs_d, q_h, k_h, v_h = cuda_ops.fused_causal_attention_from_qkv(
                qkv_d, B, T, H, hd, scale,
            )
            attn_out_d = layers.linear(
                attn_concat_d, dw[f"{prefix}.attn_out_proj"], db[f"{prefix}.attn_out_bias"],
                tracer=tracer, name=f"{prefix}.attn_out",
            )
            attn_cache = {
                "ln1_out_d": ln1_out_d, "q_d": q_h, "k_d": k_h, "v_d": v_h,
                "probs_d": probs_d, "attn_concat_d": attn_concat_d,
                "scale": scale, "B": B, "T": T, "gpu": True, "heads_layout": True,
            }
        else:
            qkv = cuda_ops.to_host(qkv_d)
            attn_concat, probs_h, q_h, k_h, v_h = _batched_attention_host(
                qkv, B, T, H, hd, scale,
            )
            attn_concat_d = cuda_ops.to_device(attn_concat)
            attn_out_d = layers.linear(
                attn_concat_d, dw[f"{prefix}.attn_out_proj"], db[f"{prefix}.attn_out_bias"],
                tracer=tracer, name=f"{prefix}.attn_out",
            )
            attn_cache = {
                "ln1_out": ln1_out_host, "q_h": q_h, "k_h": k_h, "v_h": v_h,
                "probs_h": probs_h, "attn_concat": attn_concat,
                "causal_mask": np.triu(np.ones((T, T), dtype=bool), k=1),
                "scale": scale, "B": B, "T": T,
            }

        if tracer is not None and tracer.trace_neurons and tracer.active_step:
            tracer.dump_neurons(f"{prefix}.attn_out", cuda_ops.to_host(attn_out_d))

        return attn_out_d, attn_cache

    def _mlp_forward_batch(self, ln2_out_d, ln2_out_host: np.ndarray, prefix: str, tracer: TraceContext = None):
        dw, db = self.params.device_weights, self.params.device_biases

        hidden_d = layers.linear(
            ln2_out_d, dw[f"{prefix}.mlp_expand"], db[f"{prefix}.mlp_expand_bias"],
            tracer=tracer, name=f"{prefix}.mlp_expand",
        )
        act_d = layers.gelu(hidden_d)
        mlp_out_d = layers.linear(
            act_d, dw[f"{prefix}.mlp_contract"], db[f"{prefix}.mlp_contract_bias"],
            tracer=tracer, name=f"{prefix}.mlp_contract",
        )

        if tracer is not None and tracer.trace_neurons and tracer.active_step:
            tracer.dump_neurons(f"{prefix}.mlp_out", cuda_ops.to_host(mlp_out_d))

        if _GPU_TRAINING and ln2_out_host is None:
            mlp_cache = {
                "ln2_out_d": ln2_out_d, "hidden_d": hidden_d, "act_d": act_d, "gpu": True,
            }
        else:
            mlp_cache = {
                "ln2_out": ln2_out_host,
                "hidden": cuda_ops.to_host(hidden_d),
                "act": cuda_ops.to_host(act_d),
            }
        return mlp_out_d, mlp_cache

    # ------------------------------------------------------------------
    # Backward (batched)
    # ------------------------------------------------------------------

    def backward(self, cache: Dict, dlogits) -> Dict:
        """Accepts dlogits [T,V] or [B,T,V] (NumPy) or device logits grad."""
        if cache.get("gpu") and _GPU_TRAINING:
            return self.backward_batch_gpu(cache, dlogits)
        if hasattr(dlogits, "get"):
            dlogits = dlogits.get()
        if dlogits.ndim == 2:
            dlogits = dlogits.reshape(1, *dlogits.shape)
        if not cache.get("batched"):
            return self._backward_unbatched(cache, dlogits[0])
        return self.backward_batch(cache, dlogits)

    def backward_batch_gpu(self, cache: Dict, dlogits_d) -> Dict:
        """Full backward on GPU. dlogits_d: [B*T, V] device array."""
        import pycuda.gpuarray as gpuarray

        cfg = self.config
        dw, db = self.params.device_weights, self.params.device_biases
        B, T = cache["B"], cache["T"]
        C, V = cfg.embedding_dim, cfg.vocab_size
        grads: Dict = {}

        if not hasattr(dlogits_d, "get"):
            dlogits_d = cuda_ops.to_device(np.asarray(dlogits_d, dtype=np.float32).reshape(B * T, V))

        d_h, d_lm_head, d_lm_bias = cuda_ops.linear_backward(
            dlogits_d, cache["h_final_d"], dw["lm_head"],
        )
        grads["lm_head"] = d_lm_head
        grads["lm_head_bias"] = d_lm_bias

        d_h, d_final_gamma, d_final_beta = cuda_ops.layernorm_backward(
            d_h, cache["final_xhat_d"], cache["final_invstd_d"], dw["final_ln_gamma"],
        )
        grads["final_ln_gamma"] = d_final_gamma
        grads["final_ln_beta"] = d_final_beta

        for layer in reversed(range(cfg.num_layers)):
            prefix = f"layer_{layer}"
            layer_cache = cache["layers"][layer]

            d_mlp_out = d_h
            d_resid1 = d_h

            d_ln2_out, mlp_grads = self._mlp_backward_gpu(d_mlp_out, layer_cache["mlp"], prefix, dw)
            grads.update(mlp_grads)

            d_h_from_ln2, d_ln2_gamma, d_ln2_beta = cuda_ops.layernorm_backward(
                d_ln2_out, layer_cache["ln2_xhat_d"], layer_cache["ln2_invstd_d"],
                dw[f"{prefix}.ln2_gamma"],
            )
            grads[f"{prefix}.ln2_gamma"] = d_ln2_gamma
            grads[f"{prefix}.ln2_beta"] = d_ln2_beta

            d_h = cuda_ops.add_into(d_resid1, d_h_from_ln2)

            d_attn_out = d_h
            d_resid0 = d_h

            d_ln1_out, attn_grads = self._attention_backward_batch_gpu(
                d_attn_out, layer_cache["attn"], prefix, dw,
            )
            grads.update(attn_grads)

            d_h_from_ln1, d_ln1_gamma, d_ln1_beta = cuda_ops.layernorm_backward(
                d_ln1_out, layer_cache["ln1_xhat_d"], layer_cache["ln1_invstd_d"],
                dw[f"{prefix}.ln1_gamma"],
            )
            grads[f"{prefix}.ln1_gamma"] = d_ln1_gamma
            grads[f"{prefix}.ln1_beta"] = d_ln1_beta

            d_h = cuda_ops.add_into(d_resid0, d_h_from_ln1)

        d_tok, d_pos = cuda_ops.embed_backward(
            cache["ids"].astype(np.int32), d_h, cfg.vocab_size, C,
        )
        grads["token_embedding"] = d_tok
        grads["position_embedding"] = d_pos
        return grads

    def _mlp_backward_gpu(self, d_mlp_out, mlp_cache: Dict, prefix: str, dw: Dict):
        act_d = mlp_cache["act_d"]
        hidden_d = mlp_cache["hidden_d"]
        ln2_out_d = mlp_cache["ln2_out_d"]

        d_act, d_contract, d_contract_b = cuda_ops.linear_backward(
            d_mlp_out, act_d, dw[f"{prefix}.mlp_contract"],
        )
        d_hidden = cuda_ops.gelu_backward(hidden_d, d_act)
        d_ln2_out, d_expand, d_expand_b = cuda_ops.linear_backward(
            d_hidden, ln2_out_d, dw[f"{prefix}.mlp_expand"],
        )
        grads = {
            f"{prefix}.mlp_contract": d_contract,
            f"{prefix}.mlp_contract_bias": d_contract_b,
            f"{prefix}.mlp_expand": d_expand,
            f"{prefix}.mlp_expand_bias": d_expand_b,
        }
        return d_ln2_out, grads

    def _attention_backward_batch_gpu(self, d_attn_out, attn_cache: Dict, prefix: str, dw: Dict):
        B, T = attn_cache["B"], attn_cache["T"]
        H, hd = self.config.num_heads, self.config.head_dim
        C = H * hd
        scale = attn_cache["scale"]
        ln1_out_d = attn_cache["ln1_out_d"]
        q_d, k_d, v_d = attn_cache["q_d"], attn_cache["k_d"], attn_cache["v_d"]
        probs_d = attn_cache["probs_d"]
        attn_concat_d = attn_cache["attn_concat_d"]

        d_attn_out_flat = d_attn_out.reshape(B * T, C)
        d_attn_concat, d_out_proj, d_out_bias = cuda_ops.linear_backward(
            d_attn_out_flat, attn_concat_d, dw[f"{prefix}.attn_out_proj"],
        )

        d_q, d_k, d_v = cuda_ops.attention_backward_heads(
            d_attn_concat, q_d, k_d, v_d, probs_d,
            B, T, H, hd, scale,
            heads_layout=bool(attn_cache.get("heads_layout")),
        )
        if attn_cache.get("heads_layout"):
            d_qkv = cuda_ops.pack_qkv_from_heads(d_q, d_k, d_v, B, T, H, hd)
        else:
            d_qkv = cuda_ops.pack_qkv(d_q, d_k, d_v)

        d_ln1_out, d_qkv_w, d_qkv_b = cuda_ops.linear_backward(
            d_qkv, ln1_out_d, dw[f"{prefix}.qkv_proj"],
        )

        grads = {
            f"{prefix}.attn_out_proj": d_out_proj,
            f"{prefix}.attn_out_bias": d_out_bias,
            f"{prefix}.qkv_proj": d_qkv_w,
            f"{prefix}.qkv_bias": d_qkv_b,
        }
        return d_ln1_out, grads

    def backward_batch(self, cache: Dict, dlogits: np.ndarray) -> Dict[str, np.ndarray]:
        """dlogits: [B, T, V]. Weight grads are summed over the batch."""
        cfg = self.config
        w = self.params.weights
        B, T = cache["B"], cache["T"]
        C = cfg.embedding_dim
        V = cfg.vocab_size
        grads: Dict[str, np.ndarray] = {}

        dlogits_flat = dlogits.reshape(B * T, V)
        h_final = cache["h_final"]

        d_lm_head = h_final.T @ dlogits_flat
        d_lm_head_bias = np.sum(dlogits_flat, axis=0)
        d_h_final = dlogits_flat @ w["lm_head"].T
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

            d_mlp_out = d_h
            d_resid1 = d_h

            d_ln2_out, d_mlp_grads = self._mlp_backward(d_mlp_out, layer_cache["mlp"], prefix, w)
            grads.update(d_mlp_grads)

            d_h_from_ln2, d_ln2_gamma, d_ln2_beta = _layernorm_backward(
                d_ln2_out, layer_cache["ln2_xhat"], layer_cache["ln2_invstd"], w[f"{prefix}.ln2_gamma"],
            )
            grads[f"{prefix}.ln2_gamma"] = d_ln2_gamma
            grads[f"{prefix}.ln2_beta"] = d_ln2_beta

            d_h = d_resid1 + d_h_from_ln2

            d_attn_out = d_h
            d_resid0 = d_h

            d_ln1_out, d_attn_grads = self._attention_backward_batch(
                d_attn_out, layer_cache["attn"], prefix, w,
            )
            grads.update(d_attn_grads)

            d_h_from_ln1, d_ln1_gamma, d_ln1_beta = _layernorm_backward(
                d_ln1_out, layer_cache["ln1_xhat"], layer_cache["ln1_invstd"], w[f"{prefix}.ln1_gamma"],
            )
            grads[f"{prefix}.ln1_gamma"] = d_ln1_gamma
            grads[f"{prefix}.ln1_beta"] = d_ln1_beta

            d_h = d_resid0 + d_h_from_ln1

        d_h = d_h.reshape(B, T, C)
        d_token_embedding = np.zeros_like(w["token_embedding"])
        d_position_embedding = np.zeros_like(w["position_embedding"])
        for b in range(B):
            np.add.at(d_token_embedding, cache["ids"][b], d_h[b])
            d_position_embedding[:T] += d_h[b]

        grads["token_embedding"] = d_token_embedding
        grads["position_embedding"] = d_position_embedding
        return grads

    def _attention_backward_batch(self, d_attn_out: np.ndarray, attn_cache: Dict, prefix: str, w: Dict):
        B, T = attn_cache["B"], attn_cache["T"]
        H, hd = self.config.num_heads, self.config.head_dim
        C = H * hd
        ln1_out = attn_cache["ln1_out"]
        q_h, k_h, v_h = attn_cache["q_h"], attn_cache["k_h"], attn_cache["v_h"]
        probs_h, scale = attn_cache["probs_h"], attn_cache["scale"]

        d_attn_out_flat = d_attn_out.reshape(B * T, C)
        attn_concat = attn_cache["attn_concat"]

        d_out_proj = attn_concat.T @ d_attn_out_flat
        d_out_bias = np.sum(d_attn_out_flat, axis=0)
        d_attn_concat = d_attn_out_flat @ w[f"{prefix}.attn_out_proj"].T
        d_head_out = d_attn_concat.reshape(B, T, H, hd).transpose(0, 2, 1, 3)

        d_q = np.zeros((B, T, C), dtype=np.float32)
        d_k = np.zeros((B, T, C), dtype=np.float32)
        d_v = np.zeros((B, T, C), dtype=np.float32)

        for b in range(B):
            for hidx in range(H):
                probs = probs_h[b, hidx]
                d_probs = d_head_out[b, hidx] @ v_h[b, hidx].T
                d_v_h = probs.T @ d_head_out[b, hidx]

                row_dot = np.sum(d_probs * probs, axis=-1, keepdims=True)
                d_raw = probs * (d_probs - row_dot)
                d_raw *= scale

                d_q_h = d_raw @ k_h[b, hidx]
                d_k_h = d_raw.T @ q_h[b, hidx]

                cols = slice(hidx * hd, (hidx + 1) * hd)
                d_q[b, :, cols] += d_q_h
                d_k[b, :, cols] += d_k_h
                d_v[b, :, cols] += d_v_h

        d_qkv = np.concatenate([d_q, d_k, d_v], axis=-1).reshape(B * T, 3 * C)
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

    def _backward_unbatched(self, cache: Dict, dlogits: np.ndarray) -> Dict[str, np.ndarray]:
        """Legacy path for caches without batch metadata (q_h shape [H,T,hd])."""
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

            d_mlp_out = d_h
            d_resid1 = d_h

            d_ln2_out, d_mlp_grads = self._mlp_backward(d_mlp_out, layer_cache["mlp"], prefix, w)
            grads.update(d_mlp_grads)

            d_h_from_ln2, d_ln2_gamma, d_ln2_beta = _layernorm_backward(
                d_ln2_out, layer_cache["ln2_xhat"], layer_cache["ln2_invstd"], w[f"{prefix}.ln2_gamma"],
            )
            grads[f"{prefix}.ln2_gamma"] = d_ln2_gamma
            grads[f"{prefix}.ln2_beta"] = d_ln2_beta

            d_h = d_resid1 + d_h_from_ln2

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


def _squeeze_batch_cache(cache: Dict) -> Dict:
    """Convert B=1 batched cache to legacy shapes for single-seq consumers."""
    if cache.get("B", 1) != 1:
        return cache
    out = dict(cache)
    out["ids"] = cache["ids"][0]
    out["batched"] = False
    if cache.get("gpu"):
        out.pop("B", None)
        out.pop("T", None)
        return out
    for layer_cache in out["layers"]:
        attn = layer_cache["attn"]
        B, H, T, hd = attn["B"], attn["q_h"].shape[1], attn["T"], attn["q_h"].shape[3]
        attn["q_h"] = attn["q_h"][0]
        attn["k_h"] = attn["k_h"][0]
        attn["v_h"] = attn["v_h"][0]
        attn["probs_h"] = attn["probs_h"][0]
        del attn["B"]
    out.pop("B", None)
    out.pop("T", None)
    return out


def _softmax_1d(x: np.ndarray) -> np.ndarray:
    shifted = x - np.max(x)
    exp = np.exp(shifted)
    return exp / np.sum(exp)
