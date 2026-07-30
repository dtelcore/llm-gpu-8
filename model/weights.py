"""
model/weights.py

NumPy host-side parameter storage for the GPT model. Allocates float32
arrays matching the layout used by setup/model_config.estimate_vram_footprint,
and initializes them via setup/weight_init.WeightInitializer using the
per-layer-type scales already computed by setup/training_setup.py.

V2: also keeps a persistent GPU-resident mirror of every weight/bias tensor
(device_weights / device_biases). Forward-pass matmuls read straight from
these mirrors instead of re-uploading the same weight matrix on every single
op call -- that per-call host->device transfer was the dominant cost on the
GT 730's slow PCIe link, dwarfing the actual matmul time for these small
matrices. The mirrors are refreshed once per optimizer step via sync_device(),
not once per layer op.
"""

from typing import Dict, Iterable, Optional, Tuple

import numpy as np

from model.config import GPTConfig
from setup.weight_init import WeightInitializer


class ModelParameters:
    """Owns every trainable NumPy array for a GPTConfig, plus a GPU-resident mirror."""

    def __init__(self, config: GPTConfig, init_scales: Dict[str, float] = None, seed: int = 42) -> None:
        self.config = config
        self.scales = init_scales or {}
        self.weights: Dict[str, np.ndarray] = {}
        self.biases: Dict[str, np.ndarray] = {}
        self.device_weights: Dict[str, "pycuda.gpuarray.GPUArray"] = {}
        self.device_biases: Dict[str, "pycuda.gpuarray.GPUArray"] = {}
        self._rng = np.random.default_rng(seed)
        self.allocate_and_init()
        self.upload_to_device()

    def allocate_and_init(self) -> None:
        C = self.config.embedding_dim
        V = self.config.vocab_size
        max_len = self.config.max_len
        use_rmsnorm = getattr(self.config, "use_rmsnorm", False)
        use_rope = getattr(self.config, "use_rope", False)

        self.weights["token_embedding"] = self._init("token_embedding", (V, C), V, C)
        if not use_rope:
            self.weights["position_embedding"] = self._init("position_embedding", (max_len, C), max_len, C)

        for layer in range(self.config.num_layers):
            prefix = f"layer_{layer}"

            self.weights[f"{prefix}.qkv_proj"] = self._init("qkv_proj", (C, 3 * C), C, 3 * C)
            self.biases[f"{prefix}.qkv_bias"] = WeightInitializer.bias_init((3 * C,))

            self.weights[f"{prefix}.attn_out_proj"] = self._init("attention_output_proj", (C, C), C, C)
            self.biases[f"{prefix}.attn_out_bias"] = WeightInitializer.bias_init((C,))

            gamma1, beta1 = WeightInitializer.layernorm_init((C,))
            self.weights[f"{prefix}.ln1_gamma"] = gamma1
            if not use_rmsnorm:
                self.biases[f"{prefix}.ln1_beta"] = beta1

            gamma2, beta2 = WeightInitializer.layernorm_init((C,))
            self.weights[f"{prefix}.ln2_gamma"] = gamma2
            if not use_rmsnorm:
                self.biases[f"{prefix}.ln2_beta"] = beta2

            self.weights[f"{prefix}.mlp_expand"] = self._init("mlp_expand", (C, 4 * C), C, 4 * C)
            self.biases[f"{prefix}.mlp_expand_bias"] = WeightInitializer.bias_init((4 * C,))

            self.weights[f"{prefix}.mlp_contract"] = self._init("mlp_contract", (4 * C, C), 4 * C, C)
            self.biases[f"{prefix}.mlp_contract_bias"] = WeightInitializer.bias_init((C,))

        final_gamma, final_beta = WeightInitializer.layernorm_init((C,))
        self.weights["final_ln_gamma"] = final_gamma
        if not use_rmsnorm:
            self.biases["final_ln_beta"] = final_beta

        # Tied: lm_head is a [C, V] view of token_embedding.T (same storage).
        # Untied: allocate a separate [C, V] matrix (legacy checkpoints).
        if self.config.tie_embeddings:
            self.weights["lm_head"] = self.weights["token_embedding"].T
        else:
            self.weights["lm_head"] = self._init("lm_head", (C, V), C, V)
        self.biases["lm_head_bias"] = WeightInitializer.bias_init((V,))

    # setup/weight_init.WeightInitializer.layer_init_scale only recognizes these
    # canonical type strings; our per-tensor names (token_embedding, etc.) map
    # onto them for the fallback path when no precomputed scale is supplied.
    _CANONICAL_TYPE = {
        "token_embedding": "embedding",
        "position_embedding": "embedding",
        "qkv_proj": "qkv_proj",
        "attention_output_proj": "output_proj",
        "mlp_expand": "mlp_expand",
        "mlp_contract": "mlp_contract",
        "lm_head": "lm_head",
    }

    def _init(self, layer_type: str, shape: Tuple[int, int], fan_in: int, fan_out: int) -> np.ndarray:
        scale = self.scales.get(layer_type)
        if scale is None:
            canonical = self._CANONICAL_TYPE.get(layer_type, layer_type)
            scale = WeightInitializer.layer_init_scale(canonical, fan_in, fan_out)
        return (self._rng.standard_normal(shape) * scale).astype(np.float32)

    @property
    def tie_embeddings(self) -> bool:
        return bool(getattr(self.config, "tie_embeddings", False))

    def trainable_weight_names(self) -> Tuple[str, ...]:
        """Weight keys owned by the optimizer (excludes tied lm_head view)."""
        names = [n for n in self.weights.keys() if not (self.tie_embeddings and n == "lm_head")]
        return tuple(names)

    def trainable_param_names(self) -> Tuple[str, ...]:
        return self.trainable_weight_names() + tuple(self.biases.keys())

    def all_params(self) -> Dict[str, np.ndarray]:
        """All weights and biases in a single flat dict, keyed by name.

        When embeddings are tied, lm_head is materialized as a contiguous
        transpose copy so checkpoints remain loadable by older tooling.
        """
        merged: Dict[str, np.ndarray] = {}
        for name, arr in self.weights.items():
            if self.tie_embeddings and name == "lm_head":
                merged[name] = np.ascontiguousarray(arr)
            else:
                merged[name] = arr
        merged.update(self.biases)
        return merged

    def param_count(self) -> int:
        """Unique trainable parameters (tied lm_head is not double-counted)."""
        n = sum(self.weights[k].size for k in self.trainable_weight_names())
        n += sum(arr.size for arr in self.biases.values())
        return n

    def save(self, filepath: str) -> None:
        np.savez(filepath, **self.all_params())

    def load(self, filepath: str) -> None:
        data = np.load(filepath)
        for key in data.files:
            if key == "lm_head" and self.tie_embeddings:
                continue  # view rebound after token_embedding load
            if key in self.weights:
                self.weights[key] = data[key].astype(np.float32)
            elif key in self.biases:
                self.biases[key] = data[key].astype(np.float32)

        if self.tie_embeddings:
            te = self.weights["token_embedding"]
            # Migrate untied → tied: average embedding with lm_head.T when both exist.
            if "lm_head" in data.files:
                lh = data["lm_head"].astype(np.float32)
                if lh.shape == (te.shape[1], te.shape[0]):
                    self.weights["token_embedding"] = (0.5 * (te + lh.T)).astype(np.float32)
                    te = self.weights["token_embedding"]
            self.weights["lm_head"] = te.T
        self.sync_device()

    # ------------------------------------------------------------------
    # V2: GPU-resident mirror
    # ------------------------------------------------------------------

    def _bind_tied_lm_head_device(self) -> None:
        """Point device lm_head at token_embedding.T (same gpudata, no extra VRAM)."""
        if not self.tie_embeddings:
            return
        self.device_weights["lm_head"] = self.device_weights["token_embedding"].T
        self.weights["lm_head"] = self.weights["token_embedding"].T

    def upload_to_device(self) -> None:
        """Upload every weight/bias tensor to the GPU once. Called at construction
        and after load(); training calls sync_device() after each optimizer step
        instead of re-running this from scratch."""
        from model.cuda import ops
        self.device_weights = {}
        for name, arr in self.weights.items():
            if self.tie_embeddings and name == "lm_head":
                continue
            self.device_weights[name] = ops.to_device(np.ascontiguousarray(arr))
        self.device_biases = {name: ops.to_device(arr) for name, arr in self.biases.items()}
        self._bind_tied_lm_head_device()

    def sync_device(self, names: Optional[Iterable[str]] = None) -> None:
        """Re-upload the current NumPy values to their persistent GPU mirrors.
        Call this once per optimizer step (after optimizer.step() mutates
        self.weights/self.biases in place) -- NOT once per layer op."""
        from model.cuda import ops
        if names is None:
            keys = list(self.trainable_param_names())
        else:
            keys = [n for n in names if not (self.tie_embeddings and n == "lm_head")]
        for name in keys:
            if name in self.weights:
                self.device_weights[name] = ops.to_device(np.ascontiguousarray(self.weights[name]))
            elif name in self.biases:
                self.device_biases[name] = ops.to_device(self.biases[name])
        self._bind_tied_lm_head_device()
