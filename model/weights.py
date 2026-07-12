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

        self.weights["token_embedding"] = self._init("token_embedding", (V, C), V, C)
        self.weights["position_embedding"] = self._init("position_embedding", (max_len, C), max_len, C)

        for layer in range(self.config.num_layers):
            prefix = f"layer_{layer}"

            self.weights[f"{prefix}.qkv_proj"] = self._init("qkv_proj", (C, 3 * C), C, 3 * C)
            self.biases[f"{prefix}.qkv_bias"] = WeightInitializer.bias_init((3 * C,))

            self.weights[f"{prefix}.attn_out_proj"] = self._init("attention_output_proj", (C, C), C, C)
            self.biases[f"{prefix}.attn_out_bias"] = WeightInitializer.bias_init((C,))

            gamma1, beta1 = WeightInitializer.layernorm_init((C,))
            self.weights[f"{prefix}.ln1_gamma"] = gamma1
            self.biases[f"{prefix}.ln1_beta"] = beta1

            gamma2, beta2 = WeightInitializer.layernorm_init((C,))
            self.weights[f"{prefix}.ln2_gamma"] = gamma2
            self.biases[f"{prefix}.ln2_beta"] = beta2

            self.weights[f"{prefix}.mlp_expand"] = self._init("mlp_expand", (C, 4 * C), C, 4 * C)
            self.biases[f"{prefix}.mlp_expand_bias"] = WeightInitializer.bias_init((4 * C,))

            self.weights[f"{prefix}.mlp_contract"] = self._init("mlp_contract", (4 * C, C), 4 * C, C)
            self.biases[f"{prefix}.mlp_contract_bias"] = WeightInitializer.bias_init((C,))

        final_gamma, final_beta = WeightInitializer.layernorm_init((C,))
        self.weights["final_ln_gamma"] = final_gamma
        self.biases["final_ln_beta"] = final_beta

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

    def all_params(self) -> Dict[str, np.ndarray]:
        """All weights and biases in a single flat dict, keyed by name."""
        merged = dict(self.weights)
        merged.update(self.biases)
        return merged

    def param_count(self) -> int:
        return sum(arr.size for arr in self.all_params().values())

    def save(self, filepath: str) -> None:
        np.savez(filepath, **self.all_params())

    def load(self, filepath: str) -> None:
        data = np.load(filepath)
        for key in data.files:
            if key in self.weights:
                self.weights[key] = data[key].astype(np.float32)
            elif key in self.biases:
                self.biases[key] = data[key].astype(np.float32)
        self.sync_device()

    # ------------------------------------------------------------------
    # V2: GPU-resident mirror
    # ------------------------------------------------------------------

    def upload_to_device(self) -> None:
        """Upload every weight/bias tensor to the GPU once. Called at construction
        and after load(); training calls sync_device() after each optimizer step
        instead of re-running this from scratch."""
        from model.cuda import ops
        self.device_weights = {name: ops.to_device(arr) for name, arr in self.weights.items()}
        self.device_biases = {name: ops.to_device(arr) for name, arr in self.biases.items()}

    def sync_device(self, names: Optional[Iterable[str]] = None) -> None:
        """Re-upload the current NumPy values to their persistent GPU mirrors.
        Call this once per optimizer step (after optimizer.step() mutates
        self.weights/self.biases in place) -- NOT once per layer op."""
        from model.cuda import ops
        keys = names if names is not None else list(self.weights.keys()) + list(self.biases.keys())
        for name in keys:
            if name in self.weights:
                self.device_weights[name] = ops.to_device(self.weights[name])
            elif name in self.biases:
                self.device_biases[name] = ops.to_device(self.biases[name])
