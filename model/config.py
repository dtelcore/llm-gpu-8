"""
model/config.py

Typed view over the `model` section of training_config.json.
"""

from typing import Any, Dict


class GPTConfig:
    def __init__(self, model_dict: Dict[str, Any]) -> None:
        self.name: str = model_dict.get("name", "Custom")
        self.vocab_size: int = int(model_dict["vocab_size"])
        self.max_len: int = int(model_dict["max_len"])
        self.embedding_dim: int = int(model_dict["embedding_dim"])
        self.num_heads: int = int(model_dict["num_heads"])
        self.num_layers: int = int(model_dict["num_layers"])
        self.dropout_prob: float = float(model_dict.get("dropout_prob", 0.0))
        # Share token_embedding with lm_head (lm_head = embedding.T). Default
        # False so legacy checkpoints without the key stay untied; new presets
        # set tie_embeddings=True explicitly.
        self.tie_embeddings: bool = bool(model_dict.get("tie_embeddings", False))
        # "layernorm" (legacy) | "rmsnorm" (scale-only). Default layernorm for old ckpts.
        norm = str(model_dict.get("norm_type", "layernorm")).lower()
        if norm not in ("layernorm", "rmsnorm"):
            raise ValueError(f"norm_type must be 'layernorm' or 'rmsnorm', got {norm!r}")
        self.norm_type: str = norm
        # "learned" absolute position table | "rope". Default learned for old ckpts.
        pos = str(model_dict.get("pos_encoding", "learned")).lower()
        if pos not in ("learned", "rope"):
            raise ValueError(f"pos_encoding must be 'learned' or 'rope', got {pos!r}")
        self.pos_encoding: str = pos
        self.rope_base: float = float(model_dict.get("rope_base", 10000.0))
        self.gradient_checkpointing: bool = bool(model_dict.get("gradient_checkpointing", False))

        assert self.embedding_dim % self.num_heads == 0, (
            f"embedding_dim ({self.embedding_dim}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
        self.head_dim: int = self.embedding_dim // self.num_heads

    @property
    def use_rmsnorm(self) -> bool:
        return self.norm_type == "rmsnorm"

    @property
    def use_rope(self) -> bool:
        return self.pos_encoding == "rope"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout_prob": self.dropout_prob,
            "tie_embeddings": self.tie_embeddings,
            "norm_type": self.norm_type,
            "pos_encoding": self.pos_encoding,
            "rope_base": self.rope_base,
            "gradient_checkpointing": self.gradient_checkpointing,
        }

    def __repr__(self) -> str:
        return (
            f"GPTConfig(name={self.name!r}, vocab_size={self.vocab_size}, "
            f"max_len={self.max_len}, embedding_dim={self.embedding_dim}, "
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"num_layers={self.num_layers}, tie_embeddings={self.tie_embeddings}, "
            f"norm_type={self.norm_type!r}, pos_encoding={self.pos_encoding!r}, "
            f"gradient_checkpointing={self.gradient_checkpointing})"
        )
