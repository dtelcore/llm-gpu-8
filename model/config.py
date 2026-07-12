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

        assert self.embedding_dim % self.num_heads == 0, (
            f"embedding_dim ({self.embedding_dim}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
        self.head_dim: int = self.embedding_dim // self.num_heads

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "vocab_size": self.vocab_size,
            "max_len": self.max_len,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "dropout_prob": self.dropout_prob,
        }

    def __repr__(self) -> str:
        return (
            f"GPTConfig(name={self.name!r}, vocab_size={self.vocab_size}, "
            f"max_len={self.max_len}, embedding_dim={self.embedding_dim}, "
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"num_layers={self.num_layers})"
        )
