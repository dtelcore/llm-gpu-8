"""
training/checkpoint.py

Save/load model weights + tokenizer vocab + config snapshot to models/<run>/.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np

from logging_config import logger
from model.config import GPTConfig
from model.weights import ModelParameters
from tokenizer.tokenizer import CharacterGPTTokenizer


def save_checkpoint(
    checkpoint_dir: str,
    params: ModelParameters,
    tokenizer: CharacterGPTTokenizer,
    config: Dict,
    step: int,
    epoch: int,
) -> Path:
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(out_dir / "weights.npz", **params.all_params())
    tokenizer.save_vocab(out_dir / "vocab.json")

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(out_dir / "state.json", "w", encoding="utf-8") as f:
        json.dump({"step": step, "epoch": epoch}, f, indent=2)

    logger.info(f"Checkpoint saved to {out_dir} (epoch={epoch}, step={step})")
    return out_dir


def load_checkpoint(checkpoint_dir: str):
    """Returns (GPTConfig, ModelParameters, CharacterGPTTokenizer, full_config_dict, state_dict)."""
    ckpt_dir = Path(checkpoint_dir)

    with open(ckpt_dir / "config.json", "r", encoding="utf-8") as f:
        full_config = json.load(f)

    tokenizer = CharacterGPTTokenizer.load_vocab(ckpt_dir / "vocab.json")
    gpt_config = GPTConfig(full_config["model"])

    params = ModelParameters(gpt_config, init_scales=full_config.get("weight_initialization", {}))
    params.load(str(ckpt_dir / "weights.npz"))

    state = {"step": 0, "epoch": 0}
    state_path = ckpt_dir / "state.json"
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)

    logger.info(f"Checkpoint loaded from {ckpt_dir}")
    return gpt_config, params, tokenizer, full_config, state
