"""
training/checkpoint.py

Save/load model weights + tokenizer vocab + config snapshot to models/<run>/.
"""

import copy
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

    # The training corpus can be large (thousands of sentences); keep it out of
    # config.json (which should stay a small, human-readable snapshot) and store
    # it in its own file instead. It's still restored on load so resuming works.
    config_to_save = copy.deepcopy(config)
    corpus = config_to_save.get("dataset", {}).pop("corpus", None)
    if corpus is not None:
        with open(out_dir / "corpus.json", "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False)
        config_to_save["dataset"]["num_sentences"] = len(corpus)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=2, ensure_ascii=False)

    with open(out_dir / "state.json", "w", encoding="utf-8") as f:
        json.dump({"step": step, "epoch": epoch}, f, indent=2)

    logger.info(f"Checkpoint saved to {out_dir} (epoch={epoch}, step={step})")
    return out_dir


def load_checkpoint(checkpoint_dir: str):
    """Returns (GPTConfig, ModelParameters, CharacterGPTTokenizer, full_config_dict, state_dict)."""
    ckpt_dir = Path(checkpoint_dir)

    with open(ckpt_dir / "config.json", "r", encoding="utf-8") as f:
        full_config = json.load(f)

    corpus_path = ckpt_dir / "corpus.json"
    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            full_config.setdefault("dataset", {})["corpus"] = json.load(f)

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
