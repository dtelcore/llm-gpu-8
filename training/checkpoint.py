"""
training/checkpoint.py

Save/load model weights + tokenizer vocab + config snapshot to output/checkpoints/<run>/.
Supports latest (run root), quarter_XX/, and best/ full bundles plus metrics.json.
"""

import copy
import json
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from logging_config import logger
from model.config import GPTConfig
from model.weights import ModelParameters
from paths import OUTPUT_CHECKPOINTS, OUTPUT_TOKENIZER, checkpoint_vocab_sidecar, ensure_output_dirs
from tokenizer.tokenizer import CharacterGPTTokenizer
from version import __version__


def _resolve_out_dir(checkpoint_dir: str) -> Path:
    out_dir = Path(checkpoint_dir)
    if len(out_dir.parts) == 1:
        out_dir = OUTPUT_CHECKPOINTS / out_dir
    return out_dir


def save_checkpoint(
    checkpoint_dir: str,
    params: ModelParameters,
    tokenizer: CharacterGPTTokenizer,
    config: Dict,
    step: int,
    epoch: int,
    metrics: Optional[Dict] = None,
) -> Path:
    ensure_output_dirs()
    out_dir = _resolve_out_dir(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savez(out_dir / "weights.npz", **params.all_params())
    vocab_path = out_dir / "vocab.json"
    tokenizer.save_vocab(vocab_path)
    sidecar = checkpoint_vocab_sidecar(out_dir)
    OUTPUT_TOKENIZER.mkdir(parents=True, exist_ok=True)
    shutil.copy2(vocab_path, sidecar)
    logger.info(f"Tokenizer vocab mirrored to {sidecar}")

    # The training corpus can be large (thousands of sentences); keep it out of
    # config.json (which should stay a small, human-readable snapshot) and store
    # it in its own file instead. It's still restored on load so resuming works.
    config_to_save = copy.deepcopy(config)
    config_to_save["version"] = __version__
    dataset = config_to_save.setdefault("dataset", {})
    corpus = dataset.pop("corpus", None)
    val_corpus = dataset.pop("val_corpus", None)
    if corpus is not None:
        with open(out_dir / "corpus.json", "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False)
        dataset["num_sentences"] = len(corpus)
    if val_corpus is not None:
        with open(out_dir / "val_corpus.json", "w", encoding="utf-8") as f:
            json.dump(val_corpus, f, ensure_ascii=False)
        dataset["num_val_sentences"] = len(val_corpus)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=2, ensure_ascii=False)

    state = {"step": step, "epoch": epoch, "version": __version__}
    with open(out_dir / "state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    if metrics is not None:
        save_metrics(out_dir, metrics)

    logger.info(f"Checkpoint saved to {out_dir} (epoch={epoch}, step={step})")
    return out_dir


def save_metrics(checkpoint_dir: Path, metrics: Dict) -> None:
    """Write metrics.json (latest snapshot of train/val/quality scores)."""
    out_dir = Path(checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(metrics)
    payload["version"] = __version__
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_metrics(checkpoint_dir: Path) -> Dict:
    path = Path(checkpoint_dir) / "metrics.json"
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(checkpoint_dir: str):
    """Returns (GPTConfig, ModelParameters, CharacterGPTTokenizer, full_config_dict, state_dict)."""
    ckpt_dir = Path(checkpoint_dir)

    with open(ckpt_dir / "config.json", "r", encoding="utf-8") as f:
        full_config = json.load(f)

    corpus_path = ckpt_dir / "corpus.json"
    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            full_config.setdefault("dataset", {})["corpus"] = json.load(f)

    val_corpus_path = ckpt_dir / "val_corpus.json"
    if val_corpus_path.exists():
        with open(val_corpus_path, "r", encoding="utf-8") as f:
            full_config.setdefault("dataset", {})["val_corpus"] = json.load(f)

    tokenizer = CharacterGPTTokenizer.load_vocab(ckpt_dir / "vocab.json")
    gpt_config = GPTConfig(full_config["model"])

    params = ModelParameters(gpt_config, init_scales=full_config.get("weight_initialization", {}))
    params.load(str(ckpt_dir / "weights.npz"))

    state = {"step": 0, "epoch": 0, "version": full_config.get("version", "0.0.0")}
    state_path = ckpt_dir / "state.json"
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)

    logger.info(f"Checkpoint loaded from {ckpt_dir}")
    return gpt_config, params, tokenizer, full_config, state


def copy_checkpoint_bundle(src_dir: Path, dst_dir: Path) -> Path:
    """Copy a full checkpoint bundle (weights/config/vocab/corpus/state/metrics) to dst."""
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    if dst_dir.exists():
        shutil.rmtree(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    for name in (
        "weights.npz",
        "config.json",
        "vocab.json",
        "corpus.json",
        "val_corpus.json",
        "state.json",
        "metrics.json",
    ):
        src = src_dir / name
        if src.exists():
            shutil.copy2(src, dst_dir / name)

    logger.info(f"Checkpoint bundle copied {src_dir} -> {dst_dir}")
    return dst_dir


def promote_best(
    run_dir: Path,
    source_dir: Path,
    *,
    meta: Optional[Dict] = None,
) -> Path:
    """Copy source_dir into run_dir/best/ and write best_meta.json."""
    from paths import BEST_DIR_NAME

    run_dir = Path(run_dir)
    best = run_dir / BEST_DIR_NAME
    copy_checkpoint_bundle(source_dir, best)
    payload = {
        "version": __version__,
        "source": str(source_dir),
        "source_name": Path(source_dir).name,
    }
    if meta:
        payload.update(meta)
    with open(best / "best_meta.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info(f"Promoted best checkpoint from {source_dir} -> {best}")
    return best
