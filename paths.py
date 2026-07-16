"""
paths.py

Central path layout for the GT 730 training workspace. All runtime artifacts
(logs, checkpoints, configs, tokenizer exports, caches) live under output/
at the project root. Input corpora stay in data/ (not gitignored artifacts).

Layout (mirrors llm gpu 5 output/ convention, adapted for gpu 8's
directory-based checkpoint bundles):

    output/
      logs/          training logs (.log), plots
      checkpoints/   per-run checkpoint dirs (weights.npz, config.json, ...)
      configs/       training_config.json and wizard exports
      tokenizer/     vocab.json sidecars / tokenizer exports
      cache/         optional tokenizer / encoding caches
"""

from pathlib import Path
from typing import Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent

OUTPUT_ROOT = PROJECT_ROOT / "output"
OUTPUT_LOGS = OUTPUT_ROOT / "logs"
OUTPUT_CHECKPOINTS = OUTPUT_ROOT / "checkpoints"
OUTPUT_CONFIGS = OUTPUT_ROOT / "configs"
OUTPUT_TOKENIZER = OUTPUT_ROOT / "tokenizer"
OUTPUT_CACHE = OUTPUT_ROOT / "cache"
OUTPUT_CACHE_TOKENIZER = OUTPUT_CACHE / "tokenizer"

DATA_DIR = PROJECT_ROOT / "data"
SETUP_DIR = PROJECT_ROOT / "setup"

DEFAULT_CONFIG_PATH = OUTPUT_CONFIGS / "training_config.json"
DEFAULT_CHECKPOINT_DIR = OUTPUT_CHECKPOINTS / "run1"
DEFAULT_TRAINING_LOG = OUTPUT_LOGS / "training.log"
DEFAULT_TRAINING_PLOT = OUTPUT_LOGS / "training_plot_latest.png"
DEFAULT_LANDSCAPE_PLOT = OUTPUT_LOGS / "loss_landscape_latest.png"

# Legacy locations (pre-output/ migration) — still honored when passed explicitly.
LEGACY_LOGS_DIR = PROJECT_ROOT / "logs"
LEGACY_MODELS_DIR = PROJECT_ROOT / "models"
LEGACY_CONFIG_PATH = SETUP_DIR / "training_config.json"


def ensure_output_dirs() -> None:
    """Create the output/ tree if missing."""
    for d in (
        OUTPUT_LOGS,
        OUTPUT_CHECKPOINTS,
        OUTPUT_CONFIGS,
        OUTPUT_TOKENIZER,
        OUTPUT_CACHE_TOKENIZER,
    ):
        d.mkdir(parents=True, exist_ok=True)


def resolve_checkpoints_dir(path: Optional[Union[str, Path]] = None) -> Path:
    """Return checkpoints root, defaulting to output/checkpoints."""
    if path is None:
        return OUTPUT_CHECKPOINTS
    return Path(path)


def resolve_config_path(path: Optional[Union[str, Path]] = None) -> Path:
    """Return config file path, defaulting to output/configs/training_config.json."""
    if path is None:
        return DEFAULT_CONFIG_PATH
    return Path(path)


def checkpoint_vocab_sidecar(checkpoint_dir: Union[str, Path]) -> Path:
    """Path for a vocab.json copy under output/tokenizer/ for a checkpoint run."""
    stem = Path(checkpoint_dir).name
    return OUTPUT_TOKENIZER / f"{stem}_vocab.json"
