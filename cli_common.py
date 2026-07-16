"""
cli_common.py

Shared argparse building blocks for train.py, auto_train.py, generate.py,
and interactive.py so trace flags and paths stay consistent across CLIs.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from model.trace import TraceContext
from logging_config import logger
from paths import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CONFIG_PATH,
    LEGACY_CONFIG_PATH,
    resolve_checkpoints_dir,
)


def add_trace_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("tracing (all quiet by default)")
    group.add_argument("--verbose", action="store_true", help="Enable token + top-k logit traces on traced steps")
    group.add_argument("--trace-logits", action="store_true", help="Dump top-k logits/probabilities on traced steps")
    group.add_argument("--trace-tokens", action="store_true", help="Dump token <-> id mapping on traced steps")
    group.add_argument("--trace-neurons", action="store_true", help="Dump per-layer activation mean/std/norm on traced steps")
    group.add_argument("--trace-vectorization", action="store_true", help="Print GEMM shapes and CUDA grid/block launches on traced steps")
    group.add_argument("--trace-every", type=int, default=None, help="Emit traces every N steps (default: 10%% of total training steps; ignored for generate/interactive where it defaults to every step)")


def add_config_arg(parser: argparse.ArgumentParser, default: Optional[str] = None) -> None:
    default_path = str(default or DEFAULT_CONFIG_PATH)
    parser.add_argument("--config", type=str, default=default_path, help="Path to training_config.json")


def add_training_length_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("training length")
    group.add_argument("--learning-rate", type=float, default=None, help="Override config learning_rate (prompted interactively if omitted)")
    group.add_argument("--epochs", type=int, default=None, help="Override config num_epochs (ignored if --steps is given)")
    group.add_argument("--steps", type=int, default=None, help="Total training steps across ALL epochs (overrides --epochs and config num_epochs)")
    group.add_argument("--log-every", type=int, default=100, help="Print a progress line every N steps (default: 100)")
    group.add_argument(
        "--checkpoint-every", type=int, default=None,
        help="Save a checkpoint every N steps (default: min(1000, batches/epoch); override to tune disk I/O vs crash safety)",
    )
    group.add_argument("--no-prompt", action="store_true", help="Never prompt for learning_rate/steps/epochs; silently use config/CLI defaults")


def prompt_training_length_and_lr(args: argparse.Namespace, hyperparams: Dict) -> None:
    """Interactively ask for learning_rate, and either total steps or epochs, filling
    in only whichever of `args.learning_rate` / `args.steps` / `args.epochs` weren't
    already supplied on the command line. Mutates `args` in place. No-op if --no-prompt
    was passed or stdin isn't interactive.

    Falls back silently to config/CLI values on EOF (e.g. piped/non-interactive input).
    """
    if getattr(args, "no_prompt", False):
        return

    try:
        if getattr(args, "learning_rate", None) is None:
            default_lr = hyperparams.get("learning_rate", 0.01)
            raw = input(f"Learning rate [default={default_lr}]: ").strip()
            args.learning_rate = float(raw) if raw else default_lr

        if args.steps is None and args.epochs is None:
            default_epochs = hyperparams.get("num_epochs", 10)
            raw_steps = input("Total steps (blank to specify epochs instead): ").strip()
            if raw_steps:
                args.steps = int(raw_steps)
            else:
                raw_epochs = input(f"Epochs [default={default_epochs}]: ").strip()
                args.epochs = int(raw_epochs) if raw_epochs else default_epochs
    except EOFError:
        if getattr(args, "learning_rate", None) is None:
            args.learning_rate = hyperparams.get("learning_rate", 0.01)
        if args.steps is None and args.epochs is None:
            args.epochs = hyperparams.get("num_epochs", 10)


def add_model_hyperparam_args(parser: argparse.ArgumentParser) -> None:
    """Model architecture overrides (applied to config['model'] before the GPTConfig
    is built). Weight init still runs off whatever the final values end up being."""
    group = parser.add_argument_group("model hyperparameters")
    group.add_argument("--embedding-dim", type=int, default=None, help="Override config embedding_dim")
    group.add_argument("--num-heads", type=int, default=None, help="Override config num_heads")
    group.add_argument("--num-layers", type=int, default=None, help="Override config num_layers")
    group.add_argument("--max-len", type=int, default=None, help="Override config max_len (context window)")
    group.add_argument("--dropout", type=float, default=None, help="Override config dropout_prob")
    group.add_argument("--batch-size", type=int, default=None, help="Override config batch_size")
    group.add_argument("--weight-decay", type=float, default=None, help="Override config weight_decay")
    group.add_argument("--warmup-steps", type=int, default=None, help="Override config warmup_steps")
    group.add_argument("--gradient-clip", type=float, default=None, help="Override config gradient_clip norm")


def prompt_model_hyperparams(args: argparse.Namespace, model_config: Dict, hyperparams: Dict) -> None:
    """Interactively fill in model architecture (embedding_dim/num_heads/num_layers/
    max_len/dropout) and remaining training hyperparameters (batch_size/weight_decay/
    warmup_steps/gradient_clip) that weren't already supplied on the command line.
    Mutates `model_config` and `hyperparams` in place. No-op if --no-prompt was passed.
    Falls back silently to config/CLI values on EOF (non-interactive stdin).
    """
    no_prompt = getattr(args, "no_prompt", False)

    def _ask(flag_attr: str, label: str, target: Dict, key: str, caster, current_default):
        # An explicit CLI flag always wins, --no-prompt or not.
        if getattr(args, flag_attr, None) is not None:
            target[key] = getattr(args, flag_attr)
            return
        if no_prompt:
            target[key] = current_default
            return
        try:
            raw = input(f"{label} [default={current_default}]: ").strip()
            target[key] = caster(raw) if raw else current_default
        except EOFError:
            target[key] = current_default

    _ask("embedding_dim", "Embedding dim", model_config, "embedding_dim", int, model_config.get("embedding_dim", 32))
    _ask("num_heads", "Num attention heads", model_config, "num_heads", int, model_config.get("num_heads", 4))
    _ask("num_layers", "Num transformer layers", model_config, "num_layers", int, model_config.get("num_layers", 2))
    _ask("max_len", "Max sequence length (context window)", model_config, "max_len", int, model_config.get("max_len", 16))
    _ask("dropout", "Dropout probability", model_config, "dropout_prob", float, model_config.get("dropout_prob", 0.0))

    _ask("batch_size", "Batch size", hyperparams, "batch_size", int, hyperparams.get("batch_size", 2))
    _ask("weight_decay", "Weight decay", hyperparams, "weight_decay", float, hyperparams.get("weight_decay", 0.01))
    _ask("warmup_steps", "Warmup steps", hyperparams, "warmup_steps", int, hyperparams.get("warmup_steps", 0))
    _ask("gradient_clip", "Gradient clip norm", hyperparams, "gradient_clip", float, hyperparams.get("gradient_clip", 1.0))


def add_checkpoint_arg(parser: argparse.ArgumentParser, default: Optional[str] = None) -> None:
    default_ckpt = str(default or DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Checkpoint directory")


def add_seed_arg(parser: argparse.ArgumentParser, default: int = 42) -> None:
    parser.add_argument("--seed", type=int, default=default, help="Random seed")


def build_tracer(args, default_trace_every: int = 100) -> TraceContext:
    """Build a TraceContext, resolving --trace-every to `default_trace_every`
    when the user didn't pass it explicitly (args.trace_every is None)."""
    tracer = TraceContext.from_args(args)
    resolved = args.trace_every if getattr(args, "trace_every", None) is not None else default_trace_every
    tracer.trace_every = max(1, resolved)
    return tracer


def load_config(path: str) -> Dict:
    config_path = Path(path)
    if not config_path.exists():
        if config_path.resolve() == DEFAULT_CONFIG_PATH.resolve() and LEGACY_CONFIG_PATH.exists():
            logger.info(
                "Config not found at %s; falling back to legacy %s",
                DEFAULT_CONFIG_PATH, LEGACY_CONFIG_PATH,
            )
            config_path = LEGACY_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_checkpoints(models_dir: Optional[str] = None) -> List[Path]:
    """Return checkpoint directories under checkpoints root (any dir holding config.json),
    newest first."""
    root = resolve_checkpoints_dir(models_dir)
    if not root.exists():
        return []
    found = [d for d in root.iterdir() if d.is_dir() and (d / "config.json").exists()]
    return sorted(found, key=lambda d: d.stat().st_mtime, reverse=True)


def prompt_resume_or_new(models_dir: Optional[str] = None) -> Optional[str]:
    """First-menu-option prompt: resume training from an existing checkpoint, or
    start a fresh run. Returns the checkpoint path to resume from, or None to
    signal 'start fresh' (including when no checkpoints exist or stdin is EOF)."""
    checkpoints = list_checkpoints(models_dir)
    ckpt_root = resolve_checkpoints_dir(models_dir)

    print(f"\nAvailable checkpoints in '{ckpt_root}':")
    if checkpoints:
        for i, d in enumerate(checkpoints, 1):
            print(f"  {i}. {d}")
    else:
        print("  (none found)")
    print("  n. Start a new training run")

    if not checkpoints:
        return None

    try:
        choice = input("\nResume from checkpoint (number), or 'n' for new [default=n]: ").strip()
    except EOFError:
        return None

    if not choice or choice.lower() == "n":
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(checkpoints):
        return str(checkpoints[int(choice) - 1])

    candidate = Path(choice)
    if (candidate / "config.json").exists():
        return str(candidate)

    print(f"'{choice}' not recognized; starting a new training run.")
    return None


def select_checkpoint_interactive(
    models_dir: Optional[str] = None,
    allow_new: bool = False,
    default_new_name: str = "run1",
    prompt_label: str = "checkpoint",
) -> str:
    """Interactively list checkpoints under `models_dir` and let the user pick one
    by number, type a custom path directly, or (if allow_new) create a new name.

    Returns the chosen checkpoint directory as a string path.
    """
    checkpoints = list_checkpoints(models_dir)
    ckpt_root = resolve_checkpoints_dir(models_dir)

    print(f"\nAvailable checkpoints in '{ckpt_root}':")
    if checkpoints:
        for i, d in enumerate(checkpoints, 1):
            print(f"  {i}. {d}")
    else:
        print("  (none found)")
    if allow_new:
        print("  n. Enter a new checkpoint name")

    while True:
        hint = " (number, path, or 'n' for new)" if allow_new else " (number or path)"
        choice = input(f"\nSelect {prompt_label}{hint}: ").strip()

        if not choice:
            if checkpoints:
                return str(checkpoints[0])
            if allow_new:
                choice = "n"
            else:
                print("No checkpoints available; please enter a path.")
                continue

        if allow_new and choice.lower() == "n":
            name = input(f"New checkpoint name [default={default_new_name}]: ").strip() or default_new_name
            if "/" in name or "\\" in name:
                return name
            return str(ckpt_root / name)

        if choice.isdigit() and checkpoints and 1 <= int(choice) <= len(checkpoints):
            return str(checkpoints[int(choice) - 1])

        # Fall back to treating the input as a literal path.
        candidate = Path(choice)
        if not allow_new and not (candidate / "config.json").exists():
            print(f"'{choice}' has no config.json; pick a valid checkpoint.")
            continue
        return choice
