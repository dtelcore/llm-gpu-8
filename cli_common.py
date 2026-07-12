"""
cli_common.py

Shared argparse building blocks for train.py, auto_train.py, generate.py,
and interactive.py so trace flags and paths stay consistent across CLIs.
"""

import argparse
import json
from typing import Dict

from model.trace import TraceContext


def add_trace_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("tracing (all quiet by default)")
    group.add_argument("--verbose", action="store_true", help="Enable token + top-k logit traces on traced steps")
    group.add_argument("--trace-logits", action="store_true", help="Dump top-k logits/probabilities on traced steps")
    group.add_argument("--trace-tokens", action="store_true", help="Dump token <-> id mapping on traced steps")
    group.add_argument("--trace-neurons", action="store_true", help="Dump per-layer activation mean/std/norm on traced steps")
    group.add_argument("--trace-vectorization", action="store_true", help="Print GEMM shapes and CUDA grid/block launches on traced steps")
    group.add_argument("--trace-every", type=int, default=None, help="Emit traces every N steps (default: 10%% of total training steps; ignored for generate/interactive where it defaults to every step)")


def add_config_arg(parser: argparse.ArgumentParser, default: str = "setup/training_config.json") -> None:
    parser.add_argument("--config", type=str, default=default, help="Path to training_config.json")


def add_training_length_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("training length")
    group.add_argument("--epochs", type=int, default=None, help="Override config num_epochs (ignored if --steps is given)")
    group.add_argument("--steps", type=int, default=None, help="Total training steps across ALL epochs (overrides --epochs and config num_epochs)")
    group.add_argument("--log-every", type=int, default=100, help="Print a progress line every N steps (default: 100)")
    group.add_argument("--checkpoint-every", type=int, default=None, help="Save a checkpoint every N steps (default: once per full dataset pass)")


def add_checkpoint_arg(parser: argparse.ArgumentParser, default: str = "models/run1") -> None:
    parser.add_argument("--checkpoint", type=str, default=default, help="Checkpoint directory")


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
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
