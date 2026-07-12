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
    group.add_argument("--trace-every", type=int, default=100, help="Emit traces every N steps (default: 100)")


def add_config_arg(parser: argparse.ArgumentParser, default: str = "setup/training_config.json") -> None:
    parser.add_argument("--config", type=str, default=default, help="Path to training_config.json")


def add_checkpoint_arg(parser: argparse.ArgumentParser, default: str = "models/run1") -> None:
    parser.add_argument("--checkpoint", type=str, default=default, help="Checkpoint directory")


def add_seed_arg(parser: argparse.ArgumentParser, default: int = 42) -> None:
    parser.add_argument("--seed", type=int, default=default, help="Random seed")


def build_tracer(args) -> TraceContext:
    return TraceContext.from_args(args)


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
