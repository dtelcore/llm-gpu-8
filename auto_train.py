"""
auto_train.py

Orchestrator: run train.py end-to-end, then immediately sample from the
resulting checkpoint as a smoke test. One command for
"config -> trained weights -> generated text".

Usage:
    python auto_train.py --config setup/training_config.json --checkpoint models/run1
    python auto_train.py --epochs 5 --prompt "once upon a" --trace-logits
    python auto_train.py --steps 1500 --prompt "once upon a"
"""

import argparse

import cli_common
from generate import generate as run_generate
from train import train as run_train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train then generate a smoke sample, end to end")
    cli_common.add_config_arg(parser)
    cli_common.add_checkpoint_arg(parser)
    cli_common.add_seed_arg(parser)
    cli_common.add_training_length_args(parser)
    parser.add_argument("--prompt", type=str, default="the", help="Prompt for the post-training smoke sample")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Characters to generate for the smoke sample")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature for the smoke sample")
    cli_common.add_trace_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("### STAGE 1/2: TRAINING ###")
    checkpoint_dir = run_train(args)

    print("\n### STAGE 2/2: SMOKE-TEST GENERATION ###")
    # Reuse the same trace flags for both stages; only the sampling params differ.
    args.checkpoint = checkpoint_dir
    run_generate(args)


if __name__ == "__main__":
    main()
