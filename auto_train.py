"""
auto_train.py

Orchestrator: run train.py end-to-end, then immediately sample from the
resulting checkpoint as a smoke test. One command for
"config -> trained weights -> generated text".

v0.1.0: shares quarterly checkpoints, val metrics, and quality-trial flags
with train.py.

Usage:
    python auto_train.py --config output/configs/training_config.json --checkpoint output/checkpoints/run1
    python auto_train.py --epochs 5 --prompt "once upon a" --trace-logits
    python auto_train.py --steps 1500 --prompt "once upon a"
    python auto_train.py --learning-rate 0.005 --steps 500 --no-prompt   # fully non-interactive
    python auto_train.py --menu        # wizard: Toy Run or Tiny Stories presets, or custom
    python auto_train.py --resume --checkpoint output/checkpoints/run1/quarter_50 --steps 500

If --learning-rate/--steps/--epochs aren't all given on the command line, you'll
be prompted for whichever ones are missing (pass --no-prompt to disable and
silently fall back to config/CLI defaults).
"""

import argparse

import cli_common
from logging_config import logger, setup_logging
from paths import DATA_DIR, OUTPUT_CHECKPOINTS, ensure_output_dirs
from generate import generate as run_generate
from train import train as run_train
from training.probe import (
    DEFAULT_GENERATE_PROBE_TEMPERATURE,
    DEFAULT_GENERATE_PROBE_TOP_K,
    DEFAULT_GENERATE_PROBE_TOP_P,
)
from version import __version__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train then generate a smoke sample, end to end")
    cli_common.add_config_arg(parser)
    cli_common.add_checkpoint_arg(parser)
    cli_common.add_seed_arg(parser)
    cli_common.add_training_length_args(parser)
    cli_common.add_model_hyperparam_args(parser)
    cli_common.add_probe_args(parser)
    cli_common.add_quality_args(parser)
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from --checkpoint (latest, best/, or quarter_*)",
    )
    parser.add_argument("--prompt", type=str, default="the", help="Prompt for the post-training smoke sample")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Characters to generate for the smoke sample")
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_GENERATE_PROBE_TEMPERATURE,
        help=f"Sampling temperature (default: {DEFAULT_GENERATE_PROBE_TEMPERATURE})",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_GENERATE_PROBE_TOP_K,
        help=f"Top-K filter (default: {DEFAULT_GENERATE_PROBE_TOP_K})",
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_GENERATE_PROBE_TOP_P,
        help=f"Nucleus (top-p) filter (default: {DEFAULT_GENERATE_PROBE_TOP_P})",
    )
    parser.add_argument(
        "--menu", action="store_true",
        help="Run the interactive training setup wizard before training. First prompt lets "
             "you resume from an existing checkpoint instead of starting fresh.",
    )
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR), help="Directory auto-scanned for .txt datasets when --menu is used")
    parser.add_argument("--models-dir", type=str, default=str(OUTPUT_CHECKPOINTS), help="Directory scanned for existing checkpoints by --menu (default: output/checkpoints)")
    cli_common.add_trace_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    setup_logging(log_filename="auto_train")
    logger.info("auto_train.py starting | version=%s | checkpoint=%s", __version__, args.checkpoint)
    print(f"llm-gpu-8 auto_train v{__version__}")

    print("### STAGE 1/2: TRAINING ###")
    checkpoint_dir = run_train(args)

    print("\n### STAGE 2/2: SMOKE-TEST GENERATION ###")
    # Reuse the same trace flags for both stages; only the sampling params differ.
    args.checkpoint = checkpoint_dir
    run_generate(args)


if __name__ == "__main__":
    main()
