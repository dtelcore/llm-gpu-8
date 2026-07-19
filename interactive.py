"""
interactive.py

Interactive REPL for sampling from a trained checkpoint. Each prompt you
type continues from the model; trace flags apply to every generation
you run in the session.

Usage:
    python interactive.py --checkpoint output/checkpoints/run1
    python interactive.py --checkpoint output/checkpoints/run1 --trace-tokens --trace-logits --trace-every 1

Session commands:
    :temp <value>       set sampling temperature
    :tokens <n>         set max new tokens per turn
    :trace on|off       toggle all tracing for subsequent turns
    :quit / :exit       leave the REPL
"""

import argparse

import numpy as np

import cli_common
from logging_config import logger, setup_logging
from model.gpt import GPTModel
from paths import ensure_output_dirs
from training.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive REPL for a trained checkpoint")
    cli_common.add_checkpoint_arg(parser)
    cli_common.add_seed_arg(parser)
    parser.add_argument("--temperature", type=float, default=0.8, help="Initial sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Initial characters generated per turn")
    parser.add_argument("--top-k", type=int, default=None, help="Only sample from top K characters")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling threshold (e.g. 0.9)")
    cli_common.add_trace_args(parser)
    return parser.parse_args()


def run_repl(args: argparse.Namespace) -> None:
    """Runs the interactive generation REPL for the checkpoint in `args.checkpoint`.
    Reusable by other CLIs (e.g. train.py --generate) that build their own args."""
    ensure_output_dirs()
    setup_logging(log_filename="interactive")
    logger.info("interactive.py | checkpoint=%s", args.checkpoint)

    gpt_config, params, tokenizer, _, _ = load_checkpoint(args.checkpoint)
    model = GPTModel(gpt_config, params)
    tracer = cli_common.build_tracer(args, default_trace_every=1)
    rng = np.random.default_rng(args.seed)

    temperature = args.temperature
    max_new_tokens = args.max_new_tokens
    top_k = getattr(args, "top_k", None)
    top_p = getattr(args, "top_p", None)
    trace_enabled = tracer.any_enabled

    print("=" * 70)
    print(f"INTERACTIVE GENERATION -- checkpoint: {args.checkpoint}")
    print(f"Model: {gpt_config.name} | vocab={gpt_config.vocab_size} | max_len={gpt_config.max_len}")
    print("Type a prompt and press Enter. Commands: :temp N  :tokens N  :topk N  :topp N  :trace on|off  :quit")
    print("=" * 70)

    while True:
        try:
            prompt = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue
        if prompt in (":quit", ":exit"):
            break
        if prompt.startswith(":temp "):
            temperature = float(prompt.split(maxsplit=1)[1])
            print(f"[temperature -> {temperature}]")
            continue
        if prompt.startswith(":tokens "):
            max_new_tokens = int(prompt.split(maxsplit=1)[1])
            print(f"[max_new_tokens -> {max_new_tokens}]")
            continue
        if prompt.startswith(":topk "):
            raw = prompt.split(maxsplit=1)[1].strip().lower()
            top_k = None if raw in ("none", "off", "0") else int(raw)
            print(f"[top_k -> {top_k}]")
            continue
        if prompt.startswith(":topp "):
            raw = prompt.split(maxsplit=1)[1].strip().lower()
            top_p = None if raw in ("none", "off", "1", "1.0") else float(raw)
            print(f"[top_p -> {top_p}]")
            continue
        if prompt.startswith(":trace "):
            trace_enabled = prompt.split(maxsplit=1)[1].strip().lower() == "on"
            print(f"[tracing -> {'on' if trace_enabled else 'off'}]")
            continue

        prompt_ids = tokenizer.encode(prompt)
        if not prompt_ids:
            print("[No recognized characters in prompt for this vocabulary; try different text]")
            continue

        active_tracer = tracer if trace_enabled else None
        if active_tracer is not None:
            active_tracer.dump_tokens(prompt_ids, tokenizer, label="prompt")

        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            tracer=active_tracer,
            tokenizer=tokenizer if active_tracer is not None else None,
            rng=rng,
        )
        print(tokenizer.decode(generated_ids))


def main() -> None:
    args = parse_args()
    run_repl(args)


if __name__ == "__main__":
    main()
