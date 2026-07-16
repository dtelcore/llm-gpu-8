"""
generate.py

Load a trained checkpoint and sample text from it, with optional
CLI-gated token/logit/neuron/vectorization tracing.

Usage:
    python generate.py --checkpoint output/checkpoints/run1 --prompt "once upon a" --max-new-tokens 100
    python generate.py --checkpoint output/checkpoints/run1 --prompt "the" --trace-tokens --trace-logits --trace-every 1
"""

import argparse

import numpy as np

import cli_common
from logging_config import logger, setup_logging
from model.gpt import GPTModel
from paths import ensure_output_dirs
from training.checkpoint import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample text from a trained checkpoint")
    cli_common.add_checkpoint_arg(parser)
    cli_common.add_seed_arg(parser)
    parser.add_argument("--prompt", type=str, default="the", help="Seed text to continue")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Number of characters to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    cli_common.add_trace_args(parser)
    return parser.parse_args()


def generate(args: argparse.Namespace) -> str:
    ensure_output_dirs()
    setup_logging(log_filename="generate")
    logger.info("generate.py | checkpoint=%s | prompt=%r", args.checkpoint, args.prompt)

    gpt_config, params, tokenizer, _, _ = load_checkpoint(args.checkpoint)
    model = GPTModel(gpt_config, params)
    tracer = cli_common.build_tracer(args, default_trace_every=1)
    rng = np.random.default_rng(args.seed)

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError(f"Prompt {args.prompt!r} encodes to zero known characters for this vocab")

    if tracer.any_enabled:
        tokenizer_note = f"vocab_size={tokenizer.vocab_size}"
        print(f"[Generate] Loaded checkpoint '{args.checkpoint}' ({tokenizer_note})")
        tracer.dump_tokens(prompt_ids, tokenizer, label="prompt")

    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        tracer=tracer,
        tokenizer=tokenizer if tracer.any_enabled else None,
        rng=rng,
    )

    text = tokenizer.decode(generated_ids)
    print("\n" + "=" * 70)
    print("GENERATED TEXT")
    print("=" * 70)
    print(text)
    print("=" * 70)
    return text


def main() -> None:
    args = parse_args()
    generate(args)


if __name__ == "__main__":
    main()
