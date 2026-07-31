"""
generate.py

Load a trained checkpoint and sample text from it, with optional
CLI-gated token/logit/neuron/vectorization tracing.

Usage:
    python generate.py --checkpoint output/checkpoints/run1 --prompt "once upon a" --max-new-tokens 100
    python generate.py --menu
    python generate.py --checkpoint output/checkpoints/run1 --prompt "the" --trace-tokens --trace-logits --trace-every 1
"""

import argparse
from pathlib import Path

import numpy as np

import cli_common
from logging_config import logger, setup_logging
from model.gpt import GPTModel
from paths import DEFAULT_CHECKPOINT_DIR, OUTPUT_CHECKPOINTS, ensure_output_dirs
from training.checkpoint import load_checkpoint


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample text from a trained checkpoint")
    cli_common.add_checkpoint_arg(parser)
    cli_common.add_seed_arg(parser)
    parser.add_argument("--prompt", type=str, default="the", help="Seed text to continue")
    parser.add_argument("--max-new-tokens", type=int, default=80, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-k", type=int, default=None, help="Only sample from top K tokens (e.g. 15)")
    parser.add_argument("--top-p", type=float, default=None, help="Nucleus sampling threshold (e.g. 0.9)")
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable Stage 3.2 KV cache (full recompute each token)",
    )
    parser.add_argument(
        "--cuda-graph",
        action="store_true",
        help="Stage 4: capture KV decode kernel-chain CUDA Graph (full decode stays eager GPU)",
    )
    parser.add_argument(
        "--menu", action="store_true",
        help="Pick a checkpoint then configure Sampling / Decode / Trace flag groups "
             "(Enter keeps defaults per group)",
    )
    parser.add_argument(
        "--models-dir", type=str, default=str(OUTPUT_CHECKPOINTS),
        help="Directory scanned for checkpoints when --menu is used (default: output/checkpoints)",
    )
    parser.add_argument(
        "--no-prompt", action="store_true",
        help="With --menu, skip interactive flag prompts and use defaults",
    )
    cli_common.add_trace_args(parser)
    return parser.parse_args(argv)


def _run_generate_menu(args: argparse.Namespace) -> None:
    """Interactive checkpoint pick + flag-group configuration for one-shot generate."""
    default_ckpt = str(DEFAULT_CHECKPOINT_DIR)
    # Always offer a picker in menu mode unless user already pointed at a real bundle
    # other than the argparse default placeholder.
    ckpt_has_config = bool(args.checkpoint) and (Path(args.checkpoint) / "config.json").exists()
    if not ckpt_has_config or args.checkpoint == default_ckpt:
        args.checkpoint = cli_common.select_checkpoint_interactive(
            models_dir=getattr(args, "models_dir", None),
            allow_new=False,
            prompt_label="checkpoint to generate from",
        )
    cli_common.prompt_run_flag_menu(args, fresh_run=False, entry="generate")


def generate(args: argparse.Namespace) -> str:
    ensure_output_dirs()
    setup_logging(log_filename="generate")
    logger.info(
        "generate.py | checkpoint=%s | prompt=%r | temp=%s | top_k=%s | top_p=%s",
        args.checkpoint, args.prompt, args.temperature, args.top_k, args.top_p,
    )

    gpt_config, params, tokenizer, _, _ = load_checkpoint(args.checkpoint)
    model = GPTModel(gpt_config, params)
    tracer = cli_common.build_tracer(args, default_trace_every=1)
    rng = np.random.default_rng(args.seed)

    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError(f"Prompt {args.prompt!r} encodes to zero known tokens for this vocab")

    if tracer.any_enabled:
        tokenizer_note = f"vocab_size={tokenizer.vocab_size}"
        print(f"[Generate] Loaded checkpoint '{args.checkpoint}' ({tokenizer_note})")
        tracer.dump_tokens(prompt_ids, tokenizer, label="prompt")

    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        tracer=tracer,
        tokenizer=tokenizer if tracer.any_enabled else None,
        rng=rng,
        use_kv_cache=not args.no_kv_cache,
        use_cuda_graph=bool(args.cuda_graph),
    )

    if args.cuda_graph and getattr(model, "_cuda_graph_status", None):
        logger.info("cuda_graph_status=%s", model._cuda_graph_status)

    text = tokenizer.decode(generated_ids)
    print("\n" + "=" * 70)
    print("GENERATED TEXT")
    print("=" * 70)
    print(text)
    print("=" * 70)
    return text


def main(argv=None) -> None:
    args = parse_args(argv)
    if getattr(args, "menu", False):
        _run_generate_menu(args)
    generate(args)


if __name__ == "__main__":
    main()
