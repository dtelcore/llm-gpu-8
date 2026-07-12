"""
train.py

Main training entry point. Loads setup/training_config.json, builds the
tokenizer + PyCUDA-backed GPT model + AdamW optimizer, and runs the
training loop over the corpus with CLI-gated logit/token/neuron tracing.

Usage:
    python train.py --config setup/training_config.json --checkpoint models/run1
    python train.py --epochs 3 --trace-tokens --trace-logits --trace-every 10
    python train.py --steps 1500   # total steps across ALL epochs, overrides --epochs/config
"""

import argparse
import time
from pathlib import Path

import numpy as np

import cli_common
from logging_config import logger
from model.config import GPTConfig
from model.cuda import ops as cuda_ops
from model.gpt import GPTModel
from model.weights import ModelParameters
from tokenizer.tokenizer import CharacterGPTTokenizer
from training.checkpoint import save_checkpoint
from training.dataset import WindowedDataset
from training.loss import softmax_cross_entropy, trace_predictions
from training.optimizer import AdamW
from training.probe import run_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NumPy + PyCUDA GPT training loop")
    cli_common.add_config_arg(parser)
    cli_common.add_checkpoint_arg(parser)
    cli_common.add_seed_arg(parser)
    cli_common.add_training_length_args(parser)
    cli_common.add_trace_args(parser)
    return parser.parse_args()


def build_tokenizer_and_config(config: dict) -> tuple:
    corpus = config["dataset"]["corpus"]
    tokenizer = CharacterGPTTokenizer.from_corpus(corpus)

    configured_vocab = config["model"].get("vocab_size")
    if configured_vocab != tokenizer.vocab_size:
        logger.warning(
            f"Config vocab_size ({configured_vocab}) != tokenizer vocab_size "
            f"({tokenizer.vocab_size}); overriding config to match tokenizer."
        )
        config["model"]["vocab_size"] = tokenizer.vocab_size

    gpt_config = GPTConfig(config["model"])
    return tokenizer, gpt_config


def _fmt_duration(seconds: float) -> str:
    """Format seconds as H:MM:SS (or MM:SS if under an hour)."""
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _iter_batches_forever(dataset: WindowedDataset, rng: np.random.Generator):
    """Cycles through the dataset indefinitely, reshuffling on each pass.
    Yields (batch, epoch_number) so callers can track pass count for logging/checkpointing.
    """
    epoch = 0
    while True:
        epoch += 1
        for batch in dataset.iter_batches(shuffle=True, rng=rng):
            yield batch, epoch


def train(args: argparse.Namespace) -> str:
    config = cli_common.load_config(args.config)

    tokenizer, gpt_config = build_tokenizer_and_config(config)

    params = ModelParameters(gpt_config, init_scales=config.get("weight_initialization", {}), seed=args.seed)
    model = GPTModel(gpt_config, params)

    hyperparams = config["hyperparameters"]
    optimizer = AdamW(
        params.all_params(),
        learning_rate=hyperparams["learning_rate"],
        weight_decay=hyperparams.get("weight_decay", 0.01),
        beta1=hyperparams.get("beta1", 0.9),
        beta2=hyperparams.get("beta2", 0.999),
        epsilon=hyperparams.get("epsilon", 1e-8),
        warmup_steps=hyperparams.get("warmup_steps", 0),
        gradient_clip=hyperparams.get("gradient_clip", 1.0),
    )

    dataset = WindowedDataset(config["dataset"]["corpus"], tokenizer, gpt_config.max_len, hyperparams["batch_size"])
    rng = np.random.default_rng(args.seed)
    steps_per_epoch = dataset.num_batches()

    # --steps is the TOTAL step count across all epochs and takes priority over
    # --epochs and the config's num_epochs (which only matter when --steps is absent).
    if args.steps is not None:
        total_steps = args.steps
    else:
        epochs = args.epochs if args.epochs is not None else hyperparams["num_epochs"]
        total_steps = epochs * steps_per_epoch

    checkpoint_every = args.checkpoint_every or steps_per_epoch
    log_every = max(1, args.log_every)

    # Logits (and other traces) fire every 10% of total_steps by default;
    # pass --trace-every explicitly to override that cadence.
    tracer = cli_common.build_tracer(args, default_trace_every=max(1, total_steps // 10))

    print("=" * 70)
    print(f"TRAINING: {gpt_config.name} | vocab={gpt_config.vocab_size} | "
          f"params={params.param_count():,} | windows={dataset.num_windows()} | "
          f"batches/epoch={steps_per_epoch} | total_steps={total_steps} | "
          f"checkpoint_every={checkpoint_every} steps | trace_every={tracer.trace_every} steps")
    print("=" * 70)
    logger.info(f"Training started: {gpt_config} total_steps={total_steps}")

    global_step = 0
    window_loss_sum = 0.0
    window_steps = 0
    window_start_time = time.time()
    train_start_time = window_start_time

    for batch, epoch in _iter_batches_forever(dataset, rng):
        if global_step >= total_steps:
            break

        step_start_time = time.time()
        tracer.update_step(global_step)
        batch_grads = None
        batch_loss = 0.0

        for x, y in batch:
            tracer.dump_tokens(x, tokenizer, label=f"step {global_step} input")

            logits, cache = model.forward(x, tracer=tracer)
            loss, dlogits = softmax_cross_entropy(logits, y)
            trace_predictions(logits, y, x, tokenizer, tracer, label=f"step {global_step} last-position")

            grads = model.backward(cache, dlogits)
            batch_loss += loss

            if batch_grads is None:
                batch_grads = {k: v.copy() for k, v in grads.items()}
            else:
                for k, v in grads.items():
                    batch_grads[k] += v

        batch_size_actual = len(batch)
        for k in batch_grads:
            batch_grads[k] /= batch_size_actual
        batch_loss /= batch_size_actual

        global_norm = optimizer.clip_grads_(batch_grads)
        optimizer.step(batch_grads)

        step_time_ms = (time.time() - step_start_time) * 1000.0

        global_step += 1
        window_loss_sum += batch_loss
        window_steps += 1

        if tracer.active_step:
            logger.debug(f"step={global_step} loss={batch_loss:.4f} grad_norm={global_norm:.4f} lr={optimizer.current_lr():.6g}")

        if global_step % log_every == 0 or global_step == total_steps:
            now = time.time()
            window_elapsed = now - window_start_time
            elapsed = now - train_start_time
            avg_recent_loss = window_loss_sum / max(1, window_steps)
            avg_step_ms = (window_elapsed / max(1, window_steps)) * 1000.0
            tokens_per_sec = (window_steps * hyperparams["batch_size"] * gpt_config.max_len) / max(window_elapsed, 1e-6)

            remaining_steps = max(0, total_steps - global_step)
            eta_seconds = (avg_step_ms / 1000.0) * remaining_steps

            free_bytes, total_bytes = cuda_ops.get_memory_info()
            used_mb = (total_bytes - free_bytes) / (1024 ** 2)
            free_mb = free_bytes / (1024 ** 2)

            print(
                f"Step {global_step:>7,}/{total_steps:,} | epoch={epoch} | avg_loss={avg_recent_loss:.4f} | "
                f"lr={optimizer.current_lr():.6g} | step={step_time_ms:.1f}ms | avg_step={avg_step_ms:.1f}ms | "
                f"{tokens_per_sec:.0f} tok/s | vram_used={used_mb:.0f}MB | vram_free={free_mb:.0f}MB | "
                f"elapsed={_fmt_duration(elapsed)} | eta={_fmt_duration(eta_seconds)}"
            )
            logger.info(
                f"step={global_step}/{total_steps} epoch={epoch} avg_loss={avg_recent_loss:.4f} "
                f"avg_step_ms={avg_step_ms:.1f} vram_used_mb={used_mb:.0f} vram_free_mb={free_mb:.0f} "
                f"elapsed_s={elapsed:.2f} eta_s={eta_seconds:.2f}"
            )
            window_loss_sum = 0.0
            window_steps = 0
            window_start_time = now

        if global_step % checkpoint_every == 0 or global_step == total_steps:
            ckpt_dir = save_checkpoint(args.checkpoint, params, tokenizer, config, step=global_step, epoch=epoch)
            run_probe(str(ckpt_dir))

    print(f"\nTraining complete ({global_step:,} steps). Final checkpoint: {args.checkpoint}")
    return args.checkpoint


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
