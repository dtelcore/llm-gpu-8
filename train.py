"""
train.py

Main training entry point. Loads setup/training_config.json, builds the
tokenizer + PyCUDA-backed GPT model + AdamW optimizer, and runs the
training loop over the corpus with CLI-gated logit/token/neuron tracing.

Usage:
    python train.py --config setup/training_config.json --checkpoint models/run1
    python train.py --epochs 3 --trace-tokens --trace-logits --trace-every 10
"""

import argparse
import time
from pathlib import Path

import numpy as np

import cli_common
from logging_config import logger
from model.config import GPTConfig
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
    parser.add_argument("--epochs", type=int, default=None, help="Override config num_epochs")
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


def train(args: argparse.Namespace) -> str:
    config = cli_common.load_config(args.config)
    if args.epochs is not None:
        config["hyperparameters"]["num_epochs"] = args.epochs

    tracer = cli_common.build_tracer(args)
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

    print("=" * 70)
    print(f"TRAINING: {gpt_config.name} | vocab={gpt_config.vocab_size} | "
          f"params={params.param_count():,} | windows={dataset.num_windows()} | "
          f"batches/epoch={dataset.num_batches()}")
    print("=" * 70)
    logger.info(f"Training started: {gpt_config}")

    epochs = hyperparams["num_epochs"]
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        n_steps = 0
        start_time = time.time()

        for batch in dataset.iter_batches(shuffle=True, rng=rng):
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

            epoch_loss += batch_loss
            n_steps += 1
            global_step += 1

            if tracer.active_step:
                logger.debug(f"step={global_step} loss={batch_loss:.4f} grad_norm={global_norm:.4f} lr={optimizer.current_lr():.6g}")

        avg_loss = epoch_loss / max(1, n_steps)
        elapsed = time.time() - start_time
        tokens_per_sec = (n_steps * hyperparams["batch_size"] * gpt_config.max_len) / max(elapsed, 1e-6)
        print(f"Epoch {epoch + 1:03d}/{epochs:03d} | avg_loss={avg_loss:.4f} | "
              f"lr={optimizer.current_lr():.6g} | {elapsed:.2f}s | {tokens_per_sec:.0f} tok/s")
        logger.info(f"Epoch {epoch + 1}/{epochs} avg_loss={avg_loss:.4f} elapsed={elapsed:.2f}s")

        ckpt_dir = save_checkpoint(args.checkpoint, params, tokenizer, config, step=global_step, epoch=epoch + 1)
        run_probe(str(ckpt_dir))

    print(f"\nTraining complete. Final checkpoint: {args.checkpoint}")
    return args.checkpoint


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
