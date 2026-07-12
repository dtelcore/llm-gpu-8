"""
train.py

Main training entry point. Loads setup/training_config.json, builds the
tokenizer + PyCUDA-backed GPT model + AdamW optimizer, and runs the
training loop over the corpus with CLI-gated logit/token/neuron tracing.

Usage:
    python train.py --config setup/training_config.json --checkpoint models/run1
    python train.py --epochs 3 --trace-tokens --trace-logits --trace-every 10
    python train.py --steps 1500   # total steps across ALL epochs, overrides --epochs/config
    python train.py --menu         # wizard first; first prompt offers resuming a checkpoint instead
    python train.py --generate     # skip training; pick a checkpoint and enter a generation test menu

If --learning-rate/--steps/--epochs/model-hyperparameters (--embedding-dim,
--num-heads, --num-layers, --max-len, --dropout, --batch-size, --weight-decay,
--warmup-steps, --gradient-clip) aren't all given on the command line, you'll be
prompted for whichever ones are missing (pass --no-prompt to disable and
silently fall back to config/CLI defaults).
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
    cli_common.add_model_hyperparam_args(parser)
    cli_common.add_trace_args(parser)
    parser.add_argument(
        "--menu", action="store_true",
        help="Run the interactive training setup wizard (model/dataset/init/hyperparams) "
             "before training, instead of loading --config from disk",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Directory auto-scanned for .txt datasets when --menu is used (default: data)",
    )
    parser.add_argument(
        "--models-dir", type=str, default="models",
        help="Directory scanned for existing checkpoints by --menu/--generate (default: models)",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Skip training entirely: pick a checkpoint from --models-dir and drop into an "
             "interactive generation test menu (same REPL as interactive.py)",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="After training, render loss/metrics + loss-landscape charts from logs/training.log "
             "and save them under logs/ (see training_log_plotter.py, loss_landscape_plotter.py)",
    )
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


def generate_test_menu(args: argparse.Namespace) -> None:
    """Interactive checkpoint picker + generation REPL, without touching training at all."""
    import interactive as interactive_cli

    print("=" * 70)
    print("GENERATION TEST MENU")
    print("=" * 70)
    checkpoint = cli_common.select_checkpoint_interactive(
        models_dir=args.models_dir, allow_new=False, prompt_label="checkpoint to generate from",
    )
    args.checkpoint = checkpoint
    if not hasattr(args, "temperature"):
        args.temperature = 0.8
    if not hasattr(args, "max_new_tokens"):
        args.max_new_tokens = 80
    interactive_cli.run_repl(args)


def train(args: argparse.Namespace) -> str:
    # menu/models_dir/plot are train.py-only flags; default them so this function
    # also works when called from auto_train.py's smaller argument set.
    resumed = False
    start_step = 0
    tokenizer = gpt_config = params = None

    if getattr(args, "menu", False):
        from training.checkpoint import load_checkpoint

        models_dir = getattr(args, "models_dir", "models")
        print("\n[Step 0/5] RESUME OR NEW")
        print("-" * 70)
        resume_ckpt = cli_common.prompt_resume_or_new(models_dir)

        if resume_ckpt:
            gpt_config, params, tokenizer, config, state = load_checkpoint(resume_ckpt)
            args.checkpoint = resume_ckpt
            start_step = int(state.get("step", 0))
            resumed = True
            print(f"-> Resuming '{resume_ckpt}' from step {start_step:,} "
                  f"(model architecture is fixed by the checkpoint; only training-length/LR "
                  f"prompts still apply)")
        else:
            from setup.training_setup import quickstart_training_setup
            config = quickstart_training_setup(interactive=True, data_dir=getattr(args, "data_dir", "data"))

            print("\n[Step 5/5] CHECKPOINT DESTINATION")
            print("-" * 70)
            args.checkpoint = cli_common.select_checkpoint_interactive(
                models_dir=models_dir, allow_new=True,
                default_new_name="run1", prompt_label="checkpoint to save to",
            )
            print(f"-> Training will checkpoint to '{args.checkpoint}'")
    else:
        config = cli_common.load_config(args.config)

    hyperparams = config["hyperparameters"]

    if not resumed:
        # Architecture is fixed once a checkpoint exists, so these prompts only
        # apply to fresh runs.
        cli_common.prompt_model_hyperparams(args, config["model"], hyperparams)
        tokenizer, gpt_config = build_tokenizer_and_config(config)
        params = ModelParameters(gpt_config, init_scales=config.get("weight_initialization", {}), seed=args.seed)

    model = GPTModel(gpt_config, params)

    cli_common.prompt_training_length_and_lr(args, hyperparams)
    if args.learning_rate is not None:
        hyperparams["learning_rate"] = args.learning_rate

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
    # When resuming, --steps/--epochs are interpreted as "how many MORE steps",
    # added on top of the checkpoint's saved step.
    if args.steps is not None:
        total_steps = start_step + args.steps
    else:
        epochs = args.epochs if args.epochs is not None else hyperparams["num_epochs"]
        total_steps = start_step + epochs * steps_per_epoch

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
    logger.info(f"Training started: {gpt_config} total_steps={total_steps} start_step={start_step}")

    global_step = start_step
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
        # Push updated weights to their persistent GPU mirrors once per step
        # (not once per layer op -- see ModelParameters.sync_device).
        params.sync_device(names=batch_grads.keys())

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
            # Tagged + keyed for training_log_plotter.py / loss_landscape_plotter.py to parse.
            logger.info(
                f"[train] step={global_step}/{total_steps} epoch={epoch} loss={avg_recent_loss:.4f} "
                f"step_ms={avg_step_ms:.1f} tok_s={tokens_per_sec:.0f} lr={optimizer.current_lr():.6g} "
                f"device_used_mb={used_mb:.0f} vram_free_mb={free_mb:.0f} "
                f"elapsed_s={elapsed:.2f} eta_s={eta_seconds:.2f}"
            )
            window_loss_sum = 0.0
            window_steps = 0
            window_start_time = now

        if global_step % checkpoint_every == 0 or global_step == total_steps:
            ckpt_dir = save_checkpoint(args.checkpoint, params, tokenizer, config, step=global_step, epoch=epoch)
            run_probe(str(ckpt_dir))

    print(f"\nTraining complete ({global_step:,} steps). Final checkpoint: {args.checkpoint}")
    print(f"\nTest generation with this checkpoint:")
    print(f"  python generate.py --checkpoint {args.checkpoint} --prompt \"once upon a\"")
    print(f"  python train.py --generate --models-dir {Path(args.checkpoint).parent}")

    if getattr(args, "plot", False):
        _render_post_training_plots()

    return args.checkpoint


def _render_post_training_plots() -> None:
    """Best-effort: render training_log_plotter + loss_landscape_plotter charts
    from logs/training.log and save them under logs/. Never raises on failure."""
    try:
        import training_log_plotter as tlp

        log_path = Path("logs/training.log")
        runs = tlp._load_runs([log_path])
        if runs:
            tlp.plot_runs_liveable(
                runs=runs, metric_name="tok/s", smooth_window=21, ema_alpha=0.08,
                raw_alpha=0.10, forecast_window=40, forecast_enabled=True,
                forecast_use_smoothed=True, show_raw_loss=False, show_ema_loss=False,
                show_raw_metric=True, live=False, refresh_seconds=1.0,
                source_paths=[log_path], save_path=Path("logs/training_plot_latest.png"),
            )
    except Exception as exc:
        logger.warning(f"training_log_plotter failed: {exc}")

    try:
        import loss_landscape_plotter as llp

        runs = llp.read_runs(log_dir=".", all_runs=False)
        llp.render_landscape(runs, out_path=Path("logs/loss_landscape_latest.png"), show=False)
    except Exception as exc:
        logger.warning(f"loss_landscape_plotter failed: {exc}")


def main() -> None:
    args = parse_args()
    if args.generate:
        generate_test_menu(args)
        return
    train(args)


if __name__ == "__main__":
    main()
