"""
train.py

Main training entry point. Loads output/configs/training_config.json, builds the
tokenizer + PyCUDA-backed GPT model + AdamW optimizer, and runs the
training loop over the corpus with CLI-gated logit/token/neuron tracing.

v0.1.0: quarterly milestones save latest + quarter_XX/, force full traces,
val holdout metrics, and optional sequential generation-quality trial for
manual best/ promotion. Resume from latest, best/, or any quarter_*.

Usage:
    python train.py --config output/configs/training_config.json --checkpoint output/checkpoints/run1
    python train.py --epochs 3 --trace-tokens --trace-logits --trace-every 10
    python train.py --steps 1500   # total steps across ALL epochs, overrides --epochs/config
    python train.py --menu         # wizard: Toy Run or Tiny Stories presets, or custom
    python train.py --generate     # skip training; pick a checkpoint and enter a generation test menu
    python train.py --compare-quarters --checkpoint output/checkpoints/run1
    python train.py --set-best quarter_50 --checkpoint output/checkpoints/run1

If --learning-rate/--steps/--epochs/model-hyperparameters (--embedding-dim,
--num-heads, --num-layers, --max-len, --dropout, --batch-size, --weight-decay,
--warmup-steps, --gradient-clip) aren't all given on the command line, you'll be
prompted for whichever ones are missing (pass --no-prompt to disable and
silently fall back to config/CLI defaults).
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

import cli_common
from logging_config import logger, setup_logging
from model.config import GPTConfig
from model.cuda import ops as cuda_ops
from model.gpt import GPTModel
from model.trace import TraceContext
from model.weights import ModelParameters
from paths import (
    DATA_DIR,
    DEFAULT_LANDSCAPE_PLOT,
    DEFAULT_TRAINING_LOG,
    DEFAULT_TRAINING_PLOT,
    OUTPUT_CHECKPOINTS,
    ensure_output_dirs,
    resolve_checkpoints_dir,
    run_root_for_checkpoint,
)
from tokenizer.tokenizer import CharacterGPTTokenizer
from training.checkpoint import promote_best, save_checkpoint
from training.dataset import WindowedDataset
from training.eval import (
    ensure_train_val_split,
    evaluate_val_loss,
    perplexity_from_loss,
)
from training.loss import softmax_cross_entropy_batch, softmax_cross_entropy_batch_gpu, trace_predictions
from training.gpu_optimizer import AdamWGPU
from training.probe import (
    DEFAULT_GENERATE_PROBE_TEMPERATURE,
    DEFAULT_GENERATE_PROBE_TOP_K,
    DEFAULT_GENERATE_PROBE_TOP_P,
    make_full_tracer,
    milestone_fraction_map,
    run_generate_probe,
    run_probe,
)
from training.quality import compare_quarters, parse_quality_weights, score_generation
from version import __version__


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NumPy + PyCUDA GPT training loop")
    cli_common.add_config_arg(parser)
    cli_common.add_checkpoint_arg(parser)
    cli_common.add_seed_arg(parser)
    cli_common.add_training_length_args(parser)
    cli_common.add_model_hyperparam_args(parser)
    cli_common.add_trace_args(parser)
    cli_common.add_probe_args(parser)
    cli_common.add_quality_args(parser)
    parser.add_argument(
        "--menu", action="store_true",
        help="Run the interactive training setup wizard (model/dataset/init/hyperparams) "
             "before training, instead of loading --config from disk",
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(DATA_DIR),
        help="Directory auto-scanned for .txt datasets when --menu is used (default: data)",
    )
    parser.add_argument(
        "--models-dir", type=str, default=str(OUTPUT_CHECKPOINTS),
        help="Directory scanned for existing checkpoints by --menu/--generate (default: output/checkpoints)",
    )
    parser.add_argument(
        "--generate", action="store_true",
        help="Skip training entirely: pick a checkpoint from --models-dir and drop into an "
             "interactive generation test menu (same REPL as interactive.py)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from --checkpoint (loads weights, config, vocab, and step counter)",
    )
    parser.add_argument(
        "--temperature", type=float, default=DEFAULT_GENERATE_PROBE_TEMPERATURE,
        help=f"Sampling temperature for generate probes / --generate (default: {DEFAULT_GENERATE_PROBE_TEMPERATURE})",
    )
    parser.add_argument(
        "--top-k", type=int, default=DEFAULT_GENERATE_PROBE_TOP_K,
        help=f"Top-K sampling for generate probes / --generate (default: {DEFAULT_GENERATE_PROBE_TOP_K})",
    )
    parser.add_argument(
        "--top-p", type=float, default=DEFAULT_GENERATE_PROBE_TOP_P,
        help=f"Nucleus (top-p) sampling for generate probes / --generate (default: {DEFAULT_GENERATE_PROBE_TOP_P})",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="After training, render loss/metrics + honest loss-trajectory charts from output/logs/training.log "
             "and save them under output/logs/ (see training_log_plotter.py, loss_landscape_plotter.py)",
    )
    return parser.parse_args()


def build_tokenizer_and_config(config: dict) -> tuple:
    from setup.config_loader import resolve_dataset_corpus

    corpus = resolve_dataset_corpus(config["dataset"])
    config["dataset"]["corpus"] = corpus
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


def _force_quarter_traces(tracer: TraceContext) -> Dict:
    """Enable all trace channels for this step; returns prior flag snapshot to restore."""
    snapshot = {
        "trace_logits": tracer.trace_logits,
        "trace_tokens": tracer.trace_tokens,
        "trace_neurons": tracer.trace_neurons,
        "trace_vectorization": tracer.trace_vectorization,
        "trace_every": tracer.trace_every,
        "active_step": tracer.active_step,
    }
    tracer.trace_logits = True
    tracer.trace_tokens = True
    tracer.trace_neurons = True
    tracer.trace_vectorization = True
    tracer.trace_every = 1
    tracer.active_step = True
    return snapshot


def _restore_traces(tracer: TraceContext, snapshot: Dict) -> None:
    tracer.trace_logits = snapshot["trace_logits"]
    tracer.trace_tokens = snapshot["trace_tokens"]
    tracer.trace_neurons = snapshot["trace_neurons"]
    tracer.trace_vectorization = snapshot["trace_vectorization"]
    tracer.trace_every = snapshot["trace_every"]
    tracer.active_step = snapshot["active_step"]


def generate_test_menu(args: argparse.Namespace) -> None:
    """Interactive checkpoint picker + generation REPL, without touching training at all."""
    import interactive as interactive_cli

    ensure_output_dirs()
    setup_logging(log_filename="generate_menu")
    logger.info("train.py --generate | models_dir=%s | version=%s", args.models_dir, __version__)

    print("=" * 70)
    print(f"GENERATION TEST MENU  (v{__version__})")
    print("=" * 70)
    checkpoint = cli_common.select_checkpoint_interactive(
        models_dir=args.models_dir, allow_new=False, prompt_label="checkpoint to generate from",
    )
    args.checkpoint = checkpoint
    if not hasattr(args, "temperature"):
        args.temperature = DEFAULT_GENERATE_PROBE_TEMPERATURE
    if not hasattr(args, "top_k"):
        args.top_k = DEFAULT_GENERATE_PROBE_TOP_K
    if not hasattr(args, "top_p"):
        args.top_p = DEFAULT_GENERATE_PROBE_TOP_P
    if not hasattr(args, "max_new_tokens"):
        args.max_new_tokens = 80
    interactive_cli.run_repl(args)


def run_quality_trial_for_args(args: argparse.Namespace, run_dir: Optional[str] = None) -> None:
    """Run sequential quarter comparison / optional best promotion from CLI args."""
    root = run_root_for_checkpoint(run_dir or args.checkpoint)
    prompt = getattr(args, "quality_prompt", None) or getattr(args, "generate_probe_prompt", "once upon a")
    weights = parse_quality_weights(getattr(args, "quality_weights", None))
    interactive = not getattr(args, "no_prompt", False) and not getattr(args, "set_best", None)
    compare_quarters(
        str(root),
        prompt=prompt,
        max_new_tokens=getattr(args, "generate_probe_tokens", 256),
        temperature=getattr(args, "temperature", DEFAULT_GENERATE_PROBE_TEMPERATURE),
        top_k=getattr(args, "top_k", DEFAULT_GENERATE_PROBE_TOP_K),
        top_p=getattr(args, "top_p", DEFAULT_GENERATE_PROBE_TOP_P),
        seed=args.seed,
        weights=weights,
        interactive_promote=interactive,
        set_best=getattr(args, "set_best", None),
    )


def _handle_quarterly_milestone(
    *,
    args: argparse.Namespace,
    model: GPTModel,
    params: ModelParameters,
    tokenizer,
    config: Dict,
    optimizer,
    run_dir: Path,
    global_step: int,
    total_steps: int,
    epoch: int,
    fraction: float,
    avg_recent_loss: Optional[float],
    val_dataset: Optional[WindowedDataset],
    do_generate_probe: bool = True,
) -> None:
    """Save latest + quarter_XX, val eval, probe, full-trace generate, quality score."""
    from paths import quarter_name_for_fraction

    quarter_name = quarter_name_for_fraction(fraction)
    quarter_path = run_dir / quarter_name

    print(f"\n[Quarterly] step={global_step:,}/{total_steps:,} ({fraction * 100:.0f}%) -> {quarter_name}")
    logger.info(
        "Quarterly milestone step=%s/%s fraction=%.2f quarter=%s",
        global_step, total_steps, fraction, quarter_name,
    )

    optimizer.sync_host_weights()

    val_loss, val_ppl = evaluate_val_loss(model, val_dataset, seed=args.seed)
    train_ppl = perplexity_from_loss(avg_recent_loss) if avg_recent_loss is not None else None

    metrics: Dict = {
        "step": global_step,
        "epoch": epoch,
        "fraction": fraction,
        "quarter": quarter_name,
        "loss": avg_recent_loss,
        "ppl": train_ppl,
        "val_loss": val_loss,
        "val_ppl": val_ppl,
    }

    save_checkpoint(str(run_dir), params, tokenizer, config, step=global_step, epoch=epoch, metrics=metrics)
    save_checkpoint(str(quarter_path), params, tokenizer, config, step=global_step, epoch=epoch, metrics=metrics)

    run_probe(str(quarter_path))

    generated = ""
    if do_generate_probe:
        full_tracer = make_full_tracer()
        generated = run_generate_probe(
            model,
            tokenizer,
            step=global_step,
            total_steps=total_steps,
            prompt=args.generate_probe_prompt,
            max_new_tokens=args.generate_probe_tokens,
            temperature=getattr(args, "temperature", DEFAULT_GENERATE_PROBE_TEMPERATURE),
            top_k=getattr(args, "top_k", DEFAULT_GENERATE_PROBE_TOP_K),
            top_p=getattr(args, "top_p", DEFAULT_GENERATE_PROBE_TOP_P),
            seed=args.seed,
            checkpoint_dir=str(quarter_path),
            tracer=full_tracer,
        )

    if generated:
        from training.checkpoint import save_metrics

        weights = parse_quality_weights(getattr(args, "quality_weights", None))
        scores = score_generation(generated, prompt=args.generate_probe_prompt, weights=weights)
        metrics["quality"] = scores.as_dict()
        save_metrics(run_dir, metrics)
        save_metrics(quarter_path, metrics)
        print(
            f"[Quality] {quarter_name} aggregate={scores.aggregate:.3f} "
            f"(spell={scores.spelling:.3f} punct={scores.punctuation:.3f} "
            f"gram={scores.grammar:.3f} sem={scores.semantics:.3f})"
        )
        logger.info(
            "[quality] step=%s quarter=%s aggregate=%.4f spelling=%.4f punctuation=%.4f "
            "grammar=%.4f semantics=%.4f",
            global_step, quarter_name, scores.aggregate, scores.spelling,
            scores.punctuation, scores.grammar, scores.semantics,
        )

    if val_loss is not None:
        logger.info(
            f"[train] step={global_step}/{total_steps} epoch={epoch} "
            f"loss={avg_recent_loss if avg_recent_loss is not None else float('nan'):.4f} "
            f"ppl={train_ppl if train_ppl is not None else float('nan'):.4f} "
            f"val_loss={val_loss:.4f} val_ppl={val_ppl:.4f}"
        )
        print(f"[Val] loss={val_loss:.4f} ppl={val_ppl:.4f}")


def train(args: argparse.Namespace) -> str:
    ensure_output_dirs()
    ckpt_stem = Path(args.checkpoint).name
    setup_logging(log_filename=f"training_{ckpt_stem}")
    logger.info(
        "train.py starting | version=%s | checkpoint=%s | config=%s",
        __version__, args.checkpoint, getattr(args, "config", None),
    )
    print(f"llm-gpu-8 training v{__version__}")

    # menu/models_dir/plot are train.py-only flags; default them so this function
    # also works when called from auto_train.py's smaller argument set.
    resumed = False
    start_step = 0
    tokenizer = gpt_config = params = None
    run_dir = run_root_for_checkpoint(args.checkpoint)

    if getattr(args, "resume", False):
        from training.checkpoint import load_checkpoint

        gpt_config, params, tokenizer, config, state = load_checkpoint(args.checkpoint)
        start_step = int(state.get("step", 0))
        resumed = True
        run_dir = run_root_for_checkpoint(args.checkpoint)
        # Continue writing latest/quarters into the parent run root.
        args.checkpoint = str(run_dir)
        print(f"-> Resuming '{args.checkpoint}' from step {start_step:,} "
              f"(source={state.get('version', '?')}; writing latest into run root)")
    elif getattr(args, "menu", False):
        from training.checkpoint import load_checkpoint

        models_dir = getattr(args, "models_dir", None) or str(OUTPUT_CHECKPOINTS)
        print("\n[Step 0/5] RESUME OR NEW")
        print("-" * 70)
        resume_ckpt = cli_common.prompt_resume_or_new(models_dir)

        if resume_ckpt:
            gpt_config, params, tokenizer, config, state = load_checkpoint(resume_ckpt)
            run_dir = run_root_for_checkpoint(resume_ckpt)
            args.checkpoint = str(run_dir)
            start_step = int(state.get("step", 0))
            resumed = True
            print(f"-> Resuming '{resume_ckpt}' from step {start_step:,} "
                  f"(writing latest into '{run_dir}'; "
                  f"model architecture is fixed by the checkpoint; only training-length/LR "
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
            run_dir = run_root_for_checkpoint(args.checkpoint)
            args.checkpoint = str(run_dir)
            print(f"-> Training will checkpoint to '{args.checkpoint}'")
    else:
        config = cli_common.load_config(args.config)
        run_dir = run_root_for_checkpoint(args.checkpoint)
        args.checkpoint = str(run_dir)

    hyperparams = config["hyperparameters"]

    if not resumed:
        # Architecture is fixed once a checkpoint exists, so these prompts only
        # apply to fresh runs.
        cli_common.prompt_model_hyperparams(args, config["model"], hyperparams)
        tokenizer, gpt_config = build_tokenizer_and_config(config)
        params = ModelParameters(gpt_config, init_scales=config.get("weight_initialization", {}), seed=args.seed)

    # 90/10 val holdout (stable across resume when val_corpus.json is present).
    train_corpus, val_corpus = ensure_train_val_split(config, seed=args.seed)

    model = GPTModel(gpt_config, params)

    cli_common.prompt_training_length_and_lr(args, hyperparams)
    if args.learning_rate is not None:
        hyperparams["learning_rate"] = args.learning_rate

    optimizer = AdamWGPU(
        params,
        learning_rate=hyperparams["learning_rate"],
        weight_decay=hyperparams.get("weight_decay", 0.01),
        beta1=hyperparams.get("beta1", 0.9),
        beta2=hyperparams.get("beta2", 0.999),
        epsilon=hyperparams.get("epsilon", 1e-8),
        warmup_steps=hyperparams.get("warmup_steps", 0),
        gradient_clip=hyperparams.get("gradient_clip", 1.0),
    )
    if resumed:
        optimizer.t = start_step

    dataset = WindowedDataset(train_corpus, tokenizer, gpt_config.max_len, hyperparams["batch_size"])
    val_dataset = None
    if val_corpus:
        try:
            val_dataset = WindowedDataset(val_corpus, tokenizer, gpt_config.max_len, hyperparams["batch_size"])
        except ValueError as exc:
            logger.warning("Val dataset too small for windows; skipping val eval: %s", exc)
            val_dataset = None

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

    # Default to every 1000 steps (capped by epoch length). Never fall back to
    # steps_per_epoch alone — on TinyStories that is ~26M and silently loses hours of work.
    if args.checkpoint_every is not None:
        checkpoint_every = max(1, int(args.checkpoint_every))
    else:
        checkpoint_every = min(1000, steps_per_epoch)
    log_every = max(1, args.log_every)

    # Logits (and other traces) fire every 10% of total_steps by default;
    # pass --trace-every explicitly to override that cadence.
    tracer = cli_common.build_tracer(args, default_trace_every=max(1, total_steps // 10))

    milestone_fracs = milestone_fraction_map(total_steps)
    quarterly_steps = set(milestone_fracs.keys())
    quarterly_done: set[int] = {s for s in quarterly_steps if s <= start_step}
    do_generate_probe = not getattr(args, "no_generate_probe", False)
    if quarterly_steps:
        pending = sorted(quarterly_steps - quarterly_done)
        print(f"Quarterly milestones at steps: {', '.join(f'{s:,}' for s in sorted(quarterly_steps))}")
        if quarterly_done:
            print(f"  (skipping already-passed milestones: {', '.join(f'{s:,}' for s in sorted(quarterly_done))})")
        if pending:
            print(f"  (pending: {', '.join(f'{s:,}' for s in pending)})")
        if not do_generate_probe:
            print("  (generate probes disabled via --no-generate-probe; still saving quarters + val/traces)")

    print("=" * 70)
    print(f"TRAINING v{__version__}: {gpt_config.name} | vocab={gpt_config.vocab_size} | "
          f"params={params.param_count():,} | windows={dataset.num_windows()} | "
          f"batches/epoch={steps_per_epoch} | total_steps={total_steps} | "
          f"checkpoint_every={checkpoint_every} steps | trace_every={tracer.trace_every} steps | "
          f"val_sentences={len(val_corpus)}")
    print("=" * 70)
    logger.info(
        f"Training started: {gpt_config} total_steps={total_steps} start_step={start_step} "
        f"version={__version__} run_dir={run_dir}"
    )

    global_step = start_step
    window_loss_sum = 0.0
    window_steps = 0
    window_start_time = time.time()
    train_start_time = window_start_time
    avg_recent_loss: Optional[float] = None

    for batch, epoch in _iter_batches_forever(dataset, rng):
        if global_step >= total_steps:
            break

        next_step = global_step + 1
        is_quarter = next_step in quarterly_steps and next_step not in quarterly_done
        trace_snapshot = None
        if is_quarter:
            # Force full traces on the training forward for this milestone step.
            trace_snapshot = _force_quarter_traces(tracer)
        else:
            tracer.update_step(global_step)

        step_start_time = time.time()

        xs = np.stack([x for x, _ in batch])
        ys = np.stack([y for _, y in batch])
        tracer.dump_tokens(xs[0], tokenizer, label=f"step {global_step} input")

        logits, cache = model.forward_batch(xs, tracer=tracer)
        if cache.get("gpu"):
            loss, dlogits_d = softmax_cross_entropy_batch_gpu(cache["logits_d"], ys)
            if tracer.trace_logits or tracer.trace_tokens:
                logits = cuda_ops.to_host(cache["logits_d"]).reshape(xs.shape[0], -1, model.config.vocab_size)
                trace_predictions(logits[0], ys[0], xs[0], tokenizer, tracer, label=f"step {global_step} last-position")
            batch_grads = model.backward_batch_gpu(cache, dlogits_d.reshape(-1, model.config.vocab_size))
        else:
            loss, dlogits = softmax_cross_entropy_batch(logits, ys)
            trace_predictions(logits[0], ys[0], xs[0], tokenizer, tracer, label=f"step {global_step} last-position")
            batch_grads = model.backward_batch(cache, dlogits)
        batch_loss = loss

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
            # Tagged + keyed for training_log_plotter.py / loss_landscape_plotter.py to parse.
            logger.info(
                f"[train] step={global_step}/{total_steps} epoch={epoch} loss={avg_recent_loss:.4f} "
                f"ppl={perplexity_from_loss(avg_recent_loss):.4f} "
                f"step_ms={avg_step_ms:.1f} tok_s={tokens_per_sec:.0f} lr={optimizer.current_lr():.6g} "
                f"device_used_mb={used_mb:.0f} vram_free_mb={free_mb:.0f} "
                f"elapsed_s={elapsed:.2f} eta_s={eta_seconds:.2f}"
            )
            window_loss_sum = 0.0
            window_steps = 0
            window_start_time = now

        if is_quarter:
            fraction = milestone_fracs[global_step]
            _handle_quarterly_milestone(
                args=args,
                model=model,
                params=params,
                tokenizer=tokenizer,
                config=config,
                optimizer=optimizer,
                run_dir=run_dir,
                global_step=global_step,
                total_steps=total_steps,
                epoch=epoch,
                fraction=fraction,
                avg_recent_loss=avg_recent_loss if avg_recent_loss is not None else float(batch_loss),
                val_dataset=val_dataset,
                do_generate_probe=do_generate_probe,
            )
            quarterly_done.add(global_step)
            if trace_snapshot is not None:
                _restore_traces(tracer, trace_snapshot)
        elif global_step % checkpoint_every == 0 or global_step == total_steps:
            optimizer.sync_host_weights()
            metrics = {
                "step": global_step,
                "epoch": epoch,
                "loss": avg_recent_loss if avg_recent_loss is not None else float(batch_loss),
                "ppl": perplexity_from_loss(avg_recent_loss if avg_recent_loss is not None else float(batch_loss)),
            }
            ckpt_dir = save_checkpoint(
                str(run_dir), params, tokenizer, config,
                step=global_step, epoch=epoch, metrics=metrics,
            )
            run_probe(str(ckpt_dir))

    print(f"\nTraining complete ({global_step:,} steps). Final checkpoint: {run_dir}")
    print(f"\nTest generation with this checkpoint:")
    temp = getattr(args, "temperature", DEFAULT_GENERATE_PROBE_TEMPERATURE)
    top_k = getattr(args, "top_k", DEFAULT_GENERATE_PROBE_TOP_K)
    top_p = getattr(args, "top_p", DEFAULT_GENERATE_PROBE_TOP_P)
    print(
        f"  python generate.py --checkpoint {run_dir} --prompt \"once upon a\" "
        f"--max-new-tokens 256 --temperature {temp} --top-k {top_k} --top-p {top_p}"
    )
    print(f"  python train.py --generate --models-dir {resolve_checkpoints_dir(args.models_dir)}")
    print(f"  python train.py --compare-quarters --checkpoint {run_dir}")

    if getattr(args, "set_best", None) and not getattr(args, "compare_quarters", False):
        # Non-interactive promote without full trial generation if requested mid-flow.
        source = run_dir / args.set_best
        if (source / "config.json").exists():
            promote_best(run_dir, source, meta={"step": global_step})
            print(f"Promoted {args.set_best} -> {run_dir / 'best'}")
        else:
            run_quality_trial_for_args(args, str(run_dir))
    elif cli_common.should_run_quality_trial(args):
        print("\n### QUALITY TRIAL ###")
        run_quality_trial_for_args(args, str(run_dir))

    if getattr(args, "plot", False):
        _render_post_training_plots()

    return str(run_dir)


def _render_post_training_plots() -> None:
    """Best-effort: render training_log_plotter + loss_landscape_plotter charts
    from output/logs/training.log and save them under output/logs/. Never raises on failure."""
    try:
        import training_log_plotter as tlp

        log_path = DEFAULT_TRAINING_LOG
        runs = tlp._load_runs([log_path])
        if runs:
            tlp.plot_runs_liveable(
                runs=runs, metric_name="tok/s", smooth_window=21, ema_alpha=0.08,
                raw_alpha=0.10, forecast_window=40, forecast_enabled=True,
                forecast_use_smoothed=True, show_raw_loss=False, show_ema_loss=False,
                show_raw_metric=True, live=False, refresh_seconds=1.0,
                source_paths=[log_path], save_path=DEFAULT_TRAINING_PLOT,
                show=False,
            )
    except Exception as exc:
        logger.warning(f"training_log_plotter failed: {exc}")

    try:
        import loss_landscape_plotter as llp

        runs = llp.read_runs(log_dir="output", all_runs=False)
        llp.render_landscape(runs, out_path=DEFAULT_LANDSCAPE_PLOT, show=False)
    except Exception as exc:
        logger.warning(f"loss_landscape_plotter failed: {exc}")


def main() -> None:
    args = parse_args()
    print(f"llm-gpu-8 v{__version__}")

    if args.generate:
        generate_test_menu(args)
        return

    if getattr(args, "compare_quarters", False):
        ensure_output_dirs()
        setup_logging(log_filename="quality_trial")
        run_quality_trial_for_args(args)
        return

    # --set-best alone (no train intent): promote without re-training.
    # With --compare-quarters handled above; here copy bundle only.
    train_intent = (
        getattr(args, "resume", False)
        or getattr(args, "menu", False)
        or args.steps is not None
        or args.epochs is not None
        or getattr(args, "quality_trial", False)
    )
    if getattr(args, "set_best", None) and not train_intent:
        ensure_output_dirs()
        setup_logging(log_filename="quality_trial")
        root = run_root_for_checkpoint(args.checkpoint)
        source = root / args.set_best
        if not (source / "config.json").exists():
            source = Path(args.set_best)
        if (source / "config.json").exists():
            promote_best(root, source, meta={"source_name": source.name})
            print(f"Promoted '{source}' -> {root / 'best'}")
        else:
            print(f"Cannot promote: '{args.set_best}' not found under {root}")
        return

    train(args)


if __name__ == "__main__":
    main()
