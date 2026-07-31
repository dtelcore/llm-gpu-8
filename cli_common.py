"""
cli_common.py

Shared argparse building blocks for train.py, auto_train.py, generate.py,
and interactive.py so trace flags and paths stay consistent across CLIs.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set

from model.trace import TraceContext
from logging_config import logger
from paths import (
    DEFAULT_CHECKPOINT_DIR,
    DEFAULT_CONFIG_PATH,
    LEGACY_CONFIG_PATH,
    resolve_checkpoints_dir,
)
from tokenizer.factory import DEFAULT_BPE_MERGES, DEFAULT_TOKENIZER


def add_trace_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("tracing (all quiet by default)")
    group.add_argument("--verbose", action="store_true", help="Enable token + top-k logit traces on traced steps")
    group.add_argument("--trace-logits", action="store_true", help="Dump top-k logits/probabilities on traced steps")
    group.add_argument("--trace-tokens", action="store_true", help="Dump token <-> id mapping on traced steps")
    group.add_argument("--trace-neurons", action="store_true", help="Dump per-layer activation mean/std/norm on traced steps")
    group.add_argument("--trace-vectorization", action="store_true", help="Print GEMM shapes and CUDA grid/block launches on traced steps")
    group.add_argument("--trace-every", type=int, default=None, help="Emit traces every N steps (default: 10%% of total training steps; ignored for generate/interactive where it defaults to every step)")


def add_runtime_observability_args(parser: argparse.ArgumentParser) -> None:
    """Stage 3.1: opt-in runtime metrics and ScratchPool memory timeline (off by default)."""
    group = parser.add_argument_group("runtime observability (Stage 3.1; off by default)")
    group.add_argument(
        "--runtime-metrics", action="store_true",
        help="Log grad_norm/param_norm/sync_count/sync_ms/scratch_peak_mb on [train] lines; "
             "meter host↔device transfers (no memory JSONL)",
    )
    group.add_argument(
        "--memory-timeline", action="store_true",
        help="Record ScratchPool alloc/reuse/clear to output/logs/memory_timeline_<run>.jsonl "
             "(implies --runtime-metrics)",
    )


def add_config_arg(parser: argparse.ArgumentParser, default: Optional[str] = None) -> None:
    default_path = str(default or DEFAULT_CONFIG_PATH)
    parser.add_argument("--config", type=str, default=default_path, help="Path to training_config.json")


def add_training_length_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("training length")
    group.add_argument("--learning-rate", type=float, default=None, help="Override config learning_rate (prompted interactively if omitted)")
    group.add_argument("--epochs", type=int, default=None, help="Override config num_epochs (ignored if --steps is given)")
    group.add_argument("--steps", type=int, default=None, help="Total training steps across ALL epochs (overrides --epochs and config num_epochs)")
    group.add_argument("--log-every", type=int, default=100, help="Print a progress line every N steps (default: 100)")
    group.add_argument(
        "--checkpoint-every", type=int, default=None,
        help="Save a checkpoint every N steps (default: min(1000, batches/epoch); override to tune disk I/O vs crash safety)",
    )
    group.add_argument(
        "--no-prompt", action="store_true", help="Never prompt for learning_rate/steps/epochs; silently use config/CLI defaults")
    group.add_argument(
        "--min-lr-ratio", type=float, default=None,
        help="Cosine LR floor as a fraction of base learning rate after warmup (default: 0.1)",
    )
    group.add_argument(
        "--window-stride", type=int, default=None,
        help="Token stride between training windows (default: hyperparameters window_stride or 1)",
    )
    group.add_argument(
        "--val-every", type=int, default=0,
        help="Lightweight val loss every N steps without quarterly checkpoint I/O (0=off; try 500 for sweeps)",
    )


def prompt_training_length_and_lr(args: argparse.Namespace, hyperparams: Dict) -> None:
    """Interactively ask for learning_rate, and either total steps or epochs, filling
    in only whichever of `args.learning_rate` / `args.steps` / `args.epochs` weren't
    already supplied on the command line. Mutates `args` in place. No-op if --no-prompt
    was passed or stdin isn't interactive.

    Falls back silently to config/CLI values on EOF (e.g. piped/non-interactive input).
    """
    if getattr(args, "no_prompt", False):
        return

    try:
        if getattr(args, "learning_rate", None) is None:
            default_lr = hyperparams.get("learning_rate", 0.01)
            raw = input(f"Learning rate [default={default_lr}]: ").strip()
            args.learning_rate = float(raw) if raw else default_lr

        if args.steps is None and args.epochs is None:
            default_epochs = hyperparams.get("num_epochs", 10)
            raw_steps = input("Total steps (blank to specify epochs instead): ").strip()
            if raw_steps:
                args.steps = int(raw_steps)
            else:
                raw_epochs = input(f"Epochs [default={default_epochs}]: ").strip()
                args.epochs = int(raw_epochs) if raw_epochs else default_epochs
    except EOFError:
        if getattr(args, "learning_rate", None) is None:
            args.learning_rate = hyperparams.get("learning_rate", 0.01)
        if args.steps is None and args.epochs is None:
            args.epochs = hyperparams.get("num_epochs", 10)


def add_model_hyperparam_args(parser: argparse.ArgumentParser) -> None:
    """Model architecture overrides (applied to config['model'] before the GPTConfig
    is built). Weight init still runs off whatever the final values end up being."""
    group = parser.add_argument_group("model hyperparameters")
    group.add_argument("--embedding-dim", type=int, default=None, help="Override config embedding_dim")
    group.add_argument("--num-heads", type=int, default=None, help="Override config num_heads")
    group.add_argument("--num-layers", type=int, default=None, help="Override config num_layers")
    group.add_argument("--max-len", type=int, default=None, help="Override config max_len (context window)")
    group.add_argument("--dropout", type=float, default=None, help="Override config dropout_prob")
    group.add_argument("--batch-size", type=int, default=None, help="Override config batch_size")
    group.add_argument("--weight-decay", type=float, default=None, help="Override config weight_decay")
    group.add_argument("--warmup-steps", type=int, default=None, help="Override config warmup_steps")
    group.add_argument("--gradient-clip", type=float, default=None, help="Override config gradient_clip norm")
    group.add_argument(
        "--grad-accum", "--gradient-accumulation-steps",
        dest="gradient_accumulation_steps",
        type=int, default=None,
        help="Micro-batches per optimizer step (gradient accumulation). Default: config or 1",
    )
    group.add_argument(
        "--tie-embeddings", action="store_true", default=None,
        help="Tie token_embedding with lm_head (saves V*C params). Overrides config.",
    )
    group.add_argument(
        "--no-tie-embeddings", action="store_true", default=False,
        help="Keep separate lm_head weights (legacy). Overrides config.",
    )
    group.add_argument(
        "--run-budget", type=int, default=None,
        help="Absolute step budget for quarterly 25/50/75/100%% milestones. "
             "Persisted in state.json so chunked --steps resumes do not rewrite quarter_100. "
             "Default: this session's total_steps (or prior state.run_budget_steps on resume).",
    )
    group.add_argument(
        "--norm-type", type=str, choices=("layernorm", "rmsnorm"), default=None,
        help="Normalization: layernorm (legacy) or rmsnorm (scale-only)",
    )
    group.add_argument(
        "--pos-encoding", type=str, choices=("learned", "rope"), default=None,
        help="Position encoding: learned absolute table or RoPE",
    )
    group.add_argument(
        "--grad-checkpoint", action="store_true", default=None,
        help="Recompute attention/MLP activations in backward to save VRAM",
    )
    group.add_argument(
        "--no-grad-checkpoint", action="store_true", default=False,
        help="Disable gradient checkpointing (override config)",
    )


def prompt_model_hyperparams(args: argparse.Namespace, model_config: Dict, hyperparams: Dict) -> None:
    """Interactively fill in model architecture (embedding_dim/num_heads/num_layers/
    max_len/dropout) and remaining training hyperparameters (batch_size/weight_decay/
    warmup_steps/gradient_clip) that weren't already supplied on the command line.
    Mutates `model_config` and `hyperparams` in place. No-op if --no-prompt was passed.
    Falls back silently to config/CLI values on EOF (non-interactive stdin).
    """
    no_prompt = getattr(args, "no_prompt", False)

    def _ask(flag_attr: str, label: str, target: Dict, key: str, caster, current_default):
        # An explicit CLI flag always wins, --no-prompt or not.
        if getattr(args, flag_attr, None) is not None:
            target[key] = getattr(args, flag_attr)
            return
        if no_prompt:
            target[key] = current_default
            return
        try:
            raw = input(f"{label} [default={current_default}]: ").strip()
            target[key] = caster(raw) if raw else current_default
        except EOFError:
            target[key] = current_default

    _ask("embedding_dim", "Embedding dim", model_config, "embedding_dim", int, model_config.get("embedding_dim", 32))
    _ask("num_heads", "Num attention heads", model_config, "num_heads", int, model_config.get("num_heads", 4))
    _ask("num_layers", "Num transformer layers", model_config, "num_layers", int, model_config.get("num_layers", 2))
    _ask("max_len", "Max sequence length (context window)", model_config, "max_len", int, model_config.get("max_len", 16))
    _ask("dropout", "Dropout probability", model_config, "dropout_prob", float, model_config.get("dropout_prob", 0.0))

    if getattr(args, "no_tie_embeddings", False):
        model_config["tie_embeddings"] = False
    elif getattr(args, "tie_embeddings", None):
        model_config["tie_embeddings"] = True
    elif "tie_embeddings" not in model_config:
        model_config["tie_embeddings"] = True  # new runs default tied

    if getattr(args, "norm_type", None) is not None:
        model_config["norm_type"] = args.norm_type
    elif "norm_type" not in model_config:
        model_config["norm_type"] = "rmsnorm"

    if getattr(args, "pos_encoding", None) is not None:
        model_config["pos_encoding"] = args.pos_encoding
    elif "pos_encoding" not in model_config:
        model_config["pos_encoding"] = "rope"

    if getattr(args, "no_grad_checkpoint", False):
        model_config["gradient_checkpointing"] = False
    elif getattr(args, "grad_checkpoint", None):
        model_config["gradient_checkpointing"] = True
    elif "gradient_checkpointing" not in model_config:
        model_config["gradient_checkpointing"] = False

    _ask("batch_size", "Batch size", hyperparams, "batch_size", int, hyperparams.get("batch_size", 2))
    _ask("weight_decay", "Weight decay", hyperparams, "weight_decay", float, hyperparams.get("weight_decay", 0.01))
    _ask("warmup_steps", "Warmup steps", hyperparams, "warmup_steps", int, hyperparams.get("warmup_steps", 0))
    _ask("gradient_clip", "Gradient clip norm", hyperparams, "gradient_clip", float, hyperparams.get("gradient_clip", 1.0))
    _ask(
        "gradient_accumulation_steps",
        "Gradient accumulation steps (micro-batches per optimizer step)",
        hyperparams, "gradient_accumulation_steps", int,
        hyperparams.get("gradient_accumulation_steps", 1),
    )


def add_checkpoint_arg(parser: argparse.ArgumentParser, default: Optional[str] = None) -> None:
    default_ckpt = str(default or DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--checkpoint", type=str, default=default_ckpt, help="Checkpoint directory")


def add_seed_arg(parser: argparse.ArgumentParser, default: int = 42) -> None:
    parser.add_argument("--seed", type=int, default=default, help="Random seed")


def add_tokenizer_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("tokenizer")
    group.add_argument(
        "--tokenizer", type=str, choices=("bpe", "char"), default=None,
        help=f"Tokenizer for new runs (default: {DEFAULT_TOKENIZER}; resume loads checkpoint vocab)",
    )
    group.add_argument(
        "--bpe-merges", type=int, default=None, dest="bpe_merges",
        help=f"BPE merge count when --tokenizer bpe (default: {DEFAULT_BPE_MERGES})",
    )


def build_tracer(args, default_trace_every: int = 100) -> TraceContext:
    """Build a TraceContext, resolving --trace-every to `default_trace_every`
    when the user didn't pass it explicitly (args.trace_every is None)."""
    tracer = TraceContext.from_args(args)
    resolved = args.trace_every if getattr(args, "trace_every", None) is not None else default_trace_every
    tracer.trace_every = max(1, resolved)
    return tracer


def load_config(path: str) -> Dict:
    config_path = Path(path)
    if not config_path.exists():
        if config_path.resolve() == DEFAULT_CONFIG_PATH.resolve() and LEGACY_CONFIG_PATH.exists():
            logger.info(
                "Config not found at %s; falling back to legacy %s",
                DEFAULT_CONFIG_PATH, LEGACY_CONFIG_PATH,
            )
            config_path = LEGACY_CONFIG_PATH
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _checkpoint_label(path: Path) -> str:
    """Human-readable label with step / metrics when available."""
    step = None
    state_path = path / "state.json"
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                step = json.load(f).get("step")
        except (OSError, json.JSONDecodeError, TypeError):
            step = None

    extras = []
    if step is not None:
        extras.append(f"step={step:,}" if isinstance(step, int) else f"step={step}")

    metrics_path = path / "metrics.json"
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            if metrics.get("val_loss") is not None:
                extras.append(f"val_loss={float(metrics['val_loss']):.4f}")
            elif metrics.get("loss") is not None:
                extras.append(f"loss={float(metrics['loss']):.4f}")
            quality = metrics.get("quality") or {}
            if isinstance(quality, dict) and quality.get("aggregate") is not None:
                extras.append(f"quality={float(quality['aggregate']):.3f}")
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    suffix = f" [{', '.join(extras)}]" if extras else ""
    return f"{path}{suffix}"


def list_checkpoints(models_dir: Optional[str] = None) -> List[Path]:
    """Return resumable checkpoint dirs: each run root (latest), plus best/ and
    quarter_* nested under it. Newest run roots first; nested entries follow
    their parent in order latest-implied → best → quarter_25…100."""
    from paths import BEST_DIR_NAME, QUARTER_NAMES

    root = resolve_checkpoints_dir(models_dir)
    if not root.exists():
        return []

    run_dirs = [
        d for d in root.iterdir()
        if d.is_dir() and (d / "config.json").exists()
    ]
    run_dirs = sorted(run_dirs, key=lambda d: d.stat().st_mtime, reverse=True)

    found: List[Path] = []
    for run in run_dirs:
        found.append(run)
        best = run / BEST_DIR_NAME
        if best.is_dir() and (best / "config.json").exists():
            found.append(best)
        for name in QUARTER_NAMES:
            q = run / name
            if q.is_dir() and (q / "config.json").exists():
                found.append(q)
    return found


def add_probe_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("generate probes (quarterly milestones)")
    group.add_argument(
        "--no-generate-probe", action="store_true",
        help="Skip mid-training text generation at quarterly milestones "
             "(still saves quarter_XX/, val metrics, and full traces)",
    )
    group.add_argument(
        "--generate-probe-prompt", type=str, default="once upon a",
        help="Prompt used for mid-training generation probes (default: once upon a)",
    )
    group.add_argument(
        "--generate-probe-tokens", type=int, default=256,
        help="Characters to generate for each mid-training probe (default: 256)",
    )


def add_quality_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("generation quality trial")
    group.add_argument(
        "--quality-trial", action="store_true",
        help="After training, run sequential inter-quarter generation quality trial and prompt to promote best/",
    )
    group.add_argument(
        "--no-quality-trial", action="store_true",
        help="Never run the post-training quality trial (default when --no-prompt)",
    )
    group.add_argument(
        "--quality-prompt", type=str, default=None,
        help="Prompt for quality trial generations (default: --generate-probe-prompt)",
    )
    group.add_argument(
        "--quality-weights", type=str, default=None,
        help="Comma weights spelling=1,punctuation=1,grammar=1,semantics=1",
    )
    group.add_argument(
        "--compare-quarters", action="store_true",
        help="Skip training: run quality trial on --checkpoint run (latest/quarters) and exit",
    )
    group.add_argument(
        "--set-best", type=str, default=None,
        help="Non-interactively promote a quarter (e.g. quarter_50) as best/ under the run",
    )


def should_run_quality_trial(args: argparse.Namespace) -> bool:
    """Default: on for interactive runs, off for --no-prompt; explicit flags win."""
    if getattr(args, "no_quality_trial", False):
        return False
    if getattr(args, "quality_trial", False):
        return True
    if getattr(args, "no_prompt", False):
        return False
    # Interactive menus default to offering the trial after training.
    return bool(getattr(args, "menu", False))


def prompt_resume_or_new(models_dir: Optional[str] = None) -> Optional[str]:
    """First-menu-option prompt: resume training from an existing checkpoint, or
    start a fresh run. Returns the checkpoint path to resume from, or None to
    signal 'start fresh' (including when no checkpoints exist or stdin is EOF)."""
    checkpoints = list_checkpoints(models_dir)
    ckpt_root = resolve_checkpoints_dir(models_dir)

    print(f"\nAvailable checkpoints in '{ckpt_root}' (latest / best / quarters):")
    if checkpoints:
        for i, d in enumerate(checkpoints, 1):
            print(f"  {i}. {_checkpoint_label(d)}")
    else:
        print("  (none found)")
    print("  n. Start a new training run")

    if not checkpoints:
        return None

    try:
        choice = input("\nResume from checkpoint (number), or 'n' for new [default=n]: ").strip()
    except EOFError:
        return None

    if not choice or choice.lower() == "n":
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(checkpoints):
        return str(checkpoints[int(choice) - 1])

    candidate = Path(choice)
    if (candidate / "config.json").exists():
        return str(candidate)

    print(f"'{choice}' not recognized; starting a new training run.")
    return None


def select_checkpoint_interactive(
    models_dir: Optional[str] = None,
    allow_new: bool = False,
    default_new_name: str = "run1",
    prompt_label: str = "checkpoint",
) -> str:
    """Interactively list checkpoints under `models_dir` and let the user pick one
    by number, type a custom path directly, or (if allow_new) create a new name.

    Returns the chosen checkpoint directory as a string path.
    """
    checkpoints = list_checkpoints(models_dir)
    ckpt_root = resolve_checkpoints_dir(models_dir)

    print(f"\nAvailable checkpoints in '{ckpt_root}' (latest / best / quarters):")
    if checkpoints:
        for i, d in enumerate(checkpoints, 1):
            print(f"  {i}. {_checkpoint_label(d)}")
    else:
        print("  (none found)")
    if allow_new:
        print("  n. Enter a new checkpoint name")

    while True:
        hint = " (number, path, or 'n' for new)" if allow_new else " (number or path)"
        choice = input(f"\nSelect {prompt_label}{hint}: ").strip()

        if not choice:
            if checkpoints:
                return str(checkpoints[0])
            if allow_new:
                choice = "n"
            else:
                print("No checkpoints available; please enter a path.")
                continue

        if allow_new and choice.lower() == "n":
            name = input(f"New checkpoint name [default={default_new_name}]: ").strip() or default_new_name
            if "/" in name or "\\" in name:
                return name
            return str(ckpt_root / name)

        if choice.isdigit() and checkpoints and 1 <= int(choice) <= len(checkpoints):
            return str(checkpoints[int(choice) - 1])

        # Fall back to treating the input as a literal path.
        candidate = Path(choice)
        if not allow_new and not (candidate / "config.json").exists():
            print(f"'{choice}' has no config.json; pick a valid checkpoint.")
            continue
        return choice


# ---------------------------------------------------------------------------
# --menu flag-group picker (train / auto_train / generate)
# ---------------------------------------------------------------------------

def _ask_raw(label: str, default_display: str) -> Optional[str]:
    try:
        return input(f"{label} [default={default_display}]: ").strip()
    except EOFError:
        return None


def _ask_value(args: argparse.Namespace, attr: str, label: str, caster, default):
    """Prompt for args.attr; Enter keeps current value or default."""
    current = getattr(args, attr, None)
    display = current if current is not None else default
    if getattr(args, "no_prompt", False):
        if current is None:
            setattr(args, attr, default)
        return
    raw = _ask_raw(label, display if display is not None else "None")
    if raw is None or raw == "":
        if current is None:
            setattr(args, attr, default)
        return
    try:
        setattr(args, attr, caster(raw))
    except (TypeError, ValueError):
        print(f"  Invalid value {raw!r}; keeping {display!r}")
        if current is None:
            setattr(args, attr, default)


def _ask_bool(args: argparse.Namespace, attr: str, label: str, default: bool) -> None:
    """Prompt y/n for a store_true-style flag. CLI True wins; unset uses default."""
    if getattr(args, attr, False) is True:
        return
    # Optional store_true with default=None (tie_embeddings / grad_checkpoint).
    existing = getattr(args, attr, None)
    if existing is True:
        return
    if getattr(args, "no_prompt", False):
        setattr(args, attr, default)
        return
    hint = "Y/n" if default else "y/N"
    raw = _ask_raw(f"{label} ({hint})", "Y" if default else "N")
    if raw is None or raw == "":
        setattr(args, attr, default)
        return
    setattr(args, attr, raw.lower() in ("y", "yes", "1", "true", "on"))


def _parse_group_selection(raw: str, n_groups: int) -> Set[int]:
    raw = (raw or "").strip().lower()
    if not raw:
        return set()
    if raw == "all":
        return set(range(1, n_groups + 1))
    selected: Set[int] = set()
    for part in raw.replace(" ", "").split(","):
        if part.isdigit() and 1 <= int(part) <= n_groups:
            selected.add(int(part))
    return selected


def _prompt_tokenizer_group(args: argparse.Namespace, config: Optional[Dict]) -> None:
    kind = getattr(args, "tokenizer", None) or (
        (config or {}).get("dataset", {}) or {}
    ).get("tokenizer") or DEFAULT_TOKENIZER
    merges = getattr(args, "bpe_merges", None)
    if merges is None:
        merges = ((config or {}).get("dataset", {}) or {}).get("bpe_merges", DEFAULT_BPE_MERGES)

    if getattr(args, "tokenizer", None) is None and not getattr(args, "no_prompt", False):
        raw = _ask_raw("Tokenizer (bpe/char)", kind)
        if raw:
            kind = raw.strip().lower()
            if kind not in ("bpe", "char"):
                print(f"  Unknown tokenizer {raw!r}; using {DEFAULT_TOKENIZER}")
                kind = DEFAULT_TOKENIZER
        args.tokenizer = kind
    elif getattr(args, "tokenizer", None) is None:
        args.tokenizer = kind

    if args.tokenizer == "bpe":
        if getattr(args, "bpe_merges", None) is None and not getattr(args, "no_prompt", False):
            raw = _ask_raw("BPE merges", merges)
            args.bpe_merges = int(raw) if raw else int(merges)
        elif getattr(args, "bpe_merges", None) is None:
            args.bpe_merges = int(merges)

    if config is not None:
        dataset = config.setdefault("dataset", {})
        dataset["tokenizer"] = args.tokenizer
        if args.tokenizer == "bpe":
            dataset["bpe_merges"] = int(args.bpe_merges or DEFAULT_BPE_MERGES)


def _prompt_obs_group(args: argparse.Namespace) -> None:
    _ask_bool(args, "runtime_metrics", "Enable --runtime-metrics?", False)
    _ask_bool(args, "memory_timeline", "Enable --memory-timeline? (implies metrics)", False)
    if getattr(args, "memory_timeline", False):
        args.runtime_metrics = True


def _prompt_trace_group(args: argparse.Namespace, *, generate_style: bool = False) -> None:
    _ask_bool(args, "verbose", "Enable --verbose?", False)
    _ask_bool(args, "trace_logits", "Enable --trace-logits?", False)
    _ask_bool(args, "trace_tokens", "Enable --trace-tokens?", False)
    _ask_bool(args, "trace_neurons", "Enable --trace-neurons?", False)
    _ask_bool(args, "trace_vectorization", "Enable --trace-vectorization?", False)
    any_trace = any(
        getattr(args, a, False)
        for a in ("verbose", "trace_logits", "trace_tokens", "trace_neurons", "trace_vectorization")
    )
    if any_trace and getattr(args, "trace_every", None) is None:
        default_every = 1 if generate_style else None
        if default_every is not None:
            _ask_value(args, "trace_every", "Trace every N steps", int, default_every)
        else:
            raw = _ask_raw("Trace every N steps (blank = ~10% of train steps)", "auto")
            if raw and raw.lower() != "auto":
                try:
                    args.trace_every = int(raw)
                except ValueError:
                    pass


def _prompt_probe_group(args: argparse.Namespace) -> None:
    # probes on by default → no_generate_probe default False
    skip = False
    if not getattr(args, "no_generate_probe", False):
        _ask_bool(args, "no_generate_probe", "Skip quarterly generate probes (--no-generate-probe)?", False)
        skip = bool(args.no_generate_probe)
    if skip:
        return
    _ask_value(args, "generate_probe_prompt", "Generate-probe prompt", str, "once upon a")
    _ask_value(args, "generate_probe_tokens", "Generate-probe tokens", int, 256)
    if hasattr(args, "temperature"):
        from training.probe import (
            DEFAULT_GENERATE_PROBE_TEMPERATURE,
            DEFAULT_GENERATE_PROBE_TOP_K,
            DEFAULT_GENERATE_PROBE_TOP_P,
        )
        _ask_value(args, "temperature", "Probe temperature", float, DEFAULT_GENERATE_PROBE_TEMPERATURE)
        _ask_value(args, "top_k", "Probe top-k (blank=None)", lambda s: int(s) if s else None, DEFAULT_GENERATE_PROBE_TOP_K)
        _ask_value(args, "top_p", "Probe top-p (blank=None)", lambda s: float(s) if s else None, DEFAULT_GENERATE_PROBE_TOP_P)


def _prompt_quality_group(args: argparse.Namespace) -> None:
    # Interactive menu default: trial on
    run_trial = True
    if getattr(args, "no_quality_trial", False):
        run_trial = False
    elif getattr(args, "quality_trial", False):
        run_trial = True
    else:
        _ask_bool(args, "quality_trial", "Run post-train quality trial?", True)
        run_trial = bool(args.quality_trial)
        if not run_trial:
            args.no_quality_trial = True
    if run_trial:
        args.no_quality_trial = False
        default_prompt = getattr(args, "generate_probe_prompt", None) or "once upon a"
        if getattr(args, "quality_prompt", None) is None:
            _ask_value(args, "quality_prompt", "Quality-trial prompt", str, default_prompt)
        if getattr(args, "quality_weights", None) is None:
            raw = _ask_raw("Quality weights (spelling,punctuation,grammar,semantics)", "1,1,1,1")
            if raw:
                args.quality_weights = raw


def _prompt_length_extras(args: argparse.Namespace, hyperparams: Dict) -> None:
    """Length flags beyond LR/steps/epochs (those use prompt_training_length_and_lr)."""
    _ask_value(args, "log_every", "Log every N steps", int, getattr(args, "log_every", 100) or 100)
    if getattr(args, "checkpoint_every", None) is None:
        raw = _ask_raw("Checkpoint every N steps (blank=auto)", "auto")
        if raw and raw.lower() != "auto":
            try:
                args.checkpoint_every = int(raw)
            except ValueError:
                pass
    default_min = hyperparams.get("min_lr_ratio", 0.1)
    if getattr(args, "min_lr_ratio", None) is None:
        _ask_value(args, "min_lr_ratio", "Min LR ratio (cosine floor)", float, default_min)
    default_stride = hyperparams.get("window_stride", 1)
    if getattr(args, "window_stride", None) is None:
        _ask_value(args, "window_stride", "Window stride", int, default_stride)
    if getattr(args, "val_every", 0) == 0:
        _ask_value(args, "val_every", "Val every N steps (0=off)", int, 0)


def _prompt_model_arch_extras(args: argparse.Namespace, model_config: Dict, hyperparams: Dict) -> None:
    """Architecture toggles prompted when Model group is selected (dims via prompt_model_hyperparams)."""
    if getattr(args, "norm_type", None) is None:
        default_norm = model_config.get("norm_type", "rmsnorm")
        raw = _ask_raw("Norm type (layernorm/rmsnorm)", default_norm)
        if raw:
            args.norm_type = raw.strip().lower()
        else:
            args.norm_type = default_norm
    if getattr(args, "pos_encoding", None) is None:
        default_pos = model_config.get("pos_encoding", "rope")
        raw = _ask_raw("Pos encoding (learned/rope)", default_pos)
        if raw:
            args.pos_encoding = raw.strip().lower()
        else:
            args.pos_encoding = default_pos

    if not getattr(args, "no_tie_embeddings", False) and getattr(args, "tie_embeddings", None) is None:
        tied = bool(model_config.get("tie_embeddings", True))
        _ask_bool(args, "tie_embeddings", "Tie embeddings?", tied)
        if not args.tie_embeddings:
            args.no_tie_embeddings = True

    if not getattr(args, "no_grad_checkpoint", False) and getattr(args, "grad_checkpoint", None) is None:
        gc = bool(model_config.get("gradient_checkpointing", False))
        _ask_bool(args, "grad_checkpoint", "Enable --grad-checkpoint (VRAM, not speed)?", gc)
        if not args.grad_checkpoint:
            args.no_grad_checkpoint = True

    if getattr(args, "run_budget", None) is None:
        raw = _ask_raw("Run budget steps for quarters (blank=session total)", "auto")
        if raw and raw.lower() != "auto":
            try:
                args.run_budget = int(raw)
            except ValueError:
                pass


def _prompt_sampling_group(args: argparse.Namespace, *, defaults: Optional[Dict] = None) -> None:
    d = defaults or {}
    _ask_value(args, "prompt", "Prompt", str, d.get("prompt", "the"))
    _ask_value(args, "max_new_tokens", "Max new tokens", int, d.get("max_new_tokens", 80))
    _ask_value(args, "temperature", "Temperature", float, d.get("temperature", 0.8))
    if getattr(args, "top_k", None) is None:
        raw = _ask_raw("Top-k (blank=None)", d.get("top_k", "None"))
        if raw and raw.lower() != "none":
            try:
                args.top_k = int(raw)
            except ValueError:
                pass
    if getattr(args, "top_p", None) is None:
        raw = _ask_raw("Top-p (blank=None)", d.get("top_p", "None"))
        if raw and raw.lower() != "none":
            try:
                args.top_p = float(raw)
            except ValueError:
                pass
    if hasattr(args, "seed"):
        _ask_value(args, "seed", "Seed", int, getattr(args, "seed", 42) or 42)


def _prompt_decode_group(args: argparse.Namespace) -> None:
    _ask_bool(args, "no_kv_cache", "Disable KV cache (--no-kv-cache)?", False)
    _ask_bool(args, "cuda_graph", "Enable --cuda-graph (KV kernel-chain)?", False)


def prompt_run_flag_menu(
    args: argparse.Namespace,
    *,
    fresh_run: bool,
    entry: str,
    config: Optional[Dict] = None,
    hyperparams: Optional[Dict] = None,
    model_config: Optional[Dict] = None,
) -> Set[str]:
    """Pick flag groups to customize, then prompt selected flags. Returns group keys selected.

    entry: "train" | "auto_train" | "generate"
    """
    if getattr(args, "no_prompt", False):
        return set()

    hyperparams = hyperparams or (config or {}).get("hyperparameters") or {}
    model_config = model_config or (config or {}).get("model") or {}

    if entry == "generate":
        groups: List[tuple] = [
            ("sampling", "Sampling", "prompt=the, tokens=80, temp=0.8, no top-k/p"),
            ("decode", "Decode", "KV cache on, cuda-graph off"),
            ("trace", "Tracing", "all off"),
        ]
        title = "Configure generate flags"
    else:
        groups = []
        if fresh_run:
            groups.append(("tokenizer", "Tokenizer", f"{DEFAULT_TOKENIZER} / {DEFAULT_BPE_MERGES} merges"))
        groups.extend([
            ("length", "Training length", "preset/config LR·steps; stride/val defaults"),
        ])
        if fresh_run:
            groups.append(("model", "Model / arch", "rmsnorm+rope+tied; grad-checkpoint off"))
        groups.extend([
            ("obs", "Observability", "metrics off, timeline off"),
            ("trace", "Tracing", "all off"),
            ("probe", "Generate probes", "on"),
            ("quality", "Quality trial", "on (interactive)"),
        ])
        if entry == "train":
            groups.append(("plot", "Plot", "off"))
        elif entry == "auto_train":
            groups.append(("smoke", "Smoke generate", "prompt=the, tokens=80"))
        title = "Configure run flags"

    print(f"\n{title} — groups to customize (Enter=none / e.g. 1,4 or all)")
    print("-" * 70)
    for i, (_key, label, default_note) in enumerate(groups, 1):
        print(f"  [{i}] {label:<18} default: {default_note}")

    try:
        raw = input("Groups to customize: ").strip()
    except EOFError:
        raw = ""
    selected_idx = _parse_group_selection(raw, len(groups))
    selected_keys = {groups[i - 1][0] for i in selected_idx}

    if "tokenizer" in selected_keys:
        print("\n[Tokenizer]")
        _prompt_tokenizer_group(args, config)
    elif fresh_run and entry != "generate" and config is not None:
        # Apply silent defaults into config for new runs.
        dataset = config.setdefault("dataset", {})
        if getattr(args, "tokenizer", None) is None:
            args.tokenizer = dataset.get("tokenizer") or DEFAULT_TOKENIZER
        dataset["tokenizer"] = args.tokenizer
        if args.tokenizer == "bpe":
            if getattr(args, "bpe_merges", None) is None:
                args.bpe_merges = int(dataset.get("bpe_merges", DEFAULT_BPE_MERGES))
            dataset["bpe_merges"] = int(args.bpe_merges)

    if "length" in selected_keys:
        print("\n[Training length]")
        prompt_training_length_and_lr(args, hyperparams)
        _prompt_length_extras(args, hyperparams)

    if "model" in selected_keys and fresh_run:
        print("\n[Model / arch]")
        _prompt_model_arch_extras(args, model_config, hyperparams)
        # Dims / batch etc. — interactive
        prompt_model_hyperparams(args, model_config, hyperparams)

    if "obs" in selected_keys:
        print("\n[Observability]")
        _prompt_obs_group(args)

    if "trace" in selected_keys:
        print("\n[Tracing]")
        _prompt_trace_group(args, generate_style=(entry == "generate"))

    if "probe" in selected_keys:
        print("\n[Generate probes]")
        _prompt_probe_group(args)

    if "quality" in selected_keys:
        print("\n[Quality trial]")
        _prompt_quality_group(args)

    if "plot" in selected_keys and hasattr(args, "plot"):
        print("\n[Plot]")
        _ask_bool(args, "plot", "Enable --plot after training?", False)

    if "smoke" in selected_keys:
        print("\n[Smoke generate]")
        from training.probe import (
            DEFAULT_GENERATE_PROBE_TEMPERATURE,
            DEFAULT_GENERATE_PROBE_TOP_K,
            DEFAULT_GENERATE_PROBE_TOP_P,
        )
        _prompt_sampling_group(
            args,
            defaults={
                "prompt": "the",
                "max_new_tokens": 80,
                "temperature": DEFAULT_GENERATE_PROBE_TEMPERATURE,
                "top_k": DEFAULT_GENERATE_PROBE_TOP_K,
                "top_p": DEFAULT_GENERATE_PROBE_TOP_P,
            },
        )

    if "sampling" in selected_keys:
        print("\n[Sampling]")
        _prompt_sampling_group(args)

    if "decode" in selected_keys:
        print("\n[Decode]")
        _prompt_decode_group(args)

    return selected_keys


def apply_silent_model_defaults(args: argparse.Namespace, model_config: Dict, hyperparams: Dict) -> None:
    """Apply model/hyperparam defaults without prompting (menu group skipped)."""
    saved = getattr(args, "no_prompt", False)
    args.no_prompt = True
    try:
        prompt_model_hyperparams(args, model_config, hyperparams)
    finally:
        args.no_prompt = saved

