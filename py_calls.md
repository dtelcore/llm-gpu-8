# py_calls.md — runnable entry points

**TinyStories on GT 730:** start with [`guide.md`](guide.md). Activate the project venv first (PyCUDA / sm_35):

```powershell
.\venv\Scripts\Activate.ps1
```

Shared flag groups live in [`cli_common.py`](cli_common.py) and are referenced below as **(shared: …)**.

---

## Shared flag groups (`cli_common.py`)

### Tracing `(shared: trace)`

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| `--verbose` | flag | off | Token + top-k logit traces |
| `--trace-logits` | flag | off | Dump top-k logits |
| `--trace-tokens` | flag | off | Dump token ↔ id |
| `--trace-neurons` | flag | off | Per-layer activation stats |
| `--trace-vectorization` | flag | off | GEMM shapes / CUDA grid |
| `--trace-every` | int | `None` | Every N steps (train default ≈ 10% of steps; generate/interactive default every step) |

### Runtime observability `(shared: obs)` — Stage 3.1, off by default

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| `--runtime-metrics` | flag | off | Extra `[train]` fields; meter host↔device sync |
| `--memory-timeline` | flag | off | ScratchPool JSONL under `output/logs/`; implies metrics |

### Config / seed / checkpoint

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| `--config` | str | `output/configs/training_config.json` | `(shared: config)` |
| `--checkpoint` | str | `output/checkpoints/run1` | `(shared: checkpoint)` |
| `--seed` | int | `42` | `(shared: seed)` |

### Training length `(shared: length)`

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| `--learning-rate` | float | `None` | Prompted if omitted (unless `--no-prompt`) |
| `--epochs` | int | `None` | Ignored if `--steps` set |
| `--steps` | int | `None` | Total steps across all epochs |
| `--log-every` | int | `100` | Progress line cadence |
| `--checkpoint-every` | int | `None` | Save every N steps |
| `--min-lr-ratio` | float | `None` | Cosine LR floor vs base after warmup (default **0.1** in train wiring) |
| `--window-stride` | int | `None` | Stride between sliding windows (preset **64**; else hyperparams or **1**) |
| `--val-every` | int | `0` | Val loss every N steps without quarterly I/O (**0**=off) |
| `--no-prompt` | flag | off | No interactive prompts |

### Model hyperparameters `(shared: model)`

| Flag | Type | Default |
|------|------|---------|
| `--embedding-dim` | int | `None` |
| `--num-heads` | int | `None` |
| `--num-layers` | int | `None` |
| `--max-len` | int | `None` |
| `--dropout` | float | `None` |
| `--batch-size` | int | `None` |
| `--weight-decay` | float | `None` |
| `--warmup-steps` | int | `None` |
| `--gradient-clip` | float | `None` |
| `--grad-accum` / `--gradient-accumulation-steps` | int | `None` (config or 1) |
| `--tie-embeddings` | flag | off (presets default tied) |
| `--no-tie-embeddings` | flag | off |
| `--run-budget` | int | `None` (absolute quarterly budget) |
| `--norm-type` | `layernorm`\|`rmsnorm` | `None` (presets → rmsnorm) |
| `--pos-encoding` | `learned`\|`rope` | `None` (presets → rope) |
| `--grad-checkpoint` | flag | off | VRAM: recompute attn/MLP in backward (**not** tok/s) |
| `--no-grad-checkpoint` | flag | off |

### Generate probes `(shared: probe)`

| Flag | Type | Default |
|------|------|---------|
| `--no-generate-probe` | flag | off |
| `--generate-probe-prompt` | str | `once upon a` |
| `--generate-probe-tokens` | int | `256` |

### Quality trial `(shared: quality)`

| Flag | Type | Default |
|------|------|---------|
| `--quality-trial` | flag | off |
| `--no-quality-trial` | flag | off |
| `--quality-prompt` | str | `None` |
| `--quality-weights` | str | `None` |
| `--compare-quarters` | flag | off |
| `--set-best` | str | `None` (e.g. `quarter_50`) |

---

## Training & generation

### `train.py`

```text
python train.py [flags]
```

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| *(shared: config, checkpoint, seed, length, model, trace, obs, probe, quality)* | | | |
| `--menu` | flag | off | Interactive setup wizard |
| `--data-dir` | str | `data` | Datasets for `--menu` |
| `--models-dir` | str | `output/checkpoints` | Checkpoint scan for menu / `--generate` |
| `--generate` | flag | off | Skip train; generation REPL |
| `--resume` | flag | off | Resume from `--checkpoint` |
| `--temperature` | float | probe default | Probes / `--generate` |
| `--top-k` | int | probe default | |
| `--top-p` | float | probe default | |
| `--plot` | flag | off | Post-train log / landscape plots |

**Examples**

```powershell
# Recommended new TinyStories run (wizard preset 2) — see guide.md
python train.py --menu

python train.py --resume --checkpoint output\checkpoints\BiggerTest256256 --steps 5000 --no-prompt
python train.py --resume --checkpoint output\checkpoints\ts_run --val-every 500 --steps 2000 --no-prompt
python train.py --resume --checkpoint output\checkpoints\BiggerTest256256 --runtime-metrics --memory-timeline --no-prompt --steps 200
python train.py --generate --models-dir output\checkpoints
python train.py --compare-quarters --checkpoint output\checkpoints\BiggerTest256256
```

---

### `auto_train.py`

Train then smoke-generate.

```text
python auto_train.py [flags]
```

| Flag | Type | Default |
|------|------|---------|
| *(shared: config, checkpoint, seed, length, model, probe, quality, trace, obs)* | | |
| `--resume` | flag | off |
| `--prompt` | str | `the` |
| `--max-new-tokens` | int | `80` |
| `--temperature` | float | probe default |
| `--top-k` | int | probe default |
| `--top-p` | float | probe default |
| `--menu` | flag | off |
| `--data-dir` | str | `data` |
| `--models-dir` | str | `output/checkpoints` |

---

### `generate.py`

```text
python generate.py [flags]
```

| Flag | Type | Default | Notes |
|------|------|---------|-------|
| *(shared: checkpoint, seed, trace)* | | | |
| `--prompt` | str | `the` | |
| `--max-new-tokens` | int | `80` | |
| `--temperature` | float | `0.8` | |
| `--top-k` | int | `None` | |
| `--top-p` | float | `None` | |
| `--no-kv-cache` | flag | off | Full recompute each token |

```powershell
python generate.py --checkpoint output\checkpoints\BiggerTest256256 --prompt "once upon a" --max-new-tokens 256 --temperature 0.6 --top-k 10 --top-p 0.9
```

---

### `interactive.py`

REPL; session commands: `:temp`, `:tokens`, `:trace on|off`, `:quit`.

```text
python interactive.py [flags]
```

| Flag | Type | Default |
|------|------|---------|
| *(shared: checkpoint, seed, trace)* | | |
| `--temperature` | float | `0.8` |
| `--max-new-tokens` | int | `80` |
| `--top-k` | int | `None` |
| `--top-p` | float | `None` |

---

### `setup/training_setup.py`

Interactive training setup wizard (also reachable via `train.py --menu`).

```text
python setup/training_setup.py [--data-dir DIR]
```

| Flag | Type | Default |
|------|------|---------|
| `--data-dir` | str | `data` |

---

## Benchmarks (mostly fixed hyperparameters)

### `bench_step.py`

No CLI flags. Benches `minimal` + `tiny_english` with hardcoded small model; requires metrics **disabled**.

```text
python bench_step.py
```

### `bench_profile.py`

No CLI flags. Profiles forward / backward / optimizer / sync split.

```text
python bench_profile.py
```

### `bench_mlp_fusion.py`

No CLI flags. Compares matmul+GELU path vs `fused_mlp_row`.

```text
python bench_mlp_fusion.py
```

### `tools/bench_generate.py`

Stage 3.2 generate latency (KV on vs off).

```text
python tools/bench_generate.py [flags]
```

| Flag | Type | Default |
|------|------|---------|
| `--checkpoint` | str | `None` (tiny random model if omitted) |
| `--prompt` | str | `once upon a time` |
| `--max-new-tokens` | int | `256` |
| `--seed` | int | `42` |
| `--out` | str | `output/baselines/stage32_kv_generate.json` |

```powershell
python tools\bench_generate.py --checkpoint output\checkpoints\BiggerTest256256 --max-new-tokens 256
```

---

## Plotters & logs

### `training_log_plotter.py`

```text
python training_log_plotter.py [flags]
```

| Flag | Type | Default |
|------|------|---------|
| `--logs` | paths… | |
| `--log-dir` | str | `output/logs` |
| `--multi` | flag | off |
| `--all-runs` | flag | off |
| `--keep-short` | flag | off |
| `--min-points` | int | (module default) |
| `--max-step-gap` | int | (module default) |
| `--tail-lines` | int | (module default; `0` = entire file) |
| `--metric` | str | `tok/s` |
| `--smooth-window` | int | `21` |
| `--ema-alpha` | float | `0.08` |
| `--raw-alpha` | float | `0.10` |
| `--forecast-window` | int | `40` |
| `--no-forecast` | flag | off |
| `--forecast-raw` | flag | off |
| `--show-raw-loss` | flag | off |
| `--show-ema-loss` | flag | off |
| `--hide-raw-metric` | flag | off |
| `--select` | flag | off |
| `--live` | flag | off |
| `--refresh-seconds` | float | `1.0` |
| `--save` | path | |
| `--show` | flag | off |

### `loss_landscape_plotter.py`

```text
python loss_landscape_plotter.py [flags]
```

| Flag | Type | Default |
|------|------|---------|
| `--log-dir` | str | `output` |
| `--all-runs` | flag | off |
| `--keep-short` | flag | off |
| `--min-points` | int | (module default) |
| `--max-step-gap` | int | (module default) |
| `--out` | str | `output/logs/loss_landscape_latest.png` |
| `--show` | flag | off |
| `--volatility-window` | int | `25` |

---

## Stage 3 tools

### `tools/tracing/memory_timeline.py`

```text
python -m tools.tracing.memory_timeline -i PATH [--plot] [--plot-out PATH]
# or
python tools/tracing/memory_timeline.py -i PATH ...
```

| Flag | Type | Default |
|------|------|---------|
| `--input` / `-i` | str | **required** |
| `--plot` | flag | off |
| `--plot-out` | str | `output/logs/memory_timeline.png` |

### `tools/tracing/activation_account.py`

Stage 3.4 VRAM attribution.

```text
python tools/tracing/activation_account.py [flags]
```

| Flag | Type | Default |
|------|------|---------|
| `--batch-size` | int | `4` |
| `--context` | int | `256` |
| `--embed` | int | `256` |
| `--layers` | int | `4` |
| `--heads` | int | `8` |
| `--vocab` | int | `110` |
| `--out` | str | `output/baselines/stage34_activation_account.json` |

### `tools/bpe_protocol.py`

Stage 3.3 char vs BPE experiment (does not change BiggerTest default).

```text
python tools/bpe_protocol.py [flags]
```

| Flag | Type | Default |
|------|------|---------|
| `--dataset` | str | `tiny_english` |
| `--num-merges` | int | `150` |
| `--context` | int | `64` |
| `--embed` | int | `64` |
| `--layers` | int | `2` |
| `--heads` | int | `4` |
| `--steps` | int | `3` |
| `--out` | str | `output/baselines/stage33_bpe_protocol.json` |

### `tools/stage3_milestones.py`

Runs Stage 3.4–3.7 measurement artifacts.

```text
python tools/stage3_milestones.py [--stages 34,35,36,37]
```

| Flag | Type | Default |
|------|------|---------|
| `--stages` | str | `34,35,36,37` |

### `tools/reports/evolution_report.py`

```text
python tools/reports/evolution_report.py [--out PATH]
```

| Flag | Type | Default |
|------|------|---------|
| `--out` | str | `output/reports/evolution.html` |

### `tools/releases/make_snapshot.py`

Known-good release snapshot (parity gate included).

```text
python tools/releases/make_snapshot.py [--tag v0.1.1]
```

| Flag | Type | Default |
|------|------|---------|
| `--tag` | str | `v0.1.1` |

Writes `output/releases/<tag>/{runtime,quality,generation,memory}.json`, `parity.txt`, `evolution.html`.

---

## Tests & workspace

### `tests/parity/run_parity.py`

```text
.\venv\Scripts\python.exe -m tests.parity.run_parity
```

No flags. Discovers `tests/parity/test_*.py`. Release gate: **10/10**.

Individual modules (also `unittest`-runnable):

```text
python -m unittest tests.parity.test_kv_cache -v
python -m unittest tests.parity.test_attention tests.parity.test_step -v
```

### `setup/2_test_workspace.py`

No flags. Verifies CUDA 10.1 / PyCUDA / sm_35 toolchain.

```text
python setup/2_test_workspace.py
```

---

## Not standalone CLIs (library / import only)

These are imported by the entry points above; they have no project-facing argparse CLI:

| Path | Role |
|------|------|
| `model/*`, `model/cuda/*` | GPT + kernels / ops / allocator / FP16 storage |
| `training/*` | dataset, loss, checkpoint, optimizer, probe, quality, eval |
| `tokenizer/tokenizer.py`, `tokenizer/bpe.py` | Char + experimental BPE |
| `setup/config_loader.py`, `dataset_setup.py`, `model_config.py`, `training_presets.py`, `weight_init.py` | Setup helpers |
| `tools/tracing/runtime_metrics.py` | SyncMeter / MemoryTimeline / KernelTimeline (enabled via train flags) |
| `cli_common.py`, `paths.py`, `logging_config.py`, `version.py` | Shared utilities |

---

## Quick index

| Command | Purpose |
|---------|---------|
| `train.py` | Train / resume / generate menu / quality |
| `auto_train.py` | Train + smoke generate |
| `generate.py` | One-shot sample (KV on by default) |
| `interactive.py` | Generation REPL |
| `bench_step.py` | Train-step microbench |
| `bench_profile.py` | Fwd/bwd/opt split |
| `bench_mlp_fusion.py` | MLP fusion A/B |
| `tools/bench_generate.py` | Generate KV bench |
| `training_log_plotter.py` | Loss / tok/s charts |
| `loss_landscape_plotter.py` | 3D loss trajectory |
| `tools/tracing/memory_timeline.py` | Summarize ScratchPool JSONL |
| `tools/tracing/activation_account.py` | Activation VRAM buckets |
| `tools/bpe_protocol.py` | Char vs BPE protocol |
| `tools/stage3_milestones.py` | Stages 3.4–3.7 batch |
| `tools/reports/evolution_report.py` | HTML evolution report |
| `tools/releases/make_snapshot.py` | `v0.1.1` release snapshot |
| `python -m tests.parity.run_parity` | Correctness gate (prefer `.\venv\Scripts\python.exe -m tests.parity.run_parity`) |
| [`guide.md`](guide.md) | GT 730 TinyStories fast start |
| `setup/training_setup.py` | Setup wizard |
| `setup/2_test_workspace.py` | CUDA workspace check |
