# llm-gpu-8

**Version:** 0.1.2  
**Repo:** [github.com/dtelcore/llm-gpu-8](https://github.com/dtelcore/llm-gpu-8)  
**Stack:** NumPy + PyCUDA character-level GPT — a compact **transformer runtime** targeting Kepler **GT 730** (CC 3.5)

A from-scratch training workspace for real GPU training on old Kepler hardware where modern PyTorch CUDA builds fail (`no kernel image`). Live path: **GPU forward → GPU loss → GPU manual backward → GPU AdamW**. NumPy remains the reference / fallback for verification.

---

## Table of contents

1. [Why this project](#why-this-project)
2. [Stage 3 — Runtime engineering](#stage-3--runtime-engineering)
3. [Features](#features)
4. [Hardware & software](#hardware--software)
5. [Architecture](#architecture)
6. [Repository layout](#repository-layout)
7. [Setup](#setup)
8. [Quick start](#quick-start)
9. [Training](#training)
10. [Observability (Stage 3.1)](#observability-stage-31)
11. [Verification / parity](#verification--parity)
12. [Generation & sampling](#generation--sampling)
13. [Quarterly checkpoints & quality trial](#quarterly-checkpoints--quality-trial)
14. [Tracing & diagnostics](#tracing--diagnostics)
15. [Plotting & benchmarks](#plotting--benchmarks)
16. [Configuration & presets](#configuration--presets)
17. [Notable run: BiggerTest256256](#notable-run-biggertest256256)
18. [Changelog](#changelog)
19. [Versioning](#versioning)
20. [Related docs](#related-docs)

---

## Why this project

Kepler GT 730 (CC 3.5) is unsupported by current PyTorch CUDA wheels. This repo builds a **minimal, inspectable** GPT training stack on **PyCUDA + custom kernels** so you can:

- Train a character LM on TinyStories-scale text with batch/seq sizes that fit ~4 GB DDR3 (tighter when the GPU also drives a display)
- Inspect logits, tokens, neurons, GEMM launches, **ScratchPool memory**, and **host sync stalls**
- Checkpoint, resume, and compare generation quality at quarterly milestones
- Prove CUDA math against a NumPy reference before aggressive memory/execution changes

Predecessor context and CUDA bring-up notes live in [`setup/cuda_activate.md`](setup/cuda_activate.md) (toolkit pins, MSVC 14.2, `arch=compute_35`).

---

## Stage 3 — Runtime engineering

Stages 1–2 (implement the math; move it onto GPU) are done. **Stage 3** follows:

```text
Measurement → Understanding → Optimization
```

not the reverse. Five pillars:

| Pillar | Question |
|--------|----------|
| **Memory** | How do I move and store less data? |
| **Execution** | How do I execute the same math faster? |
| **Model** | How do I make better use of the same compute? (tokenizer, positions, …) |
| **Observability** | What is the runtime actually doing? |
| **Verification** | Did an optimization preserve correctness? |

### Stage 3 exit — what exists

**Stage 3 is complete** (3.1–3.8 plus closeout 3.9–3.11). Measurement and Kepler-honest optimizations shipped; full AMP GEMM / always-on decode graphs are not viable on sm_35.

Empirical answer from the BiggerTest metrics window (step **111100**):

> Is the GT 730 limited by transfers or by the transformer workload?  
> **Not transfers.** Sync ≈ **0.018%** of step time. ScratchPool peak ≈ **28 MB** of ~**844 MB** device use. Next chapter = **memory efficiency** (activations) and **model efficiency** (KV cache, BPE) — not more CUDA plumbing.

```text
llm-gpu-8 v0.1.2

Exists
├── Runtime
│   ├── CUDA forward / loss / manual backward / GPU AdamW
│   ├── ScratchPool telemetry (pool lifetime)
│   ├── Generate KV cache (prefill + incremental decode)
│   ├── Device FP16↔FP32 cast kernels (storage; math stays FP32)
│   └── CUDA Graph API (GPU-only capture; decode falls back)
├── Observability
│   ├── training metrics (--runtime-metrics)
│   ├── sync timing (to_host / to_device)
│   ├── memory timeline (--memory-timeline)
│   └── process-only VRAM (excludes HDMI/display via baseline/NVML)
├── Verification
│   ├── NumPy reference path
│   └── CUDA parity suite (tests/parity)
└── Scientific control
    ├── output/baselines/stage31_baseline.json
    ├── output/baselines/stage32_kv_generate.json
    ├── output/baselines/stage33_bpe_protocol.json
    ├── output/baselines/stage34_activation_account.json
    ├── output/baselines/stage35_fp16_storage.json
    ├── output/baselines/stage36_allocator.json
    ├── output/baselines/stage37_timeline.json
    ├── output/baselines/stage311_cuda_graph.json
    └── output/reports/evolution.html

Exists (Stage 3.2–3.11)
├── Generate KV cache + tools/bench_generate.py
├── tokenizer/bpe.py + factory (BPE default for new runs; char via --tokenizer char)
├── Activation accounting (tools/tracing/activation_account.py)
├── FP16 activation storage + device casts (model/cuda/fp16_storage.py)
├── LifetimeAllocator (model/cuda/allocator.py)
├── Software kernel timeline (tools/tracing/runtime_metrics.kernel_timeline)
├── Evolution HTML report (tools/reports/evolution_report.py)
├── Process-only VRAM (model/cuda/ops.get_memory_usage)
└── CUDA Graph capture/fallback (model/cuda/graph.py; generate --cuda-graph)

Exists (Stage 4)
├── Device KV arenas sized to GPTConfig.max_len (model/cuda/kv_cache.py)
├── causal_mha_decode_fp32 + device KV append / embed
├── Device argmax + fixed-k top-k sampling (top-p stays host)
├── CUDA Graph kernel-chain capture (append → decode attn → argmax)
└── Baseline: output/baselines/stage4_gpu_kv_decode.json

Exists (tokenizer + menus)
├── BPE default for new train/auto_train runs (--tokenizer char to opt out)
├── train/auto_train/generate --menu flag-group picker (Enter = defaults)
└── Checkpoint vocab load dispatches char vs type=bpe

Not yet (Stage 4 follow-ons)
├── full transformer decode inside one CUDA Graph (GEMM/norm still default-stream)
└── device multinomial / top-p sampling (stochastic path remains host)
```

**Modern block (shipped with this train of work):** `norm_type=rmsnorm`, `pos_encoding=rope`, optional `--grad-checkpoint` (**VRAM** lever: selective attn/MLP recompute; ~72 MB less activation cache at BiggerTest shapes — not a tok/s speedup; see [`output/baselines/modern_activation_recompute.json`](output/baselines/modern_activation_recompute.json)). Legacy checkpoints without these keys stay LayerNorm + learned positions. RoPE path uses **QKV-split fusion** again (no full `[B·T, 3C]` buffer).

**Scientific control (BiggerTest telemetry, not a contended microbench):**

[`output/baselines/stage31_baseline.json`](output/baselines/stage31_baseline.json)

| Signal @ step 111100 | Value |
|----------------------|------:|
| tok/s | 586 |
| step_ms | 1747.9 |
| device_used_mb | 844 |
| scratch_peak_mb | 28.1 |
| sync_ms (window) | 64.7 / 200 steps ≈ 0.32 ms/step |
| grad_norm / param_norm | ≈ 0.029 |
| train loss / ppl | 0.9305 / 2.54 |

**Evidence-backed demotions (do not prioritize next):**

- Further **weight-sync** / PCIe transfer work — sync ≪ step time  
- **ScratchPool redesign** — peak ~28 MB; activations dominate VRAM  

Principle for later milestones: every change must answer (1) Did the runtime get better? (2) Did correctness hold?

**Next:** Stage 4 follow-ons (full-decode graph, device top-p) and longer BPE BiggerTest validation against the v0.1.2 / stage4 baselines.

### Stabilization release discipline

```text
Change → Parity → Benchmark → Baseline comparison → Longer training validation → Release
```

**Known-good snapshot:** [`output/releases/v0.1.2/`](output/releases/v0.1.2/)  
Rebuild with: `python tools/releases/make_snapshot.py --tag v0.1.2`

| Gate | Requirement |
|------|-------------|
| Correctness | Parity suite **OK** (blocker; live test count in `parity.txt`) |
| Runtime | No material regression vs stage31 train / stage32 generate |
| Memory | Compare stage34/35/36 artifacts (activations, FP16, allocator reuse); `device_used_mb` is process-only |
| Observability | Timeline sample + evolution report present |

### Architecture (as shipped)

```text
Training:   Dataset → Tokenizer → GPU forward → GPU loss → GPU manual backward → GPU AdamW → Checkpoint
Generation: Prompt → Tokenizer → Transformer prefill → KV cache → Incremental decode
```

| Pillar | Current state |
|--------|----------------|
| Runtime | CUDA transformer engine + CUDA Graph API (decode fallback) |
| Memory | process-only VRAM + FP16 device casts + lifetime reuse |
| Model | GPT + BPE default tokenizer (char opt-in) |
| Observability | metrics, timeline, reports |
| Verification | NumPy reference + parity |

### Runtime modes

```text
Training:
  GPU forward → loss → backward → AdamW

Generation:
  Prompt encoding
  → transformer prefill
  → KV cache
  → incremental decode
```

KV cache is generate-only (default on; `--no-kv-cache` to disable). Training path unchanged.

Future package map (**docs only** until built):

```text
Shipped:  tools/tracing/   tests/parity/   tools/bench_generate.py
          tokenizer/bpe.py   model/cuda/graph.py   model/cuda/fp16_storage.py
          output/baselines/  output/releases/v0.1.2/
Future:   training/allocator.py (ScratchPool redesign still demoted)
```

---

## Features

| Area | What you get |
|------|----------------|
| Model | Character GPT: embeddings → N× (norm → MHA → residual → norm → MLP → residual) → norm → LM head; defaults **RMSNorm + RoPE** (legacy: LayerNorm + learned pos) |
| GPU path | Tiled GEMM, fused residual norm, causal MHA, **GPU manual backward**, GPU AdamW; optional `--grad-checkpoint` |
| Host path | NumPy analytic backward as **reference / fallback**; persistent device weight mirror |
| Training | Wizard, resume, step/epoch overrides, cosine LR + warmup, val 90/10 holdout |
| Checkpoints | Latest + `quarter_*` + optional `best/` |
| Quality | Heuristic generation scores + quarter trial |
| Observability | `--runtime-metrics`, `--memory-timeline`, kernel timeline, process-only VRAM (off by default) |
| Verification | `.\venv\Scripts\python.exe -m tests.parity.run_parity` |
| Generation | KV cache on device arenas by default (`--no-kv-cache`); `--cuda-graph` captures KV kernel-chain |
| Reports | `python tools/reports/evolution_report.py` → `output/reports/evolution.html` |

---

## Hardware & software

| Component | Typical / target |
|-----------|------------------|
| GPU | NVIDIA GeForce GT 730 (GK208), **CC 3.5** |
| VRAM | ~4 GB DDR3 (less free when shared with the display) |
| Driver | 475.14 era (CUDA runtime ≤ 11.4) |
| Toolkit | CUDA **10.1** (with MSVC 14.2 / VS Build Tools for PyCUDA compile) |
| Python | **3.8** venv at project `venv/` |
| OS | Windows 10 (WDDM) |

Larger configs (e.g. 256d / 8 heads / 4 layers / T=256, batch 4) run at ~600–630 tok/s and ~840 MB device use on this card in practice.

---

## Architecture

```text
                    GPTModel (live path)

                          │
                          ▼
                    GPU forward
                          │
                          ▼
                      GPU loss
                          │
                          ▼
               GPU manual backward
                          │
                          ▼
                     GPU AdamW
                          │
                          ▼
              checkpoints under output/checkpoints/<run>/


NumPy backward path = reference implementation / testing fallback
```

**Design notes**

- Weights stay GPU-resident between steps (`upload_to_device` / `sync_device`).
- `ScratchPool` reuses temporary GPU buffers (pool lifetime until `clear()` — see Observability).
- New presets default to **RMSNorm + RoPE**; older checkpoints without those keys keep LayerNorm + learned positions.
- Character vocab (~110 for TinyStories ASCII) keeps embedding/LM-head matrices small on Kepler.

---

## Repository layout

```text
llm gpu 8/
├── README.md                 ← this file
├── guide.md                  ← GT 730 TinyStories fast start
├── VERSION / version.py      ← 0.1.2
├── train.py / auto_train.py / generate.py / interactive.py
├── cli_common.py / paths.py / logging_config.py
├── tools/
│   ├── tracing/              ← metrics, memory timeline, activation account
│   ├── reports/              ← evolution HTML report
│   ├── releases/             ← make_snapshot.py (v0.1.2 known-good pack)
│   └── bench_generate.py     ← KV generate microbench
├── tests/parity/             ← NumPy↔CUDA verification
├── model/                    ← GPT + CUDA kernels/ops (fp16_storage, graph, …)
├── training/                 ← checkpoint, loss, AdamWGPU, quality, …
├── tokenizer/                ← BPE default (factory) + char opt-in
├── setup/ / data/
└── output/                   ← runtime artifacts (baselines/releases tracked)
```

Runtime artifact convention is documented in [`output/README.md`](output/README.md). Setup details: [`setup/README.md`](setup/README.md).

---

## Setup

1. **Admin once:** run [`setup/1_new_workspace_setup.ps1`](setup/1_new_workspace_setup.ps1) (venv, CUDA 10.1 paths, MSVC 14.2 for PyCUDA).
2. Activate venv:
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
3. Smoke the workspace:
   ```powershell
   python setup\2_test_workspace.py
   ```
4. Put corpora in `data/` (e.g. TinyStories `.txt`). `data/` and `output/` are gitignored.

See [`setup/cuda_activate.md`](setup/cuda_activate.md) if kernels fail to compile or load on CC 3.5.

---

## Quick start

**GT 730 TinyStories (recommended):** see **[`guide.md`](guide.md)** for copy-paste steps. Short version:

```powershell
.\venv\Scripts\Activate.ps1
python train.py --menu
# Scaling preset 2 (Tiny Stories); put .txt corpora in data/ first
```

**Interactive train (wizard):**

```powershell
python train.py --menu
```

**Non-interactive short run** (only after you have a saved config from the wizard, or when resuming):

```powershell
python train.py --config output\configs\training_config.json `
  --checkpoint output\checkpoints\run1 `
  --steps 500 --no-prompt
```

Do **not** use legacy examples like `--learning-rate 0.0001` for new TinyStories runs — use preset **2** or the recipe in [Configuration & presets](#configuration--presets).

**Train then smoke-generate:**

```powershell
python auto_train.py --menu
# or
python auto_train.py --steps 500 --prompt "once upon a" --no-prompt
```

**Generate from a checkpoint:**

```powershell
python generate.py --checkpoint output\checkpoints\BiggerTest256256 `
  --prompt "once upon a" --max-new-tokens 256 `
  --temperature 0.6 --top-k 10 --top-p 0.9
```

---

## Training

### Entry points

| Script | Role |
|--------|------|
| `train.py` | Full loop: config/wizard → train → quarterly saves → optional quality trial / plots |
| `auto_train.py` | Same training path, then one generate sample |

### Useful flags

| Flag | Meaning |
|------|---------|
| `--menu` | Interactive model/dataset/hyperparam wizard (presets: toy / TinyStories / custom) |
| `--resume` | Continue from `--checkpoint` (run root, `best/`, or `quarter_*`) |
| `--steps N` | Total absolute step target (overrides epochs) |
| `--learning-rate` | Override AdamW base LR (preset **2** default **3e-4**) |
| `--warmup-steps` | Linear LR warmup length (preset **1000**) |
| `--min-lr-ratio` | Cosine floor as fraction of base LR after warmup (default **0.1**) |
| `--batch-size` / `--embedding-dim` / `--num-heads` / `--num-layers` / `--max-len` | Model & batch overrides |
| `--grad-accum N` | Micro-batches per optimizer step (gradient accumulation; preset **4**) |
| `--window-stride` | Token stride between sliding windows (preset **64**; `1` = dense / legacy) |
| `--val-every N` | Lightweight val loss every N steps, no quarterly checkpoint I/O (**0** = off; try **500**) |
| `--run-budget N` | Absolute step budget for quarterly 25/50/75/100% markers |
| `--tie-embeddings` / `--no-tie-embeddings` | Share or split token_embedding ↔ lm_head |
| `--norm-type layernorm\|rmsnorm` | Normalization (new presets default rmsnorm) |
| `--pos-encoding learned\|rope` | Positions (new presets default rope) |
| `--grad-checkpoint` / `--no-grad-checkpoint` | Save VRAM by recomputing attn/MLP in backward (**not** a speed lever) |
| `--dropout` | Override `dropout_prob` — values **> 0** log a warning (dropout not implemented) |
| `--log-every` / `--checkpoint-every` | Progress & disk cadence |
| `--no-prompt` | Never ask on stdin; use CLI/config defaults |
| `--compare-quarters` | Skip train; run quality trial on a run dir |
| `--set-best quarter_50` | Promote a quarter to `best/` |
| `--quality-trial` / `--no-quality-trial` | Force / suppress post-train quality trial |
| `--plot` | Render training / landscape plots after train |
| `--runtime-metrics` | Stage 3.1: log `grad_norm` / `param_norm` / `sync_*` / `scratch_peak_mb` (off by default) |
| `--memory-timeline` | Stage 3.1: ScratchPool alloc/reuse JSONL + implies `--runtime-metrics` |

Resume example toward a long target (architecture/LR schedule come from checkpoint + CLI overrides):

```powershell
python train.py --resume --checkpoint output\checkpoints\BiggerTest256256 `
  --steps 120000 --no-prompt
```

For **new** runs on the 2026 recipe, prefer `train.py --menu` → preset **2** or [`guide.md`](guide.md) instead of copying BiggerTest fine-tune LR (`1e-5`).

---

## Observability (Stage 3.1)

**Off by default** — BiggerTest throughput must not change when flags are absent (no extra CUDA syncs, no JSONL I/O).

```powershell
python train.py --config ... --checkpoint output\checkpoints\run1 `
  --steps 20 --no-prompt --runtime-metrics

python train.py ... --memory-timeline
# writes output/logs/memory_timeline_<run>.jsonl

python -m tools.tracing.memory_timeline --input output\logs\memory_timeline_run.jsonl
python -m tools.tracing.memory_timeline --input ... --plot
# optional PNG: output/logs/memory_timeline.png
```

Extended `[train]` keys when metrics are on: `grad_norm`, `param_norm`, `sync_count`, `sync_ms`, `scratch_peak_mb`.

**Limitation:** the Memory Timeline shows **ScratchPool lifetime** (buffers live until `clear()`), not per-activation free/reuse of a future arena allocator.

---

## Verification / parity

NumPy is the reference; CUDA is the device under test.

```powershell
.\venv\Scripts\python.exe -m tests.parity.run_parity
```

Progression: linear → LayerNorm → GELU → attention → one full train step. Tolerances: `rtol=1e-4`, `atol=1e-5`; NaN/Inf fail first. Tiny shapes keep display-shared VRAM safe.

---

## Generation & sampling

| Script | Role |
|--------|------|
| `generate.py` | One-shot sample |
| `interactive.py` | Prompt loop |
| `train.py --generate` | Pick a checkpoint from `--models-dir` and open a generation menu |

Shared sampling knobs: `--temperature`, `--top-k`, `--top-p`, `--max-new-tokens` / probe token count, `--seed`.

Mid-training **generate probes** fire at 25% / 50% / 75% / 100% of `total_steps` (unless `--no-generate-probe`). Defaults align with quality trial: prompt `once upon a`, temp `0.6`, top-k `10`, top-p `0.9`, 256 new tokens.

---

## Quarterly checkpoints & quality trial

Introduced in **v0.1.0** (`a2b1a6f`).

### On-disk layout (per run)

```text
output/checkpoints/<run>/
  weights.npz  config.json  state.json  vocab.json  corpus.json  metrics.json
  val_corpus.json          # holdout sidecar when present
  quarter_25/ … quarter_100/
  best/                    # only after promote
```

At each quarterly step the trainer:

1. Syncs GPU → host  
2. Saves **latest** (run root) **and** `quarter_XX/`  
3. Evaluates **val_loss / val_ppl** (10% seeded holdout)  
4. Forces full traces + probe + generate probe  
5. Scores generation into `metrics.json` / logs  

`--checkpoint-every` still updates **latest only**.

### Quality trial

```powershell
python train.py --compare-quarters --checkpoint output\checkpoints\BiggerTest256256 --no-prompt
```

Scores (0–1): **spelling**, **punctuation**, **grammar**, **semantics** → weighted **aggregate**. Optionally promote one quarter to `best/` interactively, or:

```powershell
python train.py --set-best quarter_100 --checkpoint output\checkpoints\BiggerTest256256
```

**Note:** Quarterly 25/50/75/100% markers use an absolute **`run_budget_steps`**
(persisted in `state.json`, overridable with `--run-budget`). Chunked `--steps`
resumes no longer treat each chunk end as `quarter_100`. On first run without
`--run-budget`, the session `total_steps` becomes the budget.

---

## Tracing & diagnostics

All quiet by default. Shared via `cli_common`:

| Flag | Effect |
|------|--------|
| `--trace-tokens` | Token ↔ id dumps |
| `--trace-logits` | Top-k logits / probs |
| `--trace-neurons` | Per-layer activation stats |
| `--trace-vectorization` | GEMM shapes / launch geometry |
| `--trace-every N` | Cadence (train default ~10% of total steps; generate defaults every step) |
| `--verbose` | Tokens + top-k logits together |

Quarterly milestones temporarily force a full tracer regardless of CLI cadence.

---

## Plotting & benchmarks

```powershell
# Live / recent-window training curves (default: last ~1000 steps)
python training_log_plotter.py
python training_log_plotter.py --tail-lines 0          # full log
python training_log_plotter.py --save                 # PNG, no GUI

python loss_landscape_plotter.py

python bench_step.py          # GPU step microbench + forward→backward contract smoke
python bench_profile.py
python bench_mlp_fusion.py
```

`bench_step.py` uses the batched GPU path (`forward_batch` / `backward_batch_gpu` / `AdamWGPU`) and asserts `forward()` keeps `B`/`T` on the cache for `backward()`.

Logs default under `output/logs/` (`training.log` aggregate + per-run files).

---

## Configuration & presets

- Default config path: `output/configs/training_config.json` (`paths.DEFAULT_CONFIG_PATH`)
- Wizard / presets: `setup/training_setup.py`, `setup/training_presets.py`, `setup/model_config.py`
- **Do not commit** configs that embed full corpora — `setup/training_config.json` and `output/configs/training_config.json` are gitignored
- **GT 730 TinyStories recipe (2026, preset `tiny_stories`):** `C=256`, `L=4`, 8 heads, `T=128`, rmsnorm + RoPE + tied embeddings, `dropout_prob=0`. Training: `LR=3e-4`, warmup **1000**, cosine to `min_lr_ratio=0.1` (`--min-lr-ratio`), `batch=4`, grad accum **4**, `window_stride=64` (`--window-stride`). Strided windows mean **fewer windows per epoch** than dense stride-1. Dropout is **not implemented** on GPU — keep **0** (nonzero only warns). For BiggerTest-style context, A/B with `--max-len 256`. Use `--val-every 500` for cheap mid-run val. Turn **off** `--grad-checkpoint` for tok/s baselines; turn **off** `--runtime-metrics` unless measuring. Step-by-step: [`guide.md`](guide.md).

---

## Notable run: BiggerTest256256

Long-running Kepler train used as the project’s stress reference:

| Setting | Value |
|---------|--------|
| Name | `output/checkpoints/BiggerTest256256` |
| Model | embed 256, 8 heads, 4 layers, max_len **256**, dropout **0.1 in config (never applied — no dropout kernels)** |
| Data | TinyStories character corpus (~558k train / ~62k val sentences), vocab ~110 |
| Batch / LR (late fine-tune) | 4 / **`1e-5`** (historical; new preset uses **3e-4** + cosine — see [`guide.md`](guide.md)) |
| Long-run target | **120,000** steps (first logged 2026-07-15) |
| Stage 3.1 control | step **111100**, train loss **0.9305**, ppl **2.54**, **~586 tok/s** — see [`stage31_baseline.json`](output/baselines/stage31_baseline.json) |
| Quality @ 101k (quarter) | aggregate **~0.884** (spell 0.93 / punct 0.75 / gram 1.0 / sem 0.85); val loss ~0.96 |

Generation at this stage: reliable TinyStories openers; mid-sample coherence still soft — expected for char-level ~3M-param Kepler training.

---

## Changelog

### [0.1.2+] — 2026-08-01 — BPE default + `--menu` flag groups

- **BPE default:** [`tokenizer/factory.py`](tokenizer/factory.py) — new runs use BPE (`--tokenizer bpe`, `--bpe-merges 200`); char via `--tokenizer char`; resume loads checkpoint vocab type
- **Menu flag picker:** `train.py` / `auto_train.py` / `generate.py --menu` — pick groups to customize; Enter keeps on/off defaults ([`cli_common.prompt_run_flag_menu`](cli_common.py))
- **generate.py --menu:** checkpoint picker + Sampling / Decode / Trace groups

### [0.1.2+] — 2026-07-31 — Stage 4 GPU-resident KV decode

- **S4.1 Device KV arenas:** [`model/cuda/kv_cache.py`](model/cuda/kv_cache.py) — `[B*H, max_len, hd]` packed on prefill
- **S4.2 Decode attention:** `causal_mha_decode_fp32` — no host attention on the generate hot path
- **S4.3 Append + embed:** device row write at `T` + `embedding_lookup_tokens`
- **S4.4 Device sample:** argmax + fixed-k top-k; top-p/RNG remain host
- **S4.5 Graph:** kernel-chain capture (`kv_append` → decode → argmax); [`stage4_gpu_kv_decode.json`](output/baselines/stage4_gpu_kv_decode.json)

### [0.1.2] — 2026-07-31 — Stage 3 closeout

- **3.9 Process-only VRAM:** `ops.get_memory_usage()` / `process_used_mb()` — baseline (post-context) or NVML per-PID; train/bench logs `device_used_mb` exclude HDMI/display; optional `vram_driver_used_mb`
- **3.10 Device FP16 casts:** `float_to_half` / `half_to_float` kernels; `fp16_storage` no longer host round-trips; compute remains FP32
- **3.11 CUDA Graph:** [`model/cuda/graph.py`](model/cuda/graph.py) probe/capture/replay; GPU-only cast graph works; host KV decode falls back; `generate.py --cuda-graph`; [`stage311_cuda_graph.json`](output/baselines/stage311_cuda_graph.json)
- **Release snapshot:** [`output/releases/v0.1.2/`](output/releases/v0.1.2/)

### [0.1.1-dev] — 2026-07-30 — Train recipe + throughput hygiene

- **`tiny_stories` preset:** C=256, L=4, T=128, ~3M params; LR **3e-4**, warmup **1000**, batch **4**, grad accum **4**, cosine LR (`min_lr_ratio=0.1`), **`window_stride=64`**, dropout **0** (warn if >0)
- **CLI:** `--min-lr-ratio`, `--window-stride`, `--val-every` (lightweight val without quarterly I/O)
- **Optimizer:** cosine schedule after warmup in `AdamWGPU`; residual out-proj init **1/√(2L)**
- **Throughput:** train GPU path skips host logits sync when only device CE is needed; RoPE attention restores QKV-split fusion
- **Docs:** [`guide.md`](guide.md) GT 730 quick start; README / `py_calls.md` aligned with recipe

### [0.1.1-dev] — 2026-07-22 — Stage 3.2–3.8 serial roadmap

- **3.2 KV cache** (generate-only): prefill + incremental decode; `tools/bench_generate.py`; BiggerTest **~3.9×** generate speedup @ 256 tokens; [`stage32_kv_generate.json`](output/baselines/stage32_kv_generate.json)
- **3.3 BPE experiments:** [`tokenizer/bpe.py`](tokenizer/bpe.py) + [`tools/bpe_protocol.py`](tools/bpe_protocol.py); char remains BiggerTest default
- **3.4 Activation accounting:** attention_cache largest bucket (~48 MB of ~99 MB cache @ BiggerTest shapes)
- **3.5 FP16 activation storage** + FP32 compute cast (`model/cuda/fp16_storage.py`)
- **3.6 LifetimeAllocator** for qkv_split temps (`model/cuda/allocator.py`)
- **3.7 Software kernel timeline** (`kernel_timeline` in runtime_metrics)
- **3.8 Evolution report:** [`output/reports/evolution.html`](output/reports/evolution.html)

### [0.1.1-dev] — 2026-07-22 — Stage 3.1 Observability + Verification (exit)

Measurement and correctness foundation (does **not** optimize execution):

- `tools/tracing/runtime_metrics.py` — SyncMeter + MemoryTimeline (disabled by default)
- ScratchPool alloc/reuse/clear instrumentation; `--memory-timeline` JSONL
- Richer `[train]` fields: `grad_norm`, `param_norm`, `sync_count`, `sync_ms`, `scratch_peak_mb`
- `tools/tracing/memory_timeline.py` summary CLI + optional `--plot`
- `tests/parity/` unittest harness (linear → LN → GELU → attention → full step + forward cache contract)
- **Cache contract:** `forward()` squeezes logits only; cache keeps `B`/`T`/`batched` for GPU backward
- **`bench_step.py`** repaired for batched GPU path + forward→backward smoke
- **Baseline freeze:** [`output/baselines/stage31_baseline.json`](output/baselines/stage31_baseline.json) (BiggerTest telemetry @ 111100; GT 730 not transfer-bound)
- README: five pillars, exists vs Not yet, demotions (weight-sync / ScratchPool redesign)
- Corrected GPU backward architecture docs (live path is GPU manual backward + AdamWGPU)

### [0.1.0] — 2026-07-21

Quarterly milestone pipeline, version stamp, val metrics, quality trial.

| Commit | Date | Summary |
|--------|------|---------|
| `a2b1a6f` | 2026-07-21 | **v0.1.0:** quarterly checkpoints (`quarter_*`), `metrics.json`, val loss/PPL, generate probes with full traces, `training/quality.py` trial + `--compare-quarters` / `--set-best`, resume discovery for nested dirs, `VERSION` / `version.py` |
| `b0d713d` | 2026-07-21 | Align generation defaults (temp / top-k / top-p) across training probe paths |
| `26073fa` | 2026-07-21 | Training log plotter: default to last 1000 lines for faster load / recent-scale axes |
| `7fba7da` | 2026-07-21 | Plotter sliding window: keep latest contiguous step segment; live-refresh friendly docs |

### [0.0.x] — Pre-release (2026-07-12 → 2026-07-19)

Treated as `0.0.x`–`0.9.9` relative to `version.py` policy. Building blocks before formal 0.1.0.

#### Generation & UX

| Commit | Date | Summary |
|--------|------|---------|
| `68c809a` | 2026-07-19 | Top-k and top-p sampling on generation scripts |
| `5dc5830` | 2026-07-16 | Migrate runtime layout to `output/` (logs, checkpoints, configs, tokenizer); logging path updates |

#### GPU kernels & full device training path

| Commit | Date | Summary |
|--------|------|---------|
| `3fdb142` | 2026-07-15 | GPT + CUDA attention/MLP refactors and further op optimizations |
| `a3a7043` | 2026-07-15 | Fused residual LayerNorm; GPU op optimizations |
| `2921787` | 2026-07-15 | Tiled GEMM + reusable scratch buffers |
| `e9fb039` | 2026-07-14 | Attention refactor; GPU op optimizations |
| `86f5476` | 2026-07-14 | **Full GPU training path:** device forward/backward/loss/AdamW, fused attention; `training/gpu_optimizer.py`, benches; gitignore oversized configs so corpora are not embedded in commits |

#### CLI, resume, plots, V2 weights

| Commit | Date | Summary |
|--------|------|---------|
| `959f7e2` | 2026-07-12 | Interactive checkpoint resume; `--data-dir` / `--models-dir`; wizard resume; plotter parse cache; ignore `.npz` |
| `d0d715f` | 2026-07-12 | **V2 architecture:** persistent GPU-resident weight mirror (`ModelParameters`); sync once per optimizer step; training log plotter + loss landscape plotter land |
| `9447cb9` | 2026-07-12 | Shared training-length CLI (`--steps`, `--log-every`, …); dataset path prompts; tracing defaults cleanup |

#### Initial stack

| Commit | Date | Summary |
|--------|------|---------|
| `cbcdc62` | 2026-07-12 | Complete NumPy + PyCUDA GPT stack: tokenizer, model, CUDA ops/kernels, train loop, checkpoint/dataset/loss/optimizer/probe, `generate.py` / `interactive.py` / `auto_train.py` / `cli_common.py` |
| `ed008b2` | 2026-07-12 | Implementing setup plan |
| `d0d1e0a` | 2026-07-12 | **Init:** setup system (model/dataset/weight-init/training wizard), logging, workspace PowerShell setup, `setup/README.md`, CUDA activate journey notes |

### Commit timeline (compact)

```text
2026-07-12  d0d1e0a  Init + setup system
2026-07-12  ed008b2  Setup plan implementation
2026-07-12  cbcdc62  Full NumPy/PyCUDA GPT train stack
2026-07-12  9447cb9  Training-length CLI + dataset UX
2026-07-12  d0d715f  V2 GPU-resident weights + plotters
2026-07-12  959f7e2  Resume wizard + CLI dirs
2026-07-14  86f5476  Full GPU train path + gitignore corpora configs
2026-07-14  e9fb039  Attention / GPU ops
2026-07-15  2921787  Tiled GEMM + scratch
2026-07-15  a3a7043  Fused residual LN
2026-07-15  3fdb142  GPT/CUDA attention+MLP polish
2026-07-16  5dc5830  output/ directory migration
2026-07-19  68c809a  top-k / top-p sampling
2026-07-21  b0d713d  Generation param alignment
2026-07-21  a2b1a6f  ★ v0.1.0 quarterly + quality trial
2026-07-21  26073fa  Plotter last-1000-lines default
2026-07-21  7fba7da  Plotter sliding contiguous window
2026-07-22  995e175  Stage 3.1 observability + parity foundation
2026-07-29  e19d3f7  Stage 3.2–3.8 runtime/memory/report path
2026-07-30  9ae9632  RMSNorm / RoPE / modern-block CLI
2026-07-30  cbd093c  Train recipe (cosine LR, window stride, val-every)
2026-07-31  52f5e18  ★ v0.1.2 Stage 3 closeout (process VRAM, FP16 casts, CUDA Graph)
```

---

## Versioning

- Canonical version string: [`VERSION`](VERSION) (currently `0.1.2`), exposed as `version.py` → `__version__`.
- Printed at startup by `train.py` / `auto_train.py`; stamped into checkpoint `config.json` / `state.json` / `metrics.json`.
- **Policy:** all work before `a2b1a6f` is pre-0.1.0. Stage 3 closeout ships as **`0.1.2`**.

---

## Related docs

| Doc | Topic |
|-----|--------|
| [`guide.md`](guide.md) | **GT 730 fast start** — TinyStories preset 2, verify, pitfalls |
| [`setup/README.md`](setup/README.md) | Model/dataset/init/hyperparam wizard deep dive |
| [`setup/cuda_activate.md`](setup/cuda_activate.md) | GT 730 CUDA / PyCUDA activation journey |
| [`output/README.md`](output/README.md) | Runtime artifact directories |
| [`.cursor/plans/train_recipe_speed_33e6e93c.plan.md`](.cursor/plans/train_recipe_speed_33e6e93c.plan.md) | Train recipe + throughput plan (completed) |
| [`.cursor/plans/obs_verify_foundation_da50b350.plan.md`](.cursor/plans/obs_verify_foundation_da50b350.plan.md) | Stage 3.1 Observability + Verification |
| [`.cursor/plans/quarterly_checkpoint_resume_f982639b.plan.md`](.cursor/plans/quarterly_checkpoint_resume_f982639b.plan.md) | v0.1.0 quarterly design |
| [`.cursor/plans/pycuda_gpt_training_8d816e55.plan.md`](.cursor/plans/pycuda_gpt_training_8d816e55.plan.md) | Original stack plan |
| [`.cursor/plans/port_gpu5_cuda_core_ce814de3.plan.md`](.cursor/plans/port_gpu5_cuda_core_ce814de3.plan.md) | GPU-5 CUDA core port notes |
| [`.cursor/plans/colab_t4_port_a1845e09.plan.md`](.cursor/plans/colab_t4_port_a1845e09.plan.md) | Colab T4 port notes |

---

## License / authorship

Personal training research workspace (`dtelcore/llm-gpu-8`). Commit history authored primarily by **darren hoyer** (2026-07-12 onward).
