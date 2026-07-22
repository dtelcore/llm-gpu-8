# llm-gpu-8

**Version:** 0.1.0  
**Repo:** [github.com/dtelcore/llm-gpu-8](https://github.com/dtelcore/llm-gpu-8)  
**Stack:** NumPy + PyCUDA character-level GPT, trained end-to-end on a Kepler **GT 730** (CC 3.5)

A from-scratch training workspace aimed at real GPU training on old Kepler hardware where modern PyTorch CUDA builds fail (`no kernel image`). Forward, loss, and AdamW run on device; backward is analytic NumPy with a persistent GPU weight mirror so PCIe is not flooded every matmul.

---

## Table of contents

1. [Why this project](#why-this-project)
2. [Features (v0.1.0)](#features-v010)
3. [Hardware & software](#hardware--software)
4. [Architecture](#architecture)
5. [Repository layout](#repository-layout)
6. [Setup](#setup)
7. [Quick start](#quick-start)
8. [Training](#training)
9. [Generation & sampling](#generation--sampling)
10. [Quarterly checkpoints & quality trial](#quarterly-checkpoints--quality-trial)
11. [Tracing & diagnostics](#tracing--diagnostics)
12. [Plotting & benchmarks](#plotting--benchmarks)
13. [Configuration & presets](#configuration--presets)
14. [Notable run: BiggerTest256256](#notable-run-biggertest256256)
15. [Changelog](#changelog)
16. [Versioning](#versioning)
17. [Related docs](#related-docs)

---

## Why this project

Kepler GT 730 (CC 3.5) is unsupported by current PyTorch CUDA wheels. This repo builds a **minimal, inspectable** GPT training stack on **PyCUDA + custom kernels** so you can:

- Train a character LM on TinyStories-scale text with batch/seq sizes that fit ~4 GB DDR3
- Inspect logits, tokens, neurons, and GEMM launches mid-run
- Checkpoint, resume, and compare generation quality at quarterly milestones
- Keep all runtime artifacts under a single `output/` tree

Predecessor context and CUDA bring-up notes live in [`setup/cuda_activate.md`](setup/cuda_activate.md) (toolkit pins, MSVC 14.2, `arch=compute_35`).

---

## Features (v0.1.0)

| Area | What you get |
|------|----------------|
| Model | Character GPT: embeddings → N× (LN → MHA → residual → LN → MLP → residual) → LN → LM head |
| GPU path | Fused / tiled GEMM, fused residual LayerNorm, fused causal MHA path, GPU AdamW |
| Host path | Analytic NumPy backward; `ModelParameters` keeps a persistent device weight mirror |
| Training | Interactive wizard (`--menu`), resume, step/epoch overrides, val 90/10 holdout |
| Checkpoints | Run-root “latest” + `quarter_25/50/75/100/` + optional `best/` promotion |
| Quality | Heuristic spelling / punctuation / grammar / semantics scores + sequential quarter trial |
| Sampling | Temperature, top-k, top-p; mid-training generate probes |
| Tooling | Log plotter (sliding window), loss-landscape plotter, step/MLP/profile benches |

---

## Hardware & software

| Component | Typical / target |
|-----------|------------------|
| GPU | NVIDIA GeForce GT 730 (GK208), **CC 3.5** |
| VRAM | ~4 GB DDR3 |
| Driver | 475.14 era (CUDA runtime ≤ 11.4) |
| Toolkit | CUDA **10.1** (with MSVC 14.2 / VS Build Tools for PyCUDA compile) |
| Python | **3.8** venv at project `venv/` |
| OS | Windows 10 (WDDM) |

Larger configs (e.g. 256d / 8 heads / 4 layers / T=256, batch 4) run at ~600–630 tok/s and ~840 MB device use on this card in practice.

---

## Architecture

```text
Corpus (.txt / wizard) ──► CharacterGPTTokenizer
                              │
                              ▼
                     WindowedDataset (train)
                     + 10% seeded val holdout
                              │
                              ▼
              GPTModel (PyCUDA forward + NumPy backward)
                              │
         ModelParameters: host NumPy + device gpuarray mirror
                              │
                    AdamWGPU ── sync_device() once / step
                              │
         checkpoints under output/checkpoints/<run>/
```

**Design notes**

- Weights stay GPU-resident between steps (`upload_to_device` / `sync_device`) — V2 architecture (see changelog 2026-07-12).
- Attention head split and activation caches for backward still touch host where needed; weight re-upload every linear call was removed.
- Character vocab (often ~110 for TinyStories ASCII) keeps embedding/LM-head matrices small enough for Kepler.

---

## Repository layout

```text
llm gpu 8/
├── README.md                 ← this file
├── VERSION / version.py      ← project version (0.1.0)
├── train.py                  ← main training entry
├── auto_train.py             ← train + smoke generate
├── generate.py               ← sample from a checkpoint
├── interactive.py            ← interactive generation shell
├── cli_common.py             ← shared argparse / tracers / checkpoint listing
├── paths.py                  ← output/ + quarter helpers
├── logging_config.py
├── training_log_plotter.py
├── loss_landscape_plotter.py
├── bench_step.py / bench_profile.py / bench_mlp_fusion.py
├── model/
│   ├── gpt.py                ← GPT forward/backward/generate
│   ├── layers.py             ← linear / LN / attention / MLP helpers
│   ├── weights.py            ← host + device parameter store
│   ├── config.py / trace.py
│   └── cuda/                 ← env, kernels, ops
├── training/
│   ├── checkpoint.py         ← save/load, promote_best, metrics
│   ├── dataset.py / loss.py / optimizer.py / gpu_optimizer.py
│   ├── eval.py               ← val loss / PPL
│   ├── probe.py              ← checkpoint + generate probes
│   └── quality.py            ← quality scores + compare_quarters
├── tokenizer/
├── setup/                    ← wizard, presets, CUDA activate notes
├── data/                     ← input corpora (gitignored *.txt)
├── output/                   ← logs, checkpoints, configs, tokenizer (gitignored)
├── collab/                   ← Colab / T4 helpers (optional)
└── superfinal/               ← example smaller checkpoint bundle in-tree
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

**Interactive train (wizard):**

```powershell
python train.py --menu
```

**Non-interactive short run:**

```powershell
python train.py --config output\configs\training_config.json `
  --checkpoint output\checkpoints\run1 `
  --steps 500 --learning-rate 0.0001 --no-prompt
```

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
| `--learning-rate` | Override AdamW LR |
| `--batch-size` / `--embedding-dim` / `--num-heads` / `--num-layers` / `--max-len` | Model & batch overrides |
| `--log-every` / `--checkpoint-every` | Progress & disk cadence |
| `--no-prompt` | Never ask on stdin; use CLI/config defaults |
| `--compare-quarters` | Skip train; run quality trial on a run dir |
| `--set-best quarter_50` | Promote a quarter to `best/` |
| `--quality-trial` / `--no-quality-trial` | Force / suppress post-train quality trial |
| `--plot` | Render training / landscape plots after train |

Resume example toward a long target (e.g. remaining steps to 120k):

```powershell
python train.py --resume --checkpoint output\checkpoints\BiggerTest256256 `
  --steps 19000 --learning-rate 0.00001 --batch-size 4 --no-prompt
```

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

**Note:** Chunked resumes that end each chunk at a new `total_steps` will treat the chunk end as 100% and overwrite `quarter_100`. For distinct 25/50/75/100 dirs at absolute steps, prefer one long `--steps` target (e.g. 120000) from the desired start.

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

python bench_step.py
python bench_profile.py
python bench_mlp_fusion.py
```

Logs default under `output/logs/` (`training.log` aggregate + per-run files).

---

## Configuration & presets

- Default config path: `output/configs/training_config.json` (`paths.DEFAULT_CONFIG_PATH`)
- Wizard / presets: `setup/training_setup.py`, `setup/training_presets.py`, `setup/model_config.py`
- **Do not commit** configs that embed full corpora — `setup/training_config.json` and `output/configs/training_config.json` are gitignored

---

## Notable run: BiggerTest256256

Long-running Kepler train used as the project’s stress reference:

| Setting | Value |
|---------|--------|
| Name | `output/checkpoints/BiggerTest256256` |
| Model | embed 256, 8 heads, 4 layers, max_len 256, dropout 0.1 |
| Data | TinyStories character corpus (~558k train / ~62k val sentences), vocab ~110 |
| Batch / LR (late fine-tune) | 4 / `1e-05` |
| Long-run target | **120,000** steps (first logged 2026-07-15) |
| Recent finish | **101,000** steps (~84% of 120k), train loss ~0.97, val loss ~0.96 |
| Quality @ 101k | aggregate **~0.884** (spell 0.93 / punct 0.75 / gram 1.0 / sem 0.85) |

Generation at this stage: reliable TinyStories openers; mid-sample coherence still soft — expected for char-level ~3M-param Kepler training.

---

## Changelog

All commits on `main` from init through current HEAD. Dates are author dates (UTC+local as recorded).

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
```

---

## Versioning

- Canonical version string: [`VERSION`](VERSION) (currently `0.1.0`), exposed as `version.py` → `__version__`.
- Printed at startup by `train.py` / `auto_train.py`; stamped into checkpoint `config.json` / `state.json` / `metrics.json`.
- **Policy:** all work before `a2b1a6f` is pre-0.1.0 (`0.0.x`–`0.9.9` conceptually). Bump `VERSION` when cutting a release and append a section under [Changelog](#changelog).

---

## Related docs

| Doc | Topic |
|-----|--------|
| [`setup/README.md`](setup/README.md) | Model/dataset/init/hyperparam wizard deep dive |
| [`setup/cuda_activate.md`](setup/cuda_activate.md) | GT 730 CUDA / PyCUDA activation journey |
| [`output/README.md`](output/README.md) | Runtime artifact directories |
| [`.cursor/plans/quarterly_checkpoint_resume_f982639b.plan.md`](.cursor/plans/quarterly_checkpoint_resume_f982639b.plan.md) | v0.1.0 quarterly design |
| [`.cursor/plans/pycuda_gpt_training_8d816e55.plan.md`](.cursor/plans/pycuda_gpt_training_8d816e55.plan.md) | Original stack plan |
| [`.cursor/plans/port_gpu5_cuda_core_ce814de3.plan.md`](.cursor/plans/port_gpu5_cuda_core_ce814de3.plan.md) | GPU-5 CUDA core port notes |
| [`.cursor/plans/colab_t4_port_a1845e09.plan.md`](.cursor/plans/colab_t4_port_a1845e09.plan.md) | Colab T4 port notes |

---

## License / authorship

Personal training research workspace (`dtelcore/llm-gpu-8`). Commit history authored primarily by **darren hoyer** (2026-07-12 onward).
