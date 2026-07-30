---
name: Train recipe speed
overview: "Immediate + near-term: fix TinyStories recipe (LR/cosine/batch/arch), strided windows, honest dropout, skip train logits host sync; then RoPE QKV-split fusion, residual depth init, and cheap mid-run val — BiggerTest-aligned C=256/L=4 defaults."
todos:
  - id: p1-cosine-lr
    content: "A: Cosine after warmup in AdamWGPU + train/CLI wiring (min_lr_ratio=0.1)"
    status: completed
  - id: p1-presets
    content: "B+H+C: tiny_stories C=256/L=4/T=128, LR 3e-4, warmup 1000, B=4, accum=4, dropout=0 + warn if >0"
    status: completed
  - id: p1-stride
    content: "D: window_stride on WindowedDataset + --window-stride (preset default 64)"
    status: completed
  - id: p2-logits-host
    content: "E: need_host_logits=False on train GPU path; skip to_host"
    status: completed
  - id: p3-rope-qkv
    content: "F: RoPE after linear_qkv_split + heads; parity + bench"
    status: completed
  - id: p3-residual-init
    content: "I: 1/sqrt(2L) on attn out + mlp_contract init"
    status: completed
  - id: p3-val-every
    content: "K: --val-every N lightweight val without quarterly I/O"
    status: completed
isProject: false
---

# Training speed + sample-efficiency (GT 730)

Scope locked from your answers: **Immediate (A–E) + near-term (F, H, I, K)**; **BiggerTest-aligned** preset (`C=256`, heads=8, `L=4`, `T=128`). Skip BPE (J), attention kernel rewrite (L), fused-MLP enable (M), FP16/Adam-persist/story-packing (N–P), and all Kepler skips from your analysis.

Modernization (RMSNorm / RoPE / grad-checkpoint) stays as-is; this plan only changes **recipe + a few free/cheap throughput wins**.

```mermaid
flowchart LR
  P1[P1 Recipe schedule data] --> Gate1[Parity plus short train]
  Gate1 --> P2[P2 Throughput free wins]
  P2 --> Gate2[Bench vs stage31]
  Gate2 --> P3[P3 RoPE QKV fuse plus init plus val]
  P3 --> Gate3[Parity plus A/B recipe]
```

---

## P1 — Recipe + learning (highest ROI)

### A. Cosine LR after warmup
In [`training/gpu_optimizer.py`](training/gpu_optimizer.py), extend `AdamWGPU`:

- Add `total_steps: int` and `min_lr_ratio: float = 0.1` (floor = `0.1 * base_lr`).
- `current_lr()`: linear warmup for `t < warmup_steps`, then cosine from `base_lr` → `min_lr` over remaining steps (`t` from warmup to `total_steps`).
- If `total_steps <= warmup_steps`, stay flat at `base_lr` after warmup (safe degenerate).

Wire in [`train.py`](train.py) when constructing the optimizer (pass session `total_steps` / `run_budget_steps`). CLI: `--min-lr-ratio` (default `0.1`) in [`cli_common.py`](cli_common.py); persist in hyperparams / checkpoint state if other LR knobs already round-trip.

### B. Fix `tiny_stories` defaults (BiggerTest-aligned)
[`setup/model_config.py`](setup/model_config.py) `PRESETS['tiny_stories']`:

| Knob | New default |
|------|-------------|
| `embedding_dim` | **256** |
| `num_heads` | **8** |
| `num_layers` | **4** |
| `max_len` | **128** (H: short-T default; CLI `--max-len 256` for BiggerTest-quality) |
| `dropout_prob` | **0.0** |
| keep | `rmsnorm`, `rope`, `tie_embeddings=True` |

[`setup/training_presets.py`](setup/training_presets.py) hyperparams:

| Knob | New default |
|------|-------------|
| `learning_rate` | **3e-4** |
| `warmup_steps` | **1000** |
| `batch_size` | **4** |
| `gradient_accumulation_steps` | **4** |
| keep | `weight_decay=0.01`, `gradient_clip=1.0` |

Update tagline/description (~3M params, not “~1M” / batch=32). Do **not** change legacy BiggerTest checkpoints’ loaded config.

### C. Dropout honesty
- Preset/default `dropout_prob=0.0` (above).
- In [`model/config.py`](model/config.py) / train startup: if `dropout_prob > 0`, log a clear warning that dropout is **not implemented** (no kernels). Do **not** implement residual dropout in this pass.

### D. Strided window sampling
[`training/dataset.py`](training/dataset.py):

- Add `window_stride: int = 1` (constructor + CLI `--window-stride`, default **64** for TinyStories preset / train wiring; `1` preserves old behavior).
- `starts = np.arange(0, num_dense_windows, stride)`; shuffle those.
- `num_windows()` / `num_batches()` reflect strided count.
- Document that “epoch” length shrinks vs dense starts.

### Docs / example CLI
Short README note: do not use old `tiny_stories` hyperparams; recommended GT 730 recipe matches §4 of your analysis (`--grad-checkpoint` off for tok/s baselines).

---

## P2 — Free throughput (E)

### E. Skip logits `to_host` on train path
[`model/gpt.py`](model/gpt.py) GPU `forward_batch` (~320): only call `to_host(logits_d)` when host logits are needed (tracing, NumPy path, or an explicit flag). Train already uses `cache["logits_d"]` ([`train.py`](train.py) ~629).

Concrete approach: add `need_host_logits: bool = True` kwarg defaulting True for backward compat; train/eval GPU CE path passes `False` and uses device CE only. Host `cache["logits"]` omitted or empty when False. Update callers that read host logits (parity, benches, probes) to keep default True or fetch explicitly.

Expect sub-% tok/s; still ship as hygiene.

---

## P3 — Near-term throughput + init + ops feedback

### F. Restore QKV-split fusion on RoPE path
Today RoPE forces `matmul_bias` + `split_heads_from_qkv` ([`model/cuda/ops.py`](model/cuda/ops.py) ~595–620). Change `fused_causal_attention_from_qkv` to:

1. Always use `linear_qkv_split` (no full `[B·T, 3C]` buffer).
2. Convert interleaved Q/K/V → heads layout (existing `_interleaved_to_heads_kernel`).
3. If `rope_base` set: `rope_apply_inplace` on Q and K heads.
4. Run existing fused attn / `causal_self_attention` as today.

Parity: extend [`tests/parity/test_rope.py`](tests/parity/test_rope.py) / [`tests/parity/test_modern_step.py`](tests/parity/test_modern_step.py); full `run_parity`. Measure with `bench_step.py` + short `--runtime-metrics` vs pre-change.

### H. Short-T as default (already in P1)
`max_len=128` in preset; document `--max-len 256` A/B. No extra code beyond preset + README.

### I. GPT-2 residual out-proj init `1/sqrt(2L)`
[`setup/weight_init.py`](setup/weight_init.py): for `output_proj` and `mlp_contract`, multiply fan-in scale by `1/sqrt(2 * total_layers)` when `total_layers > 0`.

[`model/weights.py`](model/weights.py) `_init` / layer construction: pass `total_layers=config.num_layers` (and depth if already threaded). Depth arg is currently unused — wire it for these two types only.

### K. Cheap mid-run val
[`train.py`](train.py) + [`cli_common.py`](cli_common.py): `--val-every N` (default **0** = off; suggested **500** in docs for recipe sweeps). When `N > 0` and `global_step % N == 0`, call existing `evaluate_val_loss` without full quarterly checkpoint/probe (log val_loss only). Quarters unchanged.

---

## Explicit non-goals (this plan)

- Grad-checkpoint as speed lever (G stays ops guidance only)
- BPE pilot, attention kernel tuning, fused MLP enable, device FP16, Adam m/v persist, story packing
- bf16 / FlashAttention / CUDA Graphs / ScratchPool redesign / host FP16 training

---

## Verification gates (after each phase)

1. `python -m tests.parity.run_parity` green
2. After P2/P3: short `bench_step.py` + optional 200-step train with `--runtime-metrics` vs stage31 reference (~586 tok/s at BiggerTest shapes — expect similar or slightly better from E/F; T=128 preset will differ)
3. After P1: 5k-step A/B (old 1e-5 flat vs new 3e-4+cosine + stride) only if you want learning evidence before a long run — not a code gate

---

## Key files

| Area | Files |
|------|--------|
| Schedule | [`training/gpu_optimizer.py`](training/gpu_optimizer.py), [`train.py`](train.py), [`cli_common.py`](cli_common.py) |
| Presets | [`setup/training_presets.py`](setup/training_presets.py), [`setup/model_config.py`](setup/model_config.py) |
| Data | [`training/dataset.py`](training/dataset.py) |
| Host sync | [`model/gpt.py`](model/gpt.py) |
| RoPE fuse | [`model/cuda/ops.py`](model/cuda/ops.py) + rope/modern parity tests |
| Init | [`setup/weight_init.py`](setup/weight_init.py), [`model/weights.py`](model/weights.py) |
| Val cadence | [`train.py`](train.py), [`training/eval.py`](training/eval.py) |
