# GT 730 quick guide — TinyStories training (2026 recipe)

Fast path to train with the **train-recipe-speed** defaults on Kepler GT 730. Full reference: [`README.md`](README.md) · CLI catalog: [`py_calls.md`](py_calls.md).

---

## Prerequisites

1. **Venv + CUDA smoke** (once per machine):
   ```powershell
   cd "c:\dev\llm gpu 8"
   .\venv\Scripts\Activate.ps1
   .\venv\Scripts\python.exe setup\2_test_workspace.py
   ```
2. **Corpus:** put TinyStories (or any) `.txt` files under `data/` (e.g. `data\tiny_stories.txt`).
3. **Use project Python** for training and parity — not a random system `python`.

---

## Quickest good TinyStories run

```powershell
cd "c:\dev\llm gpu 8"
.\venv\Scripts\Activate.ps1
python train.py --menu
```

1. Resume or **new** run → pick a checkpoint name under `output/checkpoints/`.
2. Scaling preset → **2. Tiny Stories**.
3. When asked for steps, start with **500–2000** for smoke, or **20k+** for a real run.

Same wizard from `auto_train.py --menu` if you want a generate smoke right after train.

**Non-interactive length/LR only** (after you already ran the wizard once, or when resuming):

```powershell
python train.py --resume --checkpoint output\checkpoints\YOUR_RUN `
  --steps 10000 --no-prompt
```

Preset hyperparams and architecture come from the checkpoint/config; override LR with `--learning-rate 3e-4` only if you mean to change the recipe.

---

## What preset **2** sets (defaults)

| Knob | Value | Notes |
|------|------:|-------|
| Model | C=256, 8 heads, L=4, T=128 | rmsnorm, RoPE, tied embeddings |
| Params | ~3M | BiggerTest-aligned width/depth; shorter T than BiggerTest256 |
| `dropout_prob` | 0 | Dropout is **not implemented** on GPU — nonzero logs a warning only |
| LR | 3e-4 | AdamW base |
| Warmup | 1000 steps | Linear |
| After warmup | Cosine → `min_lr_ratio` × base | Default floor **0.1** → 3e-5 at end of budget |
| Batch | 4 | Micro-batch |
| Grad accum | 4 | Effective batch **16** sequences per optimizer step |
| `window_stride` | 64 | Fewer windows per epoch vs dense stride-1 |
| LR schedule knobs | `--min-lr-ratio`, `--warmup-steps` | Cosine needs a step budget (`--steps` or epochs) |

Training skips host logits sync on the GPU CE path (`need_host_logits=False`); parity and benches still use host logits by default.

---

## Useful variations

**Context A/B (throughput vs BiggerTest-style length):**

```powershell
# Default preset T=128 (faster steps)
python train.py --menu

# Longer context (closer to BiggerTest256256), same width/depth
python train.py --resume --checkpoint output\checkpoints\YOUR_RUN --max-len 256 --no-prompt --steps ...
```

**Mid-run val without quarterly I/O:**

```powershell
python train.py --resume --checkpoint output\checkpoints\YOUR_RUN `
  --val-every 500 --no-prompt --steps ...
```

**Resume** (architecture fixed by checkpoint):

```powershell
python train.py --resume --checkpoint output\checkpoints\YOUR_RUN `
  --steps 120000 --no-prompt
```

Quarters still fire at 25/50/75/100% of `--run-budget` (or session total on first run). `--checkpoint-every` updates **latest** only.

**Stride override** (dense windows = old behavior, slower epochs):

```powershell
python train.py ... --window-stride 1
```

---

## Verify before a long run

```powershell
.\venv\Scripts\python.exe -m tests.parity.run_parity
```

Expect **10/10**. Short train smoke:

```powershell
python train.py --menu
# preset 2, then e.g. --steps 200 if prompted, or:
python train.py --resume --checkpoint output\checkpoints\YOUR_RUN --steps 200 --no-prompt
```

Optional throughput check (metrics **off** for bench honesty):

```powershell
python bench_step.py
```

Leave **`--grad-checkpoint` off** when measuring tok/s; use it only to save VRAM (~72 MB at BiggerTest shapes).

---

## Common pitfalls

| Mistake | Fix |
|---------|-----|
| Wrong Python / no PyCUDA | Always `.\venv\Scripts\python.exe` or activated venv |
| Old recipe (LR `1e-5`, batch 32, C=128/L=6, flat LR) | Use preset **2** or table above |
| `dropout_prob: 0.1` expecting regularization | Set **0**; kernels not wired |
| `--grad-checkpoint` for speed | VRAM lever only — adds recompute |
| `--runtime-metrics` / `--memory-timeline` left on | Extra sync/I/O; off for max tok/s |
| Expecting bf16 / FlashAttention / AMP GEMM on GT 730 | Out of scope for this card |
| Expecting full-step CUDA Graph speedup on generate | Kernel-chain (KV append/decode/argmax) captures; GEMM/norm stay eager device |
| FP16 **storage** vs training | Device casts for storage; training math is still FP32 |
| `device_used_mb` ≈ full card | Process-only (excludes HDMI/display); see `vram_driver_used_mb` for total-free |
| Stride 64 “short epoch” | Normal — fewer unique windows per pass over the corpus |

---

## Related

- Historical long run (T=256, late LR 1e-5): [`README.md#notable-run-biggertest256256`](README.md)
- Setup wizard details: [`setup/README.md`](setup/README.md)
