# output/

All runtime artifacts for this project live here (mirrors the `llm gpu 5` layout).

| Directory | Contents | In git? |
|-----------|----------|---------|
| `baselines/` | Frozen Stage 3 JSON baselines (scientific control) | **yes** (JSON only) |
| `releases/` | Known-good snapshots (`v0.1.2/`, …) from `tools/releases/make_snapshot.py` | **yes** (`v0.1.2/` only) |
| `reports/` | Evolution HTML and similar summaries | **yes** (`evolution.html`) |
| `logs/` | Training logs, plot PNGs, memory timelines | no |
| `checkpoints/` | Per-run checkpoint bundles | no |
| `configs/` | `training_config.json` and wizard exports | no |
| `tokenizer/` | Vocab sidecars mirrored on each checkpoint save | no |
| `cache/` | Optional encoding/tokenizer caches | no |
| `audits/` | Ad-hoc audit dumps | no |

Input corpora remain in `data/` at the project root (not under `output/`).

Defaults are set in `paths.py` and used by `train.py`, `auto_train.py`, `generate.py`, and `interactive.py`.

Rebuild the release snapshot:

```powershell
python tools/releases/make_snapshot.py --tag v0.1.2
```
