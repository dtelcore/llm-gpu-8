#!/usr/bin/env python3
"""
collab/make_toy_config.py

Copy collab toy config into output/configs/. Run from repo root:

    python collab/make_toy_config.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paths import OUTPUT_CONFIGS, ensure_output_dirs


def main() -> int:
    ensure_output_dirs()
    src = _REPO_ROOT / "collab" / "configs" / "collab_toy.json"
    if not src.exists():
        print(f"[FAIL] missing {src}", file=sys.stderr)
        return 1

    dst = OUTPUT_CONFIGS / "collab_toy.json"
    shutil.copy2(src, dst)
    print(f"[OK] wrote {dst}")

    with open(src, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    print(
        f"model={cfg['model']['name']}  batch={cfg['hyperparameters']['batch_size']}  "
        f"epochs={cfg['hyperparameters']['num_epochs']}"
    )
    print("Train with:")
    print(
        "  python collab/run_train.py --config collab/configs/collab_toy.json "
        "--checkpoint output/checkpoints/collab_smoke --epochs 2 --seed 0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
