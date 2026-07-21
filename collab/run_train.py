#!/usr/bin/env python3
"""
collab/run_train.py

Apply collab.bootstrap in a child process, then run train.main with
non-interactive defaults (--no-prompt --no-quality-trial).

Usage (from repo root):

    python collab/run_train.py --config collab/configs/collab_toy.json \\
      --checkpoint output/checkpoints/collab_smoke --epochs 2 --seed 0
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    forwarded = list(argv if argv is not None else sys.argv[1:])
    joined = " ".join(forwarded)
    if "--no-prompt" not in joined:
        forwarded.append("--no-prompt")
    if "--no-quality-trial" not in joined:
        forwarded.append("--no-quality-trial")

    argv_json = json.dumps(["train.py", *forwarded])
    preamble = (
        "from collab.bootstrap import apply; apply(); "
        "import sys; "
        f"sys.argv = json.loads({json.dumps(argv_json)}); "
        "from train import main; main()"
    )
    # Need json in the child too
    preamble = "import json; " + preamble

    child = [sys.executable, "-c", preamble]
    print("[collab/run_train] launching patched train child")
    print("[collab/run_train] args:", " ".join(forwarded))
    return subprocess.call(child, cwd=str(_REPO_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
