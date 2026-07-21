#!/usr/bin/env python3
"""
collab/run_generate.py

Bootstrap CUDA patches, then run generate.main.

    python collab/run_generate.py --checkpoint output/checkpoints/collab_smoke \\
      --prompt "once upon a" --max-new-tokens 80
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    forwarded = list(argv if argv is not None else sys.argv[1:])
    argv_json = json.dumps(["generate.py", *forwarded])
    preamble = (
        "import json; "
        "from collab.bootstrap import apply; apply(); "
        "import sys; "
        f"sys.argv = json.loads({json.dumps(argv_json)}); "
        "from generate import main; main()"
    )
    child = [sys.executable, "-c", preamble]
    print("[collab/run_generate] launching patched generate child")
    print("[collab/run_generate] args:", " ".join(forwarded))
    return subprocess.call(child, cwd=str(_REPO_ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
