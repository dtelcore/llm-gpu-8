"""
Run NumPy↔CUDA parity suite:

    python -m tests.parity.run_parity
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Ensure project root is on sys.path when run as a module
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> int:
    loader = unittest.defaultTestLoader
    suite = loader.discover(
        start_dir=str(Path(__file__).resolve().parent),
        pattern="test_*.py",
        top_level_dir=str(_ROOT),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
