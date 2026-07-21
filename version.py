"""Project version. Pre-0.1.0 work is treated as 0.0.x–0.9.9."""

from pathlib import Path

_VERSION_FILE = Path(__file__).resolve().parent / "VERSION"


def _read_version() -> str:
    try:
        return _VERSION_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return "0.1.0"


__version__ = _read_version()
