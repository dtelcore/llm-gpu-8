#!/usr/bin/env python3
"""
collab/sync_drive.py

Push/pull output/ artifacts to Google Drive (Colab Free persistence).

    python collab/sync_drive.py --push
    python collab/sync_drive.py --pull
    python collab/sync_drive.py --push --drive /content/drive/MyDrive/llm_gpu8_output
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from paths import OUTPUT_CHECKPOINTS, OUTPUT_CONFIGS, OUTPUT_LOGS, OUTPUT_TOKENIZER, ensure_output_dirs

DEFAULT_DRIVE = "/content/drive/MyDrive/llm_gpu8_output"


def _local_map() -> dict:
    return {
        "checkpoints": OUTPUT_CHECKPOINTS,
        "configs": OUTPUT_CONFIGS,
        "logs": OUTPUT_LOGS,
        "tokenizer": OUTPUT_TOKENIZER,
    }


def _copy_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"  (skip missing {src})")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        for item in src.iterdir():
            target = dst / item.name
            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)
    else:
        shutil.copytree(src, dst)
    print(f"  {src} -> {dst}")


def push(drive_root: Path) -> None:
    ensure_output_dirs()
    drive_root.mkdir(parents=True, exist_ok=True)
    print(f"PUSH local output/ -> {drive_root}")
    for name, local in _local_map().items():
        _copy_tree(local, drive_root / name)


def pull(drive_root: Path) -> None:
    ensure_output_dirs()
    print(f"PULL {drive_root} -> local output/")
    if not drive_root.exists():
        print(f"[FAIL] drive path does not exist: {drive_root}", file=sys.stderr)
        raise SystemExit(1)
    for name, local in _local_map().items():
        remote = drive_root / name
        _copy_tree(remote, local)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync output/ artifacts with Google Drive")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--push", action="store_true", help="Copy local output/ to Drive")
    g.add_argument("--pull", action="store_true", help="Copy Drive back to local output/")
    p.add_argument(
        "--drive", type=str, default=DEFAULT_DRIVE,
        help=f"Drive sync root (default: {DEFAULT_DRIVE})",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    drive_root = Path(args.drive)
    if args.push:
        push(drive_root)
    else:
        pull(drive_root)
    print("[OK] sync complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
