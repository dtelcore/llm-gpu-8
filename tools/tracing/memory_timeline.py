"""
tools/tracing/memory_timeline.py

Summarize ScratchPool Memory Timeline JSONL (pool lifetime, not tensor lifetime).

Usage:
    python -m tools.tracing.memory_timeline --input output/logs/memory_timeline_run.jsonl
    python -m tools.tracing.memory_timeline --input ... --plot
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_events(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def summarize(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    allocs = [e for e in events if e.get("event") == "alloc"]
    reuses = [e for e in events if e.get("event") == "reuse"]
    clears = [e for e in events if e.get("event") == "clear"]
    peak = 0
    for e in events:
        peak = max(peak, int(e.get("peak_pool_bytes", 0) or 0))
        peak = max(peak, int(e.get("pool_resident_bytes", 0) or 0))

    largest: Dict[str, int] = {}
    for e in allocs:
        name = e.get("name") or "<unnamed>"
        nbytes = int(e.get("nbytes", 0))
        largest[name] = max(largest.get(name, 0), nbytes)

    # Peak resident over steps (for plot)
    by_step: Dict[int, int] = {}
    resident = 0
    for e in events:
        step = int(e.get("step", 0))
        if e.get("event") == "alloc":
            resident = int(e.get("pool_resident_bytes", resident))
        elif e.get("event") == "clear":
            resident = 0
        by_step[step] = max(by_step.get(step, 0), resident)

    return {
        "allocations": len(allocs),
        "reuses": len(reuses),
        "clears": len(clears),
        "peak_pool_bytes": peak,
        "peak_pool_mb": peak / (1024.0 ** 2),
        "largest_buffers": sorted(largest.items(), key=lambda kv: -kv[1]),
        "resident_by_step": sorted(by_step.items()),
    }


def print_summary(path: Path, summary: Dict[str, Any]) -> None:
    print("=" * 60)
    print("Memory Timeline Summary")
    print("=" * 60)
    print(f"Input: {path}")
    print()
    print("NOTE: v1 visualizes ScratchPool lifetime (buffers live until clear),")
    print("      not per-activation free/reuse gaps of a future arena allocator.")
    print()
    print("Events:")
    print(f"  allocations: {summary['allocations']}")
    print(f"  reuses:      {summary['reuses']}")
    print(f"  clears:      {summary['clears']}")
    print()
    print(f"Peak pool: {summary['peak_pool_mb']:.1f} MB ({summary['peak_pool_bytes']:,} bytes)")
    print()
    print("Largest buffers:")
    for name, nbytes in summary["largest_buffers"][:15]:
        print(f"  {name:<28} {nbytes / (1024.0 ** 2):6.2f} MB")
    print("=" * 60)


def plot_summary(summary: Dict[str, Any], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    series = summary["resident_by_step"]
    if not series:
        print("No resident-by-step data to plot.")
        return
    steps = [s for s, _ in series]
    mb = [b / (1024.0 ** 2) for _, b in series]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.step(steps, mb, where="post", color="#2c5f7c", linewidth=1.5)
    ax.fill_between(steps, mb, step="post", alpha=0.25, color="#2c5f7c")
    ax.set_xlabel("Training step")
    ax.set_ylabel("ScratchPool resident (MB)")
    ax.set_title("ScratchPool memory timeline (pool lifetime)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Wrote plot: {out_path}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize ScratchPool memory timeline JSONL")
    parser.add_argument("--input", "-i", required=True, help="Path to memory_timeline_*.jsonl")
    parser.add_argument(
        "--plot", action="store_true",
        help="Write output/logs/memory_timeline.png (or --plot-out)",
    )
    parser.add_argument(
        "--plot-out", type=str, default=None,
        help="PNG path (default: output/logs/memory_timeline.png)",
    )
    args = parser.parse_args(argv)

    path = Path(args.input)
    if not path.exists():
        print(f"File not found: {path}")
        return 1
    events = load_events(path)
    if not events:
        print(f"No events in {path}")
        return 1
    summary = summarize(events)
    print_summary(path, summary)
    if args.plot:
        out = Path(args.plot_out) if args.plot_out else Path("output/logs/memory_timeline.png")
        plot_summary(summary, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
