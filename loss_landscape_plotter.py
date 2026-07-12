"""
loss_landscape_plotter.py

Renders a conceptual 3D loss-landscape trajectory (a synthetic bowl surface with
the actual training loss curve embedded as a spiral descent path) from
logs/training.log. This is illustrative, not a real Hessian-based landscape --
it maps observed loss values onto a smooth analytic surface so the descent
shape is representative of how fast/steadily the run converged.

Usage:
    python loss_landscape_plotter.py                       # latest run, logs/training.log
    python loss_landscape_plotter.py --all-runs             # overlay every run in the log
    python loss_landscape_plotter.py --out logs/foo.png --show
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

LOG_LINE_RE = re.compile(r"step=(\d+)/(\d+)")
KV_RE = re.compile(r"([a-zA-Z0-9_/]+)=([^\s]+)")


def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _losses_from_rows(rows: List[Tuple[int, float]]) -> List[float]:
    rows = sorted(rows, key=lambda item: item[0])
    deduped = {}
    for step, loss in rows:
        deduped[step] = loss
    return [deduped[step] for step in sorted(deduped.keys())]


def _segment_by_step_reset(rows: List[Tuple[int, float]]) -> List[List[Tuple[int, float]]]:
    """logs/training.log accumulates every run ever launched; split back into
    individual contiguous runs whenever the step counter resets. Rows must stay
    in file encounter order -- sorting by step would interleave overlapping
    step ranges from different runs and scramble the segmentation."""
    if not rows:
        return []
    segments: List[List[Tuple[int, float]]] = []
    current: List[Tuple[int, float]] = []
    prev_step = 0
    for step, loss in rows:
        if current and step <= prev_step:
            segments.append(current)
            current = []
        current.append((step, loss))
        prev_step = step
    if current:
        segments.append(current)
    return segments


def _rows_from_training_log(path: Path) -> List[Tuple[int, float]]:
    """Any line matching `step=N/TOTAL` counts, regardless of a "[train]" tag --
    resilient to log format tweaks. Falls back to `avg_loss=`/`loss=`, whichever
    key is present on the line."""
    rows: List[Tuple[int, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            step_match = LOG_LINE_RE.search(line)
            if not step_match:
                continue
            step = int(step_match.group(1))
            loss = None
            for key, raw in KV_RE.findall(line):
                if key in ("loss", "avg_loss"):
                    loss = _safe_float(raw)
                    break
            if loss is not None:
                rows.append((step, loss))
    return rows


def _rows_from_structured_export(path: Path) -> List[Tuple[int, float]]:
    rows: List[Tuple[int, float]] = []
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    step = int(data.get("step", 0))
                    loss = _safe_float(data.get("loss", data.get("avg_loss")))
                    if step > 0 and loss is not None:
                        rows.append((step, loss))
                except Exception:
                    pass
    elif path.suffix == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    step = int(row.get("step", 0))
                    loss = _safe_float(row.get("loss") or row.get("avg_loss"))
                    if step > 0 and loss is not None:
                        rows.append((step, loss))
                except Exception:
                    pass
    return rows


def read_runs(log_dir: str = ".", all_runs: bool = False) -> List[List[float]]:
    """Return a list of loss-curves (one list of floats per training run found).
    Looks for logs/training.log first, then falls back to structured metric exports."""
    root = Path(log_dir)
    logs_subdir = root / "logs"
    if logs_subdir.exists():
        log_paths = sorted(logs_subdir.glob("*.log"), key=lambda p: p.stat().st_mtime)
        if log_paths:
            rows = _rows_from_training_log(log_paths[-1])
            segments = _segment_by_step_reset(rows)
            if not all_runs:
                segments = segments[-1:]
            return [_losses_from_rows(seg) for seg in segments if seg]

    candidates = [root / "training_metrics_latest.jsonl", root / "training_metrics_latest.csv"]
    latest_path = next((p for p in candidates if p.exists()), None)
    if latest_path is None:
        return []

    rows = _rows_from_structured_export(latest_path)
    segments = _segment_by_step_reset(rows)
    if not all_runs:
        segments = segments[-1:]
    return [_losses_from_rows(seg) for seg in segments if seg]


def _apply_style() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#0f1117",
        "axes.facecolor":   "#161b27",
        "axes.edgecolor":   "#2a2f3d",
        "axes.labelcolor":  "#c8cdd8",
    })


_TRAJECTORY_COLORS = ["#E8738A", "#4C9BE8", "#7EC87E", "#F0C040", "#B57BED"]


def render_landscape(runs: List[List[float]], out_path: Optional[Path], show: bool) -> Optional[Path]:
    if not runs:
        return None

    _apply_style()

    X = np.linspace(-5, 5, 100)
    Y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(X, Y)
    Z = X ** 2 + 1.5 * Y ** 2

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.35, linewidth=0, antialiased=True)

    for i, losses in enumerate(runs):
        color = _TRAJECTORY_COLORS[i % len(_TRAJECTORY_COLORS)]
        loss_arr = np.array(losses, dtype=float)
        min_loss, max_loss = np.min(loss_arr), np.max(loss_arr)
        normalized_loss = (loss_arr - min_loss) / (max_loss - min_loss) if max_loss > min_loss else np.zeros_like(loss_arr)

        radii = 4.0 * np.sqrt(normalized_loss)
        angles = np.linspace(0, 4 * np.pi, len(losses))

        path_X = radii * np.cos(angles)
        path_Y = radii * np.sin(angles) / np.sqrt(1.5)
        path_Z = path_X ** 2 + 1.5 * path_Y ** 2

        label = f"run {i + 1}" if len(runs) > 1 else "training trajectory"
        ax.plot(path_X, path_Y, path_Z, color=color, linewidth=2, label=label)
        ax.scatter([path_X[-1]], [path_Y[-1]], [path_Z[-1]], color=color, s=50,
                   label=f"final loss: {losses[-1]:.4f}")

    ax.set_title("3D Loss Landscape Optimization Trajectory (conceptual)", fontsize=14, fontweight="bold", color="#e2e6f0")
    ax.set_xlabel("PCA Axis 1 (conceptual)")
    ax.set_ylabel("PCA Axis 2 (conceptual)")
    ax.set_zlabel("Loss")
    ax.legend(fontsize=8, facecolor="#1a2035", edgecolor="#2a2f3d")

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[INFO] 3D loss landscape saved to {out_path}")

    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Conceptual 3D loss-landscape trajectory plotter")
    parser.add_argument("--log-dir", default=".", help="Project root to search for logs/training.log (default: .)")
    parser.add_argument("--all-runs", action="store_true", help="Overlay every run found in the log, not just the latest")
    parser.add_argument("--out", default="logs/loss_landscape_latest.png", help="Output image path")
    parser.add_argument("--show", action="store_true", help="Also display the figure interactively")
    args = parser.parse_args()

    runs = read_runs(log_dir=args.log_dir, all_runs=args.all_runs)
    if not runs:
        print("[INFO] No loss metrics found yet.")
        return

    render_landscape(runs, out_path=Path(args.out) if args.out else None, show=args.show)


if __name__ == "__main__":
    main()
