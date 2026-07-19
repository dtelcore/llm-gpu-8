"""
loss_landscape_plotter.py

Renders an honest 3D training-loss trajectory from logged (step, loss) pairs.
Axes are real training quantities — not a synthetic PCA bowl.

  X = training step
  Y = local loss volatility (rolling std of recent losses)
  Z = logged loss (actual values)

Usage:
    python loss_landscape_plotter.py                       # longest real run
    python loss_landscape_plotter.py --all-runs             # overlay non-smoke runs
    python loss_landscape_plotter.py --out output/logs/foo.png --show
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

LOG_LINE_RE = re.compile(r"step=(\d+)/(\d+)")
KV_RE = re.compile(r"([a-zA-Z0-9_/]+)=([^\s]+)")

RunSeries = List[Tuple[int, float]]  # (step, loss)

# Ignore smoke tests / tiny probes unless --keep-short is set.
DEFAULT_MIN_POINTS = 50
# Break a segment when the step counter jumps forward by more than this
# (avoids drawing a fake bridge across disconnected runs in the aggregate log).
DEFAULT_MAX_STEP_GAP = 2000


def _safe_float(v) -> Optional[float]:
    try:
        return float(v)
    except Exception:
        return None


def _series_from_rows(rows: List[Tuple[int, float]]) -> RunSeries:
    """Deduplicate by step (last write wins), keep ascending step order."""
    deduped = {}
    for step, loss in rows:
        deduped[step] = loss
    return [(step, deduped[step]) for step in sorted(deduped.keys())]


def _segment_by_step_reset(
    rows: List[Tuple[int, float]],
    max_step_gap: int = DEFAULT_MAX_STEP_GAP,
) -> List[List[Tuple[int, float]]]:
    """Split accumulated logs into contiguous runs.

    A new segment starts when the step counter resets OR jumps forward by more
    than `max_step_gap` (disconnected runs must not be polyline-connected).
    """
    if not rows:
        return []
    segments: List[List[Tuple[int, float]]] = []
    current: List[Tuple[int, float]] = []
    prev_step = 0
    for step, loss in rows:
        if current and (step <= prev_step or step - prev_step > max_step_gap):
            segments.append(current)
            current = []
        current.append((step, loss))
        prev_step = step
    if current:
        segments.append(current)
    return segments


def _rows_from_training_log(path: Path) -> List[Tuple[int, float]]:
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


def _filter_curves(
    curves: List[RunSeries],
    all_runs: bool,
    min_points: int,
) -> List[RunSeries]:
    substantive = [c for c in curves if len(c) >= min_points]
    if not substantive:
        # Fall back to whatever we have so tiny-only logs still plot.
        substantive = [c for c in curves if len(c) >= 2]
    if not substantive:
        return []
    if all_runs:
        return substantive
    return [max(substantive, key=len)]


def read_runs(
    log_dir: str = "output",
    all_runs: bool = False,
    min_points: int = DEFAULT_MIN_POINTS,
    max_step_gap: int = DEFAULT_MAX_STEP_GAP,
) -> List[RunSeries]:
    """Return loss trajectories as lists of (step, loss) for each training run."""
    root = Path(log_dir)
    log_dirs = []
    if root.name == "logs":
        log_dirs.append(root)
    else:
        log_dirs.append(root / "logs")
        if (root / "training.log").exists():
            log_dirs.append(root)

    for logs_dir in log_dirs:
        if not logs_dir.exists():
            continue
        log_paths = sorted(logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
        if log_paths:
            preferred = logs_dir / "training.log"
            target = preferred if preferred in log_paths else log_paths[-1]
            rows = _rows_from_training_log(target)
            segments = _segment_by_step_reset(rows, max_step_gap=max_step_gap)
            curves = [_series_from_rows(seg) for seg in segments if seg]
            filtered = _filter_curves(curves, all_runs=all_runs, min_points=min_points)
            if filtered:
                return filtered

    candidates = [
        root / "training_metrics_latest.jsonl",
        root / "training_metrics_latest.csv",
        root / "logs" / "training_metrics_latest.jsonl",
    ]
    latest_path = next((p for p in candidates if p.exists()), None)
    if latest_path is None:
        return []

    rows = _rows_from_structured_export(latest_path)
    segments = _segment_by_step_reset(rows, max_step_gap=max_step_gap)
    curves = [_series_from_rows(seg) for seg in segments if seg]
    return _filter_curves(curves, all_runs=all_runs, min_points=min_points)


def _normalize_run(run: Union[RunSeries, Sequence[float]]) -> RunSeries:
    """Accept (step, loss) pairs or legacy loss-only lists (synthetic steps 1..N)."""
    if not run:
        return []
    first = run[0]
    if isinstance(first, (tuple, list)) and len(first) >= 2:
        return [(int(s), float(loss)) for s, loss in run]  # type: ignore[misc]
    return [(i + 1, float(loss)) for i, loss in enumerate(run)]  # type: ignore[arg-type]


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling standard deviation."""
    n = len(values)
    out = np.zeros(n, dtype=float)
    if n == 0:
        return out
    w = max(2, min(window, n))
    for i in range(n):
        start = max(0, i - w + 1)
        chunk = values[start : i + 1]
        out[i] = float(np.std(chunk)) if len(chunk) > 1 else 0.0
    return out


def _split_series_on_gaps(series: RunSeries, max_step_gap: int) -> List[RunSeries]:
    """Extra safety: never draw a polyline across a large step hole."""
    if not series:
        return []
    chunks: List[RunSeries] = []
    current: RunSeries = [series[0]]
    for i in range(1, len(series)):
        step, loss = series[i]
        prev_step = current[-1][0]
        if step - prev_step > max_step_gap:
            chunks.append(current)
            current = [(step, loss)]
        else:
            current.append((step, loss))
    chunks.append(current)
    return chunks


def _apply_style() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#0f1117",
        "axes.facecolor": "#161b27",
        "axes.edgecolor": "#2a2f3d",
        "axes.labelcolor": "#c8cdd8",
        "xtick.color": "#a8b0c0",
        "ytick.color": "#a8b0c0",
        "axes.titlecolor": "#e2e6f0",
    })


_TRAJECTORY_COLORS = ["#E8738A", "#4C9BE8", "#7EC87E", "#F0C040", "#B57BED", "#56C4A0"]


def render_landscape(
    runs: Sequence[Union[RunSeries, Sequence[float]]],
    out_path: Optional[Path],
    show: bool,
    volatility_window: int = 25,
    max_step_gap: int = DEFAULT_MAX_STEP_GAP,
) -> Optional[Path]:
    """Plot honest 3D trajectories: step × local volatility × logged loss."""
    series_list = [_normalize_run(run) for run in runs]
    series_list = [s for s in series_list if len(s) >= 2]
    if not series_list:
        return None

    # Primary run = longest (gets start/min/final callouts).
    primary_idx = int(np.argmax([len(s) for s in series_list]))

    _apply_style()
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    for i, series in enumerate(series_list):
        color = _TRAJECTORY_COLORS[i % len(_TRAJECTORY_COLORS)]
        chunks = _split_series_on_gaps(series, max_step_gap=max_step_gap)
        # Volatility is computed on the full series so chunk boundaries stay consistent.
        full_losses = np.array([loss for _, loss in series], dtype=float)
        full_vol = _rolling_std(full_losses, volatility_window)
        # Map step -> volatility index for chunk plotting.
        step_to_idx = {step: idx for idx, (step, _) in enumerate(series)}

        for chunk in chunks:
            if len(chunk) < 2:
                if chunk:
                    s0, l0 = chunk[0]
                    v0 = full_vol[step_to_idx[s0]]
                    ax.scatter([s0], [v0], [l0], color=color, s=25, depthshade=False)
                continue
            steps = np.array([s for s, _ in chunk], dtype=float)
            losses = np.array([loss for _, loss in chunk], dtype=float)
            volatility = np.array([full_vol[step_to_idx[int(s)]] for s in steps], dtype=float)
            # Solid per-run color — no plasma bridge that visually fuses runs.
            ax.plot(steps, volatility, losses, color=color, linewidth=2.0, alpha=0.9)

        steps_all = np.array([s for s, _ in series], dtype=float)
        losses_all = full_losses
        vol_all = full_vol
        start_loss, end_loss = float(losses_all[0]), float(losses_all[-1])
        min_idx = int(np.argmin(losses_all))
        min_loss = float(losses_all[min_idx])
        is_primary = i == primary_idx

        if is_primary:
            ax.scatter(
                [steps_all[0]], [vol_all[0]], [losses_all[0]],
                color="#F0C040", s=60, depthshade=False, zorder=5,
                label=f"start: {start_loss:.4f} @ step {int(steps_all[0]):,}",
            )
            ax.scatter(
                [steps_all[min_idx]], [vol_all[min_idx]], [losses_all[min_idx]],
                color="#7EC87E", s=60, depthshade=False, zorder=5,
                label=f"min: {min_loss:.4f} @ step {int(steps_all[min_idx]):,}",
            )
            ax.scatter(
                [steps_all[-1]], [vol_all[-1]], [losses_all[-1]],
                color=color, s=75, depthshade=False, zorder=5,
                label=f"final: {end_loss:.4f} @ step {int(steps_all[-1]):,}  ({len(series):,} pts)",
            )
            floor = float(np.min(losses_all))
            ax.plot(
                [steps_all[-1], steps_all[-1]],
                [vol_all[-1], vol_all[-1]],
                [floor, losses_all[-1]],
                color=color, linewidth=1.0, alpha=0.5, linestyle="--",
            )
        else:
            ax.scatter(
                [steps_all[-1]], [vol_all[-1]], [losses_all[-1]],
                color=color, s=45, depthshade=False, zorder=5,
                label=f"run {i + 1}: final {end_loss:.4f} ({len(series):,} pts)",
            )

    all_steps = np.concatenate([np.array([s for s, _ in ser], dtype=float) for ser in series_list])
    all_losses = np.concatenate([np.array([loss for _, loss in ser], dtype=float) for ser in series_list])
    all_vol = np.concatenate([
        _rolling_std(np.array([loss for _, loss in ser], dtype=float), volatility_window)
        for ser in series_list
    ])

    ax.set_xlim(float(all_steps.min()), float(all_steps.max()))
    ax.set_ylim(0.0, max(float(all_vol.max()) * 1.15, 1e-6))
    z_min, z_max = float(all_losses.min()), float(all_losses.max())
    pad = max((z_max - z_min) * 0.08, 1e-3)
    ax.set_zlim(max(0.0, z_min - pad), z_max + pad)

    n_points = sum(len(s) for s in series_list)
    n_runs = len(series_list)
    title = f"Training Loss Trajectory  ·  {n_points:,} points"
    if n_runs > 1:
        title += f"  ·  {n_runs} runs"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Step")
    ax.set_ylabel(f"Local volatility\n(rolling std, w={volatility_window})")
    ax.set_zlabel("Loss")
    ax.legend(loc="upper left", fontsize=8, facecolor="#1a2035", edgecolor="#2a2f3d")
    ax.view_init(elev=22, azim=-55)
    try:
        ax.set_box_aspect((2.0, 1.0, 1.1))
    except Exception:
        pass

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"[INFO] Loss trajectory saved to {out_path}")

    if show:
        plt.show()
    plt.close(fig)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Honest 3D training-loss trajectory plotter")
    parser.add_argument("--log-dir", default="output", help="Project root or logs dir (default: output)")
    parser.add_argument("--all-runs", action="store_true", help="Overlay every non-smoke run, not just the longest")
    parser.add_argument("--keep-short", action="store_true", help="Include short smoke runs (< min-points)")
    parser.add_argument("--min-points", type=int, default=DEFAULT_MIN_POINTS,
                        help=f"Drop runs with fewer logged points (default: {DEFAULT_MIN_POINTS})")
    parser.add_argument("--max-step-gap", type=int, default=DEFAULT_MAX_STEP_GAP,
                        help=f"Break polylines across step gaps larger than this (default: {DEFAULT_MAX_STEP_GAP})")
    parser.add_argument("--out", default="output/logs/loss_landscape_latest.png", help="Output image path")
    parser.add_argument("--show", action="store_true", help="Also display the figure interactively")
    parser.add_argument(
        "--volatility-window", type=int, default=25,
        help="Rolling window for local loss volatility on the Y axis (default: 25)",
    )
    args = parser.parse_args()

    min_points = 2 if args.keep_short else max(2, args.min_points)
    runs = read_runs(
        log_dir=args.log_dir,
        all_runs=args.all_runs,
        min_points=min_points,
        max_step_gap=max(1, args.max_step_gap),
    )
    if not runs:
        print("[INFO] No loss metrics found yet.")
        return

    render_landscape(
        runs,
        out_path=Path(args.out) if args.out else None,
        show=args.show,
        volatility_window=max(2, args.volatility_window),
        max_step_gap=max(1, args.max_step_gap),
    )


if __name__ == "__main__":
    main()
