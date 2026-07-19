
#!/usr/bin/env python3
"""
Training log plotter — enhanced edition.

Key behaviors:
  • Prefers aggregate training.log / richest log; plots the longest substantive run
  • Drops smoke runs (<50 points) unless --keep-short / --all-runs
  • Breaks curves across large step gaps (no fake bridges)
  • Terminal summary table (loss, Δloss, PPL, forecast, tok/s, progress)
  • Live mode only reprints the summary when step/loss changes
  • --save writes PNG without blocking on a GUI window
"""

import argparse
import re
import sys
import json
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

LOG_LINE_RE = re.compile(r"step=(\d+)/(\d+)")


def _terminal_box_chars() -> Dict[str, str]:
    """Return box-drawing chars, falling back to ASCII on narrow Windows consoles."""
    fancy = {
        "tl": "┌", "tr": "┐", "bl": "└", "br": "┘",
        "ml": "├", "mr": "┤", "h": "─", "v": "│", "dash": "—",
    }
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        for ch in fancy.values():
            ch.encode(encoding)
        return fancy
    except (UnicodeEncodeError, LookupError, TypeError):
        return {
            "tl": "+", "tr": "+", "bl": "+", "br": "+",
            "ml": "+", "mr": "+", "h": "-", "v": "|", "dash": "-",
        }


KV_RE = re.compile(r"([a-zA-Z0-9_/]+)=([^\s]+)")

# Ignore smoke / tiny probes by default (same policy as loss_landscape_plotter).
DEFAULT_MIN_POINTS = 50
DEFAULT_MAX_STEP_GAP = 2000

# ── palette (tab10-inspired but hand-picked for readability on dark bg) ──────
_COLORS = [
    "#4C9BE8",  # blue
    "#56C4A0",  # teal
    "#F28B5A",  # coral
    "#B57BED",  # purple
    "#F0C040",  # amber
    "#E8738A",  # pink
    "#7EC87E",  # green
]


# ── data model ───────────────────────────────────────────────────────────────

@dataclass
class RunSeries:
    name: str
    path: Path
    steps: List[int]
    total_steps: int
    metrics: Dict[str, List[Optional[float]]] = field(default_factory=dict)


# ── parsing ──────────────────────────────────────────────────────────────────

def _safe_float(value: str) -> Optional[float]:
    try:
        value = value.strip().rstrip(",")
        if value.lower() in {"nan", "none", "inf", "-inf"}:
            return None
        return float(value)
    except Exception:
        return None


def _rows_from_structured(path: Path) -> Tuple[List[int], Dict[str, List[Optional[float]]], int]:
    """Parse .jsonl/.csv metric exports into (steps, metrics, total_steps). One run only
    (these formats are per-run exports, unlike the shared logs/training.log)."""
    steps: List[int] = []
    metrics: Dict[str, List[Optional[float]]] = {}
    total_steps = 0

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    step = int(data.get("step", 0))
                    if not step: continue
                    steps.append(step)
                    total_steps = max(total_steps, step)
                    for k, v in data.items():
                        if k == "step": continue
                        metrics.setdefault(k, []).append(_safe_float(str(v)) if v is not None else None)
                except Exception:
                    pass
    elif path.suffix == ".csv":
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    step = int(row.get("step", 0))
                    if not step: continue
                    steps.append(step)
                    total_steps = max(total_steps, step)
                    for k, v in row.items():
                        if k == "step": continue
                        metrics.setdefault(k, []).append(_safe_float(v) if v else None)
                except Exception:
                    pass

    max_len = len(steps)
    for k in metrics:
        if len(metrics[k]) < max_len:
            metrics[k].extend([None] * (max_len - len(metrics[k])))
    return steps, metrics, total_steps


def _parse_log_line(line: str) -> Optional[Tuple[int, int, Dict[str, Optional[float]]]]:
    step_match = LOG_LINE_RE.search(line)
    if not step_match:
        return None
    step = int(step_match.group(1))
    total_steps = int(step_match.group(2))
    row: Dict[str, Optional[float]] = {}
    for key, raw in KV_RE.findall(line):
        if key == "step":
            continue
        row[key] = _safe_float(raw)
    return step, total_steps, row


# Cache of already-parsed rows per text log path, keyed so live refreshes only
# need to read/parse bytes appended since the last poll instead of re-reading
# the whole (potentially huge, ever-growing) file each time.
_TEXT_LOG_CACHE: Dict[Path, Dict[str, Any]] = {}


def _rows_from_text_log(path: Path, use_cache: bool = False) -> List[Tuple[int, int, Dict[str, Optional[float]]]]:
    """Parse a plaintext log (e.g. logs/training.log) into (step, total_steps, kv-row)
    tuples, one per line matching `step=N/TOTAL`. Any line with that pattern counts,
    whether or not it's tagged "[train]" -- keeps this resilient to log format tweaks.

    When use_cache is True (live mode), only bytes appended since the previous
    call are read and parsed; previously parsed rows are reused as-is."""
    if not use_cache:
        rows: List[Tuple[int, int, Dict[str, Optional[float]]]] = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parsed = _parse_log_line(line)
                if parsed:
                    rows.append(parsed)
        return rows

    cache = _TEXT_LOG_CACHE.get(path)
    size = path.stat().st_size
    if cache is None or size < cache["offset"]:
        # First load, or file shrank/rotated -- reparse from scratch.
        cache = {"offset": 0, "rows": [], "leftover": ""}

    if size > cache["offset"]:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            f.seek(cache["offset"])
            chunk = cache["leftover"] + f.read()
            cache["offset"] = size
        lines = chunk.split("\n")
        # Last element may be a partial line if the writer hasn't flushed a
        # trailing newline yet; stash it and prepend on the next read.
        cache["leftover"] = lines.pop() if lines else ""
        for line in lines:
            parsed = _parse_log_line(line)
            if parsed:
                cache["rows"].append(parsed)

    _TEXT_LOG_CACHE[path] = cache
    return cache["rows"]


def _segment_by_step_reset(
    rows: List[Tuple[int, int, Dict[str, Optional[float]]]],
    max_step_gap: int = DEFAULT_MAX_STEP_GAP,
) -> List[List[Tuple[int, int, Dict[str, Optional[float]]]]]:
    """Split rows into contiguous runs.

    A new segment starts when the step counter resets OR jumps forward by more
    than `max_step_gap` (avoids bridging disconnected runs in an aggregate log).
    """
    if not rows:
        return []
    segments: List[List[Tuple[int, int, Dict[str, Optional[float]]]]] = []
    current: List[Tuple[int, int, Dict[str, Optional[float]]]] = []
    prev_step = 0
    for step, total_steps, row in rows:
        if current and (step <= prev_step or step - prev_step > max_step_gap):
            segments.append(current)
            current = []
        current.append((step, total_steps, row))
        prev_step = step
    if current:
        segments.append(current)
    return segments


def _filter_runs(runs: List[RunSeries], all_runs: bool, min_points: int) -> List[RunSeries]:
    substantive = [r for r in runs if len(r.steps) >= min_points]
    if not substantive:
        substantive = [r for r in runs if len(r.steps) >= 2]
    if not substantive:
        return []
    if all_runs:
        return substantive
    # Prefer the richest trajectory, not the chronologically last (short resumes
    # / smokes otherwise hide the real training run).
    best = max(substantive, key=lambda r: len(r.steps))
    # Drop "_runN" suffix when only one series is shown.
    clean = re.sub(r"_run\d+$", "", best.name)
    if clean != best.name:
        best = RunSeries(
            name=clean, path=best.path, steps=best.steps,
            total_steps=best.total_steps, metrics=best.metrics,
        )
    return [best]


def _derive_ppl(metrics: Dict[str, List[Optional[float]]]) -> None:
    """Add a 'ppl' column derived from 'loss' (perplexity = e^loss) when not already present."""
    if "ppl" in metrics or "loss" not in metrics:
        return
    ppl: List[Optional[float]] = []
    for v in metrics["loss"]:
        if v is None:
            ppl.append(None)
        else:
            ppl.append(float(np.exp(min(v, 50.0))))
    metrics["ppl"] = ppl


def _parse_log_file(
    path: Path,
    all_runs: bool = False,
    use_cache: bool = False,
    min_points: int = DEFAULT_MIN_POINTS,
    max_step_gap: int = DEFAULT_MAX_STEP_GAP,
) -> List[RunSeries]:
    """Parse a log/metrics file into one RunSeries per contiguous training run found
    in it. Text logs (like logs/training.log) may contain many runs concatenated;
    structured .jsonl/.csv exports are treated as a single run."""
    if path.suffix in (".jsonl", ".csv"):
        steps, metrics, total_steps = _rows_from_structured(path)
        if not steps:
            return []
        _derive_ppl(metrics)
        run = _canonicalize_run(RunSeries(name=path.stem, path=path, steps=steps,
                                          total_steps=total_steps, metrics=metrics))
        return _filter_runs([run], all_runs=all_runs, min_points=min_points)

    rows = _rows_from_text_log(path, use_cache=use_cache)
    segments = _segment_by_step_reset(rows, max_step_gap=max_step_gap)
    if not segments:
        return []

    runs: List[RunSeries] = []
    for idx, segment in enumerate(segments):
        steps = [s for s, _, _ in segment]
        total_steps = max((t for _, t, _ in segment), default=0)
        metrics: Dict[str, List[Optional[float]]] = {}
        keys = sorted({k for _, _, row in segment for k in row})
        for key in keys:
            metrics[key] = [row.get(key) for _, _, row in segment]
        _derive_ppl(metrics)
        run_idx_suffix = f"_run{idx + 1}" if len(segments) > 1 else ""
        run = _canonicalize_run(RunSeries(
            name=f"{path.stem}{run_idx_suffix}", path=path,
            steps=steps, total_steps=total_steps, metrics=metrics,
        ))
        runs.append(run)
    return _filter_runs(runs, all_runs=all_runs, min_points=min_points)


def _canonicalize_run(run: RunSeries) -> RunSeries:
    """Sort by step and keep the last sample per step for clean plots."""
    if not run.steps:
        return run

    rows: List[Tuple[int, Dict[str, Optional[float]]]] = []
    metric_keys = list(run.metrics.keys())
    for i, step in enumerate(run.steps):
        row = {
            key: (run.metrics[key][i] if i < len(run.metrics[key]) else None)
            for key in metric_keys
        }
        rows.append((step, row))

    rows.sort(key=lambda item: item[0])
    deduped: Dict[int, Dict[str, Optional[float]]] = {}
    for step, row in rows:
        deduped[step] = row

    steps_sorted = sorted(deduped.keys())
    metrics_sorted: Dict[str, List[Optional[float]]] = {
        key: [deduped[step].get(key) for step in steps_sorted]
        for key in metric_keys
    }
    total_steps = max(run.total_steps, steps_sorted[-1] if steps_sorted else 0)
    return RunSeries(
        name=run.name,
        path=run.path,
        steps=steps_sorted,
        total_steps=total_steps,
        metrics=metrics_sorted,
    )


def _load_runs(
    paths: List[Path],
    all_runs: bool = False,
    use_cache: bool = False,
    min_points: int = DEFAULT_MIN_POINTS,
    max_step_gap: int = DEFAULT_MAX_STEP_GAP,
) -> List[RunSeries]:
    runs: List[RunSeries] = []
    for path in paths:
        if path.exists():
            runs.extend(_parse_log_file(
                path,
                all_runs=all_runs,
                use_cache=use_cache,
                min_points=min_points,
                max_step_gap=max_step_gap,
            ))
    if all_runs or len(paths) > 1:
        return runs
    # Single-file default path already filtered inside _parse_log_file; if multiple
    # files somehow yielded multiple runs, keep the longest.
    return _filter_runs(runs, all_runs=False, min_points=min_points) if runs else []


# ── math helpers ─────────────────────────────────────────────────────────────

def _to_arr(values: Sequence[Optional[float]]) -> np.ndarray:
    return np.array([np.nan if v is None else float(v) for v in values], dtype=float)


def _rolling_mean(values: Sequence[Optional[float]], window: int) -> np.ndarray:
    """Centered rolling mean, NaN-aware. Vectorized via cumulative sums so it
    stays O(n) regardless of window size -- the naive per-index-slice loop
    turned O(n*window) and became the dominant cost (and thus the main drag
    on how quickly the chart could redraw/autoscale) once logs grew past
    tens of thousands of steps."""
    arr = _to_arr(values)
    n = len(arr)
    if n == 0 or window <= 1:
        return arr.copy()

    half = window // 2
    mask = ~np.isnan(arr)
    filled = np.where(mask, arr, 0.0)

    # Prefix sums (padded with a leading 0) let us get any window's sum/count
    # via two subtractions instead of re-scanning the slice.
    sum_cs = np.concatenate(([0.0], np.cumsum(filled)))
    cnt_cs = np.concatenate(([0.0], np.cumsum(mask, dtype=float)))

    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)

    counts = cnt_cs[hi] - cnt_cs[lo]
    sums = sum_cs[hi] - sum_cs[lo]

    out = np.full(n, np.nan)
    valid = counts > 0
    out[valid] = sums[valid] / counts[valid]
    return out


def _ema(values: Sequence[Optional[float]], alpha: float) -> np.ndarray:
    arr = _to_arr(values)
    out = np.full_like(arr, np.nan)
    state = np.nan
    for i, v in enumerate(arr):
        if np.isnan(v):
            out[i] = state
            continue
        state = v if np.isnan(state) else (1.0 - alpha) * state + alpha * v
        out[i] = state
    return out


def _fit_line(xs: Sequence[float], ys: Sequence[float]) -> Optional[Tuple[float, float]]:
    if len(xs) < 2:
        return None
    try:
        coeffs = np.polyfit(np.array(xs, float), np.array(ys, float), 1)
        return float(coeffs[0]), float(coeffs[1])
    except Exception:
        return None


def _loss_forecast(run: RunSeries, smooth_window: int, forecast_window: int,
                   use_smoothed: bool) -> Optional[Dict[str, float]]:
    raw = [run.metrics.get("loss", [None] * len(run.steps))[i]
           if i < len(run.metrics.get("loss", [])) else None
           for i in range(len(run.steps))]
    curve = _rolling_mean(raw, smooth_window) if use_smoothed else _to_arr(raw)
    tail = run.steps[-forecast_window:]
    tail_curve = curve[-forecast_window:]
    xs = [float(s) for s, v in zip(tail, tail_curve) if not np.isnan(v)]
    ys = [float(v) for v in tail_curve if not np.isnan(v)]
    fit = _fit_line(xs, ys)
    if fit is None or not run.steps:
        return None
    slope, intercept = fit
    observed_step = float(run.steps[-1])
    target_step = float(run.total_steps or run.steps[-1])
    return {
        "observed_step": observed_step,
        "target_step": target_step,
        "predicted_loss": slope * target_step + intercept,
        "slope": slope,
        "last_smoothed": float(curve[-1]) if not np.isnan(curve[-1]) else float(curve[~np.isnan(curve)][-1]),
    }


# ── metric aliases ────────────────────────────────────────────────────────────

_ALIAS_MAP: Dict[str, List[str]] = {
    "tok/s":          ["tok/s", "tok_s", "toks", "tokens_per_sec"],
    "step_ms":        ["step_ms", "stepms", "ms_per_step"],
    "grad_norm":      ["grad_norm", "gradnorm"],
    "device_used_mb": ["device_used_mb", "device_mb", "gpu_mb"],
    "lr":             ["lr", "learning_rate"],
    "ppl":            ["ppl", "val_ppl", "perplexity"]
}


def _resolve_metric(run: RunSeries, metric_name: str) -> List[Optional[float]]:
    lowered = metric_name.lower()
    for aliases in _ALIAS_MAP.values():
        if lowered in [a.lower() for a in aliases]:
            for alias in aliases:
                if alias in run.metrics:
                    return run.metrics[alias]
    return run.metrics.get(metric_name, [None] * len(run.steps))


# ── name formatting ───────────────────────────────────────────────────────────

def _short_name(name: str) -> str:
    text = name.replace("training_", "").replace("steps_", "s_")
    # Clean up scientific or fractional learning rates (e.g. 9p0e-06lr -> 9.0e-06lr)
    text = re.sub(r'(\d+)p(\d+(?:e[-+]?\d+)?)', r'\1.\2', text)
    # Compact common architecture strings
    text = (text
            .replace("_ctx1024_", "_c1k_")
            .replace("_ctx512_", "_c512_")
            .replace("_ctx256_", "_c256_")
            .replace("_ctx128_", "_c128_")
            .replace("_deep_384d_4l_", "_d384x4_"))
    return text[:42]


# ── terminal summary ──────────────────────────────────────────────────────────

def _nan_safe(arr: np.ndarray) -> Optional[float]:
    valid = arr[~np.isnan(arr)]
    return float(valid[-1]) if len(valid) else None


def print_summary(runs: List[RunSeries], smooth_window: int, forecast_window: int) -> None:
    box = _terminal_box_chars()
    dash = box["dash"]
    W = 82
    print()
    print(box["tl"] + box["h"] * (W - 2) + box["tr"])
    print(box["v"] + "  Training run summary" + " " * (W - 24) + box["v"])
    print(box["ml"] + box["h"] * (W - 2) + box["mr"])

    headers = f"  {'Run':<28}{'Loss':>8}{'dLoss':>8}{'PPL':>8}{'Forecast':>10}{'Tok/s':>8}{'Prog':>7}"
    print(box["v"] + headers + " " * (W - 2 - len(headers)) + box["v"])
    print(box["ml"] + box["h"] * (W - 2) + box["mr"])

    for run in runs:
        name = _short_name(run.name)[:26]
        
        # Loss calculations
        loss_vals = _rolling_mean(run.metrics.get("loss", []), smooth_window)
        last_loss = _nan_safe(loss_vals)
        first_loss_arr = loss_vals[~np.isnan(loss_vals)]
        first_loss = float(first_loss_arr[0]) if len(first_loss_arr) else None
        delta = (last_loss - first_loss) if (last_loss is not None and first_loss is not None) else None

        # PPL calculations
        ppl_vals = _rolling_mean(run.metrics.get("ppl", []), smooth_window)
        last_ppl = _nan_safe(ppl_vals)

        fc = _loss_forecast(run, smooth_window, forecast_window, True)
        fc_str = f"{fc['predicted_loss']:.4f}" if fc and fc["target_step"] > fc["observed_step"] else dash

        toks_raw = run.metrics.get("tok/s") or run.metrics.get("tok_s") or []
        toks_arr = _to_arr(toks_raw)
        avg_toks = float(toks_arr[~np.isnan(toks_arr)].mean()) if len(toks_arr[~np.isnan(toks_arr)]) else None

        progress = 0.0
        if run.total_steps and run.steps:
            progress = run.steps[-1] / run.total_steps * 100

        loss_s  = f"{last_loss:.4f}"  if last_loss is not None else dash
        delta_s = (("+" if delta >= 0 else "") + f"{delta:.4f}") if delta is not None else dash
        ppl_s   = f"{last_ppl:.1f}" if last_ppl is not None else dash
        toks_s  = f"{avg_toks:,.0f}"  if avg_toks is not None else dash
        prog_s  = f"{progress:.1f}%"

        row = f"  {name:<28}{loss_s:>8}{delta_s:>8}{ppl_s:>8}{fc_str:>10}{toks_s:>8}{prog_s:>7}"
        print(box["v"] + row + " " * max(0, W - 2 - len(row)) + box["v"])

    print(box["bl"] + box["h"] * (W - 2) + box["br"])
    print()


# ── style ─────────────────────────────────────────────────────────────────────

def _apply_style() -> None:
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor":   "#0f1117",
        "axes.facecolor":     "#161b27",
        "axes.edgecolor":     "#2a2f3d",
        "axes.labelcolor":    "#c8cdd8",
        "axes.titlesize":     13,
        "axes.titleweight":   "semibold",
        "axes.titlecolor":    "#e2e6f0",
        "axes.labelsize":     10,
        "grid.color":         "#252b3b",
        "grid.linewidth":     0.7,
        "xtick.color":        "#7a8096",
        "ytick.color":        "#7a8096",
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.frameon":     True,
        "legend.framealpha":  0.25,
        "legend.facecolor":   "#1a2035",
        "legend.edgecolor":   "#2a2f3d",
        "legend.fontsize":    8.5,
        "lines.antialiased":  True,
    })


def _style_ax(ax) -> None:
    for spine in ax.spines.values():
        spine.set_color("#2a2f3d")
        spine.set_linewidth(0.8)
    ax.grid(True, alpha=0.5, linewidth=0.6)
    ax.set_axisbelow(True)


# ── progress bar ──────────────────────────────────────────────────────────────

def _draw_progress_bar(ax, run: RunSeries) -> None:
    if not run.total_steps or not run.steps:
        return
    pct = run.steps[-1] / run.total_steps
    xmin, xmax = ax.get_xlim()
    bar_y = ax.get_ylim()[0]
    width = xmax - xmin
    ax.add_patch(mpatches.FancyArrowPatch(
        (xmin, bar_y), (xmin + width, bar_y),
        arrowstyle="-", color="#2a2f3d", linewidth=3, zorder=1,
        transform=ax.transData, clip_on=False,
    ))
    ax.add_patch(mpatches.FancyArrowPatch(
        (xmin, bar_y), (xmin + pct * width, bar_y),
        arrowstyle="-", color="#4C9BE8", linewidth=3, zorder=2,
        transform=ax.transData, clip_on=False,
    ))
    ax.text(xmax, bar_y, f" {pct*100:.1f}%",
            va="center", ha="left", fontsize=8, color="#4C9BE8",
            transform=ax.transData)


# ── loss panel ────────────────────────────────────────────────────────────────

def _plot_with_gaps(ax, steps: Sequence[int], values: np.ndarray, *,
                    max_step_gap: int, color: str, lw: float, alpha: float,
                    label: Optional[str] = None, linestyle: str = "-",
                    zorder: int = 3) -> None:
    """Plot a series but break the line across large step gaps."""
    if not steps or len(values) == 0:
        return
    seg_x: List[float] = []
    seg_y: List[float] = []
    first = True
    prev_step = None
    for step, val in zip(steps, values):
        if np.isnan(val):
            if len(seg_x) >= 2:
                ax.plot(seg_x, seg_y, color=color, lw=lw, alpha=alpha,
                        label=label if first else None, linestyle=linestyle, zorder=zorder)
                first = False
            seg_x, seg_y = [], []
            prev_step = step
            continue
        if prev_step is not None and step - prev_step > max_step_gap and seg_x:
            if len(seg_x) >= 2:
                ax.plot(seg_x, seg_y, color=color, lw=lw, alpha=alpha,
                        label=label if first else None, linestyle=linestyle, zorder=zorder)
                first = False
            seg_x, seg_y = [], []
        seg_x.append(float(step))
        seg_y.append(float(val))
        prev_step = step
    if len(seg_x) >= 2:
        ax.plot(seg_x, seg_y, color=color, lw=lw, alpha=alpha,
                label=label if first else None, linestyle=linestyle, zorder=zorder)
    elif len(seg_x) == 1:
        ax.scatter(seg_x, seg_y, color=color, s=18, zorder=zorder,
                   label=label if first else None)


def _draw_loss_axis(ax, runs: Sequence[RunSeries], smooth_window: int, ema_alpha: float,
                    raw_alpha: float, forecast_window: int, forecast_enabled: bool,
                    forecast_use_smoothed: bool, show_raw: bool, show_ema: bool,
                    max_step_gap: int = DEFAULT_MAX_STEP_GAP) -> None:
    ax.clear()
    primary_idx = int(np.argmax([len(r.steps) for r in runs])) if runs else 0

    for i, run in enumerate(runs):
        color = _COLORS[i % len(_COLORS)]
        label = _short_name(run.name)
        raw = run.metrics.get("loss", [None] * len(run.steps))
        ma = _rolling_mean(raw, smooth_window)

        if show_raw:
            _plot_with_gaps(
                ax, run.steps, _to_arr(raw), max_step_gap=max_step_gap,
                color=color, lw=0.8, alpha=min(raw_alpha, 0.12), zorder=1,
            )

        last_val = _nan_safe(ma)
        legend_label = f"{label}  [{last_val:.4f}]" if last_val is not None else label
        _plot_with_gaps(
            ax, run.steps, ma, max_step_gap=max_step_gap,
            color=color, lw=2.2, alpha=0.97, label=legend_label, zorder=3,
        )

        # annotate last value on curve
        if last_val is not None:
            ax.annotate(
                f"{last_val:.4f}",
                xy=(run.steps[-1], last_val),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8, color=color, va="center",
            )

        # Mark min smoothed loss on the primary (longest) run.
        if i == primary_idx:
            valid_idx = np.where(~np.isnan(ma))[0]
            if len(valid_idx):
                min_i = int(valid_idx[np.argmin(ma[valid_idx])])
                ax.scatter(
                    [run.steps[min_i]], [ma[min_i]],
                    color="#7EC87E", s=36, zorder=6, marker="o",
                    label=f"min  [{ma[min_i]:.4f} @ {run.steps[min_i]:,}]",
                )

        if show_ema:
            _plot_with_gaps(
                ax, run.steps, _ema(raw, ema_alpha), max_step_gap=max_step_gap,
                color=color, lw=1.0, alpha=0.55, linestyle=":", zorder=2,
            )

        # val loss
        val_raw = run.metrics.get("val_loss", [])
        if any(v is not None for v in val_raw):
            val_ma = _rolling_mean(val_raw, smooth_window)
            val_last = _nan_safe(val_ma)
            vl_label = f"{label} val  [{val_last:.4f}]" if val_last is not None else f"{label} val"
            _plot_with_gaps(
                ax, run.steps, val_ma, max_step_gap=max_step_gap,
                color=color, lw=1.5, alpha=0.85, label=vl_label, linestyle="--", zorder=3,
            )

        # forecast
        if forecast_enabled:
            fc = _loss_forecast(run, smooth_window, forecast_window, forecast_use_smoothed)
            if fc and fc["target_step"] > fc["observed_step"]:
                lv = fc["last_smoothed"]
                pred = fc["predicted_loss"]
                ts = fc["target_step"]
                obs = fc["observed_step"]
                ax.plot([obs, ts], [lv, pred], color=color,
                        linestyle="-.", lw=1.2, alpha=0.6, zorder=2)
                # shaded forecast region
                ax.fill_betweenx([min(lv, pred), max(lv, pred)],
                                  obs, ts, color=color, alpha=0.06, zorder=1)
                ax.scatter([ts], [pred], color=color, s=28, zorder=5, marker="D")
                ax.annotate(
                    f"→ {pred:.4f}",
                    xy=(ts, pred), xytext=(6, 0),
                    textcoords="offset points",
                    fontsize=7.5, color=color, alpha=0.75, va="center",
                )

    ax.set_title("training loss", pad=8)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    _style_ax(ax)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))

    if runs:
        _draw_progress_bar(ax, runs[primary_idx])


# ── metric panel ──────────────────────────────────────────────────────────────

def _draw_metric_axis(ax, runs: Sequence[RunSeries], metric_name: str,
                      smooth_window: int, raw_alpha: float, show_raw: bool,
                      max_step_gap: int = DEFAULT_MAX_STEP_GAP) -> None:
    ax.clear()
    drawn = 0
    for i, run in enumerate(runs):
        color = _COLORS[i % len(_COLORS)]
        label = _short_name(run.name)
        values = _resolve_metric(run, metric_name)
        if not any(v is not None for v in values):
            continue
        drawn += 1
        arr = _to_arr(values)
        ma = _rolling_mean(values, smooth_window)

        if show_raw:
            _plot_with_gaps(
                ax, run.steps, arr, max_step_gap=max_step_gap,
                color=color, lw=0.8, alpha=min(raw_alpha, 0.14), zorder=1,
            )

        last_val = _nan_safe(ma)
        valid = arr[~np.isnan(arr)]
        stats = ""
        if len(valid):
            stats = f"  μ={valid.mean():.2g}  σ={valid.std():.2g}"
        legend_label = (f"{label}  [{last_val:.4g}]{stats}"
                        if last_val is not None else label)
        _plot_with_gaps(
            ax, run.steps, ma, max_step_gap=max_step_gap,
            color=color, lw=2.0, alpha=0.97, label=legend_label, zorder=3,
        )

        if last_val is not None:
            ax.annotate(
                f"{last_val:.4g}",
                xy=(run.steps[-1], last_val),
                xytext=(6, 0), textcoords="offset points",
                fontsize=8, color=color, va="center",
            )

    ax.set_title(f"metric: {metric_name}", pad=8)
    ax.set_xlabel("step")
    ax.set_ylabel(metric_name)
    _style_ax(ax)
    if drawn:
        ax.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
    else:
        ax.text(0.5, 0.5, f"metric '{metric_name}' not found",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="#666677")


# ── figure layout ─────────────────────────────────────────────────────────────

def _render_figure(fig, runs: Sequence[RunSeries], metric_name: str,
                   smooth_window: int, ema_alpha: float, raw_alpha: float,
                   forecast_window: int, forecast_enabled: bool,
                   forecast_use_smoothed: bool, show_raw_loss: bool,
                   show_ema_loss: bool, show_raw_metric: bool,
                   max_step_gap: int = DEFAULT_MAX_STEP_GAP) -> None:
    if len(fig.axes) < 4:
        fig.clf()
        ax_loss, ax_ppl, ax_lr, ax_metric = fig.subplots(
            4, 1, gridspec_kw={"height_ratios": [1.5, 1.0, 1.0, 1.0]})
        fig.subplots_adjust(right=0.82, hspace=0.45, left=0.07,
                            top=0.94, bottom=0.07)
    else:
        ax_loss, ax_ppl, ax_lr, ax_metric = fig.axes[:4]

    _draw_loss_axis(ax_loss, runs, smooth_window, ema_alpha, raw_alpha,
                    forecast_window, forecast_enabled, forecast_use_smoothed,
                    show_raw_loss, show_ema_loss, max_step_gap=max_step_gap)
    _draw_metric_axis(ax_ppl, runs, "perplexity", smooth_window, raw_alpha, show_raw_metric,
                      max_step_gap=max_step_gap)
    _draw_metric_axis(ax_lr, runs, "learning_rate", smooth_window, raw_alpha, show_raw_metric,
                      max_step_gap=max_step_gap)
    _draw_metric_axis(ax_metric, runs, metric_name, smooth_window,
                      raw_alpha, show_raw_metric, max_step_gap=max_step_gap)
    fig.canvas.draw_idle()


# ── file discovery ─────────────────────────────────────────────────────────────

def _count_train_lines(path: Path, sample_bytes: int = 2_000_000) -> int:
    """Cheap richness estimate: count step= lines near the end of the file."""
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > sample_bytes:
                f.seek(-sample_bytes, 2)
            data = f.read().decode("utf-8", errors="ignore")
        return data.count("step=")
    except Exception:
        return 0


def _find_default_logs(log_dir: Path) -> List[Path]:
    """Return training logs, preferring richer .log files under log_dir."""
    log_paths: List[Path] = []
    if log_dir.exists():
        log_paths = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)

    if log_paths:
        return log_paths

    # Fall back to structured metrics exports only when no .log files exist.
    search_dir = log_dir.parent if log_dir.name == "logs" else log_dir
    fallback: List[Path] = []
    for name in ("training_metrics_latest.jsonl", "training_metrics_latest.csv"):
        candidate = search_dir / name
        if candidate.exists():
            fallback.append(candidate)
    return fallback


def _pick_default_log(paths: List[Path]) -> List[Path]:
    """Prefer aggregate training.log, else the richest log by step= density."""
    if not paths:
        return []
    for p in paths:
        if p.name == "training.log":
            return [p]
    ranked = sorted(paths, key=lambda p: (_count_train_lines(p), p.stat().st_mtime), reverse=True)
    return [ranked[0]]


def _pick_logs_interactively(paths: List[Path]) -> List[Path]:
    if not paths:
        return []
    print("Available logs:")
    for i, p in enumerate(paths, 1):
        print(f"  {i}. {p.name}")
    raw = input("Select log numbers (comma-separated, blank = best default): ").strip()
    if not raw:
        return _pick_default_log(paths)
    selected = []
    for part in raw.split(","):
        try:
            idx = int(part.strip()) - 1
            if 0 <= idx < len(paths):
                selected.append(paths[idx])
        except Exception:
            pass
    return selected or _pick_default_log(paths)


# ── main loop ──────────────────────────────────────────────────────────────────

def plot_runs_liveable(
    runs: Sequence[RunSeries],
    metric_name: str,
    smooth_window: int,
    ema_alpha: float,
    raw_alpha: float,
    forecast_window: int,
    forecast_enabled: bool,
    forecast_use_smoothed: bool,
    show_raw_loss: bool,
    show_ema_loss: bool,
    show_raw_metric: bool,
    live: bool,
    refresh_seconds: float,
    source_paths: List[Path],
    save_path: Optional[Path],
    all_runs: bool = False,
    min_points: int = DEFAULT_MIN_POINTS,
    max_step_gap: int = DEFAULT_MAX_STEP_GAP,
    show: bool = True,
) -> None:
    _apply_style()
    plt.ion() if live else plt.ioff()
    fig = plt.figure(figsize=(15.0, 8.8))
    fig.suptitle("Training run monitor", fontsize=14, color="#e2e6f0",
                 fontweight="semibold", y=0.98)

    last_summary_key: Optional[Tuple] = None

    while True:
        current_runs = (
            _load_runs(
                source_paths,
                all_runs=all_runs,
                use_cache=live,
                min_points=min_points,
                max_step_gap=max_step_gap,
            )
            if live else list(runs)
        )
        if not current_runs:
            print("No valid training logs found.", file=sys.stderr)
            return

        # Live mode: only reprint the summary when the latest step/loss changes.
        summary_key = tuple(
            (r.name, r.steps[-1] if r.steps else 0,
             r.metrics.get("loss", [None])[-1] if r.metrics.get("loss") else None)
            for r in current_runs
        )
        if not live or summary_key != last_summary_key:
            print_summary(current_runs, smooth_window, forecast_window)
            last_summary_key = summary_key

        _render_figure(
            fig, current_runs, metric_name, smooth_window, ema_alpha, raw_alpha,
            forecast_window, forecast_enabled, forecast_use_smoothed,
            show_raw_loss, show_ema_loss, show_raw_metric,
            max_step_gap=max_step_gap,
        )

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            if not live:
                print(f"Saved -> {save_path}")

        if not live:
            # Headless save (e.g. train.py --plot / --save without --show) must not block.
            if show or not save_path:
                plt.show()
            else:
                plt.close(fig)
            return

        plt.pause(max(0.05, refresh_seconds))
        if not plt.fignum_exists(fig.number):
            break


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Training log plotter — enhanced edition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--logs", nargs="*", help="Explicit log file paths")
    parser.add_argument("--log-dir", default="output/logs",
                        help="Directory to scan for .log files (output/logs/training.log by default)")
    parser.add_argument("--multi", action="store_true",
                        help="Load ALL logs in log-dir (compare runs)")
    parser.add_argument("--all-runs", action="store_true",
                        help="Show every non-smoke training run found in the log; "
                             "default is the longest substantive run only")
    parser.add_argument("--keep-short", action="store_true",
                        help="Include short smoke runs (< min-points)")
    parser.add_argument("--min-points", type=int, default=DEFAULT_MIN_POINTS,
                        help="Drop runs with fewer logged points")
    parser.add_argument("--max-step-gap", type=int, default=DEFAULT_MAX_STEP_GAP,
                        help="Break curves across step gaps larger than this")
    parser.add_argument("--metric", default="tok/s",
                        help="Metric shown in the bottom panel")
    parser.add_argument("--smooth-window", type=int, default=21,
                        help="Rolling-mean window (steps)")
    parser.add_argument("--ema-alpha", type=float, default=0.08,
                        help="EMA decay factor (0–1)")
    parser.add_argument("--raw-alpha", type=float, default=0.10,
                        help="Opacity for raw (unsmoothed) traces")
    parser.add_argument("--forecast-window", type=int, default=40,
                        help="Tail steps used to fit the forecast line")
    parser.add_argument("--no-forecast", action="store_true",
                        help="Hide the loss forecast")
    parser.add_argument("--forecast-raw", action="store_true",
                        help="Fit forecast on raw loss instead of smoothed")
    parser.add_argument("--show-raw-loss", action="store_true",
                        help="Overlay raw (noisy) loss trace")
    parser.add_argument("--show-ema-loss", action="store_true",
                        help="Overlay EMA loss trace")
    parser.add_argument("--hide-raw-metric", action="store_true",
                        help="Hide raw metric trace in bottom panel")
    parser.add_argument("--select", action="store_true",
                        help="Interactively choose which logs to load")
    parser.add_argument("--live", action="store_true",
                        help="Continuously refresh the chart from disk")
    parser.add_argument("--refresh-seconds", type=float, default=1.0,
                        help="Refresh interval in live mode")
    parser.add_argument("--save", metavar="PATH",
                        help="Save the figure to this path (PNG/PDF/SVG)")
    parser.add_argument("--show", action="store_true",
                        help="Display the figure window (implied when --save is omitted)")
    args = parser.parse_args()

    min_points = 2 if args.keep_short else max(2, args.min_points)
    max_step_gap = max(1, args.max_step_gap)

    if args.logs:
        paths = [Path(p) for p in args.logs]
    else:
        all_paths = _find_default_logs(Path(args.log_dir))
        if args.select:
            paths = _pick_logs_interactively(all_paths)
        elif args.multi:
            paths = all_paths
        else:
            paths = _pick_default_log(all_paths)

    runs = _load_runs(
        paths,
        all_runs=args.all_runs,
        min_points=min_points,
        max_step_gap=max_step_gap,
    )
    if not runs:
        raise SystemExit("No valid training logs found.")

    # Show by default unless we're only saving to disk.
    do_show = args.show or args.save is None

    plot_runs_liveable(
        runs=runs,
        all_runs=args.all_runs,
        metric_name=args.metric,
        smooth_window=max(1, args.smooth_window),
        ema_alpha=args.ema_alpha,
        raw_alpha=args.raw_alpha,
        forecast_window=max(5, args.forecast_window),
        forecast_enabled=not args.no_forecast,
        forecast_use_smoothed=not args.forecast_raw,
        show_raw_loss=args.show_raw_loss,
        show_ema_loss=args.show_ema_loss,
        show_raw_metric=not args.hide_raw_metric,
        live=args.live,
        refresh_seconds=max(0.05, args.refresh_seconds),
        source_paths=paths,
        save_path=Path(args.save) if args.save else None,
        min_points=min_points,
        max_step_gap=max_step_gap,
        show=do_show and not args.live,
    )


if __name__ == "__main__":
    main()

