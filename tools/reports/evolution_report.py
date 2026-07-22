"""
tools/reports/evolution_report.py

Stage 3.8: build a local HTML evolution report from baseline JSON artifacts.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BASELINE_DIR = ROOT / "output" / "baselines"


def _load(name: str):
    path = BASELINE_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _row(cells):
    return "<tr>" + "".join(f"<td>{html.escape(str(c))}</td>" for c in cells) + "</tr>"


def build_html() -> str:
    s31 = _load("stage31_baseline.json")
    s32 = _load("stage32_kv_generate.json")
    s33 = _load("stage33_bpe_protocol.json")
    s34 = _load("stage34_activation_account.json")
    s35 = _load("stage35_fp16_storage.json")
    s36 = _load("stage36_allocator.json")
    s37 = _load("stage37_timeline_meta.json")

    rows = []
    if s31:
        rows.append(_row(["3.1 train", s31.get("runtime", {}).get("tokens_per_sec"),
                          s31.get("runtime", {}).get("step_ms"),
                          s31.get("memory", {}).get("device_used_mb"),
                          s31.get("quality", {}).get("train_loss")]))
    if s32:
        rows.append(_row(["3.2 generate KV",
                          s32.get("after_kv", {}).get("tokens_per_sec"),
                          s32.get("after_kv", {}).get("wall_s"),
                          s32.get("after_kv", {}).get("kv_peak_mb"),
                          f"speedup {s32.get('speedup')}"]))
    if s33:
        rows.append(_row(["3.3 BPE chars/tok",
                          s33.get("bpe", {}).get("chars_per_token"),
                          s33.get("bpe", {}).get("step_ms_mean"),
                          s33.get("bpe", {}).get("vocab_size"),
                          s33.get("bpe", {}).get("train_loss_last")]))
    if s34:
        rows.append(_row(["3.4 activations",
                          s34.get("largest_activation_bucket"),
                          s34.get("activation_cache_mb"),
                          s34.get("device_used_mb_after_forward"),
                          s34.get("parameter_mb")]))
    if s35:
        rows.append(_row(["3.5 FP16 storage",
                          s35.get("estimated_activation_savings_mb"),
                          s35.get("lm_head_grad_maxdiff"),
                          s35.get("parity_ok"),
                          s35.get("loss_fp16_storage_path")]))
    if s36:
        st = s36.get("allocator_stats", {})
        rows.append(_row(["3.6 allocator",
                          st.get("reuse_count"),
                          st.get("alloc_count"),
                          st.get("peak_live_bytes"),
                          s36.get("reuse_observed")]))
    if s37:
        sm = s37.get("summary", {})
        rows.append(_row(["3.7 timeline",
                          sm.get("event_count"),
                          sm.get("total_ms"),
                          list((sm.get("by_name_ms") or {}).keys())[:3],
                          s37.get("timeline_path")]))

    body = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>llm-gpu-8 Stage 3 evolution</title>
<style>
body {{ font-family: Georgia, 'Times New Roman', serif; margin: 2rem; background: #f7f4ef; color: #1c1a16; }}
h1 {{ font-size: 1.8rem; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1rem; background: #fff; }}
th, td {{ border: 1px solid #cfc8bc; padding: 0.5rem 0.7rem; text-align: left; font-size: 0.95rem; }}
th {{ background: #ebe4d8; }}
.note {{ margin-top: 1.5rem; max-width: 48rem; line-height: 1.45; }}
code {{ background: #efe9df; padding: 0.1rem 0.3rem; }}
</style>
</head>
<body>
<h1>llm-gpu-8 evolution report</h1>
<p>Generated {html.escape(datetime.now(timezone.utc).isoformat())}</p>
<p class="note">Comparison root: <code>stage31_baseline.json</code>. Each later stage should answer:
did runtime efficiency, model efficiency, or developer confidence improve — and can the baseline prove it?</p>
<table>
<thead><tr><th>Stage</th><th>Metric A</th><th>Metric B</th><th>Metric C</th><th>Metric D</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
<p class="note">Artifacts live under <code>output/baselines/</code>. Open this file locally in a browser.</p>
</body>
</html>
"""
    return body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="output/reports/evolution.html")
    args = ap.parse_args()
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
