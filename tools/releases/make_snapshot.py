"""
tools/releases/make_snapshot.py

Assemble a known-good release snapshot from frozen Stage 3 baselines + parity.

Usage:
    python tools/releases/make_snapshot.py
    python tools/releases/make_snapshot.py --tag v0.1.1
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINES = ROOT / "output" / "baselines"
REPORTS = ROOT / "output" / "reports"
RELEASES = ROOT / "output" / "releases"
PY = ROOT / "venv" / "Scripts" / "python.exe"
if not PY.exists():
    PY = Path(sys.executable)


def _load(name: str):
    path = BASELINES / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_parity(out_txt: Path) -> dict:
    proc = subprocess.run(
        [str(PY), "-m", "tests.parity.run_parity"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    out_txt.write_text(text, encoding="utf-8")
    ok = proc.returncode == 0 and "OK" in text
    # unittest summary line e.g. "Ran 10 tests in 0.538s"
    ran = None
    for line in text.splitlines():
        if line.startswith("Ran "):
            ran = line.strip()
    return {"ok": ok, "returncode": proc.returncode, "summary": ran, "path": str(out_txt)}


def build(tag: str) -> Path:
    dest = RELEASES / tag
    dest.mkdir(parents=True, exist_ok=True)

    s31 = _load("stage31_baseline.json") or {}
    s32 = _load("stage32_kv_generate.json") or {}
    s33 = _load("stage33_bpe_protocol.json") or {}
    s34 = _load("stage34_activation_account.json") or {}
    s35 = _load("stage35_fp16_storage.json") or {}
    s36 = _load("stage36_allocator.json") or {}
    s37 = _load("stage37_timeline_meta.json") or _load("stage37_timeline.json") or {}

    runtime = {
        "tag": tag,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "version_file": (ROOT / "VERSION").read_text(encoding="utf-8").strip()
        if (ROOT / "VERSION").exists()
        else None,
        "statement": "Known-good runtime state for stabilization / regression comparison.",
        "training": {
            "source": "stage31_baseline.json",
            "tokens_per_sec": (s31.get("runtime") or {}).get("tokens_per_sec"),
            "step_ms": (s31.get("runtime") or {}).get("step_ms"),
            "device_used_mb": (s31.get("memory") or {}).get("device_used_mb"),
            "scratch_peak_mb": (s31.get("runtime") or {}).get("scratch_peak_mb"),
            "train_loss": (s31.get("quality") or {}).get("train_loss"),
            "checkpoint_step": (s31.get("source") or {}).get("step"),
        },
        "hardware": s31.get("hardware"),
        "model": s31.get("model"),
        "gates": {
            "parity_required": "10/10",
            "compare_against": [
                "stage31_baseline.json",
                "stage32_kv_generate.json",
                "stage34_activation_account.json",
                "stage35_fp16_storage.json",
                "stage36_allocator.json",
                "stage37_timeline.json",
            ],
        },
    }

    quality = {
        "tag": tag,
        "source": "stage31_baseline.json + stage33_bpe_protocol.json",
        "train_loss": (s31.get("quality") or {}).get("train_loss"),
        "train_ppl": (s31.get("quality") or {}).get("train_ppl"),
        "val_loss_prior_101k": (s31.get("quality") or {}).get("val_loss_prior_readme_101k"),
        "quality_score_prior_101k": (s31.get("quality") or {}).get("quality_score_prior_readme_101k"),
        "bpe_experiment_note": (s33 or {}).get("note"),
        "bpe_chars_per_token": (s33.get("bpe") or {}).get("chars_per_token"),
    }

    generation = {
        "tag": tag,
        "source": "stage32_kv_generate.json",
        "checkpoint": s32.get("checkpoint"),
        "max_new_tokens": s32.get("max_new_tokens"),
        "before_no_kv": s32.get("before_no_kv"),
        "after_kv": s32.get("after_kv"),
        "speedup": s32.get("speedup"),
        "determinism": s32.get("determinism"),
    }

    memory = {
        "tag": tag,
        "activation_accounting": {
            "source": "stage34_activation_account.json",
            "activation_cache_mb": s34.get("activation_cache_mb"),
            "buckets_mb": s34.get("buckets_mb"),
            "largest_activation_bucket": s34.get("largest_activation_bucket"),
            "parameter_mb": s34.get("parameter_mb"),
            "device_used_mb_after_forward": s34.get("device_used_mb_after_forward"),
        },
        "fp16_storage": {
            "source": "stage35_fp16_storage.json",
            "estimated_savings_mb": s35.get("estimated_activation_savings_mb"),
            "parity_ok": s35.get("parity_ok"),
            "lm_head_grad_maxdiff": s35.get("lm_head_grad_maxdiff"),
        },
        "allocator": {
            "source": "stage36_allocator.json",
            "stats": s36.get("allocator_stats"),
            "reuse_observed": s36.get("reuse_observed"),
        },
        "timeline": {
            "source": "stage37_timeline_meta.json",
            "summary": s37.get("summary"),
            "timeline_path": s37.get("timeline_path"),
        },
        "stage31_scratch_peak_mb": (s31.get("runtime") or {}).get("scratch_peak_mb"),
    }

    (dest / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    (dest / "quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    (dest / "generation.json").write_text(json.dumps(generation, indent=2), encoding="utf-8")
    (dest / "memory.json").write_text(json.dumps(memory, indent=2), encoding="utf-8")

    parity = _run_parity(dest / "parity.txt")
    manifest = {
        "tag": tag,
        "captured_at": runtime["captured_at"],
        "parity": parity,
        "files": [
            "runtime.json",
            "quality.json",
            "generation.json",
            "memory.json",
            "parity.txt",
            "evolution.html",
            "MANIFEST.json",
        ],
        "statement": "This is the known-good runtime state.",
    }
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    evo_src = REPORTS / "evolution.html"
    if evo_src.exists():
        shutil.copy2(evo_src, dest / "evolution.html")
    else:
        # regenerate if missing
        subprocess.run(
            [str(PY), str(ROOT / "tools" / "reports" / "evolution_report.py"),
             "--out", str(dest / "evolution.html")],
            cwd=str(ROOT),
            check=False,
        )

    # Pointer copies of source baselines for auditability
    audit = dest / "sources"
    audit.mkdir(exist_ok=True)
    for name in (
        "stage31_baseline.json",
        "stage32_kv_generate.json",
        "stage34_activation_account.json",
        "stage35_fp16_storage.json",
        "stage36_allocator.json",
        "stage37_timeline.json",
    ):
        src = BASELINES / name
        if src.exists():
            shutil.copy2(src, audit / name)

    if not parity["ok"]:
        raise SystemExit(f"Parity gate failed; snapshot at {dest} is incomplete for release")

    print(json.dumps({"wrote": str(dest), "parity": parity}, indent=2))
    return dest


def main():
    ap = argparse.ArgumentParser(description="Create known-good release snapshot")
    ap.add_argument("--tag", type=str, default="v0.1.1")
    args = ap.parse_args()
    build(args.tag)


if __name__ == "__main__":
    main()
