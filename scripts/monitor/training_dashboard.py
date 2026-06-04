#!/usr/bin/env python3
"""Tiny live dashboard for Auralis training runs.

No third-party web framework is required. The server reads training logs,
learning-trace JSON/HTML reports, and semantic reports from the repo, then
serves a browser UI with live charts.

Usage:
    python scripts/monitor/training_dashboard.py --host 0.0.0.0 --port 8765
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import time
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse


STEP_RE = re.compile(
    r"step\s+(?P<step>\d+)\s+\|\s+loss\s+(?P<loss>[-+0-9.eE]+)\s+\|\s+"
    r"lr\s+(?P<lr>[-+0-9.eE]+)\s+\|\s+grad_norm\s+(?P<grad>[-+0-9.eE]+)\s+\|\s+"
    r"tok/s\s+(?P<tps>[-+0-9.eE]+)k\s+\|\s+data\s+(?P<data>[-+0-9.eE]+)%"
    r"(?:\s+\|\s+vram\s+(?P<vram_alloc>[-+0-9.eE]+)/(?P<vram_peak>[-+0-9.eE]+)GB)?"
)
EVAL_RE = re.compile(r"eval @ step\s+(?P<step>\d+):\s+(?P<body>.*)")
KV_RE = re.compile(r"(?P<key>[A-Za-z0-9_/-]+)=(?P<value>[-+0-9.eE]+)")
MIX_RE = re.compile(r"train expected rows/batch per language:\s+(?P<mix>\{.*\})")
MANIFEST_RE = re.compile(r"manifest:\s+(?P<path>.+)")

LN2 = 0.6931471805599453
# Measured tokens/byte (from eval_diagnostic.py). The training log computed bpb
# with a hand-guessed German tokens/byte of 0.2338 that inflated bpb_german ~33%.
# Recompute bpb here from per-language val_loss with the honest measured values.
CORRECT_TPB = {"german": 0.1757, "english": 0.1962}


@dataclass
class TrainPoint:
    step: int
    loss: float
    lr: float
    grad_norm: float
    tok_s: float
    data_pct: float
    vram_alloc_gb: float | None = None
    vram_peak_gb: float | None = None


def _float_or_none(value: str | None) -> float | None:
    return float(value) if value not in (None, "") else None


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def safe_relative(root: Path, rel: str) -> Path | None:
    try:
        candidate = (root / rel).resolve()
        candidate.relative_to(root.resolve())
        return candidate
    except Exception:
        return None


def default_log_dirs(root: Path) -> list[Path]:
    candidates = [
        root / "logs",
        root.parent / "NEWGPT" / "v2data" / "logs",
    ]
    return [p for p in candidates if p.exists()]


def list_logs(root: Path, log_dirs: list[Path]) -> list[dict[str, object]]:
    out = []
    for logs_dir in log_dirs:
        for p in logs_dir.glob("*.log"):
            try:
                rel = str(p.relative_to(root)).replace("\\", "/")
            except ValueError:
                rel = "abs:" + str(p.resolve())
            out.append({
                "name": f"{p.name}  [{logs_dir.name}]",
                "path": rel,
                "bytes": p.stat().st_size,
                "mtime": p.stat().st_mtime,
            })
    return sorted(out, key=lambda x: float(x["mtime"]), reverse=True)


def resolve_log_path(root: Path, log_dirs: list[Path], selected: str) -> Path | None:
    if selected.startswith("abs:"):
        p = Path(selected.removeprefix("abs:")).resolve()
        for log_dir in log_dirs:
            try:
                p.relative_to(log_dir.resolve())
                return p
            except ValueError:
                continue
        return None
    return safe_relative(root, selected)


def parse_log(path: Path, max_lines: int = 6000) -> dict[str, object]:
    if not path.exists():
        return {"error": f"log not found: {path}"}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]

    train: list[TrainPoint] = []
    evals: list[dict[str, float]] = []
    health: list[str] = []
    mix = ""
    manifest = ""
    last_error = ""

    for line in lines:
        if "health[" in line or "AUTO-STOP" in line:
            health.append(line.strip())
        if any(token in line for token in ("Traceback", "RuntimeError", "CUDA out of memory", "non-finite")):
            last_error = line.strip()
        if m := MIX_RE.search(line):
            mix = m.group("mix")
        if m := MANIFEST_RE.search(line):
            manifest = m.group("path").strip()
        if m := STEP_RE.search(line):
            train.append(TrainPoint(
                step=int(m.group("step")),
                loss=float(m.group("loss")),
                lr=float(m.group("lr")),
                grad_norm=float(m.group("grad")),
                tok_s=float(m.group("tps")) * 1000.0,
                data_pct=float(m.group("data")),
                vram_alloc_gb=_float_or_none(m.group("vram_alloc")),
                vram_peak_gb=_float_or_none(m.group("vram_peak")),
            ))
        elif m := EVAL_RE.search(line):
            row: dict[str, float] = {"step": float(m.group("step"))}
            for kv in KV_RE.finditer(m.group("body")):
                row[kv.group("key").replace("/", "_")] = float(kv.group("value"))
            # Honest bpb: recompute from per-language val_loss with the MEASURED
            # tokens/byte, overriding the logged values (which used the wrong tpb).
            for lang in ("german", "english"):
                vl = row.get(f"val_loss_{lang}")
                tpb = CORRECT_TPB.get(lang)
                if vl is not None and tpb:
                    row[f"bpb_{lang}"] = vl * tpb / LN2
            de_b, en_b = row.get("bpb_german"), row.get("bpb_english")
            if de_b and en_b:
                row["bpb_gap_max"] = max(de_b, en_b) / min(de_b, en_b)
            evals.append(row)

    latest_train = asdict(train[-1]) if train else None
    latest_eval = evals[-1] if evals else None
    mtime = path.stat().st_mtime
    return {
        "path": str(path),
        "name": path.name,
        "mtime": mtime,
        "age_seconds": time.time() - mtime,
        "active_guess": (time.time() - mtime) < 180,
        "mix": mix,
        "manifest": manifest,
        "train": [asdict(p) for p in train],
        "evals": evals,
        "latest_train": latest_train,
        "latest_eval": latest_eval,
        "health": health[-30:],
        "last_error": last_error,
        "tail": lines[-80:],
    }


def learning_reports(root: Path) -> list[dict[str, object]]:
    base = root / "reports" / "learning_trace"
    if not base.exists():
        return []
    groups: dict[str, dict[str, object]] = {}
    for p in base.iterdir():
        if not p.is_file():
            continue
        name = p.name
        stem = p.stem
        if stem.endswith("_neuro"):
            key = stem.removesuffix("_neuro")
            kind = "neuro_html"
        elif stem.endswith("_diag"):
            key = stem.removesuffix("_diag")
            kind = "diag_json"
        else:
            key = stem
            kind = p.suffix.lstrip(".") or "file"
        group = groups.setdefault(key, {"key": key, "mtime": 0.0, "files": {}})
        group["mtime"] = max(float(group["mtime"]), p.stat().st_mtime)
        group["files"][kind] = str(p.relative_to(root)).replace("\\", "/")
    return sorted(groups.values(), key=lambda x: float(x["mtime"]), reverse=True)[:50]


def _probe_to_concept(probe: dict[str, Any]) -> dict[str, Any]:
    """One probe row -> a compact concept dict (with derived status)."""
    margin = probe.get("margin")
    margin_f = float(margin) if isinstance(margin, int | float) else None
    forbidden = probe.get("forbidden_hits") or []
    if forbidden:
        status = "danger"
    elif margin_f is None:
        status = "unknown"
    elif margin_f >= 0.75:
        status = "strong"
    elif margin_f >= 0.0:
        status = "watch"
    else:
        status = "weak"
    return {
        "id": str(probe.get("id", "unknown")),
        "category": str(probe.get("category", "uncategorized")),
        "margin": margin_f,
        "target_nll": probe.get("target_nll"),
        "negative_nll": probe.get("negative_nll"),
        "forbidden": forbidden,
        "answer": str(probe.get("answer") or "").strip(),
        "status": status,
    }


def representative_snapshot(root: Path) -> dict[str, object] | None:
    """Honest cross-source bpb from eval_diagnostic.py (diag/step0_v3best.json).

    The run's own per-language val is the easy TAIL of each .bin (wiki-German;
    an anomalously trivial English source) and its German bpb used the wrong
    tokens/byte. This snapshot samples across ALL sources with the measured
    tokens/byte → the number to trust. Feeds the headline tiles + the gap.
    """
    p = root / "diag" / "step0_v3best.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
        m = (data.get("modes") or {}).get("kernels_on_bf16") or {}
        de = (m.get("german/full") or {}).get("bpb_measured")
        en = (m.get("english/full") or {}).get("bpb_measured")
        if not (de and en):
            return None
        return {
            "de": de,
            "en": en,
            "gap": max(de, en) / min(de, en),
            "de_tail": (m.get("german/tail") or {}).get("bpb_measured"),
            "en_tail": (m.get("english/tail") or {}).get("bpb_measured"),
            "step": data.get("source_step"),
            "mtime": p.stat().st_mtime,
        }
    except Exception:
        return None


def latest_neuro_summary(root: Path) -> dict[str, object]:
    reports = learning_reports(root)
    json_rel = None
    for report in reports:
        files = report.get("files", {})
        if isinstance(files, dict) and "json" in files:
            json_rel = files["json"]
            break
    if not json_rel:
        return {"concepts": [], "categories": {}}
    p = safe_relative(root, str(json_rel))
    if not p or not p.exists():
        return {"concepts": [], "categories": {}}
    try:
        data = json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception as exc:
        return {"error": str(exc), "concepts": [], "categories": {}}

    history = data.get("history") or []
    latest = history[-1] if history else {}
    concepts = [_probe_to_concept(pr) for pr in (latest.get("probes") or [])]

    categories: dict[str, dict[str, float]] = {}
    for c in concepts:
        bucket = categories.setdefault(c["category"], {"count": 0.0, "margin_sum": 0.0})
        bucket["count"] += 1.0
        if c["margin"] is not None:
            bucket["margin_sum"] += c["margin"]
    for cat, bucket in categories.items():
        bucket["avg_margin"] = bucket["margin_sum"] / max(bucket["count"], 1.0)

    # Per-step frames for the timeline scrubber (watch the graph grow).
    frames = [
        {"step": h.get("step"), "val_loss": h.get("val_loss"),
         "concepts": [_probe_to_concept(pr) for pr in (h.get("probes") or [])]}
        for h in history[-120:]
    ]
    concepts_sorted = sorted(
        concepts, key=lambda c: (c["margin"] is None, c["margin"] if c["margin"] is not None else 0.0)
    )
    return {
        "source": json_rel,
        "step": latest.get("step"),
        "val_loss": latest.get("val_loss"),
        "concepts": concepts_sorted[:80],
        "categories": categories,
        "frames": frames,
    }


INDEX_HTML = r"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Auralis · Training Monitor</title>
  <style>
    :root {
      --bg0:#070b0d; --bg1:#0b1013; --panel:rgba(22,29,33,.72); --panel-solid:#161d21;
      --stroke:rgba(120,150,160,.16); --stroke-soft:rgba(120,150,160,.10);
      --text:#eaf1f3; --muted:#8ca0a8; --faint:#5d6f77;
      --de:#3ddc84; --en:#46c6ff; --accent:#7af0c8;
      --green:#35c77b; --yellow:#f4b740; --red:#ff6b6b; --blue:#5aa7ff;
      --glow-de:rgba(61,220,132,.35); --glow-en:rgba(70,198,255,.35);
    }
    * { box-sizing:border-box; }
    html,body { height:100%; }
    body {
      margin:0; color:var(--text); font-size:14px;
      font-family:"Inter",ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
      background:var(--bg0);
      background-image:
        radial-gradient(1100px 500px at 12% -8%, rgba(61,220,132,.10), transparent 60%),
        radial-gradient(1000px 520px at 92% -10%, rgba(70,198,255,.10), transparent 60%),
        linear-gradient(180deg,#0a0f12 0%, #070b0d 100%);
      background-attachment:fixed;
      -webkit-font-smoothing:antialiased;
    }
    ::-webkit-scrollbar { width:10px; height:10px; }
    ::-webkit-scrollbar-thumb { background:rgba(120,150,160,.22); border-radius:20px; }
    ::-webkit-scrollbar-track { background:transparent; }

    /* ---------- top refresh bar ---------- */
    #refreshBar { position:fixed; top:0; left:0; height:2px; width:0%;
      background:linear-gradient(90deg,var(--de),var(--en)); z-index:50;
      box-shadow:0 0 12px var(--glow-en); transition:width .25s linear; }

    header {
      position:sticky; top:0; z-index:20;
      display:flex; align-items:center; justify-content:space-between; gap:16px;
      padding:16px 24px; flex-wrap:wrap;
      background:rgba(8,12,14,.72); backdrop-filter:blur(14px);
      border-bottom:1px solid var(--stroke);
    }
    .brand { display:flex; align-items:center; gap:13px; }
    .logo { width:36px; height:36px; border-radius:11px; position:relative; flex:0 0 auto;
      background:conic-gradient(from 200deg,var(--de),var(--en),var(--accent),var(--de));
      box-shadow:0 0 22px rgba(70,198,255,.35); animation:spin 9s linear infinite; }
    .logo::after { content:""; position:absolute; inset:3px; border-radius:8px; background:#0a0f12; }
    .logo i { position:absolute; inset:0; display:grid; place-items:center; font-style:normal;
      font-weight:800; font-size:15px; z-index:2;
      background:linear-gradient(120deg,var(--de),var(--en)); -webkit-background-clip:text; background-clip:text; color:transparent; }
    h1 { margin:0; font-size:18px; font-weight:750; letter-spacing:.2px; }
    h1 span { background:linear-gradient(110deg,var(--de),var(--en)); -webkit-background-clip:text; background-clip:text; color:transparent; }
    .sub { color:var(--muted); font-size:12px; margin-top:2px; }
    .controls { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    select,button { font:inherit; color:var(--text); background:rgba(29,36,40,.85);
      border:1px solid var(--stroke); border-radius:9px; padding:8px 11px; font-size:13px; outline:none;
      transition:border-color .2s, box-shadow .2s, transform .1s; }
    select:hover,button:hover { border-color:rgba(70,198,255,.5); }
    button { cursor:pointer; }
    button:active { transform:scale(.97); }
    .btn-accent { background:linear-gradient(120deg,rgba(61,220,132,.18),rgba(70,198,255,.18));
      border-color:rgba(70,198,255,.35); }

    .status { display:inline-flex; align-items:center; gap:8px; padding:7px 12px; border-radius:999px;
      border:1px solid var(--stroke); color:var(--muted); font-size:12px; font-weight:600;
      background:rgba(20,26,30,.7); }
    .dot { width:9px; height:9px; border-radius:50%; background:var(--faint); position:relative; }
    .active .dot { background:var(--green); box-shadow:0 0 0 0 rgba(53,199,123,.6); animation:pulse 1.8s infinite; }
    .active { color:var(--green); border-color:rgba(53,199,123,.4); }
    .warn .dot { background:var(--yellow); } .warn { color:var(--yellow); border-color:rgba(244,183,64,.4); }
    .bad .dot { background:var(--red); box-shadow:0 0 0 0 rgba(255,107,107,.6); animation:pulse 1.4s infinite; }
    .bad { color:var(--red); border-color:rgba(255,107,107,.45); }

    main { padding:22px; max-width:1560px; margin:0 auto; display:flex; flex-direction:column; gap:16px; }
    .row { display:grid; gap:16px; }
    .cards { grid-template-columns:repeat(6,minmax(0,1fr)); }
    .split { grid-template-columns:minmax(0,1.55fr) minmax(300px,1fr); }
    .split3 { grid-template-columns:minmax(0,1fr) minmax(0,1fr) minmax(0,1fr); }

    .panel { position:relative; background:var(--panel); backdrop-filter:blur(10px);
      border:1px solid var(--stroke); border-radius:16px; padding:16px 18px;
      box-shadow:0 10px 30px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.03);
      animation:rise .5s cubic-bezier(.2,.7,.2,1) both; min-width:0; }
    .panel h2 { margin:0 0 14px; font-size:13px; font-weight:700; color:var(--muted);
      text-transform:uppercase; letter-spacing:.10em; display:flex; align-items:center; gap:8px; }
    .panel h2::before { content:""; width:6px; height:6px; border-radius:2px;
      background:linear-gradient(120deg,var(--de),var(--en)); box-shadow:0 0 8px var(--glow-en); }

    .card { position:relative; overflow:hidden; padding:15px 16px 16px; }
    .card::before { content:""; position:absolute; top:0; left:0; right:0; height:3px;
      background:linear-gradient(90deg,var(--de),var(--en)); opacity:.85; }
    .card:hover { transform:translateY(-3px); transition:transform .2s; }
    .card .label { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.09em; font-weight:600; }
    .card .value { font-size:30px; font-weight:760; margin-top:9px; line-height:1; letter-spacing:-.5px;
      font-variant-numeric:tabular-nums; white-space:nowrap; }
    .card .value small { font-size:15px; color:var(--muted); font-weight:600; margin-left:2px; }
    .card .foot { color:var(--faint); font-size:11.5px; margin-top:8px; display:flex; align-items:center; gap:7px; }
    .delta { font-weight:700; font-variant-numeric:tabular-nums; }
    .delta.good { color:var(--green); } .delta.bad { color:var(--red); } .delta.flat { color:var(--faint); }
    .card.de::before { background:linear-gradient(90deg,var(--de),#bdf5d6); }
    .card.en::before { background:linear-gradient(90deg,var(--en),#bfe9ff); }
    .card.de .value { color:var(--de); } .card.en .value { color:var(--en); }

    canvas { width:100%; display:block; border-radius:10px; }
    .legend { display:flex; gap:16px; margin-bottom:6px; }
    .legend span { display:inline-flex; align-items:center; gap:7px; font-size:12px; color:var(--muted); font-weight:600; }
    .swatch { width:11px; height:11px; border-radius:3px; box-shadow:0 0 8px currentColor; }

    /* language balance */
    .balance { display:flex; flex-direction:column; gap:14px; }
    .bar-row { display:flex; flex-direction:column; gap:6px; }
    .bar-top { display:flex; justify-content:space-between; align-items:baseline; font-size:13px; }
    .bar-top b { font-variant-numeric:tabular-nums; font-size:16px; }
    .bar-track { height:10px; border-radius:999px; background:rgba(120,150,160,.12); overflow:hidden; }
    .bar-fill { height:100%; border-radius:999px; width:0%;
      transition:width .9s cubic-bezier(.2,.7,.2,1); }
    .bar-fill.de { background:linear-gradient(90deg,var(--de),#a7f0cb); box-shadow:0 0 14px var(--glow-de); }
    .bar-fill.en { background:linear-gradient(90deg,var(--en),#b6e6ff); box-shadow:0 0 14px var(--glow-en); }
    .gapwrap { display:flex; align-items:center; justify-content:space-between; gap:14px;
      padding:13px 15px; border-radius:13px; background:rgba(10,15,18,.55); border:1px solid var(--stroke-soft); }
    .gapwrap .big { font-size:30px; font-weight:780; font-variant-numeric:tabular-nums; letter-spacing:-.5px; }
    .gaptag { font-size:11.5px; font-weight:700; padding:5px 10px; border-radius:999px; }
    .gaptag.narrow { color:var(--green); background:rgba(53,199,123,.14); }
    .gaptag.widen { color:var(--yellow); background:rgba(244,183,64,.14); }
    .gaptag.flat { color:var(--muted); background:rgba(120,150,160,.12); }

    table { width:100%; border-collapse:collapse; font-size:12.5px; }
    th,td { padding:8px 6px; border-bottom:1px solid var(--stroke-soft); text-align:right; font-variant-numeric:tabular-nums; }
    th { color:var(--faint); font-weight:600; text-transform:uppercase; font-size:10.5px; letter-spacing:.08em; }
    th:first-child,td:first-child { text-align:left; }
    tbody tr { transition:background .15s; } tbody tr:hover { background:rgba(70,198,255,.05); }

    pre { margin:0; white-space:pre-wrap; word-break:break-word; color:#cdd7da;
      font-family:"JetBrains Mono",ui-monospace,SFMono-Regular,Menlo,monospace; font-size:11.5px;
      line-height:1.55; max-height:320px; overflow:auto; }
    .health-ok { color:var(--green); display:flex; align-items:center; gap:8px; font-weight:600; }
    .health-line { padding:6px 9px; border-radius:8px; margin-bottom:5px; font-size:11.5px;
      background:rgba(255,107,107,.10); border:1px solid rgba(255,107,107,.25); color:#ffd0d0;
      font-family:ui-monospace,monospace; }

    .concepts { display:grid; grid-template-columns:repeat(auto-fill,minmax(160px,1fr)); gap:9px;
      max-height:430px; overflow:auto; padding-right:4px; }
    .concept { padding:10px 11px; border-radius:11px; background:rgba(13,18,21,.6);
      border:1px solid var(--stroke); border-left-width:3px; transition:transform .15s, border-color .2s; }
    .concept:hover { transform:translateY(-2px); }
    .concept b { display:block; font-size:12px; overflow-wrap:anywhere; margin-bottom:6px; }
    .pill { display:inline-block; padding:3px 8px; border-radius:999px; font-size:10px;
      background:rgba(120,150,160,.12); color:var(--muted); font-weight:600; }
    .cmargin { float:right; font-variant-numeric:tabular-nums; font-weight:700; font-size:12px; }

    /* live neuro force-graph */
    #neuroGraph { width:100%; height:360px; display:block; border-radius:12px; cursor:grab;
      background:radial-gradient(420px 280px at 50% 45%,rgba(124,137,255,.06),transparent 70%),rgba(8,11,18,.4);
      border:1px solid var(--stroke); }
    #neuroGraph:active { cursor:grabbing; }
    .neuro-legend { display:flex; gap:13px; flex-wrap:wrap; margin-top:9px; font-size:11px; color:var(--muted); font-weight:600; }
    .neuro-legend i { display:inline-block; width:9px; height:9px; border-radius:3px; margin-right:5px; box-shadow:0 0 7px currentColor; vertical-align:-1px; }
    .nr-controls { display:flex; align-items:center; gap:8px; margin-top:10px; }
    .nr-controls button { background:rgba(29,36,40,.85); border:1px solid var(--stroke); color:var(--text);
      border-radius:8px; padding:5px 11px; font-size:12px; cursor:pointer; transition:border-color .15s,background .15s; }
    .nr-controls button:hover:not(:disabled) { border-color:rgba(70,198,255,.5); }
    .nr-controls button:disabled { opacity:.4; cursor:default; }
    .nr-controls input[type=range] { flex:1; accent-color:var(--de); height:4px; cursor:pointer; }
    .nr-label { font-size:11.5px; color:var(--muted); font-variant-numeric:tabular-nums; min-width:128px; text-align:right; }

    .links { display:flex; flex-direction:column; gap:8px; max-height:430px; overflow:auto; }
    .linkrow { padding:10px 12px; border-radius:11px; background:rgba(13,18,21,.6); border:1px solid var(--stroke); }
    .linkrow b { font-size:12px; overflow-wrap:anywhere; }
    a { color:var(--en); text-decoration:none; } a:hover { text-decoration:underline; }

    .empty { color:var(--faint); font-size:12.5px; padding:8px 0; }

    /* info badges + tooltip */
    .info { display:inline-flex; align-items:center; justify-content:center; width:15px; height:15px;
      margin-left:6px; border-radius:50%; font-size:10px; font-weight:800; font-style:normal; cursor:help;
      line-height:1; vertical-align:middle; color:var(--muted); background:rgba(120,150,160,.14);
      border:1px solid var(--stroke); transition:color .15s,background .15s,border-color .15s; }
    .info:hover,.info:focus { color:#fff; background:rgba(70,198,255,.25); border-color:rgba(70,198,255,.55); outline:none; }
    #tip { position:fixed; z-index:100; max-width:300px; padding:10px 13px; border-radius:11px;
      background:rgba(8,12,14,.97); border:1px solid rgba(120,150,160,.3); color:#eaf1f3;
      font-size:12px; line-height:1.55; box-shadow:0 12px 34px rgba(0,0,0,.6); pointer-events:none;
      opacity:0; transform:translateY(5px); transition:opacity .14s,transform .14s; }
    #tip.show { opacity:1; transform:none; }

    /* glossary */
    .glossary { display:grid; grid-template-columns:repeat(auto-fill,minmax(270px,1fr)); gap:12px; }
    .term { position:relative; overflow:hidden; padding:15px 15px 14px; border-radius:13px;
      background:rgba(13,18,21,.6); border:1px solid var(--stroke); transition:transform .15s; }
    .term:hover { transform:translateY(-2px); }
    .term::before { content:""; position:absolute; top:0; left:0; right:0; height:2px;
      background:linear-gradient(90deg,var(--de),var(--en)); opacity:.8; }
    .term h3 { margin:0 0 8px; font-size:14px; display:flex; align-items:center; gap:8px; }
    .term h3 .k { font-size:10.5px; font-weight:700; padding:2px 7px; border-radius:6px;
      background:rgba(70,198,255,.14); color:var(--en); white-space:nowrap; }
    .term p { margin:0 0 9px; color:#c4d0d4; font-size:12.5px; line-height:1.55; }
    .term .rule { font-size:11.5px; color:var(--muted); } .term .rule b { color:var(--accent); }
    .fade { animation:rise .45s cubic-bezier(.2,.7,.2,1) both; }
    .stagger>* { animation:rise .5s cubic-bezier(.2,.7,.2,1) both; }
    .stagger>*:nth-child(1){animation-delay:.02s}.stagger>*:nth-child(2){animation-delay:.06s}
    .stagger>*:nth-child(3){animation-delay:.10s}.stagger>*:nth-child(4){animation-delay:.14s}
    .stagger>*:nth-child(5){animation-delay:.18s}.stagger>*:nth-child(6){animation-delay:.22s}

    @keyframes spin { to { transform:rotate(360deg); } }
    @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(53,199,123,.55);} 70%{box-shadow:0 0 0 8px rgba(53,199,123,0);} 100%{box-shadow:0 0 0 0 rgba(53,199,123,0);} }
    @keyframes rise { from { opacity:0; transform:translateY(10px); } to { opacity:1; transform:none; } }
    @media (prefers-reduced-motion:reduce){ *{animation-duration:.001s!important; transition:none!important;} }
    @media (max-width:1180px){ .cards{grid-template-columns:repeat(3,1fr);} .split,.split3{grid-template-columns:1fr;} }
    @media (max-width:620px){ .cards{grid-template-columns:repeat(2,1fr);} header{gap:12px;} main{padding:14px;} }
  </style>
</head>
<body>
  <div id="refreshBar"></div>
  <header>
    <div class="brand">
      <div class="logo"><i>A</i></div>
      <div>
        <h1>Auralis <span>Training Monitor</span></h1>
        <div class="sub" id="subline">Live-BPB · Health · Neuro-Trace</div>
      </div>
    </div>
    <div class="controls">
      <span id="runStatus" class="status"><span class="dot"></span><span>lade…</span></span>
      <span id="updated" class="sub"></span>
      <select id="logSelect" title="Log auswählen"></select>
      <button id="refreshBtn" class="btn-accent">⟳ Aktualisieren</button>
    </div>
  </header>

  <main>
    <section class="row cards stagger" id="cards">
      <div class="panel card"><div class="label" data-tip="Trainingsschritt. 1 Schritt = 1 Optimizer-Update (hier ~65k Tokens).">Step</div><div id="step" class="value">–</div><div class="foot" id="mix">–</div></div>
      <div class="panel card"><div class="label" data-tip="Überraschung des Modells über das nächste Token (nats). Start ≈ 12,2 = reines Raten. Niedriger = besser.">Train Loss</div><div id="loss" class="value">–</div><div class="foot"><span id="lossDelta" class="delta flat">–</span><span id="lr">lr –</span></div></div>
      <div class="panel card"><div class="label" data-tip="Stärke der Gewichts-Änderung pro Schritt. Stabil/klein = gesund, plötzliche Sprünge = Explosion (Guard stoppt).">Grad Norm</div><div id="grad" class="value">–</div><div class="foot">Guard aktiv · stabil</div></div>
      <div class="panel card"><div class="label" data-tip="Verarbeitete Tokens pro Sekunde. Bestimmt, wie lange der Lauf dauert.">Durchsatz</div><div id="tps" class="value">–</div><div class="foot" id="dataPct">data –</div></div>
      <div class="panel card de"><div class="label" data-tip="Bits-pro-Byte für Deutsch — der faire, sprachübergreifend vergleichbare Schwierigkeitswert. Niedriger = besser.">Deutsch · BPB</div><div id="bpbDe" class="value">–</div><div class="foot"><span id="deDelta" class="delta flat">–</span>niedriger = besser</div></div>
      <div class="panel card en"><div class="label" data-tip="Bits-pro-Byte für Englisch. Soll nicht kollabieren — der Health-Guard wacht darüber.">Englisch · BPB</div><div id="bpbEn" class="value">–</div><div class="foot"><span id="enDelta" class="delta flat">–</span>niedriger = besser</div></div>
    </section>

    <section class="row split">
      <div class="panel fade">
        <h2>Training</h2>
        <div class="legend">
          <span><i class="swatch" style="background:var(--blue);color:var(--blue)"></i>train/loss</span>
          <span><i class="swatch" style="background:var(--yellow);color:var(--yellow)"></i>grad_norm</span>
        </div>
        <canvas id="lossChart" height="300"></canvas>
      </div>
      <div class="panel fade">
        <h2>Sprach-Balance</h2>
        <div class="balance">
          <div class="bar-row">
            <div class="bar-top"><span style="color:var(--de)">Deutsch</span><b id="balDeVal" style="color:var(--de)">–</b></div>
            <div class="bar-track"><div id="balDe" class="bar-fill de"></div></div>
          </div>
          <div class="bar-row">
            <div class="bar-top"><span style="color:var(--en)">Englisch</span><b id="balEnVal" style="color:var(--en)">–</b></div>
            <div class="bar-track"><div id="balEn" class="bar-fill en"></div></div>
          </div>
          <div class="gapwrap">
            <div>
              <div class="label" style="color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.09em">BPB-Gap (DE/EN)</div>
              <div class="big" id="gapBig">–</div>
            </div>
            <div style="text-align:right">
              <span id="gapTag" class="gaptag flat">–</span>
              <div class="sub" id="gapHint" style="margin-top:7px;max-width:200px">Ziel: Gap fällt, weil Deutsch fällt.</div>
            </div>
          </div>
        </div>
      </div>
    </section>

    <section class="row split">
      <div class="panel fade">
        <h2>Bits-per-Byte über Steps <span style="font-weight:600;color:var(--faint);font-size:11px;text-transform:none;letter-spacing:0">· Tail-Val-Trend (echte tokens/byte) · Kacheln oben = repräsentativ</span></h2>
        <div class="legend">
          <span><i class="swatch" style="background:var(--de);color:var(--de)"></i>Deutsch</span>
          <span><i class="swatch" style="background:var(--en);color:var(--en)"></i>Englisch</span>
        </div>
        <canvas id="bpbChart" height="300"></canvas>
      </div>
      <div class="panel fade">
        <h2>Letzte Eval</h2>
        <table id="evalTable"></table>
      </div>
    </section>

    <section class="row split">
      <div class="panel fade">
        <h2>Neuro- / Wissenskarte</h2>
        <div id="neuroMeta" class="empty">–</div>
        <canvas id="neuroGraph" height="360"></canvas>
        <div class="nr-controls">
          <button id="nrPlay" title="Abspielen / Pause">▶</button>
          <button id="nrPrev" title="Schritt zurück">◀◀</button>
          <button id="nrNext" title="Schritt vor">▶▶</button>
          <input id="nrSlider" type="range" min="0" max="0" value="0" />
          <span id="nrLabel" class="nr-label">–</span>
        </div>
        <div class="neuro-legend">
          <span><i style="background:#b388ff;color:#b388ff"></i>Kategorie</span>
          <span><i style="background:var(--green);color:var(--green)"></i>strong</span>
          <span><i style="background:var(--yellow);color:var(--yellow)"></i>watch</span>
          <span><i style="background:#ff9b54;color:#ff9b54"></i>weak</span>
          <span><i style="background:var(--red);color:var(--red)"></i>danger</span>
        </div>
      </div>
      <div class="panel fade">
        <h2>Health</h2>
        <div id="healthBox"></div>
      </div>
    </section>

    <section class="row split3">
      <div class="panel fade" style="grid-column:1 / -1">
        <h2>Log Tail</h2>
        <pre id="tailBox">–</pre>
      </div>
    </section>

    <section class="row">
      <div class="panel fade">
        <h2>Learning-Trace Reports</h2>
        <div id="reportLinks" class="links"></div>
      </div>
    </section>

    <section class="row">
      <div class="panel fade">
        <h2>Was bedeutet das?</h2>
        <div class="glossary">
          <div class="term"><h3>BPB <span class="k">Bits / Byte</span></h3>
            <p>Wie viele Bits das Modell pro Byte Text „braucht". Anders als der Per-Token-Loss ist BPB <b>fair über Sprachen vergleichbar</b>, weil es die unterschiedliche Tokenizer-Zerlegung herausrechnet.</p>
            <div class="rule">Faustregel: <b>niedriger = besser.</b></div></div>
          <div class="term"><h3>Gap <span class="k">DE / EN</span></h3>
            <p>Deutsch-BPB geteilt durch Englisch-BPB. Zeigt, wie weit Deutsch hinter Englisch liegt. Entscheidend ist die <b>Richtung</b>.</p>
            <div class="rule">Soll fallen, <b>weil Deutsch fällt</b> — nicht weil Englisch schlechter wird.</div></div>
          <div class="term"><h3>Train Loss <span class="k">nats</span></h3>
            <p>Negative Log-Likelihood: wie überrascht das Modell vom nächsten Token ist. Start ≈ ln(Vokabular) ≈ 12,2 (reines Raten über 200k Tokens).</p>
            <div class="rule">Niedriger = besser. Unter ~3–4 nats beginnt kohärenter Text.</div></div>
          <div class="term"><h3>Grad Norm</h3>
            <p>Länge des Gradientenvektors pro Update — wie stark sich die Gewichte ändern wollen.</p>
            <div class="rule">Stabil/klein = gesund. Sprünge = Explosion → der Guard stoppt.</div></div>
          <div class="term"><h3>Learning Rate &amp; Warmup</h3>
            <p>Schrittweite des Optimierers. Im Warmup steigt sie langsam an (damit der Start nicht explodiert), danach fällt sie per Cosine.</p>
            <div class="rule"><b>Vor Warmup-Ende</b> sind die Zahlen wenig aussagekräftig.</div></div>
          <div class="term"><h3>Margin <span class="k">Neuro</span></h3>
            <p>Bei einer Wissensfrage: wie viel wahrscheinlicher das Modell die <b>richtige</b> Fortsetzung findet als eine falsche.</p>
            <div class="rule">Positiv/grün = gelernt. Negativ/rot = noch falsch.</div></div>
          <div class="term"><h3>Durchsatz <span class="k">Tok/s</span></h3>
            <p>Verarbeitete Tokens pro Sekunde. Mal die Gesamt-Tokens ≈ ungefähre Laufzeit.</p>
            <div class="rule">Höher = schneller fertig. Multi-GPU skaliert ~linear.</div></div>
          <div class="term"><h3>Health-Guards</h3>
            <p>Automatische Wächter, die den Lauf stoppen, bevor er Geld verbrennt: Grad-Explosion, Val-Regression, BPB-Kollaps, VRAM, Durchsatz.</p>
            <div class="rule">Grün ✓ = alles ruhig.</div></div>
        </div>
      </div>
    </section>
  </main>

  <script>
    const REFRESH_MS = 10000;
    const state = { selectedLog:"", prev:{}, gapHist:[] };
    const fmt = (v,d=3) => (v===null||v===undefined||Number.isNaN(Number(v))) ? "–" : Number(v).toFixed(d);
    const easeOut = t => 1 - Math.pow(1 - t, 3);

    /* animated number tween */
    function tween(el, to, d=2, suffix=""){
      if(!el) return;
      const from = Number(el.dataset.v || "0");
      const target = (to===null||to===undefined||Number.isNaN(Number(to))) ? null : Number(to);
      if(target===null){ el.textContent="–"; el.dataset.v="0"; return; }
      el.dataset.v = String(target);
      const t0 = performance.now(), dur=650;
      function step(now){
        const k = Math.min(1,(now-t0)/dur), val = from + (target-from)*easeOut(k);
        el.innerHTML = val.toFixed(d) + suffix;
        if(k<1) requestAnimationFrame(step);
      }
      requestAnimationFrame(step);
    }
    function setDelta(el, cur, prev, lowerBetter=true){
      if(!el) return;
      if(cur===undefined||prev===undefined||Number.isNaN(cur)||Number.isNaN(prev)){ el.textContent="–"; el.className="delta flat"; return; }
      const d = cur-prev;
      if(Math.abs(d)<1e-6){ el.textContent="±0"; el.className="delta flat"; return; }
      const good = lowerBetter ? d<0 : d>0;
      el.textContent = (d<0?"▼ ":"▲ ") + Math.abs(d).toFixed(3);
      el.className = "delta " + (good?"good":"bad");
    }

    /* ---------- animated canvas line chart with hover ---------- */
    function fitCanvas(c){
      const dpr = window.devicePixelRatio||1, r = c.getBoundingClientRect();
      c.width = Math.max(320, r.width)*dpr; c.height = (c.getAttribute("height")||300)*dpr;
      const ctx=c.getContext("2d"); ctx.setTransform(dpr,0,0,dpr,0,0);
      return {ctx, w:Math.max(320,r.width), h:Number(c.getAttribute("height")||300)};
    }
    function paintChart(c, series, progress){
      const {ctx,w,h} = fitCanvas(c);
      ctx.clearRect(0,0,w,h);
      const pad={l:52,r:16,t:14,b:28};
      const pts = series.flatMap(s=>s.points);
      const ys = pts.map(p=>p.y).filter(Number.isFinite);
      const xs = pts.map(p=>p.x).filter(Number.isFinite);
      if(ys.length<2||xs.length<2){ ctx.fillStyle="#5d6f77"; ctx.font="13px Inter,system-ui"; ctx.fillText("Noch keine Daten…",pad.l,h/2); c._scales=null; return; }
      let minY=Math.min(...ys), maxY=Math.max(...ys); if(Math.abs(maxY-minY)<1e-9){maxY+=1;minY-=1;}
      const padY=(maxY-minY)*0.12; minY-=padY; maxY+=padY;
      const minX=Math.min(...xs), maxX=Math.max(...xs);
      const sx=x=>pad.l+((x-minX)/Math.max(maxX-minX,1))*(w-pad.l-pad.r);
      const sy=y=>h-pad.b-((y-minY)/(maxY-minY))*(h-pad.t-pad.b);
      /* grid + y labels */
      ctx.font="11px Inter,system-ui"; ctx.textBaseline="middle";
      for(let i=0;i<=4;i++){
        const yy=pad.t+i*(h-pad.t-pad.b)/4, val=maxY-(i/4)*(maxY-minY);
        ctx.strokeStyle="rgba(120,150,160,.10)"; ctx.lineWidth=1;
        ctx.beginPath(); ctx.moveTo(pad.l,yy); ctx.lineTo(w-pad.r,yy); ctx.stroke();
        ctx.fillStyle="#5d6f77"; ctx.textAlign="right"; ctx.fillText(val.toFixed(2),pad.l-8,yy);
      }
      ctx.textAlign="center"; ctx.textBaseline="top";
      ctx.fillStyle="#5d6f77"; ctx.fillText(String(minX),sx(minX),h-pad.b+8); ctx.fillText(String(maxX),sx(maxX),h-pad.b+8);
      const clipX = pad.l + (w-pad.l-pad.r)*progress;
      for(const s of series){
        const P=s.points.filter(p=>Number.isFinite(p.x)&&Number.isFinite(p.y));
        if(P.length<2) continue;
        ctx.save(); ctx.beginPath(); ctx.rect(0,0,clipX,h); ctx.clip();
        /* area fill */
        const grad=ctx.createLinearGradient(0,pad.t,0,h-pad.b);
        grad.addColorStop(0,s.color+"33"); grad.addColorStop(1,s.color+"00");
        ctx.beginPath(); ctx.moveTo(sx(P[0].x),sy(P[0].y));
        for(const p of P) ctx.lineTo(sx(p.x),sy(p.y));
        ctx.lineTo(sx(P[P.length-1].x),h-pad.b); ctx.lineTo(sx(P[0].x),h-pad.b); ctx.closePath();
        ctx.fillStyle=grad; ctx.fill();
        /* line with glow */
        ctx.beginPath(); P.forEach((p,i)=> i?ctx.lineTo(sx(p.x),sy(p.y)):ctx.moveTo(sx(p.x),sy(p.y)));
        ctx.strokeStyle=s.color; ctx.lineWidth=2.4; ctx.lineJoin="round"; ctx.lineCap="round";
        ctx.shadowColor=s.color; ctx.shadowBlur=10; ctx.stroke(); ctx.shadowBlur=0;
        /* last point dot */
        const last=P[P.length-1];
        ctx.beginPath(); ctx.arc(sx(last.x),sy(last.y),3.6,0,7); ctx.fillStyle=s.color; ctx.fill();
        ctx.beginPath(); ctx.arc(sx(last.x),sy(last.y),3.6,0,7); ctx.strokeStyle="#0a0f12"; ctx.lineWidth=1.5; ctx.stroke();
        ctx.restore();
      }
      c._scales={sx,sy,minX,maxX,pad,w,h,series};
    }
    function animateChart(c, series){
      const changed = c._sig !== JSON.stringify(series.map(s=>s.points.length));
      c._sig = JSON.stringify(series.map(s=>s.points.length));
      c._lastSeries=series;
      if(c._raf) cancelAnimationFrame(c._raf);
      const t0=performance.now(), dur= changed?700:1;
      const run=now=>{ const k=Math.min(1,(now-t0)/dur); paintChart(c,series,easeOut(k)); if(k<1) c._raf=requestAnimationFrame(run); };
      c._raf=requestAnimationFrame(run);
    }
    function attachHover(c){
      const tip=document.createElement("div");
      Object.assign(tip.style,{position:"fixed",pointerEvents:"none",zIndex:99,padding:"8px 10px",
        borderRadius:"9px",background:"rgba(8,12,14,.94)",border:"1px solid rgba(120,150,160,.25)",
        font:"12px Inter,system-ui",color:"#eaf1f3",boxShadow:"0 8px 24px rgba(0,0,0,.5)",
        opacity:0,transition:"opacity .12s",whiteSpace:"nowrap"});
      document.body.appendChild(tip);
      c.addEventListener("mousemove",e=>{
        const sc=c._scales; if(!sc){tip.style.opacity=0;return;}
        const r=c.getBoundingClientRect(), mx=e.clientX-r.left;
        let bx=null,best=1e9,rows=[];
        for(const s of sc.series){ for(const p of s.points){ const d=Math.abs(sc.sx(p.x)-mx); if(d<best){best=d;bx=p.x;} } }
        if(bx===null){tip.style.opacity=0;return;}
        for(const s of sc.series){ const p=s.points.find(q=>q.x===bx); if(p) rows.push(`<span style="color:${s.color}">●</span> ${s.name}: <b>${p.y.toFixed(3)}</b>`); }
        paintChart(c,c._lastSeries,1);
        const cx=sc.sx(bx); const ctx=c.getContext("2d");
        ctx.strokeStyle="rgba(120,150,160,.4)"; ctx.lineWidth=1; ctx.setLineDash([4,4]);
        ctx.beginPath(); ctx.moveTo(cx,sc.pad.t); ctx.lineTo(cx,sc.h-sc.pad.b); ctx.stroke(); ctx.setLineDash([]);
        tip.innerHTML=`<div style="color:#8ca0a8;margin-bottom:4px">step ${bx}</div>${rows.join("<br>")}`;
        tip.style.left=(e.clientX+14)+"px"; tip.style.top=(e.clientY+14)+"px"; tip.style.opacity=1;
      });
      c.addEventListener("mouseleave",()=>{ tip.style.opacity=0; if(c._lastSeries) paintChart(c,c._lastSeries,1); });
    }

    function loadJson(url){ return fetch(url,{headers:{"Cache-Control":"no-store"}}).then(r=>{ if(!r.ok) throw new Error("HTTP "+r.status); return r.json(); }); }

    function bar(){ const b=document.getElementById("refreshBar"); b.style.width="0%"; requestAnimationFrame(()=>{ b.style.transition="width "+REFRESH_MS+"ms linear"; b.style.width="100%"; }); }

    async function loadStatus(){
      bar();
      const q = state.selectedLog ? `?log=${encodeURIComponent(state.selectedLog)}` : "";
      try { render(await loadJson(`/api/status${q}`)); }
      catch(err){
        const s=document.getElementById("runStatus"); s.className="status bad";
        s.querySelector("span:last-child").textContent="API-Fehler";
      }
    }

    function render(data){
      const sel=document.getElementById("logSelect"), prevSel=state.selectedLog;
      sel.innerHTML="";
      for(const log of (data.logs||[])){ const o=document.createElement("option"); o.value=log.path; o.textContent=log.name; sel.appendChild(o); }
      if(data.selected_log){ state.selectedLog=data.selected_log; sel.value=data.selected_log; } else if(prevSel){ sel.value=prevSel; }

      const run=data.run||{}, lt=run.latest_train||{}, ev=run.latest_eval||{};
      document.getElementById("step").textContent = lt.step ?? ev.step ?? "–";
      document.getElementById("mix").textContent = run.mix || run.name || "–";
      tween(document.getElementById("loss"), lt.loss, 3);
      document.getElementById("lr").textContent = "lr " + (lt.lr ? Number(lt.lr).toExponential(2) : "–");
      setDelta(document.getElementById("lossDelta"), lt.loss, state.prev.loss, true);
      tween(document.getElementById("grad"), lt.grad_norm, 2);
      const tpsEl=document.getElementById("tps");
      tpsEl.innerHTML = lt.tok_s ? (lt.tok_s/1000).toFixed(1)+"<small>k tok/s</small>" : "–";
      document.getElementById("dataPct").textContent = "data " + fmt(lt.data_pct,1) + "%";
      const rp = data.repr || null;
      const tileDe = rp ? rp.de : ev.bpb_german;
      const tileEn = rp ? rp.en : ev.bpb_english;
      tween(document.getElementById("bpbDe"), tileDe, 3);
      tween(document.getElementById("bpbEn"), tileEn, 3);
      setDelta(document.getElementById("deDelta"), tileDe, state.prev.de, true);
      setDelta(document.getElementById("enDelta"), tileEn, state.prev.en, true);
      document.querySelectorAll("#bpbDe,#bpbEn").forEach(e=>e.title = rp ? ("repräsentativ · alle Quellen · step "+(rp.step??"?")) : "Tail-Val (Log)");

      /* status pill */
      const st=document.getElementById("runStatus");
      st.className="status "+(run.last_error?"bad":run.active_guess?"active":"warn");
      st.querySelector("span:last-child").textContent = run.last_error?"Fehler im Log":run.active_guess?"aktiv":"inaktiv";
      const now=new Date(); document.getElementById("updated").textContent = "aktualisiert "+now.toLocaleTimeString("de-DE");
      document.getElementById("subline").textContent = run.name ? run.name : "Live-BPB · Health · Neuro-Trace";

      /* language balance bars (normalize against max so both visible) */
      const de=tileDe, en=tileEn;
      if(Number.isFinite(de)&&Number.isFinite(en)){
        const mx=Math.max(de,en)*1.08;
        document.getElementById("balDe").style.width=(de/mx*100)+"%";
        document.getElementById("balEn").style.width=(en/mx*100)+"%";
        document.getElementById("balDeVal").textContent=de.toFixed(3);
        document.getElementById("balEnVal").textContent=en.toFixed(3);
      }
      const gap = rp ? rp.gap : ev.bpb_gap_max;
      tween(document.getElementById("gapBig"), gap, 3);
      const gapTag=document.getElementById("gapTag"), gapHint=document.getElementById("gapHint");
      if(rp && Number.isFinite(gap)){
        gapTag.className="gaptag narrow"; gapTag.textContent="● repräsentativ";
        gapHint.textContent="Über alle Quellen, echte tokens/byte: de "+rp.de.toFixed(2)+" / en "+rp.en.toFixed(2)+" — fast gleichauf (step "+(rp.step??"?")+").";
      } else if(Number.isFinite(gap)){
        state.gapHist.push(gap); if(state.gapHist.length>6) state.gapHist.shift();
        if(state.gapHist.length>=2){
          const d=gap-state.gapHist[0];
          if(d<-0.01){ gapTag.className="gaptag narrow"; gapTag.textContent="▼ verengt sich"; gapHint.textContent="Gut – Abstand schließt sich."; }
          else if(d>0.01){ gapTag.className="gaptag widen"; gapTag.textContent="▲ weitet sich"; gapHint.textContent="Beobachten."; }
          else { gapTag.className="gaptag flat"; gapTag.textContent="● stabil"; gapHint.textContent="Gap hält."; }
        }
      }

      animateChart(document.getElementById("lossChart"), [
        {name:"train/loss", color:"#5aa7ff", points:(run.train||[]).map(p=>({x:p.step,y:p.loss}))},
        {name:"grad_norm", color:"#f4b740", points:(run.train||[]).map(p=>({x:p.step,y:p.grad_norm}))},
      ]);
      animateChart(document.getElementById("bpbChart"), [
        {name:"Deutsch", color:"#3ddc84", points:(run.evals||[]).filter(e=>Number.isFinite(e.bpb_german)).map(e=>({x:e.step,y:e.bpb_german}))},
        {name:"Englisch", color:"#46c6ff", points:(run.evals||[]).filter(e=>Number.isFinite(e.bpb_english)).map(e=>({x:e.step,y:e.bpb_english}))},
      ]);

      const tbl=document.getElementById("evalTable");
      const rows=Object.entries(ev).sort(([a],[b])=>a.localeCompare(b));
      tbl.innerHTML = rows.length ? `<thead><tr><th>Metrik</th><th>Wert</th></tr></thead><tbody>`+
        rows.map(([k,v])=>`<tr><td>${k}</td><td>${fmt(v)}</td></tr>`).join("")+`</tbody>` :
        `<tbody><tr><td class="empty">Noch keine Eval.</td></tr></tbody>`;

      const hb=document.getElementById("healthBox");
      if(run.health&&run.health.length){ hb.innerHTML=run.health.map(h=>`<div class="health-line">${esc(h)}</div>`).join(""); }
      else { hb.innerHTML=`<div class="health-ok">✓ Keine Health-Warnungen</div>`; }

      const tail=document.getElementById("tailBox");
      tail.textContent=(run.tail||[]).join("\n"); tail.scrollTop=tail.scrollHeight;

      const rl=document.getElementById("reportLinks"); rl.innerHTML="";
      for(const r of (data.learning_reports||[]).slice(0,12)){
        const files=r.files||{}, links=Object.entries(files).map(([k,rel])=>`<a target="_blank" href="/static/${rel}">${k}</a>`);
        const d=document.createElement("div"); d.className="linkrow";
        d.innerHTML=`<b>${esc(r.key)}</b><div class="sub" style="margin-top:5px">${links.join(" · ")||"–"}</div>`;
        rl.appendChild(d);
      }
      if(!rl.children.length) rl.innerHTML=`<div class="empty">Keine Reports gefunden.</div>`;

      const neuro=data.neuro||{};
      document.getElementById("neuroMeta").textContent = neuro.source
        ? `${neuro.source} · step ${neuro.step ?? "–"} · ${(neuro.concepts||[]).length} Konzepte`
        : "Keine Neuro-Datei gefunden.";
      Neuro.setData(neuro);

      state.prev={loss:lt.loss, de:tileDe, en:tileEn};
    }
    function esc(s){ return String(s).replace(/[&<>]/g,m=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[m])); }

    /* ---------- live force-directed neuro map (in-dashboard) ---------- */
    const Neuro = (function(){
      const SC = {strong:'#35c77b', watch:'#f4b740', weak:'#ff9b54', danger:'#ff6b6b', neutral:'#7c89a4', unknown:'#8f98aa'};
      const PAL = ['#b388ff','#5ec8ff','#ff8fc8','#8cf0a0','#ffd166','#7af0c8','#ff9b54','#c0a3ff'];
      let cv=null, ctx=null, started=false, hover=null, drag=null, tip=null;
      let W=600, H=360, DPR=1;
      const nodes=[], byId=new Map(), catColor={};
      let edges=[];
      let frames=[], cursor=0, playing=false, playTimer=null, followLive=true, ctrlWired=false;
      function rnd(s){ let h=2166136261; for(let i=0;i<s.length;i++){ h^=s.charCodeAt(i); h=Math.imul(h,16777619);} return (h>>>0)/4294967295; }
      function mk(id,label,isCat,color){ const a=rnd(id)*6.2832, rad=(isCat?70:150)+rnd(id+'r')*80;
        return {id,label,isCat,color,r:isCat?16:10,status:'',margin:null,answer:'',category:'',
          x:W/2+Math.cos(a)*rad, y:H/2+Math.sin(a)*rad, vx:0, vy:0}; }
      function applyConcepts(concepts){
        cv=document.getElementById('neuroGraph'); if(!cv) return;
        if(!ctx){ ctx=cv.getContext('2d'); wire(); }
        const cats=[...new Set(concepts.map(c=>c.category||'unknown'))];
        cats.forEach((ct,i)=>{ if(!catColor['cat:'+ct]) catColor['cat:'+ct]=PAL[i%PAL.length]; });
        const seen=new Set();
        for(const ct of cats){ const id='cat:'+ct; seen.add(id); let n=byId.get(id);
          if(!n){ n=mk(id,ct,true,catColor[id]); nodes.push(n); byId.set(id,n);} n.color=catColor[id]; }
        for(const c of concepts){ const id='probe:'+c.id; seen.add(id);
          const col=SC[c.status]||(c.margin>=0.75?SC.strong:(c.margin>=0?SC.watch:SC.weak));
          let n=byId.get(id); if(!n){ n=mk(id,c.id,false,col); nodes.push(n); byId.set(id,n);}
          n.color=col; n.status=c.status; n.margin=c.margin; n.answer=c.answer; n.category=c.category; }
        for(let i=nodes.length-1;i>=0;i--){ if(!seen.has(nodes[i].id)){ byId.delete(nodes[i].id); nodes.splice(i,1);} }
        edges=concepts.map(c=>({s:'cat:'+(c.category||'unknown'), t:'probe:'+c.id})).filter(e=>byId.has(e.s)&&byId.has(e.t));
        const deg={}; edges.forEach(e=>deg[e.s]=(deg[e.s]||0)+1);
        nodes.forEach(n=>{ if(n.isCat) n.r=13+Math.min(11,(deg[n.id]||0)*1.7); });
        if(!started){ started=true; requestAnimationFrame(loop); }
      }
      function fit(){ DPR=window.devicePixelRatio||1; const r=cv.getBoundingClientRect();
        W=r.width||600; H=Number(cv.getAttribute('height')||360); cv.width=W*DPR; cv.height=H*DPR; }
      function tick(){ const cx=W/2, cy=H/2;
        for(let i=0;i<nodes.length;i++){ const a=nodes[i];
          for(let j=i+1;j<nodes.length;j++){ const b=nodes[j]; let dx=a.x-b.x,dy=a.y-b.y,d2=dx*dx+dy*dy+0.01,d=Math.sqrt(d2);
            const f=(a.isCat||b.isCat?2600:1500)/d2,ux=dx/d,uy=dy/d; a.vx+=ux*f;a.vy+=uy*f;b.vx-=ux*f;b.vy-=uy*f; }
          a.vx+=(cx-a.x)*0.002; a.vy+=(cy-a.y)*0.002; }
        for(const e of edges){ const a=byId.get(e.s),b=byId.get(e.t); let dx=b.x-a.x,dy=b.y-a.y,d=Math.hypot(dx,dy)||0.01;
          const rest=a.isCat||b.isCat?98:70,f=(d-rest)*0.02,ux=dx/d,uy=dy/d; a.vx+=ux*f;a.vy+=uy*f;b.vx-=ux*f;b.vy-=uy*f; }
        for(const n of nodes){ if(n===drag) continue; n.vx*=0.84; n.vy*=0.84; n.x+=n.vx; n.y+=n.vy;
          n.x=Math.max(n.r,Math.min(W-n.r,n.x)); n.y=Math.max(n.r,Math.min(H-n.r,n.y)); } }
      function draw(){ fit(); ctx.setTransform(DPR,0,0,DPR,0,0); ctx.clearRect(0,0,W,H);
        if(!nodes.length){ ctx.fillStyle='#5d6f77'; ctx.font='13px Inter,system-ui'; ctx.fillText('Keine Neuro-Daten im Trace…',16,28); return; }
        ctx.lineCap='round';
        for(const e of edges){ const a=byId.get(e.s),b=byId.get(e.t),hot=hover&&(hover===a||hover===b);
          const mx=(a.x+b.x)/2+(b.y-a.y)*0.13, my=(a.y+b.y)/2-(b.x-a.x)*0.13;
          const g=ctx.createLinearGradient(a.x,a.y,b.x,b.y); g.addColorStop(0,a.color); g.addColorStop(1,b.color);
          ctx.strokeStyle=g; ctx.globalAlpha=hot?0.95:0.28; ctx.lineWidth=hot?2.2:1.1; ctx.shadowColor=b.color; ctx.shadowBlur=hot?12:4;
          ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.quadraticCurveTo(mx,my,b.x,b.y); ctx.stroke(); }
        ctx.globalAlpha=1; ctx.shadowBlur=0;
        for(const n of nodes){ ctx.shadowColor=n.color; ctx.shadowBlur=hover===n?24:13; ctx.fillStyle=n.color;
          if(n.isCat){ ctx.save(); ctx.translate(n.x,n.y); ctx.rotate(Math.PI/4); const s=n.r*1.5; ctx.fillRect(-s/2,-s/2,s,s); ctx.restore(); }
          else { ctx.beginPath(); ctx.arc(n.x,n.y,n.r,0,6.2832); ctx.fill(); }
          ctx.shadowBlur=0; ctx.globalAlpha=.22; ctx.fillStyle='#fff';
          ctx.beginPath(); ctx.arc(n.x-n.r*0.28,n.y-n.r*0.28,n.r*0.3,0,6.2832); ctx.fill(); ctx.globalAlpha=1; }
        ctx.textAlign='center'; ctx.textBaseline='middle';
        for(const n of nodes){ if(!(n.isCat||hover===n||n.status==='danger')) continue;
          ctx.font=n.isCat?'700 12px Inter,system-ui':'600 10px Inter,system-ui'; const ty=n.y+n.r+(n.isCat?15:11);
          ctx.lineWidth=3; ctx.strokeStyle='rgba(7,10,16,.9)'; ctx.strokeText(n.label,n.x,ty);
          ctx.fillStyle=n.isCat?'#fff':(hover===n?'#fff':'rgba(228,236,245,.82)'); ctx.fillText(n.label,n.x,ty); } }
      function loop(){ tick(); draw(); requestAnimationFrame(loop); }
      function pick(mx,my){ let best=null,bd=1e9; for(const n of nodes){ const d=Math.hypot(n.x-mx,n.y-my); if(d<n.r+6&&d<bd){bd=d;best=n;} } return best; }
      function wire(){
        tip=document.getElementById('tip'); if(!tip){ tip=document.createElement('div'); tip.id='tip'; document.body.appendChild(tip); }
        cv.addEventListener('mousemove', ev => { const r=cv.getBoundingClientRect(),mx=ev.clientX-r.left,my=ev.clientY-r.top;
          if(drag){ drag.x=mx; drag.y=my; drag.vx=0; drag.vy=0; } hover=pick(mx,my);
          if(hover){ const m=(hover.margin==null)?'n/a':Number(hover.margin).toFixed(3);
            tip.innerHTML = hover.isCat ? `<b>${esc(hover.label)}</b>`
              : `<b>${esc(hover.label)}</b><br><span style="color:#8ca0a8">${esc(hover.category||'')} · ${esc(hover.status||'')} · margin ${m}</span>`
                + (hover.answer?`<div style="margin-top:6px;color:#dbe4f0;max-width:320px">${esc(hover.answer)}</div>`:'');
            tip.classList.add('show'); tip.style.left=Math.min(ev.clientX+14,window.innerWidth-340)+'px'; tip.style.top=(ev.clientY+14)+'px';
          } else tip.classList.remove('show'); });
        cv.addEventListener('mousedown', ev => { const r=cv.getBoundingClientRect(); drag=pick(ev.clientX-r.left,ev.clientY-r.top); });
        window.addEventListener('mouseup', () => { drag=null; });
        cv.addEventListener('mouseleave', () => { hover=null; if(tip) tip.classList.remove('show'); });
      }
      function applyFrame(){ if(frames.length && frames[cursor]) applyConcepts(frames[cursor].concepts || []); }
      function updateCtrl(){
        const sl=document.getElementById('nrSlider'), lab=document.getElementById('nrLabel');
        const pb=document.getElementById('nrPlay'), pv=document.getElementById('nrPrev'), nx=document.getElementById('nrNext');
        const multi = frames.length > 1;
        if(sl){ sl.max=Math.max(0,frames.length-1); sl.value=cursor; sl.disabled=!multi; }
        if(pb){ pb.textContent=playing?'⏸':'▶'; pb.disabled=!multi; }
        if(pv){ pv.disabled=!multi || cursor<=0; }
        if(nx){ nx.disabled=!multi || cursor>=frames.length-1; }
        if(lab){ lab.textContent = frames.length
          ? `Step ${frames[cursor] && frames[cursor].step!=null ? frames[cursor].step : '?'} · ${cursor+1}/${frames.length}${followLive?' · live':''}`
          : '–'; }
      }
      function go(i){ if(!frames.length) return; cursor=Math.max(0,Math.min(frames.length-1,i));
        followLive=(cursor===frames.length-1); applyFrame(); updateCtrl(); }
      function stop(){ playing=false; if(playTimer){ clearInterval(playTimer); playTimer=null; } updateCtrl(); }
      function play(){ if(frames.length<2) return; if(cursor>=frames.length-1) cursor=0;
        playing=true; followLive=false; if(playTimer) clearInterval(playTimer);
        playTimer=setInterval(()=>{ if(cursor>=frames.length-1){ stop(); return; } go(cursor+1); }, 750); updateCtrl(); }
      function wireCtrl(){ if(ctrlWired) return; ctrlWired=true;
        const pb=document.getElementById('nrPlay'), pv=document.getElementById('nrPrev'),
              nx=document.getElementById('nrNext'), sl=document.getElementById('nrSlider');
        if(pb) pb.addEventListener('click',()=>{ playing?stop():play(); });
        if(pv) pv.addEventListener('click',()=>{ stop(); go(cursor-1); });
        if(nx) nx.addEventListener('click',()=>{ stop(); go(cursor+1); });
        if(sl) sl.addEventListener('input',e=>{ stop(); go(parseInt(e.target.value,10)||0); });
      }
      function setData(neuro){
        wireCtrl();
        frames=(neuro && neuro.frames) || [];
        if(!frames.length){ applyConcepts((neuro && neuro.concepts) || []); updateCtrl(); return; }
        if(playing){ updateCtrl(); return; }            // don't disturb active playback
        if(followLive || cursor>frames.length-1) cursor=frames.length-1;
        applyFrame(); updateCtrl();
      }
      return { setData };
    })();

    const TIP_HEADERS = {
      "Training":"train/loss und grad_norm über die Schritte. Loss fällt = lernt; grad_norm stabil = gesund.",
      "Sprach-Balance":"Deutsch vs. Englisch in BPB, repräsentativ über alle Quellen mit den ECHTEN tokens/byte gemessen. Auf diesem Maß sind beide fast gleichauf (Gap ~1.04) — der frühere Gap ~3 war ein Artefakt (falsche tokens/byte + ein trivial leichtes englisches Val-Tail).",
      "Bits-per-Byte über Steps":"Verlauf von DE/EN-BPB über die Steps. Beide sollen fallen; der Abstand soll sich nach genug Training schließen.",
      "Letzte Eval":"Alle Roh-Metriken der letzten Evaluation, direkt aus dem Log.",
      "Neuro- / Wissenskarte":"Pro Konzept der Margin: wie viel sicherer das Modell die richtige Antwort findet als eine falsche. Grün = gelernt.",
      "Health":"Automatische Wächter (Grad-Explosion, Val-Regression, BPB-Kollaps, VRAM …). Grün = alles ruhig.",
      "Log Tail":"Die letzten Zeilen der Live-Logdatei.",
      "Learning-Trace Reports":"Verlinkte HTML/JSON-Reports früherer Auswertungen.",
      "Was bedeutet das?":"Kurzerklärungen aller Kennzahlen — fahr mit der Maus über ein ⓘ."
    };
    function setupTips(){
      let tip=document.getElementById("tip");
      if(!tip){ tip=document.createElement("div"); tip.id="tip"; document.body.appendChild(tip); }
      const show=el=>{ const t=el.getAttribute("data-tip"); if(!t) return;
        tip.innerHTML=t; tip.classList.add("show"); tip.style.left="0px"; tip.style.top="0px";
        const r=el.getBoundingClientRect(), tw=tip.offsetWidth, th=tip.offsetHeight;
        const x=Math.min(Math.max(8,r.left-4), window.innerWidth-tw-8);
        let y=r.bottom+8; if(y+th>window.innerHeight-8) y=r.top-th-8;
        tip.style.left=x+"px"; tip.style.top=y+"px"; };
      const hide=()=>tip.classList.remove("show");
      document.querySelectorAll(".panel h2").forEach(h=>{ const k=h.textContent.trim(); if(TIP_HEADERS[k]) h.setAttribute("data-tip",TIP_HEADERS[k]); });
      document.querySelectorAll("[data-tip]").forEach(el=>{
        if(el.classList.contains("info")||el.querySelector(".info")) return;
        const b=document.createElement("span"); b.className="info"; b.textContent="i"; b.tabIndex=0;
        b.setAttribute("data-tip", el.getAttribute("data-tip"));
        el.appendChild(b);
        b.addEventListener("mouseenter",()=>show(b)); b.addEventListener("mouseleave",hide);
        b.addEventListener("focus",()=>show(b)); b.addEventListener("blur",hide);
      });
    }
    document.getElementById("refreshBtn").addEventListener("click", loadStatus);
    document.getElementById("logSelect").addEventListener("change", e=>{ state.selectedLog=e.target.value; loadStatus(); });
    ["lossChart","bpbChart"].forEach(id=>attachHover(document.getElementById(id)));
    setupTips();
    window.addEventListener("resize", ()=>{ ["lossChart","bpbChart"].forEach(id=>{ const c=document.getElementById(id); if(c&&c._lastSeries) paintChart(c,c._lastSeries,1); }); });
    loadStatus();
    setInterval(loadStatus, REFRESH_MS);
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    root: Path = repo_root_from_here()
    log_dirs: list[Path] = []

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def send_json(self, obj: object) -> None:
        data = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            data = INDEX_HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/api/status":
            logs = list_logs(self.root, self.log_dirs)
            qs = parse_qs(parsed.query)
            selected = qs.get("log", [""])[0]
            if not selected and logs:
                selected = str(logs[0]["path"])
            selected_path = resolve_log_path(self.root, self.log_dirs, selected) if selected else None
            run = parse_log(selected_path) if selected_path else {}
            self.send_json({
                "root": str(self.root),
                "selected_log": selected,
                "logs": logs,
                "run": run,
                "learning_reports": learning_reports(self.root),
                "neuro": latest_neuro_summary(self.root),
                "repr": representative_snapshot(self.root),
            })
            return

        if parsed.path.startswith("/static/"):
            rel = unquote(parsed.path.removeprefix("/static/"))
            p = safe_relative(self.root, rel)
            if not p or not p.exists() or not p.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
            data = p.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(data)
            return

        self.send_error(HTTPStatus.NOT_FOUND)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=repo_root_from_here())
    ap.add_argument("--log-dir", type=Path, action="append", default=[])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    Handler.root = args.root.resolve()
    Handler.log_dirs = [p.resolve() for p in (args.log_dir or default_log_dirs(Handler.root)) if p.exists()]
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Auralis Training Monitor: http://{args.host}:{args.port}")
    print(f"root: {Handler.root}")
    print("log dirs:")
    for p in Handler.log_dirs:
        print(f"  - {p}")
    server.serve_forever()


if __name__ == "__main__":
    main()
