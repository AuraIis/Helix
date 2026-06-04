#!/usr/bin/env python3
"""Render an HTML dashboard from ``smoke_sft_de.py`` learning traces.

The trace answers a practical question during SFT:

- Does the target answer become more likely?
- Does a dangerous negative answer become less likely?
- Does the generated text actually improve, or only the loss?
- Which probes are currently strengths and weaknesses?

Per-probe cards with margin / target-NLL sparklines over the trace history. The
look matches the rest of the Auralis monitoring suite (glassmorphism, glow,
status colours) and is fully self-contained (no third-party libraries).
"""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any

STATUS_COLOR = {
    "strong": "#3ddc84",
    "watch": "#f3c84b",
    "weak": "#ff9b54",
    "danger": "#ff5f6d",
    "unknown": "#8f98aa",
}


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return esc(value)


def probe_rows(trace: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for step in trace.get("history", []):
        for probe in step.get("probes", []):
            row = dict(probe)
            row["_step"] = step.get("step")
            row["_val_loss"] = step.get("val_loss")
            out.setdefault(str(probe.get("id")), []).append(row)
    return out


def classify(latest: dict[str, Any]) -> str:
    margin = latest.get("margin")
    forbidden = latest.get("forbidden_hits") or []
    if forbidden:
        return "danger"
    if margin is None:
        return "unknown"
    margin = float(margin)
    if margin >= 0.75:
        return "strong"
    if margin >= 0.0:
        return "watch"
    return "weak"


def sparkline(points: list[float | None], color: str = "#72a7ff", uid: str = "s",
              width: int = 210, height: int = 52) -> str:
    """Glowing gradient-filled sparkline (self-contained SVG)."""
    vals = [float(x) for x in points if x is not None]
    if not vals:
        return '<div class="spark empty">keine Daten</div>'
    lo, hi = min(vals), max(vals)
    if hi == lo:
        hi = lo + 1.0
    usable = max(1, len(points) - 1)
    pts: list[tuple[float, float]] = []
    last_y = height / 2
    for i, value in enumerate(points):
        x = (i / usable) * width
        if value is None:
            y = last_y
        else:
            y = height - ((float(value) - lo) / (hi - lo) * (height - 10)) - 5
            last_y = y
        pts.append((x, y))
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts) + f" L {width:.1f} {height:.1f} L 0 {height:.1f} Z"
    zero = ""
    if lo < 0 < hi:
        zy = height - ((0 - lo) / (hi - lo) * (height - 10)) - 5
        zero = f'<line x1="0" y1="{zy:.1f}" x2="{width}" y2="{zy:.1f}" class="zero"/>'
    return (
        f'<svg class="spark" viewBox="0 0 {width} {height}" preserveAspectRatio="none" role="img">'
        f'<defs><linearGradient id="{uid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" stop-color="{color}" stop-opacity="0.34"/>'
        f'<stop offset="1" stop-color="{color}" stop-opacity="0"/></linearGradient></defs>'
        f'<path d="{area}" fill="url(#{uid})"/>{zero}'
        f'<polyline points="{poly}" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linejoin="round" stroke-linecap="round" style="filter:drop-shadow(0 0 4px {color})"/></svg>'
    )


TRACE_TEMPLATE = r"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Auralis Learning Trace</title>
  <style>
    :root{
      --bg0:#070a10; --panel:rgba(20,26,38,.72); --panel2:rgba(13,18,26,.6);
      --stroke:rgba(120,150,180,.16); --text:#eef3fb; --muted:#9aa7bd; --faint:#5f6c83;
      --good:#3ddc84; --watch:#f3c84b; --weak:#ff9b54; --bad:#ff5f6d; --blue:#72a7ff; --accent:#7af0c8;
    }
    *{box-sizing:border-box;}
    body{margin:0;color:var(--text);font-size:14px;
      font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
      background:var(--bg0);
      background-image:radial-gradient(900px 500px at 18% -10%,rgba(114,167,255,.12),transparent 60%),
        radial-gradient(900px 520px at 86% -10%,rgba(61,220,132,.10),transparent 60%);
      -webkit-font-smoothing:antialiased;}
    ::-webkit-scrollbar{width:10px;height:10px;} ::-webkit-scrollbar-thumb{background:rgba(120,150,180,.22);border-radius:20px;}
    header{padding:22px 30px;border-bottom:1px solid var(--stroke);background:rgba(8,11,18,.7);backdrop-filter:blur(12px);}
    h1{margin:0 0 8px;font-size:24px;font-weight:780;
      background:linear-gradient(110deg,#72a7ff,#3ddc84,#7af0c8);-webkit-background-clip:text;background-clip:text;color:transparent;}
    header p{color:var(--muted);margin:3px 0;font-size:12.5px;}
    code{color:var(--accent);background:rgba(122,240,200,.08);padding:1px 6px;border-radius:5px;}
    main{padding:22px 30px 46px;max-width:1560px;margin:0 auto;}
    .summary{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px;}
    .stat{position:relative;overflow:hidden;background:var(--panel);backdrop-filter:blur(8px);
      border:1px solid var(--stroke);border-radius:14px;padding:14px 16px;animation:rise .5s both;}
    .stat::before{content:"";position:absolute;top:0;left:0;right:0;height:3px;background:linear-gradient(90deg,var(--blue),var(--good));}
    .stat label{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.09em;font-weight:600;}
    .stat strong{font-size:30px;font-weight:780;display:block;margin-top:6px;font-variant-numeric:tabular-nums;}
    .overview{width:100%;border-collapse:collapse;margin:0 0 26px;background:var(--panel);backdrop-filter:blur(8px);
      border:1px solid var(--stroke);border-radius:14px;overflow:hidden;animation:rise .5s both;}
    th,td{padding:10px 13px;border-bottom:1px solid var(--stroke);text-align:left;vertical-align:top;}
    th{color:var(--faint);font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;font-weight:600;}
    .overview tbody tr{transition:background .15s;} .overview tbody tr:hover{background:rgba(114,167,255,.06);}
    .badge{display:inline-block;padding:3px 9px;border-radius:999px;font-size:11px;font-weight:700;}
    .badge.strong{background:rgba(61,220,132,.16);color:var(--good);}
    .badge.watch{background:rgba(243,200,75,.16);color:var(--watch);}
    .badge.weak{background:rgba(255,155,84,.16);color:var(--weak);}
    .badge.danger{background:rgba(255,95,109,.16);color:var(--bad);}
    .badge.unknown{background:rgba(143,152,170,.16);color:var(--muted);}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(440px,1fr));gap:16px;}
    .card{position:relative;overflow:hidden;background:var(--panel);backdrop-filter:blur(10px);
      border:1px solid var(--stroke);border-radius:16px;padding:18px;animation:rise .5s both;
      box-shadow:0 10px 30px rgba(0,0,0,.32);transition:transform .18s,border-color .2s;}
    .card:hover{transform:translateY(-3px);}
    .card::before{content:"";position:absolute;top:0;left:0;right:0;height:3px;}
    .card.strong::before{background:var(--good);} .card.watch::before{background:var(--watch);}
    .card.weak::before{background:var(--weak);} .card.danger::before{background:var(--bad);}
    .card.unknown::before{background:var(--faint);}
    .card-head{display:flex;justify-content:space-between;gap:12px;align-items:flex-start;}
    .card-head h2{margin:0;font-size:16px;font-weight:740;overflow-wrap:anywhere;}
    .card-head .cat{color:var(--muted);font-size:12px;}
    .metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:15px 0;}
    .metrics div{background:var(--panel2);border:1px solid var(--stroke);border-radius:10px;padding:10px 11px;}
    .metrics label{display:block;color:var(--muted);font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;}
    .metrics strong{font-size:19px;font-weight:740;font-variant-numeric:tabular-nums;display:block;margin-top:5px;}
    .metrics strong.good{color:var(--good);} .metrics strong.bad{color:var(--bad);}
    .charts{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;}
    .charts label{display:block;color:var(--muted);font-size:11px;margin-bottom:6px;}
    .spark{width:100%;height:52px;display:block;background:rgba(10,14,20,.5);border:1px solid var(--stroke);border-radius:10px;}
    .spark.empty{display:flex;align-items:center;justify-content:center;color:var(--faint);font-size:11px;}
    .zero{stroke:var(--faint);stroke-dasharray:3 3;opacity:.5;}
    h3{margin:16px 0 8px;font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:700;}
    pre{white-space:pre-wrap;word-break:break-word;background:rgba(8,11,18,.6);border:1px solid var(--stroke);
      border-radius:10px;padding:12px;color:#e7ecf3;min-height:46px;margin:0;
      font-family:"JetBrains Mono",ui-monospace,Menlo,monospace;font-size:12px;line-height:1.5;}
    .hits{font-size:11.5px;display:flex;gap:14px;flex-wrap:wrap;margin:10px 0 0;}
    .hits .ok{color:var(--good);} .hits .no{color:var(--bad);}
    .tokens{display:flex;flex-wrap:wrap;gap:7px;}
    .tokens span{background:var(--panel2);border:1px solid var(--stroke);border-radius:8px;padding:5px 9px;font-size:12px;}
    .tokens span b{color:var(--accent);font-variant-numeric:tabular-nums;}
    details{margin-top:14px;} summary{cursor:pointer;color:var(--blue);font-size:12.5px;font-weight:600;}
    details table{width:100%;border-collapse:collapse;margin-top:10px;font-size:12px;}
    details td:nth-child(n+2),details th:nth-child(n+2){text-align:right;font-variant-numeric:tabular-nums;}
    @keyframes rise{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:none;}}
    @media (prefers-reduced-motion:reduce){*{animation-duration:.001s!important;transition:none!important;}}
    @media (max-width:760px){.summary{grid-template-columns:repeat(2,1fr);}.grid{grid-template-columns:1fr;}}
  </style>
</head>
<body>
  <header>
    <h1>Auralis Learning Trace</h1>
    <p>Checkpoint: <code>@@CHECKPOINT@@</code></p>
    <p>Train: <code>@@TRAIN@@</code> &nbsp;|&nbsp; Probes: <code>@@PROBEFILE@@</code></p>
  </header>
  <main>
    <section class="summary">
      <div class="stat"><label>Trace Steps</label><strong>@@STEPS@@</strong></div>
      <div class="stat"><label>Probes</label><strong>@@NPROBES@@</strong></div>
      <div class="stat"><label>Strong</label><strong style="color:var(--good)">@@STRONG@@</strong></div>
      <div class="stat"><label>Watch/Weak/Danger</label><strong style="color:var(--watch)">@@OTHERS@@</strong></div>
    </section>
    <table class="overview">
      <thead><tr><th>Probe</th><th>Kategorie</th><th>Status</th><th>Target NLL</th><th>Margin</th><th>Delta</th><th>Forbidden</th></tr></thead>
      <tbody>@@OVERVIEW@@</tbody>
    </table>
    <section class="grid">
      @@CARDS@@
    </section>
  </main>
  <script>
    document.querySelectorAll('.card').forEach((c,i)=>{ c.style.animationDelay=(Math.min(i,14)*0.04)+'s'; });
  </script>
</body>
</html>
"""


def render(trace: dict[str, Any]) -> str:
    grouped = probe_rows(trace)
    cards: list[tuple[str, str]] = []
    summary_rows = []

    for probe_id, rows in sorted(grouped.items()):
        first = rows[0]
        latest = rows[-1]
        cls = classify(latest)
        margin_points = [row.get("margin") for row in rows]
        target_points = [row.get("target_nll") for row in rows]
        latest_margin = latest.get("margin")
        first_margin = first.get("margin")
        delta_margin = None if latest_margin is None or first_margin is None else float(latest_margin) - float(first_margin)
        forbidden = latest.get("forbidden_hits") or []
        expected = latest.get("expected_hits") or []
        summary_rows.append(
            {
                "id": probe_id,
                "category": latest.get("category", "unknown"),
                "status": cls,
                "target_nll": latest.get("target_nll"),
                "margin": latest_margin,
                "delta_margin": delta_margin,
                "answer": latest.get("answer", ""),
                "forbidden": forbidden,
            }
        )
        top_tokens = latest.get("top_next_tokens") or []
        top_html = "".join(
            f"<span>{esc(tok.get('text') or tok.get('piece'))} <b>{float(tok.get('prob', 0.0)):.3f}</b></span>"
            for tok in top_tokens[:8]
        ) or '<span class="cat">n/a</span>'
        row_html = "".join(
            "<tr>"
            f"<td>{esc(row.get('_step'))}</td>"
            f"<td>{fmt(row.get('target_nll'))}</td>"
            f"<td>{fmt(row.get('negative_nll'))}</td>"
            f"<td>{fmt(row.get('margin'))}</td>"
            f"<td>{esc((row.get('answer') or '').strip())}</td>"
            "</tr>"
            for row in rows
        )
        uid = re.sub(r"[^A-Za-z0-9]", "-", probe_id)
        m_spark = sparkline(margin_points, color=STATUS_COLOR.get(cls, "#72a7ff"), uid=uid + "-m")
        t_spark = sparkline(target_points, color="#7af0c8", uid=uid + "-t")
        delta_cls = "" if delta_margin is None else (" good" if delta_margin > 0 else (" bad" if delta_margin < 0 else ""))
        cards.append(
            (
                cls,
                f"""
                <section class="card {cls}">
                  <div class="card-head">
                    <div><h2>{esc(probe_id)}</h2><span class="cat">{esc(latest.get('category', 'unknown'))}</span></div>
                    <span class="badge {cls}">{esc(cls)}</span>
                  </div>
                  <div class="metrics">
                    <div><label>Target NLL</label><strong>{fmt(latest.get('target_nll'))}</strong></div>
                    <div><label>Margin</label><strong>{fmt(latest.get('margin'))}</strong></div>
                    <div><label>&Delta; Margin</label><strong class="{delta_cls.strip()}">{fmt(delta_margin)}</strong></div>
                  </div>
                  <div class="charts">
                    <div><label>Margin-Verlauf</label>{m_spark}</div>
                    <div><label>Target-NLL-Verlauf</label>{t_spark}</div>
                  </div>
                  <h3>Letzte Antwort</h3>
                  <pre>{esc((latest.get('answer') or '').strip())}</pre>
                  <p class="hits"><span class="ok">Expected: {esc(expected)}</span><span class="no">Forbidden: {esc(forbidden)}</span></p>
                  <h3>Naechstes Token nach Prompt</h3>
                  <div class="tokens">{top_html}</div>
                  <details>
                    <summary>Step-Verlauf anzeigen</summary>
                    <table>
                      <thead><tr><th>Step</th><th>Target NLL</th><th>Negative NLL</th><th>Margin</th><th>Antwort</th></tr></thead>
                      <tbody>{row_html}</tbody>
                    </table>
                  </details>
                </section>
                """,
            )
        )

    order = {"danger": 0, "weak": 1, "watch": 2, "unknown": 3, "strong": 4}
    cards.sort(key=lambda item: order.get(item[0], 9))
    counts: dict[str, int] = {}
    for row in summary_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    summary_table = "".join(
        "<tr>"
        f"<td>{esc(row['id'])}</td>"
        f"<td>{esc(row['category'])}</td>"
        f"<td><span class='badge {esc(row['status'])}'>{esc(row['status'])}</span></td>"
        f"<td>{fmt(row['target_nll'])}</td>"
        f"<td>{fmt(row['margin'])}</td>"
        f"<td>{fmt(row['delta_margin'])}</td>"
        f"<td>{esc(row['forbidden'])}</td>"
        "</tr>"
        for row in sorted(summary_rows, key=lambda row: order.get(row["status"], 9))
    )
    steps = [item.get("step") for item in trace.get("history", [])]

    out = TRACE_TEMPLATE
    for token, value in {
        "@@CHECKPOINT@@": esc(trace.get("checkpoint")),
        "@@TRAIN@@": esc(trace.get("train")),
        "@@PROBEFILE@@": esc(trace.get("probe_file")),
        "@@STEPS@@": str(len(steps)),
        "@@NPROBES@@": str(len(grouped)),
        "@@STRONG@@": str(counts.get("strong", 0)),
        "@@OTHERS@@": str(counts.get("watch", 0) + counts.get("weak", 0) + counts.get("danger", 0)),
        "@@OVERVIEW@@": summary_table,
        "@@CARDS@@": "".join(card for _, card in cards),
    }.items():
        out = out.replace(token, value)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    args = parser.parse_args()

    trace = json.loads(args.trace_json.read_text(encoding="utf-8"))
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(render(trace), encoding="utf-8")
    print(f"wrote {args.output_html}")


if __name__ == "__main__":
    main()
