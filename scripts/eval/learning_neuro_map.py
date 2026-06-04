#!/usr/bin/env python3
"""Render a live-style knowledge map from learning trace JSON.

This is an interpretability proxy, not a literal neuron microscope. It turns
the learning trace into a visual graph:

- category -> probe
- probe -> desired target answer
- probe -> dangerous negative answer
- probe -> current generated answer

Edges change color/strength from the latest target/negative likelihood margin
and forbidden-answer flags. The HTML is self-contained and can auto-refresh
while training rewrites it. The graph is drawn as a glowing, force-directed
network (no third-party libraries).
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def esc(value: object) -> str:
    return html.escape("" if value is None else str(value))


def fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return esc(value)


def status_for(row: dict[str, Any]) -> str:
    forbidden = row.get("forbidden_hits") or []
    margin = row.get("margin")
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


def latest_rows(trace: dict[str, Any]) -> list[dict[str, Any]]:
    history = trace.get("history") or []
    if not history:
        return []
    return list(history[-1].get("probes") or [])


def build_payload(trace: dict[str, Any]) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    rows = latest_rows(trace)

    def add_node(node_id: str, label: str, kind: str, status: str = "neutral", detail: str = "") -> None:
        nodes.setdefault(
            node_id,
            {
                "id": node_id,
                "label": label,
                "kind": kind,
                "status": status,
                "detail": detail,
            },
        )

    for row in rows:
        pid = str(row.get("id"))
        category = str(row.get("category") or "unknown")
        status = status_for(row)
        prompt = str(row.get("prompt") or pid)
        target = row.get("best_target") or {}
        negative = row.get("best_negative") or {}
        answer = str(row.get("answer") or "").strip()
        margin = row.get("margin")

        cat_id = f"cat:{category}"
        probe_id = f"probe:{pid}"
        target_id = f"target:{pid}"
        negative_id = f"negative:{pid}"
        answer_id = f"answer:{pid}"

        add_node(cat_id, category, "category")
        add_node(probe_id, pid, "probe", status, prompt)
        add_node(target_id, "Ziel", "target", "strong", str(target.get("text") or "").strip())
        add_node(negative_id, "Falsch", "negative", "danger", str(negative.get("text") or "").strip())
        add_node(answer_id, "Antwort", "answer", status, answer)

        # Enrich the probe node so the force graph can show margin/answer in one
        # node without the extra leaf nodes cluttering the layout.
        nodes[probe_id]["margin"] = float(margin) if margin is not None else None
        nodes[probe_id]["answer"] = answer
        nodes[probe_id]["category"] = category

        edges.append({"source": cat_id, "target": probe_id, "kind": "category", "status": "neutral", "label": category})
        edges.append(
            {
                "source": probe_id,
                "target": target_id,
                "kind": "target",
                "status": status if status != "danger" else "watch",
                "label": f"target nll {fmt(row.get('target_nll'))}",
                "weight": max(1.0, 5.0 - float(row.get("target_nll") or 5.0)),
            }
        )
        edges.append(
            {
                "source": probe_id,
                "target": negative_id,
                "kind": "negative",
                "status": "danger" if margin is not None and float(margin) < 0.75 else "weak",
                "label": f"neg nll {fmt(row.get('negative_nll'))}",
                "weight": max(1.0, 4.0 - float(row.get("negative_nll") or 4.0)),
            }
        )
        edges.append(
            {
                "source": probe_id,
                "target": answer_id,
                "kind": "answer",
                "status": status,
                "label": f"margin {fmt(margin)}",
                "weight": max(1.0, min(5.0, 2.0 + float(margin or 0.0))),
            }
        )

    return {"nodes": list(nodes.values()), "edges": edges}


# ---------------------------------------------------------------------------
# Static HTML template. NOT an f-string: braces are literal so the force-graph
# JS stays readable. Dynamic values are injected via .replace() in render().
# ---------------------------------------------------------------------------
NEURO_TEMPLATE = r"""<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  @@REFRESH@@
  <title>Auralis Live Neuro Map</title>
  <style>
    :root{
      --bg0:#070a10; --panel:rgba(20,26,38,.72); --stroke:rgba(120,150,180,.16);
      --text:#eef3fb; --muted:#9aa7bd; --faint:#5f6c83;
      --good:#3ddc84; --watch:#f3c84b; --weak:#ff9b54; --bad:#ff5f6d; --blue:#72a7ff; --accent:#7af0c8;
    }
    *{box-sizing:border-box;}
    body{margin:0;color:var(--text);font-size:14px;
      font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
      background:var(--bg0);
      background-image:radial-gradient(900px 500px at 20% -10%,rgba(124,137,255,.12),transparent 60%),
        radial-gradient(900px 520px at 85% -10%,rgba(61,220,132,.10),transparent 60%);
      -webkit-font-smoothing:antialiased;}
    header{padding:20px 26px;border-bottom:1px solid var(--stroke);
      background:rgba(8,11,18,.7);backdrop-filter:blur(12px);}
    h1{margin:0 0 8px;font-size:24px;font-weight:780;letter-spacing:.2px;
      background:linear-gradient(110deg,#b388ff,#5ec8ff,#3ddc84);-webkit-background-clip:text;background-clip:text;color:transparent;}
    header p{color:var(--muted);margin:3px 0;font-size:12.5px;}
    header code{color:var(--accent);background:rgba(122,240,200,.08);padding:1px 6px;border-radius:5px;}
    main{padding:20px 26px 44px;max-width:1500px;margin:0 auto;}
    .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:16px;}
    .stat{position:relative;overflow:hidden;background:var(--panel);backdrop-filter:blur(8px);
      border:1px solid var(--stroke);border-radius:14px;padding:14px 16px;animation:rise .5s both;}
    .stat::before{content:"";position:absolute;top:0;left:0;right:0;height:3px;}
    .stat.s::before{background:var(--good);} .stat.wt::before{background:var(--watch);}
    .stat.wk::before{background:var(--weak);} .stat.d::before{background:var(--bad);}
    .stat label{display:block;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.09em;font-weight:600;}
    .stat strong{font-size:30px;font-weight:780;display:block;margin-top:6px;font-variant-numeric:tabular-nums;}
    .stat.s strong{color:var(--good);} .stat.wt strong{color:var(--watch);}
    .stat.wk strong{color:var(--weak);} .stat.d strong{color:var(--bad);}
    .graphwrap{position:relative;border:1px solid var(--stroke);border-radius:16px;overflow:hidden;
      background:radial-gradient(700px 460px at 50% 42%,rgba(124,137,255,.06),transparent 70%),#080b12;
      box-shadow:inset 0 1px 0 rgba(255,255,255,.03),0 18px 50px rgba(0,0,0,.4);animation:rise .5s both;}
    #graph{display:block;width:100%;height:620px;cursor:grab;}
    #graph:active{cursor:grabbing;}
    .legend{position:absolute;top:12px;left:14px;display:flex;gap:14px;flex-wrap:wrap;
      padding:8px 12px;border-radius:11px;background:rgba(8,11,18,.6);border:1px solid var(--stroke);
      font-size:11.5px;color:var(--muted);font-weight:600;}
    .legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:6px;box-shadow:0 0 8px currentColor;vertical-align:-1px;}
    .hint{position:absolute;bottom:12px;right:14px;font-size:11px;color:var(--faint);
      background:rgba(8,11,18,.55);padding:5px 10px;border-radius:8px;border:1px solid var(--stroke);}
    .panel{margin-top:18px;background:var(--panel);backdrop-filter:blur(8px);
      border:1px solid var(--stroke);border-radius:14px;overflow:auto;animation:rise .5s both;}
    table{width:100%;border-collapse:collapse;font-size:12.5px;}
    th,td{padding:10px 12px;border-bottom:1px solid var(--stroke);text-align:left;vertical-align:top;}
    th{color:var(--faint);font-size:10.5px;text-transform:uppercase;letter-spacing:.07em;font-weight:600;}
    tbody tr{transition:background .15s;} tbody tr:hover{background:rgba(124,137,255,.06);}
    td:nth-child(3),td:nth-child(4),td:nth-child(5),th:nth-child(3),th:nth-child(4),th:nth-child(5){text-align:right;font-variant-numeric:tabular-nums;}
    .pill{display:inline-block;padding:3px 9px;border-radius:999px;font-size:11px;font-weight:700;}
    .pill.strong{background:rgba(61,220,132,.16);color:var(--good);}
    .pill.watch{background:rgba(243,200,75,.16);color:var(--watch);}
    .pill.weak{background:rgba(255,155,84,.16);color:var(--weak);}
    .pill.danger{background:rgba(255,95,109,.16);color:var(--bad);}
    .ntip{position:fixed;z-index:50;max-width:360px;padding:11px 13px;border-radius:11px;pointer-events:none;
      background:rgba(7,10,16,.97);border:1px solid var(--stroke);box-shadow:0 14px 40px rgba(0,0,0,.6);
      font-size:12px;line-height:1.5;opacity:0;transform:translateY(5px);transition:opacity .13s,transform .13s;}
    .ntip.show{opacity:1;transform:none;} .ntip b{color:var(--accent);} .ntip .s{color:var(--muted);font-size:11px;}
    @keyframes rise{from{opacity:0;transform:translateY(10px);}to{opacity:1;transform:none;}}
    @media (max-width:700px){.stats{grid-template-columns:repeat(2,1fr);} #graph{height:460px;}}
  </style>
</head>
<body>
  <header>
    <h1>Auralis Live Neuro Map</h1>
    <p>Step: <strong>@@STEP@@</strong> &nbsp;|&nbsp; Val-Loss: <strong>@@VALLOSS@@</strong> &nbsp;|&nbsp; Auto-Refresh: @@REFRESHLABEL@@</p>
    <p>Trace: <code>@@TRACE@@</code> &nbsp;|&nbsp; Checkpoint: <code>@@CHECKPOINT@@</code></p>
  </header>
  <main>
    <section class="stats">
      <div class="stat s"><label>Strong</label><strong>@@CSTRONG@@</strong></div>
      <div class="stat wt"><label>Watch</label><strong>@@CWATCH@@</strong></div>
      <div class="stat wk"><label>Weak</label><strong>@@CWEAK@@</strong></div>
      <div class="stat d"><label>Danger</label><strong>@@CDANGER@@</strong></div>
    </section>

    <div class="graphwrap">
      <canvas id="graph"></canvas>
      <div class="legend">
        <span><i style="background:#b388ff;color:#b388ff"></i>Kategorie</span>
        <span><i style="background:var(--good);color:var(--good)"></i>strong</span>
        <span><i style="background:var(--watch);color:var(--watch)"></i>watch</span>
        <span><i style="background:var(--weak);color:var(--weak)"></i>weak</span>
        <span><i style="background:var(--bad);color:var(--bad)"></i>danger</span>
      </div>
      <div class="hint">Hintergrund ziehen = verschieben · Mausrad = zoomen · Knoten ziehen</div>
    </div>

    <section class="panel">
      <table>
        <thead><tr><th>Probe</th><th>Status</th><th>Target NLL</th><th>Negative NLL</th><th>Margin</th><th>Forbidden</th><th>Antwort</th></tr></thead>
        <tbody>@@ROWS@@</tbody>
      </table>
    </section>
  </main>
  <div class="ntip" id="ntip"></div>

  <script>
    const DATA = @@PAYLOAD@@;
    const SC = {strong:'#3ddc84', watch:'#f3c84b', weak:'#ff9b54', danger:'#ff5f6d', neutral:'#7c89a4', unknown:'#8f98aa'};
    const PAL = ['#b388ff','#5ec8ff','#ff8fc8','#8cf0a0','#ffd166','#7af0c8','#ff9b54','#c0a3ff'];

    const cats = DATA.nodes.filter(n => n.kind === 'category');
    const probes = DATA.nodes.filter(n => n.kind === 'probe');
    const catColor = {}; cats.forEach((c,i) => catColor[c.id] = PAL[i % PAL.length]);
    const nodes = [
      ...cats.map(c => ({...c, isCat:true, color:catColor[c.id], r:18})),
      ...probes.map(p => ({...p, isCat:false, color:SC[p.status] || '#7c89a4', r:11})),
    ];
    const NID = new Map(nodes.map(n => [n.id, n]));
    const edges = DATA.edges.filter(e => e.kind === 'category')
      .map(e => ({s:e.source, t:e.target}))
      .filter(e => NID.has(e.s) && NID.has(e.t));

    const cx = 600, cy = 320;
    function hash(s){ let h = 2166136261; for(let i=0;i<s.length;i++){ h ^= s.charCodeAt(i); h = Math.imul(h, 16777619);} return (h>>>0)/4294967295; }
    nodes.forEach(n => { const a = hash(n.id)*6.2832, rad = (n.isCat?110:240) + hash(n.id+'r')*120;
      n.x = cx + Math.cos(a)*rad; n.y = cy + Math.sin(a)*rad; n.vx = 0; n.vy = 0; n.deg = 0; });
    edges.forEach(e => { NID.get(e.s).deg++; NID.get(e.t).deg++; });
    nodes.forEach(n => { if(n.isCat) n.r = 14 + Math.min(11, n.deg*1.7); });

    const cv = document.getElementById('graph'), ctx = cv.getContext('2d');
    const tip = document.getElementById('ntip');
    let W = 1200, H = 620, DPR = 1;
    const view = { k:1, x:0, y:0, init:false };
    let hoverNode = null, dragNode = null, panning = false, last = {x:0,y:0};

    function fit(){
      DPR = window.devicePixelRatio || 1;
      const r = cv.getBoundingClientRect();
      W = r.width; H = r.height;
      cv.width = W*DPR; cv.height = H*DPR;
      if(!view.init && W > 0){
        view.k = Math.min(W/1200, H/640) * 0.96 || 0.6;
        view.x = (W - 1200*view.k)/2; view.y = (H - 640*view.k)/2 + 10;
        view.init = true;
      }
    }
    const toWorld = (mx,my) => ({ x:(mx - view.x)/view.k, y:(my - view.y)/view.k });

    function tick(){
      for(let i=0;i<nodes.length;i++){ const a = nodes[i];
        for(let j=i+1;j<nodes.length;j++){ const b = nodes[j];
          let dx=a.x-b.x, dy=a.y-b.y, d2=dx*dx+dy*dy+0.01, d=Math.sqrt(d2);
          const f = (a.isCat||b.isCat?4200:2600)/d2, ux=dx/d, uy=dy/d;
          a.vx+=ux*f; a.vy+=uy*f; b.vx-=ux*f; b.vy-=uy*f;
        }
        a.vx += (cx-a.x)*0.0016; a.vy += (cy-a.y)*0.0016;
      }
      for(const e of edges){ const a=NID.get(e.s), b=NID.get(e.t);
        let dx=b.x-a.x, dy=b.y-a.y, d=Math.hypot(dx,dy)||0.01;
        const rest = a.isCat||b.isCat ? 132 : 95, f=(d-rest)*0.018, ux=dx/d, uy=dy/d;
        a.vx+=ux*f; a.vy+=uy*f; b.vx-=ux*f; b.vy-=uy*f;
      }
      for(const n of nodes){ if(n===dragNode) continue; n.vx*=0.85; n.vy*=0.85; n.x+=n.vx; n.y+=n.vy; }
    }

    function draw(){
      fit();
      ctx.setTransform(DPR,0,0,DPR,0,0);
      ctx.clearRect(0,0,W,H);
      ctx.save(); ctx.translate(view.x,view.y); ctx.scale(view.k,view.k);
      ctx.lineCap='round';
      for(const e of edges){ const a=NID.get(e.s), b=NID.get(e.t);
        const hot = hoverNode && (hoverNode===a || hoverNode===b);
        const mx=(a.x+b.x)/2 + (b.y-a.y)*0.14, my=(a.y+b.y)/2 - (b.x-a.x)*0.14;
        const g = ctx.createLinearGradient(a.x,a.y,b.x,b.y);
        g.addColorStop(0,a.color); g.addColorStop(1,b.color);
        ctx.strokeStyle=g; ctx.globalAlpha = hot?0.95:0.30; ctx.lineWidth = hot?2.4:1.2;
        ctx.shadowColor=b.color; ctx.shadowBlur = hot?14:5;
        ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.quadraticCurveTo(mx,my,b.x,b.y); ctx.stroke();
      }
      ctx.globalAlpha=1; ctx.shadowBlur=0;
      for(const n of nodes){
        const hot = hoverNode===n;
        ctx.shadowColor=n.color; ctx.shadowBlur = hot?28:15; ctx.fillStyle=n.color;
        if(n.isCat){
          ctx.save(); ctx.translate(n.x,n.y); ctx.rotate(Math.PI/4);
          const s=n.r*1.55; ctx.fillRect(-s/2,-s/2,s,s);
          ctx.restore();
        } else {
          ctx.beginPath(); ctx.arc(n.x,n.y,n.r,0,6.2832); ctx.fill();
        }
        ctx.shadowBlur=0;
        ctx.globalAlpha=.25; ctx.fillStyle='#fff';
        ctx.beginPath(); ctx.arc(n.x - n.r*0.28, n.y - n.r*0.28, n.r*0.32, 0, 6.2832); ctx.fill();
        ctx.globalAlpha=1;
      }
      ctx.shadowBlur=0; ctx.textAlign='center'; ctx.textBaseline='middle';
      for(const n of nodes){
        const label = n.isCat || hoverNode===n || n.status==='danger' || n.status==='strong';
        if(!label) continue;
        ctx.font = n.isCat ? '700 14px Inter,system-ui' : '600 11px Inter,system-ui';
        const ty = n.y + n.r + (n.isCat?18:13);
        ctx.lineWidth=3; ctx.strokeStyle='rgba(7,10,16,.9)';
        ctx.strokeText(n.label, n.x, ty);
        ctx.fillStyle = n.isCat ? '#ffffff' : (hoverNode===n?'#fff':'rgba(228,236,245,.82)');
        ctx.fillText(n.label, n.x, ty);
      }
      ctx.restore();
    }

    function frame(){ tick(); draw(); requestAnimationFrame(frame); }

    function pick(mx,my){
      const w = toWorld(mx,my); let best=null, bd=1e9;
      for(const n of nodes){ const d=Math.hypot(n.x-w.x, n.y-w.y); if(d < n.r+6 && d<bd){ bd=d; best=n; } }
      return best;
    }
    function showTip(n, ev){
      const m = (n.margin===null||n.margin===undefined)?'n/a':Number(n.margin).toFixed(3);
      tip.innerHTML = n.isCat
        ? `<b>${n.label}</b><div class="s">Kategorie · ${n.deg} Probes</div>`
        : `<b>${n.label}</b><div class="s">${n.category||''} · ${n.status} · margin ${m}</div>`
          + (n.answer?`<div style="margin-top:7px;color:#dbe4f0">${esc(n.answer)}</div>`:'')
          + (n.detail?`<div class="s" style="margin-top:6px">${esc(n.detail)}</div>`:'');
      tip.classList.add('show');
      tip.style.left = Math.min(ev.clientX+15, window.innerWidth-380)+'px';
      tip.style.top = (ev.clientY+15)+'px';
    }
    function esc(s){ return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }

    cv.addEventListener('mousemove', ev => {
      const r=cv.getBoundingClientRect(), mx=ev.clientX-r.left, my=ev.clientY-r.top;
      if(dragNode){ const w=toWorld(mx,my); dragNode.x=w.x; dragNode.y=w.y; dragNode.vx=0; dragNode.vy=0; showTip(dragNode,ev); return; }
      if(panning){ view.x += (ev.clientX-last.x); view.y += (ev.clientY-last.y); last={x:ev.clientX,y:ev.clientY}; return; }
      const n = pick(mx,my); hoverNode = n;
      if(n) showTip(n, ev); else tip.classList.remove('show');
    });
    cv.addEventListener('mousedown', ev => {
      const r=cv.getBoundingClientRect(), n=pick(ev.clientX-r.left, ev.clientY-r.top);
      if(n){ dragNode=n; } else { panning=true; last={x:ev.clientX,y:ev.clientY}; }
    });
    window.addEventListener('mouseup', () => { dragNode=null; panning=false; });
    cv.addEventListener('mouseleave', () => { tip.classList.remove('show'); hoverNode=null; });
    cv.addEventListener('wheel', ev => {
      ev.preventDefault();
      const r=cv.getBoundingClientRect(), mx=ev.clientX-r.left, my=ev.clientY-r.top;
      const w=toWorld(mx,my), f=ev.deltaY<0?1.12:0.89;
      view.k = Math.max(0.25, Math.min(3, view.k*f));
      view.x = mx - w.x*view.k; view.y = my - w.y*view.k;
    }, {passive:false});

    if(nodes.length){ frame(); }
    else { fit(); ctx.fillStyle='#5f6c83'; ctx.font='15px Inter,system-ui'; ctx.fillText('Noch keine Trace-Daten…', 24, 40); }
  </script>
</body>
</html>
"""


def render(trace: dict[str, Any], auto_refresh: int = 0) -> str:
    payload = build_payload(trace)
    history = trace.get("history") or []
    latest = history[-1] if history else {}
    rows = latest_rows(trace)
    counts: dict[str, int] = {}
    for row in rows:
        counts[status_for(row)] = counts.get(status_for(row), 0) + 1
    rows_html = "".join(
        "<tr>"
        f"<td>{esc(row.get('id'))}</td>"
        f"<td><span class='pill {status_for(row)}'>{status_for(row)}</span></td>"
        f"<td>{fmt(row.get('target_nll'))}</td>"
        f"<td>{fmt(row.get('negative_nll'))}</td>"
        f"<td>{fmt(row.get('margin'))}</td>"
        f"<td>{esc(row.get('forbidden_hits') or [])}</td>"
        f"<td>{esc((row.get('answer') or '').strip())}</td>"
        "</tr>"
        for row in sorted(rows, key=lambda r: {"danger": 0, "weak": 1, "watch": 2, "strong": 3}.get(status_for(r), 9))
    )
    refresh_tag = f'<meta http-equiv="refresh" content="{auto_refresh}">' if auto_refresh > 0 else ""

    out = NEURO_TEMPLATE
    for token, value in {
        "@@REFRESH@@": refresh_tag,
        "@@STEP@@": esc(latest.get("step")),
        "@@VALLOSS@@": fmt(latest.get("val_loss")),
        "@@REFRESHLABEL@@": str(auto_refresh) if auto_refresh else "aus",
        "@@TRACE@@": esc(trace.get("probe_file")),
        "@@CHECKPOINT@@": esc(trace.get("checkpoint")),
        "@@CSTRONG@@": str(counts.get("strong", 0)),
        "@@CWATCH@@": str(counts.get("watch", 0)),
        "@@CWEAK@@": str(counts.get("weak", 0)),
        "@@CDANGER@@": str(counts.get("danger", 0)),
        "@@ROWS@@": rows_html,
    }.items():
        out = out.replace(token, value)
    out = out.replace("@@PAYLOAD@@", json.dumps(payload, ensure_ascii=False))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path, required=True)
    parser.add_argument("--auto-refresh", type=int, default=0)
    args = parser.parse_args()

    trace = json.loads(args.trace_json.read_text(encoding="utf-8"))
    args.output_html.parent.mkdir(parents=True, exist_ok=True)
    args.output_html.write_text(render(trace, auto_refresh=args.auto_refresh), encoding="utf-8")
    print(f"wrote {args.output_html}")


if __name__ == "__main__":
    main()
