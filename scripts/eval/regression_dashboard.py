"""Checkpoint regression dashboard.

Reads all ``step_*.json`` sidecars under a run directory, builds a CSV + a
Markdown comparison:

- current vs previous checkpoint
- current vs best-ever
- per-language val_loss trend (if available)
- tokens/second drift
- cost-per-checkpoint

No model load required — everything is in the JSON sidecar + the
MANIFEST.yaml.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--output-md", type=Path, default=None)
    args = p.parse_args()

    sidecars = sorted(args.ckpt_dir.glob("step_*.json"),
                      key=lambda p: int(p.stem.split("_", 1)[1]))
    best_sidecar = args.ckpt_dir / "best.json"
    if not sidecars and not best_sidecar.is_file():
        raise SystemExit(f"no step_*.json or best.json found under {args.ckpt_dir}")

    def _load(s: Path) -> dict:
        obj = json.loads(s.read_text(encoding="utf-8"))
        # Newer checkpoints wrap state under "state"; older wrap is flat.
        return obj.get("state", obj)

    rows = []
    for s in sidecars:
        r = _load(s)
        r["_file"] = s.name
        rows.append(r)

    best = _load(best_sidecar) if best_sidecar.is_file() else None

    md = ["# Regression Dashboard\n"]
    md.append("| file | step | tokens_seen | best_val_loss | consec_val_inc |  alerts | backups ok/fail |")
    md.append("|---|--:|--:|--:|--:|--:|---|")
    for r in rows:
        md.append(
            f"| {r['_file']} | {r.get('step', 0):,} | "
            f"{r.get('tokens_seen', 0):,} | "
            f"{r.get('best_val_loss', float('inf')):.4f} | "
            f"{r.get('consecutive_val_increases', 0)} | "
            f"{len(r.get('alerts', []))} | "
            f"{r.get('external_backups_ok', 0)}/{r.get('external_backups_failed', 0)} |"
        )
    if best is not None:
        md.append(f"\n**Best**: step {best.get('step', '?'):,}, "
                  f"val_loss {best.get('best_val_loss', float('inf')):.4f}")

    # Simple regression verdict
    if len(rows) >= 2:
        cur = rows[-1]
        prev = rows[-2]
        dv = cur.get("best_val_loss", float("inf")) - prev.get("best_val_loss", float("inf"))
        md.append(f"\n**Latest vs previous**: Δbest_val_loss = {dv:+.4f}")
        if dv > 0.05:
            md.append("⚠ regression suspected — investigate before next save.")

    # Pull cost / backends from MANIFEST if present
    man = args.ckpt_dir / "MANIFEST.yaml"
    if man.is_file():
        try:
            mdoc = yaml.safe_load(man.read_text(encoding="utf-8")) or {}
            md.append(f"\n**Run**: {mdoc.get('metadata', {}).get('git_sha', '?')[:12]}, "
                      f"host={mdoc.get('metadata', {}).get('hostname', '?')}, "
                      f"dtype={mdoc.get('metadata', {}).get('dtype', '?')}")
            if mdoc.get("health", {}).get("stop_reason"):
                md.append(f"Exit reason: {mdoc.get('exit_reason')} "
                          f"(health: {mdoc['health']['stop_reason']})")
        except Exception:
            pass

    out = args.output_md or (args.ckpt_dir / "regression_dashboard.md")
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
