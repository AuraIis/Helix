#!/usr/bin/env python3
"""Calibrate adaptive-curriculum thresholds against existing checkpoints.

The mastery thresholds in a curriculum YAML are placeholders until you know what
"learned" looks like in *nats* for this probe set and this model. This tool runs
the LearningMonitor (no training) on one or more checkpoints and proposes
thresholds grounded in the data.

Key idea: the **retention probes are facts the model already knows reliably**, so
their margin distribution is the natural reference for "mastered". A target fact
is mastered when its margin reaches the level of facts the model already knows.

Run in the container:
    python scripts/eval/calibrate_adaptive_thresholds.py \
        --model-config configs/model/helix_v2_500m.yaml \
        --probes eval/adaptive_margin_probes_v1.yaml \
        --checkpoints checkpoints/.../best.pt checkpoints/.../step9000.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))


def _load_state_dict(path: Path):
    """Load a checkpoint robustly: unwrap common containers and strip the
    ``_orig_mod.`` prefix that torch.compile adds (the resume footgun)."""
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    state = obj
    for key in ("model", "model_state_dict", "state_dict"):
        if isinstance(obj, dict) and key in obj:
            state = obj[key]
            break
    if not isinstance(state, dict):
        raise SystemExit(f"{path}: could not find a state_dict")
    return { (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v
             for k, v in state.items() }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-config", type=Path, required=True)
    ap.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    ap.add_argument("--tokenizer", type=Path, default=None)
    ap.add_argument("--probes", type=Path, default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch

    from auralis.model.helix_model import build_model
    from auralis.adaptive.adapters import ModelAdapter, TokenizerAdapter
    from auralis.adaptive.monitor import LearningMonitor
    from auralis.adaptive.probes import DEFAULT_PROBES, load_margin_probes
    from statistics import mean, median

    model = build_model(args.model_config)
    model.to(args.device)
    tok_path = args.tokenizer or model.config.tokenizer_path
    tokenizer = TokenizerAdapter(tok_path)
    probes = load_margin_probes(args.probes) if args.probes else DEFAULT_PROBES
    model_adapter = ModelAdapter(model, device=args.device, autocast_dtype=torch.bfloat16)
    monitor = LearningMonitor(model_adapter, tokenizer, probes, trace_path=None)

    rows = []
    for ckpt in args.checkpoints:
        missing, unexpected = model.load_state_dict(_load_state_dict(ckpt), strict=False)
        if missing or unexpected:
            print(f"[warn] {ckpt.name}: missing={len(missing)} unexpected={len(unexpected)}")
        snap = monitor.evaluate(step=0, stage_step=0, stage_name=f"calib:{ckpt.name}")
        rows.append((ckpt.name, snap.metrics))

    # --- report ---
    cols = ["target_margin_mean", "target_pass", "retention_margin_mean", "retention_pass"]
    print("\n=== Calibration across checkpoints ===")
    print("checkpoint".ljust(40) + "".join(c.rjust(22) for c in cols))
    for name, m in rows:
        print(name.ljust(40) + "".join(f"{m.get(c, float('nan')):.4f}".rjust(22) for c in cols))

    # --- propose thresholds, anchored to the retention (already-known) scale ---
    ret_scale = [m["retention_margin_mean"] for _, m in rows if "retention_margin_mean" in m]
    if ret_scale:
        anchor = median(ret_scale)
        suggested_target_thr = round(max(0.5, 0.8 * anchor), 2)
        suggested_guard_drop = round(max(0.1, 0.25 * abs(anchor)), 2)
        print("\n=== Suggested curriculum thresholds (starting points) ===")
        print(f"# retention (known-fact) margin scale ~ {anchor:.3f} nats -> the 'mastered' reference")
        print("mastery:")
        print("  metric: target_margin_mean")
        print("  mode: either")
        print(f"  threshold: {suggested_target_thr}   # ~80% of the known-fact margin scale")
        print("  window: 4")
        print("  patience: 6")
        print("  min_delta: 0.02")
        print("guard:")
        print("  metric: retention_margin_mean")
        print(f"  max_drop: {suggested_guard_drop}   # ~25% of the known-fact margin scale")
        print("\nNote: these are starting points. The 1B run should beat the best target")
        print("margin you saw above; set 'threshold' above that if you want true gains.")
    else:
        print("\n[warn] no retention probes -> cannot anchor thresholds. Add split: retention probes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
