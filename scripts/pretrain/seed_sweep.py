"""Mini seed sweep — variance estimator.

Runs a short training loop N times with different seeds, collects final
train_loss + val_loss, reports mean ± std. Answers "is this setup robust
or just lucky?" before committing to a full run.

Uses ``scripts.pretrain.smoke_test`` under the hood so it never needs
real tokenised data — synthetic bins are fine to pick up *setup* issues
(non-deterministic init, kernel-launch races, etc.).
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import json
from pathlib import Path
import re

REPO = Path(__file__).resolve().parents[2]


def _run_one(seed: int, steps: int, model_config: Path, device: str, dtype: str,
             batch: int, seq: int) -> dict:
    """Spawn smoke_test with a given seed and parse its report."""
    env = {**__import__("os").environ, "PYTHONHASHSEED": str(seed)}
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "pretrain" / "smoke_test.py"),
         "--device", device, "--dtype", dtype,
         "--model-config", str(model_config),
         "--steps", str(steps),
         "--batch-size", str(batch),
         "--seq-length", str(seq),
         "--warmup-steps", str(max(1, steps // 10)),
         "--lr", "1e-3"],
        capture_output=True, text=True, env=env,
    )
    out = result.stdout
    rx_loss_last = re.search(r"loss last\s*:\s*([-\d.]+)", out)
    rx_loss_first = re.search(r"loss first\s*:\s*([-\d.]+)", out)
    rx_tps = re.search(r"tokens / second\s*:\s*([\d,\.]+)", out)
    return {
        "seed": seed,
        "return_code": result.returncode,
        "loss_first": float(rx_loss_first.group(1)) if rx_loss_first else None,
        "loss_last": float(rx_loss_last.group(1)) if rx_loss_last else None,
        "tokens_per_second": float(rx_tps.group(1).replace(",", "")) if rx_tps else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--model-config", type=Path,
                   default=REPO / "configs" / "model" / "helix_v2_100m.yaml")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--dtype", default="fp32")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-length", type=int, default=64)
    p.add_argument("--output", type=Path,
                   default=REPO / "checkpoints" / "seed_sweep.json")
    args = p.parse_args()

    rows = [_run_one(s, args.steps, args.model_config, args.device, args.dtype,
                     args.batch_size, args.seq_length) for s in args.seeds]

    loss_last_vals = [r["loss_last"] for r in rows if r["loss_last"] is not None]
    delta_vals = [r["loss_first"] - r["loss_last"] for r in rows
                  if r["loss_first"] is not None and r["loss_last"] is not None]

    summary = {
        "n_runs": len(rows),
        "runs": rows,
        "loss_last_mean": statistics.fmean(loss_last_vals) if loss_last_vals else None,
        "loss_last_std": statistics.pstdev(loss_last_vals) if len(loss_last_vals) > 1 else 0.0,
        "loss_delta_mean": statistics.fmean(delta_vals) if delta_vals else None,
        "loss_delta_std": statistics.pstdev(delta_vals) if len(delta_vals) > 1 else 0.0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nseed-sweep {args.seeds} over {args.steps} steps:")
    for r in rows:
        print(f"  seed {r['seed']}: loss {r['loss_first']:.3f} → {r['loss_last']:.3f} "
              f"(tps {r['tokens_per_second']}, rc {r['return_code']})")
    print(f"  loss_last  = {summary['loss_last_mean']:.4f} ± {summary['loss_last_std']:.4f}")
    print(f"  loss_delta = {summary['loss_delta_mean']:+.4f} ± {summary['loss_delta_std']:.4f}")
    if summary["loss_last_std"] > 0.5:
        print("⚠ high variance across seeds — setup may be flaky.")


if __name__ == "__main__":
    main()
