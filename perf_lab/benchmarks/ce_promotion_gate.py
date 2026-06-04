#!/usr/bin/env python3
"""Self-calibrating promotion gate for huge-vocab CE candidates.

The drift question is always relative: "does candidate X add drift *beyond* what
bf16 itself costs?". This script answers it in one command. It runs
`loss_drift_ce.py --reference-fp32` (every series vs the fp32 ground truth) for:

- the **floor**: `--impl pytorch` (PyTorch bf16 full-logits CE) — the inherent
  bf16-vs-fp32 noise a candidate cannot beat and need not exceed, and
- each requested **candidate**.

It then derives the pass bar from the *measured* floor (floor x --tolerance)
rather than a hand-picked constant, and prints a PASS/FAIL table on the metric
that actually decides promotability: max upstream-gradient L2-relative drift,
plus a flat-drift check (accumulation slope ~ 0).

Each candidate runs as a **subprocess**, so every measurement gets a fresh CUDA
context — no cross-run caching-allocator state leaking into the next memory
number (the exact artifact that made an in-process 200k memory comparison
misleading).

Examples:
    # Is Liger promotable vs the bf16 floor, small shape?
    python perf_lab/benchmarks/ce_promotion_gate.py --candidates liger \
        --steps 50 --tokens 128 --d-model 256 --input-dim 256 --vocab-size 8192

    # Head-to-head: Liger vs the hand-rolled atomic_mixed, Auralis-ish shape.
    python perf_lab/benchmarks/ce_promotion_gate.py \
        --candidates liger,triton_fused:atomic_mixed \
        --steps 100 --tokens 256 --d-model 1280 --input-dim 512 --vocab-size 200000
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
DRIFT_SCRIPT = HERE.parent / "loss_drift_ce.py"


def parse_candidate(spec: str) -> tuple[str, str | None]:
    """`liger` -> ("liger", None); `triton_fused:atomic_mixed` -> ("triton_fused", "atomic_mixed")."""
    if ":" in spec:
        impl, mode = spec.split(":", 1)
        return impl, mode
    return spec, None


def run_drift(impl: str, backward_mode: str | None, shape: dict[str, object]) -> dict:
    """Run loss_drift_ce.py as a subprocess and return its parsed JSON report."""
    cmd = [
        sys.executable, str(DRIFT_SCRIPT),
        "--impl", impl,
        "--reference-fp32",
        "--history-every", "0",
        "--steps", str(shape["steps"]),
        "--tokens", str(shape["tokens"]),
        "--input-dim", str(shape["input_dim"]),
        "--d-model", str(shape["d_model"]),
        "--vocab-size", str(shape["vocab_size"]),
        "--dtype", str(shape["dtype"]),
    ]
    if shape.get("ignore_frac"):
        cmd += ["--ignore-frac", str(shape["ignore_frac"])]
    if impl == "triton_fused" and backward_mode:
        cmd += ["--triton-backward-mode", backward_mode]
    if impl == "liger":
        cmd += ["--accum-dtype", str(shape.get("accum_dtype", "fp32"))]

    env = dict(os.environ)
    env.setdefault("TRITON_OVERRIDE_ARCH", "sm89")  # Blackwell
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if not proc.stdout.strip():
        raise RuntimeError(
            f"no JSON from {impl}{':' + backward_mode if backward_mode else ''}\n"
            f"stderr tail:\n{proc.stderr[-2000:]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"bad JSON from {impl}: {exc}\nstdout tail:\n{proc.stdout[-2000:]}") from exc


def summarize(report: dict) -> dict:
    mm = report["max_metrics"]
    acc = report["accumulation"]
    grad_l2 = max(mm["proj_grad_l2_rel"], mm["head_grad_l2_rel"])

    def slope(key: str) -> float | None:
        s = acc.get(key, {}).get("slope_per_step")
        return None if s is None else float(s)

    return {
        "grad_l2_rel": grad_l2,
        "proj_grad_l2_rel": mm["proj_grad_l2_rel"],
        "head_grad_l2_rel": mm["head_grad_l2_rel"],
        "loss_abs": mm["loss_abs"],
        "proj_grad_slope": slope("proj_grad_l2_rel"),
        "head_grad_slope": slope("head_grad_l2_rel"),
        "finite": bool(report["final_step"].get("finite", False)),
        "steps_completed": report.get("steps_completed"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--candidates", default="liger",
                   help="comma list of impls. Use impl:mode for triton_fused, e.g. "
                        "'liger,triton_fused:atomic_mixed'. The pytorch floor is always run.")
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--input-dim", type=int, default=256)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--vocab-size", type=int, default=8192)
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--ignore-frac", type=float, default=0.0)
    p.add_argument("--accum-dtype", choices=["none", "fp32"], default="fp32",
                   help="Liger gradient accumulation dtype.")
    p.add_argument("--tolerance", type=float, default=1.5,
                   help="A candidate PASSES if its max upstream-grad L2-rel drift is "
                        "<= floor * tolerance (default 1.5). The floor is the measured "
                        "PyTorch-bf16-vs-fp32 noise.")
    p.add_argument("--slope-max", type=float, default=1e-4,
                   help="A candidate must also have |grad-drift slope/step| <= this "
                        "(drift must be flat noise, not accumulating bias).")
    args = p.parse_args()

    shape = {
        "steps": args.steps, "tokens": args.tokens, "input_dim": args.input_dim,
        "d_model": args.d_model, "vocab_size": args.vocab_size, "dtype": args.dtype,
        "ignore_frac": args.ignore_frac, "accum_dtype": args.accum_dtype,
    }

    print(f"# CE promotion gate  shape: tokens={args.tokens} d_model={args.d_model} "
          f"vocab={args.vocab_size} dtype={args.dtype} steps={args.steps}", flush=True)
    print(f"# pass bar: grad_l2_rel <= floor * {args.tolerance}  AND  "
          f"|slope| <= {args.slope_max:g}  AND  finite\n", flush=True)

    # 1) Floor.
    print("running floor (pytorch bf16 vs fp32) ...", flush=True)
    floor_report = run_drift("pytorch", None, shape)
    floor = summarize(floor_report)
    bar = floor["grad_l2_rel"] * args.tolerance

    rows = [("pytorch[floor]", floor, True, [])]

    # 2) Candidates.
    for spec in [c.strip() for c in args.candidates.split(",") if c.strip()]:
        impl, mode = parse_candidate(spec)
        label = spec
        print(f"running candidate {label} ...", flush=True)
        try:
            rep = run_drift(impl, mode, shape)
        except RuntimeError as exc:
            rows.append((label, {"error": str(exc)}, False, ["run failed"]))
            continue
        s = summarize(rep)
        reasons = []
        if not s["finite"]:
            reasons.append("non-finite")
        if s["grad_l2_rel"] > bar:
            reasons.append(f"grad_l2_rel {s['grad_l2_rel']:.3g} > bar {bar:.3g}")
        worst_slope = max(abs(s["proj_grad_slope"] or 0.0), abs(s["head_grad_slope"] or 0.0))
        if worst_slope > args.slope_max:
            reasons.append(f"slope {worst_slope:.3g} > {args.slope_max:g}")
        rows.append((label, s, not reasons, reasons))

    # 3) Verdict table.
    print(f"\n{'candidate':<26} {'grad_l2_rel':>12} {'x_floor':>8} {'loss_abs':>11} "
          f"{'max_slope':>11} {'verdict':>8}")
    print("-" * 80)
    floor_grad = floor["grad_l2_rel"]
    for label, s, passed, reasons in rows:
        if "error" in s:
            print(f"{label:<26} {'ERROR':>12} {'':>8} {'':>11} {'':>11} {'FAIL':>8}")
            print(f"    {s['error'].splitlines()[0]}")
            continue
        ratio = s["grad_l2_rel"] / floor_grad if floor_grad > 0 else float("inf")
        slope = max(abs(s["proj_grad_slope"] or 0.0), abs(s["head_grad_slope"] or 0.0))
        verdict = "PASS" if passed else "FAIL"
        print(f"{label:<26} {s['grad_l2_rel']:>12.4g} {ratio:>8.2f} {s['loss_abs']:>11.4g} "
              f"{slope:>11.3g} {verdict:>8}")
        if reasons:
            print(f"    -> {'; '.join(reasons)}")

    print(f"\nfloor grad_l2_rel = {floor_grad:.4g}   pass bar = {bar:.4g}", flush=True)

    any_fail = any(not passed for label, s, passed, reasons in rows if label != "pytorch[floor]")
    raise SystemExit(2 if any_fail else 0)


if __name__ == "__main__":
    main()
