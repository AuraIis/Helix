#!/usr/bin/env python3
"""Run the existing Auralis synthetic training benchmark from the perf lab."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model/helix_v2_mid_500m_smart.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-length", type=int, default=2048)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--compile-mode", default="default")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        str(REPO / "scripts" / "debug" / "bench_training_throughput.py"),
        "--model-config",
        args.model_config,
        "--batch-size",
        str(args.batch_size),
        "--seq-length",
        str(args.seq_length),
        "--grad-accum",
        str(args.grad_accum),
        "--warmup",
        str(args.warmup),
        "--iters",
        str(args.iters),
        "--kernels",
        "--fused-optimizer",
    ]
    if args.checkpoint:
        cmd += ["--checkpoint", args.checkpoint]
    if args.compile:
        cmd += ["--compile", "--compile-mode", args.compile_mode]
    raise SystemExit(subprocess.call(cmd, cwd=REPO))


if __name__ == "__main__":
    main()

