#!/usr/bin/env python3
"""Parity + speed test for the SwiGLU silu-mul against Liger's fused kernel.

The Helix v2 FFN is silu-gated (SwiGLU): the elementwise core is
``silu(gate) * up`` over ``[tokens, d_ffn]`` (d_ffn=3584), run once per the 28
dense FFN layers. Liger fuses this silu-mul (and its backward) into one Triton
kernel. The GEMMs (gate/up/down projections) are NOT part of this op — this
measures only the fusable elementwise gate.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def torch_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def liger_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    import torch.distributed.tensor  # noqa: F401  Liger references DTensor
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    return LigerSiLUMulFunction.apply(gate, up)


def timed(fn, *, warmup: int, iters: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        fn().sum().backward()
    torch.cuda.synchronize()
    times, peaks = [], []
    for _ in range(iters):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        fn().sum().backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        peaks.append(torch.cuda.max_memory_allocated() / 1e9)
    return statistics.mean(times), min(times), statistics.mean(peaks)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=8192, help="tokens")
    p.add_argument("--dim", type=int, default=3584, help="d_ffn")
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--impl", choices=["liger"], default="liger")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    torch.manual_seed(1234)
    gate_base = torch.randn(args.rows, args.dim, device="cuda", dtype=dtype)
    up_base = torch.randn(args.rows, args.dim, device="cuda", dtype=dtype)

    def make():
        return (gate_base.detach().clone().requires_grad_(True),
                up_base.detach().clone().requires_grad_(True))

    candidate = liger_silu_mul

    g1, u1 = make()
    y_ref = torch_silu_mul(g1, u1)
    y_ref.sum().backward()
    g2, u2 = make()
    y_new = candidate(g2, u2)
    y_new.sum().backward()
    torch.cuda.synchronize()

    y_err = (y_ref.float() - y_new.float()).abs().max().item()
    gg_err = (g1.grad.float() - g2.grad.float()).abs().max().item()
    gu_err = (u1.grad.float() - u2.grad.float()).abs().max().item()

    def run_torch():
        g, u = make()
        return torch_silu_mul(g, u)

    def run_cand():
        g, u = make()
        return candidate(g, u)

    t_avg, t_min, t_peak = timed(run_torch, warmup=args.warmup, iters=args.iters)
    c_avg, c_min, c_peak = timed(run_cand, warmup=args.warmup, iters=args.iters)

    print(json.dumps({
        "gpu": torch.cuda.get_device_name(0),
        "op": "swiglu_silumul",
        "impl": args.impl,
        "rows": args.rows,
        "dim": args.dim,
        "dtype": args.dtype,
        "max_abs_y": y_err,
        "max_abs_grad_gate": gg_err,
        "max_abs_grad_up": gu_err,
        "torch_seconds_avg": t_avg,
        "candidate_seconds_avg": c_avg,
        "speedup": t_avg / c_avg if c_avg else None,
        "torch_seconds_min": t_min,
        "candidate_seconds_min": c_min,
        "torch_peak_gb": t_peak,
        "candidate_peak_gb": c_peak,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
