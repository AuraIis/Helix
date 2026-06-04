#!/usr/bin/env python3
"""Parity and speed test for the experimental RMSNorm CUDA kernel."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from perf_lab.auralis_perf_kernels import rmsnorm  # noqa: E402


def torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps).to(x.dtype) * weight.to(dtype=x.dtype)


def liger_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Liger Triton RMSNorm in 'llama' casting mode (fp32 compute, weight applied
    directly) to match `torch_rmsnorm`. Liger's op references DTensor, so the
    submodule must be imported or `torch.distributed.tensor` raises AttributeError
    on this torch build."""
    import torch.distributed.tensor  # noqa: F401
    from liger_kernel.transformers.functional import liger_rms_norm
    return liger_rms_norm(x, weight, eps, offset=0.0, casting_mode="llama", in_place=False)


def get_candidate(impl: str):
    if impl == "liger":
        return liger_rmsnorm
    return rmsnorm


def timed(fn, *, warmup: int, iters: int) -> tuple[float, list[float]]:
    for _ in range(warmup):
        y = fn()
        if isinstance(y, tuple):
            y = y[0]
        y.sum().backward()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        y = fn()
        if isinstance(y, tuple):
            y = y[0]
        y.sum().backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.mean(times), times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=1280)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--impl", choices=["custom", "liger"], default="custom",
                        help="Candidate kernel compared against the PyTorch reference: "
                             "'custom' = hand-rolled CUDA kernel, 'liger' = Liger Triton RMSNorm.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    torch.manual_seed(1234)
    x_base = torch.randn(args.rows, args.dim, device="cuda", dtype=dtype)
    weight_base = torch.randn(args.dim, device="cuda", dtype=torch.float32)

    def make_inputs():
        return (
            x_base.detach().clone().requires_grad_(True),
            weight_base.detach().clone().requires_grad_(True),
        )

    candidate = get_candidate(args.impl)

    x_ref, w_ref = make_inputs()
    y_ref = torch_rmsnorm(x_ref, w_ref, args.eps)
    y_ref.sum().backward()

    x_new, w_new = make_inputs()
    y_new = candidate(x_new, w_new, args.eps)
    y_new.sum().backward()
    torch.cuda.synchronize()

    y_err = (y_ref.float() - y_new.float()).abs().max().item()
    gx_err = (x_ref.grad.float() - x_new.grad.float()).abs().max().item()
    gw_err = (w_ref.grad.float() - w_new.grad.float()).abs().max().item()

    def run_torch():
        x, w = make_inputs()
        return torch_rmsnorm(x, w, args.eps)

    def run_custom():
        x, w = make_inputs()
        return candidate(x, w, args.eps)

    torch_avg, torch_times = timed(run_torch, warmup=args.warmup, iters=args.iters)
    custom_avg, custom_times = timed(run_custom, warmup=args.warmup, iters=args.iters)

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "impl": args.impl,
        "rows": args.rows,
        "dim": args.dim,
        "dtype": args.dtype,
        "max_abs_y": y_err,
        "max_abs_grad_x": gx_err,
        "max_abs_grad_weight": gw_err,
        "torch_seconds_avg": torch_avg,
        "custom_seconds_avg": custom_avg,
        "speedup": torch_avg / custom_avg if custom_avg else None,
        "torch_seconds_min": min(torch_times),
        "custom_seconds_min": min(custom_times),
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()

