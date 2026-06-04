#!/usr/bin/env python3
"""Forward-only benchmark for the Triton huge-vocab CE prototype."""

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
sys.path.insert(0, str(REPO / "src"))

from perf_lab.auralis_perf_kernels import triton_linear_ce_forward  # noqa: E402


def full_forward(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(F.linear(hidden, weight), labels, ignore_index=-100, reduction="none")


def time_forward(fn, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        out = fn()
        torch.cuda.synchronize()
        _ = out.detach()
    times = []
    peaks = []
    for _ in range(iters):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        peaks.append(torch.cuda.max_memory_allocated() / 1e9)
        _ = out.detach()
    return {
        "seconds_avg": statistics.mean(times),
        "seconds_min": min(times),
        "peak_alloc_gb_avg": statistics.mean(peaks),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--hidden-scale", type=float, default=1.0)
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--ignore-fraction", type=float, default=0.0)
    parser.add_argument("--block-v", type=int, default=64)
    parser.add_argument("--block-d", type=int, default=64)
    parser.add_argument("--block-m", type=int, default=8)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--num-stages", type=int, default=3)
    parser.add_argument("--mode", choices=["block", "parallel", "row"], default="block")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(1234)
    hidden = torch.randn(args.tokens, args.d_model, device="cuda", dtype=dtype) * args.hidden_scale
    weight = torch.randn(args.vocab_size, args.d_model, device="cuda", dtype=dtype) * args.weight_scale
    labels = torch.randint(0, args.vocab_size, (args.tokens,), device="cuda", dtype=torch.long)
    if args.ignore_fraction:
        if not 0.0 <= args.ignore_fraction < 1.0:
            raise ValueError("--ignore-fraction must be in [0, 1)")
        labels = labels.clone()
        labels[torch.rand(args.tokens, device="cuda") < args.ignore_fraction] = -100

    full_losses = full_forward(hidden, weight, labels).float()
    triton_losses, row_max, exp_sum = triton_linear_ce_forward(
        hidden,
        weight,
        labels,
        block_v=args.block_v,
        block_d=args.block_d,
        block_m=args.block_m,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
        mode=args.mode,
    )
    torch.cuda.synchronize()

    full = time_forward(lambda: full_forward(hidden, weight, labels), args.warmup, args.iters)
    triton = time_forward(
        lambda: triton_linear_ce_forward(
            hidden,
            weight,
            labels,
            block_v=args.block_v,
            block_d=args.block_d,
            block_m=args.block_m,
            num_warps=args.num_warps,
            num_stages=args.num_stages,
            mode=args.mode,
        )[0],
        args.warmup,
        args.iters,
    )
    print(json.dumps({
        "gpu": torch.cuda.get_device_name(0),
        "tokens": args.tokens,
        "d_model": args.d_model,
        "vocab_size": args.vocab_size,
        "dtype": args.dtype,
        "hidden_scale": args.hidden_scale,
        "weight_scale": args.weight_scale,
        "ignore_fraction": args.ignore_fraction,
        "block_v": args.block_v,
        "block_d": args.block_d,
        "block_m": args.block_m,
        "num_stages": args.num_stages,
        "mode": args.mode,
        "loss_max_abs": float((full_losses - triton_losses).abs().max().item()),
        "loss_mean_abs": float((full_losses - triton_losses).abs().mean().item()),
        "row_max_min": float(row_max.min().item()),
        "exp_sum_min": float(exp_sum.min().item()),
        "full": full,
        "triton": triton,
        "speed_ratio_full_over_triton": full["seconds_avg"] / triton["seconds_avg"],
        "memory_delta_gb": full["peak_alloc_gb_avg"] - triton["peak_alloc_gb_avg"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
