#!/usr/bin/env python3
"""Sweep chunk sizes for huge-vocab chunked CE."""

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

from perf_lab.auralis_perf_kernels import (  # noqa: E402
    chunked_linear_cross_entropy,
    triton_forward_linear_cross_entropy,
    triton_fused_linear_cross_entropy,
)


def run_step(fn, hidden, weight, labels):
    loss = fn(hidden, weight, labels)
    loss.backward()
    torch.cuda.synchronize()
    hidden.grad = None
    weight.grad = None
    return float(loss.detach().item())


def bench(fn, hidden, weight, labels, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        run_step(fn, hidden, weight, labels)

    times = []
    peaks = []
    loss = None
    for _ in range(iters):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        loss = run_step(fn, hidden, weight, labels)
        times.append(time.perf_counter() - t0)
        peaks.append(torch.cuda.max_memory_allocated() / 1e9)
    return {
        "seconds_avg": statistics.mean(times),
        "seconds_min": min(times),
        "peak_alloc_gb_avg": statistics.mean(peaks),
        "loss_last": loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=2048)
    parser.add_argument("--d-model", type=int, default=1280)
    parser.add_argument("--vocab-size", type=int, default=200000)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--hidden-scale", type=float, default=1.0)
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--chunk-sizes", default="4096,8192,16384,32768,65536")
    parser.add_argument("--impl", choices=["auto", "cpp", "python", "triton", "triton_fused"], default="cpp")
    parser.add_argument("--block-m", type=int, default=8)
    parser.add_argument("--block-v", type=int, default=256)
    parser.add_argument("--block-d", type=int, default=64)
    parser.add_argument(
        "--triton-backward-mode",
        choices=[
            "atomic",
            "atomic_lowmem",
            "atomic_mixed",
            "atomic_bf16match",
            "atomic_lowmem_bf16match",
            "atomic_mixed_bf16match",
            "partial_weight",
            "split_hidden",
            "split_weight",
        ],
        default="atomic",
    )
    parser.add_argument("--row-group-blocks", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--include-full", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    chunks = [int(x) for x in args.chunk_sizes.split(",") if x.strip()]
    torch.manual_seed(1234)
    hidden = (
        torch.randn(args.tokens, args.d_model, device="cuda", dtype=dtype) * args.hidden_scale
    ).requires_grad_(True)
    weight = (
        torch.randn(args.vocab_size, args.d_model, device="cuda", dtype=dtype) * args.weight_scale
    ).requires_grad_(True)
    labels = torch.randint(0, args.vocab_size, (args.tokens,), device="cuda", dtype=torch.long)

    out = {
        "gpu": torch.cuda.get_device_name(0),
        "tokens": args.tokens,
        "d_model": args.d_model,
        "vocab_size": args.vocab_size,
        "dtype": args.dtype,
        "hidden_scale": args.hidden_scale,
        "weight_scale": args.weight_scale,
        "impl": args.impl,
        "block_m": max(args.block_m, 16) if args.impl == "triton_fused" else args.block_m,
        "block_v": max(args.block_v, 16) if args.impl == "triton_fused" else args.block_v,
        "block_d": max(args.block_d, 16) if args.impl == "triton_fused" else args.block_d,
        "triton_backward_mode": args.triton_backward_mode if args.impl == "triton_fused" else None,
        "row_group_blocks": args.row_group_blocks if args.impl == "triton_fused" else None,
        "full_logits_gb": args.tokens * args.vocab_size * torch.empty((), dtype=dtype).element_size() / 1e9,
        "results": [],
    }

    full_result = None
    if args.include_full:
        full_result = bench(
            lambda h, w, y: F.cross_entropy(F.linear(h, w).view(-1, w.size(0)), y.view(-1), ignore_index=-100),
            hidden,
            weight,
            labels,
            args.warmup,
            args.iters,
        )
        out["full"] = full_result

    for chunk_size in chunks:
        if args.impl == "triton":
            fn = lambda h, w, y, cs=chunk_size: triton_forward_linear_cross_entropy(
                h,
                w,
                y,
                chunk_size=cs,
                block_m=args.block_m,
                block_v=args.block_v,
                block_d=args.block_d,
            )
        elif args.impl == "triton_fused":
            fn = lambda h, w, y, cs=chunk_size: triton_fused_linear_cross_entropy(
                h,
                w,
                y,
                block_m=max(args.block_m, 16),
                block_v=max(args.block_v, 16),
                block_d=max(args.block_d, 16),
                backward_mode=args.triton_backward_mode,
                row_group_blocks=args.row_group_blocks,
            )
        else:
            fn = lambda h, w, y, cs=chunk_size: chunked_linear_cross_entropy(
                h,
                w,
                y,
                chunk_size=cs,
                impl=args.impl,
            )
        result = bench(
            fn,
            hidden,
            weight,
            labels,
            args.warmup,
            args.iters,
        )
        result["chunk_size"] = chunk_size
        if full_result:
            result["speed_ratio_full_over_chunked"] = full_result["seconds_avg"] / result["seconds_avg"]
            result["memory_saved_gb"] = full_result["peak_alloc_gb_avg"] - result["peak_alloc_gb_avg"]
        out["results"].append(result)

    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
