#!/usr/bin/env python3
"""Benchmark huge-vocab projection plus cross-entropy variants.

The chunked variant computes exact CE over vocab chunks and avoids materializing
the full `[tokens, vocab_size]` logits tensor.
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
sys.path.insert(0, str(REPO / "src"))

from perf_lab.auralis_perf_kernels import (
    chunked_linear_cross_entropy,
    liger_linear_cross_entropy,
    triton_forward_linear_cross_entropy,
    triton_fused_linear_cross_entropy,
)


def step(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    logits = F.linear(hidden, weight)
    return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)


def chunked_step(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int,
    impl: str,
    block_m: int,
    block_v: int,
    block_d: int,
    triton_backward_mode: str,
    row_group_blocks: int,
) -> torch.Tensor:
    if impl == "triton":
        return triton_forward_linear_cross_entropy(
            hidden,
            weight,
            labels,
            chunk_size=chunk_size,
            block_m=block_m,
            block_v=block_v,
            block_d=block_d,
        )
    if impl == "triton_fused":
        return triton_fused_linear_cross_entropy(
            hidden,
            weight,
            labels,
            block_m=max(block_m, 16),
            block_v=max(block_v, 16),
            block_d=max(block_d, 16),
            backward_mode=triton_backward_mode,
            row_group_blocks=row_group_blocks,
        )
    if impl == "liger":
        return liger_linear_cross_entropy(hidden, weight, labels, accum_dtype=torch.float32)
    return chunked_linear_cross_entropy(hidden, weight, labels, chunk_size=chunk_size, impl=impl)


def max_abs_or_none(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    return float((a.float() - b.float()).abs().max().item())


def bench(fn, hidden, weight, labels, warmup: int, iters: int):
    for _ in range(warmup):
        loss = fn(hidden, weight, labels)
        loss.backward()
        hidden.grad = None
        weight.grad = None
    torch.cuda.synchronize()

    times: list[float] = []
    peaks: list[float] = []
    losses: list[float] = []
    for _ in range(iters):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        loss = fn(hidden, weight, labels)
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        peaks.append(torch.cuda.max_memory_allocated() / 1e9)
        losses.append(float(loss.detach().item()))
        hidden.grad = None
        weight.grad = None
    return {
        "seconds_avg": statistics.mean(times),
        "seconds_min": min(times),
        "peak_alloc_gb_avg": statistics.mean(peaks),
        "loss_last": losses[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--d-model", type=int, default=1280)
    parser.add_argument("--vocab-size", type=int, default=200000)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--weight-dtype", choices=["same", "fp32"], default="same")
    parser.add_argument("--hidden-scale", type=float, default=1.0)
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--chunked-impl", choices=["auto", "cpp", "python", "triton", "triton_fused", "liger"], default="auto")
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
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--variant", choices=["full", "chunked", "both"], default="both")
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    weight_dtype = torch.float32 if args.weight_dtype == "fp32" else dtype
    torch.manual_seed(1234)
    hidden = (
        torch.randn(args.tokens, args.d_model, device="cuda", dtype=dtype) * args.hidden_scale
    ).requires_grad_(True)
    weight = (
        torch.randn(args.vocab_size, args.d_model, device="cuda", dtype=weight_dtype) * args.weight_scale
    ).requires_grad_(True)
    labels = torch.randint(0, args.vocab_size, (args.tokens,), device="cuda", dtype=torch.long)

    results = {
        "gpu": torch.cuda.get_device_name(0),
        "tokens": args.tokens,
        "d_model": args.d_model,
        "vocab_size": args.vocab_size,
        "dtype": args.dtype,
        "weight_dtype": args.weight_dtype,
        "hidden_scale": args.hidden_scale,
        "weight_scale": args.weight_scale,
        "chunk_size": args.chunk_size,
        "chunked_impl": args.chunked_impl,
        "block_m": max(args.block_m, 16) if args.chunked_impl == "triton_fused" else args.block_m,
        "block_v": max(args.block_v, 16) if args.chunked_impl == "triton_fused" else args.block_v,
        "block_d": max(args.block_d, 16) if args.chunked_impl == "triton_fused" else args.block_d,
        "triton_backward_mode": args.triton_backward_mode if args.chunked_impl == "triton_fused" else None,
        "row_group_blocks": args.row_group_blocks if args.chunked_impl == "triton_fused" else None,
        "full_logits_gb": args.tokens * args.vocab_size * torch.empty((), dtype=dtype).element_size() / 1e9,
    }

    if not args.skip_parity and args.variant in {"chunked", "both"}:
        h1 = hidden.detach().clone().requires_grad_(True)
        w1 = weight.detach().clone().requires_grad_(True)
        h2 = hidden.detach().clone().requires_grad_(True)
        w2 = weight.detach().clone().requires_grad_(True)
        full_loss = step(h1, w1, labels)
        chunk_loss = chunked_step(
            h2,
            w2,
            labels,
            args.chunk_size,
            args.chunked_impl,
            args.block_m,
            args.block_v,
            args.block_d,
            args.triton_backward_mode,
            args.row_group_blocks,
        )
        full_loss.backward()
        chunk_loss.backward()
        torch.cuda.synchronize()
        results["parity"] = {
            "full_loss": float(full_loss.detach().item()),
            "chunked_loss": float(chunk_loss.detach().item()),
            "loss_abs": abs(float(full_loss.detach().item()) - float(chunk_loss.detach().item())),
            "grad_hidden_max_abs": max_abs_or_none(h1.grad, h2.grad),
            "grad_weight_max_abs": max_abs_or_none(w1.grad, w2.grad),
        }
        del h1, w1, h2, w2
        torch.cuda.empty_cache()

    if args.variant in {"full", "both"}:
        results["full"] = bench(step, hidden, weight, labels, args.warmup, args.iters)
    if args.variant in {"chunked", "both"}:
        results["chunked"] = bench(
            lambda h, w, y: chunked_step(
                h,
                w,
                y,
                args.chunk_size,
                args.chunked_impl,
                args.block_m,
                args.block_v,
                args.block_d,
                args.triton_backward_mode,
                args.row_group_blocks,
            ),
            hidden,
            weight,
            labels,
            args.warmup,
            args.iters,
        )
    if "full" in results and "chunked" in results:
        results["chunked_vs_full"] = {
            "speed_ratio_full_over_chunked": results["full"]["seconds_avg"] / results["chunked"]["seconds_avg"],
            "peak_memory_delta_gb": results["full"]["peak_alloc_gb_avg"] - results["chunked"]["peak_alloc_gb_avg"],
        }

    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
