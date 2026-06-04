#!/usr/bin/env python3
"""Sweep Triton fused linear-CE tiles and backward modes."""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from perf_lab.auralis_perf_kernels import triton_fused_linear_cross_entropy  # noqa: E402


def parse_csv_ints(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def full_step(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(F.linear(hidden, weight), labels, ignore_index=-100)


def fused_step(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    block_m: int,
    block_v: int,
    block_d: int,
    backward_mode: str,
    row_group_blocks: int,
) -> torch.Tensor:
    return triton_fused_linear_cross_entropy(
        hidden,
        weight,
        labels,
        block_m=block_m,
        block_v=block_v,
        block_d=block_d,
        backward_mode=backward_mode,
        row_group_blocks=row_group_blocks,
    )


def clear_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def max_abs_or_none(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    return float((a.float() - b.float()).abs().max().item())


def max_rel_or_none(a: torch.Tensor | None, b: torch.Tensor | None) -> float | None:
    if a is None or b is None:
        return None
    denom = b.float().abs().max().clamp_min(1e-12)
    return float(((a.float() - b.float()).abs().max() / denom).item())


def parity(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    block_m: int,
    block_v: int,
    block_d: int,
    backward_mode: str,
    row_group_blocks: int,
) -> dict:
    h_ref = hidden.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    h_fused = hidden.detach().clone().requires_grad_(True)
    w_fused = weight.detach().clone().requires_grad_(True)
    ref_loss = full_step(h_ref, w_ref, labels)
    fused_loss = fused_step(
        h_fused,
        w_fused,
        labels,
        block_m,
        block_v,
        block_d,
        backward_mode,
        row_group_blocks,
    )
    ref_loss.backward()
    fused_loss.backward()
    torch.cuda.synchronize()
    out = {
        "full_loss": float(ref_loss.detach().item()),
        "fused_loss": float(fused_loss.detach().item()),
        "loss_abs": abs(float(ref_loss.detach().item()) - float(fused_loss.detach().item())),
        "grad_hidden_max_abs": max_abs_or_none(h_fused.grad, h_ref.grad),
        "grad_hidden_max_rel": max_rel_or_none(h_fused.grad, h_ref.grad),
        "grad_weight_max_abs": max_abs_or_none(w_fused.grad, w_ref.grad),
        "grad_weight_max_rel": max_rel_or_none(w_fused.grad, w_ref.grad),
    }
    del h_ref, w_ref, h_fused, w_fused
    torch.cuda.empty_cache()
    return out


def bench(fn, hidden, weight, labels, warmup: int, iters: int) -> dict:
    for _ in range(warmup):
        loss = fn(hidden, weight, labels)
        loss.backward()
        torch.cuda.synchronize()
        clear_grads(hidden, weight)

    times = []
    peaks = []
    losses = []
    for _ in range(iters):
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        loss = fn(hidden, weight, labels)
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        peaks.append(torch.cuda.max_memory_allocated() / 1e9)
        losses.append(float(loss.detach().item()))
        clear_grads(hidden, weight)
    return {
        "seconds_avg": statistics.mean(times),
        "seconds_min": min(times),
        "peak_alloc_gb_avg": statistics.mean(peaks),
        "loss_last": losses[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--hidden-scale", type=float, default=1.0)
    parser.add_argument("--weight-scale", type=float, default=1.0)
    parser.add_argument("--block-ms", default="16,32")
    parser.add_argument("--block-vs", default="32,64,128")
    parser.add_argument("--block-ds", default="32,64")
    parser.add_argument(
        "--backward-modes",
        default=(
            "atomic,atomic_lowmem,atomic_mixed,"
            "atomic_bf16match,atomic_lowmem_bf16match,atomic_mixed_bf16match,"
            "partial_weight,split_hidden,split_weight"
        ),
    )
    parser.add_argument("--row-group-blocks", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=2)
    parser.add_argument("--include-full", action="store_true")
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--max-configs", type=int, default=0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    torch.manual_seed(1234)
    hidden = (
        torch.randn(args.tokens, args.d_model, device="cuda", dtype=dtype) * args.hidden_scale
    ).requires_grad_(True)
    weight = (
        torch.randn(args.vocab_size, args.d_model, device="cuda", dtype=dtype) * args.weight_scale
    ).requires_grad_(True)
    labels = torch.randint(0, args.vocab_size, (args.tokens,), device="cuda", dtype=torch.long)

    configs = list(
        itertools.product(
            parse_csv_strings(args.backward_modes),
            parse_csv_ints(args.block_ms),
            parse_csv_ints(args.block_vs),
            parse_csv_ints(args.block_ds),
        )
    )
    if args.max_configs > 0:
        configs = configs[: args.max_configs]

    out = {
        "gpu": torch.cuda.get_device_name(0),
        "tokens": args.tokens,
        "d_model": args.d_model,
        "vocab_size": args.vocab_size,
        "dtype": args.dtype,
        "hidden_scale": args.hidden_scale,
        "weight_scale": args.weight_scale,
        "full_logits_gb": args.tokens * args.vocab_size * torch.empty((), dtype=dtype).element_size() / 1e9,
        "results": [],
    }

    full_result = None
    if args.include_full:
        full_result = bench(full_step, hidden, weight, labels, args.warmup, args.iters)
        out["full"] = full_result

    for backward_mode, block_m, block_v, block_d in configs:
        item = {
            "backward_mode": backward_mode,
            "block_m": max(block_m, 16),
            "block_v": max(block_v, 16),
            "block_d": max(block_d, 16),
            "row_group_blocks": int(args.row_group_blocks) if backward_mode == "partial_weight" else None,
        }
        try:
            if not args.skip_parity:
                item["parity"] = parity(
                    hidden,
                    weight,
                    labels,
                    item["block_m"],
                    item["block_v"],
                    item["block_d"],
                    backward_mode,
                    int(args.row_group_blocks),
                )
            item["bench"] = bench(
                lambda h,
                w,
                y,
                bm=item["block_m"],
                bv=item["block_v"],
                bd=item["block_d"],
                mode=backward_mode,
                rgb=int(args.row_group_blocks): fused_step(
                    h,
                    w,
                    y,
                    bm,
                    bv,
                    bd,
                    mode,
                    rgb,
                ),
                hidden,
                weight,
                labels,
                args.warmup,
                args.iters,
            )
            if full_result:
                item["full_vs_fused"] = {
                    "speed_ratio_full_over_fused": full_result["seconds_avg"] / item["bench"]["seconds_avg"],
                    "peak_memory_delta_gb": full_result["peak_alloc_gb_avg"] - item["bench"]["peak_alloc_gb_avg"],
                }
        except Exception as exc:  # keep sweeps alive across failed tile shapes
            item["error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-8:],
            }
            torch.cuda.empty_cache()
            clear_grads(hidden, weight)
        out["results"].append(item)

    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
