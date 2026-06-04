#!/usr/bin/env python3
"""Loss-drift test for huge-vocab CE candidates.

This is a promotion gate, not a speed benchmark. It runs two identical tiny
training loops:

- reference path: `F.linear(hidden, weight)` + PyTorch cross entropy
- candidate path: custom/chunked/fused linear cross entropy

Both loops see the same synthetic batches and optimizer hyperparameters. The
script tracks loss, gradient, and parameter divergence over multiple optimizer
steps so precision shortcuts cannot pass by only matching a single backward.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from perf_lab.auralis_perf_kernels import (  # noqa: E402
    chunked_linear_cross_entropy,
    liger_linear_cross_entropy,
    triton_forward_linear_cross_entropy,
    triton_fused_linear_cross_entropy,
)


@dataclass
class TensorPair:
    ref: torch.Tensor
    cand: torch.Tensor


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def make_optimizer(name: str, params: list[torch.Tensor], lr: float, weight_decay: float) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name}")


def make_batch(
    *,
    step: int,
    seed: int,
    tokens: int,
    input_dim: int,
    vocab_size: int,
    dtype: torch.dtype,
    device: torch.device,
    input_scale: float,
    ignore_frac: float,
    resample: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_seed = seed + step if resample else seed
    gen = torch.Generator(device=device).manual_seed(batch_seed)
    x = torch.randn((tokens, input_dim), device=device, dtype=dtype, generator=gen) * input_scale
    labels = torch.randint(0, vocab_size, (tokens,), device=device, dtype=torch.long, generator=gen)
    if ignore_frac > 0:
        mask = torch.rand((tokens,), device=device, generator=gen) < ignore_frac
        labels = torch.where(mask, torch.full_like(labels, -100), labels)
        if bool(labels.ne(-100).sum().item() == 0):
            labels[0] = 0
    return x, labels


def reference_ce(hidden: torch.Tensor, weight: torch.Tensor, labels: torch.Tensor,
                 fp32: bool = False) -> torch.Tensor:
    if fp32:
        # fp32 "ground-truth" reference: upcast logits + CE to full precision so a
        # candidate can be measured against the true answer. Paired with
        # `--impl pytorch` this separates real kernel error from the inherent
        # bf16 noise floor (PyTorch-bf16-CE vs PyTorch-fp32-CE).
        logits = F.linear(hidden.float(), weight.float())
        return F.cross_entropy(logits, labels, ignore_index=-100)
    logits = F.linear(hidden, weight)
    return F.cross_entropy(logits, labels, ignore_index=-100)


def accumulation_stats(values: list[float]) -> dict[str, float | int | None]:
    """Does a drift series accumulate over steps, or stay flat (just noise)?

    Returns the linear-fit slope per step plus the late/early mean ratio. A
    promotable candidate should have slope ~0 and late_over_early ~1: the drift
    behaves like fixed bf16/reduction-order noise, not a growing systematic bias.
    """
    n = len(values)
    if n < 4:
        return {"n": n, "slope_per_step": None, "early_mean": None,
                "late_mean": None, "late_over_early": None}
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(values) / n
    den = sum((x - mx) ** 2 for x in xs) or 1.0
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, values)) / den
    k = max(1, n // 5)
    early = sum(values[:k]) / k
    late = sum(values[-k:]) / k
    return {"n": n, "slope_per_step": slope, "early_mean": early,
            "late_mean": late, "late_over_early": (late / early if early > 0 else None)}


def make_candidate_loss(args: argparse.Namespace) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    if args.impl == "pytorch":
        # PyTorch full-logits CE in the run dtype. Paired with --reference-fp32
        # this run measures the inherent bf16-vs-fp32 noise floor — the bar a
        # custom kernel must reach to count as "no extra drift".
        return lambda hidden, weight, labels: reference_ce(hidden, weight, labels, fp32=False)
    if args.impl in {"auto", "cpp", "python"}:
        return lambda hidden, weight, labels: chunked_linear_cross_entropy(
            hidden,
            weight,
            labels,
            chunk_size=args.chunk_size,
            impl=args.impl,
        )
    if args.impl == "triton":
        return lambda hidden, weight, labels: triton_forward_linear_cross_entropy(
            hidden,
            weight,
            labels,
            chunk_size=args.chunk_size,
            block_m=args.block_m,
            block_v=args.block_v,
            block_d=args.block_d,
        )
    if args.impl == "triton_fused":
        return lambda hidden, weight, labels: triton_fused_linear_cross_entropy(
            hidden,
            weight,
            labels,
            block_m=args.block_m,
            block_v=args.block_v,
            block_d=args.block_d,
            backward_mode=args.triton_backward_mode,
            row_group_blocks=args.row_group_blocks,
        )
    if args.impl == "liger":
        # Liger FLCE: exact gradients by construction (grad_weight via GEMM, not
        # atomics). accum_dtype=fp32 forces fp32 gradient accumulation for bf16.
        accum = torch.float32 if args.accum_dtype == "fp32" else None
        return lambda hidden, weight, labels: liger_linear_cross_entropy(
            hidden, weight, labels, accum_dtype=accum,
        )
    raise ValueError(f"unknown impl: {args.impl}")


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def max_rel(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    diff = (a.float() - b.float()).abs()
    denom = torch.maximum(a.float().abs(), b.float().abs()).clamp_min(eps)
    return float((diff / denom).max().item())


def l2_rel(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    af = a.float()
    bf = b.float()
    denom = torch.maximum(af.norm(), bf.norm()).clamp_min(eps)
    return float((af - bf).norm().div(denom).item())


def grad_pair_metric(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    if a.grad is None or b.grad is None:
        return math.nan, math.nan, math.nan
    return max_abs(a.grad, b.grad), max_rel(a.grad, b.grad), l2_rel(a.grad, b.grad)


def finite_param_check(params: list[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(param).all().item()) for param in params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--vocab-size", type=int, default=8192)
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--weight-dtype", choices=["same", "fp32"], default="same")
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--proj-scale", type=float, default=0.05)
    parser.add_argument("--weight-scale", type=float, default=0.05)
    parser.add_argument("--ignore-frac", type=float, default=0.0)
    parser.add_argument("--fixed-batch", action="store_true")
    parser.add_argument("--impl", choices=["pytorch", "auto", "cpp", "python", "triton", "triton_fused", "liger"], default="triton_fused")
    parser.add_argument("--accum-dtype", choices=["none", "fp32"], default="fp32",
                        help="Liger gradient accumulation dtype (--impl liger only). "
                             "fp32 forces fp32 accumulation for bf16 inputs (recommended).")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--block-m", type=int, default=32)
    parser.add_argument("--block-v", type=int, default=32)
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
        default="atomic_lowmem",
    )
    parser.add_argument("--row-group-blocks", type=int, default=8)
    parser.add_argument("--history-every", type=int, default=1)
    parser.add_argument("--fail-loss-abs", type=float, default=None)
    parser.add_argument("--fail-param-rel", type=float, default=None)
    parser.add_argument("--fail-param-l2-rel", type=float, default=None)
    parser.add_argument("--fail-grad-l2-rel", type=float, default=None,
                        help="Fail if max upstream-gradient (proj/head) L2-rel drift exceeds this. "
                             "This is the metric that actually decides promotability.")
    parser.add_argument("--reference-fp32", action="store_true",
                        help="Compute the reference loss/grad in fp32 (ground truth) instead of the "
                             "run dtype. With --impl pytorch this measures the bf16 noise floor.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if args.steps <= 0:
        raise SystemExit("--steps must be > 0")
    if not 0 <= args.ignore_frac < 1:
        raise SystemExit("--ignore-frac must be in [0, 1)")

    device = torch.device("cuda")
    dtype = dtype_from_name(args.dtype)
    weight_dtype = torch.float32 if args.weight_dtype == "fp32" else dtype
    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision("high")

    gen = torch.Generator(device=device).manual_seed(args.seed)
    proj0 = (torch.randn((args.input_dim, args.d_model), device=device, dtype=dtype, generator=gen) * args.proj_scale)
    weight0 = (
        torch.randn((args.vocab_size, args.d_model), device=device, dtype=weight_dtype, generator=gen)
        * args.weight_scale
    )

    proj = TensorPair(
        ref=proj0.detach().clone().requires_grad_(True),
        cand=proj0.detach().clone().requires_grad_(True),
    )
    head = TensorPair(
        ref=weight0.detach().clone().requires_grad_(True),
        cand=weight0.detach().clone().requires_grad_(True),
    )

    opt_ref = make_optimizer(args.optimizer, [proj.ref, head.ref], args.lr, args.weight_decay)
    opt_cand = make_optimizer(args.optimizer, [proj.cand, head.cand], args.lr, args.weight_decay)
    candidate_ce = make_candidate_loss(args)

    history: list[dict[str, float | int | bool]] = []
    max_metrics = {
        "loss_abs": 0.0,
        "proj_param_max_abs": 0.0,
        "proj_param_max_rel": 0.0,
        "proj_param_l2_rel": 0.0,
        "head_param_max_abs": 0.0,
        "head_param_max_rel": 0.0,
        "head_param_l2_rel": 0.0,
        "proj_grad_max_abs": 0.0,
        "proj_grad_max_rel": 0.0,
        "proj_grad_l2_rel": 0.0,
        "head_grad_max_abs": 0.0,
        "head_grad_max_rel": 0.0,
        "head_grad_l2_rel": 0.0,
    }

    final_step: dict[str, float | int | bool] = {}
    # Per-step drift series (every step, independent of --history-every) so we can
    # tell accumulating bias from flat noise.
    series: dict[str, list[float]] = {
        "proj_grad_l2_rel": [], "head_grad_l2_rel": [], "loss_abs": [],
    }
    for step_idx in range(1, args.steps + 1):
        x, labels = make_batch(
            step=step_idx,
            seed=args.seed + 10_000,
            tokens=args.tokens,
            input_dim=args.input_dim,
            vocab_size=args.vocab_size,
            dtype=dtype,
            device=device,
            input_scale=args.input_scale,
            ignore_frac=args.ignore_frac,
            resample=not args.fixed_batch,
        )

        opt_ref.zero_grad(set_to_none=True)
        opt_cand.zero_grad(set_to_none=True)

        hidden_ref = x.matmul(proj.ref)
        hidden_cand = x.matmul(proj.cand)
        loss_ref = reference_ce(hidden_ref, head.ref, labels, fp32=args.reference_fp32)
        loss_cand = candidate_ce(hidden_cand, head.cand, labels)
        loss_ref.backward()
        loss_cand.backward()
        torch.cuda.synchronize()

        proj_grad_abs, proj_grad_rel, proj_grad_l2_rel = grad_pair_metric(proj.ref, proj.cand)
        head_grad_abs, head_grad_rel, head_grad_l2_rel = grad_pair_metric(head.ref, head.cand)
        loss_abs = abs(float(loss_ref.detach().item()) - float(loss_cand.detach().item()))

        opt_ref.step()
        opt_cand.step()
        torch.cuda.synchronize()

        current = {
            "step": step_idx,
            "loss_ref": float(loss_ref.detach().item()),
            "loss_candidate": float(loss_cand.detach().item()),
            "loss_abs": loss_abs,
            "proj_grad_max_abs": proj_grad_abs,
            "proj_grad_max_rel": proj_grad_rel,
            "proj_grad_l2_rel": proj_grad_l2_rel,
            "head_grad_max_abs": head_grad_abs,
            "head_grad_max_rel": head_grad_rel,
            "head_grad_l2_rel": head_grad_l2_rel,
            "proj_param_max_abs": max_abs(proj.ref, proj.cand),
            "proj_param_max_rel": max_rel(proj.ref, proj.cand),
            "proj_param_l2_rel": l2_rel(proj.ref, proj.cand),
            "head_param_max_abs": max_abs(head.ref, head.cand),
            "head_param_max_rel": max_rel(head.ref, head.cand),
            "head_param_l2_rel": l2_rel(head.ref, head.cand),
            "finite": finite_param_check([proj.ref, proj.cand, head.ref, head.cand]),
        }
        final_step = current
        for key in max_metrics:
            max_metrics[key] = max(max_metrics[key], float(current[key]))
        for key in series:
            series[key].append(float(current[key]))
        if args.history_every > 0 and (step_idx == 1 or step_idx == args.steps or step_idx % args.history_every == 0):
            history.append(current)
        if not current["finite"]:
            break

    result = {
        "gpu": torch.cuda.get_device_name(0),
        "steps_requested": args.steps,
        "steps_completed": int(final_step.get("step", 0)),
        "tokens": args.tokens,
        "input_dim": args.input_dim,
        "d_model": args.d_model,
        "vocab_size": args.vocab_size,
        "dtype": args.dtype,
        "weight_dtype": args.weight_dtype,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "impl": args.impl,
        "chunk_size": args.chunk_size,
        "block_m": max(args.block_m, 16) if args.impl == "triton_fused" else args.block_m,
        "block_v": max(args.block_v, 16) if args.impl == "triton_fused" else args.block_v,
        "block_d": max(args.block_d, 16) if args.impl == "triton_fused" else args.block_d,
        "triton_backward_mode": args.triton_backward_mode if args.impl == "triton_fused" else None,
        "row_group_blocks": args.row_group_blocks if args.impl == "triton_fused" else None,
        "accum_dtype": args.accum_dtype if args.impl == "liger" else None,
        "fixed_batch": bool(args.fixed_batch),
        "ignore_frac": args.ignore_frac,
        "max_metrics": max_metrics,
        "accumulation": {k: accumulation_stats(v) for k, v in series.items()},
        "final_step": final_step,
        "history": history,
    }

    failed = False
    failure_reasons: list[str] = []
    if not bool(final_step.get("finite", False)):
        failed = True
        failure_reasons.append("non-finite parameter detected")
    if args.fail_loss_abs is not None and max_metrics["loss_abs"] > args.fail_loss_abs:
        failed = True
        failure_reasons.append(f"loss_abs {max_metrics['loss_abs']:.6g} > {args.fail_loss_abs:.6g}")
    max_param_rel = max(max_metrics["proj_param_max_rel"], max_metrics["head_param_max_rel"])
    if args.fail_param_rel is not None and max_param_rel > args.fail_param_rel:
        failed = True
        failure_reasons.append(f"param_rel {max_param_rel:.6g} > {args.fail_param_rel:.6g}")
    max_param_l2_rel = max(max_metrics["proj_param_l2_rel"], max_metrics["head_param_l2_rel"])
    if args.fail_param_l2_rel is not None and max_param_l2_rel > args.fail_param_l2_rel:
        failed = True
        failure_reasons.append(f"param_l2_rel {max_param_l2_rel:.6g} > {args.fail_param_l2_rel:.6g}")
    max_grad_l2_rel = max(max_metrics["proj_grad_l2_rel"], max_metrics["head_grad_l2_rel"])
    if args.fail_grad_l2_rel is not None and max_grad_l2_rel > args.fail_grad_l2_rel:
        failed = True
        failure_reasons.append(f"grad_l2_rel {max_grad_l2_rel:.6g} > {args.fail_grad_l2_rel:.6g}")
    result["failed"] = failed
    result["failure_reasons"] = failure_reasons

    print(json.dumps(result, indent=2), flush=True)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
