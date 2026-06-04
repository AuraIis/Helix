#!/usr/bin/env python3
"""Parity + speed test for RoPE against Liger's fused kernel.

Helix v2 applies RoPE in its 6 sparse-attention layers (n_heads=10, head_dim=128,
theta=10000). This compares a standard HF-style `apply_rotary_pos_emb`
(rotate_half) against Liger's fused Triton RoPE on `q,k = [B, H, S, D]`.
"""
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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def torch_rope(q, k, cos, sin, unsqueeze_dim: int = 1):
    cos_u = cos.unsqueeze(unsqueeze_dim)
    sin_u = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos_u) + (rotate_half(q) * sin_u)
    k_embed = (k * cos_u) + (rotate_half(k) * sin_u)
    return q_embed, k_embed


def liger_rope_fn(q, k, cos, sin, unsqueeze_dim: int = 1):
    import torch.distributed.tensor  # noqa: F401  Liger references DTensor
    from liger_kernel.transformers.functional import liger_rope
    return liger_rope(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)


def build_cos_sin(seq: int, head_dim: int, theta: float, batch: int, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    pos = torch.arange(seq, device=device).float()
    freqs = torch.outer(pos, inv_freq)             # [S, D/2]
    emb = torch.cat((freqs, freqs), dim=-1)        # [S, D]
    cos = emb.cos().to(dtype).unsqueeze(0).expand(batch, seq, head_dim).contiguous()
    sin = emb.sin().to(dtype).unsqueeze(0).expand(batch, seq, head_dim).contiguous()
    return cos, sin


def timed(fn, *, warmup: int, iters: int) -> tuple[float, float]:
    for _ in range(warmup):
        q_e, k_e = fn()
        (q_e.sum() + k_e.sum()).backward()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        q_e, k_e = fn()
        (q_e.sum() + k_e.sum()).backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.mean(times), min(times)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--heads", type=int, default=10)
    p.add_argument("--seq", type=int, default=2048)
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--theta", type=float, default=10000.0)
    p.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--impl", choices=["liger"], default="liger")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    torch.manual_seed(1234)
    shape = (args.batch, args.heads, args.seq, args.head_dim)
    q_base = torch.randn(*shape, device="cuda", dtype=dtype)
    k_base = torch.randn(*shape, device="cuda", dtype=dtype)
    cos, sin = build_cos_sin(args.seq, args.head_dim, args.theta, args.batch, "cuda", dtype)

    def make():
        return (q_base.detach().clone().requires_grad_(True),
                k_base.detach().clone().requires_grad_(True))

    candidate = liger_rope_fn

    q1, k1 = make()
    qe1, ke1 = torch_rope(q1, k1, cos, sin)
    (qe1.sum() + ke1.sum()).backward()
    q2, k2 = make()
    qe2, ke2 = candidate(q2, k2, cos, sin)
    (qe2.sum() + ke2.sum()).backward()
    torch.cuda.synchronize()

    parity = {
        "max_abs_q": (qe1.float() - qe2.float()).abs().max().item(),
        "max_abs_k": (ke1.float() - ke2.float()).abs().max().item(),
        "max_abs_grad_q": (q1.grad.float() - q2.grad.float()).abs().max().item(),
        "max_abs_grad_k": (k1.grad.float() - k2.grad.float()).abs().max().item(),
    }

    def run_torch():
        q, k = make()
        return torch_rope(q, k, cos, sin)

    def run_cand():
        q, k = make()
        return candidate(q, k, cos, sin)

    t_avg, t_min = timed(run_torch, warmup=args.warmup, iters=args.iters)
    c_avg, c_min = timed(run_cand, warmup=args.warmup, iters=args.iters)

    print(json.dumps({
        "gpu": torch.cuda.get_device_name(0),
        "op": "rope",
        "impl": args.impl,
        "batch": args.batch, "heads": args.heads, "seq": args.seq, "head_dim": args.head_dim,
        "dtype": args.dtype,
        "parity": parity,
        "torch_seconds_avg": t_avg,
        "candidate_seconds_avg": c_avg,
        "speedup": t_avg / c_avg if c_avg else None,
        "torch_seconds_min": t_min,
        "candidate_seconds_min": c_min,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
