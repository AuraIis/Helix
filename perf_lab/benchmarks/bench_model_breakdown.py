#!/usr/bin/env python3
"""Op-level forward+backward time breakdown for the Helix v2 1B model.

Goal: answer "where does a training step's time actually go?" without needing the
mixer kernels (mamba_ssm / fla / flash-attn), which require a CUDA build not
available on the clean 3090. This measures every component that is plain torch —
the 200k LM head + CE, the 28 FFN blocks, the 56 RMSNorms, RoPE — at the real
model dims, and explicitly leaves the mixer CORES (Mamba selective-scan, GLA
chunk-recurrence, sparse attention) as a separately-flagged unmeasured bucket.

It is a *lower bound* on step time (mixer cores excluded), but it cleanly settles
the practical question: do the GEMMs / LM head dominate, and how small are the
norm/rope/elementwise ops we benchmarked Liger against?

helix_v2_1b: d_model=1280, d_ffn=3584, n_layers=28 (6 mamba + 16 gla + 6 sparse),
vocab=200000, n_heads=10, head_dim=128.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
import torch.nn.functional as F

D_MODEL = 1280
D_FFN = 3584
N_LAYERS = 28
N_SPARSE = 6          # only sparse-attn layers use RoPE
VOCAB = 200000
N_HEADS = 10
HEAD_DIM = 128
EPS = 1e-6


def time_fwd_bwd(build, *, warmup: int, iters: int) -> float:
    """build() -> (loss_scalar, leaves_to_grad). Times fwd+bwd, returns avg seconds."""
    for _ in range(warmup):
        loss = build()
        loss.backward()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        loss = build()
        loss.backward()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.mean(ts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tokens", type=int, default=2048, help="tokens per micro-batch (batch*seq)")
    p.add_argument("--seq", type=int, default=2048, help="sequence length (for RoPE shape)")
    p.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=20)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    dev = "cuda"
    T = args.tokens
    torch.manual_seed(0)

    def lin(inp, outp):
        return (torch.randn(outp, inp, device=dev, dtype=dtype) * (inp ** -0.5)).requires_grad_(True)

    def x(*shape):
        return torch.randn(*shape, device=dev, dtype=dtype).requires_grad_(True)

    # --- component builders (each does fwd producing a scalar loss) ---
    def rmsnorm():
        h = x(T, D_MODEL); w = (torch.randn(D_MODEL, device=dev, dtype=torch.float32)).requires_grad_(True)
        var = h.float().pow(2).mean(-1, keepdim=True)
        y = h * torch.rsqrt(var + EPS).to(h.dtype) * w.to(h.dtype)
        return y.float().pow(2).mean()

    def ffn():  # SwiGLU: gate, up, silu-mul, down
        h = x(T, D_MODEL); wg = lin(D_MODEL, D_FFN); wu = lin(D_MODEL, D_FFN); wd = lin(D_FFN, D_MODEL)
        gated = F.silu(F.linear(h, wg)) * F.linear(h, wu)
        return F.linear(gated, wd).float().pow(2).mean()

    def attn_proj():  # rough mixer projection cost: qkv-ish in + out projection at d_model
        h = x(T, D_MODEL); w_in = lin(D_MODEL, 3 * D_MODEL); w_out = lin(D_MODEL, D_MODEL)
        proj = F.linear(h, w_in)
        return F.linear(proj[:, :D_MODEL], w_out).float().pow(2).mean()

    def lm_head_ce():  # tied head: [T,d] -> [V] + cross-entropy (full logits)
        h = x(T, D_MODEL); w = lin(D_MODEL, VOCAB)
        labels = torch.randint(0, VOCAB, (T,), device=dev)
        logits = F.linear(h, w)
        return F.cross_entropy(logits, labels)

    def rope():
        B = max(1, T // args.seq); S = args.seq
        q = x(B, N_HEADS, S, HEAD_DIM); k = x(B, N_HEADS, S, HEAD_DIM)
        inv = 1.0 / (10000.0 ** (torch.arange(0, HEAD_DIM, 2, device=dev).float() / HEAD_DIM))
        emb = torch.cat([torch.outer(torch.arange(S, device=dev).float(), inv)] * 2, dim=-1)
        cos = emb.cos().to(dtype)[None, None]; sin = emb.sin().to(dtype)[None, None]
        def rh(t):
            half = t.shape[-1] // 2
            return torch.cat((-t[..., half:], t[..., :half]), dim=-1)
        qe = q * cos + rh(q) * sin
        return qe.float().pow(2).mean()

    # one-call seconds for each, then scaled by how many run per forward
    comps = [
        ("rmsnorm",          rmsnorm,   2 * N_LAYERS),   # pre + post norm per layer
        ("ffn_swiglu",       ffn,       N_LAYERS),
        ("mixer_proj_proxy", attn_proj, N_LAYERS),       # rough; excludes scan/recurrence/attn core
        ("rope",             rope,      N_SPARSE),
        ("lm_head_ce",       lm_head_ce, 1),
    ]
    rows = []
    for name, fn, count in comps:
        per = time_fwd_bwd(fn, warmup=args.warmup, iters=args.iters)
        rows.append({"component": name, "per_call_ms": per * 1e3, "count": count,
                     "total_ms": per * count * 1e3})

    measured_total = sum(r["total_ms"] for r in rows)
    for r in rows:
        r["pct_of_measured"] = 100.0 * r["total_ms"] / measured_total if measured_total else 0.0

    print(json.dumps({
        "gpu": torch.cuda.get_device_name(0),
        "tokens": T,
        "dtype": args.dtype,
        "note": "fwd+bwd per component; mixer CORES (mamba scan / gla recurrence / "
                "sparse attn) NOT measured (need their kernels). 'mixer_proj_proxy' is a "
                "rough GEMM stand-in for per-layer in/out projections, not the mixer math.",
        "components": sorted(rows, key=lambda r: -r["total_ms"]),
        "measured_total_ms": measured_total,
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
