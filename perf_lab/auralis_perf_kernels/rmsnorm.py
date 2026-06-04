"""CUDA-backed RMSNorm experiment with safe PyTorch fallback."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXT = None


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    root = Path(__file__).resolve().parents[1]
    build_dir = Path(os.environ.get("AURALIS_PERF_BUILD_DIR", root / ".build" / "rmsnorm"))
    build_dir.mkdir(parents=True, exist_ok=True)
    _EXT = load(
        name="auralis_rmsnorm_ext",
        sources=[
            str(root / "csrc" / "rmsnorm_ext.cpp"),
            str(root / "csrc" / "rmsnorm_kernel.cu"),
        ],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        build_directory=str(build_dir),
        verbose=bool(int(os.environ.get("AURALIS_PERF_VERBOSE_BUILD", "0"))),
    )
    return _EXT


def _torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(variance + eps).to(x.dtype) * weight.to(dtype=x.dtype)


class _RMSNormFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        ext = _load_ext()
        x_contig = x.contiguous()
        weight_contig = weight.contiguous()
        y, inv_rms = ext.rmsnorm_forward(x_contig, weight_contig, float(eps))
        ctx.save_for_backward(x_contig, weight_contig, inv_rms)
        return y

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, weight, inv_rms = ctx.saved_tensors
        ext = _load_ext()
        grad_x, grad_weight = ext.rmsnorm_backward(grad_out.contiguous(), x, weight, inv_rms)
        return grad_x, grad_weight, None


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Run experimental RMSNorm when supported, otherwise use PyTorch.

    The first prototype requires CUDA tensors and a fp32 scale parameter, which
    matches the normal training path where parameters stay fp32 under autocast.
    """

    if not x.is_cuda or not weight.is_cuda or weight.dtype != torch.float32:
        return _torch_rmsnorm(x, weight, eps)
    if x.dim() < 2 or x.shape[-1] != weight.numel():
        raise ValueError(f"bad RMSNorm shape: x={tuple(x.shape)} weight={tuple(weight.shape)}")
    return _RMSNormFn.apply(x, weight, float(eps))

