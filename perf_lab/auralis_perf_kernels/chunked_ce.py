"""Chunked huge-vocab linear cross entropy.

This is the first exact prototype for `hidden @ vocab_weight.T + CE` without
materializing the full `[tokens, vocab_size]` logits tensor. It uses regular
Torch GEMMs per vocab chunk, so it is a correctness and memory prototype rather
than the final custom CUDA kernel.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_CPP_EXT = None


def _load_cpp_ext():
    global _CPP_EXT
    if _CPP_EXT is not None:
        return _CPP_EXT
    root = Path(__file__).resolve().parents[1]
    build_dir = Path(os.environ.get("AURALIS_PERF_CE_BUILD_DIR", root / ".build" / "chunked_ce"))
    build_dir.mkdir(parents=True, exist_ok=True)
    _CPP_EXT = load(
        name="auralis_chunked_ce_ext",
        sources=[str(root / "csrc" / "chunked_ce_ext.cpp")],
        extra_cflags=["-O3"],
        build_directory=str(build_dir),
        verbose=bool(int(os.environ.get("AURALIS_PERF_VERBOSE_BUILD", "0"))),
    )
    return _CPP_EXT


def _full_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int,
) -> torch.Tensor:
    logits = F.linear(hidden, weight)
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)


class _ChunkedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        chunk_size: int,
        ignore_index: int,
    ) -> torch.Tensor:
        if hidden.dim() != 2:
            raise ValueError(f"hidden must be [N, D], got {tuple(hidden.shape)}")
        if weight.dim() != 2:
            raise ValueError(f"weight must be [V, D], got {tuple(weight.shape)}")
        if hidden.size(1) != weight.size(1):
            raise ValueError(f"hidden dim {hidden.size(1)} != weight dim {weight.size(1)}")
        labels = labels.reshape(-1).contiguous()
        if labels.numel() != hidden.size(0):
            raise ValueError(f"labels length {labels.numel()} != tokens {hidden.size(0)}")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        tokens = hidden.size(0)
        vocab_size = weight.size(0)
        valid = labels.ne(ignore_index)
        valid_count = valid.sum().clamp_min(1)

        row_max = torch.full((tokens,), -torch.inf, device=hidden.device, dtype=torch.float32)
        target_logits = torch.zeros((tokens,), device=hidden.device, dtype=torch.float32)

        with torch.no_grad():
            for start in range(0, vocab_size, chunk_size):
                end = min(start + chunk_size, vocab_size)
                logits = hidden.matmul(weight[start:end].t()).float()
                row_max = torch.maximum(row_max, logits.max(dim=1).values)
                in_chunk = valid & labels.ge(start) & labels.lt(end)
                rows = in_chunk.nonzero(as_tuple=False).flatten()
                cols = labels[rows] - start
                target_logits[rows] = logits[rows, cols]

            exp_sum = torch.zeros((tokens,), device=hidden.device, dtype=torch.float32)
            for start in range(0, vocab_size, chunk_size):
                end = min(start + chunk_size, vocab_size)
                logits = hidden.matmul(weight[start:end].t()).float()
                exp_sum += torch.exp(logits - row_max[:, None]).sum(dim=1)

            losses = row_max + torch.log(exp_sum) - target_logits
            losses = torch.where(valid, losses, torch.zeros_like(losses))
            loss = losses.sum() / valid_count.to(torch.float32)

        ctx.save_for_backward(hidden, weight, labels, row_max, exp_sum, valid)
        ctx.chunk_size = int(chunk_size)
        ctx.ignore_index = int(ignore_index)
        ctx.valid_count = valid_count
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden, weight, labels, row_max, exp_sum, valid = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        vocab_size = weight.size(0)

        compute_hidden = hidden.float()
        grad_hidden = torch.zeros_like(compute_hidden)
        grad_weight = torch.zeros_like(weight)
        scale = (grad_output.float() / ctx.valid_count.to(torch.float32)).reshape(())

        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)
            weight_chunk = weight[start:end]
            logits = hidden.matmul(weight_chunk.t()).float()
            probs = torch.exp(logits - row_max[:, None]) / exp_sum[:, None]
            probs = torch.where(valid[:, None], probs, torch.zeros_like(probs))

            in_chunk = valid & labels.ge(start) & labels.lt(end)
            rows = in_chunk.nonzero(as_tuple=False).flatten()
            cols = labels[rows] - start
            probs[rows, cols] -= 1.0

            grad_logits = probs * scale
            grad_hidden += grad_logits.matmul(weight_chunk.float())
            grad_weight[start:end] = grad_logits.t().matmul(compute_hidden).to(dtype=weight.dtype)

        return (
            grad_hidden.to(dtype=hidden.dtype),
            grad_weight.to(dtype=weight.dtype),
            None,
            None,
            None,
        )


class _ChunkedLinearCrossEntropyCpp(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        chunk_size: int,
        ignore_index: int,
    ) -> torch.Tensor:
        ext = _load_cpp_ext()
        loss, row_max, exp_sum, valid, valid_count = ext.chunked_ce_forward(
            hidden,
            weight,
            labels,
            int(chunk_size),
            int(ignore_index),
        )
        ctx.save_for_backward(hidden, weight, labels, row_max, exp_sum, valid, valid_count)
        ctx.chunk_size = int(chunk_size)
        ctx.ignore_index = int(ignore_index)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden, weight, labels, row_max, exp_sum, valid, valid_count = ctx.saved_tensors
        ext = _load_cpp_ext()
        grad_hidden, grad_weight = ext.chunked_ce_backward(
            grad_output.contiguous(),
            hidden,
            weight,
            labels,
            row_max,
            exp_sum,
            valid,
            valid_count,
            int(ctx.chunk_size),
            int(ctx.ignore_index),
        )
        return grad_hidden, grad_weight, None, None, None


def chunked_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = 8192,
    ignore_index: int = -100,
    impl: str = "auto",
) -> torch.Tensor:
    """Exact mean cross-entropy over `F.linear(hidden, weight)`, chunked by vocab.

    For small vocabularies or CPU tensors, this falls back to PyTorch because the
    full logits path is simpler and faster there. The prototype target is CUDA
    with very large vocabularies such as Auralis' 200k tokenizer.
    """

    if not hidden.is_cuda or not weight.is_cuda:
        return _full_linear_cross_entropy(hidden, weight, labels.reshape(-1), ignore_index)
    if weight.size(0) <= chunk_size:
        return _full_linear_cross_entropy(hidden, weight, labels.reshape(-1), ignore_index)
    hidden = hidden.contiguous()
    weight = weight.contiguous()
    labels = labels.reshape(-1).contiguous()
    if impl not in {"auto", "cpp", "python"}:
        raise ValueError(f"unknown chunked CE impl: {impl}")
    if impl in {"auto", "cpp"}:
        try:
            return _ChunkedLinearCrossEntropyCpp.apply(
                hidden,
                weight,
                labels,
                int(chunk_size),
                int(ignore_index),
            )
        except Exception as exc:
            if impl == "cpp":
                raise
            # impl == "auto": fall back to the pure-PyTorch path, but make the
            # failure VISIBLE (once). A silently broken CUDA kernel otherwise
            # looks like "works, just slow" — the worst failure mode for a perf lab.
            if not getattr(chunked_linear_cross_entropy, "_warned", False):
                chunked_linear_cross_entropy._warned = True
                import warnings

                warnings.warn(
                    "chunked CE C++ kernel failed; using the pure-PyTorch fallback "
                    f"({type(exc).__name__}: {exc}). Pass impl='cpp' to surface the error.",
                    RuntimeWarning,
                    stacklevel=2,
                )
    return _ChunkedLinearCrossEntropy.apply(
        hidden,
        weight,
        labels,
        int(chunk_size),
        int(ignore_index),
    )


class _TritonForwardLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        chunk_size: int,
        ignore_index: int,
        block_m: int,
        block_v: int,
        block_d: int,
    ) -> torch.Tensor:
        from .triton_ce import triton_linear_ce_forward

        labels = labels.reshape(-1).contiguous()
        valid = labels.ne(ignore_index)
        valid_count = valid.sum().clamp_min(1)
        losses, row_max, exp_sum = triton_linear_ce_forward(
            hidden,
            weight,
            labels,
            ignore_index=int(ignore_index),
            block_m=int(block_m),
            block_v=int(block_v),
            block_d=int(block_d),
            mode="parallel",
        )
        loss = losses.sum() / valid_count.to(torch.float32)
        ctx.save_for_backward(hidden, weight, labels, row_max, exp_sum, valid, valid_count)
        ctx.chunk_size = int(chunk_size)
        ctx.ignore_index = int(ignore_index)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden, weight, labels, row_max, exp_sum, valid, valid_count = ctx.saved_tensors
        ext = _load_cpp_ext()
        grad_hidden, grad_weight = ext.chunked_ce_backward(
            grad_output.contiguous(),
            hidden,
            weight,
            labels,
            row_max,
            exp_sum,
            valid,
            valid_count,
            int(ctx.chunk_size),
            int(ctx.ignore_index),
        )
        return grad_hidden, grad_weight, None, None, None, None, None, None


class _TritonFusedLinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int,
        block_m: int,
        block_v: int,
        block_d: int,
        backward_mode: str,
        row_group_blocks: int,
    ) -> torch.Tensor:
        from .triton_ce import triton_linear_ce_forward

        labels = labels.reshape(-1).contiguous()
        valid_count = labels.ne(ignore_index).sum().clamp_min(1)
        round_bf16_logits = str(backward_mode).endswith("_bf16match")
        losses, row_max, exp_sum = triton_linear_ce_forward(
            hidden,
            weight,
            labels,
            ignore_index=int(ignore_index),
            block_m=int(block_m),
            block_v=int(block_v),
            block_d=int(block_d),
            mode="parallel",
            round_bf16_logits=round_bf16_logits,
        )
        loss = losses.sum() / valid_count.to(torch.float32)
        ctx.save_for_backward(hidden, weight, labels, row_max, exp_sum, valid_count)
        ctx.ignore_index = int(ignore_index)
        ctx.block_m = int(block_m)
        ctx.block_v = int(block_v)
        ctx.block_d = int(block_d)
        ctx.backward_mode = str(backward_mode)
        ctx.row_group_blocks = int(row_group_blocks)
        return loss

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        from .triton_ce import (
            triton_linear_ce_backward,
            triton_linear_ce_backward_atomic_lowmem,
            triton_linear_ce_backward_atomic_mixed,
            triton_linear_ce_backward_partial_weight,
            triton_linear_ce_backward_split_hidden,
            triton_linear_ce_backward_split_weight,
        )

        hidden, weight, labels, row_max, exp_sum, valid_count = ctx.saved_tensors
        round_bf16_logits = str(ctx.backward_mode).endswith("_bf16match")
        base_backward_mode = str(ctx.backward_mode).removesuffix("_bf16match")
        backward_fn = {
            "atomic": triton_linear_ce_backward,
            "atomic_lowmem": triton_linear_ce_backward_atomic_lowmem,
            "atomic_mixed": triton_linear_ce_backward_atomic_mixed,
            "partial_weight": triton_linear_ce_backward_partial_weight,
            "split_hidden": triton_linear_ce_backward_split_hidden,
            "split_weight": triton_linear_ce_backward_split_weight,
        }[base_backward_mode]
        kwargs = {
            "ignore_index": int(ctx.ignore_index),
            "block_m": int(ctx.block_m),
            "block_v": int(ctx.block_v),
            "block_d": int(ctx.block_d),
        }
        if base_backward_mode in {"atomic", "atomic_lowmem", "atomic_mixed"}:
            kwargs["round_bf16_logits"] = round_bf16_logits
        if base_backward_mode == "partial_weight":
            kwargs["row_group_blocks"] = int(ctx.row_group_blocks)
        grad_hidden, grad_weight = backward_fn(
            grad_output.contiguous(),
            hidden,
            weight,
            labels,
            row_max,
            exp_sum,
            valid_count,
            **kwargs,
        )
        return grad_hidden, grad_weight, None, None, None, None, None, None, None


def triton_forward_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = 4096,
    ignore_index: int = -100,
    block_m: int = 8,
    block_v: int = 256,
    block_d: int = 64,
) -> torch.Tensor:
    """Mean CE with Triton parallel forward and C++ chunked backward.

    This hybrid path is a benchmark bridge. It tests whether the parallel
    forward can improve the current chunked train step before we write a fully
    fused backward.
    """

    if not hidden.is_cuda or not weight.is_cuda:
        return _full_linear_cross_entropy(hidden, weight, labels.reshape(-1), ignore_index)
    if weight.size(0) <= block_v:
        return _full_linear_cross_entropy(hidden, weight, labels.reshape(-1), ignore_index)
    return _TritonForwardLinearCrossEntropy.apply(
        hidden.contiguous(),
        weight.contiguous(),
        labels.reshape(-1).contiguous(),
        int(chunk_size),
        int(ignore_index),
        int(block_m),
        int(block_v),
        int(block_d),
    )


def triton_fused_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    block_m: int = 16,
    block_v: int = 32,
    block_d: int = 64,
    backward_mode: str = "atomic",
    row_group_blocks: int = 8,
) -> torch.Tensor:
    """Mean CE with Triton forward and Triton backward.

    This is a correctness-first fused-backward prototype. It avoids full logits
    and the C++ chunk loop, but it is expected to need tile/atomic tuning before
    it can beat the PyTorch full-logits path.
    """

    if backward_mode not in {
        "atomic",
        "atomic_lowmem",
        "atomic_mixed",
        "atomic_bf16match",
        "atomic_lowmem_bf16match",
        "atomic_mixed_bf16match",
        "partial_weight",
        "split_hidden",
        "split_weight",
    }:
        raise ValueError(f"unknown triton fused backward_mode: {backward_mode}")
    block_m = max(int(block_m), 16)
    block_v = max(int(block_v), 16)
    block_d = max(int(block_d), 16)
    if not hidden.is_cuda or not weight.is_cuda:
        return _full_linear_cross_entropy(hidden, weight, labels.reshape(-1), ignore_index)
    if weight.size(0) <= block_v:
        return _full_linear_cross_entropy(hidden, weight, labels.reshape(-1), ignore_index)
    return _TritonFusedLinearCrossEntropy.apply(
        hidden.contiguous(),
        weight.contiguous(),
        labels.reshape(-1).contiguous(),
        int(ignore_index),
        block_m,
        block_v,
        block_d,
        str(backward_mode),
        int(row_group_blocks),
    )


def estimate_logits_gb(tokens: int, vocab_size: int, dtype: torch.dtype) -> float:
    bytes_per = torch.empty((), dtype=dtype).element_size()
    return tokens * vocab_size * bytes_per / 1e9


def suggest_chunk_size(vocab_size: int, target_logits_gb: float, dtype: torch.dtype, tokens: int) -> int:
    """Pick a chunk size that keeps transient logits near target memory."""

    bytes_per = torch.empty((), dtype=dtype).element_size()
    raw = int((target_logits_gb * 1e9) / max(tokens * bytes_per, 1))
    raw = max(1024, min(vocab_size, raw))
    # Round down to a multiple of 1024 for nicer GEMM shapes.
    return max(1024, int(math.floor(raw / 1024) * 1024))


__all__ = [
    "chunked_linear_cross_entropy",
    "estimate_logits_gb",
    "suggest_chunk_size",
    "triton_fused_linear_cross_entropy",
    "triton_forward_linear_cross_entropy",
]
