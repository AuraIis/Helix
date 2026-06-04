"""Liger-Kernel FusedLinearCrossEntropy adapter.

Liger computes ``loss = cross_entropy(hidden @ weight.T, labels)`` but FUSES the
LM-head projection into the CE op: it never materializes the ``[tokens, vocab]``
logits, and it accumulates ``grad_weight`` via a GEMM rather than a racy
``atomicAdd``. That makes the gradients **exact-by-construction** — no
reduction-order drift — which is exactly the property the hand-rolled
``triton_fused`` ``atomic*`` paths could not reach (they bottomed out at ~0.36%
upstream-gradient L2 drift). With ``accum_dtype=torch.float32`` the gradient
accumulation runs in fp32 regardless of bf16 inputs.

The contract matches the other lab CE candidates so it drops straight into the
benchmark / drift-gate harness::

    liger_linear_cross_entropy(hidden, weight, labels) -> scalar loss tensor

The returned loss is autograd-connected to BOTH ``hidden`` (so the upstream
projection gets a gradient) and ``weight`` (the LM head).
"""
from __future__ import annotations

import torch

_LIGER_FN = None


def liger_available() -> bool:
    try:
        _get_liger_fn()
        return True
    except ImportError:
        return False


def _get_liger_fn():
    global _LIGER_FN
    if _LIGER_FN is None:
        try:
            from liger_kernel.transformers.functional import (
                liger_fused_linear_cross_entropy as fn,
            )
        except Exception as exc:  # noqa: BLE001 - import guard
            raise ImportError(
                "liger-kernel is not installed. Install it in the GPU container with "
                "`pip install --no-deps liger-kernel` (pure-Python + Triton, no deps "
                "touched)."
            ) from exc
        _LIGER_FN = fn
    return _LIGER_FN


def liger_linear_cross_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: str = "mean",
    accum_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Fused linear cross-entropy via Liger.

    Args:
        hidden: ``[tokens, d_model]`` activations feeding the LM head.
        weight: ``[vocab, d_model]`` LM-head weight (nn.Linear layout).
        labels: ``[tokens]`` int64 targets; ``ignore_index`` rows are skipped.
        accum_dtype: gradient accumulation dtype. ``torch.float32`` forces fp32
            accumulation (recommended for bf16 inputs); ``None`` uses Liger's
            default.
    """
    fn = _get_liger_fn()
    out = fn(
        hidden,
        weight,
        labels,
        bias=None,
        ignore_index=ignore_index,
        reduction=reduction,
        accum_dtype=accum_dtype,
    )
    # With return_z_loss / return_token_accuracy defaults (False) Liger returns a
    # bare loss tensor, but guard against versions that return a tuple.
    if isinstance(out, tuple):
        return out[0]
    return out
