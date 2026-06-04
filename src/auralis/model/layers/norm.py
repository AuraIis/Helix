"""Normalization layers.

Helix v2 standardizes on RMSNorm (no centering, no bias). Cheaper than
LayerNorm and empirically at-least-as-good for decoder-only transformers.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root-Mean-Square Norm.

    Formula::

        y = x * rsqrt(mean(x**2, dim=-1, keepdim=True) + eps) * weight

    No bias, no centering. One learnable scale per channel.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 reduction for numerical stability regardless of input dtype
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return x * self.weight.to(dtype=x.dtype)


__all__ = ["RMSNorm"]
