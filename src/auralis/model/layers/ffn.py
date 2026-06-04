"""Feed-forward network variants.

- ``DenseFFN``: SwiGLU — the de-facto standard for modern decoder-only LMs.
- ``MoEFFN``: placeholder that falls back to ``DenseFFN`` until Phase 5
  enables MoE. Keeping the hook here (rather than later surgery) is the
  whole point of "modular from the start" in the spec.

``build_ffn(config)`` is the factory used by :class:`HelixBlock`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseFFN(nn.Module):
    """SwiGLU gated FFN.

    ``out = down(silu(gate(x)) * up(x))``.

    Bias-free, matches Llama/Mistral convention. Parameter count:
    ``3 * d_model * d_ffn``.
    """

    def __init__(self, d_model: int, d_ffn: int, activation: str = "silu_gated"):
        super().__init__()
        if activation != "silu_gated":
            raise ValueError(f"Only silu_gated (SwiGLU) is implemented, got {activation!r}")
        self.gate_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.up_proj = nn.Linear(d_model, d_ffn, bias=False)
        self.down_proj = nn.Linear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEFFN(nn.Module):
    """Mixture-of-experts FFN placeholder.

    Activates via ``config.moe.enabled = True`` (Phase 5). Until then this
    is a plain Dense FFN so the rest of the model is unaffected.
    """

    def __init__(
        self,
        d_model: int,
        d_ffn: int,
        n_experts: int = 8,
        n_experts_per_token: int = 2,
        capacity_factor: float = 1.25,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.n_experts_per_token = n_experts_per_token
        self.capacity_factor = capacity_factor
        self._fallback = DenseFFN(d_model, d_ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._fallback(x)


def build_ffn(config) -> nn.Module:
    """Factory: pick FFN variant from an ``AuralisConfig``."""
    if config.ffn.type == "moe" and config.moe.enabled:
        return MoEFFN(
            d_model=config.d_model,
            d_ffn=config.d_ffn,
            n_experts=config.moe.n_experts,
            n_experts_per_token=config.moe.n_experts_per_token,
            capacity_factor=config.moe.capacity_factor,
        )
    return DenseFFN(d_model=config.d_model, d_ffn=config.d_ffn, activation=config.ffn.activation)


__all__ = ["DenseFFN", "MoEFFN", "build_ffn"]
