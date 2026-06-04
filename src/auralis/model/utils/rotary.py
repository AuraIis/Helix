"""Rotary Position Embeddings (RoPE).

Used only by the late Sparse-Attention layers (layers 22-27 in the 28-layer
stack). Mamba and GLA do not need RoPE — state space / linear attention
layers have their own positional mechanics.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Pre-computes cos/sin caches for RoPE.

    ``forward(seq_len, device=...)`` returns ``(cos, sin)`` of shape
    ``[seq_len, d_head]`` each. Caller applies them to Q and K via
    :func:`apply_rotary_pos_emb`.
    """

    def __init__(self, dim: int, max_seq_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"RoPE dim must be even, got {dim}")
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cached: torch.Tensor | None = None
        self._sin_cached: torch.Tensor | None = None
        self._cached_len = 0

    def _build_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> None:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))                # [L, d/2]
        emb = torch.cat((freqs, freqs), dim=-1)                          # [L, d]
        self._cos_cached = emb.cos().to(dtype)
        self._sin_cached = emb.sin().to(dtype)
        self._cached_len = seq_len

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype = torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._cos_cached is None
            or self._cached_len < seq_len
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
        ):
            self._build_cache(max(seq_len, self.max_seq_len), device, dtype)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to Q and K tensors shaped ``[B, L, H, D]``.

    ``cos`` / ``sin`` are ``[L, D]`` — they broadcast over batch and heads.
    """
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, L, 1, D]
    sin = sin.unsqueeze(0).unsqueeze(2)
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot


__all__ = ["RotaryEmbedding", "apply_rotary_pos_emb"]
