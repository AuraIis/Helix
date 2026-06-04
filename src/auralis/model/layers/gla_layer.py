"""Gated Linear Attention (GLA) layer.

Two back-ends, same ``forward(x) -> (out, state)`` contract:

- **native** (default): pure-PyTorch sequential outer-product scan. Portable.
- **fla** (when ``AURALIS_USE_CUDA_KERNELS=1`` + CUDA + ``flash-linear-attention``
  installed): wraps ``fla.ops.gla.chunk_gla`` — fused Triton chunk-wise kernel,
  20–30× faster on GPU.

Both back-ends share the same trainable parameters (Q/K/V/G projections +
alpha-gate + output) so a checkpoint trained with one works with the other.
Only the inner scan math differs.
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from fla.ops.gla import chunk_gla as _chunk_gla  # type: ignore
    _FLA_AVAILABLE = True
except Exception:
    _chunk_gla = None
    _FLA_AVAILABLE = False


def _use_fla(on_cuda: bool) -> bool:
    if not (_FLA_AVAILABLE and on_cuda):
        return False
    if os.environ.get("AURALIS_USE_GLA_KERNEL", "") == "1":
        return True
    return os.environ.get("AURALIS_USE_CUDA_KERNELS", "0") == "1"


class GLALayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 16,
        d_head: int = 128,
        d_state: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_state = d_state or d_head

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.g_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        # Per-head-per-dim log-decay projection. fla's ``chunk_gla`` expects g
        # shaped [B, L, H, D] in log-space. The forward uses
        # ``log_alpha = -softplus(-raw)``; initialise raw so exp(log_alpha)
        # starts near 0.9 instead of the generic-init default 0.5.
        self.alpha_proj = nn.Linear(d_model, n_heads * d_head, bias=True)
        self._target_decay = 0.9
        self.reset_special_parameters()

        self.out_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

    def reset_special_parameters(self) -> None:
        """Restore the GLA decay bias after generic model init."""
        decay_lambda = -math.log(self._target_decay)
        # For log_alpha = -softplus(-bias), solve softplus(-bias)=lambda.
        bias = -math.log(math.expm1(decay_lambda))
        with torch.no_grad():
            self.alpha_proj.bias.fill_(bias)

    def forward(self, x, state=None):
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, L, H, D)
        k = self.k_proj(x).view(B, L, H, D)
        v = self.v_proj(x).view(B, L, H, D)
        g_out = torch.sigmoid(self.g_proj(x).view(B, L, H, D))
        # log-decay gate for the scan: keep it negative (decay in (0,1])
        log_alpha = -F.softplus(-self.alpha_proj(x).view(B, L, H, D))

        if _use_fla(x.is_cuda):
            out, new_state = _chunk_gla(q, k, v, log_alpha,
                                        scale=D ** -0.5,
                                        initial_state=state,
                                        output_final_state=False)
        else:
            out, new_state = self._native_scan(q, k, v, log_alpha, state)

        out = out * g_out                                     # per-channel output gate
        return self.out_proj(out.reshape(B, L, H * D)), new_state

    def _native_scan(self, q, k, v, log_alpha, state):
        """Sequential reference — matches chunk_gla semantics."""
        B, L, H, D = q.shape
        q = q * (D ** -0.5)
        alpha = torch.exp(log_alpha)                          # [B, L, H, D] in (0, 1]

        if state is None:
            S = torch.zeros(B, H, D, D, device=q.device, dtype=q.dtype)
        else:
            S = state.to(q.dtype)

        outs = []
        for t in range(L):
            # per-channel decay broadcast over the D key dimension of S
            a_t = alpha[:, t].unsqueeze(-2)                   # [B, H, 1, D]
            update = torch.einsum("bhd,bhe->bhde", k[:, t], v[:, t])
            S = a_t * S + update
            outs.append(torch.einsum("bhd,bhde->bhe", q[:, t], S))
        return torch.stack(outs, dim=1), S


__all__ = ["GLALayer"]
