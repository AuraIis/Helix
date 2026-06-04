"""Sparse attention layer — sliding window + optional global tokens.

Two back-ends:

- **native**: pure-PyTorch causal softmax with a custom mask (window + globals).
  Portable, O(L²) memory, quadratic in seq length.
- **flash** (``AURALIS_USE_CUDA_KERNELS=1`` + CUDA + ``flash-attn`` installed):
  uses ``flash_attn_func`` with its built-in sliding-window option. Linear
  memory in seq length, much faster. ``global_tokens`` are NOT supported by
  the stock flash-attn API, so when they are needed we fall back to native.

Same parameters for both, so swapping is transparent.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from auralis.model.layers.norm import RMSNorm
from auralis.model.utils.rotary import apply_rotary_pos_emb

try:
    from flash_attn import flash_attn_func as _flash_attn_func  # type: ignore
    _FLASH_AVAILABLE = True
except Exception:
    _flash_attn_func = None
    _FLASH_AVAILABLE = False


def _use_flash(on_cuda: bool, global_tokens: int) -> bool:
    if not (_FLASH_AVAILABLE and on_cuda):
        return False
    if global_tokens != 0:
        return False                               # stock flash-attn has no global-tokens
    if os.environ.get("AURALIS_USE_FLASH_ATTN", "") == "1":
        return True
    return os.environ.get("AURALIS_USE_CUDA_KERNELS", "0") == "1"


class SparseAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 16,
        d_head: int = 128,
        window_size: int = 1024,
        global_tokens: int = 32,
        use_rope: bool = True,
        qk_norm: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.window_size = window_size
        self.global_tokens = global_tokens
        self.use_rope = use_rope
        self.qk_norm_enabled = qk_norm

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * d_head, d_model, bias=False)
        self.q_norm = RMSNorm(d_head) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(d_head) if qk_norm else nn.Identity()

    def forward(self, x, rope=None):
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_head

        q = self.q_proj(x).view(B, L, H, D)
        k = self.k_proj(x).view(B, L, H, D)
        v = self.v_proj(x).view(B, L, H, D)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.use_rope and rope is not None:
            q, k = apply_rotary_pos_emb(q, k, rope[0], rope[1])

        if _use_flash(x.is_cuda, self.global_tokens):
            if q.dtype not in (torch.float16, torch.bfloat16):
                target_dtype = (
                    torch.get_autocast_dtype("cuda")
                    if torch.is_autocast_enabled("cuda")
                    else torch.bfloat16
                )
                q = q.to(target_dtype)
                k = k.to(target_dtype)
                v = v.to(target_dtype)
            # flash_attn expects [B, L, H, D] and takes window_size=(left, right)
            out = _flash_attn_func(
                q, k, v,
                causal=True,
                window_size=(self.window_size - 1, 0),        # causal → right=0
                softmax_scale=None,                            # uses 1/sqrt(d_head) by default
            )
        else:
            out = self._native(q, k, v)

        return self.out_proj(out.reshape(B, L, H * D)), None

    def _native(self, q, k, v):
        B, L, H, D = q.shape
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (D ** -0.5)
        mask = self._build_mask(L, device=q.device)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        return out.transpose(1, 2).contiguous()

    def _build_mask(self, L, device):
        i = torch.arange(L, device=device).unsqueeze(1)
        j = torch.arange(L, device=device).unsqueeze(0)
        causal = j > i
        outside_window = (i - j) >= self.window_size
        global_ok = j < self.global_tokens
        return causal | (outside_window & ~global_ok)


__all__ = ["SparseAttentionLayer"]
