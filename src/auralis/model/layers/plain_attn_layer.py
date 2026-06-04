"""Plain causal multi-head attention — the "control group" for ablation.

Pure softmax-attention with RoPE, no window, no globals, no sparsity.
Parameter count per layer = 4 * d_model² (standard). Used by the
``helix_v2_100m_ref.yaml`` ablation config to answer "does our hybrid
stack beat a plain transformer?". NOT meant to be wired into the
production 1B config.

flash-attn swap works the same way as SparseAttentionLayer (same KV
projections and contract) — just use SparseAttentionLayer with a
window_size ≥ max_seq_len and global_tokens=0 to get equivalent
behaviour without a separate class. This file exists mainly so that
ablation configs can say ``type: plain_attention`` explicitly in the
layer list and the intent is readable in the YAML.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from auralis.model.utils.rotary import apply_rotary_pos_emb


class PlainAttentionLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_head: int = 64,
        use_rope: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_head
        self.use_rope = use_rope

        self.q_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.k_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.v_proj = nn.Linear(d_model, n_heads * d_head, bias=False)
        self.out_proj = nn.Linear(n_heads * d_head, d_model, bias=False)

    def forward(self, x, rope=None):
        B, L, _ = x.shape
        H, D = self.n_heads, self.d_head
        q = self.q_proj(x).view(B, L, H, D)
        k = self.k_proj(x).view(B, L, H, D)
        v = self.v_proj(x).view(B, L, H, D)
        if self.use_rope and rope is not None:
            q, k = apply_rotary_pos_emb(q, k, rope[0], rope[1])

        # Use torch.nn.functional.scaled_dot_product_attention for flash-attn
        # auto-selection when available (torch>=2.0, CUDA/Metal).
        q = q.transpose(1, 2)  # [B, H, L, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, L, H * D)
        return self.out_proj(out), None


__all__ = ["PlainAttentionLayer"]
