"""Unit tests for individual layers (RMSNorm, SwiGLU, RoPE, Mamba, GLA, Sparse)."""

from __future__ import annotations

import torch

from auralis.model.layers.ffn import DenseFFN
from auralis.model.layers.gla_layer import GLALayer
from auralis.model.layers.mamba_layer import Mamba2Layer
from auralis.model.layers.norm import RMSNorm
from auralis.model.layers.sparse_attn_layer import SparseAttentionLayer
from auralis.model.utils.rotary import RotaryEmbedding, apply_rotary_pos_emb


# ---------- RMSNorm ----------

def test_rmsnorm_shape_preserved():
    x = torch.randn(2, 7, 64)
    y = RMSNorm(64)(x)
    assert y.shape == x.shape


def test_rmsnorm_unit_variance_at_init():
    x = torch.randn(4, 16, 128) * 10  # rescale to make the norm obviously active
    y = RMSNorm(128)(x)
    rms = y.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-3)


# ---------- SwiGLU ----------

def test_swiglu_shape_preserved():
    x = torch.randn(2, 5, 64)
    y = DenseFFN(64, 128)(x)
    assert y.shape == x.shape


def test_swiglu_no_nan_with_extremes():
    x = torch.tensor([[[1e3, -1e3, 0.0, 1e-3] * 16]])
    y = DenseFFN(64, 128)(x)
    assert not torch.isnan(y).any()


# ---------- RoPE ----------

def test_rope_output_shapes():
    rope = RotaryEmbedding(dim=64, max_seq_len=128)
    cos, sin = rope(32, device=torch.device("cpu"))
    assert cos.shape == (32, 64) and sin.shape == (32, 64)


def test_rope_roundtrip_preserves_vector_norm():
    rope = RotaryEmbedding(dim=32, max_seq_len=16)
    cos, sin = rope(16, device=torch.device("cpu"))
    q = torch.randn(1, 16, 2, 32)
    k = torch.randn(1, 16, 2, 32)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    # RoPE is a rotation, norms should be preserved.
    assert torch.allclose(q_rot.norm(dim=-1), q.norm(dim=-1), atol=1e-4)
    assert torch.allclose(k_rot.norm(dim=-1), k.norm(dim=-1), atol=1e-4)


# ---------- Mamba-2 ----------

def test_mamba2_forward_shape():
    layer = Mamba2Layer(d_model=64, d_state=16, d_conv=4, expand_factor=2)
    x = torch.randn(2, 10, 64)
    out, state = layer(x)
    assert out.shape == x.shape
    assert state.shape == (2, 2 * 64, 16)  # [B, d_inner, d_state]


def test_mamba2_backward_runs():
    layer = Mamba2Layer(d_model=32, d_state=8)
    x = torch.randn(2, 6, 32, requires_grad=True)
    out, _ = layer(x)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


# ---------- GLA ----------

def test_gla_forward_shape():
    layer = GLALayer(d_model=64, n_heads=4, d_head=16)
    x = torch.randn(2, 8, 64)
    out, state = layer(x)
    assert out.shape == x.shape
    assert state.shape == (2, 4, 16, 16)  # [B, H, D, D]


def test_gla_backward_runs():
    layer = GLALayer(d_model=32, n_heads=4, d_head=8)
    x = torch.randn(2, 5, 32, requires_grad=True)
    out, _ = layer(x)
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0


# ---------- Sparse attention ----------

def test_sparse_attn_forward_shape():
    layer = SparseAttentionLayer(d_model=64, n_heads=4, d_head=16,
                                 window_size=4, global_tokens=2, use_rope=True)
    rope = RotaryEmbedding(dim=16, max_seq_len=32)
    rope_out = rope(12, device=torch.device("cpu"))
    x = torch.randn(2, 12, 64)
    out, _ = layer(x, rope=rope_out)
    assert out.shape == x.shape


def test_sparse_attn_qk_norm_forward_shape_and_params():
    layer = SparseAttentionLayer(d_model=64, n_heads=4, d_head=16,
                                 window_size=4, global_tokens=0,
                                 use_rope=True, qk_norm=True)
    rope = RotaryEmbedding(dim=16, max_seq_len=32)
    rope_out = rope(12, device=torch.device("cpu"))
    x = torch.randn(2, 12, 64)
    out, _ = layer(x, rope=rope_out)
    assert out.shape == x.shape
    assert layer.q_norm.weight.shape == (16,)
    assert layer.k_norm.weight.shape == (16,)


def test_sparse_attn_causal_mask_blocks_future():
    """Future tokens must be blocked: swapping a late key should not change
    output at an earlier query position."""
    layer = SparseAttentionLayer(d_model=32, n_heads=4, d_head=8,
                                 window_size=16, global_tokens=0, use_rope=False)
    x1 = torch.randn(1, 8, 32)
    x2 = x1.clone()
    x2[:, 7] = torch.randn_like(x2[:, 7])          # only perturb last position
    out1, _ = layer(x1)
    out2, _ = layer(x2)
    # Positions 0..6 must be identical regardless of token 7 (causal).
    assert torch.allclose(out1[:, :7], out2[:, :7], atol=1e-5)


def test_sparse_attn_window_mask_blocks_out_of_range():
    """With window_size=2 and global_tokens=0, token 4 should not see token 0."""
    layer = SparseAttentionLayer(d_model=32, n_heads=4, d_head=8,
                                 window_size=2, global_tokens=0, use_rope=False)
    x1 = torch.randn(1, 5, 32)
    x2 = x1.clone()
    x2[:, 0] = torch.randn_like(x2[:, 0])          # perturb token 0 only
    out1, _ = layer(x1)
    out2, _ = layer(x2)
    # Position 4 is 4 steps away from 0 (> window=2), so output at t=4 unchanged.
    assert torch.allclose(out1[:, 4], out2[:, 4], atol=1e-5)


def test_sparse_attn_global_token_is_visible_from_anywhere():
    """global_tokens=1 means token 0 must be attendable everywhere."""
    layer = SparseAttentionLayer(d_model=32, n_heads=4, d_head=8,
                                 window_size=2, global_tokens=1, use_rope=False)
    x1 = torch.randn(1, 6, 32)
    x2 = x1.clone()
    x2[:, 0] = torch.randn_like(x2[:, 0])          # perturb global token
    out1, _ = layer(x1)
    out2, _ = layer(x2)
    # Now position 5 DOES see token 0 (as a global), so output should differ.
    assert not torch.allclose(out1[:, 5], out2[:, 5], atol=1e-5)
