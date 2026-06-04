"""Plain-attention layer + reference 100m config tests."""

from __future__ import annotations

from pathlib import Path

import torch

from auralis.model import build_model
from auralis.model.backend_info import describe_model_backends
from auralis.model.config import AuralisConfig, LayerConfig
from auralis.model.helix_model import HelixModel
from auralis.model.layers.plain_attn_layer import PlainAttentionLayer
from auralis.model.utils.rotary import RotaryEmbedding


def test_plain_attention_shape_and_causality():
    layer = PlainAttentionLayer(d_model=32, n_heads=4, d_head=8, use_rope=True)
    rope = RotaryEmbedding(dim=8, max_seq_len=16)(8, device=torch.device("cpu"))
    x = torch.randn(1, 8, 32)
    out, _ = layer(x, rope=rope)
    assert out.shape == x.shape

    # Causal: perturbing position 7 must not alter output at positions 0..6
    x2 = x.clone()
    x2[:, 7] = torch.randn_like(x2[:, 7])
    out2, _ = layer(x2, rope=rope)
    assert torch.allclose(out[:, :7], out2[:, :7], atol=1e-5)


def test_100m_ref_config_builds_and_forwards():
    repo = Path(__file__).resolve().parents[2]
    model = build_model(repo / "configs" / "model" / "helix_v2_100m_ref.yaml")
    assert all(b.layer_config.type == "plain_attention" for b in model.blocks)
    x = torch.randint(0, model.config.vocab_size, (1, 16))
    out = model(input_ids=x)
    assert out["logits"].shape == (1, 16, model.config.vocab_size)
    assert model.rope is not None


def test_backend_info_describes_plain_attention():
    repo = Path(__file__).resolve().parents[2]
    model = build_model(repo / "configs" / "model" / "helix_v2_100m_ref.yaml")
    desc = describe_model_backends(model)
    summary = desc["summary"]
    assert summary.get("plain_attention:native", 0) == 8


def test_plain_attention_only_model_builds_shared_rope():
    cfg = AuralisConfig(
        name="plain-only",
        version="1.0",
        vocab_size=512,
        d_model=32,
        n_layers=2,
        n_heads=4,
        d_head=8,
        d_ffn=64,
        layers=[
            LayerConfig(type="plain_attention", use_rope=True),
            LayerConfig(type="plain_attention", use_rope=True),
        ],
    )
    model = HelixModel(cfg)
    assert model.rope is not None
