"""Config loader + param-estimate tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from auralis.model.config import AuralisConfig, LayerConfig

REPO = Path(__file__).resolve().parents[2]
CFG_100M = REPO / "configs" / "model" / "helix_v2_100m.yaml"
CFG_1B = REPO / "configs" / "model" / "helix_v2_1b.yaml"


def test_100m_config_loads():
    c = AuralisConfig.from_yaml(CFG_100M)
    assert c.name == "helix_v2_100m"
    assert c.n_layers == 8
    assert len(c.layers) == 8
    assert c.d_head * c.n_heads == c.d_model


def test_1b_config_loads():
    c = AuralisConfig.from_yaml(CFG_1B)
    assert c.n_layers == 28
    assert len(c.layers) == 28
    assert c.d_head * c.n_heads == c.d_model


def test_layer_stack_ordering_100m():
    c = AuralisConfig.from_yaml(CFG_100M)
    types = [lc.type for lc in c.layers]
    assert types[:2] == ["mamba", "mamba"]
    assert types[2:6] == ["gla"] * 4
    assert types[6:] == ["sparse_attention", "sparse_attention"]


def test_layer_stack_ordering_1b():
    c = AuralisConfig.from_yaml(CFG_1B)
    types = [lc.type for lc in c.layers]
    assert types[:6] == ["mamba"] * 6
    assert types[6:22] == ["gla"] * 16
    assert types[22:] == ["sparse_attention"] * 6


def test_n_layers_mismatch_raises():
    import yaml
    bad = yaml.safe_load(CFG_100M.read_text(encoding="utf-8"))
    bad["model"]["n_layers"] = 16  # config has 8, say 16
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
        yaml.safe_dump(bad, fh)
        tmp_path = fh.name
    try:
        with pytest.raises(ValueError):
            AuralisConfig.from_yaml(tmp_path)
    finally:
        Path(tmp_path).unlink()


def test_d_head_mismatch_raises():
    c = AuralisConfig.from_yaml(CFG_100M)
    # Build a new config with a deliberately bad combination by editing YAML.
    import yaml
    import tempfile
    bad = yaml.safe_load(CFG_100M.read_text(encoding="utf-8"))
    bad["model"]["d_head"] = 17  # 8 * 17 != 512
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
        yaml.safe_dump(bad, fh)
        tmp_path = fh.name
    try:
        with pytest.raises(ValueError):
            AuralisConfig.from_yaml(tmp_path)
    finally:
        Path(tmp_path).unlink()


def test_param_estimate_100m_range():
    c = AuralisConfig.from_yaml(CFG_100M)
    total = c.estimate_parameters()["total"]
    # 100M test model actually estimates ~134M (200k vocab dominates).
    assert 100_000_000 <= total <= 200_000_000


def test_param_estimate_1b_range():
    c = AuralisConfig.from_yaml(CFG_1B)
    total = c.estimate_parameters()["total"]
    # 1B target; we accept 0.8B – 1.3B.
    assert 800_000_000 <= total <= 1_300_000_000


def test_mid_500m_config_loads_and_sizes():
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "configs" / "model" / "helix_v2_mid_500m.yaml"
    c = AuralisConfig.from_yaml(p)
    assert c.n_layers == 20
    assert c.d_model == 1024
    assert c.n_heads == 16 and c.d_head == 64
    # With 200k vocab + tied embeddings, this spec lands ~517M, not 250M —
    # documented in docs/TRAINING_WAVES.md. The range guards against silent
    # architecture drift, not the exact number.
    total = c.estimate_parameters()["total"]
    assert 450_000_000 <= total <= 600_000_000


def test_mid_500m_smart_config_interleaves_attention():
    p = REPO / "configs" / "model" / "helix_v2_mid_500m_smart.yaml"
    c = AuralisConfig.from_yaml(p)
    types = [lc.type for lc in c.layers]
    assert c.n_layers == 20
    assert types.count("mamba") == 4
    assert types.count("gla") == 12
    assert types.count("sparse_attention") == 4
    assert types[:4] == ["mamba", "gla", "gla", "sparse_attention"]
    assert all(lc.global_tokens == 0 for lc in c.layers if lc.type == "sparse_attention")
    assert all(lc.qk_norm for lc in c.layers if lc.type == "sparse_attention")
    assert 450_000_000 <= c.estimate_parameters()["total"] <= 600_000_000


def test_layer_repeat_sugar_works():
    """Sanity: the compact `repeat:` sugar expands to identical LayerConfigs."""
    c = AuralisConfig.from_yaml(CFG_100M)
    mambas = [lc for lc in c.layers if lc.type == "mamba"]
    assert all(lc.d_state == 64 and lc.d_conv == 4 for lc in mambas)


def test_moe_defaults_disabled():
    c = AuralisConfig.from_yaml(CFG_100M)
    assert c.moe.enabled is False
    assert c.mtp.enabled is False
