"""Helix v2 model configuration.

Single source of truth for architecture hyperparameters. Backed by YAML
(see ``configs/model/*.yaml``), no hardcoded values anywhere else.

A ``HelixModel`` is entirely reconstructible from an ``AuralisConfig`` +
the SentencePiece tokenizer model. Serializing/loading a config is how we
make experiments reproducible: a MANIFEST.yaml records the model-config
hash and the training config hash.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Per-layer config (heterogeneous stack: mamba / gla / sparse_attention)
# ---------------------------------------------------------------------------

@dataclass
class LayerConfig:
    """Config for a single layer in the heterogeneous stack."""
    type: str                                 # "mamba" | "gla" | "sparse_attention"
    # Mamba-specific
    d_state: int | None = None
    d_conv: int | None = None
    expand_factor: int | None = None
    dt_min: float | None = None
    dt_max: float | None = None
    # Sparse-attention-specific
    window_size: int | None = None
    global_tokens: int | None = None
    use_rope: bool | None = None
    qk_norm: bool | None = None


# ---------------------------------------------------------------------------
# Component sub-configs
# ---------------------------------------------------------------------------

@dataclass
class FFNConfig:
    type: str = "dense"                       # "dense" | "moe"
    activation: str = "silu_gated"            # "silu_gated" = SwiGLU


@dataclass
class MoEConfig:
    enabled: bool = False
    n_experts: int = 8
    n_experts_per_token: int = 2
    capacity_factor: float = 1.25
    load_balance_loss_weight: float = 0.01


@dataclass
class MTPConfig:
    enabled: bool = False
    n_heads: int = 1
    loss_weight: float = 0.2


@dataclass
class PositionEncodingConfig:
    type: str = "rope"                        # "rope" | "none"
    theta: float = 10000.0
    max_seq_length: int = 8192


@dataclass
class InitConfig:
    scheme: str = "scaled_normal"
    init_std: float = 0.02
    embedding_init_std: float = 0.02
    output_init_scale: float = 0.5


@dataclass
class DropoutConfig:
    embedding: float = 0.0
    attention: float = 0.0
    ffn: float = 0.0
    residual: float = 0.0


@dataclass
class AdvancedConfig:
    tie_embeddings: bool = False
    use_flash_attention: bool = True          # only when available
    gradient_checkpointing: bool = False


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class AuralisConfig:
    """Full Helix v2 model config."""

    # Identity
    name: str
    version: str

    # Core dimensions
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    d_head: int
    d_ffn: int

    # Activation / norm
    activation: str = "silu"
    norm_type: str = "rmsnorm"
    norm_eps: float = 1e-6

    # Per-layer stack
    layers: list[LayerConfig] = field(default_factory=list)

    # Sub-components
    ffn: FFNConfig = field(default_factory=FFNConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    position_encoding: PositionEncodingConfig = field(default_factory=PositionEncodingConfig)
    init: InitConfig = field(default_factory=InitConfig)
    dropout: DropoutConfig = field(default_factory=DropoutConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)

    # External refs
    tokenizer_path: str | None = None

    # ---------------- YAML loader ----------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "AuralisConfig":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        tok = data.get("tokenizer", {}) or {}
        model = data.get("model", {}) or {}
        layers = cls._expand_layers(data.get("layers", []) or [], int(model.get("n_layers", 0)))

        if len(layers) != int(model["n_layers"]):
            raise ValueError(
                f"Config error: resolved {len(layers)} layers but n_layers="
                f"{model['n_layers']}"
            )
        # Sanity: d_head * n_heads == d_model (keeps projections shape-clean)
        if int(model["d_head"]) * int(model["n_heads"]) != int(model["d_model"]):
            raise ValueError(
                f"d_head*n_heads ({model['d_head']}*{model['n_heads']}) "
                f"!= d_model ({model['d_model']})"
            )

        return cls(
            name=data.get("name", "helix_v2"),
            version=str(data.get("version", "2.0.0")),
            vocab_size=int(tok.get("vocab_size", 200000)),
            d_model=int(model["d_model"]),
            n_layers=int(model["n_layers"]),
            n_heads=int(model["n_heads"]),
            d_head=int(model["d_head"]),
            d_ffn=int(model["d_ffn"]),
            activation=str(model.get("activation", "silu")),
            norm_type=str(model.get("norm_type", "rmsnorm")),
            norm_eps=float(model.get("norm_eps", 1e-6)),
            layers=layers,
            ffn=FFNConfig(**(data.get("ffn", {}) or {})),
            moe=MoEConfig(**(data.get("moe", {}) or {})),
            mtp=MTPConfig(**(data.get("mtp", {}) or {})),
            position_encoding=PositionEncodingConfig(**(data.get("position_encoding", {}) or {})),
            init=InitConfig(**(data.get("init", {}) or {})),
            dropout=DropoutConfig(**(data.get("dropout", {}) or {})),
            advanced=AdvancedConfig(**(data.get("advanced", {}) or {})),
            tokenizer_path=tok.get("path"),
        )

    @staticmethod
    def _expand_layers(raw: list[Any], n_layers: int) -> list[LayerConfig]:
        """Expand a compact layer list into one entry per layer.

        Supports two forms so long configs stay readable:

        - plain list of dicts (one per layer)
        - block with ``repeat: N`` and ``config: {...}``, e.g.
          ``{repeat: 16, config: {type: gla, d_state: 128}}``
        """
        out: list[LayerConfig] = []
        for item in raw:
            if not isinstance(item, dict):
                raise ValueError(f"Invalid layer entry: {item!r}")
            if "repeat" in item:
                n = int(item["repeat"])
                cfg = item.get("config") or {k: v for k, v in item.items() if k != "repeat"}
                out.extend(LayerConfig(**cfg) for _ in range(n))
            else:
                out.append(LayerConfig(**item))
        return out

    # ---------------- Helpers ----------------
    def estimate_parameters(self) -> dict[str, int]:
        """Rough parameter count per component — for sanity-checking configs
        before instantiating the model.
        """
        emb = self.vocab_size * self.d_model
        head = 0 if self.advanced.tie_embeddings else self.vocab_size * self.d_model

        # SwiGLU FFN: gate + up + down = 3 * d_model * d_ffn
        ffn_total = self.n_layers * 3 * self.d_model * self.d_ffn

        # Per attention/SSM type — rough
        layers_total = 0
        for lc in self.layers:
            if lc.type == "mamba":
                d_inner = (lc.expand_factor or 2) * self.d_model
                d_state = lc.d_state or 128
                layers_total += (
                    2 * self.d_model * d_inner       # in_proj (x + z)
                    + d_inner * (d_conv := (lc.d_conv or 4))  # conv bias tiny, weight grouped
                    + d_inner * (2 * d_state + d_inner)       # x_proj
                    + d_inner * d_inner              # dt_proj
                    + d_inner * self.d_model         # out_proj
                )
            elif lc.type == "gla":
                layers_total += 5 * self.d_model * self.d_model  # q,k,v,g,o
                layers_total += self.d_model * self.n_heads       # alpha
            elif lc.type == "sparse_attention":
                layers_total += 4 * self.d_model * self.d_model  # q,k,v,o
                if lc.qk_norm:
                    layers_total += 2 * self.d_head              # q_norm + k_norm
            elif lc.type == "plain_attention":
                layers_total += 4 * self.d_model * self.d_model  # q,k,v,o (ablation)
            else:
                raise ValueError(f"Unknown layer type: {lc.type}")

        # Norms: 2 per block + final
        norms_total = (2 * self.n_layers + 1) * self.d_model
        mtp_total = 0
        if self.mtp.enabled:
            # Each MTP head is RMSNorm + d_model->d_model projection. The
            # expensive vocab projection is shared with embeddings/lm_head.
            mtp_total = int(self.mtp.n_heads) * (self.d_model + self.d_model * self.d_model)

        return {
            "embedding": emb,
            "layers": layers_total,
            "ffn": ffn_total,
            "norms": norms_total,
            "mtp": mtp_total,
            "lm_head": head,
            "total": emb + layers_total + ffn_total + norms_total + mtp_total + head,
        }


__all__ = [
    "AdvancedConfig",
    "AuralisConfig",
    "DropoutConfig",
    "FFNConfig",
    "InitConfig",
    "LayerConfig",
    "MoEConfig",
    "MTPConfig",
    "PositionEncodingConfig",
]
