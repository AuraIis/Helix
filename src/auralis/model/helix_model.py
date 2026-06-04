"""Helix v2 main model.

Heterogeneous decoder-only stack: Mamba-2 (local) → GLA (bulk) → Sparse Attention
(long-range). Architecture is entirely driven by an :class:`AuralisConfig`; the
module tree is a pure function of it.

Interfaces match Hugging Face conventions loosely (forward returns a dict with
``logits`` and optional ``loss``) so we can drop this into existing trainers.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

from auralis.model.config import AuralisConfig, LayerConfig
from auralis.model.layers.ffn import build_ffn
from auralis.model.layers.gla_layer import GLALayer
from auralis.model.layers.mamba_layer import Mamba2Layer
from auralis.model.layers.norm import RMSNorm
from auralis.model.layers.plain_attn_layer import PlainAttentionLayer
from auralis.model.layers.sparse_attn_layer import SparseAttentionLayer
from auralis.model.utils.init import scaled_normal_init
from auralis.model.utils.rotary import RotaryEmbedding


def _build_attn_sublayer(config: AuralisConfig, layer_cfg: LayerConfig) -> nn.Module:
    """Instantiate the attention/SSM sub-layer for a block, per layer type."""
    t = layer_cfg.type
    if t == "mamba":
        return Mamba2Layer(
            d_model=config.d_model,
            d_state=layer_cfg.d_state or 128,
            d_conv=layer_cfg.d_conv or 4,
            expand_factor=layer_cfg.expand_factor or 2,
            dt_min=layer_cfg.dt_min or 0.001,
            dt_max=layer_cfg.dt_max or 0.1,
        )
    if t == "gla":
        return GLALayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_head=config.d_head,
            d_state=layer_cfg.d_state,
        )
    if t == "sparse_attention":
        global_tokens = 32 if layer_cfg.global_tokens is None else layer_cfg.global_tokens
        return SparseAttentionLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_head=config.d_head,
            window_size=layer_cfg.window_size or 1024,
            global_tokens=global_tokens,
            use_rope=layer_cfg.use_rope if layer_cfg.use_rope is not None else True,
            qk_norm=bool(layer_cfg.qk_norm),
        )
    if t == "plain_attention":
        return PlainAttentionLayer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_head=config.d_head,
            use_rope=layer_cfg.use_rope if layer_cfg.use_rope is not None else True,
        )
    raise ValueError(f"Unknown layer type: {t!r}")


class HelixBlock(nn.Module):
    """Pre-norm transformer-style block with configurable attention variant."""

    def __init__(self, config: AuralisConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.layer_config: LayerConfig = config.layers[layer_idx]
        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.attn = _build_attn_sublayer(config, self.layer_config)
        self.ffn = build_ffn(config)

    def forward(
        self,
        x: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        # Attention / SSM sub-layer (call signature differs per type)
        t = self.layer_config.type
        if t in ("sparse_attention", "plain_attention"):
            attn_out, _ = self.attn(self.norm1(x), rope=rope)
        else:
            # Mamba & GLA: (x, state) → (out, new_state); we discard new_state
            # in the teacher-forcing (training) path.
            attn_out, _ = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class MTPHead(nn.Module):
    """Small future-token head that reuses the main vocab projection.

    The 200k tokenizer makes independent vocab heads too expensive. This head
    only learns a hidden-space transformation; logits are projected through the
    model's normal tied embedding or lm_head.
    """

    def __init__(self, d_model: int, norm_eps: float):
        super().__init__()
        self.norm = RMSNorm(d_model, eps=norm_eps)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.proj(self.norm(x))


class HelixModel(nn.Module):
    """Helix v2 decoder-only language model."""

    def __init__(self, config: AuralisConfig):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.d_model)

        # RoPE cache is shared by any attention layer that opts into RoPE.
        self._uses_rope = any(
            lc.type in {"sparse_attention", "plain_attention"}
            and (lc.use_rope if lc.use_rope is not None else True)
            for lc in config.layers
        )
        if self._uses_rope and config.position_encoding.type == "rope":
            self.rope = RotaryEmbedding(
                dim=config.d_head,
                max_seq_len=config.position_encoding.max_seq_length,
                theta=config.position_encoding.theta,
            )
        else:
            self.rope = None

        self.blocks = nn.ModuleList(
            [HelixBlock(config, layer_idx=i) for i in range(config.n_layers)]
        )

        # Gradient checkpointing — trades a forward recompute for ~3-5x less
        # activation memory. Driven by config.advanced.gradient_checkpointing;
        # can also be toggled at runtime via gradient_checkpointing_enable().
        self._gradient_checkpointing: bool = bool(config.advanced.gradient_checkpointing)

        self.norm_out = RMSNorm(config.d_model, eps=config.norm_eps)

        # LM head: either a separate linear or tied to embeddings.
        if config.advanced.tie_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.mtp_heads = nn.ModuleList()
        if config.mtp.enabled:
            if config.mtp.n_heads < 1:
                raise ValueError("mtp.n_heads must be >= 1 when mtp.enabled=true")
            if config.mtp.loss_weight <= 0:
                raise ValueError("mtp.loss_weight must be > 0 when mtp.enabled=true")
            self.mtp_heads = nn.ModuleList(
                MTPHead(config.d_model, config.norm_eps)
                for _ in range(int(config.mtp.n_heads))
            )

        self._init_weights()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def _init_weights(self) -> None:
        init = self.config.init
        output_modules: set[nn.Module] = set()
        if self.lm_head is not None:
            output_modules.add(self.lm_head)
        # Output projections inside blocks also get the scaled init — prevents
        # the trick from collapsing for tied-embedding models.
        for blk in self.blocks:
            attn = blk.attn
            if hasattr(attn, "out_proj"):
                output_modules.add(attn.out_proj)
            ffn = blk.ffn
            if hasattr(ffn, "down_proj"):
                output_modules.add(ffn.down_proj)
        scaled_normal_init(
            self,
            std=init.init_std,
            embedding_std=init.embedding_init_std,
            output_modules=output_modules,
            output_scale=init.output_init_scale,
        )
        # Generic init deliberately touches every Linear/Conv. A few sequence
        # layers carry non-generic bias parameterisations (Mamba dt, GLA decay)
        # that must be restored after that pass.
        for blk in self.blocks:
            reset = getattr(blk.attn, "reset_special_parameters", None)
            if callable(reset):
                reset()

    # ------------------------------------------------------------------
    # Gradient checkpointing toggles (HF-style API)
    # ------------------------------------------------------------------
    def gradient_checkpointing_enable(self) -> None:
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self._gradient_checkpointing = False

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self._gradient_checkpointing

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,                               # [B, L]
        labels: torch.Tensor | None = None,                    # [B, L]
    ) -> dict[str, torch.Tensor | None]:
        x = self.embedding(input_ids)
        rope = None
        if self.rope is not None:
            rope = self.rope(input_ids.size(1), device=x.device, dtype=x.dtype)

        # Enable checkpointing only when training (no use during eval/inference).
        use_ckpt = self._gradient_checkpointing and self.training and x.requires_grad
        for block in self.blocks:
            if use_ckpt:
                # use_reentrant=False is the modern non-reentrant autograd path
                # and preserves our custom block signature.
                x = _torch_checkpoint(block, x, rope, use_reentrant=False)
            else:
                x = block(x, rope=rope)

        x = self.norm_out(x)

        logits = self._vocab_projection(x)

        loss = None
        main_loss = None
        mtp_loss = None
        if labels is not None:
            main_loss = self._shift_loss(logits, labels)
            loss = main_loss
            if self.mtp_heads:
                mtp_loss = self._mtp_loss(x, labels)
                if mtp_loss is not None:
                    loss = main_loss + float(self.config.mtp.loss_weight) * mtp_loss
        return {
            "logits": logits,
            "loss": loss,
            "loss_main": main_loss,
            "loss_mtp": mtp_loss,
        }

    def _vocab_projection(self, hidden: torch.Tensor) -> torch.Tensor:
        if self.lm_head is not None:
            return self.lm_head(hidden)
        return F.linear(hidden, self.embedding.weight)

    @staticmethod
    def _shift_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Standard next-token prediction shift with -100 as ignore index.
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def _mtp_loss(self, hidden: torch.Tensor, labels: torch.Tensor) -> torch.Tensor | None:
        losses: list[torch.Tensor] = []
        seq_len = hidden.size(1)
        for head_idx, head in enumerate(self.mtp_heads):
            # Main LM head predicts t+1. MTP head 0 predicts t+2, head 1 t+3, ...
            offset = head_idx + 2
            if seq_len <= offset:
                continue
            target = labels[..., offset:].contiguous()
            if not torch.any(target.ne(-100)):
                continue
            mtp_hidden = head(hidden[..., :-offset, :].contiguous())
            mtp_logits = self._vocab_projection(mtp_hidden)
            losses.append(
                F.cross_entropy(
                    mtp_logits.reshape(-1, mtp_logits.size(-1)),
                    target.reshape(-1),
                    ignore_index=-100,
                )
            )
        if not losses:
            return None
        return torch.stack(losses).mean()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_model(config_path: str | Path) -> HelixModel:
    """Load config from YAML and instantiate the model."""
    cfg = AuralisConfig.from_yaml(config_path)
    return HelixModel(cfg)


__all__ = ["HelixBlock", "HelixModel", "build_model"]
