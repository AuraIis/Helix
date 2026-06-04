"""Introspect which back-end each layer actually uses.

Answers the question "did my env flags actually land in the model?" at
run-start time, without having to wait for OOM / slow speeds to hint.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn


def _mamba_backend(layer: nn.Module) -> str:
    # Our Mamba2Layer wraps either _Mamba2Native or _Mamba2CUDA.
    inner = getattr(layer, "_impl", layer)
    t = type(inner).__name__
    if t.endswith("CUDA"):
        return "mamba_ssm"
    return "native"


def _gla_backend(layer: nn.Module) -> str:
    # The runtime decision is made inside forward(); check module-level flag.
    try:
        from auralis.model.layers import gla_layer
        fla = gla_layer._FLA_AVAILABLE and gla_layer._use_fla(on_cuda=True)
        return "fla" if fla else "native"
    except Exception:
        return "native"


def _sparse_backend(layer: nn.Module) -> str:
    try:
        from auralis.model.layers import sparse_attn_layer
        gt = getattr(layer, "global_tokens", 0) or 0
        flash = sparse_attn_layer._FLASH_AVAILABLE and sparse_attn_layer._use_flash(on_cuda=True, global_tokens=gt)
        return "flash_attn" if flash else "native"
    except Exception:
        return "native"


def describe_model_backends(model: nn.Module) -> dict[str, Any]:
    """Walk HelixBlocks and return {layer_idx: (type, backend)} + counts."""
    per_layer: list[dict[str, str]] = []
    counts: dict[str, int] = {}
    blocks = getattr(model, "blocks", [])
    for idx, block in enumerate(blocks):
        attn = getattr(block, "attn", None)
        lc = getattr(block, "layer_config", None)
        layer_type = getattr(lc, "type", "unknown") if lc else "unknown"
        if layer_type == "mamba":
            backend = _mamba_backend(attn)
        elif layer_type == "gla":
            backend = _gla_backend(attn)
        elif layer_type == "sparse_attention":
            backend = _sparse_backend(attn)
        elif layer_type == "plain_attention":
            # Plain attention uses F.scaled_dot_product_attention which auto-
            # selects flash-attn / memory-efficient kernels when CUDA-available.
            backend = "native"
        else:
            backend = "unknown"
        per_layer.append({"idx": str(idx), "type": layer_type, "backend": backend})
        key = f"{layer_type}:{backend}"
        counts[key] = counts.get(key, 0) + 1
    return {"per_layer": per_layer, "summary": counts}


def format_backend_summary(desc: dict[str, Any]) -> str:
    lines = ["  layer-backend summary:"]
    for k, v in sorted(desc["summary"].items()):
        lines.append(f"    {k:32s} {v:>3d}")
    return "\n".join(lines)


__all__ = ["describe_model_backends", "format_backend_summary"]
