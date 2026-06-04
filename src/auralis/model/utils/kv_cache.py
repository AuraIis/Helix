"""KV-cache data structure for autoregressive decoding.

Kept intentionally simple — a dict-of-tensors keyed by ``(layer_idx, kind)``
where ``kind`` is one of ``"k"``, ``"v"``, ``"ssm"``. For training / teacher-
forcing we never touch this; the cache is only built by the inference path.

This file is the *home* for the cache type. Layer implementations consume it
via their ``forward(..., cache=...)`` signature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class KVCache:
    """Per-layer key/value or SSM state store.

    For attention layers: ``k`` and ``v`` grow along dim 1 (seq_len) as the
    model autoregresses. For Mamba layers: ``ssm_state`` replaces K/V.
    """

    attn_k: dict[int, torch.Tensor] = field(default_factory=dict)
    attn_v: dict[int, torch.Tensor] = field(default_factory=dict)
    ssm_state: dict[int, torch.Tensor] = field(default_factory=dict)
    current_len: int = 0

    def reset(self) -> None:
        self.attn_k.clear()
        self.attn_v.clear()
        self.ssm_state.clear()
        self.current_len = 0


__all__ = ["KVCache"]
