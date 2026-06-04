"""Weight initialization.

Uses the "scaled_normal" scheme from the config:

- Embeddings:      N(0, init_std) with init_std from config.init.embedding_init_std
- Linear layers:   N(0, init_std)
- Output head:     N(0, init_std * output_init_scale) — smaller variance on the
  final projection stabilizes early-training loss (trick from GPT-NeoX / DeepSeek).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def scaled_normal_init(
    module: nn.Module,
    std: float,
    output_modules: set[nn.Module] | None = None,
    output_scale: float = 0.5,
    embedding_std: float | None = None,
) -> None:
    """In-place init of every parameter in ``module``.

    ``output_modules`` is the set of sub-modules treated as "output heads"
    and whose ``weight`` gets ``std * output_scale`` instead of ``std``.
    Passing it explicitly (rather than by name) keeps the init fn free of
    string matching.
    """
    output_modules = output_modules or set()
    embedding_std = embedding_std if embedding_std is not None else std

    for m in module.modules():
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=embedding_std)
        elif isinstance(m, nn.Linear):
            scale = output_scale if m in output_modules else 1.0
            nn.init.normal_(m.weight, mean=0.0, std=std * scale)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv1d):
            # Kaiming fan-in for convs; conv width is already small for Mamba
            nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="linear")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        # RMSNorm / LayerNorm weights/biases stay at their default (ones/zeros).


__all__ = ["scaled_normal_init"]
