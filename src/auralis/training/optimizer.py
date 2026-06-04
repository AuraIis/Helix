"""Optimizer + LR-scheduler factories.

Weight-decay excludes 1-D parameters (biases, norm weights, embedding) —
standard trick; applying decay to norm scales or embeddings noticeably
hurts convergence.

Schedulers:

- ``cosine``: linear warmup ``warmup_steps`` → cosine decay from peak to
  ``min_lr_ratio * peak``. The canonical decoder-only pretraining choice.
- ``constant_with_warmup``: linear warmup then flat — useful for continued
  pretraining (Phase 2).
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR


def _decay_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """Split parameters into decay / no-decay groups."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 1-D tensors = biases, RMSNorm scales, learned-decay alphas → no decay
        if p.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> Optimizer:
    """Build optimizer from a training-config dict.

    Expected keys (match ``configs/training/phase1_pretrain.yaml``)::

        name: "adamw"
        lr: 3.0e-4
        betas: [0.9, 0.95]
        weight_decay: 0.1
        eps: 1.0e-8
        fused: true            # opt-in CUDA fused kernel (numerically equivalent)

    The ``fused`` flag triggers torch's fused AdamW implementation — single
    CUDA kernel for the whole step. Numerically identical (modulo floating
    point ordering); resume-from-checkpoint between fused and non-fused
    works seamlessly. Skipped silently if not on CUDA or if torch is too
    old to support the kwarg.
    """
    name = cfg.get("name", "adamw").lower()
    lr = float(cfg["lr"])
    betas = tuple(cfg.get("betas", (0.9, 0.95)))
    eps = float(cfg.get("eps", 1e-8))
    weight_decay = float(cfg.get("weight_decay", 0.0))
    fused_requested = bool(cfg.get("fused", False))
    groups = _decay_param_groups(model, weight_decay)

    if name == "adamw":
        # Detect whether torch supports `fused=` and CUDA is available.
        fused_kwargs: dict[str, Any] = {}
        if fused_requested and torch.cuda.is_available():
            try:
                # Probe: cheap construction to verify the kwarg is accepted.
                _probe = AdamW([torch.zeros(1, requires_grad=True, device="cuda")],
                               lr=1e-3, fused=True)
                del _probe
                fused_kwargs["fused"] = True
            except (TypeError, RuntimeError):
                # torch too old, or fused not available for this dtype/device.
                pass
        return AdamW(groups, lr=lr, betas=betas, eps=eps, **fused_kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: Optimizer,
    cfg: dict[str, Any],
    total_steps: int,
) -> LambdaLR:
    """Build LR scheduler from config.

    Expected keys::

        type: "cosine" | "constant_with_warmup"
        warmup_steps: 2000
        min_lr_ratio: 0.1        # only used for cosine
    """
    sched_type = cfg.get("type", "cosine")
    warmup = int(cfg.get("warmup_steps", 0))
    min_ratio = float(cfg.get("min_lr_ratio", 0.1))

    if sched_type == "cosine":
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            progress = (step - warmup) / max(1, total_steps - warmup)
            progress = min(max(progress, 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * cos
    elif sched_type == "constant_with_warmup":
        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            return 1.0
    else:
        raise ValueError(f"Unsupported scheduler: {sched_type}")

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


__all__ = ["build_optimizer", "build_scheduler"]
