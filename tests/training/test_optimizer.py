"""Tests for build_optimizer + build_scheduler."""

from __future__ import annotations

import math

import pytest
import torch

from auralis.training.optimizer import build_optimizer, build_scheduler


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(8, 4)
        self.norm = torch.nn.LayerNorm(4)
        self.embed = torch.nn.Embedding(10, 8)
        self.bias = torch.nn.Parameter(torch.zeros(4))


def test_build_optimizer_excludes_1d_params_from_decay():
    model = _TinyModel()
    opt = build_optimizer(model, {"name": "adamw", "lr": 1e-3, "weight_decay": 0.1})
    groups = opt.param_groups
    assert len(groups) == 2
    # group 0 = decay, group 1 = no decay
    assert groups[0]["weight_decay"] == 0.1
    assert groups[1]["weight_decay"] == 0.0

    decay_params = set(id(p) for p in groups[0]["params"])
    no_decay_params = set(id(p) for p in groups[1]["params"])
    # Linear.weight → decay; Linear.bias → no decay; LayerNorm.weight/bias → no decay
    assert id(model.linear.weight) in decay_params
    assert id(model.linear.bias) in no_decay_params
    assert id(model.norm.weight) in no_decay_params
    assert id(model.bias) in no_decay_params


def test_build_optimizer_unknown_raises():
    with pytest.raises(ValueError):
        build_optimizer(_TinyModel(), {"name": "adam_weird", "lr": 1e-3})


def _step_both(opt, sched):
    """Dummy opt.step() before sched.step() to silence PyTorch's order warning."""
    opt.step()
    sched.step()


def test_cosine_scheduler_warmup_then_decay():
    model = _TinyModel()
    opt = build_optimizer(model, {"name": "adamw", "lr": 1.0, "weight_decay": 0.0})
    sched = build_scheduler(
        opt,
        {"type": "cosine", "warmup_steps": 10, "min_lr_ratio": 0.1},
        total_steps=110,
    )
    lrs_warmup = []
    for _ in range(10):
        _step_both(opt, sched)
        lrs_warmup.append(sched.get_last_lr()[0])
    assert lrs_warmup[0] < 0.3
    assert math.isclose(lrs_warmup[-1], 1.0, rel_tol=0.01)
    for _ in range(100):
        _step_both(opt, sched)
    assert math.isclose(sched.get_last_lr()[0], 0.1, abs_tol=0.01)


def test_constant_with_warmup_then_flat():
    model = _TinyModel()
    opt = build_optimizer(model, {"name": "adamw", "lr": 1.0, "weight_decay": 0.0})
    sched = build_scheduler(
        opt,
        {"type": "constant_with_warmup", "warmup_steps": 5},
        total_steps=100,
    )
    for _ in range(5):
        _step_both(opt, sched)
    assert math.isclose(sched.get_last_lr()[0], 1.0, rel_tol=0.01)
    for _ in range(50):
        _step_both(opt, sched)
    assert math.isclose(sched.get_last_lr()[0], 1.0, rel_tol=0.01)
