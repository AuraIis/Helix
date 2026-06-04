"""Tests for training-script helpers."""

from __future__ import annotations

from types import SimpleNamespace

from auralis.training.utils import apply_gradient_checkpointing, resolve_gradient_checkpointing


class _DummyModel:
    def __init__(self, default_enabled: bool):
        self.config = SimpleNamespace(
            advanced=SimpleNamespace(gradient_checkpointing=default_enabled)
        )
        self._gradient_checkpointing = default_enabled

    def gradient_checkpointing_enable(self) -> None:
        self._gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self._gradient_checkpointing = False

    @property
    def is_gradient_checkpointing(self) -> bool:
        return self._gradient_checkpointing


def test_resolve_gradient_checkpointing_prefers_explicit_false_override():
    model = _DummyModel(default_enabled=True)
    gc_flag = resolve_gradient_checkpointing(model, {"gradient_checkpointing": False})
    assert gc_flag is False


def test_resolve_gradient_checkpointing_falls_back_to_model_default():
    model = _DummyModel(default_enabled=True)
    gc_flag = resolve_gradient_checkpointing(model, {})
    assert gc_flag is True


def test_apply_gradient_checkpointing_can_disable_model_default():
    model = _DummyModel(default_enabled=True)
    apply_gradient_checkpointing(model, enabled=False)
    assert model.is_gradient_checkpointing is False


def test_apply_gradient_checkpointing_can_enable_model():
    model = _DummyModel(default_enabled=False)
    apply_gradient_checkpointing(model, enabled=True)
    assert model.is_gradient_checkpointing is True
