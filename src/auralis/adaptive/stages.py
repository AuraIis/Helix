"""Curriculum stage definitions and the run-level spec.

A curriculum is an ordered list of :class:`Stage`. Each stage says:

- **what data to train on** (``data`` — an opaque spec the data adapter knows
  how to turn into a batch iterator: raw text, formatted SFT, contrastive, ...),
- **when it is done** (``mastery`` — advance to the next stage when the primary
  metric is stably above a threshold *or* has plateaued),
- **what must not break** (``guard`` — stop/hold if a guard metric regresses),
- **bounds** (``min_steps`` / ``max_steps``) and optional per-stage LR overrides.

This file is torch-free; it only describes the plan. The controller
(:mod:`auralis.adaptive.controller`) executes the plan against a metric stream,
and the trainer (:mod:`auralis.adaptive.trainer`) supplies the data and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class MasteryCriterion:
    """When is a stage 'learned enough' to advance?

    ``metric`` is a key the monitor emits each eval (e.g. ``"stage_primary"`` or
    a specific probe family like ``"margin_capital"``). Modes:

    - ``stable_above``: advance once the metric held >= ``threshold`` for
      ``window`` consecutive evals (true mastery).
    - ``plateau``: advance once the metric stopped improving for ``patience``
      evals by more than ``min_delta`` (we got everything this stage can give).
    - ``either`` (default): advance on whichever fires first. This is the
      "recognise how it learns best" behaviour — push while it climbs, move on
      when it stalls.
    """

    metric: str = "stage_primary"
    mode: str = "either"            # "stable_above" | "plateau" | "either"
    threshold: float = 0.9
    window: int = 3
    patience: int = 4
    min_delta: float = 0.005

    def __post_init__(self) -> None:
        if self.mode not in ("stable_above", "plateau", "either"):
            raise ValueError(f"unknown mastery mode: {self.mode}")


@dataclass
class GuardCriterion:
    """What must not regress while a stage trains.

    If ``metric`` drops by >= ``max_drop`` below its recent peak, the controller
    emits STOP (or HOLD, see the controller's ``on_guard`` policy). The classic
    use is retention: do not let the model forget already-correct facts while
    chasing new ones.
    """

    metric: str = "retention"
    max_drop: float = 0.001        # > 0; retention is a pass-rate in [0, 1]
    lookback: int = 0              # 0 = compare against all history


@dataclass
class Stage:
    """One curriculum phase."""

    name: str
    data: dict[str, Any] = field(default_factory=dict)
    mastery: MasteryCriterion = field(default_factory=MasteryCriterion)
    guard: GuardCriterion | None = None
    min_steps: int = 100
    max_steps: int = 100_000
    eval_every: int | None = None          # override the global cadence
    lr_scale: float = 1.0                  # multiply the base LR for this stage
    seq_length: int | None = None          # per-stage context length (curriculum
                                           # by length, e.g. file -> repo scale);
                                           # None = use the trainer default
    notes: str = ""

    def __post_init__(self) -> None:
        if self.max_steps < self.min_steps:
            raise ValueError(
                f"stage {self.name}: max_steps ({self.max_steps}) < min_steps "
                f"({self.min_steps})"
            )


@dataclass
class CurriculumSpec:
    """The whole adaptive run: stages + global defaults."""

    name: str
    stages: list[Stage]
    eval_every: int = 200                  # global eval cadence (steps)
    default_guard: GuardCriterion | None = field(
        default_factory=lambda: GuardCriterion("retention", 0.001)
    )
    # If a stage has no guard, fall back to default_guard. Set to None to
    # disable guarding by default.
    on_guard: str = "stop"                 # "stop" | "hold" | "rollback"

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("curriculum needs at least one stage")
        names = [s.name for s in self.stages]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate stage names: {names}")
        if self.on_guard not in ("stop", "hold", "rollback"):
            raise ValueError(f"unknown on_guard policy: {self.on_guard}")
        # Materialise the effective guard per stage (stage guard or default).
        for s in self.stages:
            if s.guard is None:
                s.guard = self.default_guard

    # ---------------- YAML loader ----------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "CurriculumSpec":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CurriculumSpec":
        default_guard = _guard_from(data.get("default_guard"), allow_none=True,
                                    fallback=GuardCriterion("retention", 0.001))
        stages = [_stage_from(s) for s in (data.get("stages") or [])]
        return cls(
            name=str(data.get("name", "curriculum")),
            stages=stages,
            eval_every=int(data.get("eval_every", 200)),
            default_guard=default_guard,
            on_guard=str(data.get("on_guard", "stop")),
        )


def _stage_from(d: dict[str, Any]) -> Stage:
    mastery_d = d.get("mastery") or {}
    guard = _guard_from(d.get("guard"), allow_none=True, fallback=None)
    return Stage(
        name=str(d["name"]),
        data=dict(d.get("data") or {}),
        mastery=MasteryCriterion(**mastery_d),
        guard=guard,
        min_steps=int(d.get("min_steps", 100)),
        max_steps=int(d.get("max_steps", 100_000)),
        eval_every=(int(d["eval_every"]) if d.get("eval_every") is not None else None),
        lr_scale=float(d.get("lr_scale", 1.0)),
        seq_length=(int(d["seq_length"]) if d.get("seq_length") is not None else None),
        notes=str(d.get("notes", "")),
    )


def _guard_from(
    d: dict[str, Any] | None,
    allow_none: bool,
    fallback: GuardCriterion | None,
) -> GuardCriterion | None:
    if d is None:
        return fallback if allow_none else GuardCriterion()
    if d is False or d == "off":
        return None
    return GuardCriterion(**d)


__all__ = ["Stage", "MasteryCriterion", "GuardCriterion", "CurriculumSpec"]
