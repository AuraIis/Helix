"""Auralis adaptive curriculum trainer.

A thin, additive orchestration layer on top of the existing Helix pieces
(``HelixModel``, ``MixedDataLoader``, ``build_optimizer``) that adds two things
the plain trainer cannot give you:

1. **Learning observability** — during training it continuously measures whether
   the model is *actually* acquiring the target capabilities, not just lowering
   loss. It tracks, per concept, the teacher-forced margin between the correct
   and the wrong continuation (P(correct) vs P(wrong)), a deterministic canary
   loss, and per-domain loss. See :mod:`auralis.adaptive.monitor`.

2. **Adaptive curriculum** — the run is split into ordered stages (e.g. first
   raw text so the model acquires the knowledge, then formatted/prompted data so
   it learns the chat/instruction surface). The controller advances to the next
   stage *on its own* when the current stage is mastered or has plateaued, and
   stops/holds when a guard metric (e.g. retention) regresses. See
   :mod:`auralis.adaptive.controller`.

Design rule: the decision logic (signals + controller + stages) is pure Python
and torch-free, so it is unit-testable without a GPU. Everything that touches
torch (scoring, the training loop) is isolated in :mod:`auralis.adaptive.scoring`,
:mod:`auralis.adaptive.adapters` and :mod:`auralis.adaptive.trainer`.
"""

from __future__ import annotations

from .signals import (
    ema,
    trend_slope,
    is_plateaued,
    is_stable_above,
    detect_regression,
    relative_improvement,
)
from .stages import Stage, MasteryCriterion, GuardCriterion, CurriculumSpec
from .controller import (
    CurriculumController,
    MetricSnapshot,
    Decision,
    DecisionKind,
)
from .frozen_gate import FrozenGateLiveEvaluator, summarize_frozen_results

__all__ = [
    # signals
    "ema",
    "trend_slope",
    "is_plateaued",
    "is_stable_above",
    "detect_regression",
    "relative_improvement",
    # stages
    "Stage",
    "MasteryCriterion",
    "GuardCriterion",
    "CurriculumSpec",
    # controller
    "CurriculumController",
    "MetricSnapshot",
    "Decision",
    "DecisionKind",
    "FrozenGateLiveEvaluator",
    "summarize_frozen_results",
]
