"""The curriculum controller: a torch-free state machine.

It consumes a stream of :class:`MetricSnapshot` (one per eval) and decides, after
each, whether to keep training the current stage, advance to the next, hold, or
stop. All the "is it learning?" intelligence lives here and in
:mod:`auralis.adaptive.signals`, so it can be unit-tested without a model.

Metric convention: every metric driven by the controller is **higher-is-better**
(pass-rate, margin). Loss-like metrics must be negated by the caller before being
put in the snapshot (e.g. ``"neg_loss": -loss``). Declare any exceptions in
``higher_is_better`` if you must.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from .signals import detect_regression, is_plateaued, is_stable_above, trend_slope
from .stages import CurriculumSpec, Stage


class DecisionKind(str, Enum):
    CONTINUE = "continue"      # keep training the current stage
    ADVANCE = "advance"        # move to the next stage
    HOLD = "hold"              # pause/freeze; guard tripped but not fatal
    STOP = "stop"              # terminate the run (guard fatal or done badly)
    DONE = "done"             # finished the last stage successfully


@dataclass
class Decision:
    kind: DecisionKind
    reason: str
    from_stage: str
    to_stage: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def should_switch_data(self) -> bool:
        return self.kind == DecisionKind.ADVANCE

    @property
    def is_terminal(self) -> bool:
        return self.kind in (DecisionKind.STOP, DecisionKind.DONE)


@dataclass
class MetricSnapshot:
    """Everything the controller needs for one decision point."""

    step: int                                  # global training step
    stage_step: int                            # steps trained in the current stage
    metrics: dict[str, float] = field(default_factory=dict)


class CurriculumController:
    """Drives a :class:`CurriculumSpec` against a metric stream."""

    def __init__(
        self,
        spec: CurriculumSpec,
        start_stage: int = 0,
        higher_is_better: dict[str, bool] | None = None,
    ) -> None:
        self.spec = spec
        self.stage_index = start_stage
        self.finished = False
        # Per-metric history (only the metrics we have seen). Kept full; the
        # signal helpers slice trailing windows themselves.
        self._history: dict[str, list[float]] = {}
        self._snapshots: list[MetricSnapshot] = []
        self.higher_is_better = higher_is_better or {}
        # Snapshot index at which the current stage began. Mastery/plateau are
        # judged on the *stage-local* trajectory (from here on); the guard looks
        # at the whole run so retention can never silently decay across stages.
        # This assumes the monitor emits the mastery/guard metrics on every eval.
        self._stage_start_index = 0

    # ----------------------------------------------------------------- state
    @property
    def stage(self) -> Stage:
        return self.spec.stages[self.stage_index]

    @property
    def is_last_stage(self) -> bool:
        return self.stage_index >= len(self.spec.stages) - 1

    def history(self, metric: str) -> list[float]:
        return self._history.get(metric, [])

    # --------------------------------------------------------------- update
    def update(self, snap: MetricSnapshot) -> Decision:
        """Record a snapshot and return the next decision."""
        if self.finished:
            return Decision(DecisionKind.DONE, "already finished", self.stage.name)

        self._snapshots.append(snap)
        for k, v in snap.metrics.items():
            self._history.setdefault(k, []).append(float(v))

        stage = self.stage

        # 1) Guard first: never let a regression survive a single eval.
        guard_decision = self._check_guard(stage, snap)
        if guard_decision is not None:
            return guard_decision

        # 2) Respect the minimum dwell time so we don't advance on noise.
        if snap.stage_step < stage.min_steps:
            return self._continue(stage, snap, "below min_steps")

        # 3) Mastery / plateau => advance.
        mastered, why = self._check_mastery(stage)
        if mastered:
            return self._advance(stage, snap, why)

        # 4) Stage timeout => advance (or stop if it's the last stage).
        if snap.stage_step >= stage.max_steps:
            if self.is_last_stage:
                return self._finish(stage, snap, "last stage hit max_steps")
            return self._advance(stage, snap, "max_steps reached (timeout)")

        return self._continue(stage, snap, "training")

    # ------------------------------------------------------------- internals
    def _global_series(self, metric: str) -> list[float]:
        """Whole-run history, normalised to higher-is-better."""
        series = self._history.get(metric, [])
        if self.higher_is_better.get(metric, True):
            return series
        return [-v for v in series]   # normalise to higher-is-better

    def _stage_series(self, metric: str) -> list[float]:
        """Current-stage-only history, normalised to higher-is-better.

        Assumes the metric is present in every snapshot since the stage began,
        which is how the monitor is expected to emit mastery/guard metrics.
        """
        full = self._history.get(metric, [])
        local = full[self._stage_start_index:]
        if self.higher_is_better.get(metric, True):
            return local
        return [-v for v in local]

    def _check_mastery(self, stage: Stage) -> tuple[bool, str]:
        crit = stage.mastery
        series = self._stage_series(crit.metric)
        if not series:
            return False, ""
        stable = is_stable_above(series, crit.threshold, crit.window)
        plateau = is_plateaued(series, crit.patience, crit.min_delta)
        if crit.mode == "stable_above":
            return (stable, "mastered: stable above threshold") if stable else (False, "")
        if crit.mode == "plateau":
            return (plateau, "advanced: metric plateaued") if plateau else (False, "")
        # "either"
        if stable:
            return True, "mastered: stable above threshold"
        if plateau:
            return True, "advanced: metric plateaued (no more gains)"
        return False, ""

    def _check_guard(self, stage: Stage, snap: MetricSnapshot) -> Decision | None:
        guard = stage.guard
        if guard is None:
            return None
        series = self._global_series(guard.metric)
        if not detect_regression(series, guard.max_drop, guard.lookback):
            return None
        reason = (
            f"guard '{guard.metric}' regressed >= {guard.max_drop} "
            f"(now {self._history[guard.metric][-1]:.4f})"
        )
        policy = self.spec.on_guard
        if policy == "stop":
            self.finished = True
            return Decision(DecisionKind.STOP, reason, stage.name, metrics=dict(snap.metrics))
        if policy == "hold":
            return Decision(DecisionKind.HOLD, reason, stage.name, metrics=dict(snap.metrics))
        # rollback => signal STOP-with-rollback intent to the trainer
        self.finished = True
        return Decision(DecisionKind.STOP, "rollback: " + reason, stage.name,
                        metrics=dict(snap.metrics))

    def _continue(self, stage: Stage, snap: MetricSnapshot, why: str) -> Decision:
        return Decision(DecisionKind.CONTINUE, why, stage.name, metrics=dict(snap.metrics))

    def _advance(self, stage: Stage, snap: MetricSnapshot, why: str) -> Decision:
        if self.is_last_stage:
            return self._finish(stage, snap, why + " (last stage)")
        self.stage_index += 1
        # The new stage's mastery trajectory starts after the snapshot that
        # triggered this advance.
        self._stage_start_index = len(self._snapshots)
        nxt = self.spec.stages[self.stage_index]
        return Decision(DecisionKind.ADVANCE, why, stage.name, to_stage=nxt.name,
                        metrics=dict(snap.metrics))

    def _finish(self, stage: Stage, snap: MetricSnapshot, why: str) -> Decision:
        self.finished = True
        return Decision(DecisionKind.DONE, why, stage.name, metrics=dict(snap.metrics))

    # --------------------------------------------------------------- helpers
    def trend(self, metric: str, window: int = 5) -> float:
        """Recent learning rate of a metric (slope over the last ``window``)."""
        series = self._stage_series(metric)
        return trend_slope(series[-window:]) if series else 0.0

    def replay(self, snaps: Iterable[MetricSnapshot]) -> list[Decision]:
        """Convenience for tests: feed many snapshots, collect decisions."""
        return [self.update(s) for s in snaps]


__all__ = ["CurriculumController", "MetricSnapshot", "Decision", "DecisionKind"]
