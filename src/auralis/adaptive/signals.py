"""Pure-Python signal processing for learning detection.

These helpers turn a noisy stream of eval metrics (one value per eval step) into
the boolean/numeric judgements the curriculum controller needs:

- has the metric *plateaued* (no more gains to be had from this stage)?
- is it *stably above* a mastery threshold?
- did it *regress* (guard tripped)?
- what is its *trend* (learning fast / slow / stalled)?

Everything here is torch-free and side-effect-free so it can be unit-tested
without a GPU. "Higher is better" is the convention for every metric the
controller drives (pass-rate, margin); for loss-like metrics, negate before
feeding them in (see ``higher_is_better`` in the controller).
"""

from __future__ import annotations

from typing import Sequence


def ema(values: Sequence[float], alpha: float = 0.3) -> list[float]:
    """Exponential moving average. ``alpha`` in (0, 1]; higher = less smoothing."""
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    out: list[float] = []
    acc: float | None = None
    for v in values:
        acc = v if acc is None else alpha * v + (1.0 - alpha) * acc
        out.append(acc)
    return out


def trend_slope(values: Sequence[float]) -> float:
    """Least-squares slope of ``values`` over their integer index.

    Positive => improving (for higher-is-better metrics). Units are
    "metric units per eval step". Returns 0.0 for fewer than 2 points.
    """
    n = len(values)
    if n < 2:
        return 0.0
    xs = range(n)
    mean_x = (n - 1) / 2.0
    mean_y = sum(values) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    den = sum((x - mean_x) ** 2 for x in xs)
    return num / den if den else 0.0


def relative_improvement(values: Sequence[float], window: int) -> float:
    """Improvement of the latest value vs the best in the trailing ``window``.

    Returns latest - max(previous window). Positive => new ground gained.
    Looks at the ``window`` points *before* the latest one.
    """
    if len(values) < 2:
        return 0.0
    latest = values[-1]
    prev = values[-(window + 1):-1] if window > 0 else values[:-1]
    if not prev:
        return 0.0
    return latest - max(prev)


def is_plateaued(values: Sequence[float], patience: int, min_delta: float) -> bool:
    """True when the metric has stopped improving.

    Plateau = over the last ``patience`` evals, no value exceeded the best value
    seen *before* that window by more than ``min_delta``. Intuition: "the stage
    has given us everything it can; advance." Requires at least
    ``patience + 1`` points so there is a baseline to compare against.
    """
    if patience <= 0:
        raise ValueError("patience must be >= 1")
    if len(values) < patience + 1:
        return False
    baseline = max(values[: len(values) - patience])
    recent = values[len(values) - patience:]
    return max(recent) <= baseline + min_delta


def is_stable_above(values: Sequence[float], threshold: float, window: int) -> bool:
    """True when the last ``window`` evals are all >= ``threshold``.

    This is the mastery test: not "touched the threshold once" but "held it".
    """
    if window <= 0:
        raise ValueError("window must be >= 1")
    if len(values) < window:
        return False
    return all(v >= threshold for v in values[-window:])


def detect_regression(
    values: Sequence[float],
    max_drop: float,
    lookback: int = 0,
) -> bool:
    """True when the latest value dropped >= ``max_drop`` below a recent peak.

    ``lookback`` limits the peak search to the trailing N points (0 = all
    history). Use this for guard metrics (retention) and divergence (negated
    loss) so a single bad regression can stop the run.
    """
    if len(values) < 2:
        return False
    history = values[:-1]
    if lookback > 0:
        history = history[-lookback:]
    if not history:
        return False
    peak = max(history)
    return (peak - values[-1]) >= max_drop


__all__ = [
    "ema",
    "trend_slope",
    "relative_improvement",
    "is_plateaued",
    "is_stable_above",
    "detect_regression",
]
