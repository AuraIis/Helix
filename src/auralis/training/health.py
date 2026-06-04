"""Training-health guards.

Detects the four classes of trouble that turn a 3-week pretraining run into a
3-week bill:

- **Gradient explosion**: ``grad_norm`` above threshold for K consecutive steps
- **Gradient collapse**: ``grad_norm`` below ~0 for K steps (dead model)
- **Loss spike**: current loss > ``factor × running_average`` — sudden jump
- **Throughput drop**: ``tokens_per_second`` below ``min_ratio × peak`` sustained

Each guard is configured independently (see ``HealthConfig``). On breach, the
monitor either just flags it (``AlertLevel.WARN``) or requests an immediate
abort (``AlertLevel.STOP``). The trainer checks for STOP after each logged
step and raises ``HealthStop`` if set.

Design note: guards are observational-only — they never mutate model state.
They only read the metrics dict the trainer already produces, so they can be
tested without needing a full training loop.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from auralis.adaptive.signals import detect_regression


class AlertLevel(str, Enum):
    INFO = "info"
    WARN = "warn"
    STOP = "stop"


@dataclass
class HealthConfig:
    # Gradient explosion: grad_norm above this for K consecutive logged windows.
    grad_explosion_threshold: float = 50.0
    grad_explosion_k: int = 3
    # Gradient collapse: grad_norm below this for K consecutive windows.
    grad_collapse_threshold: float = 1e-5
    grad_collapse_k: int = 10
    # Loss spike: current > factor * running_avg, window of last N.
    loss_spike_factor: float = 3.0
    loss_spike_avg_window: int = 20
    # Throughput floor: tps below min_ratio of peak (after warmup).
    tps_min_ratio: float = 0.3
    tps_warmup_logs: int = 5
    tps_k: int = 3
    # Val-loss regression: stop if val rose K consecutive evals.
    val_regression_stop_k: int = 5
    # Per-language bpb regression: lower is better. Disabled by default so
    # legacy configs keep their exact behavior.
    bpb_regression_enabled: bool = False
    bpb_regression_languages: list[str] = field(default_factory=list)
    bpb_regression_max_increase: float = 0.20
    bpb_regression_k: int = 2
    bpb_regression_lookback: int = 4
    bpb_regression_warmup_evals: int = 2
    # VRAM pressure (CUDA only). alloc/total ratio.
    vram_frac_warn: float = 0.95
    vram_frac_stop: float = 0.98
    # Checkpoint write-time anomaly: write > factor * rolling median of last N.
    ckpt_time_factor: float = 3.0
    ckpt_time_median_window: int = 5


@dataclass
class HealthState:
    grad_explosion_count: int = 0
    grad_collapse_count: int = 0
    tps_below_count: int = 0
    tps_peak: float = 0.0
    loss_window: deque[float] = field(default_factory=lambda: deque(maxlen=32))
    ckpt_times: deque[float] = field(default_factory=lambda: deque(maxlen=5))
    bpb_series: dict[str, list[float]] = field(default_factory=dict)
    bpb_regression_counts: dict[str, int] = field(default_factory=dict)
    alerts: list[tuple[int, AlertLevel, str]] = field(default_factory=list)
    stop_requested: bool = False
    stop_reason: str = ""


class HealthStop(RuntimeError):
    """Raised by the trainer when a health guard demands an immediate stop."""


class HealthMonitor:
    """Stateful observer over training metrics. Consumes the same dicts the
    trainer logs (``train/loss``, ``train/grad_norm``, ``train/tokens_per_second``)
    and an optional ``eval/val_loss`` on eval steps.
    """

    def __init__(self, config: HealthConfig | None = None):
        self.config = config or HealthConfig()
        self.state = HealthState()
        self.state.loss_window = deque(maxlen=max(8, self.config.loss_spike_avg_window))
        self.state.ckpt_times = deque(maxlen=max(3, self.config.ckpt_time_median_window))

    # ------------------------------------------------------------------
    def observe(self, metrics: dict[str, float], step: int) -> list[tuple[AlertLevel, str]]:
        """Ingest a metrics dict. Returns any NEW alerts produced by this call."""
        fresh: list[tuple[AlertLevel, str]] = []
        c = self.config

        grad = metrics.get("train/grad_norm")
        if grad is not None:
            if grad > c.grad_explosion_threshold:
                self.state.grad_explosion_count += 1
                if self.state.grad_explosion_count >= c.grad_explosion_k:
                    fresh.append((AlertLevel.STOP,
                                  f"grad_norm={grad:.2f} > {c.grad_explosion_threshold} "
                                  f"for {self.state.grad_explosion_count} windows"))
                    self._request_stop("grad_explosion")
            else:
                self.state.grad_explosion_count = 0

            if grad < c.grad_collapse_threshold:
                self.state.grad_collapse_count += 1
                if self.state.grad_collapse_count >= c.grad_collapse_k:
                    fresh.append((AlertLevel.WARN,
                                  f"grad_norm={grad:.2e} near zero for "
                                  f"{self.state.grad_collapse_count} windows"))
            else:
                self.state.grad_collapse_count = 0

        loss = metrics.get("train/loss")
        if loss is not None:
            if len(self.state.loss_window) >= 4:
                avg = sum(self.state.loss_window) / len(self.state.loss_window)
                if avg > 0 and loss > avg * c.loss_spike_factor:
                    fresh.append((AlertLevel.WARN,
                                  f"loss spike: {loss:.3f} > "
                                  f"{c.loss_spike_factor}× running_avg {avg:.3f}"))
            self.state.loss_window.append(loss)

        tps = metrics.get("train/tokens_per_second")
        if tps is not None:
            self.state.tps_peak = max(self.state.tps_peak, tps)
            if len(self.state.loss_window) >= c.tps_warmup_logs:
                if self.state.tps_peak > 0 and tps < self.state.tps_peak * c.tps_min_ratio:
                    self.state.tps_below_count += 1
                    if self.state.tps_below_count >= c.tps_k:
                        fresh.append((AlertLevel.WARN,
                                      f"tok/s collapsed: {tps:.0f} < "
                                      f"{c.tps_min_ratio:.0%} of peak "
                                      f"{self.state.tps_peak:.0f}"))
                else:
                    self.state.tps_below_count = 0

        for level, msg in fresh:
            self.state.alerts.append((step, level, msg))
        return fresh

    # ------------------------------------------------------------------
    def observe_val(self, val_loss: float, best_val_loss: float, consecutive_increases: int, step: int) -> list[tuple[AlertLevel, str]]:
        """Called by the trainer after each val. Separate path because the
        trainer already owns ``consecutive_val_increases`` — we just decide
        whether it's long enough to stop."""
        fresh: list[tuple[AlertLevel, str]] = []
        if consecutive_increases >= self.config.val_regression_stop_k:
            fresh.append((AlertLevel.STOP,
                          f"val_loss rose {consecutive_increases} evals "
                          f"(current {val_loss:.4f}, best {best_val_loss:.4f})"))
            self._request_stop("val_regression")
        for level, msg in fresh:
            self.state.alerts.append((step, level, msg))
        return fresh

    # ------------------------------------------------------------------
    def observe_bpb(self, metrics: dict[str, float], step: int) -> list[tuple[AlertLevel, str]]:
        """Stop when a per-language bits-per-byte metric regresses.

        BPB is lower-is-better. This catches the multilingual failure mode
        where aggregate val_loss improves while one language quietly gets
        worse.
        """
        fresh: list[tuple[AlertLevel, str]] = []
        c = self.config
        if not c.bpb_regression_enabled:
            return fresh

        available = {
            key.rsplit("/", 1)[-1]: float(value)
            for key, value in metrics.items()
            if key.startswith("eval/bpb/")
        }
        if not available:
            return fresh

        langs = c.bpb_regression_languages or sorted(available)
        for lang in langs:
            if lang not in available:
                continue
            bpb = available[lang]
            series = self.state.bpb_series.setdefault(lang, [])
            series.append(bpb)

            if len(series) <= c.bpb_regression_warmup_evals:
                self.state.bpb_regression_counts[lang] = 0
                continue

            # detect_regression assumes higher-is-better, so negate bpb.
            regressed = detect_regression(
                [-v for v in series],
                max_drop=c.bpb_regression_max_increase,
                lookback=c.bpb_regression_lookback,
            )
            if regressed:
                count = self.state.bpb_regression_counts.get(lang, 0) + 1
                self.state.bpb_regression_counts[lang] = count
                if count >= c.bpb_regression_k:
                    lookback = series[-(c.bpb_regression_lookback + 1):-1]
                    if not lookback:
                        lookback = series[:-1]
                    recent_best = min(lookback) if lookback else min(series[:-1])
                    fresh.append((
                        AlertLevel.STOP,
                        f"bpb/{lang} regressed {count} evals "
                        f"(current {bpb:.4f}, recent_best {recent_best:.4f}, "
                        f"max_increase {c.bpb_regression_max_increase:.4f})",
                    ))
                    self._request_stop(f"bpb_regression:{lang}")
            else:
                self.state.bpb_regression_counts[lang] = 0

        for level, msg in fresh:
            self.state.alerts.append((step, level, msg))
        return fresh

    # ------------------------------------------------------------------
    def observe_vram(self, alloc_gb: float, total_gb: float, step: int) -> list[tuple[AlertLevel, str]]:
        """Allocated/total ratio — fires at 95 % / 98 %."""
        fresh: list[tuple[AlertLevel, str]] = []
        if total_gb <= 0:
            return fresh
        frac = alloc_gb / total_gb
        if frac >= self.config.vram_frac_stop:
            msg = f"VRAM {alloc_gb:.1f}/{total_gb:.1f}GB = {frac*100:.1f}% (stop threshold)"
            fresh.append((AlertLevel.STOP, msg))
            self._request_stop("vram_saturated")
        elif frac >= self.config.vram_frac_warn:
            msg = f"VRAM {alloc_gb:.1f}/{total_gb:.1f}GB = {frac*100:.1f}% (warn threshold)"
            fresh.append((AlertLevel.WARN, msg))
        for level, m in fresh:
            self.state.alerts.append((step, level, m))
        return fresh

    # ------------------------------------------------------------------
    def observe_checkpoint_write(self, seconds: float, step: int) -> list[tuple[AlertLevel, str]]:
        """Catch write-time drift (NFS hiccup, disk pressure)."""
        fresh: list[tuple[AlertLevel, str]] = []
        window = self.state.ckpt_times
        if len(window) >= 3:
            import statistics
            median = statistics.median(window)
            if median > 0 and seconds > self.config.ckpt_time_factor * median:
                fresh.append((AlertLevel.WARN,
                              f"ckpt write {seconds:.1f}s > "
                              f"{self.config.ckpt_time_factor}× median "
                              f"{median:.1f}s (last {len(window)} saves)"))
        window.append(seconds)
        for level, m in fresh:
            self.state.alerts.append((step, level, m))
        return fresh

    # ------------------------------------------------------------------
    def observe_backup(self, ok: bool, fail_count: int, consecutive_fail_threshold: int = 3) -> list[tuple[AlertLevel, str]]:
        """Repeated backup failures ⇒ stop before the checkpoint graveyard."""
        fresh: list[tuple[AlertLevel, str]] = []
        if not ok and fail_count >= consecutive_fail_threshold:
            fresh.append((AlertLevel.STOP,
                          f"{fail_count} consecutive backup failures"))
            self._request_stop("backup_failures")
        return fresh

    # ------------------------------------------------------------------
    def _request_stop(self, reason: str) -> None:
        self.state.stop_requested = True
        if not self.state.stop_reason:
            self.state.stop_reason = reason

    def should_stop(self) -> bool:
        return self.state.stop_requested

    def summary(self) -> dict[str, Any]:
        return {
            "stop_requested": self.state.stop_requested,
            "stop_reason": self.state.stop_reason,
            "n_alerts": len(self.state.alerts),
            "grad_explosion_count": self.state.grad_explosion_count,
            "grad_collapse_count": self.state.grad_collapse_count,
            "bpb_regression_counts": dict(self.state.bpb_regression_counts),
            "bpb_latest": {
                lang: values[-1]
                for lang, values in self.state.bpb_series.items()
                if values
            },
            "tps_peak": self.state.tps_peak,
            "alerts": [
                {"step": s, "level": lvl.value, "msg": m}
                for (s, lvl, m) in self.state.alerts[-20:]
            ],
        }


__all__ = ["AlertLevel", "HealthConfig", "HealthMonitor", "HealthState", "HealthStop"]
