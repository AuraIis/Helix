"""LearningMonitor: turns the model's current state into learning metrics.

Every eval step it computes, from the margin probes and a fixed canary batch, a
flat ``metrics`` dict and wraps it in a :class:`MetricSnapshot` for the
controller. It also appends a JSONL trace line (per-probe margins included, so
you can reconstruct a per-concept "neuro-map" over time) and optionally logs to
Weights & Biases / TensorBoard.

Emitted metrics (all higher-is-better unless prefixed ``neg_``):

- ``target_pass`` / ``retention_pass`` — fraction of probes with positive margin
- ``target_margin_mean`` / ``retention_margin_mean`` — mean margin per split
- ``margin_<family>`` — mean margin per concept family (capital, photo, ...)
- ``canary_loss`` (lower better) and ``neg_canary_loss`` (for the controller)
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Callable, Sequence

import torch

from .adapters import ModelAdapter, TokenizerAdapter
from .controller import MetricSnapshot
from .probes import MarginProbe
from .scoring import canary_loss, margin


class LearningMonitor:
    def __init__(
        self,
        model_adapter: ModelAdapter,
        tokenizer: TokenizerAdapter,
        probes: Sequence[MarginProbe],
        canary_batch: dict[str, torch.Tensor] | None = None,
        trace_path: str | Path | None = None,
        use_wandb: bool = False,
        extra_metrics_fn: Callable[[int], dict[str, float]] | None = None,
    ):
        self.ma = model_adapter
        self.tok = tokenizer
        self.probes = list(probes)
        self.canary_batch = canary_batch
        self.trace_path = Path(trace_path) if trace_path else None
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self.extra_metrics_fn = extra_metrics_fn
        self._wandb = None
        if use_wandb:
            try:
                import wandb

                self._wandb = wandb
            except ImportError:
                self._wandb = None

    # ------------------------------------------------------------------
    def evaluate(self, step: int, stage_step: int, stage_name: str) -> MetricSnapshot:
        per_probe: list[dict] = []
        for p in self.probes:
            prompt_ids, correct_ids, wrong_ids = self.tok.probe_ids(p)
            m = margin(self.ma.model, prompt_ids, correct_ids, wrong_ids, self.ma.device)
            per_probe.append({"id": p.id, "family": p.family, "split": p.split, **m})

        metrics: dict[str, float] = {}
        self._split_metrics(per_probe, "target", metrics)
        self._split_metrics(per_probe, "retention", metrics)
        self._family_metrics(per_probe, metrics)

        if self.canary_batch is not None:
            cl = canary_loss(self.ma.model, self.canary_batch["input_ids"],
                             self.canary_batch["labels"])
            metrics["canary_loss"] = cl
            metrics["neg_canary_loss"] = -cl

        if self.extra_metrics_fn is not None:
            try:
                metrics.update(self.extra_metrics_fn(step))
            except Exception as exc:   # never let optional metrics kill the run
                metrics["extra_metrics_error"] = 1.0
                print(f"[monitor] extra_metrics_fn failed at step {step}: {exc}")

        self._write_trace(step, stage_step, stage_name, metrics, per_probe)
        self._log_wandb(step, metrics)
        self._print(step, stage_name, metrics)
        return MetricSnapshot(step=step, stage_step=stage_step, metrics=metrics)

    # ------------------------------------------------------------------
    @staticmethod
    def _split_metrics(per_probe: list[dict], split: str, out: dict[str, float]) -> None:
        rows = [r for r in per_probe if r["split"] == split]
        if not rows:
            return
        margins = [r["margin"] for r in rows]
        out[f"{split}_pass"] = sum(1 for m in margins if m > 0.0) / len(margins)
        out[f"{split}_margin_mean"] = mean(margins)

    @staticmethod
    def _family_metrics(per_probe: list[dict], out: dict[str, float]) -> None:
        fams: dict[str, list[float]] = {}
        for r in per_probe:
            fams.setdefault(r["family"], []).append(r["margin"])
        for fam, vals in fams.items():
            out[f"margin_{fam}"] = mean(vals)

    def _write_trace(self, step, stage_step, stage_name, metrics, per_probe) -> None:
        if not self.trace_path:
            return
        line = {
            "step": step,
            "stage_step": stage_step,
            "stage": stage_name,
            "metrics": metrics,
            "probes": per_probe,
        }
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")

    def _log_wandb(self, step, metrics) -> None:
        if self._wandb is not None:
            self._wandb.log({f"eval/{k}": v for k, v in metrics.items()}, step=step)

    @staticmethod
    def _print(step, stage_name, metrics) -> None:
        keys = ["target_pass", "retention_pass", "target_margin_mean",
                "retention_margin_mean", "frozen_target_pass",
                "frozen_retention_pass", "canary_loss"]
        parts = [f"{k}={metrics[k]:.3f}" for k in keys if k in metrics]
        print(f"[eval step {step} | {stage_name}] " + " ".join(parts))


__all__ = ["LearningMonitor"]
