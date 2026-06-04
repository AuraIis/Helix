"""Live bridge from adaptive training to the frozen target/retention gate.

The adaptive margin probes are training telemetry. The frozen response gate is
the promotion guard. This module lets a training run periodically generate
answers for the frozen gate and expose split metrics to ``LearningMonitor``:

- ``frozen_target_pass``
- ``frozen_retention_pass``
- ``frozen_promotable``
- ``frozen_target_failures``
- ``frozen_retention_failures``

It is deliberately add-on only: no normal trainer or offline gate behavior is
changed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:
    from .adapters import ModelAdapter, TokenizerAdapter


def _load_gate_helpers():
    """Import the existing gate code lazily.

    ``scripts`` is a namespace package in the repo root when run through the
    adaptive CLI. Keeping this import lazy avoids making the pure controller
    tests depend on script imports.
    """

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from scripts.eval.frozen_response_gate import evaluate_answer, load_probes

    return load_probes, evaluate_answer


def summarize_frozen_results(results: Sequence[dict[str, Any]]) -> dict[str, float]:
    """Turn per-probe frozen-gate results into flat monitor metrics."""

    metrics: dict[str, float] = {}
    promotable = True
    for split in ("target", "retention"):
        rows = [r for r in results if r.get("split") == split]
        total = len(rows)
        passed = sum(1 for r in rows if float(r.get("semantic_score", 0.0)) == 1.0)
        failures = total - passed
        score = passed / total if total else 0.0
        metrics[f"frozen_{split}_pass"] = score
        metrics[f"frozen_{split}_failures"] = float(failures)
        if failures:
            promotable = False
    metrics["frozen_promotable"] = 1.0 if promotable else 0.0
    return metrics


class FrozenGateLiveEvaluator:
    """Callable extra-metrics provider for :class:`LearningMonitor`.

    The evaluator runs greedy generation for each frozen probe, scores the
    answers with the same semantic code as ``frozen_response_gate.py``, writes an
    optional JSONL trace, and returns flat numeric metrics.
    """

    def __init__(
        self,
        model_adapter: ModelAdapter,
        tokenizer: TokenizerAdapter,
        probe_file: str | Path,
        *,
        max_new_tokens: int = 64,
        trace_path: str | Path | None = None,
        every_n_evals: int = 1,
    ) -> None:
        if every_n_evals < 1:
            raise ValueError(f"every_n_evals must be >= 1, got {every_n_evals}")
        load_probes, evaluate_answer = _load_gate_helpers()
        self.ma = model_adapter
        self.tok = tokenizer
        self.probe_file = Path(probe_file)
        self.probes = load_probes(self.probe_file)
        self.evaluate_answer = evaluate_answer
        self.max_new_tokens = max_new_tokens
        self.every_n_evals = every_n_evals
        self._calls = 0
        self._last_metrics: dict[str, float] | None = None
        self.trace_path = Path(trace_path) if trace_path else None
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)

        end_ids = self.tok.encode("<|end|>")
        self.stop_id = end_ids[-1] if end_ids else self.tok.eos_id
        self.stop_ids = {self.tok.eos_id, self.stop_id}
        if end_ids:
            self.stop_ids.add(end_ids[-1])
        self.stop_ids = {i for i in self.stop_ids if i is not None and i >= 0}

    def __call__(self, step: int) -> dict[str, float]:
        self._calls += 1
        should_run = self._last_metrics is None or self._calls % self.every_n_evals == 0
        if not should_run:
            metrics = dict(self._last_metrics or {})
            metrics["frozen_gate_ran"] = 0.0
            return metrics

        results: list[dict[str, Any]] = []
        for probe in self.probes:
            prompt_ids = self.tok.encode(self.tok.chat_prompt(probe.prompt))
            answer = self._generate(prompt_ids)
            results.append(self.evaluate_answer(probe, answer))
        metrics = summarize_frozen_results(results)
        metrics["frozen_gate_ran"] = 1.0
        self._last_metrics = dict(metrics)
        self._write_trace(step, metrics, results)
        return metrics

    def _generate(self, prompt_ids: Sequence[int]) -> str:
        import torch

        from .scoring import greedy_generate

        # greedy_generate accepts one eos id; using the chat <|end|> terminal is
        # enough for this repo's probe style, and eos_id remains a fallback when
        # SentencePiece exposes it as the only terminal.
        with torch.no_grad():
            new_ids = greedy_generate(
                self.ma.model,
                prompt_ids,
                max_new_tokens=self.max_new_tokens,
                eos_id=self.stop_id,
                device=self.ma.device,
            )
        answer = self.tok.decode([i for i in new_ids if i not in self.stop_ids])
        return answer.replace("<|end|>", "").strip()

    def _write_trace(
        self,
        step: int,
        metrics: dict[str, float],
        results: Sequence[dict[str, Any]],
    ) -> None:
        if not self.trace_path:
            return
        payload = {
            "step": step,
            "probe_file": str(self.probe_file),
            "metrics": metrics,
            "results": list(results),
        }
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


__all__ = ["FrozenGateLiveEvaluator", "summarize_frozen_results"]
