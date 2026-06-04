"""Baseline evaluation runner.

Runs every question in ``eval/baseline_questions.yaml`` against a callable that
takes a prompt string and returns the model's answer, then writes a JSON report
to ``eval/results/<tag>.json`` and prints an aggregate score.

This is the honesty gate: every checkpoint during Phase 1/2/3/4/5 must pass
through here. Never mutate baseline questions to improve scores — only append.

Usage (standalone smoke test against a dummy generator)::

    python scripts/eval/run_baseline.py --tag dry_run --dry
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Question:
    id: str
    category: str
    language: str
    question: str
    expected_keywords: list[str] = field(default_factory=list)
    expected: str | None = None
    tool_required: bool = False
    notes: str | None = None


@dataclass
class Result:
    id: str
    category: str
    language: str
    score: float
    matched: list[str]
    answer: str


def load_questions(path: Path) -> list[Question]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [Question(**{k: v for k, v in q.items() if k in Question.__dataclass_fields__}) for q in data["questions"]]


def score_answer(answer: str, q: Question) -> tuple[float, list[str]]:
    """Simple keyword / exact-substring scoring.

    Returns (score in [0.0, 1.0], matched_tokens).
    """
    lower = answer.lower()
    matched: list[str] = []

    if q.expected is not None:
        if q.expected.lower() in lower:
            matched.append(q.expected)
            return 1.0, matched
        return 0.0, matched

    for kw in q.expected_keywords:
        if kw.lower() in lower:
            matched.append(kw)

    if not q.expected_keywords:
        return 0.0, matched
    return (1.0 if matched else 0.0), matched


def run_baseline(
    generator: Callable[[str], str],
    questions_path: Path,
    results_dir: Path,
    tag: str,
) -> dict[str, Any]:
    questions = load_questions(questions_path)
    results: list[Result] = []
    for q in questions:
        answer = generator(q.question)
        score, matched = score_answer(answer, q)
        results.append(Result(q.id, q.category, q.language, score, matched, answer))

    by_cat: dict[str, list[float]] = {}
    by_lang: dict[str, list[float]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r.score)
        by_lang.setdefault(r.language, []).append(r.score)

    report: dict[str, Any] = {
        "tag": tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "questions_file": str(questions_path),
        "aggregate_score": sum(r.score for r in results) / len(results),
        "by_category": {k: sum(v) / len(v) for k, v in by_cat.items()},
        "by_language": {k: sum(v) / len(v) for k, v in by_lang.items()},
        "num_questions": len(results),
        "results": [asdict(r) for r in results],
    }

    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{tag}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _dummy_generator(prompt: str) -> str:
    """Placeholder generator — always returns a canned string. Used for --dry."""
    return "I don't know yet; I am the Auralis v2 baseline dry-run generator."


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Auralis v2 baseline eval.")
    parser.add_argument("--questions", type=Path, default=Path("eval/baseline_questions.yaml"))
    parser.add_argument("--results-dir", type=Path, default=Path("eval/results"))
    parser.add_argument("--tag", required=True, help="Unique name for this run, e.g. 'phase1_step_10000'.")
    parser.add_argument("--dry", action="store_true", help="Use a dummy generator (no model loaded).")
    args = parser.parse_args()

    if args.dry:
        gen: Callable[[str], str] = _dummy_generator
    else:
        raise SystemExit(
            "No real generator wired yet. Pass --dry for a smoke test, or import run_baseline() "
            "from Python and hand it a real callable."
        )

    report = run_baseline(gen, args.questions, args.results_dir, args.tag)
    print(f"Aggregate score: {report['aggregate_score'] * 100:.1f}% over {report['num_questions']} questions")
    for cat, score in sorted(report["by_category"].items()):
        print(f"  {cat:12s} {score * 100:5.1f}%")


if __name__ == "__main__":
    main()
