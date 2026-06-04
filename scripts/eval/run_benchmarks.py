"""Industry-standard benchmark runner for Auralis.

Complements ``run_baseline.py`` (custom-question keyword scorer) with
HF-dataset-backed canonical benchmarks (HellaSwag, ARC, GSM8K, MMLU-Pro,
BBH, HumanEval, GPQA, plus DE-specific MMLU-DE / GermanQuAD / PAWS-X / XNLI).

Modes:
* multiple-choice (accuracy_mc, accuracy_mc2/mc3/mc4)  — uses log-prob of
  each choice's continuation; no generation needed.
* exact-match (exact_match_number, exact_match_choice)  — generates
  with --max_new_tokens, parses the answer.
* f1_squad — token-level F1 against gold spans.
* pass_at_1 — generates code, runs it in a tests/sandbox subprocess.

Usage::

    # All Phase-1-relevant benchmarks against the current best.pt:
    python scripts/eval/run_benchmarks.py \\
        --tier pretrain \\
        --ckpt /workspace/v2data/checkpoints/phase1_pretrain/best.pt \\
        --tag step18000

    # One specific benchmark, full set:
    python scripts/eval/run_benchmarks.py \\
        --benchmark gsm8k --n_samples -1 \\
        --ckpt path/to/ckpt.pt --tag debug

Results land in ``eval/results/benchmarks/<tag>__<benchmark>.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BENCHMARKS_YAML = REPO_ROOT / "eval" / "benchmarks_v1.yaml"
DEFAULT_RESULTS_DIR = REPO_ROOT / "eval" / "results" / "benchmarks"


# ============================================================================
# Generic generator/scorer interface
# ============================================================================
@dataclass
class GenerateAdapter:
    """Wraps the LLM inference call. The runner passes this around so
    benchmarks don't need to know how the model is loaded.
    """
    generate: Callable[[str, dict[str, Any]], str]              # (prompt, sampling_kwargs) -> text
    score_choice: Callable[[str, str], float]                   # (prompt, continuation) -> log-prob sum
    name: str = "auralis"


@dataclass
class BenchmarkResult:
    benchmark: str
    metric: str
    score: float
    n_samples: int
    n_correct: int
    sampling: dict[str, Any]
    expected_floor: float | None = None
    expected_target: float | None = None
    elapsed_seconds: float = 0.0
    started_at: str = ""
    finished_at: str = ""
    notes: list[str] = field(default_factory=list)


# ============================================================================
# Per-metric scorers — each consumes (predictions, gold_set) → score
# ============================================================================
def _accuracy_mc_scorer(adapter: GenerateAdapter, examples: Iterable[dict],
                         choices_key: str, label_key: str, prompt_fn) -> tuple[int, int]:
    """For each example, score every choice via log-prob, pick argmax.
    Returns (n_correct, n_total)."""
    n_correct = 0
    n_total = 0
    for ex in examples:
        prompt = prompt_fn(ex)
        choices = ex[choices_key]
        if isinstance(choices, dict):
            # ARC-style {"text": [...], "label": [...]}
            texts = choices["text"]
            labels = choices["label"]
        else:
            texts = list(choices)
            labels = list(range(len(choices)))
        scores = [adapter.score_choice(prompt, t) for t in texts]
        pred_idx = max(range(len(scores)), key=lambda i: scores[i])
        gold_label = ex[label_key]
        if str(labels[pred_idx]) == str(gold_label):
            n_correct += 1
        n_total += 1
    return n_correct, n_total


_NUMBER_RE = re.compile(r"-?\d+(?:[.,]\d+)?")


def _exact_match_number_scorer(adapter: GenerateAdapter, examples, prompt_fn,
                                gold_fn, sampling) -> tuple[int, int]:
    """Generate, extract last number, compare with gold."""
    n_correct = 0
    n_total = 0
    for ex in examples:
        prompt = prompt_fn(ex)
        gen = adapter.generate(prompt, sampling)
        gen_nums = _NUMBER_RE.findall(gen.replace(",", ""))
        gen_num = gen_nums[-1] if gen_nums else None
        gold = str(gold_fn(ex)).strip()
        gold_nums = _NUMBER_RE.findall(gold.replace(",", ""))
        gold_num = gold_nums[-1] if gold_nums else gold
        try:
            if gen_num is not None and float(gen_num) == float(gold_num):
                n_correct += 1
        except ValueError:
            pass
        n_total += 1
    return n_correct, n_total


def _f1_squad_scorer(adapter: GenerateAdapter, examples, prompt_fn,
                      gold_answers_fn, sampling) -> tuple[float, int]:
    """Token-level F1 vs gold answers. Returns (sum_f1, n_total)."""
    def _normalize(s: str) -> list[str]:
        s = re.sub(r"[\W_]+", " ", s.lower()).strip()
        return s.split()

    sum_f1 = 0.0
    n_total = 0
    for ex in examples:
        prompt = prompt_fn(ex)
        pred = adapter.generate(prompt, sampling)
        pred_tokens = _normalize(pred)
        best_f1 = 0.0
        for gold in gold_answers_fn(ex):
            gold_tokens = _normalize(gold)
            if not pred_tokens or not gold_tokens:
                continue
            common = set(pred_tokens) & set(gold_tokens)
            if not common:
                continue
            precision = len(common) / len(set(pred_tokens))
            recall = len(common) / len(set(gold_tokens))
            f1 = 2 * precision * recall / (precision + recall)
            best_f1 = max(best_f1, f1)
        sum_f1 += best_f1
        n_total += 1
    return sum_f1, n_total


# ============================================================================
# Benchmark dispatch — declarative registry
# ============================================================================
def _hellaswag_prompt(ex):
    ctx = ex.get("ctx") or ex.get("ctx_a") or ""
    return ctx

def _hellaswag_run(adapter, ds, n):
    examples = list(ds.select(range(min(n, len(ds))))) if n > 0 else list(ds)
    return _accuracy_mc_scorer(adapter, examples, "endings", "label", _hellaswag_prompt)

def _arc_prompt(ex):
    return f"Question: {ex['question']}\nAnswer:"

def _arc_run(adapter, ds, n):
    examples = list(ds.select(range(min(n, len(ds))))) if n > 0 else list(ds)
    return _accuracy_mc_scorer(adapter, examples, "choices", "answerKey", _arc_prompt)

def _winogrande_prompt(ex):
    return ex["sentence"]

def _winogrande_run(adapter, ds, n):
    examples = list(ds.select(range(min(n, len(ds))))) if n > 0 else list(ds)
    n_correct = 0
    n_total = 0
    for ex in examples:
        prompt = _winogrande_prompt(ex)
        s1 = adapter.score_choice(prompt, ex["option1"])
        s2 = adapter.score_choice(prompt, ex["option2"])
        pred = "1" if s1 > s2 else "2"
        if pred == str(ex["answer"]):
            n_correct += 1
        n_total += 1
    return n_correct, n_total

def _gsm8k_prompt(ex):
    return f"Frage: {ex['question']}\nDenke schrittweise und gib die Endantwort an.\nAntwort:"

def _gsm8k_run(adapter, ds, n, sampling):
    examples = list(ds.select(range(min(n, len(ds))))) if n > 0 else list(ds)
    return _exact_match_number_scorer(
        adapter, examples, _gsm8k_prompt,
        gold_fn=lambda ex: ex["answer"].split("####")[-1],
        sampling=sampling,
    )

def _mmlu_prompt(ex):
    choices = ex.get("choices") or [ex.get("option_a", ""), ex.get("option_b", ""),
                                       ex.get("option_c", ""), ex.get("option_d", "")]
    text = f"{ex['question']}\n"
    for i, c in enumerate(choices):
        text += f"{chr(65 + i)}) {c}\n"
    text += "Antwort:"
    return text

def _mmlu_run(adapter, ds, n):
    examples = list(ds.select(range(min(n, len(ds))))) if n > 0 else list(ds)
    n_correct = 0
    n_total = 0
    for ex in examples:
        prompt = _mmlu_prompt(ex)
        choices = ex.get("choices") or [ex.get("option_a"), ex.get("option_b"),
                                          ex.get("option_c"), ex.get("option_d")]
        # Score each letter A/B/C/D continuation
        scores = [adapter.score_choice(prompt, f" {chr(65+i)}") for i in range(len(choices))]
        pred_idx = max(range(len(scores)), key=lambda i: scores[i])
        gold = ex.get("answer") or ex.get("answer_index") or ex.get("label")
        if isinstance(gold, str) and len(gold) == 1 and gold.isalpha():
            gold_idx = ord(gold.upper()) - 65
        else:
            gold_idx = int(gold)
        if pred_idx == gold_idx:
            n_correct += 1
        n_total += 1
    return n_correct, n_total

def _germanquad_run(adapter, ds, n, sampling):
    examples = list(ds.select(range(min(n, len(ds))))) if n > 0 else list(ds)
    return _f1_squad_scorer(
        adapter, examples,
        prompt_fn=lambda ex: f"Kontext: {ex['context']}\nFrage: {ex['question']}\nAntwort:",
        gold_answers_fn=lambda ex: ex["answers"]["text"] if ex.get("answers") else [],
        sampling=sampling,
    )


BENCHMARK_RUNNERS: dict[str, Callable] = {
    "hellaswag":     _hellaswag_run,
    "arc_easy":      _arc_run,
    "arc_challenge": _arc_run,
    "winogrande":    _winogrande_run,
    "gsm8k":         _gsm8k_run,
    "mmlu_pro":      _mmlu_run,
    "mmlu_de":       _mmlu_run,
    "bbh":           _mmlu_run,                 # most BBH tasks are multiple-choice-like
    "humaneval":     None,                       # needs sandbox; placeholder
    "gpqa_diamond":  _mmlu_run,                  # 4-way MC
    "aime_2024":     None,                       # needs careful prompting; placeholder
    "livecodebench": None,                       # needs sandbox; placeholder
    "germanquad":    _germanquad_run,
    "paws_x_de":     _mmlu_run,
    "xnli_de":       _mmlu_run,
}


# ============================================================================
# Driver
# ============================================================================
def load_suite(path: Path = DEFAULT_BENCHMARKS_YAML) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def list_for_tier(suite: dict, tier: str) -> list[str]:
    tiers = suite.get("tiers", {})
    if tier not in tiers:
        raise ValueError(f"unknown tier {tier!r}; have {sorted(tiers.keys())}")
    return list(tiers[tier]["benchmarks"])


def run_one(name: str, cfg: dict, adapter: GenerateAdapter, n_override: int | None = None) -> BenchmarkResult:
    runner = BENCHMARK_RUNNERS.get(name)
    if runner is None:
        return BenchmarkResult(
            benchmark=name, metric=cfg.get("metric", "?"),
            score=0.0, n_samples=0, n_correct=0,
            sampling={}, notes=[f"runner not implemented for {name!r}"],
        )

    print(f"  [{name}] loading {cfg['hf_dataset']!r}...", flush=True)
    from datasets import load_dataset
    load_kwargs: dict[str, Any] = {}
    if "config" in cfg:
        load_kwargs["name"] = cfg["config"]
    ds = load_dataset(cfg["hf_dataset"], split=cfg.get("split", "test"), **load_kwargs)
    n = n_override if n_override is not None else cfg.get("n_samples", 200)

    started = datetime.now(timezone.utc).isoformat()
    t0 = time.time()
    metric = cfg["metric"]

    # Some runners take sampling kwargs (generation), others don't.
    sampling = {
        "temperature": 0.0,
        "max_new_tokens": cfg.get("max_new_tokens", 256),
    }
    if metric.startswith("accuracy_mc") or name in ("winogrande",):
        n_correct, n_total = runner(adapter, ds, n)
        score = n_correct / n_total if n_total else 0.0
    elif metric == "exact_match_number":
        n_correct, n_total = runner(adapter, ds, n, sampling)
        score = n_correct / n_total if n_total else 0.0
    elif metric == "f1_squad":
        sum_f1, n_total = runner(adapter, ds, n, sampling)
        score = sum_f1 / n_total if n_total else 0.0
        n_correct = int(round(sum_f1))
    else:
        return BenchmarkResult(
            benchmark=name, metric=metric, score=0.0, n_samples=0, n_correct=0,
            sampling=sampling, notes=[f"metric {metric!r} not handled"],
        )
    elapsed = time.time() - t0
    return BenchmarkResult(
        benchmark=name, metric=metric, score=score,
        n_samples=n_total, n_correct=n_correct,
        sampling=sampling,
        expected_floor=cfg.get("expected_floor"),
        expected_target=cfg.get("expected_target"),
        elapsed_seconds=elapsed,
        started_at=started, finished_at=datetime.now(timezone.utc).isoformat(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tier", help="run all benchmarks in this tier (smoke|pretrain|post_pretrain|post_sft|frontier)")
    parser.add_argument("--benchmark", help="run a single benchmark by name")
    parser.add_argument("--n_samples", type=int, default=None, help="override n_samples for all benchmarks")
    parser.add_argument("--ckpt", help="path to model checkpoint (loaded by the adapter — not used in --dry mode)")
    parser.add_argument("--tag", required=True, help="run identifier; used in result filenames")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--dry", action="store_true", help="don't load model; use random scorer (sanity-check the runner)")
    parser.add_argument("--suite", type=Path, default=DEFAULT_BENCHMARKS_YAML)
    args = parser.parse_args()

    suite = load_suite(args.suite)

    if args.benchmark:
        benchmarks_to_run = [args.benchmark]
    elif args.tier:
        benchmarks_to_run = list_for_tier(suite, args.tier)
    else:
        sys.exit("--tier or --benchmark required")

    if args.dry:
        # Random adapter for runner-sanity testing without model load.
        import random
        rng = random.Random(0)
        adapter = GenerateAdapter(
            generate=lambda p, k: str(rng.randint(0, 100)),
            score_choice=lambda p, c: rng.random(),
            name="dry-run-random",
        )
    else:
        # Real model load — implementation lives in the calling project; the
        # runner imports it lazily so --dry works without torch installed.
        from auralis.eval.adapter import build_adapter        # type: ignore
        adapter = build_adapter(args.ckpt)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary = {"tag": args.tag, "ckpt": args.ckpt, "results": []}
    for name in benchmarks_to_run:
        cfg = suite["benchmarks"].get(name)
        if cfg is None:
            print(f"  [{name}] WARN unknown benchmark, skipping", flush=True)
            continue
        result = run_one(name, cfg, adapter, n_override=args.n_samples)
        out_path = args.results_dir / f"{args.tag}__{name}.json"
        out_path.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=False), encoding="utf-8")
        summary["results"].append(asdict(result))
        bar = "█" * int(20 * result.score)
        print(f"  [{name}] {result.score:.3f} ({result.n_correct}/{result.n_samples})  {bar}", flush=True)

    summary_path = args.results_dir / f"{args.tag}__summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
