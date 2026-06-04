"""LightEval bridge — runs eval/benchmarks_v1.yaml tiers via LightEval.

Maps the suite definitions in benchmarks_v1.yaml to LightEval's CLI task
strings (format: ``suite|task|fewshot[|truncate]``) and shells out to
``lighteval accelerate``. Result JSONs are read back, parsed, and an
Auralis-format summary is appended to ``eval/results/lighteval/<tag>__summary.json``.

Why this exists: LightEval is the canonical HF eval tool but uses its
own task naming + fewshot conventions. Our benchmarks_v1.yaml is the
project's source of truth for "what to evaluate when". This bridge
keeps our YAML as the human-edited config while letting LightEval do
the heavy lifting.

Usage:
    python scripts/eval/lighteval_bridge.py \\
        --tier pretrain \\
        --model-name HuggingFaceTB/SmolLM2-135M \\
        --tag smoke_smollm
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUITE = REPO_ROOT / "eval" / "benchmarks_v1.yaml"
DEFAULT_RESULTS_DIR = REPO_ROOT / "eval" / "results" / "lighteval"


# Map our benchmark IDs to LightEval task strings.
# Format: "suite|task|fewshot|truncate" — fewshot=0 for zero-shot, truncate=0
# means no input-truncation (LightEval handles model max-length internally).
LIGHTEVAL_TASK_MAP: dict[str, str] = {
    "hellaswag":     "lighteval|hellaswag|0|0",
    "arc_easy":      "lighteval|arc:easy|0|0",
    "arc_challenge": "lighteval|arc:challenge|0|0",
    "winogrande":    "lighteval|winogrande|0|0",
    "gsm8k":         "lighteval|gsm8k|5|0",       # 5-shot is the canonical GSM8K setup
    "mmlu_pro":      "lighteval|mmlu_pro|5|0",
    "bbh":           "lighteval|bbh|3|0",
    "gpqa_diamond":  "lighteval|gpqa:diamond|0|0",
    # The DE-specific and code benchmarks may need lighteval-multilingual or
    # custom task definitions; we include them but mark unsupported until
    # verified.
    "humaneval":     None,                          # may need lighteval-extended
    "aime_2024":     None,
    "livecodebench": None,
    "mmlu_de":       None,                          # multilingual subset, see lighteval/multilingual
    "germanquad":    None,
    "paws_x_de":     None,
    "xnli_de":       None,
}


def load_suite(path: Path = DEFAULT_SUITE) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def benchmarks_for_tier(suite: dict[str, Any], tier: str) -> list[str]:
    tiers = suite.get("tiers", {})
    if tier not in tiers:
        raise SystemExit(f"unknown tier {tier!r}; have {sorted(tiers)}")
    return list(tiers[tier]["benchmarks"])


def resolve_to_lighteval(benchmarks: list[str]) -> tuple[list[str], list[str]]:
    """Returns (supported_task_strings, skipped_benchmarks)."""
    supported, skipped = [], []
    for b in benchmarks:
        task_str = LIGHTEVAL_TASK_MAP.get(b)
        if task_str is None:
            skipped.append(b)
        else:
            supported.append(task_str)
    return supported, skipped


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", help="benchmarks_v1.yaml tier name")
    p.add_argument("--benchmark", help="single benchmark ID (mutually exclusive with --tier)")
    p.add_argument("--model-name", required=True, help="HuggingFace model identifier or local path")
    p.add_argument("--tag", required=True, help="run identifier; appended to result dir")
    p.add_argument("--max-samples", type=int, default=None, help="cap per task (for smoke tests)")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    p.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    p.add_argument("--dry-run", action="store_true", help="print the lighteval command but don't run it")
    args = p.parse_args()

    suite = load_suite(args.suite)
    if args.benchmark:
        benchmarks = [args.benchmark]
    elif args.tier:
        benchmarks = benchmarks_for_tier(suite, args.tier)
    else:
        sys.exit("--tier or --benchmark required")

    supported, skipped = resolve_to_lighteval(benchmarks)
    if not supported:
        sys.exit(f"None of the requested benchmarks have LightEval mappings.\n"
                 f"  requested: {benchmarks}\n"
                 f"  skipped:   {skipped}")
    if skipped:
        print(f"⚠️  Skipping (no LightEval mapping yet): {skipped}", flush=True)

    out_dir = args.results_dir / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "lighteval", "accelerate",
        f"model_name={args.model_name}",
        ",".join(supported),
        "--output-dir", str(out_dir),
    ]
    if args.max_samples is not None:
        cmd += ["--max-samples", str(args.max_samples)]

    print("=" * 60)
    print(f"Tier:          {args.tier or '(single)'}")
    print(f"Model:         {args.model_name}")
    print(f"Tag:           {args.tag}")
    print(f"Benchmarks:    {benchmarks}")
    print(f"  supported:   {supported}")
    print(f"  skipped:     {skipped}")
    print(f"Output dir:    {out_dir}")
    print("=" * 60)
    print(f"Running: {' '.join(cmd)}")
    if args.dry_run:
        return 0

    env = os.environ.copy()
    if Path("/root/.hf_token").exists():
        env["HF_TOKEN"] = Path("/root/.hf_token").read_text().strip()

    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"\nlighteval failed with rc={result.returncode}")
        return result.returncode

    # LightEval writes one JSON per task into out_dir; collect them into a summary.
    summary = {
        "tag": args.tag,
        "model_name": args.model_name,
        "tier": args.tier,
        "supported": supported,
        "skipped": skipped,
        "results": {},
    }
    for json_file in sorted(out_dir.rglob("*.json")):
        try:
            data = json.loads(json_file.read_text())
            summary["results"][json_file.name] = data.get("results", data)
        except Exception:
            continue
    summary_path = out_dir / "_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
