"""Run a specific baseline tier (smoke / pretrain / chat / domain / style_honesty).

Thin wrapper around :func:`run_baseline.run_baseline` — loads
``eval/baseline_tiers.yaml``, picks the requested tier, optionally merges
its ``extra_questions`` (e.g. the honesty probes) on top of the master
``baseline_questions.yaml``, and writes a tier-specific result JSON.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.eval.run_baseline import run_baseline              # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tier", required=True,
                   choices=["smoke", "pretrain", "chat", "domain", "style_honesty"])
    p.add_argument("--tag", required=True)
    p.add_argument("--tiers-file", type=Path,
                   default=REPO / "eval" / "baseline_tiers.yaml")
    p.add_argument("--questions", type=Path,
                   default=REPO / "eval" / "baseline_questions.yaml")
    p.add_argument("--results-dir", type=Path,
                   default=REPO / "eval" / "results")
    p.add_argument("--dry", action="store_true")
    args = p.parse_args()

    tiers_doc = yaml.safe_load(args.tiers_file.read_text(encoding="utf-8"))
    tier = tiers_doc["tiers"][args.tier]
    master_doc = yaml.safe_load(args.questions.read_text(encoding="utf-8"))
    all_qs = master_doc["questions"]

    # Select question subset
    ids = tier.get("question_ids") or []
    selected = list(all_qs) if not ids else [q for q in all_qs if q["id"] in ids]

    # Append tier-specific extras
    extras = tier.get("extra_questions") or []
    selected.extend(extras)

    # Re-serialise as a temporary questions YAML so run_baseline can consume it
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as fh:
        yaml.safe_dump({"version": master_doc.get("version", 1),
                         "total": len(selected),
                         "questions": selected}, fh, allow_unicode=True)
        tmp_q = Path(fh.name)

    def _dummy_gen(prompt: str) -> str:                        # placeholder
        return "I don't know yet; placeholder."

    try:
        report = run_baseline(
            generator=_dummy_gen if args.dry else _dummy_gen,  # TODO: real model
            questions_path=tmp_q,
            results_dir=args.results_dir,
            tag=f"{args.tier}_{args.tag}",
        )
    finally:
        tmp_q.unlink()

    print(f"Tier {args.tier}: {report['aggregate_score']*100:.1f}% "
          f"over {report['num_questions']} qs")


if __name__ == "__main__":
    main()
