#!/usr/bin/env python3
"""Audit coral-nlp/german-commons by source split.

This is intentionally a bounded sampler: it uses the Hugging Face datasets
server first-rows endpoint instead of downloading the full 258 GB corpus.
The goal is to decide which German Commons sources are worth a full local
sample-clean run and which should be excluded or aggressively filtered.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.data.dataset_market_app import _hf_json, _row_text, _sample_quality  # noqa: E402
from scripts.data.structure_clean_pretrain import clean_document  # noqa: E402


DATASET_ID = "coral-nlp/german-commons"


def median(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return round(statistics.median(clean), 3)


def mean(values: list[float]) -> float | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    return round(sum(clean) / len(clean), 3)


def sample_rows(config: str, split: str, limit: int) -> tuple[list[dict[str, Any]], str]:
    try:
        payload = _hf_json(
            "first-rows",
            {"dataset": DATASET_ID, "config": config, "split": split},
            timeout=25,
        )
    except Exception as exc:
        return [], repr(exc)
    rows = []
    for item in payload.get("rows", [])[:limit]:
        row = item.get("row", item)
        if isinstance(row, dict):
            rows.append(row)
    return rows, ""


def clean_pass(text: str) -> tuple[bool, str | None, float | None]:
    doc, reason = clean_document(
        text,
        min_words=70,
        min_score=0.62,
        target_paragraph_chars=900,
        max_paragraph_chars=1_800,
        min_language_signal=0.06,
    )
    if doc is None:
        return False, reason, None
    return True, None, doc.score


def classify(report: dict[str, Any]) -> tuple[str, str]:
    keep = report["cleaner_keep_rate"]
    sample_keep = report["sample_quality"].get("estimated_keep_rate")
    warnings = " ".join(report["sample_quality"].get("warnings", [])).lower()
    ocr = report.get("median_ocr_score")
    ppl = report.get("median_perplexity")
    split = report["split"]

    if report["sample_count"] == 0:
        return "manual_check", "no samples available"
    if "ocr" in warnings or (ocr is not None and ocr < 75):
        return "hard_filter", "OCR/character noise detected"
    if keep < 0.35:
        return "exclude", "too few samples pass the structure cleaner"
    if split in {"youtubecommons", "wikidiscussions", "onemillionposts", "thestack", "wikiquote"}:
        return "small_or_special", "use only as small specialty share"
    if sample_keep is not None and sample_keep < 0.65:
        return "hard_filter", "low estimated keep rate"
    if ppl is not None and ppl > 1800:
        return "hard_filter", "very high median perplexity"
    if keep >= 0.70:
        return "take", "good first-row cleaner pass rate"
    return "hard_filter", "usable, but needs source-specific filtering"


def audit_split(config: str, split: str, limit: int) -> dict[str, Any]:
    rows, error = sample_rows(config, split, limit)
    samples = [_row_text(row) for row in rows]
    samples = [text for text in samples if text.strip()]
    quality = _sample_quality(samples, goal="base_pretrain", language="de")

    clean_ok = 0
    clean_scores: list[float] = []
    reject_reasons: dict[str, int] = {}
    for text in samples:
        ok, reason, score = clean_pass(text)
        clean_ok += int(ok)
        if score is not None:
            clean_scores.append(score)
        if reason:
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1

    token_counts = [row.get("num_tokens") for row in rows if isinstance(row.get("num_tokens"), (int, float))]
    perplexities = [row.get("perplexity") for row in rows if isinstance(row.get("perplexity"), (int, float))]
    ocr_scores = [row.get("ocr_score") for row in rows if isinstance(row.get("ocr_score"), (int, float))]
    licenses = sorted({str(item) for row in rows for item in (row.get("license") or [])})

    report = {
        "config": config,
        "split": split,
        "sample_count": len(samples),
        "error": error,
        "cleaner_keep_rate": round(clean_ok / max(1, len(samples)), 3),
        "cleaner_avg_score": mean(clean_scores),
        "reject_reasons": dict(sorted(reject_reasons.items(), key=lambda kv: kv[1], reverse=True)),
        "sample_quality": quality,
        "median_tokens": median(token_counts),
        "median_perplexity": median(perplexities),
        "median_ocr_score": median(ocr_scores),
        "licenses": licenses[:8],
        "examples": quality.get("examples", [])[:3],
    }
    decision, reason = classify(report)
    report["decision"] = decision
    report["decision_reason"] = reason
    return report


def markdown(report: dict[str, Any]) -> str:
    rows = report["splits"]
    order = {"take": 0, "small_or_special": 1, "hard_filter": 2, "manual_check": 3, "exclude": 4}
    rows = sorted(rows, key=lambda r: (order.get(r["decision"], 9), r["config"], r["split"]))
    lines = [
        "# German Commons Audit",
        "",
        f"Dataset: `{DATASET_ID}`",
        f"Generated: `{report['generated_at']}`",
        f"Samples per split: `{report['samples_per_split']}`",
        "",
        "## Summary",
        "",
    ]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["decision"]] = counts.get(row["decision"], 0) + 1
    for key in ("take", "small_or_special", "hard_filter", "manual_check", "exclude"):
        if key in counts:
            lines.append(f"- `{key}`: {counts[key]}")
    lines.extend(
        [
            "",
            "## Ranked Splits",
            "",
            "| Decision | Config | Split | Cleaner keep | Est. keep | OCR | PPL | Reason |",
            "|---|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in rows:
        sample_keep = row["sample_quality"].get("estimated_keep_rate")
        lines.append(
            "| {decision} | {config} | {split} | {cleaner_keep_rate:.2f} | {sample_keep} | {ocr} | {ppl} | {reason} |".format(
                decision=row["decision"],
                config=row["config"],
                split=row["split"],
                cleaner_keep_rate=row["cleaner_keep_rate"],
                sample_keep="" if sample_keep is None else sample_keep,
                ocr="" if row["median_ocr_score"] is None else row["median_ocr_score"],
                ppl="" if row["median_perplexity"] is None else row["median_perplexity"],
                reason=row["decision_reason"],
            )
        )
    lines.extend(["", "## Notes", ""])
    lines.append("- `take` still means run the full local cleaner and dedup before tokenizing.")
    lines.append("- `hard_filter` is not always trash; it means use source-specific thresholds like `ocr_score`, `perplexity`, and stricter character-noise checks.")
    lines.append("- This audit uses first-row samples, so it is a fast triage pass, not a full corpus measurement.")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    splits_payload = _hf_json("splits", {"dataset": DATASET_ID}, timeout=25)
    split_rows = [
        row
        for row in splits_payload.get("splits", [])
        if row.get("config") != "default" and (not args.config or row.get("config") in args.config)
    ]
    if args.max_splits:
        split_rows = split_rows[: args.max_splits]

    reports = []
    for index, row in enumerate(split_rows, start=1):
        config = row["config"]
        split = row["split"]
        print(f"[{index:02d}/{len(split_rows):02d}] auditing {config}/{split}", flush=True)
        reports.append(audit_split(config, split, args.samples_per_split))

    payload = {
        "dataset": DATASET_ID,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "samples_per_split": args.samples_per_split,
        "splits": reports,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.write_text(markdown(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-split", type=int, default=12)
    parser.add_argument("--max-splits", type=int, default=0)
    parser.add_argument("--config", action="append", help="Only audit this config; can be repeated.")
    parser.add_argument("--output-json", type=Path, default=Path("data/eval/german_commons_audit.json"))
    parser.add_argument("--output-md", type=Path, default=Path("data/eval/german_commons_audit.md"))
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
