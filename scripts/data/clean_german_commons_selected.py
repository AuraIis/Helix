#!/usr/bin/env python3
"""Clean downloaded German Commons source splits for base pretraining.

The downloader stores compressed JSONL records with metadata. This step reads
those records, applies source-level metadata guards, runs the structure-aware
prose cleaner on the text field only, and writes both per-source and combined
training text outputs.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.download_german_commons_selected import safe_slug
from scripts.data.structure_clean_pretrain import clean_document


@dataclass
class SplitManifest:
    config: str
    split: str
    input_file: str
    output_jsonl: str
    output_text: str
    docs_in: int = 0
    docs_written: int = 0
    bytes_written_text: int = 0
    dropped: Counter = field(default_factory=Counter)


def load_plan(plan_path: Path) -> dict[str, Any]:
    return json.loads(plan_path.read_text(encoding="utf-8"))


def iter_targets(plan: dict[str, Any], include_special: bool, include_hard_filter: bool) -> Iterable[dict[str, Any]]:
    yield from plan.get("take_first", [])
    if include_special:
        yield from plan.get("small_specialty", [])
    if include_hard_filter:
        yield from plan.get("hard_filter_candidates", [])


def metadata_reject_reason(row: dict[str, Any], filters: dict[str, Any]) -> str | None:
    ocr_score = row.get("ocr_score")
    min_ocr = filters.get("min_ocr_score")
    if min_ocr is not None and isinstance(ocr_score, (int, float)) and ocr_score < float(min_ocr):
        return "low_ocr_score"

    perplexity = row.get("perplexity")
    max_perplexity = filters.get("max_perplexity")
    if max_perplexity is not None and isinstance(perplexity, (int, float)) and perplexity > float(max_perplexity):
        return "high_perplexity"

    return None


def source_paths(input_root: Path, output_root: Path, config: str, split: str) -> tuple[Path, Path, Path]:
    slug = safe_slug(config, split)
    input_path = input_root / config / f"{slug}.jsonl.gz"
    out_dir = output_root / config
    return (
        input_path,
        out_dir / f"{slug}.clean.jsonl",
        out_dir / f"{slug}.clean.txt",
    )


def clean_split(
    *,
    input_path: Path,
    output_jsonl: Path,
    output_text: Path,
    config: str,
    split: str,
    global_filters: dict[str, Any],
    source_filters: dict[str, Any],
    max_docs: int,
    flush_every: int,
) -> SplitManifest:
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    manifest = SplitManifest(
        config=config,
        split=split,
        input_file=str(input_path),
        output_jsonl=str(output_jsonl),
        output_text=str(output_text),
    )
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_text.parent.mkdir(parents=True, exist_ok=True)

    min_words = int(global_filters.get("min_words", 70))
    min_score = float(global_filters.get("min_quality_score", 0.62))
    min_language_signal = float(global_filters.get("min_language_signal", 0.06))

    with gzip.open(input_path, "rt", encoding="utf-8", errors="replace") as src, output_jsonl.open(
        "w", encoding="utf-8", newline="\n"
    ) as jsonl, output_text.open("w", encoding="utf-8", newline="\n") as text_out:
        for line in src:
            manifest.docs_in += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                manifest.dropped["bad_json"] += 1
                continue

            reason = metadata_reject_reason(row, source_filters)
            if reason:
                manifest.dropped[reason] += 1
                continue

            doc, reason = clean_document(
                str(row.get("text") or ""),
                min_words=min_words,
                min_score=min_score,
                target_paragraph_chars=650,
                max_paragraph_chars=1100,
                min_language_signal=min_language_signal,
            )
            if doc is None:
                manifest.dropped[reason or "unknown"] += 1
                continue

            record = {
                "text": doc.text,
                "quality_score": doc.score,
                "metrics": doc.metrics,
                "hash": doc.hash,
                "source": row.get("source"),
                "subset": row.get("subset"),
                "license": row.get("license"),
                "num_tokens": row.get("num_tokens"),
                "perplexity": row.get("perplexity"),
                "ocr_score": row.get("ocr_score"),
                "config": config,
                "split": split,
            }
            jsonl.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            clean_text = " ".join(doc.text.split())
            text_out.write(clean_text + "\n")
            manifest.bytes_written_text += len((clean_text + "\n").encode("utf-8"))
            manifest.docs_written += 1

            if flush_every and manifest.docs_written % flush_every == 0:
                jsonl.flush()
                text_out.flush()
                print(
                    f"[progress] {config}/{split} docs_in={manifest.docs_in:,} "
                    f"docs_written={manifest.docs_written:,}",
                    flush=True,
                )
            if max_docs and manifest.docs_written >= max_docs:
                break

    return manifest


def combine_texts(split_manifests: list[SplitManifest], output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    docs = 0
    bytes_written = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        for manifest in split_manifests:
            with Path(manifest.output_text).open("r", encoding="utf-8") as src:
                for line in src:
                    if not line.strip():
                        continue
                    out.write(line)
                    docs += 1
                    bytes_written += len(line.encode("utf-8"))
    return {"output_file": str(output_path), "docs": docs, "bytes": bytes_written}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started_at = time.time()
    plan = load_plan(args.plan)
    global_filters = plan.get("global_filters", {})
    manifests: list[SplitManifest] = []

    for target in iter_targets(plan, args.include_special, args.include_hard_filter):
        config = target["config"]
        split = target["split"]
        input_path, output_jsonl, output_text = source_paths(args.input_root, args.output_root, config, split)
        if args.skip_missing and not input_path.is_file():
            print(f"[skip] {config}/{split} missing: {input_path}", flush=True)
            continue
        source_filters = dict(target.get("filters", {}))
        print(f"[start] {config}/{split}", flush=True)
        manifest = clean_split(
            input_path=input_path,
            output_jsonl=output_jsonl,
            output_text=output_text,
            config=config,
            split=split,
            global_filters=global_filters,
            source_filters=source_filters,
            max_docs=args.max_docs_per_split,
            flush_every=args.flush_every,
        )
        manifests.append(manifest)
        print(
            f"[done] {config}/{split} kept={manifest.docs_written:,}/{manifest.docs_in:,} "
            f"dropped={sum(manifest.dropped.values()):,}",
            flush=True,
        )

    combined = combine_texts(manifests, args.combined_text)
    summary = {
        "dataset": plan.get("dataset", "coral-nlp/german-commons"),
        "plan": str(args.plan),
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "combined": combined,
        "docs_in": sum(item.docs_in for item in manifests),
        "docs_written": sum(item.docs_written for item in manifests),
        "bytes_written_text": sum(item.bytes_written_text for item in manifests),
        "elapsed_seconds": round(time.time() - started_at, 2),
        "splits": [
            {
                **asdict(item),
                "dropped": dict(item.dropped.most_common()),
            }
            for item in manifests
        ],
    }
    args.manifest.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] kept={summary['docs_written']:,}/{summary['docs_in']:,}")
    print(f"[summary] wrote {args.combined_text}")
    print(f"[summary] wrote {args.manifest}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/data/german_commons_clean_plan_v1.json"))
    parser.add_argument("--input-root", type=Path, default=Path("I:/KI/Auralis_datasets/german_commons_selected_raw"))
    parser.add_argument("--output-root", type=Path, default=Path("I:/KI/Auralis_datasets/german_commons_selected_clean"))
    parser.add_argument(
        "--combined-text",
        type=Path,
        default=Path("I:/KI/Auralis_datasets/german_commons_selected_clean/german_commons_selected.clean.txt"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("I:/KI/Auralis_datasets/german_commons_selected_clean/manifest.json"),
    )
    parser.add_argument("--include-special", action="store_true")
    parser.add_argument("--include-hard-filter", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--max-docs-per-split", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=10_000)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    run(args)


if __name__ == "__main__":
    main()
