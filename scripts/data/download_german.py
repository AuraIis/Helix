"""Download German pretraining data for Phase 1.

Sources (see ``Doc/SPEC_DATASETS.md`` §2.2):

- german-commons (filtered, no Gutenberg/DTA) — 3B tokens
- Wikipedia DE 20231101                        — 1B tokens
- OSCAR-2301 DE (quality_warnings = none)      — 1B tokens

v1 lesson: the ``cultural`` subset of german-commons is dominated by
historical German (Gutenberg, DTA) and poisoned the style distribution of
Helix v1. We cap it at ``cultural_keep_ratio`` (default 5%).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data._common import (
    DownloadStats,
    atomic_text_writer,
    check_free_space,
    clean_text,
    load_paths,
    now_iso,
    resolve,
)

# German is less compressible than English by typical tokenizers.
DE_BYTES_PER_TOKEN = 5.0


def _parse_target_overrides(values: list[str] | None) -> dict[str, int]:
    """Parse CLI overrides of the form ``source=123456``."""
    overrides: dict[str, int] = {}
    for raw in values or []:
        if "=" not in raw:
            raise ValueError(f"Invalid target override {raw!r}; expected source=tokens.")
        name, value = raw.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid target override {raw!r}; empty source name.")
        overrides[name] = int(value.replace("_", "").strip())
    return overrides


def _open_streaming(path: str, **kwargs: Any):
    from datasets import load_dataset
    return load_dataset(path, split="train", streaming=True, **kwargs)


def _write_source(
    source_name: str,
    output_path: Path,
    target_tokens: int,
    records: Iterator[tuple[str, dict[str, int]]],
    filters_applied: dict[str, Any],
) -> DownloadStats:
    stats = DownloadStats(
        source=source_name,
        output_file=str(output_path),
        target_tokens=target_tokens,
        estimated_bytes_per_token=DE_BYTES_PER_TOKEN,
        started_at=now_iso(),
        filters_applied=filters_applied,
    )
    target_bytes = stats.target_bytes()
    with atomic_text_writer(output_path) as fh:
        for line, reasons in tqdm(records, desc=source_name, unit="doc"):
            if line is None:
                for reason, count in reasons.items():
                    stats.filtered_reasons[reason] = stats.filtered_reasons.get(reason, 0) + count
                    stats.filtered_total += count
                continue
            fh.write(line + "\n")
            stats.final_bytes += len(line.encode("utf-8")) + 1
            stats.final_docs += 1
            if stats.final_bytes >= target_bytes:
                break
    stats.finished_at = now_iso()
    stats.write_manifest(output_path.with_suffix(output_path.suffix + ".manifest.json"))
    return stats


def _stream_german_commons(filters: dict[str, Any]) -> Iterator[tuple[str, dict[str, int]]]:
    ds = _open_streaming("coral-nlp/german-commons")
    max_ppl = filters["max_perplexity"]
    min_len = filters["min_length"]
    max_len = filters["max_length"]
    cultural_keep = filters["cultural_keep_ratio"]
    excluded = set(filters["excluded_subsets"])
    rng = random.Random(42)  # deterministic sub-sampling of cultural subset

    for ex in ds:
        subset = str(ex.get("subset", "")).lower()
        if subset in excluded:
            yield None, {"excluded_subset": 1}
            continue
        ppl = ex.get("perplexity", 1000) or 1000
        if ppl > max_ppl:
            yield None, {"high_perplexity": 1}
            continue
        if subset in {"cultural", "gutenberg", "dta"}:
            if rng.random() > cultural_keep:
                yield None, {"cultural_subsampled": 1}
                continue

        text = ex.get("text", "") or ""
        if len(text) < min_len:
            yield None, {"too_short": 1}
            continue
        if len(text) > max_len:
            text = text[:max_len]
        yield clean_text(text), {}


def _stream_wikipedia_de(filters: dict[str, Any]) -> Iterator[tuple[str, dict[str, int]]]:
    ds = _open_streaming("wikimedia/wikipedia", name="20231101.de")
    min_len = filters["min_length"]
    for ex in ds:
        text = ex.get("text", "") or ""
        title = ex.get("title", "") or ""
        if "Begriffsklärung" in title or title.startswith("Liste "):
            yield None, {"disambiguation_or_list": 1}
            continue
        if len(text) < max(min_len, 500):
            yield None, {"too_short": 1}
            continue
        yield clean_text(text), {}


def _stream_oscar_de(filters: dict[str, Any]) -> Iterator[tuple[str, dict[str, int]]]:
    ds = _open_streaming("oscar-corpus/OSCAR-2301", name="de")
    min_len = filters["min_length"]
    max_len = filters["max_length"]
    for ex in ds:
        meta = ex.get("meta", {}) or {}
        warnings = meta.get("quality_warnings") or []
        if warnings:
            yield None, {"oscar_quality_warnings": 1}
            continue
        text = ex.get("text", "") or ""
        if len(text) < max(min_len, 400):
            yield None, {"too_short": 1}
            continue
        if len(text) > max_len:
            text = text[:max_len]
        yield clean_text(text), {}


def _stream_fineweb2_de(filters: dict[str, Any]) -> Iterator[tuple[str, dict[str, int]]]:
    """FineWeb-2 German slice (deu_Latn), streamed from Hugging Face."""
    ds = _open_streaming("HuggingFaceFW/fineweb-2", name="deu_Latn")
    min_len = filters["min_length"]
    max_len = filters["max_length"]
    for ex in ds:
        text = ex.get("text", "") or ""
        if len(text) < min_len:
            yield None, {"too_short": 1}
            continue
        if len(text) > max_len:
            yield None, {"too_long": 1}
            continue
        yield clean_text(text), {}


SOURCES: dict[str, Any] = {
    "german_commons": {"stream": _stream_german_commons, "filename": "german_commons.txt"},
    "wikipedia_de":   {"stream": _stream_wikipedia_de,   "filename": "wikipedia_de.txt"},
    "oscar_de":       {"stream": _stream_oscar_de,       "filename": "oscar_de.txt"},
    "fineweb2_de":    {"stream": _stream_fineweb2_de,    "filename": "fineweb2_de.txt"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Phase 1 German pretraining data.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument(
        "--target-tokens-override",
        nargs="*",
        default=None,
        help="Per-source token override(s), e.g. german_commons=4500000000 fineweb2_de=2500000000",
    )
    parser.add_argument("--required-free-gb", type=float, default=25.0)
    args = parser.parse_args()

    cfg = load_paths(args.config) if args.config else load_paths()
    out_dir = resolve(cfg, "raw", "german")
    out_dir.mkdir(parents=True, exist_ok=True)
    check_free_space(out_dir, args.required_free_gb)

    targets = dict(cfg["phase1_targets"]["german"])
    targets.update(_parse_target_overrides(args.target_tokens_override))
    filters = cfg["filters"]["german"]

    summaries: list[DownloadStats] = []
    for name in args.sources:
        spec = SOURCES[name]
        target_tokens = int(targets[name])
        output_path = out_dir / spec["filename"]
        print(f"\n=== {name} → {output_path} (target {target_tokens/1e9:.2f}B tokens) ===")
        stats = _write_source(
            source_name=name,
            output_path=output_path,
            target_tokens=target_tokens,
            records=spec["stream"](filters),
            filters_applied=filters,
        )
        summaries.append(stats)
        print(
            f"  docs={stats.final_docs:,} "
            f"bytes={stats.final_bytes/1e9:.2f}GB "
            f"filtered={stats.filtered_total:,} "
            f"(cultural_dropped={stats.filtered_reasons.get('cultural_subsampled', 0):,})"
        )

    print("\n=== Summary ===")
    for s in summaries:
        print(f"  {s.source:15s} {s.final_docs:>10,} docs  {s.final_bytes/1e9:>6.2f} GB")


if __name__ == "__main__":
    main()
