"""Download English pretraining data for Phase 1.

Sources (see ``Doc/SPEC_DATASETS.md`` §2.1):

- FineWeb-Edu sample-10BT   — 10B tokens (primary)
- Wikipedia EN 20231101      — 3B tokens
- SlimPajama subset          — 3B tokens (arxiv/stackexchange/book/wikipedia)
- OpenMathInstruct-2         — 2B tokens (reasoning)

Writes one ``*.txt`` per source (one document per line, whitespace-normalised)
plus a ``*.manifest.json`` next to each file. All paths come from
``configs/data_paths.yaml`` — pass ``--config`` to override.

Streaming mode is used throughout so we never materialise a full source to disk.
Kill with Ctrl-C at any time: atomic writes mean the previous ``.txt`` stays
valid and unchanged — the partial ``.txt.tmp`` is cleaned up.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm

# Make "scripts/data/_common.py" importable when run as a script.
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

# Rough bytes-per-token estimates for EN data (SentencePiece-ish). Used only to
# decide when we've collected enough raw text to cover the token target.
EN_BYTES_PER_TOKEN = 4.0


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


def _open_dataset_streaming(path: str, **kwargs: Any):
    """Lazy import of ``datasets`` so the module imports cheaply for --help.

    The first positional arg is renamed ``path`` to avoid a kwargs collision
    with HF's ``name`` (the config identifier).
    """
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
        estimated_bytes_per_token=EN_BYTES_PER_TOKEN,
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


def _stream_fineweb_edu(filters: dict[str, Any]) -> Iterator[tuple[str, dict[str, int]]]:
    ds = _open_dataset_streaming("HuggingFaceFW/fineweb-edu", name="sample-10BT")
    min_len = filters["min_length"]
    max_len = filters["max_length"]
    min_score = filters["fineweb_min_score"]
    for ex in ds:
        text = ex.get("text", "") or ""
        score = ex.get("score", 0) or 0
        if score < min_score:
            yield None, {"low_education_score": 1}
            continue
        if len(text) < min_len:
            yield None, {"too_short": 1}
            continue
        if len(text) > max_len:
            yield None, {"too_long": 1}
            continue
        yield clean_text(text), {}


def _stream_wikipedia_en(filters: dict[str, Any]) -> Iterator[tuple[str, dict[str, int]]]:
    ds = _open_dataset_streaming("wikimedia/wikipedia", name="20231101.en")
    min_len = filters["min_length"]
    for ex in ds:
        text = ex.get("text", "") or ""
        title = (ex.get("title", "") or "").lower()
        if "disambiguation" in title or title.startswith("list of"):
            yield None, {"disambiguation_or_list": 1}
            continue
        if len(text) < max(min_len, 500):
            yield None, {"too_short": 1}
            continue
        yield clean_text(text), {}


def _stream_dolma(filters: dict[str, Any]) -> Iterator[tuple[str, dict[str, int]]]:
    """AllenAI Dolma — replaces SlimPajama (removed from HF).

    Dolma is a curated multi-source mix (Common Crawl, Wikipedia, books, arXiv,
    StackExchange, code, etc.). We filter by `source` field to stay close to
    what SlimPajama provided: diversity-giving sources *outside* of straight
    web crawl (which FineWeb-Edu already covers).
    """
    ds = _open_dataset_streaming("allenai/dolma")
    # Prefer non-CC sources — FineWeb-Edu already covers general web.
    wanted_substrings = ("wiki", "book", "arxiv", "stack", "peS2o", "pes2o")
    min_len = filters["min_length"]
    max_len = filters["max_length"]
    for ex in ds:
        source = str(ex.get("source", "") or "").lower()
        if source and not any(w in source for w in wanted_substrings):
            yield None, {"source_not_wanted": 1}
            continue
        text = ex.get("text", "") or ""
        if len(text) < min_len:
            yield None, {"too_short": 1}
            continue
        if len(text) > max_len:
            yield None, {"too_long": 1}
            continue
        yield clean_text(text), {}


def _stream_openmath(_filters: dict[str, Any]) -> Iterator[tuple[str, dict[str, int]]]:
    ds = _open_dataset_streaming("nvidia/OpenMathInstruct-2")
    for ex in ds:
        problem = (ex.get("problem", "") or "").strip()
        solution = (ex.get("generated_solution", "") or "").strip()
        if not problem or not solution:
            yield None, {"empty_fields": 1}
            continue
        combined = f"Problem: {problem}\n\nSolution: {solution}"
        yield clean_text(combined), {}


def _stream_fineweb2_en(filters: dict[str, Any]) -> Iterator[tuple[str, dict[str, int]]]:
    """FineWeb-2 eng_Latn subset — modern web EN top-up over FineWeb-Edu.

    Parquet format (no script loader), works on HF datasets v4+. Intended
    to be run ON THE POD (Gigabit HF connection) — too slow over SMB.
    """
    ds = _open_dataset_streaming("HuggingFaceFW/fineweb-2", name="eng_Latn")
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


def _stream_dclm_edu(filters: dict[str, Any]) -> Iterator[tuple[str, dict[str, int]]]:
    """HuggingFaceTB/dclm-edu — education-filtered DCLM text.

    This is a compact high-quality English web/edu source used by small-LM
    recipes. It complements FineWeb-Edu with a different upstream crawl/mix.
    """
    ds = _open_dataset_streaming("HuggingFaceTB/dclm-edu")
    min_len = filters["min_length"]
    max_len = filters["max_length"]
    for ex in ds:
        text = ex.get("text", "") or ex.get("content", "") or ""
        if len(text) < min_len:
            yield None, {"too_short": 1}
            continue
        if len(text) > max_len:
            yield None, {"too_long": 1}
            continue
        yield clean_text(text), {}


SOURCES: dict[str, Any] = {
    "fineweb_edu":    {"stream": _stream_fineweb_edu,   "filename": "fineweb_edu.txt"},
    "wikipedia_en":   {"stream": _stream_wikipedia_en,  "filename": "wikipedia_en.txt"},
    "dolma":          {"stream": _stream_dolma,         "filename": "dolma.txt"},
    "openmath":       {"stream": _stream_openmath,      "filename": "openmath.txt"},
    "fineweb2_en":    {"stream": _stream_fineweb2_en,   "filename": "fineweb2_en.txt"},
    "dclm_edu":       {"stream": _stream_dclm_edu,      "filename": "dclm_edu.txt"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Phase 1 English pretraining data.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--sources", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument(
        "--target-tokens-override",
        nargs="*",
        default=None,
        help="Per-source token override(s), e.g. fineweb2_en=8000000000 openmath=1500000000",
    )
    parser.add_argument("--required-free-gb", type=float, default=50.0,
                        help="Abort if data_root has less free space than this.")
    args = parser.parse_args()

    cfg = load_paths(args.config) if args.config else load_paths()
    out_dir = resolve(cfg, "raw", "english")
    out_dir.mkdir(parents=True, exist_ok=True)
    check_free_space(out_dir, args.required_free_gb)

    targets = dict(cfg["phase1_targets"]["english"])
    targets.update(_parse_target_overrides(args.target_tokens_override))
    filters = cfg["filters"]["english"]

    total_stats: list[DownloadStats] = []
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
        total_stats.append(stats)
        print(
            f"  docs={stats.final_docs:,} "
            f"bytes={stats.final_bytes/1e9:.2f}GB "
            f"filtered={stats.filtered_total:,}"
        )

    print("\n=== Summary ===")
    for s in total_stats:
        print(f"  {s.source:15s} {s.final_docs:>10,} docs  {s.final_bytes/1e9:>6.2f} GB")


if __name__ == "__main__":
    main()
