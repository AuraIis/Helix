"""Reuse Auralis v1 German pretraining data as Phase 1 DE corpus.

v1 already produced a filtered + deduplicated pretraining pool under
``I:/Auralis/NEWGPT/data/german_clean/``. Re-doing that work from scratch
via ``download_german.py`` would waste a day of streaming. This script
converts the existing v1 artifacts to the v2 ``cleaned/german.txt``
format (one whitespace-normalised document per line) on the NAS data-root.

Preset source groups (``--preset``):

- ``deduped_only``  (default): ``all_deduped.jsonl`` — v1's aggregated,
  Perplexity-filtered, MinHash-deduped superset. ~9.8 GB → est. 1.6B
  tokens. Cleanest single source, no hash-dedup needed.

- ``deduped_plus_fineweb2``: adds ``train/downloaded/fineweb2_de/*.txt``
  for extra volume (~16 GB). Activates content-hash dedup to drop
  collisions with ``all_deduped``.

- ``everything``: all DE pretraining jsonl/txt in the v1 tree that we
  can identify as pretrain-shaped (``text`` field, ``source_type``
  ``prose``). Content-hash dedup enabled.

Writes atomically to ``<data_root>/cleaned/german.txt`` and emits a
manifest JSON beside it. Run with ``--dry`` to tally bytes/docs
without touching the NAS.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data._common import (
    atomic_text_writer,
    check_free_space,
    clean_text,
    load_paths,
    now_iso,
)

# Presets: list of (relative_path_under_v1_root, kind). kind is either
# 'jsonl' (field "text") or 'txt' (one doc per line, possibly multi-para).
PRESETS: dict[str, list[tuple[str, str]]] = {
    # Minimal: only the v1-aggregated pool. No dedup needed — already deduped.
    "deduped_only": [
        ("german_clean/stage1_deduped/all_deduped.jsonl", "jsonl"),
    ],
    # Recommended: complementary — all_deduped has Wiki/News/SFT-recycled but
    # NOT FineWeb-2; fineweb2_de.jsonl adds modern web DE. Dedup on.
    "v1_recommended": [
        ("german_clean/stage1_deduped/all_deduped.jsonl", "jsonl"),
        ("german_clean/stage1_collected/fineweb2_de.jsonl", "jsonl"),
    ],
    # Same as recommended plus original FineWeb-2 txt shards (may overlap with
    # the .jsonl — dedup catches it, but wastes read time).
    "v1_plus_fineweb_txt": [
        ("german_clean/stage1_deduped/all_deduped.jsonl", "jsonl"),
        ("german_clean/stage1_collected/fineweb2_de.jsonl", "jsonl"),
        ("train/downloaded/fineweb2_de/", "txt_dir"),
    ],
}


@dataclass
class ReuseStats:
    preset: str
    output_file: str
    sources_read: list[str] = field(default_factory=list)
    docs_written: int = 0
    docs_duplicate: int = 0
    docs_too_short: int = 0
    docs_too_long: int = 0
    bytes_written: int = 0
    started_at: str = ""
    finished_at: str = ""
    dedup_enabled: bool = False
    dry_run: bool = False


def _iter_jsonl_texts(path: Path) -> Iterator[str]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            text = obj.get("text", "") if isinstance(obj, dict) else ""
            if text:
                yield text


def _iter_txt_lines(path: Path) -> Iterator[str]:
    """For v1 .txt shards we treat each non-empty line as a document."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                yield raw


def _iter_txt_dir(dir_path: Path) -> Iterator[str]:
    for f in sorted(dir_path.glob("*.txt")):
        yield from _iter_txt_lines(f)


def _collect_sources(v1_root: Path, preset: str) -> Iterator[tuple[str, Iterator[str]]]:
    for rel, kind in PRESETS[preset]:
        resolved = v1_root / rel
        if kind == "jsonl":
            if not resolved.exists():
                print(f"  warn: missing {resolved}", file=sys.stderr)
                continue
            yield str(resolved), _iter_jsonl_texts(resolved)
        elif kind == "txt_dir":
            if not resolved.exists() or not resolved.is_dir():
                print(f"  warn: missing dir {resolved}", file=sys.stderr)
                continue
            yield str(resolved), _iter_txt_dir(resolved)
        else:
            raise ValueError(f"Unknown kind: {kind}")


def _content_hash(text: str) -> bytes:
    # blake2b on first 500 chars — catches near-duplicates from different
    # v1 stages without needing to hash full 4-kB documents. 16-byte digest.
    return hashlib.blake2b(text[:500].encode("utf-8"), digest_size=16).digest()


def run_reuse(
    v1_root: Path,
    output_path: Path,
    preset: str,
    dedup: bool,
    min_len: int,
    max_len: int,
    dry_run: bool,
) -> ReuseStats:
    stats = ReuseStats(
        preset=preset,
        output_file=str(output_path),
        dedup_enabled=dedup,
        dry_run=dry_run,
        started_at=now_iso(),
    )
    seen: set[bytes] = set() if dedup else set()  # empty when dedup disabled

    def _process(fh) -> None:
        for source_label, iterator in _collect_sources(v1_root, preset):
            stats.sources_read.append(source_label)
            for text in tqdm(iterator, desc=Path(source_label).name, unit="doc"):
                if len(text) < min_len:
                    stats.docs_too_short += 1
                    continue
                if len(text) > max_len:
                    text = text[:max_len]
                    stats.docs_too_long += 1
                cleaned = clean_text(text)
                if not cleaned:
                    stats.docs_too_short += 1
                    continue
                if dedup:
                    h = _content_hash(cleaned)
                    if h in seen:
                        stats.docs_duplicate += 1
                        continue
                    seen.add(h)
                if fh is not None:
                    fh.write(cleaned + "\n")
                stats.docs_written += 1
                stats.bytes_written += len(cleaned.encode("utf-8")) + 1

    if dry_run:
        _process(None)
    else:
        with atomic_text_writer(output_path) as fh:
            _process(fh)

    stats.finished_at = now_iso()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--preset", choices=list(PRESETS), default="deduped_only")
    parser.add_argument("--dedup", action=argparse.BooleanOptionalAction, default=None,
                        help="Content-hash dedup. Default: auto (on unless preset is deduped_only).")
    parser.add_argument("--min-len", type=int, default=300)
    parser.add_argument("--max-len", type=int, default=100_000)
    parser.add_argument("--dry", action="store_true",
                        help="Count docs/bytes but write nothing.")
    parser.add_argument("--required-free-gb", type=float, default=15.0)
    args = parser.parse_args()

    cfg = load_paths(args.config) if args.config else load_paths()
    v1_root = Path(cfg["v1_data_root"])
    if not v1_root.exists():
        sys.exit(f"v1_data_root missing: {v1_root}")

    # Default: dedup on for multi-source presets, off for the single deduped source
    dedup = args.dedup if args.dedup is not None else (args.preset != "deduped_only")

    output_path = Path(cfg["_data_root"]) / cfg["cleaned"]["german"]
    if not args.dry:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        check_free_space(output_path.parent, args.required_free_gb)

    print(f"preset     : {args.preset}")
    print(f"dedup      : {dedup}")
    print(f"v1 root    : {v1_root}")
    print(f"output     : {output_path}")
    print(f"dry run    : {args.dry}")
    print()

    stats = run_reuse(
        v1_root=v1_root,
        output_path=output_path,
        preset=args.preset,
        dedup=dedup,
        min_len=args.min_len,
        max_len=args.max_len,
        dry_run=args.dry,
    )

    if not args.dry:
        manifest_path = output_path.with_suffix(".txt.manifest.json")
        manifest_path.write_text(json.dumps(asdict(stats), ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        print(f"\nManifest: {manifest_path}")

    print("\n=== Summary ===")
    print(f"  docs_written   : {stats.docs_written:>10,}")
    print(f"  bytes_written  : {stats.bytes_written/1e9:>10.2f} GB")
    print(f"  docs_duplicate : {stats.docs_duplicate:>10,}")
    print(f"  docs_too_short : {stats.docs_too_short:>10,}")
    print(f"  docs_too_long  : {stats.docs_too_long:>10,}")


if __name__ == "__main__":
    main()
