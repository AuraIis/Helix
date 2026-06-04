#!/usr/bin/env python3
"""Assemble the strict pretraining-v2 text mix.

This keeps the step explicit and manifest-backed. Each input is copied as
one-document-per-line into a single text file that can be tokenized by the
existing pretraining tokenizer.

The trainer reserves the tail of the token stream for validation. Therefore the
last part of the mixed file must be representative; a naive concatenation would
make validation come only from the last source. This builder holds back a small
tail from every source, interleaves the training section, then shuffles and
appends the held-out tail.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = (
    "data/training/pretrain_clean_v2/german_commons.strict.txt",
    "data/training/pretrain_clean_v2/german.strict.txt",
    "data/training/pretrain_clean_v2/wikipedia_de.strict.txt",
    "data/training/pretrain_clean_v2/openmath.strict.txt",
    "data/training/pretrain_booster_de_v1m.txt",
)


def _source_sizes(sources: list[Path]) -> dict[Path, int]:
    return {src: src.stat().st_size for src in sources}


def _reserve_bytes_by_source(sources: list[Path], val_tail_bytes: int) -> dict[Path, int]:
    sizes = _source_sizes(sources)
    total = sum(sizes.values())
    if total <= 0 or val_tail_bytes <= 0:
        return {src: 0 for src in sources}
    return {
        src: min(sizes[src], max(0, int(val_tail_bytes * (sizes[src] / total))))
        for src in sources
    }


def build_mix(
    sources: list[Path],
    output: Path,
    *,
    val_tail_bytes: int,
    chunk_lines: int,
    seed: int,
) -> dict:
    sizes = _source_sizes(sources)
    reserve = _reserve_bytes_by_source(sources, val_tail_bytes)
    cutoffs = {src: max(0, sizes[src] - reserve[src]) for src in sources}
    stats = {
        src: {
            "path": str(src),
            "input_bytes": sizes[src],
            "reserved_val_bytes_target": reserve[src],
            "train_documents": 0,
            "train_bytes": 0,
            "val_documents": 0,
            "val_bytes": 0,
        }
        for src in sources
    }

    handles = {src: src.open("r", encoding="utf-8") for src in sources}
    bytes_seen = {src: 0 for src in sources}
    active = list(sources)
    val_rows: list[tuple[str, str]] = []

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output.open("w", encoding="utf-8", newline="\n") as out_fh:
            while active:
                next_active: list[Path] = []
                for src in active:
                    fh = handles[src]
                    wrote_or_read = False
                    for _ in range(chunk_lines):
                        line = fh.readline()
                        if not line:
                            break
                        wrote_or_read = True
                        line = line.rstrip("\n")
                        if not line:
                            continue
                        out_line = line + "\n"
                        line_bytes = len(out_line.encode("utf-8"))
                        bytes_seen[src] += line_bytes
                        if bytes_seen[src] <= cutoffs[src]:
                            out_fh.write(out_line)
                            stats[src]["train_documents"] += 1
                            stats[src]["train_bytes"] += line_bytes
                        else:
                            val_rows.append((str(src), out_line))
                            stats[src]["val_documents"] += 1
                            stats[src]["val_bytes"] += line_bytes
                    if wrote_or_read:
                        next_active.append(src)
                active = next_active

            rng = random.Random(seed)
            rng.shuffle(val_rows)
            for _, out_line in val_rows:
                out_fh.write(out_line)
    finally:
        for fh in handles.values():
            fh.close()

    total_train_docs = sum(row["train_documents"] for row in stats.values())
    total_val_docs = sum(row["val_documents"] for row in stats.values())
    total_train_bytes = sum(row["train_bytes"] for row in stats.values())
    total_val_bytes = sum(row["val_bytes"] for row in stats.values())
    return {
        "output_file": str(output),
        "chunk_lines": chunk_lines,
        "seed": seed,
        "val_tail_bytes_target": val_tail_bytes,
        "documents": total_train_docs + total_val_docs,
        "bytes_written": total_train_bytes + total_val_bytes,
        "train_documents": total_train_docs,
        "train_bytes": total_train_bytes,
        "val_tail_documents": total_val_docs,
        "val_tail_bytes": total_val_bytes,
        "sources": list(stats.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=REPO / "data" / "training" / "pretrain_clean_v2" / "mix_full.txt")
    parser.add_argument("--source", action="append", default=None)
    parser.add_argument("--val-tail-bytes", type=int, default=80_000_000)
    parser.add_argument("--chunk-lines", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260509)
    args = parser.parse_args()

    source_args = args.source if args.source else list(DEFAULT_SOURCES)
    sources = [Path(p) if Path(p).is_absolute() else REPO / p for p in source_args]
    missing = [str(p) for p in sources if not p.is_file()]
    if missing:
        raise SystemExit("missing sources:\n" + "\n".join(missing))

    manifest = build_mix(
        sources,
        args.output,
        val_tail_bytes=args.val_tail_bytes,
        chunk_lines=args.chunk_lines,
        seed=args.seed,
    )

    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output} ({manifest['documents']:,} docs)")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
