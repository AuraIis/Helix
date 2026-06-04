"""Tokenize the cleaned Phase-1 corpora into memmap-ready binary files.

For each language configured under ``cleaned`` in ``configs/data_paths.yaml``,
this script concatenates its source files, encodes each document through the
trained Helix v2 SentencePiece model, inserts ``</s>`` between documents, and
writes the result as a flat ``uint32`` stream on the NAS.

Output layout (under ``<data_root>/tokenized/phase1/``):

- ``english.bin`` — flat uint32 token stream (no doc boundaries, <eos> separates)
- ``english.idx`` — int64 pairs ``[offset_tokens, n_tokens]`` per document
- ``english.manifest.json`` — source files, counts, elapsed, SP hash

Same for ``german`` and ``code``. A single pass on the NAS, atomic renames.

Resume-safe: an already-existing ``*.bin`` (not ``.tmp``) is skipped, so you can
re-run the script after restarting the machine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import sentencepiece as spm
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data._common import check_free_space, load_paths, now_iso

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TOKENIZER = REPO_ROOT / "tokenizer" / "helix_v2_tokenizer.model"


@dataclass
class TokenizeStats:
    language: str
    output_bin: str
    output_idx: str
    sources: list[str] = field(default_factory=list)
    documents: int = 0
    tokens: int = 0
    empty_lines: int = 0
    bytes_in: int = 0
    tokenizer_sha256: str = ""
    tokens_per_byte: float = 0.0   # measured from the written bin (for bpb)
    started_at: str = ""
    finished_at: str = ""
    elapsed_seconds: float = 0.0


def _expand_sources(data_root: Path, entries: list[str] | str) -> list[Path]:
    if isinstance(entries, str):
        entries = [entries]
    out: list[Path] = []
    for e in entries:
        p = data_root / e
        if any(ch in e for ch in "*?["):
            out.extend(sorted(data_root.glob(e)))
        elif p.is_file():
            out.append(p)
    return out


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _iter_documents(paths: list[Path]) -> Iterable[tuple[str, str]]:
    """Yield (source_path, text) one document at a time."""
    for p in paths:
        with p.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line:
                    yield str(p), line


def _tokenize_language(
    lang: str,
    sources: list[Path],
    out_bin: Path,
    out_idx: Path,
    sp: spm.SentencePieceProcessor,
    batch_size: int,
    tokenizer_hash: str,
) -> TokenizeStats:
    stats = TokenizeStats(
        language=lang,
        output_bin=str(out_bin),
        output_idx=str(out_idx),
        sources=[str(p) for p in sources],
        started_at=now_iso(),
        tokenizer_sha256=tokenizer_hash,
        bytes_in=sum(p.stat().st_size for p in sources if p.exists()),
    )
    if not sources:
        raise FileNotFoundError(
            f"[{lang}] no source files resolved from config. "
            "Refuse to write an empty tokenized corpus."
        )

    eos = sp.eos_id()
    if eos is None or int(eos) < 0:
        raise ValueError(
            f"[{lang}] tokenizer reports no valid EOS id (got {eos!r}). Appending it and "
            "casting to uint32 would wrap to a huge out-of-vocab token (e.g. -1 -> 4294967295) "
            "and silently corrupt the entire corpus. Export the SentencePiece model with a "
            "defined </s> (eos_id >= 0) before tokenizing."
        )

    out_bin.parent.mkdir(parents=True, exist_ok=True)
    tmp_bin = out_bin.with_suffix(out_bin.suffix + ".tmp")
    tmp_idx = out_idx.with_suffix(out_idx.suffix + ".tmp")

    t0 = time.time()
    buffer: list[str] = []
    offset_tokens = 0

    def flush(f_bin, f_idx):
        nonlocal offset_tokens
        if not buffer:
            return
        # SentencePiece batched encode for throughput
        ids_batch = sp.encode(buffer, out_type=int)
        for ids in ids_batch:
            if not ids:
                stats.empty_lines += 1
                continue
            ids.append(eos)
            arr = np.asarray(ids, dtype=np.uint32)
            arr.tofile(f_bin)
            idx = np.asarray([offset_tokens, len(ids)], dtype=np.int64)
            idx.tofile(f_idx)
            offset_tokens += len(ids)
            stats.tokens += len(ids)
            stats.documents += 1
        buffer.clear()

    total_bytes = stats.bytes_in
    pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc=f"tok {lang}")
    try:
        with tmp_bin.open("wb") as f_bin, tmp_idx.open("wb") as f_idx:
            last_pos = 0
            for src_path, text in _iter_documents(sources):
                buffer.append(text)
                pbar.update(len(text.encode("utf-8")))
                if len(buffer) >= batch_size:
                    flush(f_bin, f_idx)
            flush(f_bin, f_idx)
        os.replace(tmp_bin, out_bin)
        os.replace(tmp_idx, out_idx)
    finally:
        pbar.close()
        for t in (tmp_bin, tmp_idx):
            if t.exists():
                t.unlink()

    # Measure tokens/byte from the middle of the finished bin (the honest value
    # for bits-per-byte — decode tokens back to text, count UTF-8 bytes). The
    # config previously carried a hand-guessed 0.2338 for German which inflated
    # bpb ~30%; record the real number here so configs can use it.
    try:
        mm = np.memmap(out_bin, dtype=np.uint32, mode="r")
        n = int(mm.shape[0])
        take = min(300_000, n)
        lo = max(0, (n - take) // 2)
        sample_ids = [int(x) for x in mm[lo : lo + take]]
        sample_bytes = len(sp.decode(sample_ids).encode("utf-8"))
        stats.tokens_per_byte = round(take / max(1, sample_bytes), 6)
        del mm
    except Exception as exc:  # measurement is diagnostic; never fail the run
        print(f"  warn: tokens/byte measurement failed for {lang}: {exc}")

    stats.elapsed_seconds = round(time.time() - t0, 1)
    stats.finished_at = now_iso()
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-config", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--languages", nargs="+", default=["english", "german", "code"],
                        choices=["english", "german", "code"])
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Lines fed to SP.encode() per call.")
    parser.add_argument("--required-free-gb", type=float, default=100.0)
    parser.add_argument("--force", action="store_true",
                        help="Re-tokenize even if output already exists.")
    parser.add_argument("--output-subdir", default="phase1",
                        help="Subdirectory under <data_root>/tokenized/ to write the "
                             ".bin/.idx/.manifest files into. Default 'phase1'. Use e.g. "
                             "'curated_40b' to keep a phase-1 rollback anchor intact.")
    args = parser.parse_args()

    cfg = load_paths(args.data_config) if args.data_config else load_paths()
    data_root = Path(cfg["_data_root"])
    if not args.tokenizer.exists():
        sys.exit(f"tokenizer missing: {args.tokenizer}")

    tokenizer_hash = _sha256_file(args.tokenizer)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    print(f"tokenizer : {args.tokenizer} (sha={tokenizer_hash[:12]}…)")
    print(f"vocab     : {sp.GetPieceSize():,}")

    out_dir = data_root / "tokenized" / args.output_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    check_free_space(out_dir, args.required_free_gb)
    print(f"output dir: {out_dir}")

    all_stats: list[TokenizeStats] = []
    for lang in args.languages:
        out_bin = out_dir / f"{lang}.bin"
        out_idx = out_dir / f"{lang}.idx"
        if out_bin.exists() and not args.force:
            print(f"\n[{lang}] already tokenized → {out_bin} (pass --force to redo)")
            continue
        print(f"\n[{lang}] tokenizing → {out_bin}")
        sources = _expand_sources(data_root, cfg["cleaned"][lang])
        for s in sources:
            print(f"  - {s}")
        stats = _tokenize_language(
            lang=lang,
            sources=sources,
            out_bin=out_bin,
            out_idx=out_idx,
            sp=sp,
            batch_size=args.batch_size,
            tokenizer_hash=tokenizer_hash,
        )
        manifest_path = out_bin.with_suffix(".bin.manifest.json")
        manifest_path.write_text(
            json.dumps(asdict(stats), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        all_stats.append(stats)
        print(
            f"  {stats.documents:,} docs | {stats.tokens/1e9:.2f} B tokens | "
            f"{stats.elapsed_seconds/60:.1f} min"
        )

    print("\n=== Summary ===")
    for s in all_stats:
        print(f"  {s.language:8s} {s.documents:>12,} docs  {s.tokens/1e9:>6.2f} B tokens")


if __name__ == "__main__":
    main()
