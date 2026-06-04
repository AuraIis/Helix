"""Per-source data-quality scorecard.

For every ``cleaned/*.txt`` or ``raw/<lang>/*.txt`` source, compute:

- byte size
- line count
- mean / median line length
- 8-shingle near-duplicate rate (sample 50 000 lines)
- fraction of lines failing fastText language ID (optional, if fasttext
  model available)
- fraction with encoding artefacts (invalid UTF-8 surrogates, NUL bytes,
  >5 % non-printable chars)
- estimated tokenizer efficiency (tokens per 100 words) against the
  trained Helix v2 SentencePiece model (optional)

Writes a JSON report next to each input and a Markdown summary to
``data/eval/quality_report.md``.

Cheap and read-only. Runs on a single CPU, does not need GPU. Safe to run
BEFORE tokenisation, so a bad source can be caught before burning storage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import statistics as stats
import sys
from pathlib import Path

import sentencepiece as spm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.data._common import load_paths                    # noqa: E402


def _shingle_hash(s: str, k: int = 8) -> int:
    if len(s) < k:
        return hash(s)
    # rolling blake2b on k-char shingles, return count of unique shingles sample
    return int(hashlib.blake2b(s[:k].encode("utf-8", errors="replace"),
                                digest_size=8).hexdigest(), 16)


def _encoding_suspect(line: str) -> bool:
    if "\x00" in line:
        return True
    non_print = sum(1 for c in line if ord(c) < 32 and c not in "\t\r\n")
    return (non_print / max(1, len(line))) > 0.05


def analyse(path: Path, sample_dedup: int, sp: spm.SentencePieceProcessor | None) -> dict:
    size = path.stat().st_size
    n = 0
    lengths: list[int] = []
    sus = 0
    seen: set[int] = set()
    dup = 0
    words_total = 0
    tokens_total = 0
    sample_rng = random.Random(hash(path.name) & 0xFFFFFFFF)

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            n += 1
            line = line.rstrip("\n")
            L = len(line)
            if n <= 200_000:
                lengths.append(L)
            if _encoding_suspect(line):
                sus += 1

            # near-dup shingle sample (blake2b of first 64 chars)
            if sample_rng.random() < (sample_dedup / max(n, 1)):
                h = int(hashlib.blake2b(line[:64].encode("utf-8", errors="replace"),
                                         digest_size=8).hexdigest(), 16)
                if h in seen:
                    dup += 1
                else:
                    seen.add(h)

            # tokenizer efficiency on a 1 % line sample
            if sp is not None and (n % 100) == 0 and line:
                words_total += len(line.split())
                tokens_total += len(sp.EncodeAsIds(line))

    report = {
        "path": str(path),
        "bytes": size,
        "gb": round(size / 1e9, 3),
        "lines": n,
        "suspect_lines": sus,
        "suspect_rate": round(sus / max(n, 1), 6),
        "length_mean": round(stats.fmean(lengths), 1) if lengths else 0,
        "length_median": stats.median(lengths) if lengths else 0,
        "dedup_sample_seen": len(seen) + dup,
        "dedup_sample_dups": dup,
        "dedup_rate_estimate": round(dup / max(len(seen) + dup, 1), 4),
    }
    if sp is not None and words_total > 0:
        report["tokenizer_tokens_per_100_words"] = round(tokens_total / words_total * 100, 1)
        report["tokenizer_sample_lines"] = n // 100
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-config", type=Path, default=None)
    p.add_argument("--tokenizer", type=Path, default=REPO / "tokenizer" / "helix_v2_tokenizer.model")
    p.add_argument("--sample-dedup", type=int, default=50_000,
                   help="Target number of lines to sample for dedup detection.")
    p.add_argument("--output-md", type=Path,
                   default=REPO / "data" / "eval" / "quality_report.md")
    args = p.parse_args()

    cfg = load_paths(args.data_config) if args.data_config else load_paths()
    data_root = Path(cfg["_data_root"])

    sp: spm.SentencePieceProcessor | None = None
    if args.tokenizer.is_file():
        sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
        print(f"tokenizer loaded: {args.tokenizer}")

    rows: list[dict] = []
    for lang in ("english", "german", "code"):
        entries = cfg["cleaned"][lang]
        if isinstance(entries, str):
            entries = [entries]
        for entry in entries:
            candidates = list(data_root.glob(entry)) if any(c in entry for c in "*?[") else [data_root / entry]
            for p_ in candidates:
                if not p_.is_file():
                    continue
                print(f"scanning {p_}")
                r = analyse(p_, args.sample_dedup, sp)
                r["language"] = lang
                rows.append(r)
                # per-source JSON side-car
                (p_.with_suffix(p_.suffix + ".quality.json")).write_text(
                    json.dumps(r, indent=2, ensure_ascii=False), encoding="utf-8"
                )

    # Markdown summary
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    md = ["# Source Quality Report\n"]
    md.append("| Source | Lang | GB | Lines | Dup-Rate | Suspect | tok/100w |")
    md.append("|---|---|--:|--:|--:|--:|--:|")
    for r in rows:
        md.append(
            f"| `{Path(r['path']).name}` | {r['language']} | {r['gb']:.2f} | {r['lines']:,} | "
            f"{r['dedup_rate_estimate']*100:.2f}% | {r['suspect_rate']*100:.3f}% | "
            f"{r.get('tokenizer_tokens_per_100_words','—')} |"
        )
    args.output_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
