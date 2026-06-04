#!/usr/bin/env python3
"""Download and lightly clean public-domain book-style sources.

The goal is a small, auditable book booster, not a blind OCR dump. The script
streams public datasets, keeps only German/English public/open book-like rows,
chunks long books into paragraph-preserving training documents, and rejects
obvious OCR/boilerplate/list/table noise.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


DEFAULT_OUT = Path("/workspace/v2data/data/book_sources_v1")

LANG_MAP = {
    "de": "de",
    "deu": "de",
    "ger": "de",
    "german": "de",
    "Deutsch": "de",
    "German": "de",
    "en": "en",
    "eng": "en",
    "english": "en",
    "English": "en",
}

PROJECT_GUTENBERG_HEADER_RE = re.compile(
    r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    re.I | re.S,
)
PROJECT_GUTENBERG_FOOTER_RE = re.compile(
    r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*",
    re.I | re.S,
)
BOILERPLATE_RE = re.compile(
    r"project gutenberg|distributed proofreading|transcriber's note|"
    r"produced by|end of the project gutenberg|"
    r"copyright status|terms of use|www\.gutenberg\.org",
    re.I,
)
HARD_NOISE_RE = re.compile(
    r"<\s*/?\s*(html|body|div|script|style|table)\b|&(?:amp|lt|gt|quot);|"
    r"https?://|www\.|isbn\s*:|all rights reserved|onlyfans|casino|porn|xxx",
    re.I,
)
TOC_RE = re.compile(r"\b(?:contents|inhalt|inhaltsverzeichnis|chapter\s+[ivxlcdm]+|kapitel\s+[ivxlcdm\d]+)\b", re.I)
LIST_RE = re.compile(r"(^|\n)\s*(?:[-*]|\d{1,3}[.)])\s+", re.M)
OCR_RE = re.compile(r"[�]|Ã.|Â.|â€|ﬁ|ﬂ|[A-Za-zÄÖÜäöüß]-\s+[A-Za-zÄÖÜäöüß]")
OLD_PRINT_RE = re.compile(r"[ſ]|thut|daſs|ſein|ſich|muſs", re.I)
WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")


@dataclass
class SourceStats:
    name: str
    dataset: str
    config: str | None
    split: str
    rows_seen: int = 0
    rows_kept: int = 0
    chunks_written: int = 0
    bytes_text: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    languages: dict[str, int] = field(default_factory=dict)
    titles_sample: list[str] = field(default_factory=list)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def inc(stats: SourceStats, reason: str) -> None:
    stats.skipped[reason] = stats.skipped.get(reason, 0) + 1


def norm_lang(value: Any) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if value is None:
        return None
    return LANG_MAP.get(str(value).strip(), LANG_MAP.get(str(value).strip().lower()))


def clean_book_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = PROJECT_GUTENBERG_HEADER_RE.sub("", text)
    text = PROJECT_GUTENBERG_FOOTER_RE.sub("", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{4,}", "\n\n", text)
    return text.strip()


def too_noisy(text: str, *, min_chars: int, max_chars: int, allow_old_print: bool) -> str | None:
    stripped = text.strip()
    if len(stripped) < min_chars:
        return "too_short"
    if len(stripped) > max_chars:
        return "too_long"
    if HARD_NOISE_RE.search(stripped):
        return "hard_noise"
    if BOILERPLATE_RE.search(stripped) and len(stripped) < 3500:
        return "boilerplate"
    if OCR_RE.search(stripped):
        return "ocr_mojibake"
    if OLD_PRINT_RE.search(stripped) and not allow_old_print:
        return "old_print"
    words = WORD_RE.findall(stripped)
    if len(words) < 80:
        return "too_few_words"
    alpha = sum(1 for c in stripped if c.isalpha()) / max(len(stripped), 1)
    if alpha < 0.55:
        return "low_alpha"
    unique = len(set(w.lower() for w in words)) / max(len(words), 1)
    if unique < 0.18:
        return "repetitive"
    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    if len(lines) > 4:
        list_lines = sum(1 for ln in lines if re.match(r"^(?:[-*]|\d{1,3}[.)])\s+", ln))
        if list_lines / len(lines) > 0.35:
            return "list_heavy"
    if TOC_RE.search(stripped[:1200]) and LIST_RE.search(stripped[:2500]):
        return "toc_or_index"
    return None


def chunk_text(text: str, *, min_chars: int, target_chars: int, max_chars: int) -> Iterable[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    buf: list[str] = []
    n = 0
    for para in paras:
        if len(para) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if not sent.strip():
                    continue
                if n + len(sent) > target_chars and n >= min_chars:
                    yield "\n\n".join(buf)
                    buf, n = [], 0
                buf.append(sent.strip())
                n += len(sent) + 1
            continue
        if n + len(para) > target_chars and n >= min_chars:
            yield "\n\n".join(buf)
            buf, n = [], 0
        buf.append(para)
        n += len(para) + 2
    if n >= min_chars and buf:
        yield "\n\n".join(buf)


def stable_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=12).hexdigest()


def write_record(out_txt, out_jsonl, *, source: str, lang: str, title: str, author: str, license_: str, text: str) -> int:
    one_line = re.sub(r"\s+", " ", text).strip()
    out_txt.write(one_line + "\n")
    rec = {
        "source": source,
        "language": lang,
        "title": title,
        "author": author,
        "license": license_,
        "text": one_line,
    }
    out_jsonl.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len((one_line + "\n").encode("utf-8"))


def row_from_common_pile(row: dict[str, Any]) -> tuple[str | None, str, str, str, str]:
    meta = row.get("metadata") or {}
    if isinstance(meta, str):
        try:
            parsed = ast.literal_eval(meta)
            meta = parsed if isinstance(parsed, dict) else {}
        except Exception:
            meta = {}
    return (
        norm_lang(meta.get("language")),
        str(meta.get("title") or ""),
        "",
        str(meta.get("license") or ""),
        str(row.get("text") or ""),
    )


def row_from_zkeown(row: dict[str, Any]) -> tuple[str | None, str, str, str, str]:
    return (
        norm_lang(row.get("language")),
        str(row.get("title") or ""),
        str(row.get("author") or ""),
        str(row.get("rights") or ""),
        str(row.get("text") or ""),
    )


def row_from_pleias(row: dict[str, Any]) -> tuple[str | None, str, str, str, str]:
    collection = str(row.get("collection") or "")
    title = str(row.get("title") or "")
    creator = str(row.get("creator") or "")
    hay = f"{collection} {title} {creator}".lower()
    # Common Corpus is broad. Keep only obvious book/library-like records here.
    if not re.search(r"gutenberg|book|books|library|biblioth|gallica|hathi|internet archive|open library", hay):
        return None, title, creator, str(row.get("license") or ""), ""
    return (
        norm_lang(row.get("language")),
        title,
        creator,
        str(row.get("license") or ""),
        str(row.get("text") or ""),
    )


def stream_source(
    *,
    stats: SourceStats,
    row_reader,
    out_txt,
    out_jsonl,
    seen_hashes: set[str],
    max_bytes: int,
    args: argparse.Namespace,
) -> SourceStats:
    ds = load_dataset(stats.dataset, stats.config, split=stats.split, streaming=True) if stats.config else load_dataset(stats.dataset, split=stats.split, streaming=True)
    for row in ds:
        stats.rows_seen += 1
        lang, title, author, license_, text = row_reader(row)
        if lang not in args.languages:
            inc(stats, "language")
            continue
        stats.languages[lang] = stats.languages.get(lang, 0) + 1
        text = clean_book_text(text)
        if not text:
            inc(stats, "empty")
            continue
        if title and len(stats.titles_sample) < 20:
            stats.titles_sample.append(title[:180])
        wrote_row = False
        for chunk in chunk_text(text, min_chars=args.min_chars, target_chars=args.target_chars, max_chars=args.max_chars):
            reason = too_noisy(
                chunk,
                min_chars=args.min_chars,
                max_chars=args.max_chars,
                allow_old_print=args.allow_old_print,
            )
            if reason:
                inc(stats, reason)
                continue
            h = stable_hash(chunk)
            if h in seen_hashes:
                inc(stats, "duplicate")
                continue
            seen_hashes.add(h)
            stats.bytes_text += write_record(
                out_txt,
                out_jsonl,
                source=stats.name,
                lang=lang,
                title=title,
                author=author,
                license_=license_,
                text=chunk,
            )
            stats.chunks_written += 1
            wrote_row = True
            if stats.bytes_text >= max_bytes:
                return stats
        if wrote_row:
            stats.rows_kept += 1
        if stats.rows_seen % args.progress_every == 0:
            print(
                f"[progress] {stats.name} rows={stats.rows_seen:,} chunks={stats.chunks_written:,} "
                f"gb={stats.bytes_text/1e9:.2f}",
                flush=True,
            )
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--languages", nargs="+", default=["de", "en"], choices=["de", "en"])
    parser.add_argument("--gutenberg-gb", type=float, default=6.0)
    parser.add_argument("--zkeown-gb", type=float, default=4.0)
    parser.add_argument("--pleias-gb", type=float, default=2.0)
    parser.add_argument("--min-chars", type=int, default=700)
    parser.add_argument("--target-chars", type=int, default=5000)
    parser.add_argument("--max-chars", type=int, default=9000)
    parser.add_argument("--allow-old-print", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    txt_path = args.output_dir / "book_sources.txt"
    jsonl_path = args.output_dir / "book_sources.jsonl"
    manifest_path = args.output_dir / "manifest.json"

    started = time.time()
    manifest = {
        "version": 1,
        "started_at": now_iso(),
        "output_text": str(txt_path),
        "output_jsonl": str(jsonl_path),
        "languages": args.languages,
        "sources": [],
    }
    seen_hashes: set[str] = set()
    sources = [
        (SourceStats("common_pile_project_gutenberg", "common-pile/project_gutenberg", None, "train"), row_from_common_pile, int(args.gutenberg_gb * 1e9)),
        (SourceStats("zkeown_gutenberg_books", "zkeown/gutenberg-corpus", "books", "train"), row_from_zkeown, int(args.zkeown_gb * 1e9)),
        (SourceStats("pleias_common_corpus_books", "PleIAs/common_corpus", None, "train"), row_from_pleias, int(args.pleias_gb * 1e9)),
    ]

    with txt_path.open("w", encoding="utf-8", newline="\n") as out_txt, jsonl_path.open("w", encoding="utf-8", newline="\n") as out_jsonl:
        for stats, reader, cap in sources:
            if cap <= 0:
                continue
            print(f"[source] {stats.name} cap_gb={cap/1e9:.2f}", flush=True)
            try:
                stream_source(
                    stats=stats,
                    row_reader=reader,
                    out_txt=out_txt,
                    out_jsonl=out_jsonl,
                    seen_hashes=seen_hashes,
                    max_bytes=cap,
                    args=args,
                )
            except Exception as exc:  # keep partial output + manifest
                stats.skipped["ERROR_" + type(exc).__name__] = stats.skipped.get("ERROR_" + type(exc).__name__, 0) + 1
                stats.skipped["ERROR_message"] = str(exc)[:500]  # type: ignore[assignment]
                print(f"[error] {stats.name}: {type(exc).__name__}: {exc}", flush=True)
            manifest["sources"].append(asdict(stats))
            manifest["documents"] = sum(s["chunks_written"] for s in manifest["sources"])
            manifest["bytes_text"] = sum(s["bytes_text"] for s in manifest["sources"])
            manifest["elapsed_seconds"] = round(time.time() - started, 1)
            manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest["finished_at"] = now_iso()
    manifest["elapsed_seconds"] = round(time.time() - started, 1)
    manifest["documents"] = sum(s["chunks_written"] for s in manifest["sources"])
    manifest["bytes_text"] = sum(s["bytes_text"] for s in manifest["sources"])
    manifest["top_skips"] = dict(Counter(k for s in manifest["sources"] for k, v in s["skipped"].items() for _ in range(v if isinstance(v, int) else 0)).most_common(20))
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {txt_path} ({manifest['documents']:,} docs, {manifest['bytes_text']/1e9:.2f} GB)", flush=True)
    print(f"wrote {jsonl_path}", flush=True)
    print(f"wrote {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
