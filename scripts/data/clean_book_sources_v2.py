#!/usr/bin/env python3
"""Second-pass cleaner for Gutenberg/book corpora.

The first book downloader already chunks and removes obvious OCR/boilerplate.
This pass is stricter about book-specific artifacts: Gutenberg license blocks,
front/back matter, table of contents, indexes, errata, page lists, and editorial
transcription notes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


SPACE_RE = re.compile(r"\s+")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+")

GUTENBERG_RE = re.compile(
    r"project gutenberg|gutenberg\.org|gutenberg etext|gutenberg ebook|"
    r"gutenberg-tm|electronic work|etext\s+#|ebook\s+#|"
    r"distributed proofreading|produced by|prepared by volunteers|"
    r"transcriber'?s note|transcriber's notes|proofreaders|"
    r"start of (?:the|this) project gutenberg|end of (?:the|this) project gutenberg|"
    r"full project gutenberg license|project gutenberg literary archive",
    re.I,
)
LICENSE_RE = re.compile(
    r"copyright|all rights reserved|terms of use|permission is granted|"
    r"redistribution|public domain|creative commons|license|licence|"
    r"without warranty|limited warranty|trademark|royalties|"
    r"foundation|donations?|tax deductible",
    re.I,
)
FRONT_BACK_RE = re.compile(
    r"^(\s*)("
    r"title page|half-title|preface|foreword|introduction|publisher'?s note|"
    r"editor'?s note|author'?s note|dedication|acknowledg(e)?ments|"
    r"contents|table of contents|inhaltsverzeichnis|inhalt|"
    r"list of illustrations|list of plates|errata|bibliography|"
    r"index|appendix|glossary|notes?|footnotes?"
    r")\b",
    re.I,
)
INDEX_RE = re.compile(
    r"\b(index|general index|name index|subject index|register|glossary)\b",
    re.I,
)
TOC_RE = re.compile(
    r"\b(contents|table of contents|inhaltsverzeichnis|inhalt|list of illustrations|list of plates)\b",
    re.I,
)
PAGE_REF_RE = re.compile(
    r"(\bpage\s+\d{1,4}\b|\bp\.\s*\d{1,4}\b|\.{3,}\s*\d{1,4}\b|\[\s*page\s+\d{1,4}\s*\])",
    re.I,
)
CHAPTER_PAGE_RE = re.compile(r"\b(chapter|kapitel)\s+[ivxlcdm\d]+\b.{0,80}\b(page|seite)\b", re.I)
ALL_CAPS_WORD_RE = re.compile(r"\b[A-Z]{4,}\b")
ROMAN_LINE_RE = re.compile(r"^\s*(?:chapter|kapitel)?\s*[ivxlcdm]{1,12}\.?\s*$", re.I)
SCENE_PAGE_RE = re.compile(r"^\s*(?:act|scene|chapter|book|part|canto)\s+[ivxlcdm\d]+\.?\s*$", re.I)
MIXED_DOC_RE = re.compile(
    r"\b(the constitution of the united states|bill of rights|inaugural address|"
    r"gettysburg address|magna carta)\b",
    re.I,
)
MOJIBAKE_RE = re.compile(r"\u00c3.|\u00c2.|\ufffd|\u00e2\u20ac|\u00ef\u00ac")
URL_RE = re.compile(r"https?://|www\.", re.I)


def clean(text: Any) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def one_line(text: str) -> str:
    return SPACE_RE.sub(" ", clean(text)).strip()


def stable_hash(text: str) -> str:
    return hashlib.sha256(one_line(text).lower().encode("utf-8", errors="replace")).hexdigest()


def ratio_all_caps(text: str) -> float:
    words = WORD_RE.findall(text)
    if not words:
        return 0.0
    caps = sum(1 for w in words if len(w) >= 4 and w.upper() == w)
    return caps / len(words)


def listish_score(text: str) -> float:
    parts = re.split(r"[.;]\s+|\n+", text)
    short = sum(1 for p in parts if 0 < len(p.split()) <= 7)
    return short / max(1, len(parts))


def reject_reason(text: str, title: str = "") -> str | None:
    t = one_line(text)
    lower_head = t[:2500].lower()
    if len(t) < 700:
        return "too_short"
    if MOJIBAKE_RE.search(t):
        return "mojibake"
    if URL_RE.search(t):
        return "url"
    if GUTENBERG_RE.search(t):
        return "gutenberg_boilerplate"
    if LICENSE_RE.search(lower_head) and ("project gutenberg" in lower_head or "redistribution" in lower_head):
        return "license_block"
    if LICENSE_RE.search(lower_head) and len(t) < 2500:
        return "license_short"
    if FRONT_BACK_RE.search(t[:400]):
        return "front_back_matter"
    if TOC_RE.search(lower_head) and (PAGE_REF_RE.search(lower_head) or CHAPTER_PAGE_RE.search(lower_head)):
        return "table_of_contents"
    if INDEX_RE.search(t[:800]) and (PAGE_REF_RE.search(t) or listish_score(t) > 0.45):
        return "index_glossary"
    if PAGE_REF_RE.search(t[:2000]) and listish_score(t[:3000]) > 0.40:
        return "page_list"
    if ratio_all_caps(t[:3000]) > 0.28:
        return "all_caps"
    if MIXED_DOC_RE.search(t) and "Declaration of Independence" in title:
        return "mixed_gutenberg_bundle"
    alpha = sum(1 for c in t if c.isalpha()) / max(1, len(t))
    if alpha < 0.52:
        return "low_alpha"
    words = WORD_RE.findall(t)
    if len(words) < 120:
        return "too_few_words"
    unique = len(set(w.lower() for w in words)) / max(1, len(words))
    if unique < 0.16:
        return "repetitive"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-text", type=Path, required=True)
    parser.add_argument("--reject-samples", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--flush-every", type=int, default=25000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_text.parent.mkdir(parents=True, exist_ok=True)
    if args.reject_samples:
        args.reject_samples.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".manifest.json")

    stats: dict[str, Any] = {
        "input_jsonl": str(args.input_jsonl),
        "output_jsonl": str(args.output_jsonl),
        "output_text": str(args.output_text),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "docs_in": 0,
        "docs_written": 0,
        "bytes_written_text": 0,
        "dropped": {},
        "sources": {},
        "languages": {},
    }
    dropped: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    seen: set[str] = set()

    reject_fh = args.reject_samples.open("w", encoding="utf-8", newline="\n") if args.reject_samples else None
    try:
        with args.input_jsonl.open("r", encoding="utf-8", errors="replace") as src, args.output_jsonl.open(
            "w", encoding="utf-8", newline="\n"
        ) as out_jsonl, args.output_text.open("w", encoding="utf-8", newline="\n") as out_txt:
            for line in src:
                stats["docs_in"] += 1
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    dropped["bad_json"] += 1
                    continue
                text = clean(obj.get("text"))
                title = clean(obj.get("title"))
                reason = reject_reason(text, title)
                if not reason:
                    h = stable_hash(text)
                    if h in seen:
                        reason = "duplicate"
                    else:
                        seen.add(h)
                if reason:
                    dropped[reason] += 1
                    if reject_fh and dropped[reason] <= 20:
                        reject_fh.write(
                            json.dumps(
                                {
                                    "reason": reason,
                                    "source": obj.get("source"),
                                    "language": obj.get("language"),
                                    "title": title[:180],
                                    "preview": one_line(text)[:700],
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                    continue
                text_out = one_line(text)
                obj["text"] = text_out
                obj["book_clean_v2"] = True
                out_jsonl.write(json.dumps(obj, ensure_ascii=False, sort_keys=True) + "\n")
                out_txt.write(text_out + "\n")
                stats["docs_written"] += 1
                stats["bytes_written_text"] += len((text_out + "\n").encode("utf-8"))
                sources[str(obj.get("source") or "unknown")] += 1
                languages[str(obj.get("language") or "unknown")] += 1
                if args.flush_every and stats["docs_in"] % args.flush_every == 0:
                    out_jsonl.flush()
                    out_txt.flush()
                    print(
                        f"progress docs_in={stats['docs_in']:,} docs_written={stats['docs_written']:,} "
                        f"dropped={sum(dropped.values()):,}",
                        flush=True,
                    )
    finally:
        if reject_fh:
            reject_fh.close()

    stats["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    stats["dropped"] = dict(dropped.most_common())
    stats["sources"] = dict(sources.most_common())
    stats["languages"] = dict(languages.most_common())
    stats["unique_hashes"] = len(seen)
    manifest_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
