#!/usr/bin/env python3
"""Audit pretraining text sources for clean-v3 triage.

The strict clean-v2 filter caught many obvious problems, but a base model can
still learn bad style from "valid-looking" documents: tables of contents, OCR
fragments, bibliographies, archive/catalogue pages, repetition, long scan
blocks, list-heavy lines, and synthetic/assistant bleed-through.

This audit samples each one-document-per-line source with a small head sample
and random byte-offset samples. It writes JSON + Markdown reports with:

- per-source keep/repair/trash estimates
- problem-class counts
- representative bad examples
- recommended clean-v3 action

It is read-only with respect to source data.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[2]

DEFAULT_SOURCES = {
    "german_commons": REPO / "data/training/pretrain_clean_v2/german_commons.strict.txt",
    "german_legacy": REPO / "data/training/pretrain_clean_v2/german.strict.txt",
    "wikipedia_de": REPO / "data/training/pretrain_clean_v2/wikipedia_de.strict.txt",
    "openmath": REPO / "data/training/pretrain_clean_v2/openmath.strict.txt",
    "booster": REPO / "data/training/pretrain_booster_de_v1m.txt",
}

WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")
URL_RE = re.compile(r"https?://|www\.|\.com\b|\.de\b|\.org\b", re.I)
HTML_RE = re.compile(r"<\s*/?\s*(html|body|div|span|script|style|table|a|p|br|iframe)\b", re.I)
CHAT_RE = re.compile(r"<\|(?:system|user|assistant|end|im_start|im_end)\|>|###\s*(?:Aufgabe|Antwort):", re.I)
MOJIBAKE_RE = re.compile(r"Ã.|Â.|â€|�|ï¿½")
TOC_RE = re.compile(
    r"\b(contents?|inhaltsverzeichnis|table of contents|page|seite)\b|"
    r"(^|\s)([ivxlcdm]{1,8}|\d{1,4})\s+[\.\-–—]+\s+\d{1,5}(\s|$)|"
    r"(\.{3,}|…{2,})\s*\d{1,5}(\s|$)",
    re.I,
)
BIBLIO_RE = re.compile(
    r"\b(verlag|druck|druckerei|isbn|doi:|journal|vol\.|volume|chapter|"
    r"herausgegeben|published by|copyright|all rights reserved|bibliothek)\b",
    re.I,
)
OCR_RE = re.compile(
    r"\b(?:thl|bd\.|s\.\s*\d+|a\.\s*a\.\s*o\.|vgl\.|nr\.|fig\.|"
    r"tafel|abb\.|cap\.|ibid|ſ|flg\.|u\.\s*s\.\s*w\.)\b|"
    # OCR/text extraction often inserts spaces inside words. Normal hyphenated
    # compounds like "US-amerikanisch" must not count.
    r"\b[A-Za-zÄÖÜäöüß](?:\s+[A-Za-zÄÖÜäöüß]){4,}\b",
    re.I,
)
BOILERPLATE_RE = re.compile(
    r"cookie|privacy policy|javascript|newsletter|subscribe|sign in|"
    r"alle rechte vorbehalten|nutzungsbedingungen|datenschutzerklärung",
    re.I,
)
CODE_RE = re.compile(r"\b(def|class|import|return|function|const|let|var|public static|#include)\b|[{};]{3,}")
MATH_RE = re.compile(r"\\\[|\\\(|\b(theorem|lemma|proof|beweis|gleichung|integral|matrix)\b|[=+\-*/^]{3,}")


@dataclass
class Sample:
    line_no: int | None
    text: str
    origin: str


@dataclass
class SourceReport:
    name: str
    path: str
    bytes: int
    samples: int = 0
    class_counts: Counter = field(default_factory=Counter)
    flag_counts: Counter = field(default_factory=Counter)
    length_sum: int = 0
    word_sum: int = 0
    examples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))


def clean_preview(text: str, limit: int = 320) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isalpha()) / len(text)


def symbol_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)


def digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isdigit()) / len(text)


def repetition_ratio(words: list[str]) -> float:
    if len(words) < 30:
        return 0.0
    return 1.0 - len(set(words)) / len(words)


def repeated_ngram_ratio(words: list[str], n: int = 4) -> float:
    if len(words) < n * 4:
        return 0.0
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def line_list_score(text: str) -> float:
    # Many strict sources are one physical line per doc. Count list/table-ish
    # separators inside that document line.
    markers = len(re.findall(r"(^|\s)(?:[-*•]|\d{1,3}[.)])\s+", text))
    page_refs = len(re.findall(r"\b\d{1,4}\s*(?:\.{2,}|…|—|-)\s*\d{1,5}\b", text))
    separators = text.count(" | ") + text.count("\t")
    return markers + page_refs + separators


def classify(text: str) -> tuple[str, list[str]]:
    flags: list[str] = []
    stripped = text.strip()
    length = len(stripped)
    words = [w.lower() for w in WORD_RE.findall(stripped)]

    if length < 180:
        flags.append("too_short")
    if length > 50_000:
        flags.append("very_long_doc")
    if MOJIBAKE_RE.search(stripped):
        flags.append("mojibake")
    if HTML_RE.search(stripped):
        flags.append("html")
    if URL_RE.search(stripped) and len(URL_RE.findall(stripped)) >= 3:
        flags.append("url_dense")
    if CHAT_RE.search(stripped):
        flags.append("chat_or_sft_marker")
    if BOILERPLATE_RE.search(stripped):
        flags.append("boilerplate")
    toc_hits = len(TOC_RE.findall(stripped))
    biblio_hits = len(BIBLIO_RE.findall(stripped))
    ocr_hits = len(OCR_RE.findall(stripped))

    if toc_hits >= 2 or (toc_hits >= 1 and line_list_score(stripped) >= 4):
        flags.append("toc_or_index")
    if biblio_hits >= 3:
        flags.append("bibliography_or_metadata")
    if ocr_hits >= 2:
        flags.append("ocr_scan_hint")

    ar = alpha_ratio(stripped)
    sr = symbol_ratio(stripped)
    dr = digit_ratio(stripped)
    rr = repetition_ratio(words)
    ngr = repeated_ngram_ratio(words)
    lscore = line_list_score(stripped)

    if ar < 0.45:
        flags.append("low_alpha")
    if sr > 0.22:
        flags.append("symbol_dense")
    if dr > 0.16:
        flags.append("digit_dense")
    if rr > 0.68 or ngr > 0.28:
        flags.append("repetitive")
    if lscore >= 8:
        flags.append("list_or_table_heavy")
    if CODE_RE.search(stripped):
        flags.append("code_like")
    if MATH_RE.search(stripped):
        flags.append("math_like")

    hard_trash = {
        "html",
        "url_dense",
        "chat_or_sft_marker",
        "boilerplate",
        "mojibake",
        "repetitive",
    }
    repairish = {
        "toc_or_index",
        "bibliography_or_metadata",
        "ocr_scan_hint",
        "list_or_table_heavy",
        "very_long_doc",
        "symbol_dense",
        "digit_dense",
        "low_alpha",
    }

    if hard_trash.intersection(flags):
        quality = "trash"
    elif repairish.intersection(flags):
        quality = "repair"
    elif length >= 240 and len(words) >= 35:
        quality = "keep"
    else:
        quality = "repair"
    return quality, flags


def iter_head_samples(path: Path, n: int) -> Iterable[Sample]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for idx, line in enumerate(fh, start=1):
            if idx > n:
                break
            line = line.rstrip("\n")
            if line:
                yield Sample(idx, line, "head")


def iter_random_samples(path: Path, n: int, seed: int) -> Iterable[Sample]:
    size = path.stat().st_size
    if size <= 0 or n <= 0:
        return
    rng = random.Random(seed)
    with path.open("rb") as fh:
        for _ in range(n):
            pos = rng.randrange(0, size)
            fh.seek(pos)
            if pos > 0:
                fh.readline()  # discard partial line
            raw = fh.readline()
            if not raw:
                fh.seek(0)
                raw = fh.readline()
            text = raw.decode("utf-8", errors="replace").rstrip("\n")
            if text:
                yield Sample(None, text, "random")


def audit_source(name: str, path: Path, head: int, random_n: int, seed: int, examples_per_flag: int) -> SourceReport:
    report = SourceReport(name=name, path=str(path), bytes=path.stat().st_size)
    for sample in list(iter_head_samples(path, head)) + list(iter_random_samples(path, random_n, seed)):
        report.samples += 1
        text = sample.text
        words = WORD_RE.findall(text)
        quality, flags = classify(text)
        report.class_counts[quality] += 1
        report.length_sum += len(text)
        report.word_sum += len(words)
        for flag in flags:
            report.flag_counts[flag] += 1
            if len(report.examples[flag]) < examples_per_flag:
                report.examples[flag].append(clean_preview(text))
        if quality != "keep" and len(report.examples[quality]) < examples_per_flag:
            report.examples[quality].append(clean_preview(text))
    return report


def pct(n: int, d: int) -> float:
    return 100.0 * n / max(d, 1)


def recommendation(report: SourceReport) -> str:
    trash = pct(report.class_counts["trash"], report.samples)
    repair = pct(report.class_counts["repair"], report.samples)
    flags = report.flag_counts
    if trash >= 25 or flags["chat_or_sft_marker"] > 0:
        return "exclude_or_reclean_hard"
    if repair + trash >= 45:
        return "reclean_with_v3_filters"
    if flags["toc_or_index"] + flags["ocr_scan_hint"] > report.samples * 0.15:
        return "split_and_drop_ocr_index"
    return "keep_with_light_filters"


def as_dict(report: SourceReport) -> dict:
    return {
        "name": report.name,
        "path": report.path,
        "bytes": report.bytes,
        "gib": round(report.bytes / 1024**3, 3),
        "samples": report.samples,
        "class_counts": dict(report.class_counts),
        "class_rates": {k: round(v / max(report.samples, 1), 4) for k, v in report.class_counts.items()},
        "flag_counts": dict(report.flag_counts.most_common()),
        "flag_rates": {k: round(v / max(report.samples, 1), 4) for k, v in report.flag_counts.items()},
        "avg_chars": round(report.length_sum / max(report.samples, 1), 1),
        "avg_words": round(report.word_sum / max(report.samples, 1), 1),
        "recommendation": recommendation(report),
        "examples": dict(report.examples),
    }


def write_markdown(reports: list[SourceReport], output: Path) -> None:
    lines: list[str] = []
    lines.append("# Pretrain Clean-v2 Data Audit\n")
    lines.append(
        "Sample-based triage for clean-v3. Rates combine a small head sample "
        "with random byte-offset samples, so treat them as directional, not exact.\n"
    )
    lines.append("| Source | GiB | Samples | Keep | Repair | Trash | Top Flags | Recommendation |")
    lines.append("|---|---:|---:|---:|---:|---:|---|---|")
    for r in reports:
        top_flags = ", ".join(f"{k} {pct(v, r.samples):.1f}%" for k, v in r.flag_counts.most_common(4))
        lines.append(
            f"| `{r.name}` | {r.bytes / 1024**3:.2f} | {r.samples:,} | "
            f"{pct(r.class_counts['keep'], r.samples):.1f}% | "
            f"{pct(r.class_counts['repair'], r.samples):.1f}% | "
            f"{pct(r.class_counts['trash'], r.samples):.1f}% | "
            f"{top_flags or '-'} | `{recommendation(r)}` |"
        )

    total_bytes = sum(r.bytes for r in reports)
    lines.append("\n## Weighted Estimate\n")
    for cls in ("keep", "repair", "trash"):
        weighted = sum((r.class_counts[cls] / max(r.samples, 1)) * r.bytes for r in reports)
        lines.append(f"- {cls}: ~{weighted / 1024**3:.2f} GiB of {total_bytes / 1024**3:.2f} GiB")

    lines.append("\n## Source Notes\n")
    for r in reports:
        lines.append(f"### {r.name}\n")
        lines.append(f"- Action: `{recommendation(r)}`")
        lines.append(f"- Avg size: {r.length_sum / max(r.samples, 1):.0f} chars, {r.word_sum / max(r.samples, 1):.0f} words")
        if r.flag_counts:
            lines.append("- Main flags: " + ", ".join(f"`{k}` {pct(v, r.samples):.1f}%" for k, v in r.flag_counts.most_common(8)))
        for flag, examples in list(r.examples.items())[:6]:
            if not examples:
                continue
            lines.append(f"\nBad sample for `{flag}`:")
            lines.append("```text")
            lines.append(examples[0])
            lines.append("```")
        lines.append("")

    lines.append("## Clean-v3 Filter Priorities\n")
    lines.append("1. Drop TOC/index/bibliography/catalogue documents before tokenization.")
    lines.append("2. Split or exclude very long scanned book lines instead of feeding whole pages as one document.")
    lines.append("3. Add OCR-fragment scoring for old book sources, especially legacy `german.strict`.")
    lines.append("4. Keep German Commons/Wikipedia as high-priority base, then add code/math in controlled shares.")
    lines.append("5. Keep synthetic/SFT-style data out of base pretraining unless explicitly marked as a small booster.")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", action="append", default=None, help="name=path. Can be repeated.")
    parser.add_argument("--head-samples", type=int, default=2_000)
    parser.add_argument("--random-samples", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--examples-per-flag", type=int, default=3)
    parser.add_argument("--output-json", type=Path, default=REPO / "data/eval/pretrain_clean_v2_audit_v3.json")
    parser.add_argument("--output-md", type=Path, default=REPO / "data/eval/pretrain_clean_v2_audit_v3.md")
    args = parser.parse_args()

    if args.source:
        sources: dict[str, Path] = {}
        for item in args.source:
            if "=" not in item:
                raise SystemExit(f"--source must be name=path, got {item!r}")
            name, raw_path = item.split("=", 1)
            p = Path(raw_path)
            sources[name] = p if p.is_absolute() else REPO / p
    else:
        sources = DEFAULT_SOURCES

    reports: list[SourceReport] = []
    for idx, (name, path) in enumerate(sources.items()):
        if not path.is_file():
            print(f"skip missing: {name} {path}")
            continue
        print(f"auditing {name}: {path}", flush=True)
        reports.append(
            audit_source(
                name,
                path,
                head=args.head_samples,
                random_n=args.random_samples,
                seed=args.seed + idx,
                examples_per_flag=args.examples_per_flag,
            )
        )

    payload = {
        "head_samples": args.head_samples,
        "random_samples": args.random_samples,
        "seed": args.seed,
        "reports": [as_dict(r) for r in reports],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(reports, args.output_md)
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")


if __name__ == "__main__":
    main()
