#!/usr/bin/env python3
"""Audit short documents from a one-document-per-line corpus.

The goal is to decide whether short documents are useful compact knowledge
units or scrape fragments that should be removed before a training mix.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import sentencepiece as spm


HTML_RE = re.compile(r"<\s*/?\s*(html|body|div|script|style|table|iframe|a)\b|&(?:amp|gt|lt|quot|#x?[0-9a-f]+);", re.I)
CHAT_RE = re.compile(r"<\|(?:im_start|im_end|endoftext|user|assistant|system)\|>|_end_of_the_data|</?think>", re.I)
URL_RE = re.compile(r"https?://|www\.", re.I)
ADULT_CASINO_RE = re.compile(r"\b(?:onlyfans|porn|xxx|casino|jackpot|free spins|sportwetten|sexkontakte)\b", re.I)
SHOP_RE = re.compile(r"\b(?:warenkorb|checkout|rabattcode|gutschein|trusted shops|lieferzeit|versandkosten)\b", re.I)
TOC_RE = re.compile(r"\b(?:inhaltsverzeichnis|table of contents|seite|page)\b|\.{3,}\s*\d{1,5}", re.I)
OCR_RE = re.compile(r"Ã.|�|Å¿|\b[a-zA-ZÄÖÜäöüß](?:\s+[a-zA-ZÄÖÜäöüß]){4,}\b")
LIST_RE = re.compile(r"(^|\s)(?:[-*•]|\d{1,3}[.)])\s+|[|]{2,}|\t")
WIKI_TALK_RE = re.compile(
    r"\b(?:nicht signierter beitrag|ce[st]\)|wikipedia:|qs-baustein|"
    r"redaktion[_ ]|diskussion:|benutzer:|--\s*[A-Za-zÄÖÜäöüß0-9_-])",
    re.I,
)
TABLE_RE = re.compile(r"\|\s*[-:]+\s*\||\{\||\|\}|^\s*\|", re.I)
INDEX_RE = re.compile(r"\b(?:kategorie:|liste der|personen nach|artikel des tages|portal:)\b|\.{3,}\s*\d{1,5}", re.I)
WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")


def clean_preview(text: str, limit: int = 1000) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def pct(value: int | float, total: int | float) -> float:
    return round(float(value) * 100.0 / float(total), 3) if total else 0.0


def hard_flags(text: str) -> list[str]:
    flags = []
    if HTML_RE.search(text):
        flags.append("html")
    if CHAT_RE.search(text):
        flags.append("chat_marker")
    if len(URL_RE.findall(text)) >= 3:
        flags.append("url_dense")
    if ADULT_CASINO_RE.search(text):
        flags.append("adult_or_casino")
    if SHOP_RE.search(text):
        flags.append("shop_spam")
    return flags


def short_kind(text: str) -> str:
    stripped = text.strip()
    words = WORD_RE.findall(stripped)
    lower_start = bool(stripped[:1] and stripped[:1].islower())
    terminal = stripped.endswith((".", "!", "?", ":", ")", "]", "\""))
    flags = hard_flags(stripped)
    if flags:
        return "hard_noise"
    if OCR_RE.search(stripped):
        return "ocr_or_mojibake"
    if WIKI_TALK_RE.search(stripped):
        return "wiki_talk"
    if TABLE_RE.search(stripped):
        return "table_fragment"
    if INDEX_RE.search(stripped) or TOC_RE.search(stripped):
        return "index_or_toc"
    if LIST_RE.search(stripped):
        return "list_like"
    if len(words) < 8:
        return "tiny_fragment"
    if lower_start and not terminal:
        return "broken_sentence_fragment"
    if lower_start or not terminal:
        return "sentence_fragment"
    if ":" in stripped and len(words) < 28:
        return "definition_or_label"
    return "compact_valid"


def reservoir_add(rng: random.Random, reservoir: list[dict], item: dict, seen: int, limit: int) -> None:
    if len(reservoir) < limit:
        reservoir.append(item)
        return
    j = rng.randrange(0, seen)
    if j < limit:
        reservoir[j] = item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--threshold-tokens", type=int, default=100)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument(
        "--sample-kinds",
        default="",
        help="Comma-separated short_kind categories to sample separately, e.g. compact_valid,sentence_fragment.",
    )
    parser.add_argument("--per-kind-sample-size", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--max-docs", type=int, default=0, help="0 means full file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    sample_kinds = {k.strip() for k in args.sample_kinds.split(",") if k.strip()}

    total_docs = 0
    short_docs = 0
    hard_counts = Counter()
    kind_counts = Counter()
    short_samples: list[dict] = []
    kind_samples: dict[str, list[dict]] = {kind: [] for kind in sample_kinds}
    kind_seen: Counter = Counter()

    with args.input.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            total_docs += 1
            flags = hard_flags(text)
            hard_counts.update(flags)
            token_count = len(sp.encode(text, out_type=int)) + 1
            if token_count < args.threshold_tokens:
                short_docs += 1
                kind = short_kind(text)
                kind_counts[kind] += 1
                if kind in sample_kinds:
                    kind_seen[kind] += 1
                    reservoir_add(
                        rng,
                        kind_samples[kind],
                        {
                            "line": line_no,
                            "tokens": token_count,
                            "kind": kind,
                            "hard_flags": flags,
                            "text": clean_preview(text, 1400),
                        },
                        kind_seen[kind],
                        args.per_kind_sample_size,
                    )
                reservoir_add(
                    rng,
                    short_samples,
                    {
                        "line": line_no,
                        "tokens": token_count,
                        "kind": kind,
                        "hard_flags": flags,
                        "text": clean_preview(text, 1400),
                    },
                    short_docs,
                    args.sample_size,
                )
            if args.max_docs and total_docs >= args.max_docs:
                break

    results = {
        "input": str(args.input),
        "scanned_docs": total_docs,
        "threshold_tokens": args.threshold_tokens,
        "short_docs": short_docs,
        "short_docs_pct": pct(short_docs, total_docs),
        "hard_flag_counts": dict(hard_counts),
        "short_kind_counts": dict(kind_counts),
        "samples": short_samples,
        "kind_samples": kind_samples,
    }
    (args.output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Short Document Audit",
        "",
        f"- input: `{args.input}`",
        f"- scanned_docs: `{total_docs:,}`",
        f"- short_docs <{args.threshold_tokens}: `{short_docs:,}` (`{pct(short_docs, total_docs)}%`)",
        f"- hard_flag_counts: `{dict(hard_counts)}`",
        f"- short_kind_counts: `{dict(kind_counts)}`",
        "",
        "## Samples",
        "",
    ]
    for i, sample in enumerate(short_samples, start=1):
        lines.extend(
            [
                f"### Sample {i}",
                "",
                f"- line: `{sample['line']}`",
                f"- tokens: `{sample['tokens']}`",
                f"- kind: `{sample['kind']}`",
                f"- hard_flags: `{sample['hard_flags'] or 'none'}`",
                "",
                "```text",
                sample["text"],
                "```",
                "",
            ]
        )
    if kind_samples:
        lines.extend(["", "## Per-Kind Samples", ""])
        for kind, samples in kind_samples.items():
            lines.extend([f"## Kind: {kind}", ""])
            for i, sample in enumerate(samples, start=1):
                lines.extend(
                    [
                        f"### {kind} sample {i}",
                        "",
                        f"- line: `{sample['line']}`",
                        f"- tokens: `{sample['tokens']}`",
                        f"- hard_flags: `{sample['hard_flags'] or 'none'}`",
                        "",
                        "```text",
                        sample["text"],
                        "```",
                        "",
                    ]
                )
    (args.output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
