#!/usr/bin/env python3
"""Profile-aware NeMo Curator audit for the v5 pretraining mix.

NeMo Curator ships useful text filters, but many defaults are tuned for
English web prose. This audit keeps the scoring layer from Curator and adds a
small source-aware decision layer so math, QA, Reddit/chat, and German prose do
not get judged by the same thresholds.

The script is read-only. It samples one-document-per-line mixes, writes a
report, and emits enough examples to tune a future v6/v7 builder profile.
"""

from __future__ import annotations

import argparse
import bisect
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


try:
    from nemo_curator.stages.text.filters.heuristic.repetition.repetition import (
        RepeatedLinesByCharFilter,
        RepeatedLinesFilter,
        RepeatedParagraphsByCharFilter,
        RepeatedParagraphsFilter,
        RepeatingDuplicateNGramsFilter,
        RepeatingTopNGramsFilter,
    )
    from nemo_curator.stages.text.filters.heuristic.string import (
        BulletsFilter,
        EllipsisFilter,
        LongWordFilter,
        MeanWordLengthFilter,
        NonAlphaNumericFilter,
        NumbersFilter,
        ParenthesesFilter,
        PunctuationFilter,
        SymbolsToWordsFilter,
        UrlsFilter,
        WhiteSpaceFilter,
        WordCountFilter,
        WordsWithoutAlphabetsFilter,
    )
except Exception as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "NeMo Curator is required. Activate the curator venv first, for example:\n"
        "  source /workspace/v2data/.venvs/nemo-curator/bin/activate\n"
        f"Import error: {exc}"
    ) from exc


REPO = Path(__file__).resolve().parents[2]
DEFAULT_MIX = Path("/workspace/v2data/data/training/pretrain_mix_v5_boosted/mix_full.txt")
DEFAULT_MANIFEST = Path("/workspace/v2data/data/training/pretrain_mix_v5_boosted/manifest.json")
DEFAULT_OUTPUT_DIR = Path("/workspace/v2data/data/eval/nemo_curator_v5_profile_audit")

HTML_RE = re.compile(
    r"<\s*/?\s*(html|body|div|span|script|style|table|tr|td|iframe)\b|"
    r"<br\s*/?>|&(?:nbsp|amp|lt|gt|quot|#x?[0-9a-f]+);",
    re.I,
)
CHAT_RE = re.compile(
    r"<\|(?:system|user|assistant|im_start|im_end|endoftext)\|>|"
    r"^\s*###\s*(?:Instruction|Response|Human|Assistant)\b|"
    r"^\s*(?:User|Assistant|Human):\s|</?think>",
    re.I | re.M,
)
ADULT_SHOP_RE = re.compile(
    r"\b(?:onlyfans|porn|xxx|sexcam|escort|viagra|casino|spielautomat|"
    r"jackpot|free spins|sportwetten|rabattcode|warenkorb|checkout|"
    r"trusted shops|lieferzeit|versandkosten|kundenservice)\b",
    re.I,
)
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
LIST_RE = re.compile(
    r"(^|\n)\s*(?:[-*•]|\d{1,3}[.)]|[a-zA-Z][.)])\s+|"
    r"\|[^\n]+\||\b(?:Kategorie:|Datei:|Vorlage:|Liste der)\b",
    re.I,
)
MOJIBAKE_RE = re.compile(r"Ãƒ.|ï¿½|Ã…Â¿|â€™|â€œ|â€|â€“|â€”")
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
ALPHA_RE = re.compile(r"[A-Za-zÄÖÜäöüß]")
LONG_TOKEN_RE = re.compile(r"\S{180,}")
MATH_CONTENT_RE = re.compile(
    r"\b(?:Problem|Solution|Loesung|Lösung):|"
    r"\\(?:frac|sqrt|begin|end|boxed|times|cdot|mathbf|overline)|"
    r"\$\s*[-+*/=\\\w]+\s*\$|\[[Aa][Ss][Yy]\]",
    re.I,
)
QA_CONTENT_RE = re.compile(r"\b(?:QA\. Frage:|Question:|Answer:|Instruction:|Response:)\b", re.I)
DIALOGUE_CONTENT_RE = re.compile(r"\bDialog: Nutzer:|Reddit-Thread-QA\. Frage:", re.I)


@dataclass(frozen=True)
class SourceSpan:
    start: int
    end: int
    name: str
    profile: str


@dataclass
class AuditDoc:
    line_no: int
    source: str
    profile: str
    text: str
    chars: int
    words: int
    scores: dict[str, Any]
    warnings: list[str]
    hard_drops: list[str]


def pct(value: int | float, total: int | float) -> float:
    return round(float(value) * 100.0 / float(total), 3) if total else 0.0


def q(values: list[int | float], quantile: float) -> int | float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * quantile))
    return ordered[max(0, min(index, len(ordered) - 1))]


def clean_preview(text: str, limit: int = 600) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def url_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(len(match.group(0)) for match in URL_RE.finditer(text)) / len(text)


def alpha_word_ratio(words: list[str]) -> float:
    if not words:
        return 0.0
    return sum(1 for word in words if ALPHA_RE.search(word)) / len(words)


def profile_for_source(source: str) -> str:
    lower = source.lower()
    if "math/" in lower or "openmath" in lower:
        return "math"
    if "reddit" in lower or "wildchat" in lower:
        return "dialogue"
    if "qa/" in lower or "large_qa" in lower or "orca" in lower:
        return "qa"
    if "validation" in lower:
        return "validation"
    return "german_prose"


def effective_profile(source_profile: str, text: str) -> str:
    """Override source profile when the document surface clearly belongs elsewhere."""
    if MATH_CONTENT_RE.search(text):
        return "math"
    if DIALOGUE_CONTENT_RE.search(text):
        return "dialogue"
    if QA_CONTENT_RE.search(text):
        return "qa"
    return source_profile


def load_spans(manifest_path: Path) -> list[SourceSpan]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    spans: list[SourceSpan] = []
    cursor = 1
    for source in manifest.get("sources", []):
        docs = int(source.get("documents") or 0)
        if docs <= 0:
            continue
        name = str(source.get("name") or "unknown")
        spans.append(SourceSpan(cursor, cursor + docs - 1, name, profile_for_source(name)))
        cursor += docs
    return spans


def source_for_line(line_no: int, spans: list[SourceSpan], ends: list[int]) -> SourceSpan:
    idx = bisect.bisect_left(ends, line_no)
    if idx >= len(spans):
        return SourceSpan(line_no, line_no, "unknown", "german_prose")
    span = spans[idx]
    if span.start <= line_no <= span.end:
        return span
    return SourceSpan(line_no, line_no, "unknown", "german_prose")


def sample_line_numbers(total_docs: int, samples: int, seed: int) -> list[int]:
    samples = min(samples, total_docs)
    rng = random.Random(seed)
    return sorted(rng.sample(range(1, total_docs + 1), samples))


def read_samples(mix_path: Path, line_numbers: list[int], spans: list[SourceSpan]) -> list[tuple[int, SourceSpan, str]]:
    wanted = set(line_numbers)
    ends = [span.end for span in spans]
    rows: list[tuple[int, SourceSpan, str]] = []
    with mix_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            if line_no in wanted:
                rows.append((line_no, source_for_line(line_no, spans, ends), line.rstrip("\n")))
                if len(rows) >= len(line_numbers):
                    break
    return rows


def curator_filters() -> dict[str, Any]:
    """Return language-neutral/de-tolerant filters used as signal generators."""
    return {
        "word_count_de": WordCountFilter(min_words=1, max_words=100000, lang="de"),
        "non_alpha_numeric": NonAlphaNumericFilter(max_non_alpha_numeric_to_text_ratio=1.0),
        "numbers": NumbersFilter(max_number_to_text_ratio=1.0),
        "urls": UrlsFilter(max_url_to_text_ratio=1.0),
        "whitespace": WhiteSpaceFilter(max_white_space_ratio=1.0),
        "bullets": BulletsFilter(max_bullet_lines_ratio=1.0),
        "punctuation": PunctuationFilter(max_num_sentences_without_endmark_ratio=1.0),
        "symbols_to_words_de": SymbolsToWordsFilter(max_symbol_to_word_ratio=1.0, lang="de"),
        "long_word_de": LongWordFilter(max_word_length=10_000, lang="de"),
        "mean_word_len_de": MeanWordLengthFilter(min_mean_word_length=0, max_mean_word_length=10_000, lang="de"),
        "words_with_alpha_de": WordsWithoutAlphabetsFilter(min_words_with_alphabets=0.0, lang="de"),
        "parentheses": ParenthesesFilter(max_parentheses_ratio=1.0),
        "ellipsis": EllipsisFilter(max_num_lines_ending_with_ellipsis_ratio=1.0),
        "repeated_lines": RepeatedLinesFilter(max_repeated_line_fraction=1.0),
        "repeated_lines_char": RepeatedLinesByCharFilter(max_repeated_lines_char_ratio=1.0),
        "repeated_paragraphs": RepeatedParagraphsFilter(max_repeated_paragraphs_ratio=1.0),
        "repeated_paragraphs_char": RepeatedParagraphsByCharFilter(max_repeated_paragraphs_char_ratio=1.0),
        "duplicate_3gram_de": RepeatingDuplicateNGramsFilter(
            n=3, max_repeating_duplicate_ngram_ratio=1.0, lang="de"
        ),
        "top_3gram_de": RepeatingTopNGramsFilter(n=3, max_repeating_ngram_ratio=1.0, lang="de"),
    }


def score_text(text: str, filters: dict[str, Any]) -> dict[str, Any]:
    scores: dict[str, Any] = {}
    for name, filt in filters.items():
        try:
            value = filt.score_document(text)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                scores[name] = round(float(value), 6)
            else:
                scores[name] = value
        except Exception as exc:  # pragma: no cover - third-party edge case
            scores[name] = f"ERROR:{type(exc).__name__}:{exc}"
    return scores


def decide(profile: str, text: str, scores: dict[str, Any], words: int) -> tuple[list[str], list[str]]:
    """Return warnings and hard drops for a profile.

    The thresholds are intentionally conservative for hard drops. Curator
    signals are treated as warnings unless they are profile-safe to remove.
    """
    warnings: list[str] = []
    hard: list[str] = []

    non_alpha = float(scores.get("non_alpha_numeric") or 0.0)
    numbers = float(scores.get("numbers") or 0.0)
    url_ratio = url_char_ratio(text)
    whitespace = float(scores.get("whitespace") or 0.0)
    bullets = float(scores.get("bullets") or 0.0)
    punctuation = float(scores.get("punctuation") or 0.0)
    symbols = float(scores.get("symbols_to_words_de") or 0.0)
    long_word = float(scores.get("long_word_de") or 0.0)
    alpha_words = float(scores.get("words_with_alpha_de") or 1.0)
    ellipsis = float(scores.get("ellipsis") or 0.0)
    dup_3gram = float(scores.get("duplicate_3gram_de") or 0.0)
    top_3gram = float(scores.get("top_3gram_de") or 0.0)
    repeated_lines = float(scores.get("repeated_lines") or 0.0)
    repeated_paragraphs = float(scores.get("repeated_paragraphs") or 0.0)
    line_count = max(1, text.count("\n") + 1)
    paragraph_count = len([p for p in re.split(r"\n\s*\n", text) if p.strip()])

    html_hits = len(HTML_RE.findall(text))
    if html_hits >= 3 or re.search(r"<\s*/?\s*(script|style|iframe|table)\b", text, re.I):
        hard.append("html_or_entity")
    elif html_hits:
        warnings.append("html_or_entity_hint")
    if CHAT_RE.search(text):
        hard.append("chat_marker")
    mojibake_hits = len(MOJIBAKE_RE.findall(text))
    if mojibake_hits >= 2:
        hard.append("mojibake_or_broken_encoding")
    elif mojibake_hits:
        warnings.append("mojibake_or_broken_encoding_hint")
    if ADULT_SHOP_RE.search(text):
        # Casino as a hotel/place can be false positive, so this is a warning
        # for prose and a hard drop only when paired with spammy surface cues.
        lower = text.lower()
        if "checkout" in lower or "rabattcode" in lower or "warenkorb" in lower:
            hard.append("adult_shop_spam")
        else:
            warnings.append("adult_shop_casino_term")
    url_count = len(URL_RE.findall(text))
    if profile in {"qa", "dialogue", "math"}:
        if url_ratio > 0.25 or url_count >= 5:
            hard.append("url_dense")
        elif url_count:
            warnings.append("url_present")
    elif url_ratio > 0.25 or url_count >= 3:
        hard.append("url_dense")
    elif url_count:
        warnings.append("url_present")

    if words < 16:
        hard.append("too_few_words")
    elif words < 32 and profile in {"german_prose", "validation"}:
        warnings.append("short_doc")

    if whitespace > 0.45:
        hard.append("whitespace_extreme")
    elif whitespace > 0.35:
        warnings.append("whitespace_high")

    if profile in {"german_prose", "validation"}:
        if long_word > 300:
            hard.append("very_long_token")
        elif long_word > 80:
            warnings.append("long_token")
    elif long_word > 80:
        warnings.append("long_token")

    if (line_count >= 4 and repeated_lines > 0.85) or (
        paragraph_count >= 4 and repeated_paragraphs > 0.85
    ):
        hard.append("line_or_paragraph_repetition_extreme")
    elif (line_count >= 4 and repeated_lines > 0.70) or (
        paragraph_count >= 4 and repeated_paragraphs > 0.70
    ):
        warnings.append("line_or_paragraph_repetition")

    if profile in {"german_prose", "validation"} and (dup_3gram > 0.75 or top_3gram > 0.65):
        hard.append("ngram_repetition_extreme")
    elif dup_3gram > 0.35 or top_3gram > 0.25:
        warnings.append("ngram_repetition")

    if profile == "math":
        if non_alpha > 0.65 or symbols > 0.55:
            warnings.append("math_symbol_heavy")
        if alpha_words < 0.45:
            warnings.append("math_low_alpha_words")
        if long_word > 160:
            warnings.append("math_long_token")
        if LIST_RE.search(text) and bullets > 0.80:
            warnings.append("math_list_like")
        return warnings, hard

    if profile in {"qa", "dialogue"}:
        if non_alpha > 0.40:
            warnings.append("non_alpha_numeric_high")
        if numbers > 0.35:
            warnings.append("number_heavy")
        if long_word > 160:
            warnings.append("long_token")
        if LIST_RE.search(text):
            warnings.append("list_like_hint")
        if punctuation > 0.95:
            warnings.append("punctuation_unusual")
        return warnings, hard

    # German prose / validation profile.
    if non_alpha > 0.45:
        hard.append("non_alpha_numeric_extreme")
    elif non_alpha > 0.25:
        warnings.append("non_alpha_numeric_high")
    if numbers > 0.30:
        warnings.append("number_heavy")
    if alpha_words < 0.55:
        hard.append("too_few_alpha_words")
    elif alpha_words < 0.65:
        warnings.append("low_alpha_words")
    if bullets > 0.80:
        warnings.append("bullet_list")
    if LIST_RE.search(text):
        warnings.append("list_like_hint")
    if punctuation > 0.95:
        warnings.append("punctuation_unusual")
    if ellipsis > 0.30:
        warnings.append("ellipsis_heavy")

    return warnings, hard


def audit_rows(rows: list[tuple[int, SourceSpan, str]], examples_per_reason: int) -> tuple[list[AuditDoc], dict[str, Any]]:
    filters = curator_filters()
    audited: list[AuditDoc] = []
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_source_total: Counter[str] = Counter()
    per_profile_total: Counter[str] = Counter()
    per_source_hard: Counter[str] = Counter()
    per_profile_hard: Counter[str] = Counter()
    per_source_warn: Counter[str] = Counter()
    per_profile_warn: Counter[str] = Counter()
    hard_reasons: Counter[str] = Counter()
    warning_reasons: Counter[str] = Counter()
    score_values: dict[str, list[float]] = defaultdict(list)
    char_lengths: list[int] = []
    word_lengths: list[int] = []

    for line_no, span, text in rows:
        profile = effective_profile(span.profile, text)
        scores = score_text(text, filters)
        words = len(WORD_RE.findall(text))
        warnings, hard_drops = decide(profile, text, scores, words)
        doc = AuditDoc(
            line_no=line_no,
            source=span.name,
            profile=profile,
            text=text,
            chars=len(text),
            words=words,
            scores=scores,
            warnings=warnings,
            hard_drops=hard_drops,
        )
        audited.append(doc)
        per_source_total[span.name] += 1
        per_profile_total[profile] += 1
        char_lengths.append(len(text))
        word_lengths.append(words)

        for key, value in scores.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                score_values[key].append(float(value))

        if hard_drops:
            per_source_hard[span.name] += 1
            per_profile_hard[profile] += 1
        if warnings:
            per_source_warn[span.name] += 1
            per_profile_warn[profile] += 1

        for reason in hard_drops:
            hard_reasons[reason] += 1
            bucket = "HARD/" + reason
            if len(examples[bucket]) < examples_per_reason:
                examples[bucket].append(example_row(doc))
        for reason in warnings:
            warning_reasons[reason] += 1
            bucket = "WARN/" + reason
            if len(examples[bucket]) < examples_per_reason:
                examples[bucket].append(example_row(doc))

    total = len(audited)
    score_summary = {
        key: {
            "median": q(values, 0.5),
            "p95": q(values, 0.95),
            "max": max(values) if values else None,
        }
        for key, values in score_values.items()
    }
    summary = {
        "docs": total,
        "hard_drop_docs": sum(1 for d in audited if d.hard_drops),
        "warning_docs": sum(1 for d in audited if d.warnings),
        "hard_drop_pct": pct(sum(1 for d in audited if d.hard_drops), total),
        "warning_pct": pct(sum(1 for d in audited if d.warnings), total),
        "chars": {
            "min": min(char_lengths) if char_lengths else None,
            "median": q(char_lengths, 0.5),
            "p95": q(char_lengths, 0.95),
            "max": max(char_lengths) if char_lengths else None,
            "mean": round(statistics.mean(char_lengths), 1) if char_lengths else None,
        },
        "words": {
            "min": min(word_lengths) if word_lengths else None,
            "median": q(word_lengths, 0.5),
            "p95": q(word_lengths, 0.95),
            "max": max(word_lengths) if word_lengths else None,
            "mean": round(statistics.mean(word_lengths), 1) if word_lengths else None,
        },
        "per_source": source_profile_table(per_source_total, per_source_hard, per_source_warn),
        "per_profile": source_profile_table(per_profile_total, per_profile_hard, per_profile_warn),
        "hard_reasons": counter_table(hard_reasons, total),
        "warning_reasons": counter_table(warning_reasons, total),
        "score_summary": score_summary,
        "examples": dict(examples),
    }
    return audited, summary


def example_row(doc: AuditDoc) -> dict[str, Any]:
    return {
        "line_no": doc.line_no,
        "source": doc.source,
        "profile": doc.profile,
        "chars": doc.chars,
        "words": doc.words,
        "hard_drops": doc.hard_drops,
        "warnings": doc.warnings,
        "text": clean_preview(doc.text),
    }


def counter_table(counter: Counter[str], denominator: int) -> list[dict[str, Any]]:
    return [
        {"reason": key, "count": value, "pct": pct(value, denominator)}
        for key, value in counter.most_common()
    ]


def source_profile_table(
    totals: Counter[str], hard: Counter[str], warnings: Counter[str]
) -> list[dict[str, Any]]:
    rows = []
    for key, total in totals.most_common():
        rows.append(
            {
                "name": key,
                "sample_docs": total,
                "hard_drop_docs": hard[key],
                "hard_drop_pct": pct(hard[key], total),
                "warning_docs": warnings[key],
                "warning_pct": pct(warnings[key], total),
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_report(path: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    lines: list[str] = []
    lines.append("# NeMo Curator Profile Audit")
    lines.append("")
    lines.append("This is a source-aware audit. Curator filters provide scores; profile logic decides whether a signal is a hard drop or a warning.")
    lines.append("")
    lines.append("## Run")
    lines.append("")
    lines.append(f"- mix: `{args.mix}`")
    lines.append(f"- manifest: `{args.manifest}`")
    lines.append(f"- sample docs: {summary['docs']:,}")
    lines.append(f"- seed: {args.seed}")
    lines.append(f"- hard-drop candidates: {summary['hard_drop_docs']:,} ({summary['hard_drop_pct']}%)")
    lines.append(f"- warning-only candidates: {summary['warning_docs']:,} ({summary['warning_pct']}%)")
    lines.append(
        f"- chars median/p95/max: {summary['chars']['median']:,} / {summary['chars']['p95']:,} / {summary['chars']['max']:,}"
    )
    lines.append(
        f"- words median/p95/max: {summary['words']['median']:,} / {summary['words']['p95']:,} / {summary['words']['max']:,}"
    )
    lines.append("")
    lines.append("## Recommended Interpretation")
    lines.append("")
    lines.append("- Treat `hard_drop` reasons as safe candidates for the next builder/profile gate.")
    lines.append("- Treat warnings as audit signals; do not drop them globally without a per-profile review.")
    lines.append("- Math and QA intentionally tolerate symbol and number density.")
    lines.append("- Punctuation/list signals are warnings for German prose because lexicon/list articles can be useful but may hurt long-form style.")
    lines.append("")
    lines.append("## v6 Builder Gate Draft")
    lines.append("")
    lines.append("```python")
    lines.extend(builder_gate_draft(summary))
    lines.append("```")
    lines.append("")
    lines.append("## Per Profile")
    lines.append("")
    lines.append("| profile | sample | hard drop | warning |")
    lines.append("|---|---:|---:|---:|")
    for row in summary["per_profile"]:
        lines.append(
            f"| `{row['name']}` | {row['sample_docs']:,} | "
            f"{row['hard_drop_docs']:,} ({row['hard_drop_pct']}%) | "
            f"{row['warning_docs']:,} ({row['warning_pct']}%) |"
        )
    lines.append("")
    lines.append("## Per Source")
    lines.append("")
    lines.append("| source | sample | hard drop | warning |")
    lines.append("|---|---:|---:|---:|")
    for row in summary["per_source"]:
        lines.append(
            f"| `{row['name']}` | {row['sample_docs']:,} | "
            f"{row['hard_drop_docs']:,} ({row['hard_drop_pct']}%) | "
            f"{row['warning_docs']:,} ({row['warning_pct']}%) |"
        )
    lines.append("")
    lines.append("## Hard Drop Reasons")
    lines.append("")
    lines.append("| reason | count | pct |")
    lines.append("|---|---:|---:|")
    for row in summary["hard_reasons"]:
        lines.append(f"| `{row['reason']}` | {row['count']:,} | {row['pct']}% |")
    lines.append("")
    lines.append("## Warning Reasons")
    lines.append("")
    lines.append("| reason | count | pct |")
    lines.append("|---|---:|---:|")
    for row in summary["warning_reasons"][:30]:
        lines.append(f"| `{row['reason']}` | {row['count']:,} | {row['pct']}% |")
    lines.append("")
    lines.append("## Examples")
    for bucket, examples in summary["examples"].items():
        lines.append("")
        lines.append(f"### `{bucket}`")
        lines.append("")
        for ex in examples[: args.examples_per_reason]:
            lines.append(
                f"- line `{ex['line_no']}` source=`{ex['source']}` profile=`{ex['profile']}` "
                f"chars={ex['chars']} words={ex['words']}: {ex['text']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def builder_gate_draft(summary: dict[str, Any]) -> list[str]:
    hard_reasons = {row["reason"] for row in summary.get("hard_reasons", [])}
    lines = [
        "# Conservative hard gates inferred from this audit.",
        "# Keep warnings as observability counters until manually reviewed.",
        "hard_reasons = set()",
    ]
    for reason in [
        "html_or_entity",
        "mojibake_or_broken_encoding",
        "very_long_token",
        "chat_marker",
        "url_dense",
        "adult_shop_spam",
        "line_or_paragraph_repetition_extreme",
        "ngram_repetition_extreme",
        "non_alpha_numeric_extreme",
        "whitespace_extreme",
    ]:
        if reason in hard_reasons:
            lines.append(f"hard_reasons.add({reason!r})")
    if len(lines) == 3:
        lines.append("# no recurring hard gate was observed in this sample")
    lines.extend(
        [
            "",
            "# Do not auto-drop these yet; report them per source/profile.",
            "warning_only = {",
            "    'punctuation_unusual', 'list_like_hint', 'ellipsis_heavy',",
            "    'adult_shop_casino_term', 'url_present', 'long_token',",
            "}",
        ]
    )
    return lines


def hard_signal_reasons(profile: str, text: str) -> list[str]:
    """Fast, profile-light hard signal pass for full-mix scanning."""
    words = WORD_RE.findall(text)
    reasons: list[str] = []
    html_hits = len(HTML_RE.findall(text))
    if html_hits >= 3 or re.search(r"<\s*/?\s*(script|style|iframe|table)\b", text, re.I):
        reasons.append("html_or_entity")
    if CHAT_RE.search(text):
        reasons.append("chat_marker")
    if len(MOJIBAKE_RE.findall(text)) >= 2:
        reasons.append("mojibake_or_broken_encoding")
    if profile in {"german_prose", "validation"} and LONG_TOKEN_RE.search(text):
        reasons.append("very_long_token")
    if URL_RE.search(text) and url_char_ratio(text) > 0.25:
        reasons.append("url_dense")
    if profile in {"german_prose", "validation"} and len(words) >= 30 and alpha_word_ratio(words) < 0.55:
        reasons.append("too_few_alpha_words")
    return sorted(set(reasons))


def hard_signal_fullscan(
    mix_path: Path,
    spans: list[SourceSpan],
    examples_per_reason: int,
    progress_every: int,
    max_docs: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ends = [span.end for span in spans]
    reason_counts: Counter[str] = Counter()
    source_counts: dict[str, Counter[str]] = defaultdict(Counter)
    profile_counts: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    candidates: list[dict[str, Any]] = []
    docs_with_signal = 0
    total = 0
    started = time.time()

    with mix_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            if max_docs is not None and line_no > max_docs:
                break
            total = line_no
            text = line.rstrip("\n")
            span = source_for_line(line_no, spans, ends)
            profile = effective_profile(span.profile, text)
            reasons = hard_signal_reasons(profile, text)
            if not reasons:
                if progress_every and line_no % progress_every == 0:
                    elapsed = time.time() - started
                    print(
                        f"progress docs={line_no:,} signaled={docs_with_signal:,} elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                continue

            docs_with_signal += 1
            row = {
                "line_no": line_no,
                "source": span.name,
                "profile": profile,
                "chars": len(text),
                "words": len(WORD_RE.findall(text)),
                "hard_drops": reasons,
                "text": clean_preview(text, 900),
            }
            candidates.append(row)
            for reason in reasons:
                reason_counts[reason] += 1
                source_counts[span.name][reason] += 1
                profile_counts[profile][reason] += 1
                if len(examples[reason]) < examples_per_reason:
                    examples[reason].append(row)

            if progress_every and line_no % progress_every == 0:
                elapsed = time.time() - started
                print(
                    f"progress docs={line_no:,} signaled={docs_with_signal:,} elapsed={elapsed:.1f}s",
                    flush=True,
                )

    summary = {
        "mode": "hard-scan",
        "docs": total,
        "max_docs": max_docs,
        "hard_signal_docs": docs_with_signal,
        "hard_signal_pct": pct(docs_with_signal, total),
        "elapsed_sec": round(time.time() - started, 1),
        "reason_counts": counter_table(reason_counts, total),
        "per_source_reason_counts": {
            source: counter_table(counter, total)
            for source, counter in sorted(
                source_counts.items(), key=lambda item: sum(item[1].values()), reverse=True
            )
        },
        "per_profile_reason_counts": {
            profile: counter_table(counter, total)
            for profile, counter in sorted(
                profile_counts.items(), key=lambda item: sum(item[1].values()), reverse=True
            )
        },
        "examples": dict(examples),
    }
    return summary, candidates


def write_hard_scan_report(path: Path, summary: dict[str, Any], args: argparse.Namespace) -> None:
    lines: list[str] = []
    lines.append("# NeMo Curator Hard Signal Fullscan")
    lines.append("")
    lines.append("This mode skips expensive Curator scores and scans the full mix for the conservative hard signals tuned by the sample audit.")
    lines.append("")
    lines.append("## Run")
    lines.append("")
    lines.append(f"- mix: `{args.mix}`")
    lines.append(f"- manifest: `{args.manifest}`")
    lines.append(f"- docs: {summary['docs']:,}")
    if summary.get("max_docs"):
        lines.append(f"- max docs: {summary['max_docs']:,}")
    lines.append(f"- hard-signal docs: {summary['hard_signal_docs']:,} ({summary['hard_signal_pct']}%)")
    lines.append(f"- elapsed sec: {summary['elapsed_sec']}")
    lines.append("")
    lines.append("## v6 Builder Gate Draft")
    lines.append("")
    lines.append("```python")
    lines.extend(builder_gate_draft({"hard_reasons": summary["reason_counts"]}))
    lines.append("```")
    lines.append("")
    lines.append("## Reason Counts")
    lines.append("")
    lines.append("| reason | count | pct |")
    lines.append("|---|---:|---:|")
    for row in summary["reason_counts"]:
        lines.append(f"| `{row['reason']}` | {row['count']:,} | {row['pct']}% |")
    lines.append("")
    lines.append("## Per Profile Reason Counts")
    for profile, rows in summary["per_profile_reason_counts"].items():
        lines.append("")
        lines.append(f"### `{profile}`")
        lines.append("")
        lines.append("| reason | count | pct of full mix |")
        lines.append("|---|---:|---:|")
        for row in rows:
            lines.append(f"| `{row['reason']}` | {row['count']:,} | {row['pct']}% |")
    lines.append("")
    lines.append("## Top Source Reason Counts")
    for source, rows in list(summary["per_source_reason_counts"].items())[:30]:
        lines.append("")
        lines.append(f"### `{source}`")
        lines.append("")
        lines.append("| reason | count | pct of full mix |")
        lines.append("|---|---:|---:|")
        for row in rows:
            lines.append(f"| `{row['reason']}` | {row['count']:,} | {row['pct']}% |")
    lines.append("")
    lines.append("## Examples")
    for reason, examples in summary["examples"].items():
        lines.append("")
        lines.append(f"### `{reason}`")
        for ex in examples[: args.examples_per_reason]:
            lines.append(
                f"- line `{ex['line_no']}` source=`{ex['source']}` profile=`{ex['profile']}` "
                f"chars={ex['chars']} words={ex['words']}: {ex['text']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", type=Path, default=DEFAULT_MIX)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--mode",
        choices=("sample", "hard-scan", "both"),
        default="sample",
        help="sample runs the Curator-backed audit; hard-scan scans the full mix for conservative regex hard signals.",
    )
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=5240524)
    parser.add_argument("--examples-per-reason", type=int, default=8)
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000_000,
        help="For hard-scan mode, print progress every N documents. Use 0 to disable.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=0,
        help="For hard-scan smoke tests, stop after N documents. 0 means scan the whole mix.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    spans = load_spans(args.manifest)
    if not spans:
        raise SystemExit(f"No source spans in manifest: {args.manifest}")
    total_docs = max(span.end for span in spans)

    if args.mode in {"sample", "both"}:
        sample_dir = args.output_dir if args.mode == "sample" else args.output_dir / "sample_audit"
        sample_dir.mkdir(parents=True, exist_ok=True)
        line_numbers = sample_line_numbers(total_docs, args.samples, args.seed)
        rows = read_samples(args.mix, line_numbers, spans)
        audited, summary = audit_rows(rows, args.examples_per_reason)

        sample_rows = [
            {
                "line_no": line_no,
                "source": span.name,
                "source_profile": span.profile,
                "profile": effective_profile(span.profile, text),
                "text": text,
            }
            for line_no, span, text in rows
        ]
        write_jsonl(sample_dir / "sample.jsonl", sample_rows)
        write_jsonl(
            sample_dir / "hard_drop_candidates.jsonl",
            [example_row(doc) | {"scores": doc.scores} for doc in audited if doc.hard_drops],
        )
        write_jsonl(
            sample_dir / "warning_candidates.jsonl",
            [example_row(doc) | {"scores": doc.scores} for doc in audited if doc.warnings],
        )
        (sample_dir / "results.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_report(sample_dir / "report.md", summary, args)
        print(
            json.dumps(
                {
                    "mode": "sample",
                    **{
                        k: summary[k]
                        for k in (
                            "docs",
                            "hard_drop_docs",
                            "hard_drop_pct",
                            "warning_docs",
                            "warning_pct",
                        )
                    },
                    "report": str(sample_dir / "report.md"),
                },
                indent=2,
            )
        )

    if args.mode in {"hard-scan", "both"}:
        scan_dir = args.output_dir if args.mode == "hard-scan" else args.output_dir / "hard_signal_fullscan"
        scan_dir.mkdir(parents=True, exist_ok=True)
        summary, candidates = hard_signal_fullscan(
            args.mix,
            spans,
            args.examples_per_reason,
            args.progress_every,
            args.max_docs or None,
        )
        (scan_dir / "results.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_jsonl(scan_dir / "hard_signal_candidates.jsonl", candidates)
        write_hard_scan_report(scan_dir / "report.md", summary, args)
        print(
            json.dumps(
                {
                    "mode": "hard-scan",
                    "docs": summary["docs"],
                    "hard_signal_docs": summary["hard_signal_docs"],
                    "hard_signal_pct": summary["hard_signal_pct"],
                    "report": str(scan_dir / "report.md"),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
