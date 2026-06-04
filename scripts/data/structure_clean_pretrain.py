#!/usr/bin/env python3
"""Structure-aware cleaning pass for prose pretraining data.

This pass is meant to run after raw extraction and before final mixing. It is
stricter than ``filter_quality.py``: instead of only accepting or rejecting
lines, it rewrites surviving documents into clean prose paragraphs, drops
navigation/list/HTML leftovers, scores the result, and writes a manifest with
drop reasons.

The JSONL output keeps paragraph structure for auditing. The optional text
output is one clean training document per line, which matches the current
tokenizer/pretraining pipeline.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
HTML_TAG_RE = re.compile(r"<[^>]{1,160}>")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
EMAIL_RE = re.compile(r"\b\S+@\S+\.\S+\b")
SPACE_RE = re.compile(r"[ \t\u00a0]{2,}")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)]|[a-zA-Z][.)])\s+")
HEADER_NOISE_RE = re.compile(
    r"^(?:home|menu|navigation|login|log in|sign in|register|teilen|share|"
    r"mehr lesen|read more|weiterlesen|zurück|kontakt|impressum|datenschutz|"
    r"cookie|cookies|newsletter|abo|subscribe|advertisement|anzeige)$",
    re.I,
)
BAD_FRAGMENT_RE = re.compile(
    r"(?:_end_of_the_data|_user-data|<\|im_start\|>|<\|im_end\|>|"
    r"javascript is disabled|accept all cookies|cookie policy|privacy policy|"
    r"all rights reserved|subscribe to our newsletter|sign in to continue|"
    r"\bnewsletter\b|advertisement|anzeige)",
    re.I,
)
TEMPLATE_RE = re.compile(
    r"(?:###\s*(?:aufgabe|antwort|instruction|response)|\bfrage\s*:|\bantwort\s*:)",
    re.I,
)
LIST_ARTICLE_RE = re.compile(
    r"(?:\b(?:heissen|heißen) folgende\b|\bfolgende geographische objekte\b|"
    r"\bsteht (?:fuer|für)\s*:|\bbezeichnet\s*:|\bliste der\b)",
    re.I,
)
MOJIBAKE_HINTS = (
    "Ã¤",
    "Ã¶",
    "Ã¼",
    "ÃŸ",
    "Ã„",
    "Ã–",
    "Ãœ",
    "â€",
    "Â ",
    "�",
)
DE_STOPWORDS = {
    "der",
    "die",
    "das",
    "und",
    "oder",
    "nicht",
    "ist",
    "sind",
    "ein",
    "eine",
    "einer",
    "einen",
    "mit",
    "auf",
    "von",
    "zu",
    "im",
    "im",
    "den",
    "dem",
    "des",
    "dass",
    "auch",
    "als",
    "wie",
    "wird",
    "werden",
    "fuer",
    "ueber",
    "für",
    "durch",
    "bei",
    "aus",
    "nach",
    "über",
}


@dataclass
class CleanedDocument:
    text: str
    paragraphs: list[str]
    score: float
    metrics: dict[str, float]
    hash: str


@dataclass
class Manifest:
    input_file: str
    jsonl_output: str
    text_output: str | None
    started_at: float
    finished_at: float = 0.0
    docs_in: int = 0
    docs_written: int = 0
    bytes_in: int = 0
    bytes_written_jsonl: int = 0
    bytes_written_text: int = 0
    dropped: Counter = field(default_factory=Counter)


def mojibake_score(text: str) -> int:
    return sum(text.count(marker) for marker in MOJIBAKE_HINTS) + text.count("�") * 3


def repair_mojibake(text: str) -> str:
    if not any(marker in text for marker in MOJIBAKE_HINTS):
        return text
    candidates = [text]
    for enc in ("latin1", "cp1252"):
        try:
            candidates.append(text.encode(enc).decode("utf-8"))
        except UnicodeError:
            pass
    return min(candidates, key=mojibake_score)


def normalize_text(text: str) -> str:
    text = repair_mojibake(text)
    text = html.unescape(text)
    text = CONTROL_RE.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = text.replace("“", '"').replace("”", '"').replace("„", '"')
    text = text.replace("’", "'").replace("‘", "'").replace("`", "'")
    text = text.replace("�", " ")  # drop stray Unicode replacement chars (encoding damage)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(SPACE_RE.sub(" ", line).strip() for line in text.splitlines()).strip()


def _line_is_noise(line: str) -> bool:
    stripped = LIST_PREFIX_RE.sub("", line).strip(" .:;|/-")
    if not stripped:
        return True
    if HEADER_NOISE_RE.match(stripped):
        return True
    words = WORD_RE.findall(stripped)
    if len(words) <= 2 and len(stripped) < 32:
        return True
    if BAD_FRAGMENT_RE.search(stripped):
        return True
    return False


_CRUMB_LABELS = (
    "sie befinden sich hier", "sie sind hier", "aktuelle seite",
    "you are here", "navigation:", "startseite", "you are at",
)
# On a line we already KNOW starts with a nav label, these separators reliably
# delimit crumbs, so cutting at the last one in the head is safe.
_CRUMB_SEP_RE = re.compile(r"\s*(?:>>|>|//|/|›|»|·|\|)\s*")


def _cut_at_duplicated_phrase(text: str, window: int = 30) -> str:
    """Breadcrumb pages typically end the trail with the page title, then repeat
    it as the article heading ('... Entropie Isothermer Prozess Isothermer
    Prozess Zustand...'). Cut at the second occurrence of the first back-to-back
    repeated phrase in the head, keeping the content. Longer repeats win."""
    toks = text.split()
    n = min(len(toks), window)
    for k in range(6, 0, -1):
        for i in range(0, n - 2 * k + 1):
            if toks[i:i + k] == toks[i + k:i + 2 * k]:
                return " ".join(toks[i + k:])
    return text


def strip_breadcrumbs(line: str) -> str:
    """Remove a leading breadcrumb/navigation trail that sits on the same line
    as real content, e.g. 'Home | A | B | Echter Text' or 'Sie befinden sich
    hier: Startseite > X > Echter Text'. Conservative on un-labeled lines (only
    stacked '|'), more aggressive once a nav label is matched."""
    head = line[:160]
    low = head.lower()
    if low.startswith(_CRUMB_LABELS):
        seps = list(_CRUMB_SEP_RE.finditer(head))
        if seps:
            return line[seps[-1].end():].strip()
        # no nav separator (space-separated crumbs): drop the label, then cut
        # at the duplicated page-title that breadcrumb pages repeat as the H1.
        colon = head.find(":")
        rest = line[colon + 1:].strip() if 0 <= colon <= 24 else line
        return _cut_at_duplicated_phrase(rest)
    # bare pipe trail: 'X | Y | Z | content' with short crumb segments
    if head.count("|") >= 2:
        cut = head.rfind("|") + 1
        segs = [s.strip() for s in head[:cut].split("|") if s.strip()]
        if segs and all(len(s) <= 40 for s in segs):
            return line[cut:].strip()
    return line


def remove_boilerplate_lines(text: str) -> str:
    kept: list[str] = []
    for raw in text.splitlines():
        line = strip_breadcrumbs(raw.strip())
        if _line_is_noise(line):
            continue
        kept.append(LIST_PREFIX_RE.sub("", line))
    return "\n".join(kept)


def sentence_split(text: str) -> list[str]:
    text = " ".join(text.split())
    parts = SENTENCE_END_RE.split(text)
    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if len(part) < 24:
            continue
        if BAD_FRAGMENT_RE.search(part):
            continue
        sentences.append(part)
    return sentences


def build_paragraphs(sentences: Iterable[str], *, target_chars: int, max_chars: int) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        add_len = len(sentence) + (1 if current else 0)
        if current and current_len + add_len > max_chars:
            paragraphs.append(" ".join(current))
            current = []
            current_len = 0
        current.append(sentence)
        current_len += add_len
        if current_len >= target_chars and len(current) >= 2:
            paragraphs.append(" ".join(current))
            current = []
            current_len = 0
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def repetition_ratio(words: list[str], ngram: int = 4) -> float:
    if len(words) < ngram * 2:
        return 0.0
    grams = [tuple(words[i : i + ngram]) for i in range(len(words) - ngram + 1)]
    return 1.0 - (len(set(grams)) / max(1, len(grams)))


def language_signal(words: list[str]) -> float:
    if not words:
        return 0.0
    hits = sum(1 for word in words if word.lower() in DE_STOPWORDS)
    return hits / max(1, len(words))


def compute_metrics(text: str, paragraphs: list[str]) -> dict[str, float]:
    chars = len(text)
    words = WORD_RE.findall(text)
    alpha = sum(1 for c in text if c.isalpha()) / max(1, chars)
    symbol = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(1, chars)
    digits = sum(1 for c in text if c.isdigit()) / max(1, chars)
    sentence_count = sum(max(1, len(sentence_split(p))) for p in paragraphs)
    avg_word_len = sum(len(w) for w in words) / max(1, len(words))
    return {
        "chars": float(chars),
        "words": float(len(words)),
        "paragraphs": float(len(paragraphs)),
        "sentences": float(sentence_count),
        "alpha_ratio": round(alpha, 4),
        "symbol_ratio": round(symbol, 4),
        "digit_ratio": round(digits, 4),
        "repetition_ratio": round(repetition_ratio([w.lower() for w in words]), 4),
        "language_signal": round(language_signal(words), 4),
        "avg_word_len": round(avg_word_len, 2),
    }


def quality_score(metrics: dict[str, float]) -> float:
    score = 1.0
    if metrics["words"] < 80:
        score -= 0.25
    if metrics["sentences"] < 3:
        score -= 0.2
    if metrics["alpha_ratio"] < 0.55:
        score -= 0.2
    if metrics["symbol_ratio"] > 0.18:
        score -= min(0.25, (metrics["symbol_ratio"] - 0.18) * 1.5)
    if metrics["digit_ratio"] > 0.25:
        score -= 0.15
    if metrics["repetition_ratio"] > 0.18:
        score -= min(0.35, metrics["repetition_ratio"])
    if metrics["language_signal"] < 0.035:
        score -= 0.18
    if not 3.0 <= metrics["avg_word_len"] <= 10.5:
        score -= 0.08
    return round(max(0.0, min(1.0, score)), 4)


def stable_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


def looks_like_list_article(text: str) -> bool:
    head = text[:2_500]
    if not LIST_ARTICLE_RE.search(head):
        return False
    punctuation_items = head.count(",") + head.count(";")
    short_fragments = sum(1 for part in re.split(r"[.;]", head) if 0 < len(part.split()) <= 8)
    return punctuation_items >= 8 or short_fragments >= 8


def clean_document(
    raw: str,
    *,
    min_words: int,
    min_score: float,
    target_paragraph_chars: int,
    max_paragraph_chars: int,
    min_language_signal: float = 0.10,
) -> tuple[CleanedDocument | None, str | None]:
    text = normalize_text(raw)
    if not text:
        return None, "empty"
    if mojibake_score(text) > 0:
        return None, "mojibake"
    if BAD_FRAGMENT_RE.search(text):
        return None, "boilerplate"
    if len(TEMPLATE_RE.findall(text)) >= 2:
        return None, "template"
    if looks_like_list_article(text):
        return None, "list_article"

    text = remove_boilerplate_lines(text)
    sentences = sentence_split(text)
    if len(sentences) < 2:
        return None, "too_few_sentences"
    paragraphs = build_paragraphs(
        sentences,
        target_chars=target_paragraph_chars,
        max_chars=max_paragraph_chars,
    )
    paragraphs = [p for p in paragraphs if len(WORD_RE.findall(p)) >= 12]
    if not paragraphs:
        return None, "too_short"

    structured = "\n\n".join(paragraphs).strip()
    metrics = compute_metrics(structured, paragraphs)
    if metrics["words"] < min_words:
        return None, "too_few_words"
    if metrics["language_signal"] < min_language_signal:
        return None, "language_mismatch"
    score = quality_score(metrics)
    if score < min_score:
        return None, "low_quality"
    return (
        CleanedDocument(
            text=structured,
            paragraphs=paragraphs,
            score=score,
            metrics=metrics,
            hash=stable_hash(structured),
        ),
        None,
    )


def manifest_payload(manifest: Manifest) -> dict:
    payload = asdict(manifest)
    payload["elapsed_seconds"] = round(manifest.finished_at - manifest.started_at, 3)
    payload["dropped"] = dict(manifest.dropped.most_common())
    return payload


def run(args: argparse.Namespace) -> dict:
    manifest = Manifest(
        input_file=str(args.input),
        jsonl_output=str(args.output_jsonl),
        text_output=str(args.output_text) if args.output_text else None,
        started_at=time.time(),
        bytes_in=args.input.stat().st_size,
    )
    seen: set[str] = set()
    seen_fifo: deque[str] = deque()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.output_text:
        args.output_text.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", encoding="utf-8", errors="replace") as src, args.output_jsonl.open(
        "w", encoding="utf-8", newline="\n"
    ) as jsonl:
        text_fh = args.output_text.open("w", encoding="utf-8", newline="\n") if args.output_text else None
        try:
            for raw in src:
                manifest.docs_in += 1
                doc, reason = clean_document(
                    raw,
                    min_words=args.min_words,
                    min_score=args.min_score,
                    target_paragraph_chars=args.target_paragraph_chars,
                    max_paragraph_chars=args.max_paragraph_chars,
                    min_language_signal=args.min_language_signal,
                )
                if doc is None:
                    manifest.dropped[reason or "unknown"] += 1
                    continue
                if args.dedup_cache:
                    if doc.hash in seen:
                        manifest.dropped["duplicate"] += 1
                        continue
                    seen.add(doc.hash)
                    seen_fifo.append(doc.hash)
                    if len(seen_fifo) > args.dedup_cache:
                        seen.discard(seen_fifo.popleft())

                record = {
                    "text": doc.text,
                    "paragraphs": doc.paragraphs,
                    "quality_score": doc.score,
                    "metrics": doc.metrics,
                    "hash": doc.hash,
                }
                line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                jsonl.write(line)
                manifest.bytes_written_jsonl += len(line.encode("utf-8"))
                if text_fh:
                    train_text = " ".join(doc.text.split())
                    text_fh.write(train_text + "\n")
                    manifest.bytes_written_text += len((train_text + "\n").encode("utf-8"))
                manifest.docs_written += 1
                if args.flush_every and manifest.docs_written % args.flush_every == 0:
                    jsonl.flush()
                    if text_fh:
                        text_fh.flush()
                    print(
                        f"progress docs_in={manifest.docs_in:,} docs_written={manifest.docs_written:,} "
                        f"dropped={sum(manifest.dropped.values()):,}",
                        flush=True,
                    )
                if args.max_docs and manifest.docs_written >= args.max_docs:
                    break
        finally:
            if text_fh:
                text_fh.close()

    manifest.finished_at = time.time()
    out_manifest = args.output_jsonl.with_suffix(args.output_jsonl.suffix + ".manifest.json")
    out_manifest.write_text(json.dumps(manifest_payload(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output_jsonl} ({manifest.docs_written:,} docs)")
    if args.output_text:
        print(f"wrote {args.output_text}")
    print(f"wrote {out_manifest}")
    print(f"dropped: {dict(manifest.dropped.most_common(12))}")
    return manifest_payload(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--output-text", type=Path, default=None)
    parser.add_argument("--min-words", type=int, default=80)
    parser.add_argument("--min-score", type=float, default=0.62)
    parser.add_argument("--min-language-signal", type=float, default=0.10)
    parser.add_argument("--target-paragraph-chars", type=int, default=650)
    parser.add_argument("--max-paragraph-chars", type=int, default=1100)
    parser.add_argument("--dedup-cache", type=int, default=1_000_000)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--flush-every", type=int, default=0)
    args = parser.parse_args()
    if args.max_paragraph_chars < args.target_paragraph_chars:
        raise SystemExit("--max-paragraph-chars must be >= --target-paragraph-chars")
    run(args)


if __name__ == "__main__":
    main()
