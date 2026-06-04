#!/usr/bin/env python3
"""Strict streaming filter for base-model pretraining text.

Use this for a new-from-scratch run when we care more about clean signal than
raw volume. It reads one document per line, repairs common encoding drift, drops
HTML/URL/boilerplate/repetition, applies a light language check for prose, and
writes a manifest with drop reasons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df]+")
HTML_RE = re.compile(r"<\s*/?\s*(?:html|body|div|script|style|a|span|table|iframe|form|nav|footer)\b", re.I)
URL_RE = re.compile(r"https?://|www\.", re.I)
BAD_RE = re.compile(
    r"(cookie policy|privacy policy|accept all cookies|javascript is disabled|"
    r"_end_of_the_data|user-data|<\|im_start\|>|<\|im_end\|>|"
    r"sign in to continue|subscribe to our newsletter)",
    re.I,
)
WEB_BOILERPLATE_RE = re.compile(
    r"(zwei klicks f(?:ü|u)r mehr datenschutz|teilen-button|"
    r"verfassen sie die erste bewertung|erste bewertung verfassen|bewertung verfassen|"
    r"newsletter abonnieren|cookie-einstellungen|alle preise inkl|gratis versand|"
    r"jetzt kaufen|in den warenkorb|warenkorb|lieferzeit|"
    r"weiterlesen\s+[A-ZÄÖÜ0-9]|read more|"
    r"facebook,\s*google\+|twitter senden|"
    r"bitte f(?:ü|u)llen sie .*formular|kein rechtsanspruch)",
    re.I,
)
COMMERCIAL_BOILERPLATE_RE = re.compile(
    r"(about us|ueber uns|über uns|impressum|kontaktformular|"
    r"agb|allgemeine geschaeftsbedingungen|allgemeine geschäftsbedingungen|"
    r"datenschutzerklaerung|datenschutzerklärung|"
    r"versandkosten|versand und zahlung|lieferung und zahlung|"
    r"preisvergleich|rabattcode|gutschein|warenkorb|checkout|"
    r"produktbeschreibung|kunden kauften auch|"
    r"trusted shops|bewertungen? schreiben|"
    r"erotik,\s*wohlbefinden|wellness,\s*partnerschaft|"
    r"singleboerse|singlebörse)",
    re.I,
)
ADULT_GAMBLING_SPAM_RE = re.compile(
    r"(online casino|casino bonus|slot machine|sportwetten|"
    r"porn(?:o|hub)?|xxx|sexkontakte|escort|cams? live|"
    r"nackte|erotik\s+chat|onlyfans|"
    r"bitcoin casino|krypto casino|no deposit bonus|"
    r"free spins|jackpot|roulette|blackjack)",
    re.I,
)
OLDPRINT_RE = re.compile(
    r"(\bthun|\bthats(?:a|ä)ch|\bth(?:e|a)il|\bselb(?:st|ige)n?\b|"
    r"\bverh(?:a|ä)ltni|\bge\s+stellt|\bbe\s+stimm|\bent\s+sch|\bver\s+fahr|"
    r"\bVerzeichni(?:ß|ss)\b|\bUeber\b|"
    r"\bGesetzsammlung\b|\bAmtsblatt\b|\bAbtheilung\b|\bCivilproze(?:ß|ss)\b|"
    r"\bVerwaltungsgericht|[□�]|Å¿)",
    re.I,
)
MOJIBAKE_HINTS = (
    "\ufffd", "\u00c3\u00a4", "\u00c3\u00b6", "\u00c3\u00bc", "\u00c3\u009f",
    "\u00c3\u00a2\u00e2\u201a\u00ac", "\u00c2 ",
)
CHAT_MARKER_RE = re.compile(r"<\|(?:system|user|assistant|end|im_start|im_end)\|>|###\s*(?:Aufgabe|Antwort):", re.I)
TOC_RE = re.compile(
    r"\b(contents?|inhaltsverzeichnis|table of contents|page|seite)\b|"
    r"(\.{3,}|…{2,})\s*\d{1,5}(\s|$)|"
    r"(^|\s)(?:[ivxlcdm]{1,8}|\d{1,4})\s+[\.\-–—]{2,}\s+\d{1,5}(\s|$)",
    re.I,
)
BIBLIO_RE = re.compile(
    r"\b(isbn|doi:|copyright|all rights reserved|published by|"
    r"herausgegeben|verlag|druckerei|bibliothek)\b",
    re.I,
)
OCR_RE = re.compile(
    r"\b(?:thl|bd\.|s\.\s*\d+|a\.\s*a\.\s*o\.|vgl\.|nr\.|fig\.|"
    r"tafel|abb\.|cap\.|ibid|ſ|flg\.|u\.\s*s\.\s*w\.)\b|"
    r"\b[A-Za-zÄÖÜäöüß](?:\s+[A-Za-zÄÖÜäöüß]){4,}\b",
    re.I,
)
LIST_NAME_RE = re.compile(
    r"(?:\*|\(\*\s*\d{3,4}\)|\(\d{3,4}[–-]\d{2,4}\)|\b[A-ZÄÖÜ][a-zäöüß]+,\s+[A-ZÄÖÜ])"
)
MATH_RE = re.compile(r"\\\[|\\\(|\b(problem|solution|theorem|lemma|proof|beweis)\b|[=+\-*/^]{3,}", re.I)
CODE_RE = re.compile(r"\b(def|class|import|return|function|const|let|var|public static|#include)\b|[{};]{3,}")
SPEAKER_LABEL_RE = re.compile(r"(?<![A-Za-zÄÖÜäöüß])([A-ZÄÖÜ][A-ZÄÖÜ]{2,}(?:\s+[A-ZÄÖÜ][A-ZÄÖÜ]{2,}){0,2})\.")

DE_WORDS = {
    "der", "die", "das", "und", "oder", "nicht", "ist", "sind", "ein", "eine",
    "einer", "einen", "mit", "fuer", "fur", "auf", "von", "zu", "im", "den",
    "dem", "des", "dass", "ich", "du", "sie", "wir", "kann", "wird", "werden",
    "wenn", "weil", "aber", "auch", "als", "wie", "was", "warum", "deutsch",
}
EN_WORDS = {
    "the", "and", "or", "not", "is", "are", "a", "an", "with", "for", "to",
    "of", "in", "that", "this", "you", "your", "we", "can", "will", "if",
    "because", "please", "explain", "example", "answer",
}


@dataclass
class Manifest:
    input_file: str
    output_file: str
    language: str
    lines_in: int = 0
    lines_written: int = 0
    bytes_in: int = 0
    bytes_written: int = 0
    dropped: Counter = field(default_factory=Counter)


def mojibake_score(text: str) -> int:
    return sum(text.count(h) for h in MOJIBAKE_HINTS) + text.count("\ufffd") * 3


def repair_mojibake(text: str) -> str:
    if not any(h in text for h in MOJIBAKE_HINTS):
        return text
    candidates = [text]
    for enc in ("latin1", "cp1252"):
        try:
            candidates.append(text.encode(enc).decode("utf-8"))
        except UnicodeError:
            pass
    return min(candidates, key=mojibake_score)


def normalize(text: str) -> str:
    text = repair_mojibake(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def ascii_fold(text: str) -> str:
    return (
        text.lower()
        .replace("\u00e4", "ae")
        .replace("\u00f6", "oe")
        .replace("\u00fc", "ue")
        .replace("\u00df", "ss")
    )


def language_ok(text: str, language: str) -> bool:
    if language == "code":
        return True
    sample = ascii_fold(text[:10_000])
    words = WORD_RE.findall(sample)
    if len(words) < 30:
        return False
    de_hits = sum(1 for w in words if w in DE_WORDS)
    en_hits = sum(1 for w in words if w in EN_WORDS)
    umlauts = sum(text.lower().count(ch) for ch in "\u00e4\u00f6\u00fc\u00df")
    if language == "german":
        return de_hits >= 6 and de_hits >= en_hits * 1.1 or (umlauts >= 2 and de_hits >= en_hits)
    if language == "english":
        return en_hits >= 6 and en_hits >= de_hits * 1.1
    return True


def repetition_score(text: str) -> float:
    words = WORD_RE.findall(ascii_fold(text))
    if len(words) < 50:
        return 0.0
    return 1.0 - (len(set(words)) / len(words))


def repeated_ngram_score(text: str, n: int = 4) -> float:
    words = WORD_RE.findall(ascii_fold(text))
    if len(words) < n * 4:
        return 0.0
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return 1.0 - (len(set(grams)) / len(grams))


def char_density(text: str) -> tuple[float, float]:
    if not text:
        return 0.0, 0.0
    alpha = sum(1 for c in text if c.isalpha()) / len(text)
    symbol = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
    return alpha, symbol


def digit_density(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isdigit()) / len(text)


def line_list_score(text: str) -> int:
    markers = len(re.findall(r"(^|\s)(?:[-*•]|\d{1,3}[.)])\s+", text))
    dotted_refs = len(re.findall(r"\b\d{1,4}\s*(?:\.{2,}|…|—|-)\s*\d{1,5}\b", text))
    separators = text.count(" | ") + text.count("\t")
    return markers + dotted_refs + separators


def web_boilerplate_score(text: str) -> int:
    return len(WEB_BOILERPLATE_RE.findall(text))


def commercial_boilerplate_score(text: str) -> int:
    return len(COMMERCIAL_BOILERPLATE_RE.findall(text))


def adult_gambling_spam_score(text: str) -> int:
    return len(ADULT_GAMBLING_SPAM_RE.findall(text))


def oldprint_score(text: str) -> int:
    sample = text[:20_000]
    hits = len(OLDPRINT_RE.findall(sample))
    hyphen_breaks = len(re.findall(r"\b[A-Za-zÄÖÜäöüß]{2,}-\s+[A-Za-zÄÖÜäöüß]{2,}\b", sample))
    spaced_words = len(re.findall(r"\b[A-Za-zÄÖÜäöüß]{2,}\s+[A-Za-zÄÖÜäöüß]{1,3}\s+[A-Za-zÄÖÜäöüß]{2,}\b", sample))
    return hits + min(hyphen_breaks, 6) + min(spaced_words // 8, 6)


def structure_reject_reason(text: str, args: argparse.Namespace) -> str | None:
    """Reject valid-looking but low-value pretraining documents.

    These checks are intentionally profile-aware. Math/code contain symbols by
    design; prose should be much stricter about lists, OCR, and catalogue pages.
    """
    if not args.v3_structure_filters:
        return None
    if CHAT_MARKER_RE.search(text):
        return "chat_marker"

    profile = args.profile
    toc_hits = len(TOC_RE.findall(text))
    biblio_hits = len(BIBLIO_RE.findall(text))
    ocr_hits = len(OCR_RE.findall(text))
    list_score = line_list_score(text)
    name_hits = len(LIST_NAME_RE.findall(text[:20_000]))
    year_hits = len(re.findall(r"\(\*?\s?\d{3,4}[–-]\d{2,4}\)|\*\s?\d{3,4}", text[:20_000]))
    speaker_hits = len(SPEAKER_LABEL_RE.findall(text[:20_000]))
    unique_speakers = len(set(SPEAKER_LABEL_RE.findall(text[:20_000])))
    repeated_ngrams = repeated_ngram_score(text)
    digits = digit_density(text)

    if profile == "booster":
        # Booster examples are intentionally short and number-heavy. Only keep
        # them out of the base if they contain chat/HTML/boilerplate, handled
        # elsewhere.
        return None
    if profile == "math":
        # Preserve math structure, but drop obvious navigation/metadata/list junk.
        if toc_hits >= 2 and not MATH_RE.search(text):
            return "toc_or_index"
        if biblio_hits >= 4:
            return "bibliography_or_metadata"
        if repeated_ngrams > args.max_ngram_repetition:
            return "repetitive_ngram"
        return None
    if profile == "code":
        if toc_hits >= 2:
            return "toc_or_index"
        if repeated_ngrams > args.max_ngram_repetition:
            return "repetitive_ngram"
        return None

    if toc_hits >= 2 or (toc_hits >= 1 and list_score >= 4):
        return "toc_or_index"
    if args.drop_web_boilerplate and web_boilerplate_score(text) >= args.max_web_boilerplate_hits:
        return "web_boilerplate"
    if args.drop_commercial_boilerplate and commercial_boilerplate_score(text) >= args.max_commercial_boilerplate_hits:
        return "commercial_boilerplate"
    if args.drop_adult_gambling_spam and adult_gambling_spam_score(text) >= args.max_adult_gambling_hits:
        return "adult_or_gambling_spam"
    if args.drop_old_ocr and oldprint_score(text) >= args.max_old_ocr_hits:
        return "old_ocr_or_fraktur"
    if biblio_hits >= args.max_bibliography_hits:
        return "bibliography_or_metadata"
    if ocr_hits >= args.max_ocr_hits:
        return "ocr_scan_hint"
    if speaker_hits >= args.max_speaker_labels and unique_speakers <= 12:
        return "dialogue_script"
    if list_score >= args.max_list_score:
        return "list_or_table_heavy"
    if name_hits >= args.max_name_list_hits or (year_hits >= 10 and text.count(",") >= 10):
        return "name_catalogue"
    if digits > args.max_digit_ratio:
        return "digit_dense"
    if repeated_ngrams > args.max_ngram_repetition:
        return "repetitive_ngram"
    return None


def reject_reason(text: str, args: argparse.Namespace) -> str | None:
    if len(text) < args.min_chars:
        return "too_short"
    if len(text) > args.max_chars:
        return "too_long"
    early_structure_reason = structure_reject_reason(text, args)
    if early_structure_reason in {"chat_marker", "toc_or_index", "name_catalogue"}:
        return early_structure_reason
    if mojibake_score(text) > 0:
        return "mojibake"
    if HTML_RE.search(text):
        return "html"
    if BAD_RE.search(text):
        return "boilerplate"
    urls = len(URL_RE.findall(text))
    if urls > args.max_urls:
        return "url_dense"
    alpha, symbol = char_density(text)
    if args.language != "code" and alpha < args.min_alpha:
        return "low_alpha"
    if symbol > args.max_symbol:
        return "symbol_dense"
    if repetition_score(text) > args.max_repetition:
        return "repetitive"
    if early_structure_reason is not None:
        return early_structure_reason
    if args.profile != "booster" and not language_ok(text, args.language):
        return "language_mismatch"
    return None


def stable_hash(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=16).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=["german", "english", "code"], required=True)
    parser.add_argument(
        "--profile",
        choices=["prose", "math", "code", "booster"],
        default=None,
        help="clean-v3 structural profile. Defaults to language-specific profile.",
    )
    parser.add_argument("--min-chars", type=int, default=None)
    parser.add_argument("--max-chars", type=int, default=80_000)
    parser.add_argument("--max-urls", type=int, default=1)
    parser.add_argument("--min-alpha", type=float, default=0.45)
    parser.add_argument("--max-symbol", type=float, default=0.28)
    parser.add_argument("--max-repetition", type=float, default=0.62)
    parser.add_argument("--v3-structure-filters", action="store_true")
    parser.add_argument("--max-list-score", type=int, default=8)
    parser.add_argument("--max-name-list-hits", type=int, default=30)
    parser.add_argument("--max-ocr-hits", type=int, default=2)
    parser.add_argument("--max-bibliography-hits", type=int, default=3)
    parser.add_argument("--max-speaker-labels", type=int, default=18)
    parser.add_argument("--max-digit-ratio", type=float, default=0.16)
    parser.add_argument("--max-ngram-repetition", type=float, default=0.28)
    parser.add_argument("--drop-web-boilerplate", action="store_true")
    parser.add_argument("--max-web-boilerplate-hits", type=int, default=1)
    parser.add_argument("--drop-commercial-boilerplate", action="store_true")
    parser.add_argument("--max-commercial-boilerplate-hits", type=int, default=1)
    parser.add_argument("--drop-adult-gambling-spam", action="store_true")
    parser.add_argument("--max-adult-gambling-hits", type=int, default=1)
    parser.add_argument("--drop-old-ocr", action="store_true")
    parser.add_argument("--max-old-ocr-hits", type=int, default=8)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument(
        "--dedup-cache",
        type=int,
        default=1_000_000,
        help="Maximum exact-document hashes kept in RAM. 0 disables dedup.",
    )
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    if args.profile is None:
        args.profile = "code" if args.language == "code" else "prose"
    if args.min_chars is None:
        if args.profile == "booster":
            args.min_chars = 40
        elif args.profile == "math":
            args.min_chars = 80
        else:
            args.min_chars = {"german": 240, "english": 180, "code": 8}[args.language]

    rng = random.Random(args.seed)
    manifest = Manifest(
        input_file=str(args.input),
        output_file=str(args.output),
        language=args.language,
        bytes_in=args.input.stat().st_size,
    )
    seen: set[str] = set()
    seen_fifo: deque[str] = deque()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", encoding="utf-8", errors="replace") as src, args.output.open(
        "w", encoding="utf-8", newline="\n"
    ) as out:
        for line in src:
            manifest.lines_in += 1
            if args.sample_rate < 1.0 and rng.random() > args.sample_rate:
                manifest.dropped["sample_skip"] += 1
                continue
            text = normalize(line)
            reason = reject_reason(text, args)
            if reason is not None:
                manifest.dropped[reason] += 1
                continue
            if args.dedup_cache:
                key = stable_hash(text)
                if key in seen:
                    manifest.dropped["duplicate"] += 1
                    continue
                seen.add(key)
                seen_fifo.append(key)
                if len(seen_fifo) > args.dedup_cache:
                    seen.discard(seen_fifo.popleft())
            out.write(text.replace("\n", " ") + "\n")
            manifest.lines_written += 1
            manifest.bytes_written += len((text + "\n").encode("utf-8"))
            if args.max_docs and manifest.lines_written >= args.max_docs:
                break

    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_payload = {
        "input_file": manifest.input_file,
        "output_file": manifest.output_file,
        "language": manifest.language,
        "lines_in": manifest.lines_in,
        "lines_written": manifest.lines_written,
        "bytes_in": manifest.bytes_in,
        "bytes_written": manifest.bytes_written,
        "dropped": dict(manifest.dropped.most_common()),
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output} ({manifest.lines_written:,} docs)")
    print(f"wrote {manifest_path}")
    print(f"dropped: {dict(manifest.dropped.most_common(12))}")


if __name__ == "__main__":
    main()
