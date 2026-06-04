#!/usr/bin/env python3
"""Rescue legacy ChatML SFT JSONL into canonical Helix prompt data.

The first Phase-3 SFT exports used Qwen-style markers:

    <|im_start|>user ... <|im_end|>

Helix v2's tokenizer/template uses different special tokens:

    <|user|> ... <|end|>

This script salvages those records without rewriting the actual answers. It
parses the ChatML-ish text, repairs common mojibake, deduplicates, and writes
two clean variants:

- ``all_neutral`` keeps valid examples but replaces the contradictory German
  system prompt with a neutral one.
- ``de_strict`` keeps only records whose user/assistant content looks German
  enough, and uses a German-answer system prompt.

Input format:
    {"text": "...chatml...", "source": "...", "category": "..."}

Output format:
    {"text": "...helix prompt...", "source": "...", "category": "..."}

Usage:
    python scripts/data/rescue_sft_data.py \
        --input-dir sft_train_balanced \
        --output-dir data/training/sft_rescued/balanced
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


CANON_SYSTEM_NEUTRAL = "Du bist Auralis, ein hilfreicher KI-Assistent."
CANON_SYSTEM_DE = "Du bist Auralis, ein hilfreicher KI-Assistent. Antworte auf Deutsch."

ROLE_ORDER = {"system", "user", "assistant"}
CHATML_RE = re.compile(
    r"<\|im_start\|>(system|user|assistant)\n(.*?)<\|im_end\|>",
    re.DOTALL,
)

GERMAN_WORDS = {
    "der", "die", "das", "und", "oder", "nicht", "ist", "sind", "ein", "eine",
    "einer", "einen", "mit", "für", "auf", "von", "zu", "im", "im", "den",
    "dem", "des", "dass", "ich", "du", "sie", "wir", "kann", "können",
    "wird", "werden", "wenn", "weil", "aber", "auch", "als", "wie", "was",
    "warum", "bitte", "erkläre", "beispiel", "lösung", "antwort",
}
ENGLISH_WORDS = {
    "the", "and", "or", "not", "is", "are", "a", "an", "with", "for", "to",
    "of", "in", "that", "this", "you", "your", "we", "can", "will", "if",
    "because", "please", "explain", "example", "solution", "answer",
}
MOJIBAKE_HINTS = ("Ã", "Â", "â€", "â€“", "â€”", "â€ž", "â€œ", "â€™", "ðŸ")
HTML_HINT_RE = re.compile(r"<\s*/?\s*(?:html|body|div|script|style|a|span|table)\b", re.I)
COMPLETE_END_RE = re.compile(r"(\.|\!|\?|…|```|\)|\]|\}|\"|'|»|”)$")


@dataclass
class Stats:
    input_records: int = 0
    parsed_records: int = 0
    all_neutral_records: int = 0
    de_strict_records: int = 0
    dropped: Counter = field(default_factory=Counter)
    source_counts_all: Counter = field(default_factory=Counter)
    category_counts_all: Counter = field(default_factory=Counter)
    source_counts_de: Counter = field(default_factory=Counter)
    category_counts_de: Counter = field(default_factory=Counter)


def _mojibake_score(text: str) -> int:
    return sum(text.count(h) for h in MOJIBAKE_HINTS) + text.count("\ufffd") * 3


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8-as-Latin-1 artifacts if the result looks better."""
    if not any(h in text for h in MOJIBAKE_HINTS) and "\ufffd" not in text:
        return text
    candidates = [text]
    for enc in ("latin1", "cp1252"):
        try:
            candidates.append(text.encode(enc).decode("utf-8"))
        except UnicodeError:
            pass
    return min(candidates, key=_mojibake_score)


def clean_content(text: str) -> str:
    text = repair_mojibake(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x00", "")
    # Preserve paragraphs and code indentation, but trim line-edge drift.
    lines = [ln.rstrip() for ln in text.split("\n")]
    text = "\n".join(lines).strip()
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text


def parse_chatml(text: str) -> list[dict[str, str]] | None:
    text = text.replace("<|EOT|>", "")
    turns = [{"role": m.group(1), "content": clean_content(m.group(2))} for m in CHATML_RE.finditer(text)]
    if not turns:
        return None
    if any(t["role"] not in ROLE_ORDER or not t["content"] for t in turns):
        return None
    if not any(t["role"] == "user" for t in turns) or turns[-1]["role"] != "assistant":
        return None
    expected = "user"
    for turn in turns:
        if turn["role"] == "system":
            continue
        if turn["role"] != expected:
            return None
        expected = "assistant" if expected == "user" else "user"
    return turns


def looks_german(text: str) -> bool:
    sample = text[:8000].lower()
    words = re.findall(r"[a-zA-ZäöüÄÖÜß]+", sample)
    if not words:
        return False
    de_hits = sum(1 for w in words if w in GERMAN_WORDS)
    en_hits = sum(1 for w in words if w in ENGLISH_WORDS)
    umlauts = sum(sample.count(ch) for ch in "äöüß")
    # German generated with mojibake repair may still have no umlauts; do not
    # require them. The stopword gap avoids keeping English examples whose
    # only German text is the old system prompt.
    return de_hits >= 4 and de_hits >= en_hits * 1.35 or (umlauts >= 2 and de_hits >= en_hits)


def is_probably_garbage(messages: list[dict[str, str]]) -> bool:
    joined = "\n".join(m["content"] for m in messages if m["role"] != "system")
    if len(joined) < 24:
        return True
    if _mojibake_score(joined) >= 4:
        return True
    if HTML_HINT_RE.search(joined):
        return True
    if joined.count("<|im_start|>") or joined.count("<|im_end|>"):
        return True
    return False


def assistant_turns_look_complete(messages: list[dict[str, str]]) -> bool:
    """Drop likely max-token truncations.

    Several teacher-generated German records end mid-word/mid-sentence. Those
    examples teach the model to trail off, so they are worse than losing a bit
    of volume. Code answers may end in ``}`` or a closing code fence; prose
    should end with normal sentence punctuation.
    """
    for msg in messages:
        if msg["role"] != "assistant":
            continue
        text = msg["content"].rstrip()
        if not text:
            return False
        if text.count("```") % 2:
            return False
        last_line = next((ln.strip() for ln in reversed(text.splitlines()) if ln.strip()), "")
        if re.fullmatch(r"(?:\d+[\.)]?|[-*+]|\#{1,6})", last_line):
            return False
        if last_line.endswith(("- ", ":", ";")):
            return False
        if len(text) >= 80 and not COMPLETE_END_RE.search(text):
            return False
    return True


def render_helix(messages: list[dict[str, str]], *, system_prompt: str) -> str:
    rendered: list[str] = [f"<|system|>\n{system_prompt}\n<|end|>\n"]
    for m in messages:
        if m["role"] == "system":
            continue
        rendered.append(f"<|{m['role']}|>\n{m['content']}\n<|end|>\n")
    return "".join(rendered)


def dedup_key(messages: list[dict[str, str]]) -> str:
    payload = []
    for m in messages:
        if m["role"] == "system":
            continue
        norm = re.sub(r"\s+", " ", m["content"].strip().lower())
        payload.append(f"{m['role']}:{norm}")
    return hashlib.sha1("\n".join(payload).encode("utf-8")).hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def process_split(input_path: Path, all_path: Path, de_path: Path) -> Stats:
    stats = Stats()
    seen_all: set[str] = set()
    seen_de: set[str] = set()
    all_rows: list[dict] = []
    de_rows: list[dict] = []

    for rec in iter_jsonl(input_path):
        stats.input_records += 1
        messages = parse_chatml(str(rec.get("text", "")))
        if messages is None:
            stats.dropped["parse_failed"] += 1
            continue
        stats.parsed_records += 1

        if is_probably_garbage(messages):
            stats.dropped["garbage_or_html"] += 1
            continue
        if not assistant_turns_look_complete(messages):
            stats.dropped["truncated_assistant"] += 1
            continue

        key = dedup_key(messages)
        if key in seen_all:
            stats.dropped["duplicate_all"] += 1
        else:
            seen_all.add(key)
            out = {
                "text": render_helix(messages, system_prompt=CANON_SYSTEM_NEUTRAL),
                "source": rec.get("source", ""),
                "category": rec.get("category", ""),
                "rescue_variant": "all_neutral",
            }
            all_rows.append(out)
            stats.source_counts_all[out["source"]] += 1
            stats.category_counts_all[out["category"]] += 1

        non_system = "\n".join(m["content"] for m in messages if m["role"] != "system")
        assistant_text = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
        if looks_german(non_system) and looks_german(assistant_text):
            if key in seen_de:
                stats.dropped["duplicate_de"] += 1
            else:
                seen_de.add(key)
                out = {
                    "text": render_helix(messages, system_prompt=CANON_SYSTEM_DE),
                    "source": rec.get("source", ""),
                    "category": rec.get("category", ""),
                    "rescue_variant": "de_strict",
                }
                de_rows.append(out)
                stats.source_counts_de[out["source"]] += 1
                stats.category_counts_de[out["category"]] += 1
        else:
            stats.dropped["not_de_strict"] += 1

    stats.all_neutral_records = write_jsonl(all_path, all_rows)
    stats.de_strict_records = write_jsonl(de_path, de_rows)
    return stats


def stats_to_json(stats: Stats) -> dict:
    return {
        "input_records": stats.input_records,
        "parsed_records": stats.parsed_records,
        "all_neutral_records": stats.all_neutral_records,
        "de_strict_records": stats.de_strict_records,
        "dropped": dict(stats.dropped.most_common()),
        "source_counts_all": dict(stats.source_counts_all.most_common()),
        "category_counts_all": dict(stats.category_counts_all.most_common()),
        "source_counts_de": dict(stats.source_counts_de.most_common()),
        "category_counts_de": dict(stats.category_counts_de.most_common()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", type=Path, default=Path("sft_train_balanced"))
    ap.add_argument("--output-dir", type=Path, default=Path("data/training/sft_rescued/balanced"))
    ap.add_argument("--train-name", default="train.chatml.jsonl")
    ap.add_argument("--val-name", default="val.chatml.jsonl")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_stats: dict[str, dict] = {}
    for split, name in (("train", args.train_name), ("val", args.val_name)):
        input_path = args.input_dir / name
        if not input_path.exists():
            raise SystemExit(f"missing input file: {input_path}")
        print(f"processing {split}: {input_path}", flush=True)
        stats = process_split(
            input_path,
            args.output_dir / "all_neutral" / f"{split}.helix.jsonl",
            args.output_dir / "de_strict" / f"{split}.helix.jsonl",
        )
        final_stats[split] = stats_to_json(stats)
        print(
            f"  parsed={stats.parsed_records:,} "
            f"all={stats.all_neutral_records:,} de={stats.de_strict_records:,}",
            flush=True,
        )

    stats_path = args.output_dir / "rescue_stats.json"
    stats_path.write_text(json.dumps(final_stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"stats: {stats_path}", flush=True)


if __name__ == "__main__":
    main()
