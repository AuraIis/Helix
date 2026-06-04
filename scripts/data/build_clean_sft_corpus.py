#!/usr/bin/env python3
"""Build a strict, German-first SFT corpus from local rescued/raw sources.

This is intentionally conservative. It prefers losing volume over teaching the
model broken formatting, English answers behind a German system prompt, HTML,
generic "please provide the input" answers, truncated responses, or duplicates.

Output records are JSONL with a single ``text`` field using Helix tokens:

    <|system|> ... <|end|>
    <|user|> ... <|end|>
    <|assistant|> ... <|end|>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parents[2]
SYSTEM_DE = "Du bist Auralis, ein hilfreicher KI-Assistent. Antworte auf Deutsch."

DEFAULT_SOURCES = (
    "data/training/sft_rescued/balanced/de_strict/train.helix.jsonl",
    "data/training/sft_rescued/balanced/de_strict/val.helix.jsonl",
    "raw/sft/synth/helix_curated_de_25000.jsonl",
    "raw/sft/synth/helix_curated_de_history_25000.jsonl",
    "raw/sft/synth/evol_de_gpt55_premium.jsonl",
    "raw/sft/synth/evol_de_gemini31pro_premium.jsonl",
    "raw/sft/synth/evol_de_clarify_aware.jsonl",
    "raw/sft/synth/evol_instruct_de_v1.jsonl",
    "raw/sft/synth/multiturn_de_v1.jsonl",
    "raw/sft/synth/outputs/phase3_sft_final.jsonl",
)

ROLE_RE = re.compile(r"<\|(system|user|assistant)\|>\n(.*?)\n<\|end\|>", re.DOTALL)
WORD_RE = re.compile(r"[A-Za-z\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df]+")
HTML_RE = re.compile(r"<\s*/?\s*(?:html|body|div|script|style|a|span|table|iframe|form)\b", re.I)
URL_RE = re.compile(r"https?://|www\.", re.I)
COMPLETE_RE = re.compile(r'(\.|\!|\?|\u2026|```|\)|\]|\}|\"|\'|\u00bb|\u201d)$')

DE_WORDS = {
    "der", "die", "das", "und", "oder", "nicht", "ist", "sind", "ein", "eine",
    "einer", "einen", "mit", "fuer", "fur", "auf", "von", "zu", "im", "den",
    "dem", "des", "dass", "ich", "du", "sie", "wir", "kann", "koennen",
    "konnen", "wird", "werden", "wenn", "weil", "aber", "auch", "als", "wie",
    "was", "warum", "bitte", "erklaere", "erklare", "beispiel", "loesung",
    "losung", "antwort", "deutsch", "nein", "ja", "dies", "diese", "dieser",
}
EN_WORDS = {
    "the", "and", "or", "not", "is", "are", "a", "an", "with", "for", "to",
    "of", "in", "that", "this", "you", "your", "we", "can", "will", "if",
    "because", "please", "explain", "example", "solution", "answer",
}
MOJIBAKE_HINTS = (
    "\ufffd", "\u00c3\u00a4", "\u00c3\u00b6", "\u00c3\u00bc", "\u00c3\u009f",
    "\u00c3\u00a2\u00e2\u201a\u00ac", "\u00c2 ",
)
GENERIC_NO_INPUT_RE = re.compile(
    r"(bitte\s+(poste|gib|sende|teile).{0,80}(text|kommentar|code|datei|frage)|"
    r"sobald\s+du\s+mir\s+.{0,80}(gibst|sendest|postest)|"
    r"ich\s+brauche\s+.{0,80}(text|input|eingabe))",
    re.I | re.S,
)
BAD_PHRASE_RE = re.compile(
    r"(as an ai language model|i cannot browse|knowledge cutoff|<\|im_start\|>|"
    r"<\|im_end\|>|_end_of_the_data|user-data|cookie policy|privacy policy|"
    r"accept all cookies|javascript is disabled)",
    re.I,
)


@dataclass
class Stats:
    records_in: int = 0
    records_kept: int = 0
    dropped: Counter = field(default_factory=Counter)
    kept_by_source: Counter = field(default_factory=Counter)
    kept_by_category: Counter = field(default_factory=Counter)


def _ascii_fold(text: str) -> str:
    return (
        text.lower()
        .replace("\u00e4", "ae")
        .replace("\u00f6", "oe")
        .replace("\u00fc", "ue")
        .replace("\u00df", "ss")
    )


def _mojibake_score(text: str) -> int:
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
    return min(candidates, key=_mojibake_score)


def clean_text(text: object) -> str:
    out = repair_mojibake(str(text))
    out = out.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    lines = [line.rstrip() for line in out.splitlines()]
    out = "\n".join(lines).strip()
    return re.sub(r"\n{4,}", "\n\n\n", out)


def looks_german(text: str) -> bool:
    sample = _ascii_fold(text[:12_000])
    words = WORD_RE.findall(sample)
    if not words:
        return False
    de_hits = sum(1 for w in words if w in DE_WORDS)
    en_hits = sum(1 for w in words if w in EN_WORDS)
    umlauts = sum(text.lower().count(ch) for ch in "\u00e4\u00f6\u00fc\u00df")
    return (de_hits >= 4 and de_hits >= en_hits * 1.25) or (umlauts >= 2 and de_hits >= en_hits)


def repetition_score(text: str) -> float:
    words = WORD_RE.findall(_ascii_fold(text))
    if len(words) < 24:
        return 0.0
    return 1.0 - (len(set(words)) / len(words))


def max_line_repeat(text: str) -> int:
    counts = Counter(line.strip() for line in text.splitlines() if line.strip())
    return max(counts.values(), default=0)


def role_messages_from_helix(text: str) -> list[dict[str, str]] | None:
    turns = [{"role": m.group(1), "content": clean_text(m.group(2))} for m in ROLE_RE.finditer(text)]
    return turns or None


def normalize_messages(raw: object) -> list[dict[str, str]] | None:
    if not isinstance(raw, list):
        return None
    messages: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        role = item.get("role")
        content = clean_text(item.get("content", ""))
        if role not in {"system", "user", "assistant"} or not content:
            return None
        messages.append({"role": str(role), "content": content})
    return messages


def messages_from_record(record: dict, source_path: Path) -> list[dict[str, str]] | None:
    if isinstance(record.get("text"), str):
        parsed = role_messages_from_helix(record["text"])
        if parsed:
            return parsed

    if "messages" in record:
        return normalize_messages(record["messages"])

    if "evolved_prompt" in record and "response" in record:
        return [
            {"role": "user", "content": clean_text(record["evolved_prompt"])},
            {"role": "assistant", "content": clean_text(record["response"])},
        ]

    if "instruction" in record and "output" in record:
        user = clean_text(record["instruction"])
        inp = clean_text(record.get("input", ""))
        if inp:
            user = f"{user}\n\nInput:\n{inp}"
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": clean_text(record["output"])},
        ]

    return None


def role_order_ok(messages: list[dict[str, str]]) -> bool:
    expected = "user"
    seen_user = False
    for i, msg in enumerate(messages):
        role = msg["role"]
        if role == "system":
            if i != 0:
                return False
            continue
        if role != expected:
            return False
        seen_user = seen_user or role == "user"
        expected = "assistant" if expected == "user" else "user"
    return seen_user and messages[-1]["role"] == "assistant"


def assistant_complete(text: str) -> bool:
    stripped = text.rstrip()
    if not stripped or stripped.count("```") % 2:
        return False
    last_line = next((line.strip() for line in reversed(stripped.splitlines()) if line.strip()), "")
    if re.fullmatch(r"(?:\d+[\.)]?|[-*+]|\#{1,6})", last_line):
        return False
    if last_line.endswith(("- ", ":", ";")):
        return False
    if len(stripped) >= 80 and not COMPLETE_RE.search(stripped):
        return False
    return True


def reject_reason(messages: list[dict[str, str]], record: dict) -> str | None:
    if not role_order_ok(messages):
        return "bad_role_order"

    non_system = "\n".join(m["content"] for m in messages if m["role"] != "system")
    assistants = [m["content"] for m in messages if m["role"] == "assistant"]
    users = [m["content"] for m in messages if m["role"] == "user"]
    assistant_text = "\n".join(assistants)

    if len(non_system) < 48 or min(len(u) for u in users) < 8 or min(len(a) for a in assistants) < 20:
        return "too_short"
    if len(non_system) > 24_000 or max(len(a) for a in assistants) > 12_000:
        return "too_long"
    if _mojibake_score(non_system) >= 3:
        return "mojibake"
    if HTML_RE.search(non_system):
        return "html"
    if BAD_PHRASE_RE.search(non_system):
        return "bad_phrase"
    if URL_RE.search(non_system) and len(URL_RE.findall(non_system)) > 2:
        return "url_dense"
    if GENERIC_NO_INPUT_RE.search(assistant_text):
        return "generic_missing_input"
    if repetition_score(assistant_text) > 0.72 or max_line_repeat(assistant_text) >= 4:
        return "repetitive"
    if any(not assistant_complete(a) for a in assistants):
        return "truncated_assistant"
    if not looks_german(assistant_text):
        return "assistant_not_german"

    quality_score = record.get("quality_score")
    if quality_score is not None:
        try:
            if float(quality_score) < 4:
                return "low_quality_score"
        except (TypeError, ValueError):
            return "bad_quality_score"

    filter_score = record.get("filter_score")
    if filter_score is not None:
        try:
            if float(filter_score) < 4:
                return "low_filter_score"
        except (TypeError, ValueError):
            return "bad_filter_score"

    return None


def render_helix(messages: list[dict[str, str]]) -> str:
    chunks = [f"<|system|>\n{SYSTEM_DE}\n<|end|>\n"]
    for msg in messages:
        if msg["role"] == "system":
            continue
        chunks.append(f"<|{msg['role']}|>\n{msg['content']}\n<|end|>\n")
    return "".join(chunks)


def dedup_key(messages: list[dict[str, str]]) -> str:
    payload = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        norm = re.sub(r"\s+", " ", msg["content"].strip().lower())
        payload.append(f"{msg['role']}:{norm}")
    return hashlib.blake2b("\n".join(payload).encode("utf-8"), digest_size=16).hexdigest()


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError:
                yield lineno, {"_json_error": True}


def assistant_key(messages: list[dict[str, str]]) -> str:
    text = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=16).hexdigest()


def collect_rows(
    sources: list[Path],
    reject_sample_limit: int,
    max_same_assistant: int,
) -> tuple[list[dict], list[dict], Stats]:
    stats = Stats()
    rows: list[dict] = []
    rejects: list[dict] = []
    seen: set[str] = set()
    assistant_seen: Counter = Counter()

    for source in sources:
        if not source.exists():
            stats.dropped[f"missing:{source}"] += 1
            continue
        for lineno, record in iter_jsonl(source):
            stats.records_in += 1
            source_name = source.as_posix()
            if record.get("_json_error"):
                reason = "json_error"
                stats.dropped[reason] += 1
                if len(rejects) < reject_sample_limit:
                    rejects.append({"source": source_name, "line": lineno, "reason": reason})
                continue

            messages = messages_from_record(record, source)
            if messages is None:
                reason = "unknown_schema"
            else:
                reason = reject_reason(messages, record)

            if reason is not None:
                stats.dropped[reason] += 1
                if len(rejects) < reject_sample_limit:
                    preview = ""
                    if messages:
                        preview = "\n".join(m["content"] for m in messages if m["role"] != "system")[:600]
                    rejects.append({"source": source_name, "line": lineno, "reason": reason, "preview": preview})
                continue

            assert messages is not None
            key = dedup_key(messages)
            if key in seen:
                stats.dropped["duplicate"] += 1
                continue
            seen.add(key)

            akey = assistant_key(messages)
            if assistant_seen[akey] >= max_same_assistant:
                stats.dropped["assistant_duplicate_cap"] += 1
                continue
            assistant_seen[akey] += 1

            category = str(record.get("category") or record.get("task_type") or source.parent.name)
            row = {
                "text": render_helix(messages),
                "source": source_name,
                "category": category,
                "quality_variant": "clean_de_v1",
            }
            rows.append(row)
            stats.records_kept += 1
            stats.kept_by_source[source_name] += 1
            stats.kept_by_category[category] += 1

    return rows, rejects, stats


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=REPO / "data" / "training" / "sft_clean_de_v1")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--reject-sample-limit", type=int, default=500)
    parser.add_argument(
        "--max-same-assistant",
        type=int,
        default=3,
        help="Keep at most this many records with exactly the same assistant text.",
    )
    parser.add_argument("--source", action="append", default=None, help="Source JSONL path. Repeatable.")
    args = parser.parse_args()

    source_args = args.source if args.source else list(DEFAULT_SOURCES)
    sources = [Path(p) if Path(p).is_absolute() else REPO / p for p in source_args]

    rows, rejects, stats = collect_rows(sources, args.reject_sample_limit, args.max_same_assistant)
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val_count = max(1, int(len(rows) * args.val_ratio)) if rows else 0
    val_rows = rows[:val_count]
    train_rows = rows[val_count:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_n = write_jsonl(args.output_dir / "train.helix.jsonl", train_rows)
    val_n = write_jsonl(args.output_dir / "val.helix.jsonl", val_rows)
    write_jsonl(args.output_dir / "reject_samples.jsonl", rejects)

    manifest = {
        "variant": "clean_de_v1",
        "sources": [str(p) for p in sources],
        "max_same_assistant": args.max_same_assistant,
        "train_records": train_n,
        "val_records": val_n,
        "stats": {
            "records_in": stats.records_in,
            "records_kept": stats.records_kept,
            "dropped": dict(stats.dropped.most_common()),
            "kept_by_source": dict(stats.kept_by_source.most_common()),
            "kept_by_category": dict(stats.kept_by_category.most_common()),
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"wrote {args.output_dir / 'train.helix.jsonl'} ({train_n:,})")
    print(f"wrote {args.output_dir / 'val.helix.jsonl'} ({val_n:,})")
    print(f"kept {stats.records_kept:,} / {stats.records_in:,}")
    print(f"dropped: {dict(stats.dropped.most_common(12))}")


if __name__ == "__main__":
    main()
