#!/usr/bin/env python3
"""Download larger QA/instruct sources with light quality filters.

These sources are meant as scale boosters for QA/reasoning style data. They are
kept separate from Reddit so we can audit and mix them conservatively.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from datasets import load_dataset


WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")
BAD_RE = re.compile(
    r"\[(?:deleted|removed)\]|onlyfans|discord\.gg|telegram|whatsapp|free karma|"
    r"upvote if|subscribe to|promo code|coupon code|buy now|nsfw|porn|sex chat|"
    r"i am a bot|beep boop|performed automatically",
    re.I,
)
URL_RE = re.compile(r"https?://|www\.", re.I)


SOURCES = {
    "openorca": {
        "dataset": "Open-Orca/OpenOrca",
        "split": "train",
        "kind": "openorca",
    },
    "orca_agent_open_domain": {
        "dataset": "microsoft/orca-agentinstruct-1M-v1",
        "split": "open_domain_qa",
        "kind": "messages",
    },
    "orca_agent_analytical": {
        "dataset": "microsoft/orca-agentinstruct-1M-v1",
        "split": "analytical_reasoning",
        "kind": "messages",
    },
}


def clean(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def text_ok(text: str, *, min_chars: int, max_chars: int, min_words: int, allow_urls: bool) -> str | None:
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    if BAD_RE.search(text):
        return "bad_marker"
    if not allow_urls and URL_RE.search(text):
        return "has_url"
    words = WORD_RE.findall(text)
    if len(words) < min_words:
        return "too_few_words"
    alpha = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha < 0.35:
        return "low_alpha_density"
    return None


def extract_messages(raw: Any) -> list[dict[str, str]]:
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, list):
        return []
    return [m for m in raw if isinstance(m, dict)]


def records_from_row(row: dict[str, Any], source_name: str, kind: str) -> Iterable[dict[str, str]]:
    if kind == "openorca":
        yield {
            "source": source_name,
            "system": clean(row.get("system_prompt", "")),
            "question": clean(row.get("question", "")),
            "answer": clean(row.get("response", "")),
        }
        return
    if kind == "messages":
        messages = extract_messages(row.get("messages"))
        user = ""
        assistant = ""
        system = ""
        for msg in messages:
            role = msg.get("role")
            content = clean(msg.get("content", ""))
            if role == "system" and content and not system:
                system = content
            elif role == "user" and content and not user:
                user = content
            elif role == "assistant" and content and user:
                assistant = content
                break
        if user and assistant:
            yield {
                "source": source_name,
                "system": system,
                "question": user,
                "answer": assistant,
            }


def write_one(name: str, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    spec = SOURCES[name]
    stats: Counter[str] = Counter()
    seen = 0
    written = 0
    out_jsonl = out_dir / f"{name}.jsonl"
    out_txt = out_dir / f"{name}.txt"
    ds = load_dataset(spec["dataset"], split=spec["split"], streaming=True)
    with out_jsonl.open("w", encoding="utf-8", newline="\n") as jsonl, out_txt.open(
        "w", encoding="utf-8", newline="\n"
    ) as txt:
        for row in ds:
            seen += 1
            emitted = False
            for rec in records_from_row(row, name, spec["kind"]):
                q_reason = text_ok(
                    rec["question"],
                    min_chars=args.min_question_chars,
                    max_chars=args.max_question_chars,
                    min_words=args.min_question_words,
                    allow_urls=args.allow_urls,
                )
                if q_reason:
                    stats[f"question_{q_reason}"] += 1
                    continue
                a_reason = text_ok(
                    rec["answer"],
                    min_chars=args.min_answer_chars,
                    max_chars=args.max_answer_chars,
                    min_words=args.min_answer_words,
                    allow_urls=args.allow_urls,
                )
                if a_reason:
                    stats[f"answer_{a_reason}"] += 1
                    continue
                jsonl.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                system = f"System: {rec['system']}\n" if rec.get("system") else ""
                txt.write(f"{system}Frage: {rec['question']}\nAntwort: {rec['answer']}\n\n")
                written += 1
                emitted = True
            if not emitted:
                stats["not_written"] += 1
            if seen >= args.max_scan_per_source or written >= args.max_records_per_source:
                break
            if seen % args.log_every == 0:
                print(f"{name}: seen={seen:,} written={written:,}", flush=True)
    return {
        "source": name,
        "hf_dataset": spec["dataset"],
        "split": spec["split"],
        "seen": seen,
        "written": written,
        "bytes_text": out_txt.stat().st_size,
        "stats": dict(stats),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", action="append", choices=sorted(SOURCES), default=None)
    parser.add_argument("--max-records-per-source", type=int, default=300_000)
    parser.add_argument("--max-scan-per-source", type=int, default=1_500_000)
    parser.add_argument("--min-question-chars", type=int, default=25)
    parser.add_argument("--max-question-chars", type=int, default=5_000)
    parser.add_argument("--min-question-words", type=int, default=5)
    parser.add_argument("--min-answer-chars", type=int, default=40)
    parser.add_argument("--max-answer-chars", type=int, default=10_000)
    parser.add_argument("--min-answer-words", type=int, default=8)
    parser.add_argument("--allow-urls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=50_000)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = args.source or ["openorca", "orca_agent_open_domain", "orca_agent_analytical"]
    manifest = {
        "sources": [],
        "filters": {
            "max_records_per_source": args.max_records_per_source,
            "max_scan_per_source": args.max_scan_per_source,
            "min_question_chars": args.min_question_chars,
            "max_question_chars": args.max_question_chars,
            "min_answer_chars": args.min_answer_chars,
            "max_answer_chars": args.max_answer_chars,
            "allow_urls": args.allow_urls,
        },
    }
    for name in selected:
        print(f"=== {name} ===", flush=True)
        manifest["sources"].append(write_one(name, args, args.output_dir))
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    manifest["documents"] = sum(s["written"] for s in manifest["sources"])
    manifest["bytes_text"] = sum(s["bytes_text"] for s in manifest["sources"])
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {manifest['documents']:,} records ({manifest['bytes_text'] / 1e9:.2f} GB)")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
