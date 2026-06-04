#!/usr/bin/env python3
"""Stream and filter large supplemental training sources.

Outputs stay source-separated so audits can decide final mix weights. This
script intentionally writes filtered JSONL/TXT only; it does not preserve raw
HF shards beyond the local streaming cache.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")
BAD_RE = re.compile(
    r"\[(?:deleted|removed)\]|onlyfans|discord\.gg|telegram|whatsapp|free karma|"
    r"upvote if|subscribe to|promo code|coupon code|buy now|nsfw|porn|sex chat|"
    r"i am a bot|beep boop|performed automatically",
    re.I,
)
URL_RE = re.compile(r"https?://|www\.", re.I)
REDDIT_META_RE = re.compile(r"\b(upvote|downvote|karma|mods?|subreddit|crosspost|thanks for the gold)\b", re.I)


def clean(text: Any) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def ok_text(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
    min_words: int,
    allow_urls: bool,
    reject_reddit_meta: bool,
) -> str | None:
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    if BAD_RE.search(text):
        return "bad_marker"
    if reject_reddit_meta and REDDIT_META_RE.search(text):
        return "reddit_meta"
    if not allow_urls and URL_RE.search(text):
        return "has_url"
    words = WORD_RE.findall(text)
    if len(words) < min_words:
        return "too_few_words"
    alpha = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha < 0.35:
        return "low_alpha_density"
    return None


def safe_detox(row: dict[str, Any], threshold: float) -> bool:
    if row.get("toxic") or row.get("redacted"):
        return False
    mods = row.get("detoxify_moderation") or []
    if not isinstance(mods, list):
        return True
    for mod in mods:
        if not isinstance(mod, dict):
            continue
        for key in ("toxicity", "severe_toxicity", "obscene", "identity_attack", "sexual_explicit", "threat", "insult"):
            try:
                if float(mod.get(key) or 0.0) > threshold:
                    return False
            except (TypeError, ValueError):
                continue
    return True


def write_wildchat(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    ds = load_dataset("allenai/WildChat-4.8M", split="train", streaming=True)
    stats: Counter[str] = Counter()
    seen = written = 0
    txt_path = out / "wildchat_en.txt"
    jsonl_path = out / "wildchat_en.jsonl"
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as jsonl, txt_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as txt:
        for row in ds:
            seen += 1
            if row.get("language") != "English":
                stats["non_english"] += 1
                continue
            if not safe_detox(row, args.detox_threshold):
                stats["unsafe"] += 1
                continue
            conv = row.get("conversation") or []
            if not isinstance(conv, list):
                stats["bad_conversation"] += 1
                continue
            turns = []
            for msg in conv[: args.max_turns_per_conversation]:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                if role not in {"user", "assistant"}:
                    continue
                content = clean(msg.get("content"))
                reason = ok_text(
                    content,
                    min_chars=args.min_chat_turn_chars,
                    max_chars=args.max_chat_turn_chars,
                    min_words=args.min_chat_turn_words,
                    allow_urls=False,
                    reject_reddit_meta=False,
                )
                if reason:
                    stats[f"turn_{reason}"] += 1
                    continue
                turns.append({"role": role, "content": content})
            if len(turns) < 2 or turns[0]["role"] != "user":
                stats["not_enough_turns"] += 1
                continue
            rec = {"source": "allenai/WildChat-4.8M", "model": row.get("model"), "turns": turns}
            jsonl.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            rendered = "\n".join(("Nutzer: " if t["role"] == "user" else "Assistent: ") + t["content"] for t in turns)
            txt.write(rendered + "\n\n")
            written += 1
            stats["written"] += 1
            if seen >= args.max_scan_per_source or written >= args.max_records_per_source:
                break
            if seen % args.log_every == 0:
                print(f"wildchat_en: seen={seen:,} written={written:,}", flush=True)
    return {"source": "wildchat_en", "seen": seen, "written": written, "bytes_text": txt_path.stat().st_size, "stats": dict(stats)}


def write_reddit_title_body(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    ds = load_dataset("BEE-spoke-data/reddit-title-body-hf", "deduped", split="train", streaming=True)
    stats: Counter[str] = Counter()
    seen = written = 0
    txt_path = out / "reddit_title_body.txt"
    jsonl_path = out / "reddit_title_body.jsonl"
    subreddit_keep = {s.lower() for s in args.reddit_subreddit_keep}
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as jsonl, txt_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as txt:
        for row in ds:
            seen += 1
            subreddit = clean(row.get("subreddit")).lower()
            if subreddit_keep and subreddit not in subreddit_keep:
                stats["subreddit_not_wanted"] += 1
                continue
            title = clean(row.get("title"))
            body = clean(row.get("body"))
            text = clean(f"{title}\n\n{body}")
            reason = ok_text(
                text,
                min_chars=args.min_reddit_doc_chars,
                max_chars=args.max_reddit_doc_chars,
                min_words=args.min_reddit_doc_words,
                allow_urls=False,
                reject_reddit_meta=True,
            )
            if reason:
                stats[reason] += 1
                continue
            rec = {"source": "BEE-spoke-data/reddit-title-body-hf", "subreddit": subreddit, "title": title, "body": body}
            jsonl.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            txt.write(f"Titel: {title}\nText: {body}\n\n")
            written += 1
            stats["written"] += 1
            if seen >= args.max_scan_per_source or written >= args.max_records_per_source:
                break
            if seen % args.log_every == 0:
                print(f"reddit_title_body: seen={seen:,} written={written:,}", flush=True)
    return {"source": "reddit_title_body", "seen": seen, "written": written, "bytes_text": txt_path.stat().st_size, "stats": dict(stats)}


def write_openmath_reasoning(args: argparse.Namespace, out: Path) -> dict[str, Any]:
    stats_total = {"source": "openmath_reasoning", "splits": [], "written": 0, "bytes_text": 0}
    for split in args.openmath_split:
        ds = load_dataset("nvidia/OpenMathReasoning", split=split, streaming=True)
        stats: Counter[str] = Counter()
        seen = written = 0
        txt_path = out / f"openmath_reasoning_{split}.txt"
        jsonl_path = out / f"openmath_reasoning_{split}.jsonl"
        with jsonl_path.open("w", encoding="utf-8", newline="\n") as jsonl, txt_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as txt:
            for row in ds:
                seen += 1
                problem = clean(row.get("problem"))
                solution = clean(row.get("generated_solution"))
                if solution == "n/a":
                    stats["no_solution"] += 1
                    continue
                if args.strip_think:
                    solution = re.sub(r"<think>.*?</think>", "", solution, flags=re.I | re.S).strip() or solution
                    solution = solution.replace("<think>", "").replace("</think>", "").strip()
                p_reason = ok_text(problem, min_chars=20, max_chars=6000, min_words=3, allow_urls=True, reject_reddit_meta=False)
                s_reason = ok_text(solution, min_chars=80, max_chars=args.max_math_solution_chars, min_words=8, allow_urls=True, reject_reddit_meta=False)
                if p_reason:
                    stats[f"problem_{p_reason}"] += 1
                    continue
                if s_reason:
                    stats[f"solution_{s_reason}"] += 1
                    continue
                rec = {
                    "source": "nvidia/OpenMathReasoning",
                    "split": split,
                    "problem": problem,
                    "solution": solution,
                    "expected_answer": clean(row.get("expected_answer")),
                    "problem_source": row.get("problem_source"),
                }
                jsonl.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                txt.write(f"Problem: {problem}\n\nSolution: {solution}\n\n")
                written += 1
                stats["written"] += 1
                if seen >= args.max_scan_per_source or written >= args.max_records_per_source:
                    break
                if seen % args.log_every == 0:
                    print(f"openmath_reasoning/{split}: seen={seen:,} written={written:,}", flush=True)
        split_stats = {"split": split, "seen": seen, "written": written, "bytes_text": txt_path.stat().st_size, "stats": dict(stats)}
        stats_total["splits"].append(split_stats)
        stats_total["written"] += written
        stats_total["bytes_text"] += txt_path.stat().st_size
    return stats_total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source", action="append", choices=["wildchat", "reddit_title_body", "openmath_reasoning"], default=None)
    parser.add_argument("--max-records-per-source", type=int, default=200_000)
    parser.add_argument("--max-scan-per-source", type=int, default=1_000_000)
    parser.add_argument("--log-every", type=int, default=50_000)
    parser.add_argument("--detox-threshold", type=float, default=0.15)
    parser.add_argument("--max-turns-per-conversation", type=int, default=6)
    parser.add_argument("--min-chat-turn-chars", type=int, default=20)
    parser.add_argument("--max-chat-turn-chars", type=int, default=4000)
    parser.add_argument("--min-chat-turn-words", type=int, default=4)
    parser.add_argument("--min-reddit-doc-chars", type=int, default=400)
    parser.add_argument("--max-reddit-doc-chars", type=int, default=6000)
    parser.add_argument("--min-reddit-doc-words", type=int, default=80)
    parser.add_argument(
        "--reddit-subreddit-keep",
        nargs="*",
        default=[
            "askscience",
            "explainlikeimfive",
            "AskHistorians",
            "science",
            "history",
            "space",
            "programming",
            "learnprogramming",
            "personalfinance",
            "buildapc",
            "DIY",
            "gardening",
            "EatCheapAndHealthy",
        ],
    )
    parser.add_argument("--openmath-split", action="append", default=["cot", "tir"])
    parser.add_argument("--max-math-solution-chars", type=int, default=12000)
    parser.add_argument("--strip-think", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = args.source or ["wildchat", "reddit_title_body", "openmath_reasoning"]
    manifest = {"sources": [], "filters": vars(args)}
    for source in selected:
        print(f"=== {source} ===", flush=True)
        if source == "wildchat":
            result = write_wildchat(args, args.output_dir)
        elif source == "reddit_title_body":
            result = write_reddit_title_body(args, args.output_dir)
        elif source == "openmath_reasoning":
            result = write_openmath_reasoning(args, args.output_dir)
        else:
            raise AssertionError(source)
        manifest["sources"].append(result)
        (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    manifest["documents"] = sum(s.get("written", 0) for s in manifest["sources"])
    manifest["bytes_text"] = sum(s.get("bytes_text", 0) for s in manifest["sources"])
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"wrote {manifest['documents']:,} docs ({manifest['bytes_text'] / 1e9:.2f} GB)")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
