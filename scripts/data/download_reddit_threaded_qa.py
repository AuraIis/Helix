#!/usr/bin/env python3
"""Download filtered Reddit threaded Q&A pairs.

Source: HuggingFaceGECLM/REDDIT_threaded. Each row is a short conversation
thread. This script converts high-signal threads into simple question/answer
pairs suitable for a small QA/dialogue booster, not for raw base pretraining.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_SPLITS = [
    "explainlikeimfive",
    "askscience",
    "AskHistorians",
    "IWantToLearn",
    "LifeProTips",
    "YouShouldKnow",
    "personalfinance",
    "buildapc",
    "DIY",
    "programming",
    "books",
    "technology",
    "space",
    "history",
]

TURN_RE = re.compile(r"(?:^|\s)([A-Za-z0-9_-]{2,24}):\s")
URL_RE = re.compile(r"https?://|www\.", re.I)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")
BAD_RE = re.compile(
    r"\[(?:deleted|removed)\]|onlyfans|discord\.gg|telegram|whatsapp|free karma|"
    r"upvote if|subscribe to|promo code|coupon code|buy now|nsfw|porn|sex chat|"
    r"fuck|fucking|shit|bullshit|bitch|asshole|i am a bot|beep boop|performed automatically|"
    r"question answered|thanks everyone|thanks in advance|edit:|eta:",
    re.I,
)
META_RE = re.compile(
    r"\b(reddit|subreddit|upvote|downvote|karma|mods?|moderators?|crosspost|"
    r"front page|reddit gold|thanks for the gold|pm me|throwaway)\b",
    re.I,
)
QUESTION_START_RE = re.compile(
    r"^\s*(?:why|what|how|when|where|who|which|can|could|would|should|does|do|did|is|are|"
    r"was|were|if|eli5)\b",
    re.I,
)
QUOTE_LINE_RE = re.compile(r"^\s*>.*$", re.M)


def normalize(text: str) -> str:
    text = html.unescape(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = text.replace("&gt;", ">").replace("&lt;", "<").replace("&amp;", "&")
    text = text.replace("\u200b", " ")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_reddit_quotes(text: str) -> str:
    text = QUOTE_LINE_RE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return normalize(text)


def split_turns(text: str) -> list[tuple[str, str]]:
    matches = list(TURN_RE.finditer(text))
    turns: list[tuple[str, str]] = []
    for i, match in enumerate(matches):
        author = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = normalize(text[start:end])
        if body:
            turns.append((author, body))
    return turns


def text_ok(text: str, *, min_chars: int, max_chars: int, allow_meta: bool = False) -> str | None:
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    if BAD_RE.search(text):
        return "bad_marker"
    if not allow_meta and META_RE.search(text):
        return "reddit_meta"
    words = WORD_RE.findall(text)
    if len(words) < 12:
        return "too_few_words"
    urls = len(URL_RE.findall(text))
    question_marks = text.count("?")
    if question_marks > 3:
        return "too_many_questions"
    if urls > 1:
        return "too_many_urls"
    if urls == 1 and len(words) < 40:
        return "link_dominant"
    alpha = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha < 0.45:
        return "low_alpha_density"
    return None


def clean_question(text: str, *, require_question_mark: bool) -> str:
    text = re.sub(r"^\s*(ELI5|CMV|LPT|YSK)\s*:\s*", "", text, flags=re.I)
    text = strip_reddit_quotes(text)
    if require_question_mark and "?" not in text:
        return ""
    if require_question_mark and not QUESTION_START_RE.search(text):
        return ""
    return normalize(text)


def make_pair(row: dict[str, Any], split: str, args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    total_score = float(row.get("total_score") or 0.0)
    avg_score = float(row.get("avg_score") or 0.0)
    num_messages = int(row.get("num_messages") or 0)
    if total_score < args.min_total_score:
        return None, "low_total_score"
    if avg_score < args.min_avg_score:
        return None, "low_avg_score"
    if num_messages < 2:
        return None, "too_few_messages"

    turns = split_turns(str(row.get("text") or ""))
    if len(turns) < 2:
        return None, "parse_failed"
    question_author, question = turns[0]
    question = clean_question(question, require_question_mark=args.require_question_mark)
    if not question:
        return None, "question_not_question_like"
    q_reason = text_ok(question, min_chars=args.min_question_chars, max_chars=args.max_question_chars, allow_meta=False)
    if q_reason:
        return None, f"question_{q_reason}"

    answer_parts: list[str] = []
    answer_authors: list[str] = []
    for author, body in turns[1:]:
        if author == question_author:
            continue
        body = strip_reddit_quotes(body)
        reason = text_ok(body, min_chars=args.min_answer_chars, max_chars=args.max_answer_chars, allow_meta=False)
        if reason:
            continue
        answer_parts.append(body)
        answer_authors.append(author)
        if len(answer_parts) >= args.max_answers_per_thread:
            break
    if not answer_parts:
        return None, "no_good_answer"
    answer = normalize("\n\n".join(answer_parts))
    pair = {
        "source": "HuggingFaceGECLM/REDDIT_threaded",
        "split": split,
        "thread_id": row.get("thread_id"),
        "total_score": row.get("total_score"),
        "avg_score": row.get("avg_score"),
        "num_messages": row.get("num_messages"),
        "question": question,
        "answer": answer,
        "answer_authors": answer_authors,
    }
    return pair, "written"


def stream_split(split: str, *, jsonl, txt, args: argparse.Namespace) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    seen = 0
    written = 0
    ds = load_dataset(args.dataset, split=split, streaming=True, trust_remote_code=False)
    for row in ds:
        seen += 1
        pair, reason = make_pair(row, split, args)
        stats[reason] += 1
        if pair:
            jsonl.write(json.dumps(pair, ensure_ascii=False, sort_keys=True) + "\n")
            txt.write(f"Frage: {pair['question']}\nAntwort: {pair['answer'].replace(chr(10), ' ')}\n\n")
            written += 1
        if seen >= args.max_scan_per_split or written >= args.max_pairs_per_split:
            break
        if seen % args.log_every == 0:
            print(f"{split}: seen={seen:,} written={written:,}", flush=True)
    stats["seen"] = seen
    stats["written"] = written
    return {"split": split, **dict(stats)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="HuggingFaceGECLM/REDDIT_threaded")
    parser.add_argument("--split", action="append", default=None)
    parser.add_argument("--max-pairs-per-split", type=int, default=25_000)
    parser.add_argument("--max-scan-per-split", type=int, default=400_000)
    parser.add_argument("--min-total-score", type=float, default=5.0)
    parser.add_argument("--min-avg-score", type=float, default=1.0)
    parser.add_argument("--min-question-chars", type=int, default=40)
    parser.add_argument("--max-question-chars", type=int, default=900)
    parser.add_argument("--min-answer-chars", type=int, default=120)
    parser.add_argument("--max-answer-chars", type=int, default=3_500)
    parser.add_argument("--max-answers-per-thread", type=int, default=2)
    parser.add_argument("--require-question-mark", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=50_000)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "reddit_threaded_qa.jsonl"
    txt_path = args.output_dir / "reddit_threaded_qa.txt"
    manifest_path = args.output_dir / "manifest.json"
    splits = args.split or DEFAULT_SPLITS
    summary = {
        "dataset": args.dataset,
        "splits": [],
        "requested_splits": splits,
        "filters": {
            "max_pairs_per_split": args.max_pairs_per_split,
            "max_scan_per_split": args.max_scan_per_split,
            "min_total_score": args.min_total_score,
            "min_avg_score": args.min_avg_score,
            "min_question_chars": args.min_question_chars,
            "max_question_chars": args.max_question_chars,
            "min_answer_chars": args.min_answer_chars,
            "max_answer_chars": args.max_answer_chars,
            "max_answers_per_thread": args.max_answers_per_thread,
            "require_question_mark": args.require_question_mark,
        },
    }
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as jsonl, txt_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as txt:
        for split in splits:
            print(f"=== {split} ===", flush=True)
            tmp_jsonl = args.output_dir / f"{split}.jsonl.tmp"
            tmp_txt = args.output_dir / f"{split}.txt.tmp"
            stats = None
            last_error = None
            for attempt in range(1, args.retries + 1):
                try:
                    with tmp_jsonl.open("w", encoding="utf-8", newline="\n") as tj, tmp_txt.open(
                        "w", encoding="utf-8", newline="\n"
                    ) as tt:
                        stats = stream_split(split, jsonl=tj, txt=tt, args=args)
                    break
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    last_error = repr(exc)
                    print(f"{split}: attempt {attempt}/{args.retries} failed: {last_error}", flush=True)
                    time.sleep(min(60, attempt * 5))
            if stats is None:
                stats = {"split": split, "error": last_error}
            else:
                with tmp_jsonl.open("r", encoding="utf-8") as tj:
                    shutil.copyfileobj(tj, jsonl, length=1024 * 1024)
                with tmp_txt.open("r", encoding="utf-8") as tt:
                    shutil.copyfileobj(tt, txt, length=1024 * 1024)
                tmp_jsonl.unlink(missing_ok=True)
                tmp_txt.unlink(missing_ok=True)
            summary["splits"].append(stats)
            manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["documents"] = sum(s.get("written", 0) for s in summary["splits"])
    summary["bytes_text"] = txt_path.stat().st_size if txt_path.exists() else 0
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {summary['documents']:,} Q&A pairs to {txt_path} ({summary['bytes_text'] / 1e9:.2f} GB)")
    print(f"wrote {manifest_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
