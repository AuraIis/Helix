#!/usr/bin/env python3
"""Download and filter high-signal Reddit comments for pretraining.

This intentionally does *not* mirror all of Reddit. It streams selected
subreddit splits from ``HuggingFaceGECLM/REDDIT_comments`` and keeps only
comments that are useful as a small dialogue/explanation booster.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


DEFAULT_SUBREDDITS = [
    "explainlikeimfive",
    "askscience",
    "AskHistorians",
    "science",
    "todayilearned",
    "programming",
    "IWantToLearn",
    "DIY",
    "books",
    "technology",
]

URL_RE = re.compile(r"https?://|www\.", re.I)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")
SPAM_RE = re.compile(
    r"(onlyfans|discord\.gg|telegram|whatsapp|free karma|upvote if|subscribe to|"
    r"check out my|promo code|coupon code|buy now|nsfw|porn|sex chat)",
    re.I,
)
BOT_RE = re.compile(r"\b(i am a bot|beep boop|this action was performed automatically)\b", re.I)
MARKDOWN_JUNK_RE = re.compile(r"^\s*(?:&gt;|>|\*|\-|_|\[deleted\]|\[removed\])\s*$", re.I)
BAD_VALUES = {"[deleted]", "[removed]", "deleted", "removed", ""}


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def reject_reason(row: dict[str, Any], *, min_score: int, min_chars: int, max_chars: int) -> str | None:
    body = normalize(str(row.get("body") or ""))
    if body.lower() in BAD_VALUES:
        return "deleted_or_removed"
    if len(body) < min_chars:
        return "too_short"
    if len(body) > max_chars:
        return "too_long"
    try:
        score = int(row.get("score") or 0)
    except (TypeError, ValueError):
        score = 0
    if score < min_score:
        return "low_score"
    if MARKDOWN_JUNK_RE.match(body):
        return "markdown_junk"
    if BOT_RE.search(body):
        return "bot"
    if SPAM_RE.search(body):
        return "spam_or_nsfw"
    words = WORD_RE.findall(body)
    if len(words) < 12:
        return "too_few_words"
    url_count = len(URL_RE.findall(body))
    if url_count > 1:
        return "too_many_urls"
    if url_count == 1 and len(words) < 35:
        return "link_dominant"
    alpha = sum(1 for c in body if c.isalpha()) / max(len(body), 1)
    if alpha < 0.45:
        return "low_alpha_density"
    if body.count("\n") > 12:
        return "too_many_lines"
    return None


def row_to_text(row: dict[str, Any], subreddit: str) -> str:
    body = normalize(str(row.get("body") or ""))
    score = row.get("score")
    return f"Subreddit: r/{subreddit}\nScore: {score}\nKommentar: {body}"


def stream_subreddit(
    subreddit: str,
    *,
    output_jsonl,
    output_txt,
    args: argparse.Namespace,
) -> dict[str, Any]:
    stats: Counter[str] = Counter()
    written = 0
    seen = 0
    dataset = load_dataset(
        args.dataset,
        split=subreddit,
        streaming=True,
        trust_remote_code=False,
    )
    for row in dataset:
        seen += 1
        reason = reject_reason(
            row,
            min_score=args.min_score,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )
        if reason:
            stats[reason] += 1
        else:
            text = row_to_text(row, subreddit)
            out_row = {
                "subreddit": subreddit,
                "score": row.get("score"),
                "created_utc": row.get("created_utc"),
                "permalink": row.get("permalink"),
                "body": normalize(str(row.get("body") or "")),
            }
            output_jsonl.write(json.dumps(out_row, ensure_ascii=False, sort_keys=True) + "\n")
            output_txt.write(text.replace("\n", " ") + "\n")
            written += 1
            stats["written"] += 1
        if seen >= args.max_scan_per_subreddit or written >= args.max_docs_per_subreddit:
            break
        if seen % args.log_every == 0:
            print(f"{subreddit}: seen={seen:,} written={written:,}", flush=True)
    stats["seen"] = seen
    return {"subreddit": subreddit, **dict(stats)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="HuggingFaceGECLM/REDDIT_comments")
    parser.add_argument("--subreddit", action="append", default=None)
    parser.add_argument("--max-docs-per-subreddit", type=int, default=100_000)
    parser.add_argument("--max-scan-per-subreddit", type=int, default=1_000_000)
    parser.add_argument("--min-score", type=int, default=5)
    parser.add_argument("--min-chars", type=int, default=120)
    parser.add_argument("--max-chars", type=int, default=2_500)
    parser.add_argument("--log-every", type=int, default=50_000)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "reddit_quality.jsonl"
    txt_path = args.output_dir / "reddit_quality.txt"
    manifest_path = args.output_dir / "manifest.json"
    subreddits = args.subreddit or DEFAULT_SUBREDDITS
    summary = {
        "dataset": args.dataset,
        "subreddits": subreddits,
        "filters": {
            "max_docs_per_subreddit": args.max_docs_per_subreddit,
            "max_scan_per_subreddit": args.max_scan_per_subreddit,
            "min_score": args.min_score,
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
        },
        "splits": [],
    }
    with jsonl_path.open("w", encoding="utf-8", newline="\n") as jsonl, txt_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as txt:
        for subreddit in subreddits:
            print(f"=== {subreddit} ===", flush=True)
            split_jsonl = args.output_dir / f"{subreddit}.jsonl.tmp"
            split_txt = args.output_dir / f"{subreddit}.txt.tmp"
            last_error = None
            split_stats = None
            for attempt in range(1, args.retries + 1):
                try:
                    with split_jsonl.open("w", encoding="utf-8", newline="\n") as sj, split_txt.open(
                        "w", encoding="utf-8", newline="\n"
                    ) as st:
                        split_stats = stream_subreddit(subreddit, output_jsonl=sj, output_txt=st, args=args)
                    break
                except KeyboardInterrupt:
                    print("interrupted", file=sys.stderr)
                    raise
                except Exception as exc:  # network streams can fail mid-fragment
                    last_error = repr(exc)
                    print(f"{subreddit}: attempt {attempt}/{args.retries} failed: {last_error}", flush=True)
                    time.sleep(min(60, attempt * 5))
            if split_stats is None:
                summary["splits"].append({"subreddit": subreddit, "error": last_error})
                manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
                continue
            with split_jsonl.open("r", encoding="utf-8") as sj:
                shutil.copyfileobj(sj, jsonl, length=1024 * 1024)
            with split_txt.open("r", encoding="utf-8") as st:
                shutil.copyfileobj(st, txt, length=1024 * 1024)
            split_jsonl.unlink(missing_ok=True)
            split_txt.unlink(missing_ok=True)
            summary["splits"].append(split_stats)
            manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    docs = sum(split.get("written", 0) for split in summary["splits"])
    bytes_written = txt_path.stat().st_size if txt_path.exists() else 0
    summary["documents"] = docs
    summary["bytes_text"] = bytes_written
    manifest_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {docs:,} docs to {txt_path} ({bytes_written / 1e9:.2f} GB)")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
