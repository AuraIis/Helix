#!/usr/bin/env python3
"""Download filtered Reddit question/best-answer pairs.

Source: nreimers/reddit_question_best_answers. Compared with raw Reddit
threads, this dataset already groups a post with scored answers, so it is a
better candidate for a larger QA/dialogue booster.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import load_dataset


URL_RE = re.compile(r"https?://|www\.", re.I)
WORD_RE = re.compile(r"[A-Za-z][A-Za-z']+")
QUESTION_START_RE = re.compile(
    r"^\s*(?:why|what|how|when|where|who|which|can|could|would|should|does|do|did|is|are|"
    r"was|were|if|eli5)\b",
    re.I,
)
BAD_RE = re.compile(
    r"\[(?:deleted|removed)\]|onlyfans|discord\.gg|telegram|whatsapp|free karma|"
    r"upvote if|subscribe to|promo code|coupon code|buy now|nsfw|porn|sex chat|"
    r"fuck|fucking|shit|bullshit|bitch|asshole|i am a bot|beep boop|performed automatically|"
    r"question answered|thanks everyone|thanks in advance|edit:|eta:",
    re.I,
)
META_RE = re.compile(
    r"\b(reddit|subreddit|upvote|downvote|karma|mods?|moderators?|crosspost|"
    r"reddit gold|thanks for the gold|throwaway)\b",
    re.I,
)
CASUAL_QUESTION_RE = re.compile(
    r"\b(what(?:'s| is) your (?:favorite|favourite|best)|who else|do you like your life|"
    r"if you could|what would you do|does anyone know|can anyone recommend|where to start with|"
    r"when you call a girl|stoned|cannabis|weed|gun show|buy a gun|sex is dirty|"
    r"best story|plot holes|favourite emacs feature)\b",
    re.I,
)
QUOTE_LINE_RE = re.compile(r"^\s*>.*$", re.M)


def normalize(text: str) -> str:
    text = html.unescape(str(text or ""))
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = text.replace("\u200b", " ")
    text = QUOTE_LINE_RE.sub("", text)
    text = re.sub(r"\[[^\]]{1,120}\]\((?:https?://|www\.)[^)]+\)", "", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def text_ok(
    text: str,
    *,
    min_chars: int,
    max_chars: int,
    min_words: int,
    allow_question: bool,
    require_question: bool = False,
) -> str | None:
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    if BAD_RE.search(text):
        return "bad_marker"
    if META_RE.search(text):
        return "reddit_meta"
    if require_question and ("?" not in text or not QUESTION_START_RE.search(text)):
        return "not_question_like"
    if not allow_question and text.count("?") > 1:
        return "too_many_questions"
    words = WORD_RE.findall(text)
    if len(words) < min_words:
        return "too_few_words"
    urls = len(URL_RE.findall(text))
    if urls > 0:
        return "has_url"
    alpha = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha < 0.45:
        return "low_alpha_density"
    return None


def build_question(row: dict[str, Any], args: argparse.Namespace) -> str:
    title = normalize(row.get("title", ""))
    body = normalize(row.get("body", ""))
    if body and len(title) + len(body) + 2 <= args.max_question_chars:
        return normalize(f"{title}\n{body}")
    return title


def choose_answer(row: dict[str, Any], args: argparse.Namespace) -> tuple[str | None, str]:
    answers = row.get("answers") or []
    if not isinstance(answers, list):
        return None, "answers_not_list"
    candidates = []
    for answer in answers:
        if not isinstance(answer, dict):
            continue
        score = float(answer.get("score") or 0.0)
        if score < args.min_answer_score:
            continue
        body = normalize(answer.get("body", ""))
        reason = text_ok(
            body,
            min_chars=args.min_answer_chars,
            max_chars=args.max_answer_chars,
            min_words=args.min_answer_words,
            allow_question=False,
        )
        if reason:
            continue
        candidates.append((score, body))
    if not candidates:
        return None, "no_good_answer"
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1], "written"


def make_pair(row: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, str]:
    score = float(row.get("score") or 0.0)
    if score < args.min_question_score:
        return None, "low_question_score"
    question = build_question(row, args)
    q_reason = text_ok(
        question,
        min_chars=args.min_question_chars,
        max_chars=args.max_question_chars,
        min_words=args.min_question_words,
        allow_question=True,
        require_question=args.require_question,
    )
    if q_reason:
        return None, f"question_{q_reason}"
    if args.reject_casual_questions and CASUAL_QUESTION_RE.search(question):
        return None, "question_casual_or_sensitive"
    answer, reason = choose_answer(row, args)
    if answer is None:
        return None, reason
    return {
        "source": args.dataset,
        "question_score": row.get("score"),
        "question": question,
        "answer": answer,
    }, "written"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", default="nreimers/reddit_question_best_answers")
    parser.add_argument("--max-pairs", type=int, default=300_000)
    parser.add_argument("--max-scan", type=int, default=2_000_000)
    parser.add_argument("--min-question-score", type=float, default=10.0)
    parser.add_argument("--min-answer-score", type=float, default=5.0)
    parser.add_argument("--min-question-chars", type=int, default=35)
    parser.add_argument("--max-question-chars", type=int, default=1_000)
    parser.add_argument("--min-question-words", type=int, default=6)
    parser.add_argument("--min-answer-chars", type=int, default=220)
    parser.add_argument("--max-answer-chars", type=int, default=2_500)
    parser.add_argument("--min-answer-words", type=int, default=45)
    parser.add_argument("--require-question", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reject-casual-questions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=50_000)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = args.output_dir / "reddit_best_answers.jsonl"
    txt_path = args.output_dir / "reddit_best_answers.txt"
    manifest_path = args.output_dir / "manifest.json"
    stats: Counter[str] = Counter()
    seen = 0
    written = 0
    last_error = None

    for attempt in range(1, args.retries + 1):
        try:
            ds = load_dataset(args.dataset, split="train", streaming=True)
            with jsonl_path.open("w", encoding="utf-8", newline="\n") as jsonl, txt_path.open(
                "w", encoding="utf-8", newline="\n"
            ) as txt:
                for row in ds:
                    seen += 1
                    pair, reason = make_pair(row, args)
                    stats[reason] += 1
                    if pair:
                        jsonl.write(json.dumps(pair, ensure_ascii=False, sort_keys=True) + "\n")
                        txt.write(f"Frage: {pair['question']}\nAntwort: {pair['answer'].replace(chr(10), ' ')}\n\n")
                        written += 1
                    if seen >= args.max_scan or written >= args.max_pairs:
                        break
                    if seen % args.log_every == 0:
                        print(f"seen={seen:,} written={written:,}", flush=True)
            break
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last_error = repr(exc)
            print(f"attempt {attempt}/{args.retries} failed: {last_error}", flush=True)
            time.sleep(min(60, attempt * 5))
    else:
        raise RuntimeError(f"download failed after {args.retries} attempts: {last_error}")

    manifest = {
        "dataset": args.dataset,
        "documents": written,
        "seen": seen,
        "bytes_text": txt_path.stat().st_size if txt_path.exists() else 0,
        "filters": {
            "max_pairs": args.max_pairs,
            "max_scan": args.max_scan,
            "min_question_score": args.min_question_score,
            "min_answer_score": args.min_answer_score,
            "min_question_chars": args.min_question_chars,
            "max_question_chars": args.max_question_chars,
            "min_answer_chars": args.min_answer_chars,
            "max_answer_chars": args.max_answer_chars,
            "require_question": args.require_question,
            "reject_casual_questions": args.reject_casual_questions,
        },
        "stats": dict(stats),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {written:,} Q&A pairs to {txt_path} ({manifest['bytes_text'] / 1e9:.2f} GB)")
    print(f"wrote {manifest_path}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
