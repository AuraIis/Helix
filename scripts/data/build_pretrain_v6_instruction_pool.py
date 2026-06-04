#!/usr/bin/env python3
"""Build a strict German instruction pool from v6 candidate datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


BAD_RE = re.compile(
    r"\[(?:deleted|removed)\]|onlyfans|discord\.gg|telegram|whatsapp|free karma|"
    r"promo code|coupon code|buy now|nsfw|porn|sexkontakte|casino|sportwetten|"
    r"warenkorb|rabattcode|xxxHHxxx",
    re.I,
)
URL_RE = re.compile(r"https?://|www\.", re.I)
HTML_RE = re.compile(r"<\s*/?\s*(html|body|div|script|style|iframe)\b|href=|&(?:amp|gt|lt|quot);", re.I)
EN_RE = re.compile(r"\b(the|and|you|write|explain|answer|question|therefore|because)\b", re.I)
DE_RE = re.compile(r"\b(der|die|das|und|ist|nicht|ich|du|sie|wir|frage|antwort|schritt)\b", re.I)
WEAK_TASK_RE = re.compile(r"\b(text-to-speech|audiodatei|audio controls|mein hund.*abschlussball)\b", re.I)
GATE_NEAR_RE = re.compile(r"\bhauptstadt von deutschland\b", re.I)


SOURCES = {
    "oasst_de_conversations": "oasst_de_conversations/oasst_de_conversations.jsonl",
    "alpaca_gpt4_de": "alpaca_gpt4_deutsch/alpaca_gpt4_deutsch.jsonl",
}


def clean(text: Any) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", clean(text)).lower()


def sha(text: str) -> str:
    return hashlib.sha256(norm(text).encode("utf-8", errors="replace")).hexdigest()


def load_holdout(manifest: Path) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    if not manifest.is_file():
        return out
    for line in manifest.open("r", encoding="utf-8", errors="replace"):
        rec = json.loads(line)
        if rec.get("split") == "holdout":
            out.add((str(rec.get("source")), int(rec.get("line_no"))))
    return out


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                yield line_no, json.loads(line)


def reject(question: str, answer: str) -> str | None:
    q = clean(question)
    a = clean(answer)
    text = f"{q}\n{a}"
    if len(q) < 8 or len(a) < 20:
        return "too_short"
    if len(text) > 6000:
        return "too_long"
    if BAD_RE.search(text):
        return "bad_marker"
    if HTML_RE.search(text):
        return "html"
    if len(URL_RE.findall(text)) > 0:
        return "url"
    if WEAK_TASK_RE.search(text):
        return "weak_task"
    if GATE_NEAR_RE.search(text):
        return "gate_near"
    if EN_RE.search(a) and not DE_RE.search(a):
        return "english_answer"
    if not DE_RE.search(text):
        return "weak_de_signal"
    alpha = sum(1 for c in text if c.isalpha()) / max(1, len(text))
    if alpha < 0.35:
        return "low_alpha"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/training/pretrain_v6_candidates"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/training/pretrain_v6_instruction_de_strict"))
    parser.add_argument("--manifest", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.manifest or args.input_dir / "source_disjoint_manifest_v2.jsonl"
    holdout = load_holdout(manifest)
    seen_hashes: set[str] = set()

    train_jsonl = args.out_dir / "instruction_de_train.jsonl"
    holdout_jsonl = args.out_dir / "instruction_de_holdout.jsonl"
    train_txt = args.out_dir / "instruction_de_train.txt"
    reject_path = args.out_dir / "reject_samples.jsonl"

    stats: dict[str, Any] = {
        "input_dir": str(args.input_dir),
        "holdout_manifest": str(manifest),
        "outputs": {
            "train_jsonl": str(train_jsonl),
            "holdout_jsonl": str(holdout_jsonl),
            "train_txt": str(train_txt),
        },
        "sources": {},
    }

    with train_jsonl.open("w", encoding="utf-8", newline="\n") as train_j, holdout_jsonl.open(
        "w", encoding="utf-8", newline="\n"
    ) as hold_j, train_txt.open("w", encoding="utf-8", newline="\n") as train_t, reject_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as reject_f:
        for source_name, rel_path in SOURCES.items():
            path = args.input_dir / rel_path
            counts: Counter[str] = Counter()
            for line_no, obj in iter_jsonl(path):
                counts["seen"] += 1
                q = clean(obj.get("question"))
                a = clean(obj.get("answer"))
                reason = reject(q, a)
                if reason:
                    counts[f"reject_{reason}"] += 1
                    if counts[f"reject_{reason}"] <= 5:
                        reject_f.write(
                            json.dumps(
                                {"source": source_name, "line_no": line_no, "reason": reason, "question": q[:300], "answer": a[:300]},
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                    continue
                h = sha(f"{q}\n{a}")
                if h in seen_hashes:
                    counts["reject_duplicate"] += 1
                    continue
                seen_hashes.add(h)
                rec = {
                    "source": obj.get("source", source_name),
                    "source_file": source_name,
                    "source_line": line_no,
                    "license": obj.get("license", "Apache-2.0"),
                    "messages": [
                        {"role": "system", "content": "Du bist Auralis, ein hilfreicher deutscher KI-Assistent."},
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                    "question": q,
                    "answer": a,
                }
                if (source_name, line_no) in holdout:
                    hold_j.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                    counts["holdout"] += 1
                else:
                    train_j.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                    train_t.write(f"System: Du bist Auralis, ein hilfreicher deutscher KI-Assistent.\nFrage: {q}\nAntwort: {a}\n\n")
                    counts["train"] += 1
            stats["sources"][source_name] = dict(counts)

    stats["bytes_train_txt"] = train_txt.stat().st_size
    stats["estimated_train_tokens_4bpt"] = int(stats["bytes_train_txt"] / 4)
    stats["train_records"] = sum(v.get("train", 0) for v in stats["sources"].values())
    stats["holdout_records"] = sum(v.get("holdout", 0) for v in stats["sources"].values())
    (args.out_dir / "manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
