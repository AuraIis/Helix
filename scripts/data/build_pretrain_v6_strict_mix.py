#!/usr/bin/env python3
"""Build a strict v6 candidate mix from audited source-separated data."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable


BAD_RE = re.compile(
    r"\[(?:deleted|removed)\]|onlyfans|discord\.gg|telegram|whatsapp|free karma|"
    r"subscribe to|promo code|coupon code|buy now|nsfw|porn|sexkontakte|casino|"
    r"sportwetten|warenkorb|rabattcode|cookie policy|xxxHHxxx",
    re.I,
)
URL_RE = re.compile(r"https?://|www\.", re.I)
HTML_RE = re.compile(r"<\s*/?\s*(html|body|div|script|style|iframe)\b|href=|&(?:amp|gt|lt|quot);", re.I)
COMMERCE_RE = re.compile(r"\b(warenkorb|rabatt|gutschein|checkout|kaufen|shop|preis|automarkt)\b", re.I)
BOILER_RE = re.compile(r"\b(cookie|privacy policy|datenschutz|impressum|all rights reserved)\b", re.I)
LIST_TABLE_RE = re.compile(r"(^|\s)([-*]|\d{1,3}[.)])\s+|\|\s*[-:]+\s*\||\t|\.{3,}\s*\d{1,5}", re.I)
DE_SENT_RE = re.compile(r"\b(der|die|das|und|ist|nicht|frage|antwort|schritt|erklaere|erklare)\b", re.I)
SECRET_RE = re.compile(r"api[_-]?key|secret|password|private key|token", re.I)
GATE_NEAR_RE = re.compile(r"\bhauptstadt von deutschland\b", re.I)


def clean(text: Any) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def one_line(text: str) -> str:
    return re.sub(r"\s+", " ", clean(text)).strip()


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                yield line_no, json.loads(line)


def load_holdout(manifest: Path) -> set[tuple[str, int]]:
    holdout: set[tuple[str, int]] = set()
    with manifest.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("split") == "holdout":
                holdout.add((str(rec.get("source")), int(rec.get("line_no"))))
    return holdout


def reject_common(text: str, *, min_chars: int, max_chars: int, max_urls: int) -> str | None:
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    if BAD_RE.search(text):
        return "bad_marker"
    if HTML_RE.search(text):
        return "html"
    if GATE_NEAR_RE.search(text):
        return "gate_near"
    if len(URL_RE.findall(text)) > max_urls:
        return "url_dense"
    alpha = sum(1 for c in text if c.isalpha()) / max(1, len(text))
    if alpha < 0.32:
        return "low_alpha"
    return None


def keep_instruction(obj: dict[str, Any]) -> tuple[str | None, str | None]:
    text = clean(obj.get("text"))
    if not text:
        question = clean(obj.get("question"))
        answer = clean(obj.get("answer"))
        text = (
            "System: Du bist Auralis, ein hilfreicher deutscher KI-Assistent.\n"
            f"Frage: {question}\n"
            f"Antwort: {answer}"
        )
    reason = reject_common(text, min_chars=80, max_chars=8000, max_urls=0)
    if reason:
        return None, reason
    return one_line(text), None


def keep_fineweb(obj: dict[str, Any]) -> tuple[str | None, str | None]:
    text = clean(obj.get("text"))
    reason = reject_common(text, min_chars=700, max_chars=12000, max_urls=0)
    if reason:
        return None, reason
    if COMMERCE_RE.search(text):
        return None, "commerce"
    if BOILER_RE.search(text):
        return None, "boilerplate"
    if LIST_TABLE_RE.search(text):
        return None, "list_table"
    if not DE_SENT_RE.search(text):
        return None, "weak_de_signal"
    return one_line(text), None


def keep_openmath(obj: dict[str, Any]) -> tuple[str | None, str | None]:
    problem = clean(obj.get("problem"))
    solution = clean(obj.get("solution"))
    expected = clean(obj.get("expected_answer"))
    text = f"Problem: {problem}\nLoesung: {solution}"
    if expected:
        text += f"\nAntwort: {expected}"
    reason = reject_common(text, min_chars=180, max_chars=7000, max_urls=0)
    if reason:
        return None, reason
    return one_line(text), None


def keep_code(obj: dict[str, Any]) -> tuple[str | None, str | None]:
    content = clean(obj.get("content"))
    path = clean(obj.get("path"))
    repo = clean(obj.get("repo_name"))
    text = f"<|code|>[python]\n<filename>{path}\n<reponame>{repo}\n{content}\n<|endcode|>"
    if len(content) < 160:
        return None, "too_short"
    if len(content) > 18000:
        return None, "too_long"
    if SECRET_RE.search(content):
        return None, "secret_word"
    if HTML_RE.search(content):
        return None, "html"
    if len(URL_RE.findall(content)) > 2:
        return None, "url_dense"
    return one_line(text), None


SourceSpec = tuple[str, Path, str, Callable[[dict[str, Any]], tuple[str | None, str | None]], int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/training/pretrain_v6_candidates"))
    parser.add_argument("--instruction-dir", type=Path, default=Path("data/training/pretrain_v6_instruction_de_strict"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/training/pretrain_v6_strict_mix"))
    parser.add_argument("--manifest", type=Path, default=Path("data/training/pretrain_v6_candidates/source_disjoint_manifest_v2.jsonl"))
    parser.add_argument("--max-instruction", type=int, default=25000)
    parser.add_argument("--max-fineweb", type=int, default=5000)
    parser.add_argument("--max-openmath", type=int, default=5000)
    parser.add_argument("--max-code", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    holdout = load_holdout(args.manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "mix_full.txt"
    reject_path = args.out_dir / "reject_samples.jsonl"
    sources: list[SourceSpec] = [
        (
            "instruction_de_strict",
            args.instruction_dir / "instruction_de_train.jsonl",
            "instruction_de_strict",
            keep_instruction,
            args.max_instruction,
        ),
        (
            "fineweb2_deu_latn",
            args.input_dir / "fineweb2_deu_latn" / "fineweb2_deu_latn.jsonl",
            "fineweb2_deu_latn",
            keep_fineweb,
            args.max_fineweb,
        ),
        (
            "openmathinstruct2",
            args.input_dir / "openmathinstruct2_capped" / "openmathinstruct2_capped.jsonl",
            "openmathinstruct2",
            keep_openmath,
            args.max_openmath,
        ),
        (
            "codeparrot_permissive",
            args.input_dir / "codeparrot_clean_python_permissive" / "codeparrot_clean_python_permissive.jsonl",
            "codeparrot_permissive",
            keep_code,
            args.max_code,
        ),
    ]
    stats: dict[str, Any] = {
        "input_dir": str(args.input_dir),
        "instruction_dir": str(args.instruction_dir),
        "holdout_manifest": str(args.manifest),
        "out": str(out_path),
        "sources": {},
    }
    total_docs = 0
    total_bytes = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as out, reject_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as rejects:
        for public_name, path, manifest_name, keeper, max_written in sources:
            counts: Counter[str] = Counter()
            if not path.is_file():
                counts["missing_file"] += 1
                stats["sources"][public_name] = dict(counts)
                continue
            for line_no, obj in iter_jsonl(path):
                counts["seen"] += 1
                if (manifest_name, line_no) in holdout:
                    counts["holdout_skip"] += 1
                    continue
                text, reason = keeper(obj)
                if reason:
                    counts[f"reject_{reason}"] += 1
                    if counts[f"reject_{reason}"] <= 5:
                        rejects.write(
                            json.dumps(
                                {
                                    "line_no": line_no,
                                    "preview": str(obj)[:500],
                                    "reason": reason,
                                    "source": public_name,
                                },
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                            + "\n"
                        )
                    continue
                out.write(text + "\n")
                counts["written"] += 1
                total_docs += 1
                total_bytes += len(text.encode("utf-8"))
                if counts["written"] >= max_written:
                    break
            stats["sources"][public_name] = dict(counts)
    stats["documents"] = total_docs
    stats["bytes"] = total_bytes
    stats["estimated_tokens_4bpt"] = int(total_bytes / 4)
    (args.out_dir / "manifest.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
