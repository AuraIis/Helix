#!/usr/bin/env python3
"""Build a tiny strict v6 canary mix from downloaded candidate sources."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


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
LIST_TABLE_RE = re.compile(r"(^|\s)([-*\u2022]|\d{1,3}[.)])\s+|\|\s*[-:]+\s*\||\t|\.{3,}\s*\d{1,5}", re.I)
EN_SENT_RE = re.compile(r"\b(the|and|you|write|explain|answer|question|therefore|because)\b", re.I)
DE_SENT_RE = re.compile(r"\b(der|die|das|und|ist|nicht|frage|antwort|schritt|erklaere|erkl\u00e4re)\b", re.I)
SECRET_RE = re.compile(r"api[_-]?key|secret|password|private key|token", re.I)


def clean(text: Any) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def one_line(text: str) -> str:
    return re.sub(r"\s+", " ", clean(text)).strip()


def load_holdout(manifest: Path) -> set[tuple[str, int]]:
    holdout: set[tuple[str, int]] = set()
    if not manifest.is_file():
        return holdout
    with manifest.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("split") == "holdout":
                holdout.add((str(rec.get("source")), int(rec.get("line_no"))))
    return holdout


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                yield line_no, json.loads(line)


def reject_common(text: str, *, min_chars: int, max_chars: int, max_urls: int) -> str | None:
    if len(text) < min_chars:
        return "too_short"
    if len(text) > max_chars:
        return "too_long"
    if BAD_RE.search(text):
        return "bad_marker"
    if HTML_RE.search(text):
        return "html"
    if len(URL_RE.findall(text)) > max_urls:
        return "url_dense"
    alpha = sum(1 for c in text if c.isalpha()) / max(1, len(text))
    if alpha < 0.32:
        return "low_alpha"
    return None


def keep_oasst(obj: dict[str, Any]) -> tuple[str | None, str | None]:
    q = clean(obj.get("question"))
    a = clean(obj.get("answer"))
    text = f"System: Du bist Auralis und antwortest hilfreich, knapp und auf Deutsch.\nFrage: {q}\nAntwort: {a}"
    reason = reject_common(text, min_chars=120, max_chars=5000, max_urls=0)
    if reason:
        return None, reason
    if EN_SENT_RE.search(a) and not DE_SENT_RE.search(a):
        return None, "english_answer"
    return one_line(text), None


def keep_fineweb(obj: dict[str, Any]) -> tuple[str | None, str | None]:
    text = clean(obj.get("text"))
    reason = reject_common(text, min_chars=500, max_chars=12000, max_urls=0)
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
    reason = reject_common(text, min_chars=160, max_chars=7000, max_urls=0)
    if reason:
        return None, reason
    return one_line(text), None


def keep_code(obj: dict[str, Any]) -> tuple[str | None, str | None]:
    content = clean(obj.get("content"))
    path = clean(obj.get("path"))
    repo = clean(obj.get("repo_name"))
    text = f"<|code|>[python]\n<filename>{path}\n<reponame>{repo}\n{content}\n<|endcode|>"
    if len(content) < 120:
        return None, "too_short"
    if len(content) > 18000:
        return None, "too_long"
    if SECRET_RE.search(content):
        return None, "secret_word"
    if HTML_RE.search(content):
        return None, "html"
    # URLs in code comments are common, but excessive URL/comment boilerplate is
    # bad for this tiny canary.
    if len(URL_RE.findall(content)) > 2:
        return None, "url_dense"
    return one_line(text), None


SOURCES = {
    "oasst1_de": ("oasst1_de/oasst1_de.jsonl", "oasst1_de", keep_oasst),
    "fineweb2_deu_latn": ("fineweb2_deu_latn/fineweb2_deu_latn.jsonl", "fineweb2_deu_latn", keep_fineweb),
    "openmathinstruct2": ("openmathinstruct2_capped/openmathinstruct2_capped.jsonl", "openmathinstruct2", keep_openmath),
    "codeparrot_permissive": (
        "codeparrot_clean_python_permissive/codeparrot_clean_python_permissive.jsonl",
        "codeparrot_permissive",
        keep_code,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("data/training/pretrain_v6_candidates"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/training/pretrain_v6_canary_strict"))
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--max-per-source", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.manifest or args.input_dir / "source_disjoint_manifest.jsonl"
    holdout = load_holdout(manifest)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "mix_full.txt"
    reject_path = args.out_dir / "reject_samples.jsonl"
    stats: dict[str, Any] = {
        "input_dir": str(args.input_dir),
        "out": str(out_path),
        "holdout_manifest": str(manifest),
        "sources": {},
    }
    total_docs = 0
    total_bytes = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as out, reject_path.open(
        "w", encoding="utf-8", newline="\n"
    ) as rejects:
        for public_name, (rel_path, manifest_name, keeper) in SOURCES.items():
            path = args.input_dir / rel_path
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
                                {"source": public_name, "line_no": line_no, "reason": reason, "preview": str(obj)[:500]},
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
                if counts["written"] >= args.max_per_source:
                    break
            stats["sources"][public_name] = dict(counts)
    stats["documents"] = total_docs
    stats["bytes"] = total_bytes
    stats["estimated_tokens_4bpt"] = int(total_bytes / 4)
    (args.out_dir / "manifest.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
