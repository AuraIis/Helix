#!/usr/bin/env python3
"""Build an expanded v6 test mix from strict base plus extra audited sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SPACE_RE = re.compile(r"\s+")


def clean(text: Any) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def one_line(text: str) -> str:
    return SPACE_RE.sub(" ", clean(text)).strip()


def sha(text: str) -> str:
    norm = one_line(text).lower()
    return hashlib.sha256(norm.encode("utf-8", errors="replace")).hexdigest()


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            if line.strip():
                yield line_no, json.loads(line)


def load_holdout(path: Path) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("split") == "holdout":
                out.add((str(rec.get("source")), int(rec.get("line_no"))))
    return out


def source_text(source: str, obj: dict[str, Any]) -> str:
    if source.startswith("german_commons_"):
        return one_line(obj.get("text"))
    if source == "codeparrot_clean_python_permissive_plus":
        content = clean(obj.get("content"))
        path = clean(obj.get("path"))
        repo = clean(obj.get("repo_name"))
        return one_line(f"<|code|>[python]\n<filename>{path}\n<reponame>{repo}\n{content}\n<|endcode|>")
    if source == "natural_questions_german":
        return one_line(f"Frage: {clean(obj.get('question'))}\nAntwort: {clean(obj.get('answer'))}")
    if source.startswith("avemio_german_rag_sft"):
        return one_line(
            f"System: {clean(obj.get('system'))}\n"
            f"Frage: {clean(obj.get('question'))}\n"
            f"Antwort: {clean(obj.get('answer'))}"
        )
    return one_line(obj.get("text") or obj)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-mix", type=Path, default=Path("data/training/pretrain_v6_strict_mix/mix_full.txt"))
    parser.add_argument("--extra-dir", type=Path, default=Path("data/training/pretrain_v6_extra_candidates"))
    parser.add_argument("--extra-manifest", type=Path, default=Path("data/training/pretrain_v6_extra_candidates/source_disjoint_manifest.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/training/pretrain_v6_expanded_test_mix"))
    parser.add_argument("--include-sharealike", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    holdout = load_holdout(args.extra_manifest)
    out_path = args.out_dir / "mix_full.txt"
    seen: set[str] = set()
    stats: dict[str, Any] = {
        "base_mix": str(args.base_mix),
        "extra_dir": str(args.extra_dir),
        "extra_manifest": str(args.extra_manifest),
        "include_sharealike": args.include_sharealike,
        "out": str(out_path),
        "sources": {},
    }
    source_paths = {
        "german_commons_web_wikipedia": args.extra_dir / "german_commons_web_wikipedia" / "german_commons_web_wikipedia.jsonl",
        "german_commons_web_wikidiscussions": args.extra_dir / "german_commons_web_wikidiscussions" / "german_commons_web_wikidiscussions.jsonl",
        "german_commons_scientific_wikibooks": args.extra_dir / "german_commons_scientific_wikibooks" / "german_commons_scientific_wikibooks.jsonl",
        "german_commons_scientific_wikiversity": args.extra_dir / "german_commons_scientific_wikiversity" / "german_commons_scientific_wikiversity.jsonl",
        "codeparrot_clean_python_permissive_plus": args.extra_dir / "codeparrot_clean_python_permissive_plus" / "codeparrot_clean_python_permissive_plus.jsonl",
    }
    if args.include_sharealike:
        source_paths.update(
            {
                "natural_questions_german": args.extra_dir / "natural_questions_german" / "natural_questions_german.jsonl",
                "avemio_german_rag_sft_qa": args.extra_dir
                / "avemio_german_rag_sft_qa_without_timedifference"
                / "avemio_german_rag_sft_qa_without_timedifference.jsonl",
                "avemio_german_rag_sft_summarizations": args.extra_dir
                / "avemio_german_rag_sft_summarizations"
                / "avemio_german_rag_sft_summarizations.jsonl",
            }
        )

    total_docs = 0
    total_bytes = 0
    with out_path.open("w", encoding="utf-8", newline="\n") as out:
        base_counts: Counter[str] = Counter()
        with args.base_mix.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                text = one_line(line)
                if not text:
                    continue
                h = sha(text)
                if h in seen:
                    base_counts["duplicate_skip"] += 1
                    continue
                seen.add(h)
                out.write(text + "\n")
                base_counts["written"] += 1
                total_docs += 1
                total_bytes += len(text.encode("utf-8"))
        stats["sources"]["base_strict_mix"] = dict(base_counts)

        for source, path in source_paths.items():
            counts: Counter[str] = Counter()
            if not path.is_file():
                counts["missing_file"] += 1
                stats["sources"][source] = dict(counts)
                continue
            for line_no, obj in iter_jsonl(path):
                counts["seen"] += 1
                if (source, line_no) in holdout:
                    counts["holdout_skip"] += 1
                    continue
                text = source_text(source, obj)
                if not text:
                    counts["empty_skip"] += 1
                    continue
                h = sha(text)
                if h in seen:
                    counts["duplicate_skip"] += 1
                    continue
                seen.add(h)
                out.write(text + "\n")
                counts["written"] += 1
                total_docs += 1
                total_bytes += len(text.encode("utf-8"))
            stats["sources"][source] = dict(counts)

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
