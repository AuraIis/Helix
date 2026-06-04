#!/usr/bin/env python3
"""Filter Auralis code corpora into a safer code-booster candidate."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.data.audit_code_corpus import (
    FILENAME_RE,
    GENERATED_RE,
    HTML_RE,
    LOG_RE,
    NOTEBOOK_RE,
    SECRET_PATTERNS,
    TODO_RE,
    URL_RE,
    VENDOR_RE,
    extract_code_for_syntax,
    iter_docs,
    line_length_stats,
    stable_hash,
)


def reject_reasons(text: str, lang: str, seen_hashes: set[str], min_chars: int) -> list[str]:
    reasons: list[str] = []
    stripped = text.strip()
    char_len = len(text)
    lines, long_lines, avg_line_len = line_length_stats(text)

    if char_len < min_chars:
        reasons.append("too_short")
    content_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
        and not line.startswith("<filename>")
        and not line.startswith("<reponame>")
        and not line.strip().startswith("#")
    ]
    import_only = content_lines and all(
        line.startswith(("import ", "from ")) or line.startswith("__all__")
        for line in content_lines
    )
    if import_only:
        reasons.append("import_only")
    if char_len > 60_000:
        reasons.append("too_long")
    if "\ufffd" in text:
        reasons.append("replacement_chars")
    if any((ord(ch) < 32 and ch not in "\n\r\t") for ch in text):
        reasons.append("control_chars")
    if lines <= 2 and char_len > 800:
        reasons.append("collapsed_or_minified")
    if long_lines >= max(3, int(lines * 0.20)) or avg_line_len > 180:
        reasons.append("long_line_dense")
    if GENERATED_RE.search(text):
        reasons.append("generated_or_compiled")
    if NOTEBOOK_RE.search(text):
        reasons.append("notebook_json")
    if HTML_RE.search(text):
        reasons.append("html_embedded")
    if LOG_RE.search(text):
        reasons.append("log_or_traceback")
    if URL_RE.search(text) and len(URL_RE.findall(text)) >= 4:
        reasons.append("url_dense")
    if TODO_RE.search(text) and len(TODO_RE.findall(text)) >= 8:
        reasons.append("todo_dense")

    filename_match = FILENAME_RE.search(text)
    if filename_match and VENDOR_RE.search(filename_match.group("path")):
        reasons.append("vendor_or_dependency_path")

    for name, pattern in SECRET_PATTERNS.items():
        if pattern.search(text):
            reasons.append(f"possible_secret_{name}")

    h = stable_hash(stripped)
    if h in seen_hashes:
        reasons.append("exact_duplicate")
    else:
        seen_hashes.add(h)

    if lang == "python":
        try:
            ast.parse(extract_code_for_syntax(type("DocLike", (), {"text": text, "lang": lang})()))
        except SyntaxError:
            reasons.append("python_syntax_error")

    return reasons


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter code corpus into clean wrapped docs.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--mode", choices=["auto", "code-wrapped", "helix-jsonl", "plain-lines"], default="auto")
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reject-samples", type=Path, default=None)
    args = parser.parse_args()

    seen_hashes: set[str] = set()
    stats = {
        "input": str(args.input),
        "output": str(args.output),
        "docs_seen": 0,
        "docs_kept": 0,
        "bytes_kept": 0,
        "kept_by_lang": Counter(),
        "reject_reasons": Counter(),
    }
    reject_samples: list[dict] = []

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as out:
        for doc in iter_docs(args.input, args.mode, code_only=False):
            stats["docs_seen"] += 1
            reasons = reject_reasons(doc.text, doc.lang, seen_hashes, min_chars=args.min_chars)
            if reasons:
                for reason in reasons:
                    stats["reject_reasons"][reason] += 1
                if len(reject_samples) < 200:
                    reject_samples.append(
                        {
                            "index": doc.index,
                            "lang": doc.lang,
                            "reasons": reasons,
                            "preview": doc.text[:800],
                        }
                    )
            else:
                wrapped = f"<|code|>[{doc.lang}]\n{doc.text.rstrip()}\n<|endcode|>\n"
                out.write(wrapped)
                stats["docs_kept"] += 1
                stats["bytes_kept"] += len(wrapped.encode("utf-8"))
                stats["kept_by_lang"][doc.lang] += 1
            if args.max_docs and stats["docs_seen"] >= args.max_docs:
                break

    payload = {
        **{k: v for k, v in stats.items() if k not in {"kept_by_lang", "reject_reasons"}},
        "kept_by_lang": dict(stats["kept_by_lang"].most_common()),
        "reject_reasons": dict(stats["reject_reasons"].most_common()),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.reject_samples:
        args.reject_samples.write_text(
            "\n".join(json.dumps(x, ensure_ascii=False) for x in reject_samples) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
