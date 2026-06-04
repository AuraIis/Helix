#!/usr/bin/env python3
"""Build the clean-v3.1 + booster v4 pretraining mix.

The output is one document per line, which matches the existing tokenizer.
JSONL booster sources are rendered into single-line documents so multi-line
chat/QA records do not get split into broken training examples.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_BASE = Path("/workspace/v2data/data/training/pretrain_clean_v31/mix_full.txt")
DEFAULT_OUT = Path("/workspace/v2data/data/training/pretrain_mix_v4_boosted")
DEFAULT_DNA = Path("/workspace/v2data/data/eval/knowledge_dna_v2_expanded_750/hybrid_corpus.txt")
DEFAULT_LARGE_QA = Path("/disk5v2data/data/large_qa_sources_v1")
DEFAULT_SUPP = Path("/disk5v2data/data/supplemental_large_sources_v1")
DEFAULT_REDDIT_BEST = Path("/disk5v2data/data/reddit_best_answers_v1/reddit_best_answers.jsonl")

CHAT_MARKER_RE = re.compile(r"<\|(?:im_start|im_end|endoftext|user|assistant|system)\|>|_end_of_the_data", re.I)
HTML_RE = re.compile(r"<\s*/?\s*(html|body|div|script|style|table|iframe)\b|&(?:amp|gt|lt|quot|#x200b);", re.I)
BAD_RE = re.compile(r"\[(?:deleted|removed)\]|discord\.gg|onlyfans|free karma|subscribe to|promo code", re.I)
THINK_RE = re.compile(r"</?think>", re.I)


@dataclass
class SourceStats:
    name: str
    kind: str
    path: str
    cap_bytes: int | None
    documents: int = 0
    bytes_written: int = 0
    skipped: dict[str, int] = field(default_factory=dict)


def inc(stats: SourceStats, reason: str) -> None:
    stats.skipped[reason] = stats.skipped.get(reason, 0) + 1


def normalize_doc(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = THINK_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def should_skip(text: str) -> str | None:
    if len(text) < 80:
        return "too_short"
    if CHAT_MARKER_RE.search(text):
        return "chat_marker"
    if HTML_RE.search(text):
        return "html_or_entity"
    if BAD_RE.search(text):
        return "bad_marker"
    alpha = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha < 0.30:
        return "low_alpha"
    return None


def iter_text_lines(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = normalize_doc(line)
            if line:
                yield line


def iter_dna_records(path: Path) -> Iterable[str]:
    if not path.is_file():
        return
    block: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                block.append(line.rstrip("\n"))
                continue
            if block:
                part = normalize_doc("\n".join(block))
                if part:
                    yield part
                block.clear()
    if block:
        part = normalize_doc("\n".join(block))
        if part:
            yield part


def iter_qa_jsonl(path: Path, prefix: str) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            obj = json.loads(line)
            question = normalize_doc(obj.get("question", ""))
            answer = normalize_doc(obj.get("answer", ""))
            system = normalize_doc(obj.get("system", ""))
            if system:
                yield f"{prefix} System: {system} Frage: {question} Antwort: {answer}"
            else:
                yield f"{prefix} Frage: {question} Antwort: {answer}"


def iter_wildchat_jsonl(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            obj = json.loads(line)
            turns = obj.get("turns") or []
            rendered = []
            for turn in turns[:6]:
                role = turn.get("role")
                content = normalize_doc(turn.get("content", ""))
                if not content:
                    continue
                label = "Nutzer" if role == "user" else "Assistent"
                rendered.append(f"{label}: {content}")
            if len(rendered) >= 2:
                yield "Dialog: " + " ".join(rendered)


def iter_math_jsonl(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            obj = json.loads(line)
            problem = normalize_doc(obj.get("problem", ""))
            solution = normalize_doc(obj.get("solution", ""))
            expected = normalize_doc(obj.get("expected_answer", ""))
            suffix = f" Erwartete Antwort: {expected}" if expected else ""
            yield f"Mathematik. Problem: {problem} Lösung: {solution}{suffix}"


def write_source(
    out_fh,
    *,
    name: str,
    kind: str,
    path: Path,
    docs: Iterable[str],
    cap_bytes: int | None,
) -> SourceStats:
    stats = SourceStats(name=name, kind=kind, path=str(path), cap_bytes=cap_bytes)
    for doc in docs:
        doc = normalize_doc(doc)
        reason = should_skip(doc)
        if reason:
            inc(stats, reason)
            continue
        line = doc + "\n"
        line_bytes = len(line.encode("utf-8"))
        if cap_bytes is not None and stats.bytes_written + line_bytes > cap_bytes:
            break
        out_fh.write(line)
        stats.documents += 1
        stats.bytes_written += line_bytes
    return stats


def write_audit_samples(mix_path: Path, out_dir: Path, *, n: int, seed: int) -> dict:
    rng = random.Random(seed)
    size = mix_path.stat().st_size
    samples = []
    flags = {"chat_marker": 0, "html_or_entity": 0, "bad_marker": 0, "too_short": 0, "low_alpha": 0}
    with mix_path.open("rb") as fh:
        for _ in range(n):
            fh.seek(rng.randrange(0, max(size, 1)))
            fh.readline()
            raw = fh.readline() or b""
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            reason = should_skip(text)
            if reason in flags:
                flags[reason] += 1
            samples.append(text[:800])
    (out_dir / "audit_samples.txt").write_text(
        "\n\n--- SAMPLE ---\n\n".join(samples),
        encoding="utf-8",
    )
    return {"sample_count": len(samples), "flag_counts": flags}


def copy_base_fast(src: Path, dst: Path) -> SourceStats:
    stats = SourceStats(name="clean_v31_base", kind="text_lines_fast_copy", path=str(src), cap_bytes=None)
    last_byte = b""
    with src.open("rb") as in_fh, dst.open("wb") as out_fh:
        for chunk in iter(lambda: in_fh.read(16 * 1024 * 1024), b""):
            out_fh.write(chunk)
            stats.bytes_written += len(chunk)
            stats.documents += chunk.count(b"\n")
            last_byte = chunk[-1:]
        if last_byte and last_byte != b"\n":
            out_fh.write(b"\n")
            stats.bytes_written += 1
            stats.documents += 1
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--large-qa-dir", type=Path, default=DEFAULT_LARGE_QA)
    parser.add_argument("--supp-dir", type=Path, default=DEFAULT_SUPP)
    parser.add_argument("--reddit-best", type=Path, default=DEFAULT_REDDIT_BEST)
    parser.add_argument("--dna", type=Path, default=DEFAULT_DNA)
    parser.add_argument("--large-qa-mb", type=int, default=850)
    parser.add_argument("--math-mb", type=int, default=900)
    parser.add_argument("--wildchat-mb", type=int, default=220)
    parser.add_argument("--reddit-best-mb", type=int, default=80)
    parser.add_argument("--dna-mb", type=int, default=80)
    parser.add_argument("--audit-samples", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260515)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mix_path = args.out_dir / "mix_full.txt"
    stats: list[SourceStats] = []

    stats.append(copy_base_fast(args.base, mix_path))
    with mix_path.open("a", encoding="utf-8", newline="\n") as out_fh:
        large_cap = args.large_qa_mb * 1_000_000
        large_sources = [
            args.large_qa_dir / "openorca.jsonl",
            args.large_qa_dir / "orca_agent_open_domain.jsonl",
            args.large_qa_dir / "orca_agent_analytical.jsonl",
        ]
        per_large = max(1, large_cap // len(large_sources))
        for src in large_sources:
            if src.is_file():
                stats.append(
                    write_source(
                        out_fh,
                        name=f"large_qa/{src.stem}",
                        kind="qa_jsonl",
                        path=src,
                        docs=iter_qa_jsonl(src, "QA."),
                        cap_bytes=per_large,
                    )
                )

        math_cap = args.math_mb * 1_000_000
        math_sources = [
            args.supp_dir / "openmath_reasoning_cot.jsonl",
            args.supp_dir / "openmath_reasoning_tir.jsonl",
        ]
        per_math = max(1, math_cap // len(math_sources))
        for src in math_sources:
            if src.is_file():
                stats.append(
                    write_source(
                        out_fh,
                        name=f"math/{src.stem}",
                        kind="math_jsonl",
                        path=src,
                        docs=iter_math_jsonl(src),
                        cap_bytes=per_math,
                    )
                )

        wildchat = args.supp_dir / "wildchat_en.jsonl"
        if wildchat.is_file():
            stats.append(
                write_source(
                    out_fh,
                    name="wildchat_en",
                    kind="wildchat_jsonl",
                    path=wildchat,
                    docs=iter_wildchat_jsonl(wildchat),
                    cap_bytes=args.wildchat_mb * 1_000_000,
                )
            )

        if args.reddit_best.is_file():
            stats.append(
                write_source(
                    out_fh,
                    name="reddit_best_answers",
                    kind="qa_jsonl",
                    path=args.reddit_best,
                    docs=iter_qa_jsonl(args.reddit_best, "Reddit-QA."),
                    cap_bytes=args.reddit_best_mb * 1_000_000,
                )
            )

        if args.dna.is_file():
            stats.append(
                write_source(
                    out_fh,
                    name="knowledge_dna_hybrid",
                    kind="dna_text",
                    path=args.dna,
                    docs=iter_dna_records(args.dna),
                    cap_bytes=args.dna_mb * 1_000_000,
                )
            )

    total_bytes = mix_path.stat().st_size
    manifest = {
        "output_file": str(mix_path),
        "bytes_written": total_bytes,
        "documents": sum(s.documents for s in stats),
        "source_count": len(stats),
        "sources": [s.__dict__ for s in stats],
    }
    manifest["audit"] = write_audit_samples(mix_path, args.out_dir, n=args.audit_samples, seed=args.seed)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {mix_path} ({manifest['documents']:,} docs, {total_bytes / 1e9:.2f} GB)")
    print(f"wrote {args.out_dir / 'manifest.json'}")
    print(f"wrote {args.out_dir / 'audit_samples.txt'}")


if __name__ == "__main__":
    main()
