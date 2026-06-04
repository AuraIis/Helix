#!/usr/bin/env python3
"""Build the clean-v3.2 + strict booster v5 pretraining mix.

Compared with v4 this defaults to clean-v3.2, rejects URL/shop/adult/casino
fragments more aggressively, can include strict Reddit threaded QA, and leaves
Knowledge-DNA disabled by default until the ablation is clearly positive.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

try:
    import sentencepiece as spm
except Exception:  # pragma: no cover
    spm = None


DEFAULT_BASE = Path("/workspace/v2data/data/training/pretrain_clean_v32/mix_full.txt")
DEFAULT_OUT = Path("/workspace/v2data/data/training/pretrain_mix_v5_boosted")
DEFAULT_DNA = Path("/workspace/v2data/data/eval/knowledge_dna_v2_expanded_750/hybrid_corpus.txt")
DEFAULT_LARGE_QA = Path("/disk5v2data/data/large_qa_sources_v1")
DEFAULT_SUPP = Path("/disk5v2data/data/supplemental_large_sources_v1")
DEFAULT_REDDIT_BEST = Path("/disk5v2data/data/reddit_best_answers_v1/reddit_best_answers.jsonl")
DEFAULT_REDDIT_THREADED = Path("/disk5v2data/data/reddit_threaded_qa_v2_strict/reddit_threaded_qa.jsonl")
DEFAULT_TOKENIZER = Path("/workspace/v2data/tokenizer/helix_v2_tokenizer.model")

CHAT_MARKER_RE = re.compile(r"<\|(?:im_start|im_end|endoftext|user|assistant|system)\|>|_end_of_the_data", re.I)
HTML_RE = re.compile(r"<\s*/?\s*(html|body|div|script|style|table|iframe)\b|&(?:amp|gt|lt|quot|#x200b);", re.I)
URL_RE = re.compile(r"https?://|www\.", re.I)
BAD_RE = re.compile(
    r"\[(?:deleted|removed)\]|discord\.gg|onlyfans|free karma|subscribe to|promo code|"
    r"online casino|casino bonus|free spins|jackpot|sportwetten|sexkontakte|"
    r"erotik\s+(?:chat|kontakte|shop)|erotik,\s*wohlbefinden|porn|xxx|"
    r"warenkorb|checkout|rabattcode|gutschein|trusted shops|"
    r"all rights reserved|cookie policy|privacy policy",
    re.I,
)
WIKI_TALK_RE = re.compile(
    r"\b(?:nicht signierter beitrag|ce[st]\)|wikipedia:|qs-baustein|"
    r"redaktion[_ ]|diskussion:|benutzer:|--\s*[A-Za-zÄÖÜäöüß0-9_-])",
    re.I,
)
TABLE_RE = re.compile(r"\|\s*[-:]+\s*\||\{\||\|\}|^\s*\|", re.I)
INDEX_RE = re.compile(r"\b(?:kategorie:|liste der|personen nach|artikel des tages|portal:)\b|\.{3,}\s*\d{1,5}", re.I)
LIST_RE = re.compile(r"(^|\s)(?:[-*•]|\d{1,3}[.)])\s+|[|]{2,}|\t")
TERMINAL_RE = re.compile(r"[.!?:\)\]\"”]$")
THINK_RE = re.compile(r"</?think>", re.I)
REDDIT_META_RE = re.compile(
    r"\b(upvote|downvote|subreddit|moderator|automod|this post has been removed|"
    r"edit:\s*thanks|thanks for the gold|throwaway account)\b",
    re.I,
)


@dataclass
class SourceStats:
    name: str
    kind: str
    path: str
    cap_bytes: int | None
    documents: int = 0
    bytes_written: int = 0
    observed: dict[str, int] = field(default_factory=dict)
    skipped: dict[str, int] = field(default_factory=dict)


def inc(stats: SourceStats, reason: str) -> None:
    stats.skipped[reason] = stats.skipped.get(reason, 0) + 1


def observe(stats: SourceStats | None, signal: str) -> None:
    if stats is None:
        return
    stats.observed[signal] = stats.observed.get(signal, 0) + 1


def source_manifest(stats: SourceStats) -> dict:
    item = stats.__dict__.copy()
    item["skip_rates"] = {
        reason: {
            "skipped": skipped,
            "observed": stats.observed.get(reason, 0),
            "pct_of_observed": round(skipped * 100.0 / stats.observed[reason], 3)
            if stats.observed.get(reason)
            else None,
        }
        for reason, skipped in sorted(stats.skipped.items())
        if not reason.startswith("source_docs/")
    }
    return item


def normalize_doc(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = THINK_RE.sub("", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_count(sp, text: str) -> int:
    if sp is None:
        return 0
    return len(sp.encode(text, out_type=int)) + 1


def looks_like_broken_sentence_fragment(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped[:1] and stripped[:1].islower() and not TERMINAL_RE.search(stripped))


def should_skip(
    text: str,
    *,
    max_urls: int,
    min_tokens: int = 0,
    sp=None,
    drop_wiki_talk: bool = False,
    drop_table_index: bool = False,
    drop_short_list_like_under_tokens: int = 0,
    drop_broken_sentence_under_tokens: int = 0,
    stats: SourceStats | None = None,
) -> str | None:
    if len(text) < 90:
        observe(stats, "too_short")
        return "too_short"
    if CHAT_MARKER_RE.search(text):
        observe(stats, "chat_marker")
        return "chat_marker"
    if HTML_RE.search(text):
        observe(stats, "html_or_entity")
        return "html_or_entity"
    if BAD_RE.search(text):
        observe(stats, "bad_marker")
        return "bad_marker"
    url_count = len(URL_RE.findall(text))
    if url_count:
        observe(stats, "has_url")
    if url_count > max_urls:
        observe(stats, "url_dense")
        return "url_dense"
    wiki_talk = bool(WIKI_TALK_RE.search(text))
    if wiki_talk:
        observe(stats, "wiki_talk")
        if drop_wiki_talk:
            return "wiki_talk"
    table_or_index = bool(TABLE_RE.search(text) or INDEX_RE.search(text))
    if table_or_index:
        observe(stats, "table_or_index")
        if drop_table_index:
            return "table_or_index"
    alpha = sum(1 for c in text if c.isalpha()) / max(len(text), 1)
    if alpha < 0.32:
        observe(stats, "low_alpha")
        return "low_alpha"
    list_like = bool(LIST_RE.search(text))
    if list_like:
        observe(stats, "list_like")
    broken_sentence_fragment = looks_like_broken_sentence_fragment(text)
    if broken_sentence_fragment:
        observe(stats, "broken_sentence_fragment")
    n_tokens = 0
    if (min_tokens > 0 or drop_short_list_like_under_tokens > 0 or drop_broken_sentence_under_tokens > 0) and sp is None:
        observe(stats, "missing_tokenizer_for_min_tokens")
        return "missing_tokenizer_for_min_tokens"
    if sp is not None and (min_tokens > 0 or drop_short_list_like_under_tokens > 0 or drop_broken_sentence_under_tokens > 0):
        n_tokens = token_count(sp, text)
    if drop_short_list_like_under_tokens > 0 and n_tokens < drop_short_list_like_under_tokens and list_like:
        observe(stats, "short_list_like")
        return "short_list_like"
    if drop_broken_sentence_under_tokens > 0 and n_tokens < drop_broken_sentence_under_tokens and broken_sentence_fragment:
        observe(stats, "short_broken_sentence")
        return "short_broken_sentence"
    if min_tokens > 0:
        if n_tokens < min_tokens:
            observe(stats, "too_few_tokens")
            return "too_few_tokens"
    return None


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


def iter_qa_jsonl(path: Path, prefix: str, *, reddit: bool = False) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            question = normalize_doc(obj.get("question", ""))
            answer = normalize_doc(obj.get("answer", ""))
            system = normalize_doc(obj.get("system", ""))
            if reddit and (REDDIT_META_RE.search(question) or REDDIT_META_RE.search(answer)):
                continue
            if not question or not answer:
                continue
            if system:
                yield f"{prefix} System: {system} Frage: {question} Antwort: {answer}"
            else:
                yield f"{prefix} Frage: {question} Antwort: {answer}"


def iter_wildchat_jsonl(path: Path) -> Iterable[str]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
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
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            problem = normalize_doc(obj.get("problem", ""))
            solution = normalize_doc(obj.get("solution", ""))
            expected = normalize_doc(obj.get("expected_answer", ""))
            if not problem or not solution:
                continue
            suffix = f" Erwartete Antwort: {expected}" if expected else ""
            yield f"Mathematik. Problem: {problem} Loesung: {solution}{suffix}"


def write_source(
    out_fh,
    *,
    name: str,
    kind: str,
    path: Path,
    docs: Iterable[str],
    cap_bytes: int | None,
    max_urls: int,
    min_tokens: int = 0,
    sp=None,
    drop_wiki_talk: bool = False,
    drop_table_index: bool = False,
    drop_short_list_like_under_tokens: int = 0,
    drop_broken_sentence_under_tokens: int = 0,
) -> SourceStats:
    stats = SourceStats(name=name, kind=kind, path=str(path), cap_bytes=cap_bytes)
    for doc in docs:
        doc = normalize_doc(doc)
        reason = should_skip(
            doc,
            max_urls=max_urls,
            min_tokens=min_tokens,
            sp=sp,
            drop_wiki_talk=drop_wiki_talk,
            drop_table_index=drop_table_index,
            drop_short_list_like_under_tokens=drop_short_list_like_under_tokens,
            drop_broken_sentence_under_tokens=drop_broken_sentence_under_tokens,
            stats=stats,
        )
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


def write_balanced_val_tail(
    out_fh,
    *,
    base: Path,
    large_qa_dir: Path,
    supp_dir: Path,
    reddit_best: Path,
    reddit_threaded: Path,
    dna: Path,
    cap_bytes: int,
    max_urls: int,
    include_dna: bool,
    min_tokens: int = 0,
    sp=None,
    drop_wiki_talk: bool = False,
    drop_table_index: bool = False,
    drop_short_list_like_under_tokens: int = 0,
    drop_broken_sentence_under_tokens: int = 0,
) -> SourceStats:
    """Append an interleaved tail so validation is not dominated by the last source.

    The trainer currently reserves the last N binary bytes as validation. If the
    mix is built by copying the base and appending boosters, validation becomes
    a narrow slice of whatever source was appended last. This tail intentionally
    writes a small, mixed corpus at the end, large enough to fully cover the
    configured validation split after tokenization.
    """

    stats = SourceStats(
        name="balanced_validation_tail",
        kind="interleaved_text",
        path="multiple",
        cap_bytes=cap_bytes,
    )

    streams: list[tuple[str, Iterable[str]]] = []
    if base.is_file():
        streams.append(("base", (line.strip() for line in base.open("r", encoding="utf-8", errors="replace") if line.strip())))

    large_sources = [
        large_qa_dir / "openorca.jsonl",
        large_qa_dir / "orca_agent_open_domain.jsonl",
        large_qa_dir / "orca_agent_analytical.jsonl",
    ]
    for src in large_sources:
        if src.is_file():
            streams.append((f"large_qa/{src.stem}", iter_qa_jsonl(src, "QA.")))

    math_sources = [
        supp_dir / "openmath_reasoning_cot.jsonl",
        supp_dir / "openmath_reasoning_tir.jsonl",
    ]
    for src in math_sources:
        if src.is_file():
            streams.append((f"math/{src.stem}", iter_math_jsonl(src)))

    wildchat = supp_dir / "wildchat_en.jsonl"
    if wildchat.is_file():
        streams.append(("wildchat_en", iter_wildchat_jsonl(wildchat)))

    if reddit_best.is_file():
        streams.append(("reddit_best_answers", iter_qa_jsonl(reddit_best, "Reddit-QA.", reddit=True)))
    if reddit_threaded.is_file():
        streams.append(("reddit_threaded_qa_strict", iter_qa_jsonl(reddit_threaded, "Reddit-Thread-QA.", reddit=True)))
    if include_dna and dna.is_file():
        streams.append(("knowledge_dna_hybrid", iter_dna_records(dna)))

    active = [(name, iter(docs)) for name, docs in streams]
    per_source_docs: dict[str, int] = {}
    while active and stats.bytes_written < cap_bytes:
        next_active = []
        for name, docs in active:
            try:
                doc = next(docs)
            except StopIteration:
                continue
            doc = normalize_doc(doc)
            reason = should_skip(
                doc,
                max_urls=max_urls,
                min_tokens=min_tokens,
                sp=sp,
                drop_wiki_talk=drop_wiki_talk,
                drop_table_index=drop_table_index,
                drop_short_list_like_under_tokens=drop_short_list_like_under_tokens,
                drop_broken_sentence_under_tokens=drop_broken_sentence_under_tokens,
                stats=stats,
            )
            if reason:
                inc(stats, reason)
                next_active.append((name, docs))
                continue
            line = doc + "\n"
            line_bytes = len(line.encode("utf-8"))
            if stats.bytes_written + line_bytes > cap_bytes:
                next_active.append((name, docs))
                break
            out_fh.write(line)
            stats.documents += 1
            stats.bytes_written += line_bytes
            per_source_docs[name] = per_source_docs.get(name, 0) + 1
            next_active.append((name, docs))
            if stats.bytes_written >= cap_bytes:
                break
        active = next_active

    for name, count in per_source_docs.items():
        stats.skipped[f"source_docs/{name}"] = count
    return stats


def copy_base_fast(src: Path, dst: Path) -> SourceStats:
    stats = SourceStats(name="clean_v32_base", kind="text_lines_fast_copy", path=str(src), cap_bytes=None)
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


def copy_base_filtered(
    src: Path,
    dst: Path,
    *,
    min_tokens: int,
    sp,
    drop_wiki_talk: bool,
    drop_table_index: bool,
    drop_short_list_like_under_tokens: int,
    drop_broken_sentence_under_tokens: int,
) -> SourceStats:
    stats = SourceStats(name="clean_v32_base", kind="text_lines_filtered_copy", path=str(src), cap_bytes=None)
    with src.open("r", encoding="utf-8", errors="replace") as in_fh, dst.open("w", encoding="utf-8", newline="\n") as out_fh:
        for line in in_fh:
            text = normalize_doc(line)
            if not text:
                inc(stats, "empty")
                continue
            reason = should_skip(
                text,
                max_urls=1,
                min_tokens=min_tokens,
                sp=sp,
                drop_wiki_talk=drop_wiki_talk,
                drop_table_index=drop_table_index,
                drop_short_list_like_under_tokens=drop_short_list_like_under_tokens,
                drop_broken_sentence_under_tokens=drop_broken_sentence_under_tokens,
                stats=stats,
            )
            if reason:
                inc(stats, reason)
                continue
            out_line = text + "\n"
            out_fh.write(out_line)
            stats.documents += 1
            stats.bytes_written += len(out_line.encode("utf-8"))
    return stats


def write_audit_samples(mix_path: Path, out_dir: Path, *, n: int, seed: int, max_urls: int) -> dict:
    rng = random.Random(seed)
    size = mix_path.stat().st_size
    samples = []
    flags = {
        "chat_marker": 0,
        "html_or_entity": 0,
        "bad_marker": 0,
        "url_dense": 0,
        "too_short": 0,
        "low_alpha": 0,
    }
    with mix_path.open("rb") as fh:
        for _ in range(n):
            fh.seek(rng.randrange(0, max(size, 1)))
            fh.readline()
            raw = fh.readline() or b""
            text = raw.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            reason = should_skip(text, max_urls=max_urls)
            if reason in flags:
                flags[reason] += 1
            samples.append(text[:800])
    (out_dir / "audit_samples.txt").write_text(
        "\n\n--- SAMPLE ---\n\n".join(samples),
        encoding="utf-8",
    )
    return {"sample_count": len(samples), "flag_counts": flags}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--large-qa-dir", type=Path, default=DEFAULT_LARGE_QA)
    parser.add_argument("--supp-dir", type=Path, default=DEFAULT_SUPP)
    parser.add_argument("--reddit-best", type=Path, default=DEFAULT_REDDIT_BEST)
    parser.add_argument("--reddit-threaded", type=Path, default=DEFAULT_REDDIT_THREADED)
    parser.add_argument("--dna", type=Path, default=DEFAULT_DNA)
    parser.add_argument("--large-qa-mb", type=int, default=900)
    parser.add_argument("--math-mb", type=int, default=900)
    parser.add_argument("--wildchat-mb", type=int, default=160)
    parser.add_argument("--reddit-best-mb", type=int, default=70)
    parser.add_argument("--reddit-threaded-mb", type=int, default=70)
    parser.add_argument("--dna-mb", type=int, default=0)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--base-min-tokens",
        type=int,
        default=32,
        help=(
            "Minimum token count for base-mix lines. This is a conservative "
            "fragment gate for future mixes. Set 0 to keep the old fast-copy path."
        ),
    )
    parser.add_argument(
        "--booster-min-tokens",
        type=int,
        default=0,
        help="Optional minimum token count for appended booster records.",
    )
    parser.add_argument("--drop-wiki-talk", action="store_true", default=True)
    parser.add_argument("--keep-wiki-talk", action="store_false", dest="drop_wiki_talk")
    parser.add_argument("--drop-table-index", action="store_true", default=True)
    parser.add_argument("--keep-table-index", action="store_false", dest="drop_table_index")
    parser.add_argument(
        "--drop-short-list-like-under-tokens",
        type=int,
        default=100,
        help="Drop list-like base/booster docs below this token length. Set 0 to disable.",
    )
    parser.add_argument(
        "--drop-broken-sentence-under-tokens",
        type=int,
        default=100,
        help=(
            "Drop short docs that both start mid-sentence and lack a terminal "
            "sentence boundary. Set 0 to disable."
        ),
    )
    parser.add_argument(
        "--balanced-val-tail-mb",
        type=int,
        default=320,
        help=(
            "Append a mixed validation tail. The trainer holds out the final "
            "binary bytes, so this prevents validation from being just the last "
            "booster source. Use 0 to disable."
        ),
    )
    parser.add_argument("--max-urls", type=int, default=1)
    parser.add_argument("--audit-samples", type=int, default=240)
    parser.add_argument("--seed", type=int, default=20260517)
    args = parser.parse_args()

    if not args.base.is_file():
        raise SystemExit(f"missing base mix: {args.base}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mix_path = args.out_dir / "mix_full.txt"
    stats: list[SourceStats] = []
    sp = None
    if args.base_min_tokens > 0 or args.booster_min_tokens > 0 or args.drop_broken_sentence_under_tokens > 0:
        if spm is None:
            raise SystemExit("sentencepiece is required for --*-min-tokens")
        if not args.tokenizer.is_file():
            raise SystemExit(f"missing tokenizer: {args.tokenizer}")
        sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))

    if (
        args.base_min_tokens > 0
        or args.drop_wiki_talk
        or args.drop_table_index
        or args.drop_short_list_like_under_tokens > 0
        or args.drop_broken_sentence_under_tokens > 0
    ):
        if sp is None and (
            args.base_min_tokens > 0
            or args.drop_short_list_like_under_tokens > 0
            or args.drop_broken_sentence_under_tokens > 0
        ):
            if spm is None:
                raise SystemExit("sentencepiece is required for base filtering")
            if not args.tokenizer.is_file():
                raise SystemExit(f"missing tokenizer: {args.tokenizer}")
            sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
        stats.append(
            copy_base_filtered(
                args.base,
                mix_path,
                min_tokens=args.base_min_tokens,
                sp=sp,
                drop_wiki_talk=args.drop_wiki_talk,
                drop_table_index=args.drop_table_index,
                drop_short_list_like_under_tokens=args.drop_short_list_like_under_tokens,
                drop_broken_sentence_under_tokens=args.drop_broken_sentence_under_tokens,
            )
        )
    else:
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
                        max_urls=args.max_urls,
                        min_tokens=args.booster_min_tokens,
                        sp=sp,
                        drop_wiki_talk=args.drop_wiki_talk,
                        drop_table_index=args.drop_table_index,
                        drop_short_list_like_under_tokens=args.drop_short_list_like_under_tokens,
                        drop_broken_sentence_under_tokens=args.drop_broken_sentence_under_tokens,
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
                        max_urls=args.max_urls,
                        min_tokens=args.booster_min_tokens,
                        sp=sp,
                        drop_wiki_talk=args.drop_wiki_talk,
                        drop_table_index=args.drop_table_index,
                        drop_short_list_like_under_tokens=args.drop_short_list_like_under_tokens,
                        drop_broken_sentence_under_tokens=args.drop_broken_sentence_under_tokens,
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
                    max_urls=args.max_urls,
                    min_tokens=args.booster_min_tokens,
                    sp=sp,
                    drop_wiki_talk=args.drop_wiki_talk,
                    drop_table_index=args.drop_table_index,
                    drop_short_list_like_under_tokens=args.drop_short_list_like_under_tokens,
                    drop_broken_sentence_under_tokens=args.drop_broken_sentence_under_tokens,
                )
            )

        if args.reddit_best.is_file():
            stats.append(
                write_source(
                    out_fh,
                    name="reddit_best_answers",
                    kind="qa_jsonl",
                    path=args.reddit_best,
                    docs=iter_qa_jsonl(args.reddit_best, "Reddit-QA.", reddit=True),
                    cap_bytes=args.reddit_best_mb * 1_000_000,
                    max_urls=args.max_urls,
                    min_tokens=args.booster_min_tokens,
                    sp=sp,
                    drop_wiki_talk=args.drop_wiki_talk,
                    drop_table_index=args.drop_table_index,
                    drop_short_list_like_under_tokens=args.drop_short_list_like_under_tokens,
                    drop_broken_sentence_under_tokens=args.drop_broken_sentence_under_tokens,
                )
            )

        if args.reddit_threaded.is_file():
            stats.append(
                write_source(
                    out_fh,
                    name="reddit_threaded_qa_strict",
                    kind="qa_jsonl",
                    path=args.reddit_threaded,
                    docs=iter_qa_jsonl(args.reddit_threaded, "Reddit-Thread-QA.", reddit=True),
                    cap_bytes=args.reddit_threaded_mb * 1_000_000,
                    max_urls=args.max_urls,
                    min_tokens=args.booster_min_tokens,
                    sp=sp,
                    drop_wiki_talk=args.drop_wiki_talk,
                    drop_table_index=args.drop_table_index,
                    drop_short_list_like_under_tokens=args.drop_short_list_like_under_tokens,
                    drop_broken_sentence_under_tokens=args.drop_broken_sentence_under_tokens,
                )
            )

        if args.dna_mb > 0 and args.dna.is_file():
            stats.append(
                write_source(
                    out_fh,
                    name="knowledge_dna_hybrid",
                    kind="dna_text",
                    path=args.dna,
                    docs=iter_dna_records(args.dna),
                    cap_bytes=args.dna_mb * 1_000_000,
                    max_urls=args.max_urls,
                    min_tokens=args.booster_min_tokens,
                    sp=sp,
                    drop_wiki_talk=args.drop_wiki_talk,
                    drop_table_index=args.drop_table_index,
                    drop_short_list_like_under_tokens=args.drop_short_list_like_under_tokens,
                    drop_broken_sentence_under_tokens=args.drop_broken_sentence_under_tokens,
                )
            )

        if args.balanced_val_tail_mb > 0:
            stats.append(
                write_balanced_val_tail(
                    out_fh,
                    base=args.base,
                    large_qa_dir=args.large_qa_dir,
                    supp_dir=args.supp_dir,
                    reddit_best=args.reddit_best,
                    reddit_threaded=args.reddit_threaded,
                    dna=args.dna,
                    cap_bytes=args.balanced_val_tail_mb * 1_000_000,
                    max_urls=args.max_urls,
                    include_dna=args.dna_mb > 0,
                    min_tokens=args.booster_min_tokens,
                    sp=sp,
                    drop_wiki_talk=args.drop_wiki_talk,
                    drop_table_index=args.drop_table_index,
                    drop_short_list_like_under_tokens=args.drop_short_list_like_under_tokens,
                    drop_broken_sentence_under_tokens=args.drop_broken_sentence_under_tokens,
                )
            )

    total_bytes = mix_path.stat().st_size
    manifest = {
        "output_file": str(mix_path),
        "bytes_written": total_bytes,
        "documents": sum(s.documents for s in stats),
        "source_count": len(stats),
        "sources": [source_manifest(s) for s in stats],
        "dna_enabled": args.dna_mb > 0,
        "balanced_val_tail_mb": args.balanced_val_tail_mb,
        "base_min_tokens": args.base_min_tokens,
        "booster_min_tokens": args.booster_min_tokens,
        "drop_wiki_talk": args.drop_wiki_talk,
        "drop_table_index": args.drop_table_index,
        "drop_short_list_like_under_tokens": args.drop_short_list_like_under_tokens,
        "drop_broken_sentence_under_tokens": args.drop_broken_sentence_under_tokens,
    }
    manifest["audit"] = write_audit_samples(mix_path, args.out_dir, n=args.audit_samples, seed=args.seed, max_urls=args.max_urls)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {mix_path} ({manifest['documents']:,} docs, {total_bytes / 1e9:.2f} GB)")
    print(f"wrote {args.out_dir / 'manifest.json'}")
    print(f"wrote {args.out_dir / 'audit_samples.txt'}")


if __name__ == "__main__":
    main()
