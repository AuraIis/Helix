#!/usr/bin/env python3
"""Generate a pretraining data forensics report.

This is a read-only gate for mixed pretraining corpora. It checks whether a
mix looks healthy before we spend GPU time on it:

- global token/document distribution from the tokenized .idx file
- exact per-source token/doc stats when the mix was tokenized in manifest order
- per-source byte/doc/token estimates from the mix manifest as fallback context
- sampled token lengths per source when the original source files are present
- tail/validation composition estimate from the end of mix_full.txt
- tokenizer marker sanity for prompt/control strings
- readable random samples with spam/HTML/chat-marker flags

The script intentionally avoids mutating data. It writes report.md,
results.json and audit_samples.md into an output directory.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency guard
    np = None

try:
    import sentencepiece as spm
except Exception:  # pragma: no cover - optional runtime dependency guard
    spm = None


HTML_RE = re.compile(r"<\s*/?\s*(html|body|div|script|style|table|iframe|a)\b|&(?:amp|gt|lt|quot|#x?[0-9a-f]+);", re.I)
CHAT_RE = re.compile(r"<\|(?:im_start|im_end|endoftext|user|assistant|system)\|>|_end_of_the_data|</?think>", re.I)
URL_RE = re.compile(r"https?://|www\.", re.I)
ADULT_CASINO_RE = re.compile(r"\b(?:onlyfans|porn|xxx|casino|jackpot|free spins|sportwetten|sexkontakte)\b", re.I)
SHOP_RE = re.compile(r"\b(?:warenkorb|checkout|rabattcode|gutschein|trusted shops|lieferzeit|versandkosten)\b", re.I)
MATH_RE = re.compile(r"\b(?:mathematik|problem:|loesung:|lösung:|beweis|theorem|lemma|integral|gleichung)\b|[=+\-*/^]{4,}", re.I)
QA_RE = re.compile(r"\b(?:frage:|antwort:|question:|answer:|instruction:|response:)\b", re.I)
CODE_RE = re.compile(r"\b(?:def|class|import|return|function|const|let|var|public static|#include)\b|[{};]{4,}", re.I)
DNA_RE = re.compile(r"</?(?:memory|recall)>|<\|end\|>", re.I)
GERMAN_RE = re.compile(r"\b(?:der|die|das|und|ist|nicht|eine|einer|mit|für|ueber|über|werden|wurde)\b|[äöüÄÖÜß]", re.I)
ENGLISH_RE = re.compile(r"\b(?:the|and|that|with|this|from|were|would|should|because|there)\b", re.I)
TOC_RE = re.compile(r"\b(?:inhaltsverzeichnis|table of contents|seite|page)\b|\.{3,}\s*\d{1,5}", re.I)
OCR_RE = re.compile(r"Ã.|�|Å¿|\b[a-zA-ZÄÖÜäöüß](?:\s+[a-zA-ZÄÖÜäöüß]){4,}\b")
WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")


@dataclass
class DocSample:
    text: str
    origin: str


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def human_bytes(value: int | float | None) -> str:
    if value is None:
        return "-"
    value = float(value)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} PB"


def pct(value: int | float, total: int | float) -> float:
    if not total:
        return 0.0
    return round(float(value) * 100.0 / float(total), 3)


def clean_preview(text: str, limit: int = 700) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def token_stats_from_idx(idx_path: Path) -> dict:
    if np is None:
        return {"error": "numpy_not_available"}
    if not idx_path.is_file():
        return {"error": f"missing_idx: {idx_path}"}
    if idx_path.stat().st_size % 16 != 0:
        return {"error": f"bad_idx_size: {idx_path.stat().st_size}"}

    arr = np.memmap(idx_path, dtype=np.int64, mode="r").reshape(-1, 2)
    lengths = arr[:, 1]
    docs = int(lengths.shape[0])
    if docs == 0:
        return {"documents": 0}

    qs = np.quantile(lengths, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    short_100 = int(np.sum(lengths < 100))
    short_200 = int(np.sum(lengths < 200))
    long_4096 = int(np.sum(lengths > 4096))
    long_8192 = int(np.sum(lengths > 8192))
    total = int(np.sum(lengths, dtype=np.int64))
    max_offset = int(arr[-1, 0] + arr[-1, 1])

    return {
        "documents": docs,
        "tokens": total,
        "avg_tokens_per_doc": round(total / docs, 3),
        "min_tokens": int(np.min(lengths)),
        "p01_tokens": round(float(qs[0]), 3),
        "p05_tokens": round(float(qs[1]), 3),
        "p25_tokens": round(float(qs[2]), 3),
        "median_tokens": round(float(qs[3]), 3),
        "p75_tokens": round(float(qs[4]), 3),
        "p95_tokens": round(float(qs[5]), 3),
        "p99_tokens": round(float(qs[6]), 3),
        "max_tokens": int(np.max(lengths)),
        "docs_lt_100": short_100,
        "docs_lt_100_pct": pct(short_100, docs),
        "docs_lt_200": short_200,
        "docs_lt_200_pct": pct(short_200, docs),
        "docs_gt_4096": long_4096,
        "docs_gt_4096_pct": pct(long_4096, docs),
        "docs_gt_8192": long_8192,
        "docs_gt_8192_pct": pct(long_8192, docs),
        "offset_matches_total_tokens": bool(max_offset == total),
    }


def idx_lengths(idx_path: Path):
    if np is None or not idx_path.is_file() or idx_path.stat().st_size % 16 != 0:
        return None
    arr = np.memmap(idx_path, dtype=np.int64, mode="r").reshape(-1, 2)
    return arr[:, 1]


def length_stats(lengths) -> dict:
    if np is None or lengths is None or int(lengths.shape[0]) == 0:
        return {}
    docs = int(lengths.shape[0])
    total = int(np.sum(lengths, dtype=np.int64))
    qs = np.quantile(lengths, [0.05, 0.50, 0.95])
    short_64 = int(np.sum(lengths < 64))
    short_100 = int(np.sum(lengths < 100))
    short_200 = int(np.sum(lengths < 200))
    return {
        "exact_tokens": total,
        "exact_avg_tokens_per_doc": round(total / docs, 3),
        "exact_p05_tokens": round(float(qs[0]), 3),
        "exact_median_tokens": round(float(qs[1]), 3),
        "exact_p95_tokens": round(float(qs[2]), 3),
        "exact_docs_lt_64": short_64,
        "exact_docs_lt_64_pct": pct(short_64, docs),
        "exact_docs_lt_100": short_100,
        "exact_docs_lt_100_pct": pct(short_100, docs),
        "exact_docs_lt_200": short_200,
        "exact_docs_lt_200_pct": pct(short_200, docs),
    }


def classify_domain(text: str) -> tuple[str, list[str]]:
    flags: list[str] = []
    if HTML_RE.search(text):
        flags.append("html")
    if CHAT_RE.search(text):
        flags.append("chat_marker")
    if ADULT_CASINO_RE.search(text):
        flags.append("adult_or_casino")
    if SHOP_RE.search(text):
        flags.append("shop_spam")
    if len(URL_RE.findall(text)) >= 3:
        flags.append("url_dense")
    if TOC_RE.search(text):
        flags.append("toc_or_index")
    if OCR_RE.search(text):
        flags.append("ocr_or_mojibake")
    if len(text) < 120:
        flags.append("too_short")

    if DNA_RE.search(text):
        return "dna", flags
    if MATH_RE.search(text):
        return "math", flags
    if CODE_RE.search(text):
        return "code", flags
    if QA_RE.search(text):
        return "qa_or_instruction", flags
    german_hits = len(GERMAN_RE.findall(text))
    english_hits = len(ENGLISH_RE.findall(text))
    if german_hits > english_hits * 1.25 and german_hits >= 2:
        return "german_prose", flags
    if english_hits >= max(2, german_hits):
        return "english_prose", flags
    return "other", flags


def iter_random_line_samples(path: Path, n: int, seed: int) -> Iterable[DocSample]:
    if not path.is_file() or n <= 0:
        return
    size = path.stat().st_size
    if size <= 0:
        return
    rng = random.Random(seed)
    with path.open("rb") as fh:
        for _ in range(n):
            pos = rng.randrange(0, size)
            fh.seek(pos)
            if pos > 0:
                fh.readline()
            raw = fh.readline()
            if not raw:
                fh.seek(0)
                raw = fh.readline()
            text = raw.decode("utf-8", errors="replace").strip()
            if text:
                yield DocSample(text=text, origin=f"byte:{pos}")


def iter_tail_docs(path: Path, max_docs: int, tail_bytes: int) -> list[str]:
    if not path.is_file() or max_docs <= 0:
        return []
    size = path.stat().st_size
    start = max(0, size - tail_bytes)
    docs: list[str] = []
    with path.open("rb") as fh:
        fh.seek(start)
        if start:
            fh.readline()
        for raw in fh:
            text = raw.decode("utf-8", errors="replace").strip()
            if text:
                docs.append(text)
                if len(docs) >= max_docs:
                    break
    return docs


def normalize_doc(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"</?think>", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def render_jsonl_doc(obj: dict, kind: str) -> str:
    if kind == "wildchat_jsonl":
        turns = obj.get("turns") or []
        parts = []
        for turn in turns[:6]:
            content = normalize_doc(turn.get("content", ""))
            if not content:
                continue
            role = turn.get("role", "")
            label = "Nutzer" if role == "user" else "Assistent"
            parts.append(f"{label}: {content}")
        return "Dialog: " + " ".join(parts) if len(parts) >= 2 else ""
    if kind == "math_jsonl":
        problem = normalize_doc(obj.get("problem", ""))
        solution = normalize_doc(obj.get("solution", ""))
        expected = normalize_doc(obj.get("expected_answer", ""))
        if problem and solution:
            suffix = f" Erwartete Antwort: {expected}" if expected else ""
            return f"Mathematik. Problem: {problem} Loesung: {solution}{suffix}"
        return ""
    question = normalize_doc(obj.get("question", "") or obj.get("instruction", "") or obj.get("prompt", ""))
    answer = normalize_doc(obj.get("answer", "") or obj.get("output", "") or obj.get("response", "") or obj.get("completion", ""))
    system = normalize_doc(obj.get("system", ""))
    if question and answer:
        prefix = f"System: {system} " if system else ""
        return f"{prefix}Frage: {question} Antwort: {answer}"
    text = normalize_doc(obj.get("text", ""))
    return text


def sample_source_docs(path: Path, kind: str, n: int, seed: int) -> list[str]:
    if not path.is_file() or n <= 0:
        return []
    rng = random.Random(seed)
    docs: list[str] = []
    if kind.endswith("jsonl"):
        size = path.stat().st_size
        with path.open("rb") as fh:
            attempts = 0
            while len(docs) < n and attempts < n * 12:
                attempts += 1
                pos = rng.randrange(0, max(size, 1))
                fh.seek(pos)
                if pos > 0:
                    fh.readline()
                raw = fh.readline()
                if not raw:
                    fh.seek(0)
                    raw = fh.readline()
                try:
                    obj = json.loads(raw.decode("utf-8", errors="replace"))
                except json.JSONDecodeError:
                    continue
                doc = render_jsonl_doc(obj, kind)
                if doc:
                    docs.append(doc)
        return docs
    return [s.text for s in iter_random_line_samples(path, n, seed)]


def marker_tokenizer_report(tokenizer: Path | None, markers: list[str]) -> dict:
    if tokenizer is None:
        return {"status": "skipped_no_tokenizer"}
    if spm is None:
        return {"status": "skipped_sentencepiece_not_available"}
    if not tokenizer.is_file():
        return {"status": f"missing_tokenizer: {tokenizer}"}
    sp = spm.SentencePieceProcessor()
    sp.load(str(tokenizer))
    rows = {}
    unk_id = sp.unk_id()
    for marker in markers:
        ids = sp.encode(marker, out_type=int)
        pieces = sp.encode(marker, out_type=str)
        rows[marker] = {
            "piece_count": len(pieces),
            "id_count": len(ids),
            "has_unk": any(i == unk_id for i in ids),
            "pieces": pieces,
        }
    return {"status": "ok", "markers": rows}


def sampled_token_lengths(sp, docs: list[str]) -> dict:
    if not sp or not docs:
        return {}
    lengths = [len(sp.encode(doc, out_type=int)) + 1 for doc in docs if doc]
    if not lengths:
        return {}
    return {
        "sample_docs": len(lengths),
        "sample_avg_tokens_per_doc": round(statistics.mean(lengths), 3),
        "sample_median_tokens_per_doc": round(statistics.median(lengths), 3),
        "sample_min_tokens": min(lengths),
        "sample_max_tokens": max(lengths),
        "sample_docs_lt_100_pct": pct(sum(1 for x in lengths if x < 100), len(lengths)),
    }


def build_source_report(
    mix_manifest: dict,
    tokenized_manifest: dict,
    tokenizer: Path | None,
    samples_per_source: int,
    seed: int,
    idx_path: Path | None = None,
) -> list[dict]:
    total_mix_bytes = mix_manifest.get("bytes_written") or tokenized_manifest.get("bytes_in") or 0
    total_tokens = tokenized_manifest.get("tokens") or 0
    sp = None
    if tokenizer and tokenizer.is_file() and spm is not None:
        sp = spm.SentencePieceProcessor()
        sp.load(str(tokenizer))

    lengths = idx_lengths(idx_path) if idx_path else None
    source_doc_total = sum(int(src.get("documents") or 0) for src in mix_manifest.get("sources", []))
    exact_source_lengths = lengths is not None and int(lengths.shape[0]) == int(source_doc_total)
    cursor = 0
    reports = []
    for idx, src in enumerate(mix_manifest.get("sources", [])):
        bytes_written = int(src.get("bytes_written") or 0)
        docs = int(src.get("documents") or 0)
        approx_tokens = int(total_tokens * (bytes_written / total_mix_bytes)) if total_mix_bytes and total_tokens else None
        row = {
            "name": src.get("name"),
            "kind": src.get("kind"),
            "path": src.get("path"),
            "documents": docs,
            "bytes_written": bytes_written,
            "avg_bytes_per_doc": round(bytes_written / docs, 3) if docs else None,
            "byte_share_pct": pct(bytes_written, total_mix_bytes),
            "approx_tokens_by_byte_share": approx_tokens,
            "approx_avg_tokens_per_doc": round(approx_tokens / docs, 3) if approx_tokens and docs else None,
            "skipped": src.get("skipped", {}),
        }
        if exact_source_lengths:
            row.update(length_stats(lengths[cursor : cursor + docs]))
            cursor += docs
        elif lengths is not None:
            row["exact_token_stats_error"] = (
                f"manifest source docs ({source_doc_total}) do not match idx docs ({int(lengths.shape[0])})"
            )
        path = Path(str(src.get("path", "")))
        sample_docs = sample_source_docs(path, str(src.get("kind", "")), samples_per_source, seed + idx)
        row.update(sampled_token_lengths(sp, sample_docs))
        domain_counts = Counter()
        flag_counts = Counter()
        for doc in sample_docs[:samples_per_source]:
            domain, flags = classify_domain(doc)
            domain_counts[domain] += 1
            flag_counts.update(flags)
        if sample_docs:
            row["sample_domain_counts"] = dict(domain_counts)
            row["sample_flag_counts"] = dict(flag_counts)
        reports.append(row)
    return reports


def analyze_tail(docs: list[str]) -> dict:
    domain_counts = Counter()
    flag_counts = Counter()
    lengths = []
    for doc in docs:
        domain, flags = classify_domain(doc)
        domain_counts[domain] += 1
        flag_counts.update(flags)
        lengths.append(len(doc))
    return {
        "sample_docs": len(docs),
        "domain_counts": dict(domain_counts),
        "flag_counts": dict(flag_counts),
        "avg_chars_per_doc": round(statistics.mean(lengths), 3) if lengths else 0,
        "median_chars_per_doc": round(statistics.median(lengths), 3) if lengths else 0,
    }


def write_audit_samples(samples: list[DocSample], out_path: Path) -> list[dict]:
    rows = []
    lines = ["# Audit Samples", ""]
    for i, sample in enumerate(samples, start=1):
        domain, flags = classify_domain(sample.text)
        row = {
            "n": i,
            "origin": sample.origin,
            "domain": domain,
            "flags": flags,
            "chars": len(sample.text),
            "preview": clean_preview(sample.text, 900),
        }
        rows.append(row)
        lines.extend(
            [
                f"## Sample {i}",
                "",
                f"- origin: `{sample.origin}`",
                f"- domain: `{domain}`",
                f"- flags: `{', '.join(flags) if flags else 'none'}`",
                "",
                "```text",
                clean_preview(sample.text, 1500),
                "```",
                "",
            ]
        )
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return rows


def markdown_report(results: dict) -> str:
    token_stats = results.get("token_stats", {})
    lines = [
        "# Pretraining Forensics Report",
        "",
        f"- mix: `{results['mix_dir']}`",
        f"- tokenized: `{results['tokenized_dir']}`",
        f"- generated_at: `{results['generated_at']}`",
        "",
        "## Token/Document Health",
        "",
    ]
    if "error" in token_stats:
        lines.append(f"- error: `{token_stats['error']}`")
    else:
        lines.extend(
            [
                f"- documents: `{token_stats.get('documents'):,}`",
                f"- tokens: `{token_stats.get('tokens'):,}`",
                f"- avg tokens/doc: `{token_stats.get('avg_tokens_per_doc')}`",
                f"- median tokens/doc: `{token_stats.get('median_tokens')}`",
                f"- p05/p95 tokens/doc: `{token_stats.get('p05_tokens')}` / `{token_stats.get('p95_tokens')}`",
                f"- docs <100 tokens: `{token_stats.get('docs_lt_100'):,}` (`{token_stats.get('docs_lt_100_pct')}%`)",
                f"- docs >4096 tokens: `{token_stats.get('docs_gt_4096'):,}` (`{token_stats.get('docs_gt_4096_pct')}%`)",
                f"- idx offsets match token total: `{token_stats.get('offset_matches_total_tokens')}`",
            ]
        )
    lines.extend(["", "## Sources", ""])
    lines.append("| Source | Docs | GB | Share | Exact tok/doc | <64 tok | <100 tok | Sample flags |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for src in results.get("sources", []):
        flags = ", ".join(f"{k}:{v}" for k, v in (src.get("sample_flag_counts") or {}).items()) or "-"
        lines.append(
            "| {name} | {docs:,} | {gb:.2f} | {share:.2f}% | {tokdoc} | {lt64}% | {lt100}% | {flags} |".format(
                name=src.get("name"),
                docs=src.get("documents") or 0,
                gb=(src.get("bytes_written") or 0) / (1024**3),
                share=src.get("byte_share_pct") or 0,
                tokdoc=src.get("exact_avg_tokens_per_doc") or src.get("approx_avg_tokens_per_doc") or "-",
                lt64=src.get("exact_docs_lt_64_pct", "-"),
                lt100=src.get("exact_docs_lt_100_pct", "-"),
                flags=flags,
            )
        )
    lines.extend(["", "## Tail/Val Composition Estimate", ""])
    tail = results.get("tail_analysis", {})
    lines.append(f"- sampled tail docs: `{tail.get('sample_docs', 0)}`")
    lines.append(f"- domain counts: `{tail.get('domain_counts', {})}`")
    lines.append(f"- flag counts: `{tail.get('flag_counts', {})}`")
    lines.extend(["", "## Tokenizer Markers", ""])
    marker = results.get("marker_tokenizer", {})
    lines.append(f"- status: `{marker.get('status')}`")
    for text, row in (marker.get("markers") or {}).items():
        pieces = " ".join(row.get("pieces", []))
        lines.append(f"- `{text.replace(chr(10), '<NL>')}` -> {row.get('piece_count')} pieces, unk={row.get('has_unk')}: `{pieces}`")
    lines.extend(["", "## Gate Notes", ""])
    notes = results.get("gate_notes", [])
    for note in notes:
        lines.append(f"- {note}")
    lines.extend(["", "See `audit_samples.md` for readable examples and `results.json` for full details.", ""])
    return "\n".join(lines)


def gate_notes(results: dict) -> list[str]:
    notes: list[str] = []
    stats = results.get("token_stats", {})
    if stats.get("docs_lt_100_pct", 0) > 10:
        notes.append("WARN: More than 10% of docs are under 100 tokens; check doc boundaries and short fragments.")
    elif stats:
        notes.append("OK: Short-document share is not obviously broken.")
    avg = stats.get("avg_tokens_per_doc")
    if avg and avg < 150:
        notes.append("WARN: Average tokens/doc is very low; likely too many fragments.")
    elif avg and avg > 2500:
        notes.append("WARN: Average tokens/doc is very high; check packed mega-docs or missing line breaks.")
    elif avg:
        notes.append("OK: Average tokens/doc is in a plausible range for one-line documents.")
    tail = results.get("tail_analysis", {})
    domains = tail.get("domain_counts") or {}
    if domains:
        top_domain, top_count = max(domains.items(), key=lambda kv: kv[1])
        share = pct(top_count, sum(domains.values()))
        if share > 70:
            notes.append(f"WARN: Tail/val estimate is dominated by {top_domain} ({share}%).")
        else:
            notes.append("OK: Tail/val estimate is not dominated by a single simple domain.")
    flags = tail.get("flag_counts") or {}
    bad_tail = sum(flags.get(k, 0) for k in ("html", "chat_marker", "adult_or_casino", "shop_spam", "url_dense"))
    if bad_tail:
        notes.append(f"WARN: Tail sample still contains {bad_tail} hard-noise flags.")
    else:
        notes.append("OK: Tail sample has no hard HTML/chat/adult/shop/url-spam flags.")
    marker = results.get("marker_tokenizer", {}).get("markers") or {}
    for special in ("<memory>", "</memory>", "<recall>", "</recall>", "<|end|>"):
        row = marker.get(special)
        if row and row.get("piece_count") != 1:
            notes.append(f"WARN: Intended special marker {special} is not single-token ({row.get('piece_count')} pieces).")
    task = marker.get("### Aufgabe:\n")
    if task:
        if task.get("has_unk"):
            notes.append("WARN: Alpaca marker contains <unk>; this is dangerous.")
        elif task.get("piece_count", 99) <= 6:
            notes.append("OK: Alpaca text marker is compact enough; single-token is not required for this marker.")
        else:
            notes.append("WARN: Alpaca marker splits into many pieces; consider a registered special token in tokenizer-v2.")
    sources = results.get("sources") or []
    short_sources = [
        (src.get("name"), src.get("exact_docs_lt_100_pct", 0), src.get("exact_docs_lt_100", 0))
        for src in sources
        if src.get("exact_docs_lt_100_pct", 0) > 10
    ]
    if short_sources:
        short_sources.sort(key=lambda item: (item[2] or 0), reverse=True)
        pretty = ", ".join(f"{name}={share}%" for name, share, _ in short_sources[:5])
        notes.append(f"WARN: Short-doc pressure is dominated by: {pretty}.")
    return notes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix-dir", required=True, type=Path, help="Directory with mix_full.txt and manifest.json.")
    parser.add_argument("--tokenized-dir", required=True, type=Path, help="Directory with german.idx and german.bin.manifest.json.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--sample-docs", type=int, default=30)
    parser.add_argument("--samples-per-source", type=int, default=80)
    parser.add_argument("--tail-docs", type=int, default=5000)
    parser.add_argument("--tail-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import datetime as _dt

    mix_manifest_path = args.mix_dir / "manifest.json"
    mix_path = args.mix_dir / "mix_full.txt"
    tokenized_manifest_path = args.tokenized_dir / "german.bin.manifest.json"
    idx_path = args.tokenized_dir / "german.idx"

    if not mix_manifest_path.is_file():
        raise SystemExit(f"missing mix manifest: {mix_manifest_path}")
    if not tokenized_manifest_path.is_file():
        raise SystemExit(f"missing tokenized manifest: {tokenized_manifest_path}")
    if not mix_path.is_file():
        raise SystemExit(f"missing mix text: {mix_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mix_manifest = load_json(mix_manifest_path)
    tokenized_manifest = load_json(tokenized_manifest_path)

    samples = list(iter_random_line_samples(mix_path, args.sample_docs, args.seed))
    audit_rows = write_audit_samples(samples, args.output_dir / "audit_samples.md")
    tail_docs = iter_tail_docs(mix_path, args.tail_docs, args.tail_bytes)

    markers = [
        "### Aufgabe:\n",
        "### Antwort:\n",
        "<memory>",
        "</memory>",
        "<recall>",
        "</recall>",
        "<|end|>",
    ]

    results = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "mix_dir": str(args.mix_dir),
        "tokenized_dir": str(args.tokenized_dir),
        "mix_manifest": {
            "documents": mix_manifest.get("documents"),
            "bytes_written": mix_manifest.get("bytes_written"),
            "source_count": mix_manifest.get("source_count"),
        },
        "tokenized_manifest": tokenized_manifest,
        "token_stats": token_stats_from_idx(idx_path),
        "sources": build_source_report(
            mix_manifest,
            tokenized_manifest,
            args.tokenizer,
            args.samples_per_source,
            args.seed,
            idx_path=idx_path,
        ),
        "tail_analysis": analyze_tail(tail_docs),
        "marker_tokenizer": marker_tokenizer_report(args.tokenizer, markers),
        "audit_samples": audit_rows,
    }
    results["gate_notes"] = gate_notes(results)

    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "report.md").write_text(markdown_report(results), encoding="utf-8")
    print(f"wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
