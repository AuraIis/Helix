#!/usr/bin/env python3
"""Forensics gate for cleaned text sources before building a pretraining mix.

This runs before tokenization and before the v5 mix build. It answers one
question cheaply: are the cleaned source files still dominated by tiny
fragments or obvious web/OCR/shop noise?

It uses the source manifests for exact doc/byte counts and deterministic
random line samples for tokenizer-length and domain/noise estimates. A full
token scan is intentionally optional because it is nearly as expensive as
tokenizing the corpus.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from collections import Counter
from pathlib import Path

try:
    import sentencepiece as spm
except Exception:  # pragma: no cover
    spm = None


HTML_RE = re.compile(r"<\s*/?\s*(html|body|div|script|style|table|iframe|a)\b|&(?:amp|gt|lt|quot|#x?[0-9a-f]+);", re.I)
CHAT_RE = re.compile(r"<\|(?:im_start|im_end|endoftext|user|assistant|system)\|>|_end_of_the_data|</?think>", re.I)
URL_RE = re.compile(r"https?://|www\.", re.I)
ADULT_CASINO_RE = re.compile(r"\b(?:onlyfans|porn|xxx|casino|jackpot|free spins|sportwetten|sexkontakte)\b", re.I)
SHOP_RE = re.compile(r"\b(?:warenkorb|checkout|rabattcode|gutschein|trusted shops|lieferzeit|versandkosten)\b", re.I)
TOC_RE = re.compile(r"\b(?:inhaltsverzeichnis|table of contents|seite|page)\b|\.{3,}\s*\d{1,5}", re.I)
OCR_RE = re.compile(r"Ã.|�|Å¿|\b[a-zA-ZÄÖÜäöüß](?:\s+[a-zA-ZÄÖÜäöüß]){4,}\b")
MATH_RE = re.compile(r"\b(?:mathematik|problem:|loesung:|lösung:|beweis|theorem|lemma|integral|gleichung)\b|[=+\-*/^]{4,}", re.I)
QA_RE = re.compile(r"\b(?:frage:|antwort:|question:|answer:|instruction:|response:)\b", re.I)
CODE_RE = re.compile(r"\b(?:def|class|import|return|function|const|let|var|public static|#include)\b|[{};]{4,}", re.I)
GERMAN_RE = re.compile(r"\b(?:der|die|das|und|ist|nicht|eine|einer|mit|für|ueber|über|werden|wurde)\b|[äöüÄÖÜß]", re.I)
ENGLISH_RE = re.compile(r"\b(?:the|and|that|with|this|from|were|would|should|because|there)\b", re.I)


def pct(value: int | float, total: int | float) -> float:
    return round(float(value) * 100.0 / float(total), 3) if total else 0.0


def clean_preview(text: str, limit: int = 900) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def source_files(clean_dir: Path) -> list[Path]:
    files = sorted(clean_dir.glob("*.v32.txt"))
    if not files:
        files = sorted(p for p in clean_dir.glob("*.txt") if p.name != "mix_full.txt")
    return files


def random_line_samples(path: Path, n: int, seed: int) -> list[str]:
    if not path.is_file() or n <= 0 or path.stat().st_size <= 0:
        return []
    rng = random.Random(seed)
    size = path.stat().st_size
    docs: list[str] = []
    with path.open("rb") as fh:
        attempts = 0
        while len(docs) < n and attempts < n * 4:
            attempts += 1
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
                docs.append(text)
    return docs


def classify(text: str) -> tuple[str, list[str]]:
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
        flags.append("too_short_chars")

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


def token_length_stats(samples: list[str], tokenizer: Path | None) -> dict:
    if not samples:
        return {}
    if tokenizer is None or spm is None or not tokenizer.is_file():
        char_lengths = [len(s) for s in samples]
        return {
            "sample_docs": len(samples),
            "sample_avg_chars": round(statistics.mean(char_lengths), 3),
            "sample_median_chars": round(statistics.median(char_lengths), 3),
        }
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
    lengths = [len(sp.encode(s, out_type=int)) + 1 for s in samples]
    return {
        "sample_docs": len(lengths),
        "sample_avg_tokens_per_doc": round(statistics.mean(lengths), 3),
        "sample_median_tokens_per_doc": round(statistics.median(lengths), 3),
        "sample_p05_tokens": round(sorted(lengths)[max(0, int(len(lengths) * 0.05) - 1)], 3),
        "sample_p95_tokens": round(sorted(lengths)[min(len(lengths) - 1, int(len(lengths) * 0.95))], 3),
        "sample_docs_lt_64": sum(1 for x in lengths if x < 64),
        "sample_docs_lt_64_pct": pct(sum(1 for x in lengths if x < 64), len(lengths)),
        "sample_docs_lt_100": sum(1 for x in lengths if x < 100),
        "sample_docs_lt_100_pct": pct(sum(1 for x in lengths if x < 100), len(lengths)),
    }


def full_token_scan(path: Path, tokenizer: Path) -> dict:
    if spm is None or not tokenizer.is_file():
        return {"error": "sentencepiece_or_tokenizer_missing"}
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer))
    docs = 0
    tokens = 0
    lt64 = 0
    lt100 = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            n = len(sp.encode(text, out_type=int)) + 1
            docs += 1
            tokens += n
            lt64 += int(n < 64)
            lt100 += int(n < 100)
    return {
        "exact_docs": docs,
        "exact_tokens": tokens,
        "exact_avg_tokens_per_doc": round(tokens / docs, 3) if docs else 0,
        "exact_docs_lt_64_pct": pct(lt64, docs),
        "exact_docs_lt_100_pct": pct(lt100, docs),
    }


def source_manifest(path: Path) -> dict:
    manifest = path.with_suffix(path.suffix + ".manifest.json")
    if manifest.is_file():
        return load_json(manifest)
    return {
        "output_file": str(path),
        "lines_written": None,
        "bytes_written": path.stat().st_size if path.exists() else 0,
        "dropped": {},
    }


def analyze_source(path: Path, *, tokenizer: Path | None, samples_per_source: int, seed: int, full_scan: bool) -> dict:
    manifest = source_manifest(path)
    samples = random_line_samples(path, samples_per_source, seed)
    domain_counts = Counter()
    flag_counts = Counter()
    for sample in samples:
        domain, flags = classify(sample)
        domain_counts[domain] += 1
        flag_counts.update(flags)
    row = {
        "name": path.stem,
        "path": str(path),
        "docs": manifest.get("lines_written"),
        "bytes_written": manifest.get("bytes_written", path.stat().st_size),
        "dropped": manifest.get("dropped", {}),
        "sample_domain_counts": dict(domain_counts),
        "sample_flag_counts": dict(flag_counts),
        "sample_previews": [clean_preview(s) for s in samples[:5]],
    }
    row.update(token_length_stats(samples, tokenizer))
    if full_scan and tokenizer is not None:
        row.update(full_token_scan(path, tokenizer))
    return row


def gate_notes(sources: list[dict], sample_lt100_warn: float) -> list[str]:
    notes: list[str] = []
    hard_flags = ("html", "chat_marker", "adult_or_casino", "shop_spam", "url_dense")
    bad_sources = []
    short_sources = []
    for src in sources:
        flags = src.get("sample_flag_counts") or {}
        hard_count = sum(flags.get(k, 0) for k in hard_flags)
        if hard_count:
            bad_sources.append(f"{src['name']}={hard_count}")
        lt100 = src.get("exact_docs_lt_100_pct", src.get("sample_docs_lt_100_pct", 0))
        if lt100 and lt100 > sample_lt100_warn:
            short_sources.append(f"{src['name']}={lt100}%")
    if bad_sources:
        notes.append("WARN: hard-noise sample flags remain in " + ", ".join(bad_sources[:8]))
    else:
        notes.append("OK: samples show no hard HTML/chat/adult/shop/url-spam flags.")
    if short_sources:
        notes.append("WARN: short-doc rate above threshold in " + ", ".join(short_sources[:8]))
    else:
        notes.append("OK: sampled <100-token rates are below threshold.")
    return notes


def markdown(results: dict) -> str:
    lines = [
        "# Clean Corpus Forensics Report",
        "",
        f"- clean_dir: `{results['clean_dir']}`",
        f"- generated_at: `{results['generated_at']}`",
        "",
        "## Sources",
        "",
        "| Source | Docs | GB | Sample tok/doc | <64 tok | <100 tok | Domain sample | Flags |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for src in results["sources"]:
        flags = ", ".join(f"{k}:{v}" for k, v in (src.get("sample_flag_counts") or {}).items()) or "-"
        domains = ", ".join(f"{k}:{v}" for k, v in (src.get("sample_domain_counts") or {}).items()) or "-"
        lines.append(
            "| {name} | {docs} | {gb:.2f} | {tokdoc} | {lt64}% | {lt100}% | {domains} | {flags} |".format(
                name=src["name"],
                docs=f"{src.get('docs'):,}" if isinstance(src.get("docs"), int) else "-",
                gb=(src.get("bytes_written") or 0) / (1024**3),
                tokdoc=src.get("exact_avg_tokens_per_doc", src.get("sample_avg_tokens_per_doc", "-")),
                lt64=src.get("exact_docs_lt_64_pct", src.get("sample_docs_lt_64_pct", "-")),
                lt100=src.get("exact_docs_lt_100_pct", src.get("sample_docs_lt_100_pct", "-")),
                domains=domains,
                flags=flags,
            )
        )
    lines.extend(["", "## Gate Notes", ""])
    lines.extend(f"- {note}" for note in results["gate_notes"])
    lines.extend(["", "## Sample Previews", ""])
    for src in results["sources"]:
        lines.append(f"### {src['name']}")
        for sample in src.get("sample_previews", []):
            lines.extend(["", "```text", sample, "```"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--samples-per-source", type=int, default=2000)
    parser.add_argument("--sample-lt100-warn", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--full-token-scan", action="store_true")
    return parser.parse_args()


def main() -> None:
    import datetime as _dt

    args = parse_args()
    files = source_files(args.clean_dir)
    if not files:
        raise SystemExit(f"no source .txt files in {args.clean_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        analyze_source(
            path,
            tokenizer=args.tokenizer,
            samples_per_source=args.samples_per_source,
            seed=args.seed + idx,
            full_scan=args.full_token_scan,
        )
        for idx, path in enumerate(files)
    ]
    results = {
        "generated_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "clean_dir": str(args.clean_dir),
        "samples_per_source": args.samples_per_source,
        "full_token_scan": args.full_token_scan,
        "sources": sources,
        "gate_notes": gate_notes(sources, args.sample_lt100_warn),
    }
    (args.output_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "report.md").write_text(markdown(results), encoding="utf-8")
    print(f"wrote {args.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
