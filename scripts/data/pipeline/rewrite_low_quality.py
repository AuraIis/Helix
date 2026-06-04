"""Rewrite low-quality docs into clean prose (Nemotron-CC / SwallowCode pattern).

Picks documents from an Ask-LLM-scored JSONL within a chosen score band
(default 2-3 = "salvageable but mediocre"), re-loads the full document
text from the original blank-separated corpus, and asks an LLM to
rewrite each one into clean informative prose suitable for pretraining.

The rewriter prompt is anti-hallucination (per L-017): no new facts,
SKIP marker for unsalvageable inputs, language preserved.

Output JSONL has one record per attempt with the original metadata,
the rewrite text (or null if SKIP), and a flag for downstream re-scoring.

Usage:
    OPENROUTER_API_KEY=sk-or-... \\
    python scripts/data/pipeline/rewrite_low_quality.py \\
        --scored /staging/cleaned/ask_llm/fineweb_10bt_5k_qwen36_scored.jsonl \\
        --source /staging/raw/fineweb_10bt/fineweb_10bt.txt \\
        --output /staging/cleaned/rewrites/fineweb_10bt_rewrite_qwen36.jsonl \\
        --model  qwen/qwen3.6-35b-a3b \\
        --max-docs 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable


# IMPORTANT: prompts are language-matched to the source doc.
# A first-pass with a single German prompt caused DeepSeek-V3.2 to translate
# 92% and Qwen3.6 82% of English source docs into German — the models follow
# the language of the prompt, not the rule "preserve source language".
# Fix: detect lang first, render prompt in that language so the model's
# default behaviour aligns with the requirement.

REWRITE_PROMPT_DE = """\
Du bekommst einen Webdokument-Auszug von wechselhafter Qualität.
Schreibe ihn um in klare, informative deutsche Prosa für ein LLM-Pretraining-Korpus.

REGELN:
- Antworte AUSSCHLIESSLICH auf Deutsch. Übersetze nichts.
- Erfinde KEINE neuen Fakten. Wenn ein Detail im Original nicht steht, lass es weg.
- Entferne Boilerplate: Navigation, Werbung, Cookie-Banner, Datums-/Autoren-Zeilen
  ohne Kontext, Login-Aufforderungen.
- Behalte alle Sachaussagen, Namen, Zahlen, Daten, Zitate, Quellenangaben.
- Wenn das Dokument zu fragmentiert ist um etwas Sinnvolles zu retten,
  antworte exakt mit dem Wort: SKIP

Original-Dokument:
{doc}

Schreibe den umformulierten Text direkt — ohne Vorrede, ohne Meta-Kommentare.
Nur die Prosa selbst, oder das Wort SKIP."""

REWRITE_PROMPT_EN = """\
You are given a web-document excerpt of variable quality.
Rewrite it as clean, informative English prose for an LLM pretraining corpus.

RULES:
- Reply ONLY in English. Do NOT translate.
- Do NOT invent new facts. If a detail isn't in the original, leave it out.
- Strip boilerplate: navigation, ads, cookie banners, "page 1 of N", login walls,
  "Email this article" links, byline-only date stubs.
- Keep all factual statements, names, numbers, dates, quotes, source citations.
- If the document is too fragmentary or content-free to salvage,
  reply with exactly the word: SKIP

Original document:
{doc}

Write the rewritten prose directly — no preamble, no meta-commentary, no
"Here is the rewrite:". Just the prose itself, or the word SKIP."""


def detect_language(text: str) -> str:
    """Lightweight DE-vs-EN classifier based on stop-word counts.
    No dependency on external libraries. Returns 'de' or 'en'."""
    sample = text[:1000].lower()
    de_markers = (" der ", " die ", " das ", " und ", " ist ", " nicht ",
                  " sich ", " wird ", " mit ", " auf ", " den ", " für ",
                  " eine ", " auch ", " von ", " im ", " zu ")
    en_markers = (" the ", " and ", " of ", " is ", " that ", " to ",
                  " a ", " in ", " for ", " with ", " on ", " was ",
                  " it ", " be ", " as ", " at ", " by ")
    de = sum(sample.count(m) for m in de_markers)
    en = sum(sample.count(m) for m in en_markers)
    return "de" if de > en else "en"


def render_prompt(doc: str) -> tuple[str, str]:
    """Return (lang_code, rendered_prompt) for the source doc."""
    lang = detect_language(doc)
    template = REWRITE_PROMPT_DE if lang == "de" else REWRITE_PROMPT_EN
    return lang, template.format(doc=doc)


def read_blank_separated_docs(path: Path) -> Iterable[tuple[int, str]]:
    """Stream blank-line-separated docs from a text file. Same logic as
    the Ask-LLM scorer so doc_ids align across runs."""
    buf: list[str] = []
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip() == "":
                if buf:
                    text = " ".join(b.strip() for b in buf if b.strip())
                    if text:
                        yield n, text
                        n += 1
                    buf.clear()
            else:
                buf.append(line)
        if buf:
            text = " ".join(b.strip() for b in buf if b.strip())
            if text:
                yield n, text


def load_scored_band(scored_path: Path, score_min: int, score_max: int,
                     max_docs: int | None) -> dict[int, int]:
    """Returns {doc_id -> original_score} for docs in the chosen score band."""
    keep: dict[int, int] = {}
    with scored_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            score = rec.get("score")
            if score is None:
                continue
            if score_min <= score <= score_max:
                keep[rec["doc_id"]] = score
                if max_docs and len(keep) >= max_docs:
                    break
    return keep


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scored", type=Path, required=True,
                   help="Ask-LLM-scored JSONL (input)")
    p.add_argument("--source", type=Path, required=True,
                   help="Original blank-separated raw text corpus")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--score-min", type=int, default=2,
                   help="Pick docs with score >= this (default 2)")
    p.add_argument("--score-max", type=int, default=3,
                   help="Pick docs with score <= this (default 3)")
    p.add_argument("--max-docs", type=int, default=50,
                   help="Cap rewrite candidates (default 50, use 0 for unlimited)")
    p.add_argument("--head-chars", type=int, default=4000,
                   help="Truncate source doc to this many chars before rewrite (default 4000)")
    p.add_argument("--model", default="qwen/qwen3.6-35b-a3b",
                   help="OpenRouter model id (default qwen3.6-35b-a3b)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-output-tokens", type=int, default=1024,
                   help="Max tokens for the rewrite (default 1024)")
    p.add_argument("--chunk-size", type=int, default=200,
                   help="Process this many docs per pipeline.run() call. "
                        "Each rewrite is ~1024 output tokens, so chunks of 200 "
                        "≈ 200 s of work + ~1 s overhead. Smaller = better "
                        "crash-resumability. Set 0 to disable chunking.")
    p.add_argument("--resume", action="store_true",
                   help="If output exists, skip doc_ids already present and "
                        "append new ones. Default: fresh run (truncate).")
    args = p.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("FATAL: OPENROUTER_API_KEY env var is required")

    # Resume: load doc_ids already present in output.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done_ids: set[int] = set()
    if args.resume and args.output.exists():
        with args.output.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("doc_id") is not None:
                        done_ids.add(rec["doc_id"])
                except json.JSONDecodeError:
                    continue
        print(f"Resume: {len(done_ids)} doc_ids already in output, will skip.",
              flush=True)
    elif args.output.exists() and not args.resume:
        args.output.write_text("")

    print(f"Loading scored band [{args.score_min}, {args.score_max}] from "
          f"{args.scored} ...", flush=True)
    candidates = load_scored_band(args.scored, args.score_min, args.score_max,
                                   args.max_docs if args.max_docs > 0 else None)
    # Drop already-done IDs from candidates so the source-walk only loads
    # the texts we still need (saves disk I/O on big corpora).
    n_resumed = sum(1 for did in candidates if did in done_ids)
    candidates = {did: s for did, s in candidates.items() if did not in done_ids}
    print(f"  {len(candidates)} candidate doc_ids "
          f"({n_resumed} already in output, skipped)", flush=True)

    if not candidates:
        print("Nothing to rewrite. Exiting.", flush=True)
        return 0

    # Re-load full doc text from source by doc_id, render lang-matched prompt.
    print(f"Pulling full text from {args.source} ...", flush=True)
    docs: list[dict] = []
    lang_counts = {"de": 0, "en": 0}
    seen = 0
    for doc_id, text in read_blank_separated_docs(args.source):
        if doc_id in candidates:
            truncated = text[:args.head_chars]
            lang, rendered = render_prompt(truncated)
            lang_counts[lang] += 1
            docs.append({
                "doc_id": doc_id,
                "original_score": candidates[doc_id],
                "original_chars": len(text),
                "source_lang": lang,
                "instruction": rendered,
                "head": text[:200],
            })
            seen += 1
            if seen == len(candidates):
                break
    print(f"  {len(docs)} docs loaded "
          f"(detected language: DE={lang_counts['de']}, EN={lang_counts['en']})",
          flush=True)

    from distilabel.pipeline import Pipeline
    from distilabel.steps import LoadDataFromDicts
    from distilabel.steps.tasks import TextGeneration
    from distilabel.models import OpenAILLM

    # extra_body disables reasoning for Qwen3.6 / GPT-5 / etc.
    # Non-reasoning models silently ignore it.
    llm = OpenAILLM(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1",
        model=args.model,
        generation_kwargs={
            "temperature": 0.3,
            "max_new_tokens": args.max_output_tokens,
            "extra_body": {"reasoning": {"enabled": False}},
        },
    )

    chunk_size = args.chunk_size if args.chunk_size > 0 else len(docs)
    n_chunks = (len(docs) + chunk_size - 1) // chunk_size

    n_total = 0
    n_skip = 0
    n_rewritten = 0
    n_empty = 0

    def _iter_rows(distiset):
        for leaf_name, leaf in distiset.items():
            for split_name, ds in leaf.items():
                for row in ds:
                    yield row

    print(f"Processing {len(docs)} docs in {n_chunks} chunk(s) of "
          f"{chunk_size} (streaming-append, resumable).", flush=True)

    with args.output.open("a", encoding="utf-8") as out_f:
        for chunk_idx in range(n_chunks):
            chunk_docs = docs[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
            print(f"  chunk {chunk_idx + 1}/{n_chunks}: "
                  f"{len(chunk_docs)} docs ...", flush=True)

            with Pipeline(name=f"rewrite-{args.model.replace('/', '_')}") as pipeline:
                loader = LoadDataFromDicts(data=chunk_docs)
                rewrite_step = TextGeneration(
                    llm=llm,
                    input_batch_size=args.batch_size,
                    num_generations=1,
                )
                loader >> rewrite_step

            distiset = pipeline.run(use_cache=False)

            chunk_rewritten = 0
            chunk_skip = 0
            for row in _iter_rows(distiset):
                n_total += 1
                generation = (row.get("generation") or "").strip()
                skipped = generation.upper().strip() == "SKIP" or generation == ""
                if generation == "":
                    n_empty += 1
                elif skipped:
                    n_skip += 1
                    chunk_skip += 1
                else:
                    n_rewritten += 1
                    chunk_rewritten += 1
                rec = {
                    "doc_id": row.get("doc_id"),
                    "original_score": row.get("original_score"),
                    "original_chars": row.get("original_chars"),
                    "source_lang": row.get("source_lang"),
                    "head": row.get("head"),
                    "rewrite": None if skipped else generation,
                    "rewrite_chars": 0 if skipped else len(generation),
                    "skipped": skipped,
                    "model": args.model,
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"    chunk {chunk_idx + 1} done: {chunk_rewritten} rewritten, "
                  f"{chunk_skip} SKIP — running total: {n_rewritten}/{n_total}",
                  flush=True)

    print()
    print(f"=== Rewrite results ({args.model}) ===")
    print(f"  candidates:  {n_total} (this run)")
    if n_resumed:
        print(f"  resumed:     {n_resumed} previously-done docs not re-rewritten")
    print(f"  rewritten:   {n_rewritten}  ({100*n_rewritten/max(n_total,1):.1f}%)")
    print(f"  skipped:     {n_skip}")
    print(f"  empty:       {n_empty}")
    print(f"  output:      {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
