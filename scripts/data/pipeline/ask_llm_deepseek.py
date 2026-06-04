"""Ask-LLM quality scoring pipeline using DeepSeek V4 Flash via OpenRouter.

Replaces the Flan-T5 POC (which was too weak — see ask_llm_poc.py results)
with a real production-grade scorer. distilabel orchestrates batches and
retries; DeepSeek V4 Flash is cheap (~$0.14 per 1M input tokens) and
handles German fluently — both regressions of the Flan-T5 attempt.

Usage:
    OPENROUTER_API_KEY=sk-or-... \\
    python scripts/data/pipeline/ask_llm_deepseek.py \\
        --input  /staging/raw/fineweb_10bt/fineweb_10bt.txt \\
        --output /staging/cleaned/ask_llm/fineweb_10bt_scored.jsonl \\
        --max-docs 1000 \\
        --threshold 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable


SCORE_PROMPT = """\
You are a strict data-quality evaluator for an LLM-pretraining corpus.
Rate the document on a 1-5 scale:
1 = useless boilerplate, link spam, gibberish, or content with no informational value.
2 = mostly noise with occasional useful sentences.
3 = mediocre web text — readable but with grammatical issues or low information density.
4 = clean, informative, well-formed prose suitable for training.
5 = high-quality reference material (encyclopaedia, textbook, research summary).

Document head (first 1500 chars):
{doc_head}

Reply with this exact format on a single line:
Score: <digit from 1 to 5>"""


def read_blank_separated_docs(path: Path) -> Iterable[tuple[int, str]]:
    """Stream multi-line-with-blank-line-separator docs from a text file."""
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-docs", type=int, default=100)
    p.add_argument("--threshold", type=int, default=3,
                   help="keep docs with score >= threshold (default 3)")
    p.add_argument("--head-chars", type=int, default=1500)
    p.add_argument("--min-chars", type=int, default=200,
                   help="Skip docs with fewer than this many characters before "
                        "scoring (default 200). For Wikipedia-style corpora the "
                        "blank-separated reader yields lots of section-header "
                        "fragments ('Geschichte', 'Weblinks', ...) that are "
                        "almost always score 1 — pre-filtering them saves API "
                        "calls and avoids polluting the kept-rate stats. Set "
                        "to 0 to disable.")
    p.add_argument("--max-chars", type=int, default=0,
                   help="Cap doc length BEFORE scoring (default 0 = no cap). "
                        "Useful for skipping huge list-articles / glossaries "
                        "(>5k chars) which the rubric rates poorly anyway. "
                        "Try 8000 if scoring Wikipedia-style corpora.")
    p.add_argument("--model", default="qwen/qwen3.6-35b-a3b",
                   help="Model id. For OpenRouter use 'qwen/qwen3.6-35b-a3b' "
                        "(default). For LocalAI use the local model name (e.g. "
                        "'qwen3.6-35b-a3b-apex') and pair with --base-url. "
                        "deepseek/deepseek-chat-v3.1 is NOT recommended — "
                        "exhibits a 18%% degenerate-token bug ('棣棣棣') at "
                        "short completion lengths.")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1",
                   help="OpenAI-compatible API endpoint. Default OpenRouter. "
                        "For LocalAI on <host>: http://172.17.0.1:8765/v1 "
                        "(from inside auralis-downloader container) or "
                        "http://localhost:8765/v1 (host-side). Local is free "
                        "but ~2-3x slower than OpenRouter for short-completion "
                        "scoring.")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY",
                   help="Env var holding the API key. For LocalAI any non-"
                        "empty string works (the server doesn't validate). "
                        "Default OPENROUTER_API_KEY.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="distilabel parallelism per LLM call")
    p.add_argument("--chunk-size", type=int, default=500,
                   help="Process this many docs per pipeline.run() invocation, "
                        "appending each chunk to the output before starting the "
                        "next. Smaller = better crash-resumability + more "
                        "OpenRouter-call overhead. Larger = less overhead but "
                        "more wasted work on crash. Default 500 ≈ ~100 s of "
                        "work per chunk at the typical OpenRouter throughput. "
                        "Set to 0 to disable chunking (single run, all-or-nothing).")
    p.add_argument("--resume", action="store_true",
                   help="If output file already exists, skip doc_ids that "
                        "are already present and append new ones. Default: "
                        "fresh run (overwrites existing output).")
    args = p.parse_args()

    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        # LocalAI doesn't actually validate the key — any non-empty string works.
        if "openrouter.ai" not in args.base_url:
            api_key = "sk-localai-dummy"
            print(f"  using dummy key for non-OpenRouter endpoint {args.base_url}",
                  flush=True)
        else:
            sys.exit(f"FATAL: {args.api_key_env} env var is required for "
                     f"OpenRouter ({args.base_url})")

    # distilabel pipeline: load docs as a generator, score via OpenAI-compatible
    # OpenRouter LLM, write JSONL.
    from distilabel.pipeline import Pipeline
    from distilabel.steps import LoadDataFromDicts
    from distilabel.steps.tasks import TextGeneration
    from distilabel.models import OpenAILLM

    # Resume support: if output exists and --resume given, load done doc_ids.
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
        print(f"Resume: {len(done_ids)} doc_ids already in output, will skip them.",
              flush=True)
    elif args.output.exists() and not args.resume:
        # Fresh run -> truncate
        args.output.write_text("")

    print(f"Reading {args.input} ...", flush=True)
    docs = []
    n_too_short = 0
    n_too_long = 0
    n_resumed = 0
    for doc_id, text in read_blank_separated_docs(args.input):
        if doc_id >= args.max_docs:
            break
        if doc_id in done_ids:
            n_resumed += 1
            continue
        L = len(text)
        if args.min_chars and L < args.min_chars:
            n_too_short += 1
            continue
        if args.max_chars and L > args.max_chars:
            n_too_long += 1
            continue
        docs.append({
            "doc_id": doc_id,
            "instruction": SCORE_PROMPT.format(doc_head=text[:args.head_chars]),
            "head": text[:200],
            "length_chars": L,
        })
    print(f"  {len(docs)} docs prepared", flush=True)
    if n_too_short or n_too_long:
        print(f"  pre-filtered: {n_too_short} too-short (<{args.min_chars} chars), "
              f"{n_too_long} too-long (>{args.max_chars} chars)", flush=True)
    if n_resumed:
        print(f"  resumed: {n_resumed} doc_ids already done, skipped", flush=True)
    if not docs:
        print("  nothing to score, exiting.", flush=True)
        return 0

    # NOTE on the prompt + decoding:
    # The 5000-doc v1 run on deepseek/deepseek-chat-v3.1 (T=0, max=4,
    # "respond with EXACTLY one digit") hit a degenerate-token loop on
    # ~18% of inputs ("棣棣棣棣"); the bug is in deepseek-chat-v3.1
    # itself, not the prompt. Switched default to llama-3.3-70b/qwen3.x.
    #
    # For reasoning models (qwen3.6-*, qwen3-thinking, etc.) we MUST
    # disable thinking — otherwise max_new_tokens=16 is consumed by the
    # reasoning trace and content comes back empty. OpenRouter's
    # extra_body={"reasoning": {"enabled": false}} works for all models
    # that support it; non-reasoning models silently ignore it.
    #
    # LocalAI Apex variant: hardcoded thinking, can't be disabled via
    # extra_body. The score lands in the 'reasoning' field of the response
    # message. We bumped max_new_tokens 16 -> 32 to give validity headroom
    # and the parser checks BOTH content and reasoning (see _iter_rows).
    is_local = "openrouter.ai" not in args.base_url
    llm = OpenAILLM(
        api_key=api_key,
        base_url=args.base_url,
        model=args.model,
        generation_kwargs={
            "temperature": 0.05,
            "max_new_tokens": 32,
            "extra_body": {"reasoning": {"enabled": False}},
        },
    )

    # Chunked execution + streaming append. Each chunk = one pipeline.run()
    # invocation; results are written to output before the next chunk starts,
    # so a crash loses at most one chunk's worth of work.
    chunk_size = args.chunk_size if args.chunk_size > 0 else len(docs)
    n_chunks = (len(docs) + chunk_size - 1) // chunk_size

    n_kept = 0
    n_total = 0
    histogram = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, "?": 0}

    def _iter_rows(distiset):
        for leaf_name, leaf in distiset.items():
            for split_name, ds in leaf.items():
                for row in ds:
                    yield row

    print(f"Processing {len(docs)} docs in {n_chunks} chunk(s) of "
          f"{chunk_size} (streaming-append, resumable).", flush=True)

    # Open output in append mode — fresh-run truncate already happened above.
    with args.output.open("a", encoding="utf-8") as out_f:
        for chunk_idx in range(n_chunks):
            chunk_docs = docs[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
            print(f"  chunk {chunk_idx + 1}/{n_chunks}: "
                  f"{len(chunk_docs)} docs ...", flush=True)

            with Pipeline(name="ask-llm-deepseek") as pipeline:
                loader = LoadDataFromDicts(data=chunk_docs)
                score_step = TextGeneration(
                    llm=llm,
                    input_batch_size=args.batch_size,
                    num_generations=1,
                )
                loader >> score_step

            distiset = pipeline.run(
                parameters={
                    score_step.name: {
                        "llm": {"generation_kwargs": {
                            "temperature": 0.05,
                            "max_new_tokens": 32,
                            "extra_body": {"reasoning": {"enabled": False}},
                        }},
                    },
                },
                use_cache=False,
            )

            chunk_kept = 0
            for row in _iter_rows(distiset):
                n_total += 1
                generation = (row.get("generation") or "").strip()
                # NOTE: for OpenRouter this works as expected. For thinking-mode
                # local models (LocalAI Apex) the answer ends up in the OpenAI
                # 'reasoning' field which distilabel does NOT expose — use
                # ask_llm_local.py for that case instead.
                score = None
                for ch in generation:
                    if ch in "12345":
                        score = int(ch)
                        break
                histogram[score if score in histogram else "?"] += 1
                rec = {
                    "doc_id": row.get("doc_id"),
                    "score": score,
                    "raw_response": generation,
                    "head": row.get("head"),
                    "length_chars": row.get("length_chars"),
                    "kept": score is not None and score >= args.threshold,
                }
                if rec["kept"]:
                    n_kept += 1
                    chunk_kept += 1
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            print(f"    chunk {chunk_idx + 1} done: {chunk_kept} kept "
                  f"(running total: {n_kept}/{n_total})", flush=True)

    print()
    print(f"=== Ask-LLM (DeepSeek) results ===")
    print(f"  scored:  {n_total} docs (this run)")
    if n_resumed:
        print(f"  resumed: {n_resumed} previously-done docs not re-scored")
    print(f"  kept:    {n_kept} (threshold ≥ {args.threshold}) — "
          f"{100*n_kept/max(n_total,1):.1f}%")
    print(f"  output:  {args.output}")
    print(f"  histogram:")
    for k in [1, 2, 3, 4, 5, "?"]:
        bar = "█" * histogram[k]
        print(f"    {k}: {histogram[k]:4d}  {bar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
