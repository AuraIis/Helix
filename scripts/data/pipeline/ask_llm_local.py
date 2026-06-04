"""Ask-LLM scoring against a LocalAI / vLLM / llama.cpp-server endpoint.

Companion to ask_llm_deepseek.py — same JSONL schema, same chunked-streaming
semantics, same --resume support. Differs in two ways:

* Uses raw HTTP via requests + ThreadPoolExecutor instead of distilabel.
  distilabel's OpenAILLM strips the OpenAI 'reasoning' / 'reasoning_content'
  field and only returns 'content', which is empty for thinking-mode Qwen3
  variants where the actual answer ends up in 'reasoning'. Going direct
  lets us read both.

* Tunable --parallel concurrency (default 8, sweet spot for the <host>
  Blackwell + Qwen3.6-35B-A3B-Apex GGUF).

Usage:
    python scripts/data/pipeline/ask_llm_local.py \\
        --input  /staging/raw/fineweb_10bt/fineweb_10bt.txt \\
        --output /staging/cleaned/ask_llm/fineweb_10bt_local.jsonl \\
        --base-url http://172.17.0.1:8765/v1 \\
        --model qwen3.6-35b-a3b-apex \\
        --max-docs 100000 --min-chars 200 --parallel 8 --resume
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import requests


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
    """Stream blank-line-separated docs from a text file."""
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


def score_one(session: requests.Session, base_url: str, model: str,
              doc: dict, max_tokens: int, temperature: float,
              timeout: int) -> dict:
    """Score a single doc. Returns the same JSONL row schema as ask_llm_deepseek."""
    try:
        r = session.post(
            f"{base_url.rstrip('/')}/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": doc["instruction"]}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=timeout,
        )
        r.raise_for_status()
        j = r.json()
        msg = j["choices"][0]["message"]
        # Read content first, then reasoning (LocalAI Apex puts answer there).
        text = ((msg.get("content") or "").strip()
                or (msg.get("reasoning") or "").strip()
                or (msg.get("reasoning_content") or "").strip())
        score = None
        m = re.search(r"[1-5]", text)
        if m:
            score = int(m.group())
    except Exception as e:
        text = f"ERROR: {type(e).__name__}: {e}"
        score = None
    return {
        "doc_id": doc["doc_id"],
        "score": score,
        "raw_response": text,
        "head": doc["head"],
        "length_chars": doc["length_chars"],
        "kept": score is not None and score >= doc["_threshold"],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-docs", type=int, default=100)
    p.add_argument("--threshold", type=int, default=3)
    p.add_argument("--head-chars", type=int, default=1500)
    p.add_argument("--min-chars", type=int, default=200)
    p.add_argument("--max-chars", type=int, default=0,
                   help="0 = no upper cap (default).")
    p.add_argument("--model", default="qwen3.6-35b-a3b-apex")
    p.add_argument("--base-url", default="http://172.17.0.1:8765/v1",
                   help="LocalAI / vLLM / llama-cpp-server endpoint. "
                        "Default points at LocalAI on <host>'s docker bridge.")
    p.add_argument("--parallel", type=int, default=8,
                   help="Concurrent HTTP calls (sweet spot ≈ 8 for "
                        "Blackwell + Qwen3.6-35B; raising past saturates GPU).")
    p.add_argument("--max-tokens", type=int, default=32,
                   help="max_tokens per call. Apex needs ≥16 for the score "
                        "to fit after thinking; 32 gives validity headroom.")
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--chunk-size", type=int, default=500,
                   help="Docs per write-flush (kept for parity with the "
                        "OpenRouter scorer; this script flushes per-row anyway).")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Resume — read existing output for done doc_ids.
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
        print(f"Resume: {len(done_ids)} doc_ids already done.", flush=True)
    elif args.output.exists():
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
            "_threshold": args.threshold,
        })
    print(f"  {len(docs)} docs prepared", flush=True)
    if n_too_short or n_too_long:
        print(f"  pre-filtered: {n_too_short} too-short, {n_too_long} too-long",
              flush=True)
    if n_resumed:
        print(f"  resumed: {n_resumed} doc_ids already done", flush=True)
    if not docs:
        print("  nothing to score, exiting.")
        return 0

    # Quick connectivity check.
    print(f"Pinging {args.base_url} ...", flush=True)
    sess = requests.Session()
    try:
        ping = sess.get(f"{args.base_url.rstrip('/')}/models", timeout=10)
        ping.raise_for_status()
    except Exception as e:
        sys.exit(f"FATAL: cannot reach {args.base_url}: {e}")
    print(f"  endpoint OK, model={args.model!r}", flush=True)

    n_kept = 0
    n_total = 0
    histogram = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, "?": 0}

    chunk_size = args.chunk_size if args.chunk_size > 0 else len(docs)
    n_chunks = (len(docs) + chunk_size - 1) // chunk_size
    print(f"Processing {len(docs)} docs in {n_chunks} chunk(s) of "
          f"{chunk_size}, parallel={args.parallel}.", flush=True)

    import time
    t0 = time.time()
    with args.output.open("a", encoding="utf-8") as out_f, \
         concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
        for chunk_idx in range(n_chunks):
            chunk_docs = docs[chunk_idx * chunk_size:(chunk_idx + 1) * chunk_size]
            futures = [ex.submit(score_one, sess, args.base_url, args.model, d,
                                 args.max_tokens, args.temperature, args.timeout)
                       for d in chunk_docs]
            chunk_kept = 0
            for fut in concurrent.futures.as_completed(futures):
                rec = fut.result()
                # Strip _threshold sentinel before writing.
                rec_out = {k: v for k, v in rec.items() if k != "_threshold"}
                n_total += 1
                histogram[rec["score"] if rec["score"] in histogram else "?"] += 1
                if rec["kept"]:
                    n_kept += 1
                    chunk_kept += 1
                out_f.write(json.dumps(rec_out, ensure_ascii=False) + "\n")
            out_f.flush()
            elapsed = time.time() - t0
            rate = n_total / elapsed
            eta = (len(docs) - n_total) / rate if rate > 0 else 0
            print(f"  chunk {chunk_idx + 1}/{n_chunks} done: {chunk_kept} kept "
                  f"(running {n_kept}/{n_total}, {rate:.2f} d/s, ETA {eta/60:.0f} min)",
                  flush=True)

    print()
    print(f"=== Ask-LLM-Local results ({args.model}) ===")
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
