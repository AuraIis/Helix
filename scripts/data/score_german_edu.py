#!/usr/bin/env python3
"""Score German documents for educational/informational value (FineWeb-Edu style).

The German corpus is clean of garbage but was never *education*-filtered the way
English fineweb_edu was. This tool brings the same methodology to German:

  1. ANNOTATE: a strong LLM scores a sample of German docs 0-5 on educational
     value (additive rubric, German-adapted).
  2. (later) train a cheap classifier on these labels and score the full corpus.

It reuses the OpenAI-compatible QwenClient, so the backend is whatever
OPENAI_API_BASE points at — a local Qwen via vLLM, or DeepSeek's API, etc.

Run per source to see the score distribution (validates the hypothesis +
informs the mix-rebalance):

    OPENAI_API_BASE=http://localhost:8000/v1 OPENAI_MODEL=Qwen2.5-32B-Instruct \
    python scripts/data/score_german_edu.py \
        --input data/training/cleaned/fineweb2_de.filtered.txt \
        --source fineweb2_de --sample 400 \
        --output-jsonl eval/results/de_edu/fineweb2_de.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.data.synth.qwen_client import GenRequest, QwenClient  # noqa: E402

RUBRIC = (
    "Du bewertest den Bildungs- und Informationswert eines deutschen Textauszugs "
    "für das Vortraining eines Sprachmodells. Nutze ein additives 0–5-Punkte-System:\n"
    "- +1: enthält grundlegende Information (auch mit etwas Werbung/Navigation/Boilerplate).\n"
    "- +1: behandelt konkrete Aspekte eines Sachthemas, auch wenn unvollständig.\n"
    "- +1: klar, kohärent, sachlich geschrieben; vermittelt nachvollziehbares Wissen.\n"
    "- +1: hoch relevant und lehrreich, gut strukturiert, kaum irrelevanter Inhalt "
    "(wie ein Lehrbuch-/Enzyklopädie-Abschnitt).\n"
    "- +1: herausragender Bildungswert: tiefgehend, präzise, frei von Boilerplate/Werbung/Listen-Müll.\n\n"
    "Antworte mit max. einem Satz Begründung und dann GENAU dieser Schlusszeile:\n"
    "Bewertung: <Zahl 0-5>"
)


def build_messages(doc: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": RUBRIC},
        {"role": "user", "content": f'Textauszug:\n"""\n{doc}\n"""'},
    ]


def parse_score(text: str) -> int | None:
    m = re.findall(r"Bewertung:\s*([0-5])", text)
    if m:
        return int(m[-1])
    m2 = re.findall(r"\b([0-5])\b", text)
    return int(m2[-1]) if m2 else None


def sample_docs(path: Path, n: int, scan_lines: int, max_chars: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    pool: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= scan_lines:
                break
            line = line.strip()
            if len(line) >= 200:                     # skip trivially short lines
                pool.append(line[:max_chars])
    if not pool:
        raise SystemExit(f"no usable docs in {path}")
    rng.shuffle(pool)
    return pool[:n]


async def annotate(docs: list[str], source: str, *, base_url: str, api_key: str,
                   model: str, concurrency: int, max_tokens: int = 1024,
                   reasoning_effort: str = "") -> list[dict]:
    client = QwenClient(base_url=base_url, api_key=api_key, model=model,
                        max_concurrency=concurrency)
    extra_body = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    reqs = [
        GenRequest(request_id=f"{source}-{i}", messages=build_messages(doc),
                   max_tokens=max_tokens, temperature=0.0,
                   extra={"source": source, "snippet": doc[:160], "text": doc},
                   extra_body=extra_body)
        for i, doc in enumerate(docs)
    ]
    rows: list[dict] = []
    async for res in client.run_batch(reqs):
        rows.append({
            "id": res.request_id,
            "source": (res.extra or {}).get("source"),
            "score": None if res.error else parse_score(res.completion),
            "snippet": (res.extra or {}).get("snippet"),
            "text": (res.extra or {}).get("text"),
            "raw": res.completion[:300],
            "error": res.error,
        })
    print("client stats:", json.dumps(client.stats_summary()), file=sys.stderr)
    return rows


def summarize(rows: list[dict], source: str) -> None:
    scored = [r["score"] for r in rows if r["score"] is not None]
    n_err = sum(1 for r in rows if r["error"])
    n_unparsed = sum(1 for r in rows if r["error"] is None and r["score"] is None)
    hist = {s: sum(1 for x in scored if x == s) for s in range(6)}
    mean = sum(scored) / len(scored) if scored else float("nan")
    keep = sum(1 for x in scored if x >= 3) / len(scored) if scored else 0.0
    print(f"\n=== {source}: {len(scored)} scored "
          f"(errors {n_err}, unparsed {n_unparsed}) ===")
    for s in range(6):
        bar = "#" * round(40 * hist[s] / max(1, len(scored)))
        print(f"  score {s}: {hist[s]:4d} {bar}")
    print(f"  mean: {mean:.2f} | fraction >= 3 (keep-bar): {keep*100:.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--source", required=True, help="label for this source, e.g. fineweb2_de")
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--scan-lines", type=int, default=200_000)
    ap.add_argument("--max-chars", type=int, default=2000)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=1024,
                    help="Output token budget. Thinking models (Gemini 3.x) count "
                         "reasoning against this, so keep it generous (>=512) or the "
                         "visible 'Bewertung:' line gets truncated.")
    ap.add_argument("--reasoning-effort", default="",
                    help="Pass-through to thinking models (e.g. 'low'/'medium'/'high' "
                         "for Gemini 3.x). Empty = provider default. 'low' cuts latency "
                         "and cost on Gemini 3.5 Flash with negligible quality loss for "
                         "a 0-5 rating task.")
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_API_BASE", "http://localhost:8000/v1"))
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "Qwen2.5-32B-Instruct"))
    args = ap.parse_args()

    docs = sample_docs(args.input, args.sample, args.scan_lines, args.max_chars, args.seed)
    print(f"scoring {len(docs)} docs from {args.input} (source={args.source}) "
          f"via {args.model} @ {args.base_url}", file=sys.stderr)
    rows = asyncio.run(annotate(docs, args.source, base_url=args.base_url,
                                api_key=args.api_key, model=args.model,
                                concurrency=args.concurrency, max_tokens=args.max_tokens,
                                reasoning_effort=args.reasoning_effort))
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summarize(rows, args.source)
    print(f"wrote {args.output_jsonl}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
