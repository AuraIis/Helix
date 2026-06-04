"""Ask-LLM proof-of-concept.

Score a small batch of documents 1-5 on quality using a small instruction-
tuned model. The reference paper used Flan-T5-large; we use the smaller
Flan-T5-base (~250M params) for the smoke test.

Usage:
    python scripts/eval/ask_llm_poc.py \
        --input /tmp/dt_smoke/sample.txt \
        --output /tmp/ask_llm_scores.jsonl \
        --max-docs 50

Output: JSONL with {doc_id, score, length_chars, head}.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path


PROMPT_TEMPLATE = """\
You are a strict data-quality evaluator for an LLM-pretraining corpus.
Rate the following document on a 1-5 scale where:
1 = useless boilerplate, navigation menu, link spam, gibberish.
2 = mostly noise, occasional useful sentence.
3 = mediocre web text, some grammatical issues.
4 = clean, informative, well-formed prose.
5 = high-quality reference / encyclopaedia / textbook.

Document:
{doc_head}

Respond with a single digit 1-5 and nothing else.
"""


def read_blank_separated_docs(path: Path):
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-docs", type=int, default=50)
    parser.add_argument("--model", default="google/flan-t5-base")
    parser.add_argument("--head-chars", type=int, default=1500,
                        help="how many chars of each doc to feed the scorer")
    args = parser.parse_args()

    print(f"Loading {args.model} ...", flush=True)
    from transformers import T5Tokenizer, T5ForConditionalGeneration
    import torch

    tokenizer = T5Tokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"  device: {device}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    n_scored = 0
    histogram = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, "?": 0}
    t0 = time.time()

    digit_re = re.compile(r"[1-5]")
    with args.output.open("w", encoding="utf-8") as f_out:
        for doc_id, doc in read_blank_separated_docs(args.input):
            if n_scored >= args.max_docs:
                break
            head = doc[: args.head_chars]
            prompt = PROMPT_TEMPLATE.format(doc_head=head)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=2, do_sample=False)
            text = tokenizer.decode(out[0], skip_special_tokens=True).strip()
            m = digit_re.search(text)
            score = int(m.group()) if m else None

            rec = {
                "doc_id": doc_id,
                "score": score,
                "raw_response": text,
                "length_chars": len(doc),
                "head": head[:200],
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            histogram[score if score in histogram else "?"] += 1
            n_scored += 1
            if n_scored % 10 == 0:
                rate = n_scored / max(time.time() - t0, 0.01)
                print(f"  {n_scored}/{args.max_docs} ({rate:.1f} docs/s)", flush=True)

    elapsed = time.time() - t0
    print(f"\n=== Ask-LLM POC results ===")
    print(f"  scored: {n_scored} docs in {elapsed:.1f}s ({n_scored/elapsed:.1f} docs/s)")
    print(f"  output: {args.output}")
    print(f"  score distribution:")
    for k in [1, 2, 3, 4, 5, "?"]:
        bar = "█" * histogram[k]
        print(f"    {k}: {histogram[k]:3d}  {bar}")
    avg_known = sum(k * v for k, v in histogram.items() if isinstance(k, int))
    n_known = sum(v for k, v in histogram.items() if isinstance(k, int))
    if n_known:
        print(f"  mean score (excl. ?): {avg_known / n_known:.2f}")


if __name__ == "__main__":
    main()
