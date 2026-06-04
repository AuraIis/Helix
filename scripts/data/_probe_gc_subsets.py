#!/usr/bin/env python3
"""Probe coral-nlp/german-commons: real fields, source/subset mix, and whether
clean MODERN German (low perplexity) is identifiable/abundant vs OCR-historical."""
import itertools
from collections import Counter

from datasets import load_dataset

ds = load_dataset("coral-nlp/german-commons", split="train", streaming=True)
src = Counter()
sub = Counter()
ppl_buckets = Counter()
clean = []
first_keys = None
N = 8000
for i, ex in enumerate(itertools.islice(ds, N)):
    if first_keys is None:
        first_keys = list(ex.keys())
        print("KEYS:", first_keys)
    s = str(ex.get("source", ""))
    su = str(ex.get("subset", ""))
    src[s] += 1
    sub[su] += 1
    ppl = ex.get("perplexity") or 9999
    ocr = ex.get("ocr_score")
    b = "<200" if ppl < 200 else "200-500" if ppl < 500 else "500-1000" if ppl < 1000 else ">=1000"
    ppl_buckets[b] += 1
    if ppl < 200 and len(clean) < 12:
        clean.append((round(float(ppl), 1), ocr, s, su, (ex.get("text", "") or "").replace("\n", " ")[:160]))

print(f"\nsource counts (first {N}):", dict(src))
print("subset counts:", dict(sub))
print("perplexity buckets:", dict(ppl_buckets))
print("\nLOW-ppl (<200) modern-candidate samples:")
for ppl, ocr, s, su, t in clean:
    print(f"  ppl={ppl} ocr={ocr} source={s!r} subset={su!r} :: {t!r}")
