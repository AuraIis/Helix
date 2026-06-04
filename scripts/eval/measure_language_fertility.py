#!/usr/bin/env python3
"""Measure tokenizer fertility per language and convert per-token loss to
bits-per-byte so languages are actually comparable.

Why: per-token cross-entropy is NOT comparable across languages, because the
tokenizer splits them into different numbers of pieces. The canary showed
German per-token val loss 8.283 vs English 5.216 — but that ratio is partly a
tokenization artifact. The fair metric is bits-per-byte:

    bpb = per_token_nats * (tokens / byte) / ln(2)

This tells you the TRUE difficulty/quality gap between languages, independent of
how the tokenizer chops them up. Run inside the container.

Example:
    python scripts/eval/measure_language_fertility.py \
      --tokenizer tokenizer/helix_v2_tokenizer.model \
      --input english=/workspace/v2data/data/training/curated_40b/english.txt \
      --input german=/workspace/v2data/data/training/curated_40b/german.txt \
      --loss english=5.216 --loss german=8.283
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def _kv(pairs: list[str]) -> dict[str, str]:
    out = {}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"expected lang=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k] = v
    return out


def sample_text(path: str, n_bytes: int) -> str:
    with open(path, "rb") as fh:
        return fh.read(n_bytes).decode("utf-8", errors="ignore")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--input", action="append", required=True,
                    help="lang=path (repeatable)")
    ap.add_argument("--loss", action="append", default=[],
                    help="lang=per_token_nats (repeatable; optional)")
    ap.add_argument("--sample-bytes", type=int, default=5_000_000)
    args = ap.parse_args()

    import sentencepiece as spm

    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer)
    inputs = _kv(args.input)
    losses = {k: float(v) for k, v in _kv(args.loss).items()}
    print(f"tokenizer vocab: {sp.get_piece_size()}  | sample: {args.sample_bytes/1e6:.1f} MB/lang\n")

    rows = []
    for lang, path in inputs.items():
        if not Path(path).exists():
            print(f"[skip] {lang}: {path} not found")
            continue
        text = sample_text(path, args.sample_bytes)
        lines = [ln for ln in text.split("\n") if ln.strip()]
        lines = lines[:-1] if len(lines) > 1 else lines
        nbytes = sum(len(ln.encode("utf-8")) for ln in lines)
        nwords = sum(len(ln.split()) for ln in lines)
        ntok = sum(len(ids) for ids in sp.encode(lines, out_type=int))
        tpb = ntok / max(1, nbytes)
        row = {
            "lang": lang, "bytes": nbytes, "words": nwords, "tokens": ntok,
            "tokens_per_byte": tpb, "tokens_per_word": ntok / max(1, nwords),
            "chars_per_token": nbytes / max(1, ntok),
        }
        if lang in losses:
            row["loss_tok_nats"] = losses[lang]
            row["bits_per_byte"] = losses[lang] * tpb / math.log(2)
        rows.append(row)
        print(f"[{lang}]")
        for k, v in row.items():
            if k == "lang":
                continue
            print(f"  {k:18s}: {v:,.4f}" if isinstance(v, float) else f"  {k:18s}: {v:,}")
        print()

    # Pairwise fairness (first two langs with losses)
    scored = [r for r in rows if "bits_per_byte" in r]
    if len(scored) >= 2:
        a, b = scored[0], scored[1]
        print("=== fairness analysis ===")
        print(f"fertility ratio ({b['lang']}/{a['lang']} tokens/byte): "
              f"{b['tokens_per_byte']/a['tokens_per_byte']:.3f}")
        print(f"per-token loss ratio ({b['lang']}/{a['lang']}):        "
              f"{b['loss_tok_nats']/a['loss_tok_nats']:.3f}")
        print(f"bits-per-byte: {a['lang']}={a['bits_per_byte']:.3f}  "
              f"{b['lang']}={b['bits_per_byte']:.3f}")
        print(f"TRUE gap ({b['lang']}/{a['lang']} bpb): "
              f"{b['bits_per_byte']/a['bits_per_byte']:.3f}x")
        print("\nReading: if the bpb gap is much smaller than the per-token-loss")
        print("gap, most of the German lag is tokenization, not real difficulty.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
