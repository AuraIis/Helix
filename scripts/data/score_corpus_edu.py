#!/usr/bin/env python3
"""Score a full corpus with the trained edu classifier and keep docs >= threshold.

Streams a line-per-doc text file, embeds in batches with the SAME frozen model
used at train time (read from the artifact), predicts a 0-5 score per doc, and
writes the kept docs to --output plus a .manifest.json with the score histogram,
keep rate, and mean predicted score.

    python scripts/data/score_corpus_edu.py \
        --input cleaned/fineweb2_de.filtered.txt \
        --artifact eval/results/de_edu/edu_clf.pt \
        --output cleaned/edu/fineweb2_de.edu.txt

Big files take a while (GPU-bound on the embedder). Use --limit to dry-run a slice.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.data.edu_embed import EduEmbedder  # noqa: E402


def predict(X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    n = X.shape[0]
    Xb = torch.cat([X, torch.ones(n, 1)], dim=1)
    return (Xb @ w).clamp(0, 5)


def clip_excerpt(text: str, cap: int) -> tuple[str, bool]:
    """Cut at a word boundary (not mid-word) and mark it, so the human labeling
    UI shows a clean excerpt instead of a string that reads as broken. The kept
    corpus still stores the FULL doc; this only affects the review-pool preview."""
    if len(text) <= cap:
        return text, False
    head = text[:cap]
    sp = head.rfind(" ")
    if sp >= cap - 80:   # only back off to the last space if we lose little
        head = head[:sp]
    return head.rstrip() + " […]", True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True, help="kept docs (>= threshold)")
    ap.add_argument("--artifact", type=Path, required=True, help="edu_clf.pt from train_edu_classifier.py")
    ap.add_argument("--threshold", type=float, default=None, help="override artifact threshold")
    ap.add_argument("--min-length", type=int, default=200, help="skip lines shorter than this (chars)")
    ap.add_argument("--max-chars", type=int, default=2000, help="truncate before embedding (match training)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="stop after N scored docs (0 = all)")
    ap.add_argument("--skip-lines", type=int, default=0,
                    help="skip the first N raw lines (probe a deeper, less position-biased slice)")
    ap.add_argument("--scores-jsonl", type=Path, default=None,
                    help="optional: write {len, score} per scored doc for inspection")
    ap.add_argument("--review-pool", type=Path, default=None,
                    help="optional: write borderline docs {text, model_score, source} for human "
                         "review in the data-game app (active learning on the decision boundary)")
    ap.add_argument("--review-band", type=float, nargs=2, default=(1.5, 3.0), metavar=("LO", "HI"),
                    help="predicted-score band counted as 'borderline' for the review pool")
    ap.add_argument("--review-max", type=int, default=3000, help="cap on review-pool docs")
    args = ap.parse_args()

    art = torch.load(args.artifact, map_location="cpu", weights_only=False)
    w = art["w"].float()
    threshold = args.threshold if args.threshold is not None else float(art["threshold"])
    emb = EduEmbedder(art["emb_model"], art["prefix"], max_length=art["max_length"])
    print(f"artifact: emb={art['emb_model']} dim={art['dim']} threshold={threshold} "
          f"(val_metrics={art.get('val_metrics')})")

    hist = {s: 0 for s in range(6)}
    lines_in = kept = scored = 0
    score_sum = 0.0
    t0 = time.monotonic()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_fh = args.output.open("w", encoding="utf-8")
    sc_fh = args.scores_jsonl.open("w", encoding="utf-8") if args.scores_jsonl else None
    review_fh = None
    if args.review_pool:
        args.review_pool.parent.mkdir(parents=True, exist_ok=True)
        review_fh = args.review_pool.open("w", encoding="utf-8")
    r_lo, r_hi = args.review_band
    review_written = 0

    buf_orig: list[str] = []   # original lines (written verbatim if kept)
    buf_text: list[str] = []   # truncated text fed to the embedder

    def flush() -> None:
        nonlocal kept, scored, score_sum, review_written
        if not buf_text:
            return
        X = emb.embed(buf_text, batch_size=args.batch_size)
        preds = predict(X, w)
        for orig, pr in zip(buf_orig, preds.tolist()):
            scored += 1
            score_sum += pr
            hist[int(round(pr))] = hist.get(int(round(pr)), 0) + 1
            if sc_fh:
                sc_fh.write(json.dumps({"len": len(orig), "score": round(pr, 3)}) + "\n")
            if (review_fh is not None and review_written < args.review_max
                    and r_lo <= pr <= r_hi):
                excerpt, was_trunc = clip_excerpt(orig, args.max_chars)
                review_fh.write(json.dumps(
                    {"text": excerpt, "model_score": round(pr, 3),
                     "truncated": was_trunc, "orig_len": len(orig),
                     "source": args.input.name, "source_line": scored},
                    ensure_ascii=False) + "\n")
                review_written += 1
            if pr >= threshold:
                out_fh.write(orig + "\n")
                kept += 1
        buf_orig.clear()
        buf_text.clear()

    try:
        with args.input.open("r", encoding="utf-8", errors="replace") as fh:
            skipped = 0
            for line in fh:
                if skipped < args.skip_lines:
                    skipped += 1
                    continue
                lines_in += 1
                stripped = line.strip()
                if len(stripped) < args.min_length:
                    continue
                buf_orig.append(stripped)
                buf_text.append(stripped[: args.max_chars])
                if len(buf_text) >= args.batch_size:
                    flush()
                    if scored % (args.batch_size * 20) == 0:
                        rate = scored / max(1e-9, time.monotonic() - t0)
                        print(f"  scored {scored} | kept {kept} ({100*kept/max(1,scored):.1f}%) "
                              f"| {rate:.0f} docs/s", flush=True)
                if args.limit and scored + len(buf_text) >= args.limit:
                    break
            flush()
    finally:
        out_fh.close()
        if sc_fh:
            sc_fh.close()
        if review_fh:
            review_fh.close()

    keep_rate = kept / max(1, scored)
    mean_score = score_sum / max(1, scored)
    manifest = {
        "input": str(args.input), "output": str(args.output),
        "artifact": str(args.artifact), "threshold": threshold,
        "lines_in": lines_in, "scored": scored, "kept": kept,
        "keep_rate": round(keep_rate, 4), "mean_score": round(mean_score, 4),
        "review_pool_written": review_written,
        "score_hist": hist, "elapsed_s": round(time.monotonic() - t0, 1),
    }
    man_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== {args.input.name} ===")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"wrote {args.output}\nwrote {man_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
