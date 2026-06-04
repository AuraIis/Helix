#!/usr/bin/env python3
"""Train a cheap edu-quality regressor on LLM 0-5 labels (FineWeb-Edu style).

Loads labeled JSONL (rows with 'text' and integer 'score'), embeds each doc with
a frozen multilingual model, fits a closed-form ridge-regression head, and reports
held-out RMSE/MAE/Pearson + keep/drop precision/recall/F1 at a threshold. Saves a
small artifact (just the regression weights + embedder config) that
score_corpus_edu.py uses to filter the full corpus.

Closed-form ridge (no iterative training, deterministic): w = (XᵀX + λI)⁻¹ Xᵀy,
with the bias term left unregularised. The embedding is the expensive part; the
head is a tiny (dim+1,) vector.

    python scripts/data/train_edu_classifier.py \
        --labels eval/results/de_edu/train/*.jsonl \
        --output eval/results/de_edu/edu_clf.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.data.edu_embed import DEFAULT_MODEL, DEFAULT_PREFIX, EduEmbedder  # noqa: E402


def load_labeled(paths: list[Path]) -> list[tuple[str, int, str]]:
    rows: list[tuple[str, int, str]] = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a truncated final line (e.g. killed mid-write)
                text = r.get("text")
                score = r.get("score")
                if text and score is not None:
                    rows.append((text, int(score), r.get("source") or "?"))
    return rows


def ridge_fit(X: torch.Tensor, y: torch.Tensor, lam: float) -> torch.Tensor:
    """X: (N, D). Returns w: (D+1,) including a trailing (unregularised) bias."""
    X = X.double()
    y = y.double()
    n, d = X.shape
    Xb = torch.cat([X, torch.ones(n, 1, dtype=X.dtype)], dim=1)  # (N, D+1)
    A = Xb.t() @ Xb                                              # (D+1, D+1)
    reg = lam * torch.eye(d + 1, dtype=X.dtype)
    reg[d, d] = 0.0                                             # don't regularise bias
    w = torch.linalg.solve(A + reg, Xb.t() @ y.unsqueeze(1))   # (D+1, 1)
    return w.squeeze(1).float()


def predict(X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    n = X.shape[0]
    Xb = torch.cat([X, torch.ones(n, 1)], dim=1)
    return (Xb @ w).clamp(0, 5)


def metrics(pred: torch.Tensor, true: torch.Tensor, pred_threshold: float,
            label_threshold: float = 3.0) -> dict:
    """Keep/drop is judged against the FIXED rubric bar (true >= label_threshold).
    pred_threshold is the (calibratable) cutoff on the regressor output."""
    err = pred - true
    rmse = float(torch.sqrt((err ** 2).mean()))
    mae = float(err.abs().mean())
    pm, tm = pred - pred.mean(), true - true.mean()
    pear = float((pm @ tm) / (pm.norm() * tm.norm() + 1e-9))
    pk, tk = pred >= pred_threshold, true >= label_threshold
    tp = int((pk & tk).sum()); fp = int((pk & ~tk).sum())
    fn = int((~pk & tk).sum()); tn = int((~pk & ~tk).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / max(1, len(pred))
    return {
        "pred_threshold": round(float(pred_threshold), 3),
        "rmse": round(rmse, 4), "mae": round(mae, 4), "pearson": round(pear, 4),
        "keep_precision": round(prec, 4), "keep_recall": round(rec, 4),
        "keep_f1": round(f1, 4), "accuracy": round(acc, 4),
        "kept_frac_pred": round(float(pk.float().mean()), 4),
        "kept_frac_true": round(float(tk.float().mean()), 4),
    }


def best_threshold(pred: torch.Tensor, true: torch.Tensor, label_threshold: float = 3.0,
                   lo: float = 1.0, hi: float = 4.0, step: float = 0.05) -> float:
    """Pick the regressor-output cutoff that maximises keep-F1 against the rubric
    bar (true >= label_threshold). Counters ridge shrinkage (predictions pulled
    toward the mean would otherwise under-keep at a naive 3.0 cutoff)."""
    tk = true >= label_threshold
    best_t, best_f1 = label_threshold, -1.0
    t = lo
    while t <= hi + 1e-9:
        pk = pred >= t
        tp = int((pk & tk).sum()); fp = int((pk & ~tk).sum()); fn = int((~pk & tk).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if f1 > best_f1:
            best_f1, best_t = f1, round(t, 3)
        t += step
    return best_t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, nargs="+", required=True,
                    help="one or more JSONL files with 'text' + 'score' rows")
    ap.add_argument("--output", type=Path, required=True, help="artifact .pt path")
    ap.add_argument("--emb-model", default=DEFAULT_MODEL)
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--lam", type=float, default=1.0, help="ridge L2 strength")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=20260531)
    args = ap.parse_args()

    rows = load_labeled(args.labels)
    if len(rows) < 20:
        raise SystemExit(f"only {len(rows)} usable labeled rows — need more (check 'text' field)")
    print(f"loaded {len(rows)} labeled docs from {len(args.labels)} file(s)")

    hist = {s: sum(1 for _, sc, _ in rows if sc == s) for s in range(6)}
    print("label histogram:", hist)
    by_src: dict[str, int] = {}
    for _, _, s in rows:
        by_src[s] = by_src.get(s, 0) + 1
    print("by source:", by_src)

    texts = [r[0] for r in rows]
    scores = torch.tensor([r[1] for r in rows], dtype=torch.float32)

    emb = EduEmbedder(args.emb_model, args.prefix, max_length=args.max_length)
    print(f"embedding {len(texts)} docs via {args.emb_model} (dim={emb.dim}) on {emb.device} ...")
    X = emb.embed(texts, batch_size=args.batch_size, progress_every=10)

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(rows), generator=g)
    n_val = max(1, int(len(rows) * args.val_frac))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    Xtr, ytr = X[tr_idx], scores[tr_idx]
    Xval, yval = X[val_idx], scores[val_idx]

    w = ridge_fit(Xtr, ytr, args.lam)
    pred_tr, pred_val = predict(Xtr, w), predict(Xval, w)

    # Naive cutoff = the rubric bar itself (what a hardcoded threshold would do).
    val_raw = metrics(pred_val, yval, args.threshold, args.threshold)

    # Calibrated cutoff: maximise keep-F1 on TRAIN, then report on VAL (no leakage).
    cal_t = best_threshold(pred_tr, ytr, args.threshold)
    val_cal = metrics(pred_val, yval, cal_t, args.threshold)

    # Baseline: always predict the train-mean score (sanity floor for RMSE).
    base_pred = torch.full_like(yval, float(ytr.mean()))
    base_rmse = float(torch.sqrt(((base_pred - yval) ** 2).mean()))

    print(f"\n=== val @ naive cutoff {args.threshold} ===\n{json.dumps(val_raw, indent=2)}")
    print(f"\ncalibrated cutoff (max train keep-F1): {cal_t}")
    print(f"=== val @ calibrated {cal_t} ===\n{json.dumps(val_cal, indent=2)}")
    print(f"\nbaseline val RMSE (predict train-mean): {base_rmse:.4f}  "
          f"(classifier val RMSE {val_cal['rmse']} should be clearly lower)")

    art = {
        "emb_model": args.emb_model, "prefix": args.prefix, "max_length": args.max_length,
        "label_threshold": args.threshold,   # fixed rubric keep bar (true >= this)
        "threshold": cal_t,                   # CALIBRATED regressor cutoff (used by scorer)
        "naive_threshold": args.threshold,
        "lam": args.lam, "dim": emb.dim, "w": w,
        "n_train": len(tr_idx), "n_val": len(val_idx),
        "val_metrics": val_cal, "val_metrics_naive": val_raw, "label_hist": hist,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(art, args.output)
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
