"""Per-language bits-per-byte — the only fair cross-language loss metric.

Per-token cross-entropy is NOT comparable across languages: the tokenizer splits
them into different numbers of pieces. The canary showed German per-token val
loss 8.283 vs English 5.216, but most of that ratio is tokenization. The fair
metric is bits-per-byte:

    bpb = per_token_nats * (tokens / byte) / ln(2)

This module computes it during training without touching Codex's trainer:

- :func:`bits_per_byte` / :func:`bpb_gap` — pure, unit-tested.
- :func:`measure_tokens_per_byte` — measures fertility from a .bin sample.
- :class:`LanguageBpbEvaluator` — an ``extra_metrics_fn`` for ``LearningMonitor``
  that logs ``val_loss_<lang>``, ``bpb_<lang>`` and ``bpb_gap_max`` each eval.
- :func:`combine_extra_metrics` — compose several extra-metrics callables
  (e.g. the frozen gate + this) into the monitor's single ``extra_metrics_fn``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Sequence

LN2 = math.log(2)


# --------------------------------------------------------------------------
# Pure helpers (torch-free, unit-tested)
# --------------------------------------------------------------------------
def bits_per_byte(loss_tok_nats: float, tokens_per_byte: float) -> float:
    """Convert a per-token NLL (nats) to bits-per-byte."""
    return loss_tok_nats * tokens_per_byte / LN2


def bpb_gap(bpb_by_lang: dict[str, float]) -> float:
    """Worst/best bits-per-byte ratio across languages (1.0 if <2 langs)."""
    vals = [v for v in bpb_by_lang.values() if v > 0]
    if len(vals) < 2:
        return 1.0
    return max(vals) / min(vals)


def combine_extra_metrics(*fns: Callable[[int], dict] | None) -> Callable[[int], dict] | None:
    """Merge several ``extra_metrics_fn`` callables into one.

    The monitor accepts a single ``extra_metrics_fn``; this lets you run, say,
    the frozen gate AND the bpb evaluator. A failing component is isolated so it
    cannot take down the others or the run.
    """
    active = [f for f in fns if f is not None]
    if not active:
        return None
    if len(active) == 1:
        return active[0]

    def _combined(step: int) -> dict:
        out: dict = {}
        for f in active:
            try:
                out.update(f(step) or {})
            except Exception as exc:   # isolate a broken component
                name = getattr(f, "__name__", type(f).__name__)
                out[f"extra_metrics_error_{name}"] = 1.0
                print(f"[bpb/combine] {name} failed at step {step}: {exc}")
        return out

    return _combined


# --------------------------------------------------------------------------
# Fertility measurement (numpy, no torch)
# --------------------------------------------------------------------------
def measure_tokens_per_byte(
    decode: Callable[[list[int]], str],
    bin_path: str | Path,
    sample_tokens: int = 200_000,
    offset_tokens: int = 0,
) -> float:
    """Measure tokens/byte by decoding a sample of a uint32 .bin back to text."""
    import numpy as np

    mm = np.memmap(Path(bin_path), dtype=np.uint32, mode="r")
    n = int(mm.shape[0])
    if n == 0:
        raise ValueError(f"empty bin: {bin_path}")
    take = min(sample_tokens, n)
    lo = min(max(0, offset_tokens), n - take)
    ids = [int(x) for x in mm[lo:lo + take]]
    nbytes = len(decode(ids).encode("utf-8"))
    return take / max(1, nbytes)


# --------------------------------------------------------------------------
# Live evaluator (torch)
# --------------------------------------------------------------------------
class LanguageBpbEvaluator:
    """``extra_metrics_fn`` that reports per-language val loss and bits-per-byte.

    It samples a fixed number of validation batches per language from the
    held-out split, computes the mean per-token loss, and converts it to bpb
    using each language's measured tokens/byte (measured once at construction
    unless provided).
    """

    def __init__(
        self,
        model_adapter,
        tokenizer,
        data_dir: str | Path,
        langs: Sequence[str],
        *,
        seq_length: int,
        val_split_bytes: int,
        batch_size: int = 8,
        batches: int = 20,
        tokens_per_byte: dict[str, float] | None = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 20260530,
    ) -> None:
        from auralis.training.dataset import MixedDataLoader

        self.ma = model_adapter
        self.langs = list(langs)
        self.batch_size = batch_size
        self.batches = batches
        eq = {lang: 1.0 / len(self.langs) for lang in self.langs}
        self.loader = MixedDataLoader(
            data_dir, eq, batch_size, seq_length, seed=seed,
            split="val", val_split_bytes=val_split_bytes,
            rank=rank, world_size=world_size,
        )
        if tokens_per_byte is None:
            tokens_per_byte = {
                lang: measure_tokens_per_byte(tokenizer.decode, Path(data_dir) / f"{lang}.bin")
                for lang in self.langs
            }
        self.tpb = dict(tokens_per_byte)

    def __call__(self, step: int) -> dict:
        from statistics import mean

        from .scoring import canary_loss

        out: dict = {}
        bpbs: dict[str, float] = {}
        for lang in self.langs:
            losses = [
                canary_loss(self.ma.model, *self._batch(lang))
                for _ in range(self.batches)
            ]
            vl = mean(losses)
            out[f"val_loss_{lang}"] = vl
            if lang in self.tpb:
                out[f"bpb_{lang}"] = bits_per_byte(vl, self.tpb[lang])
                bpbs[lang] = out[f"bpb_{lang}"]
        if len(bpbs) >= 2:
            out["bpb_gap_max"] = bpb_gap(bpbs)
        return out

    def _batch(self, lang: str):
        b = self.loader.sample_language(lang, self.batch_size)
        return b["input_ids"], b["labels"]


__all__ = [
    "bits_per_byte",
    "bpb_gap",
    "combine_extra_metrics",
    "measure_tokens_per_byte",
    "LanguageBpbEvaluator",
]
