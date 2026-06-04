"""Pretraining dataset: memmap .bin streams with language-mix batching.

The tokenized corpus from ``tokenize_for_pretraining.py`` lives as flat
``uint32`` files per language (``english.bin``, ``german.bin``, ``code.bin``)
on the NAS. Random-access via ``numpy.memmap`` is cheap even at 80+ GB —
we never load the file, only slice into it.

Two classes:

- :class:`PretrainDataset` — infinite sampler over a single .bin file. Each
  ``__getitem__`` returns ``seq_length+1`` tokens (the extra one is shifted
  inside the model to form labels).
- :class:`MixedDataLoader` — composes multiple PretrainDatasets (one per
  language) and yields batches whose rows are drawn according to
  ``mix_ratios``. Uses a deterministic-per-seed RNG so runs are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


def _mmap_bin(path: Path) -> np.memmap:
    """Open a flat uint32 .bin file as a read-only memmap."""
    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise ValueError(f"empty bin: {path}")
    n_tokens = size_bytes // 4
    return np.memmap(path, dtype=np.uint32, mode="r", shape=(n_tokens,))


def _rank_shard_window(
    start: int,
    end: int,
    *,
    seq_length: int,
    rank: int,
    world_size: int,
    name: str,
) -> tuple[int, int]:
    """Return the per-rank token window for a contiguous corpus slice."""
    if world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
    if world_size == 1:
        return start, end

    span = end - start
    shard_start = start + (span * rank) // world_size
    shard_end = start + (span * (rank + 1)) // world_size
    min_tokens = seq_length + 1
    if shard_end - shard_start < min_tokens:
        raise ValueError(
            f"{name} rank shard [{shard_start}, {shard_end}) has "
            f"{shard_end - shard_start} tokens, need >= seq_length+1 "
            f"({min_tokens}). Reduce world_size or seq_length."
        )
    return shard_start, shard_end


@dataclass
class PretrainDataset:
    """Infinite token-stream sampler over one .bin file.

    Each call to :meth:`sample` draws a random start offset inside the
    ``[train_start, train_end)`` window and returns a contiguous block of
    ``seq_length + 1`` tokens as an ``int64`` tensor.

    The ``train_end`` / ``val_start`` split supports a disjoint validation
    holdout: the last ``val_split_tokens`` tokens of the .bin file are
    reserved for validation and are never yielded by a train-mode sampler.
    """

    bin_path: Path
    seq_length: int
    rng: np.random.Generator
    train_start: int = 0
    train_end: int | None = None          # exclusive; None → all tokens

    def __post_init__(self) -> None:
        self._mmap = _mmap_bin(Path(self.bin_path))
        self._n_tokens = self._mmap.shape[0]
        if self.train_end is None:
            self.train_end = self._n_tokens
        # A window of exactly seq_length+1 tokens is the smallest valid one:
        # sample() picks start ∈ [train_start, train_end - seq_length) which
        # then has exactly one legal offset. Strict `<` keeps the validation
        # consistent with sample()'s upper bound (Codex P3).
        if self.train_end - self.train_start < self.seq_length + 1:
            raise ValueError(
                f"{self.bin_path} window [{self.train_start}, {self.train_end}) has "
                f"{self.train_end - self.train_start} tokens, need >= seq_length+1"
            )

    @property
    def num_tokens(self) -> int:
        return int(self._n_tokens)

    def close(self) -> None:
        mmap = getattr(self, "_mmap", None)
        if mmap is None:
            return
        backing = getattr(mmap, "_mmap", None)
        if backing is not None:
            backing.close()
        self._mmap = None

    def __del__(self) -> None:
        self.close()

    def sample(self) -> torch.Tensor:
        if self._mmap is None:
            raise RuntimeError(f"{self.bin_path} is closed")
        lo = self.train_start
        hi = self.train_end - self.seq_length
        start = int(self.rng.integers(lo, hi))
        block = self._mmap[start : start + self.seq_length + 1].astype(np.int64, copy=True)
        return torch.from_numpy(block)


class MixedDataLoader:
    """Batches tokens from multiple language streams according to mix ratios.

    Yields dicts with ``input_ids`` and ``labels`` of shape
    ``[batch_size, seq_length]``. ``labels`` is just ``input_ids`` shifted
    inside the model's own loss computation, so we simply emit the same
    sequence for both.

    The expected number of rows per language per batch is proportional to
    ``mix_ratios[lang]``. Actual per-batch counts are apportioned across time,
    so low-share languages (for example 5% code with batch size 4) still
    appear regularly instead of being rounded down to zero forever.
    """

    def __init__(
        self,
        data_dir: str | Path,
        mix_ratios: dict[str, float],
        batch_size: int,
        seq_length: int,
        seed: int = 42,
        split: str = "train",                    # "train" | "val"
        val_split_bytes: int = 0,                # last N BYTES of each .bin reserved for val
        rank: int = 0,
        world_size: int = 1,
    ):
        self.data_dir = Path(data_dir)
        self.mix_ratios = dict(mix_ratios)
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.split = split
        self.rank = int(rank)
        self.world_size = int(world_size)
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        if self.world_size < 1:
            raise ValueError(f"world_size must be >= 1, got {self.world_size}")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}), got {self.rank}")

        total = sum(self.mix_ratios.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(f"mix_ratios must sum to 1, got {total}")
        self.mix_ratios = {lang: p / total for lang, p in self.mix_ratios.items()}

        # uint32 = 4 bytes per token
        val_split_tokens = int(val_split_bytes) // 4

        # Dedicated RNG for batch-row shuffling inside __next__, so
        # reproducibility is independent of global torch state.
        rank_seed_offset = self.rank * 1_000_000_007
        self._shuffle_seed = (seed + rank_seed_offset) ^ 0xC0FFEE
        self._shuffle_rng = np.random.default_rng(self._shuffle_seed)

        # Per-language RNGs (distinct seeds so draws do not correlate).
        # Val gets its own offset in the seed so train and val do not align.
        # Seeds are remembered so ``reset_rngs()`` can rewind the val loader to
        # an identical token stream for every eval (see _evaluate): otherwise
        # the stateful RNG advances and each eval samples DIFFERENT val tokens,
        # turning the val trajectory into model-change + sampling-noise.
        self._lang_rng_seeds: dict[str, int] = {}
        self.datasets: dict[str, PretrainDataset] = {}
        for i, lang in enumerate(sorted(self.mix_ratios)):
            bin_path = self.data_dir / f"{lang}.bin"
            if not bin_path.exists():
                raise FileNotFoundError(bin_path)
            n_tokens_total = bin_path.stat().st_size // 4

            # --- Hard split validation (fail loudly, never silently repair) ---
            if val_split_tokens < 0:
                raise ValueError(f"val_split_bytes must be >= 0, got {val_split_bytes}")
            if val_split_tokens >= n_tokens_total:
                raise ValueError(
                    f"val_split_bytes ({val_split_bytes}) >= {lang}.bin size "
                    f"({n_tokens_total*4}). Would leave zero or negative tokens "
                    f"for training."
                )
            train_window = n_tokens_total - val_split_tokens
            if train_window <= seq_length + 1:
                raise ValueError(
                    f"val_split_bytes={val_split_bytes} leaves only "
                    f"{train_window} train tokens for {lang}.bin, need > "
                    f"seq_length+1 ({seq_length + 1}). Reduce val_split_bytes."
                )
            # Val must also have room for at least one seq+1 block (checked below).

            if split == "train":
                train_start, train_end = _rank_shard_window(
                    0,
                    train_window,
                    seq_length=seq_length,
                    rank=self.rank,
                    world_size=self.world_size,
                    name=f"{lang}.bin train",
                )
                rng_seed = seed + i * 7919 + rank_seed_offset
                rng = np.random.default_rng(rng_seed)
            else:
                train_start, train_end = _rank_shard_window(
                    train_window,
                    n_tokens_total,
                    seq_length=seq_length,
                    rank=self.rank,
                    world_size=self.world_size,
                    name=f"{lang}.bin val",
                )
                if train_end - train_start < seq_length + 1:
                    raise ValueError(
                        f"val split for {lang} too small: {train_end - train_start} tokens "
                        f"(need >= seq_length+1={seq_length+1}). Increase val_split_bytes."
                    )
                rng_seed = seed + i * 7919 + rank_seed_offset + 1_000_003
                rng = np.random.default_rng(rng_seed)
            self._lang_rng_seeds[lang] = rng_seed

            # Invariant: train/val windows must be disjoint. With train_end ==
            # train_window == val_start this is exactly adjacency, no overlap.
            assert train_start < train_end, (train_start, train_end)
            self.datasets[lang] = PretrainDataset(
                bin_path=bin_path, seq_length=seq_length, rng=rng,
                train_start=train_start, train_end=train_end,
            )

        # Expected rows-per-batch plus a carried deficit/surplus so small
        # ratios are scheduled fairly across batches rather than rounded away.
        self._lang_order = list(sorted(self.mix_ratios))
        self._expected_rows_per_lang = {
            lang: self.batch_size * self.mix_ratios[lang] for lang in self._lang_order
        }
        self._row_credit = {lang: 0.0 for lang in self._lang_order}

    def reset_rngs(self) -> None:
        """Rewind every RNG to its construction seed.

        Call this at the START of each evaluation so the val loader yields the
        IDENTICAL token stream every time. Without it, ``sample()`` advances a
        stateful RNG, so eval@250 and eval@500 see different val tokens and the
        loss trajectory mixes real model change with ~1σ sampling noise. With
        it, the trajectory is apples-to-apples. Harmless on the train loader,
        but only ever called on the val loader.
        """
        self._shuffle_rng = np.random.default_rng(self._shuffle_seed)
        for lang, ds in self.datasets.items():
            ds.rng = np.random.default_rng(self._lang_rng_seeds[lang])
        self._row_credit = {lang: 0.0 for lang in self._lang_order}

    def _allocate_rows_for_batch(self) -> dict[str, int]:
        """Allocate concrete row counts for the next batch.

        Each batch adds the language's expected row budget into a running
        credit balance. We then emit all whole rows that are available and
        distribute any remainder to the languages with the largest residual
        credit. This preserves the target mix over time while ensuring that
        positive-weight languages are not starved by small micro-batches.
        """
        for lang in self._lang_order:
            self._row_credit[lang] += self._expected_rows_per_lang[lang]

        rows = {lang: max(0, int(self._row_credit[lang])) for lang in self._lang_order}
        remaining = self.batch_size - sum(rows.values())
        if remaining:
            frac = sorted(
                self._lang_order,
                key=lambda lang: (-(self._row_credit[lang] - rows[lang]), lang),
            )
            for lang in frac[:remaining]:
                rows[lang] += 1

        for lang in self._lang_order:
            self._row_credit[lang] -= rows[lang]
        return rows

    @property
    def rows_per_language(self) -> dict[str, float]:
        """Expected rows per batch, not the concrete count of a single batch."""
        return dict(self._expected_rows_per_lang)

    def sample_language(self, lang: str, batch_size: int) -> dict[str, torch.Tensor]:
        """Draw a pure-single-language batch (used for per-language val_loss).

        Bypasses mix_ratios — caller picks the language, the sampler still
        uses the language's own RNG so repeated calls do not collide.
        """
        if lang not in self.datasets:
            raise KeyError(f"unknown language {lang!r}; have {list(self.datasets)}")
        ds = self.datasets[lang]
        rows = [ds.sample() for _ in range(batch_size)]
        batch = torch.stack(rows, dim=0)
        input_ids = batch[:, :-1].contiguous()
        return {"input_ids": input_ids, "labels": input_ids.clone()}

    def __iter__(self):
        return self

    def __next__(self) -> dict[str, torch.Tensor]:
        rows: list[torch.Tensor] = []
        batch_rows = self._allocate_rows_for_batch()
        for lang, n in batch_rows.items():
            ds = self.datasets[lang]
            for _ in range(n):
                rows.append(ds.sample())
        # Shuffle so a batch does not begin with all of one language. Uses the
        # loader-owned RNG (seeded at construction) so a rerun with the same
        # seed produces byte-identical batches — independent of the global
        # torch RNG state.
        self._shuffle_rng.shuffle(rows)
        batch = torch.stack(rows, dim=0)
        # Drop the extra sampled token — HelixModel._shift_loss shifts labels
        # internally, so both tensors are the same length and the same content.
        # Passing ``labels = input_ids`` (not a pre-shifted copy) avoids the
        # classic "off-by-two" bug where loss accidentally predicts t+2 from t.
        input_ids = batch[:, :-1].contiguous()
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels}

    def close(self) -> None:
        for ds in getattr(self, "datasets", {}).values():
            ds.close()

    def __del__(self) -> None:
        self.close()


__all__ = ["MixedDataLoader", "PretrainDataset"]
