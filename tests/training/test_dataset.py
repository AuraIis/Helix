"""Tests for MixedDataLoader + PretrainDataset (memmap-based)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from auralis.training.dataset import MixedDataLoader, PretrainDataset


def _write_bin(path: Path, n_tokens: int, rng: np.random.Generator) -> None:
    arr = rng.integers(0, 200_000, size=n_tokens, dtype=np.uint32)
    arr.tofile(path)


@pytest.fixture
def three_bins(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    for lang, n in [("english", 20_000), ("german", 10_000), ("code", 5_000)]:
        _write_bin(tmp_path / f"{lang}.bin", n, rng)
    return tmp_path


def test_pretrain_dataset_sample_shape(three_bins: Path):
    ds = PretrainDataset(
        bin_path=three_bins / "english.bin",
        seq_length=128,
        rng=np.random.default_rng(1),
    )
    sample = ds.sample()
    assert sample.shape == (129,)
    assert sample.dtype == torch.int64


def test_pretrain_dataset_raises_on_short_bin(tmp_path: Path):
    p = tmp_path / "tiny.bin"
    np.zeros(10, dtype=np.uint32).tofile(p)
    with pytest.raises(ValueError):
        PretrainDataset(bin_path=p, seq_length=64, rng=np.random.default_rng(0))


def test_pretrain_dataset_can_sample_last_valid_window(tmp_path: Path):
    p = tmp_path / "ordered.bin"
    np.arange(10, dtype=np.uint32).tofile(p)
    ds = PretrainDataset(bin_path=p, seq_length=3, rng=np.random.default_rng(123))

    seen_starts: set[int] = set()
    for _ in range(512):
        sample = ds.sample()
        seen_starts.add(int(sample[0].item()))

    assert seen_starts == set(range(7))


def test_pretrain_dataset_close_releases_mmap(tmp_path: Path):
    p = tmp_path / "ordered.bin"
    np.arange(10, dtype=np.uint32).tofile(p)
    ds = PretrainDataset(bin_path=p, seq_length=3, rng=np.random.default_rng(0))
    ds.close()

    with pytest.raises(RuntimeError, match="closed"):
        ds.sample()


def test_mixed_dataloader_reports_expected_rows_per_language(three_bins: Path):
    loader = MixedDataLoader(
        data_dir=three_bins,
        mix_ratios={"english": 0.70, "german": 0.25, "code": 0.05},
        batch_size=16,
        seq_length=64,
    )
    rows = loader.rows_per_language
    assert sum(rows.values()) == pytest.approx(16.0)
    assert rows["english"] == pytest.approx(11.2)
    assert rows["german"] == pytest.approx(4.0)
    assert rows["code"] == pytest.approx(0.8)


def test_mixed_dataloader_small_batch_mix_is_preserved_over_time(three_bins: Path):
    loader = MixedDataLoader(
        data_dir=three_bins,
        mix_ratios={"english": 0.70, "german": 0.25, "code": 0.05},
        batch_size=4,
        seq_length=64,
        seed=123,
    )

    sample_counts = {lang: 0 for lang in loader.datasets}
    for lang, ds in loader.datasets.items():
        original_sample = ds.sample

        def counted_sample(lang: str = lang, original_sample=original_sample):
            sample_counts[lang] += 1
            return original_sample()

        ds.sample = counted_sample  # type: ignore[method-assign]

    for _ in range(20):
        next(loader)

    assert sample_counts == {"code": 4, "english": 56, "german": 20}


def test_mixed_dataloader_batch_shape_and_types(three_bins: Path):
    loader = MixedDataLoader(
        data_dir=three_bins,
        mix_ratios={"english": 0.5, "german": 0.5, "code": 0.0},
        batch_size=4,
        seq_length=32,
    )
    batch = next(loader)
    assert batch["input_ids"].shape == (4, 32)
    assert batch["labels"].shape == (4, 32)
    assert batch["input_ids"].dtype == torch.int64


def test_mixed_dataloader_labels_equal_input_ids(three_bins: Path):
    """MixedDataLoader passes unshifted labels; HelixModel shifts internally."""
    loader = MixedDataLoader(
        data_dir=three_bins,
        mix_ratios={"english": 1.0, "german": 0.0, "code": 0.0},
        batch_size=2,
        seq_length=16,
    )
    batch = next(loader)
    assert torch.equal(batch["input_ids"], batch["labels"])


def test_mixed_dataloader_mix_ratios_must_sum_to_one(three_bins: Path):
    with pytest.raises(ValueError):
        MixedDataLoader(
            data_dir=three_bins,
            mix_ratios={"english": 0.5, "german": 0.2, "code": 0.1},
            batch_size=4,
            seq_length=16,
        )


def test_mixed_dataloader_missing_bin_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        MixedDataLoader(
            data_dir=tmp_path,
            mix_ratios={"english": 1.0},
            batch_size=2,
            seq_length=8,
        )


def test_mixed_dataloader_train_val_split_disjoint(three_bins: Path):
    """Train and val should sample from disjoint regions of the same .bin."""
    val_bytes = 16_000
    train = MixedDataLoader(
        data_dir=three_bins,
        mix_ratios={"english": 0.5, "german": 0.5, "code": 0.0},
        batch_size=8,
        seq_length=32,
        seed=0,
        split="train",
        val_split_bytes=val_bytes,
    )
    val = MixedDataLoader(
        data_dir=three_bins,
        mix_ratios={"english": 0.5, "german": 0.5, "code": 0.0},
        batch_size=8,
        seq_length=32,
        seed=0,
        split="val",
        val_split_bytes=val_bytes,
    )
    en_train = train.datasets["english"]
    en_val = val.datasets["english"]
    assert en_train.train_end <= en_val.train_start
    assert en_val.train_end > en_val.train_start
    a = train.datasets["english"].sample()
    b = val.datasets["english"].sample()
    assert not torch.equal(a, b)


def test_mixed_dataloader_val_too_small_raises(three_bins: Path):
    with pytest.raises(ValueError, match="val split for"):
        MixedDataLoader(
            data_dir=three_bins,
            mix_ratios={"english": 1.0, "german": 0.0, "code": 0.0},
            batch_size=2,
            seq_length=64,
            seed=0,
            split="val",
            val_split_bytes=100,
        )


def test_mixed_dataloader_split_name_validated(three_bins: Path):
    with pytest.raises(ValueError, match="split must be"):
        MixedDataLoader(
            data_dir=three_bins,
            mix_ratios={"english": 1.0, "german": 0.0, "code": 0.0},
            batch_size=2,
            seq_length=16,
            split="holdout",
        )


def test_mixed_dataloader_val_split_too_big_raises(three_bins: Path):
    """val_split_bytes that would leave 0 train tokens must hard fail."""
    with pytest.raises(ValueError, match="val_split_bytes"):
        MixedDataLoader(
            data_dir=three_bins,
            mix_ratios={"english": 1.0, "german": 0.0, "code": 0.0},
            batch_size=2,
            seq_length=16,
            split="train",
            val_split_bytes=80_000,
        )


def test_mixed_dataloader_val_split_bytes_negative_rejected(three_bins: Path):
    with pytest.raises(ValueError, match=">= 0"):
        MixedDataLoader(
            data_dir=three_bins,
            mix_ratios={"english": 1.0, "german": 0.0, "code": 0.0},
            batch_size=2,
            seq_length=16,
            split="train",
            val_split_bytes=-4,
        )


def test_mixed_dataloader_shuffle_is_deterministic(three_bins: Path):
    """Same seed means byte-identical first batch."""
    kw = dict(
        data_dir=three_bins,
        mix_ratios={"english": 0.5, "german": 0.5, "code": 0.0},
        batch_size=6,
        seq_length=16,
        seed=1234,
    )
    a = MixedDataLoader(**kw)
    b = MixedDataLoader(**kw)
    assert torch.equal(next(a)["input_ids"], next(b)["input_ids"])


def test_mixed_dataloader_rank_shards_are_disjoint(tmp_path: Path):
    np.arange(1_000, dtype=np.uint32).tofile(tmp_path / "english.bin")
    kw = dict(
        data_dir=tmp_path,
        mix_ratios={"english": 1.0},
        batch_size=8,
        seq_length=16,
        seed=123,
        world_size=2,
    )
    rank0 = MixedDataLoader(**kw, rank=0)
    rank1 = MixedDataLoader(**kw, rank=1)

    for _ in range(8):
        b0 = next(rank0)["input_ids"]
        b1 = next(rank1)["input_ids"]
        assert int(b0.max().item()) < rank0.datasets["english"].train_end
        assert int(b1.min().item()) >= rank1.datasets["english"].train_start
        assert rank0.datasets["english"].train_end <= rank1.datasets["english"].train_start


def test_mixed_dataloader_rank_validation(three_bins: Path):
    with pytest.raises(ValueError, match="rank must be"):
        MixedDataLoader(
            data_dir=three_bins,
            mix_ratios={"english": 1.0, "german": 0.0, "code": 0.0},
            batch_size=2,
            seq_length=16,
            rank=2,
            world_size=2,
        )


def test_mixed_dataloader_rank_shard_too_small_raises(tmp_path: Path):
    np.arange(100, dtype=np.uint32).tofile(tmp_path / "english.bin")
    with pytest.raises(ValueError, match="rank shard"):
        MixedDataLoader(
            data_dir=tmp_path,
            mix_ratios={"english": 1.0},
            batch_size=2,
            seq_length=64,
            rank=1,
            world_size=2,
        )


def test_sample_language_bypasses_mix_ratios(three_bins: Path):
    loader = MixedDataLoader(
        data_dir=three_bins,
        mix_ratios={"english": 0.5, "german": 0.5, "code": 0.0},
        batch_size=4,
        seq_length=16,
        seed=0,
    )
    en_batch = loader.sample_language("english", batch_size=2)
    assert en_batch["input_ids"].shape == (2, 16)
    with pytest.raises(KeyError):
        loader.sample_language("klingon", batch_size=1)
