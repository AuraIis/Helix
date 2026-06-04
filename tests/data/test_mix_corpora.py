"""Tests for ``scripts/data/mix_corpora.py``."""

from __future__ import annotations

import io
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.mix_corpora import _copy_budget


def test_copy_budget_stops_at_whole_lines(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("aaaa\nbbbb\ncccc\n", encoding="utf-8")
    out = io.StringIO()

    lines_written, bytes_written = _copy_budget(source, out, target_bytes=len("aaaa\nbbbb\n".encode("utf-8")))

    assert lines_written == 2
    assert bytes_written == len("aaaa\nbbbb\n".encode("utf-8"))
    assert out.getvalue() == "aaaa\nbbbb\n"


def test_copy_budget_writes_first_line_even_if_target_is_tiny(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("abcdef\nsecond\n", encoding="utf-8")
    out = io.StringIO()

    lines_written, bytes_written = _copy_budget(source, out, target_bytes=1)

    assert lines_written == 1
    assert bytes_written == len("abcdef\n".encode("utf-8"))
    assert out.getvalue() == "abcdef\n"
