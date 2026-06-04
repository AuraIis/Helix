"""Tests for ``scripts/data/_common.py`` helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data._common import atomic_text_writer, clean_text


def test_clean_text_normalises_whitespace() -> None:
    assert clean_text("foo\n\nbar\t  baz\r\n") == "foo bar baz"
    assert clean_text("   multi   spaces   ") == "multi spaces"


def test_atomic_writer_commits_on_success(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    with atomic_text_writer(target) as fh:
        fh.write("hello\n")
    assert target.read_text(encoding="utf-8") == "hello\n"
    assert not target.with_suffix(".txt.tmp").exists()


def test_atomic_writer_discards_on_exception(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("ORIGINAL", encoding="utf-8")

    class Boom(RuntimeError):
        pass

    with pytest.raises(Boom):
        with atomic_text_writer(target) as fh:
            fh.write("partial")
            raise Boom()

    # Original is preserved, no partial/.tmp left behind.
    assert target.read_text(encoding="utf-8") == "ORIGINAL"
    assert not target.with_suffix(".txt.tmp").exists()


def test_atomic_writer_replaces_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    target.write_text("OLD", encoding="utf-8")

    with atomic_text_writer(target) as fh:
        fh.write("NEW")

    assert target.read_text(encoding="utf-8") == "NEW"
