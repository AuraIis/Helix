"""Tests for ``scripts/data/filter_quality.py``."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.filter_quality import _normalise, _passes


def test_normalise_collapses_whitespace_for_text() -> None:
    assert _normalise("foo\n\nbar\t baz\r\n", preserve_newlines=False) == "foo bar baz"


def test_normalise_preserves_code_newlines_mode() -> None:
    assert _normalise("print('x')\r\n", preserve_newlines=True) == "print('x') "


def test_boilerplate_is_rejected() -> None:
    line = "Please accept all cookies before reading the rest of this article."
    assert _passes(
        line,
        min_length=10,
        max_length=1000,
        preserve_newlines=False,
        allow_mojibake=False,
    ) == "boilerplate"


def test_repetitive_text_is_rejected() -> None:
    line = "spam " * 30
    assert _passes(
        line,
        min_length=10,
        max_length=1000,
        preserve_newlines=False,
        allow_mojibake=False,
    ) == "repetitive"


def test_clean_line_passes() -> None:
    line = "This is a clean sentence about machine learning and data curation."
    assert _passes(
        line,
        min_length=10,
        max_length=1000,
        preserve_newlines=False,
        allow_mojibake=False,
    ) is None
