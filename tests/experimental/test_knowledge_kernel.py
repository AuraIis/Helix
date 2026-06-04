from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.experimental.knowledge_kernel import (  # noqa: E402
    current_block,
    future_block,
    plain_block,
    qa_rows,
    sample_entries,
)


def test_sample_entries_are_valid_and_have_qa() -> None:
    entries = sample_entries()

    assert len(entries) >= 5
    assert len(qa_rows(entries)) >= 8
    for entry in entries:
        entry.validate()


def test_current_block_uses_existing_memory_tags() -> None:
    block = current_block(sample_entries()[0])

    assert block.startswith("<memory>")
    assert block.endswith("</memory>")
    assert "Typ: definition" in block
    assert "Begriff: Photosynthese" in block


def test_future_block_uses_planned_dna_tags() -> None:
    block = future_block(sample_entries()[0])

    assert "<|definition|>" in block
    assert "<|fact|>" in block
    assert "<|example|>" in block
    assert "<|source|>" in block
    assert block.endswith("<|end|>")


def test_plain_baseline_contains_same_core_fact() -> None:
    entry = sample_entries()[1]

    plain = plain_block(entry)
    structured = current_block(entry)

    assert "Berlin ist die Hauptstadt" in plain
    assert "Berlin ist die Hauptstadt" in structured

