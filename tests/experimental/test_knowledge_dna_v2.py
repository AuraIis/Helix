from __future__ import annotations

import sys
from pathlib import Path

import sentencepiece as spm
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.experimental.knowledge_dna_v2 import (  # noqa: E402
    CURRENT_SPECIALS,
    dna_block,
    hybrid_block,
    plain_block,
    probe_rows,
    sample_entries,
    tokenizer_audit,
    variant_texts,
)
from scripts.experimental.knowledge_dna_v2_ablation import (  # noqa: E402
    encode_probe_loss_batch,
    make_lm_batch,
    score_generation,
    variant_prompt,
)


def test_sample_entries_have_counterfacts_and_transfer_probes() -> None:
    entries = sample_entries()
    rows = probe_rows(entries)

    assert len(entries) >= 5
    assert any(row["kind"] == "counterfact" for row in rows)
    assert any(row["kind"] == "transfer" for row in rows)
    for entry in entries:
        entry.validate()


def test_variants_share_core_fact_but_use_different_shapes() -> None:
    entry = next(row for row in sample_entries() if row.term == "Berlin")

    plain = plain_block(entry)
    dna = dna_block(entry)
    hybrid = hybrid_block(entry)

    assert "Berlin ist die Hauptstadt" in plain
    assert "Berlin ist die Hauptstadt" in dna
    assert "Berlin ist die Hauptstadt" in hybrid
    assert "<memory>" not in plain
    assert "<memory>" in dna
    assert "<recall>" in dna
    assert "<memory>" in hybrid
    assert "<recall>" in hybrid


def test_variant_texts_include_plain_dna_and_hybrid() -> None:
    texts = variant_texts(sample_entries())

    assert set(texts) == {"plain", "dna", "hybrid"}
    assert "<memory>" not in texts["plain"]
    assert "<memory>" in texts["dna"]
    assert "<memory>" in texts["hybrid"]
    assert len(texts["hybrid"]) > len(texts["plain"])


def test_tokenizer_audit_uses_only_current_specials() -> None:
    texts = variant_texts(sample_entries())
    audit = tokenizer_audit(REPO_ROOT / "tokenizer/helix_v2_tokenizer.model", texts)

    assert audit["available"]
    assert set(audit["special_tokens"]) == set(CURRENT_SPECIALS)
    assert audit["special_tokens"]["<memory>"]["registered"]
    assert audit["texts"]["hybrid"]["tokens"] > audit["texts"]["plain"]["tokens"]


def test_probe_loss_batch_masks_prompt_and_keeps_answer() -> None:
    sp = spm.SentencePieceProcessor(model_file=str(REPO_ROOT / "tokenizer/helix_v2_tokenizer.model"))
    rows = probe_rows(sample_entries())[:3]

    batch = encode_probe_loss_batch(sp, rows, variant="hybrid", device=torch.device("cpu"))

    assert batch["input_ids"].shape == batch["labels"].shape
    assert torch.any(batch["labels"] == -100)
    assert torch.any(batch["labels"] != -100)


def test_lm_batches_can_be_capped_for_fair_comparison() -> None:
    sp = spm.SentencePieceProcessor(model_file=str(REPO_ROOT / "tokenizer/helix_v2_tokenizer.model"))
    texts = variant_texts(sample_entries())

    batches = [
        make_lm_batch(
            sp,
            text,
            seq_len=32,
            repeat=20,
            max_rows=4,
            device=torch.device("cpu"),
        )
        for text in texts.values()
    ]

    assert {tuple(batch["input_ids"].shape) for batch in batches} == {(4, 32)}


def test_generation_scoring_detects_forbidden_and_tag_echo() -> None:
    row = {
        "term": "Berlin",
        "question": "Was ist die Hauptstadt Deutschlands?",
        "answer": "Berlin.",
        "kind": "fact",
        "aliases": ["Berlin"],
        "forbidden": ["Frankfurt"],
    }

    metric = score_generation(row, "<memory> Berlin bei Frankfurt")

    assert metric.matched
    assert metric.forbidden_hit
    assert metric.tag_echo
    assert variant_prompt(row, "hybrid").startswith("<recall>")
