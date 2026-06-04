from __future__ import annotations

import sys
from pathlib import Path

import sentencepiece as spm
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.experimental.knowledge_kernel import qa_rows, sample_entries  # noqa: E402
from scripts.experimental.knowledge_kernel_ablation import (  # noqa: E402
    encode_answer_loss_batch,
    make_lm_batch,
)


def test_make_lm_batch_shapes() -> None:
    sp = spm.SentencePieceProcessor(model_file=str(REPO_ROOT / "tokenizer/helix_v2_tokenizer.model"))

    batch = make_lm_batch(
        sp,
        "Berlin ist die Hauptstadt Deutschlands.\n",
        seq_len=16,
        repeat=20,
        device=torch.device("cpu"),
    )

    assert batch["input_ids"].shape[1] == 16
    assert batch["labels"].shape == batch["input_ids"].shape


def test_make_lm_batch_can_cap_rows() -> None:
    sp = spm.SentencePieceProcessor(model_file=str(REPO_ROOT / "tokenizer/helix_v2_tokenizer.model"))

    batch = make_lm_batch(
        sp,
        "Berlin ist die Hauptstadt Deutschlands.\n",
        seq_len=8,
        repeat=50,
        max_rows=3,
        device=torch.device("cpu"),
    )

    assert batch["input_ids"].shape == (3, 8)


def test_answer_loss_batch_masks_prompt() -> None:
    sp = spm.SentencePieceProcessor(model_file=str(REPO_ROOT / "tokenizer/helix_v2_tokenizer.model"))
    rows = qa_rows(sample_entries())[:2]

    batch = encode_answer_loss_batch(sp, rows, style="kernel", device=torch.device("cpu"))

    assert batch["input_ids"].shape == batch["labels"].shape
    assert torch.any(batch["labels"] == -100)
    assert torch.any(batch["labels"] != -100)
