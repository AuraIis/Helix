from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.experimental.memory_kernel import (  # noqa: E402
    ChatMessage,
    MemoryExtractor,
    build_adapter_pairs,
    build_kernel_corpus,
)


def test_extracts_trainable_preferences_but_not_temporary_status() -> None:
    messages = [
        ChatMessage(
            role="user",
            content=(
                "Ich möchte, dass Auralis auf Deutsch direkt antwortet. "
                "Trainingsdaten sollen auf /mnt/disk5 liegen."
            ),
        ),
        ChatMessage(
            role="assistant",
            content="Status: clean-v3.1 läuft gerade noch an german_legacy.",
        ),
    ]

    memories = MemoryExtractor(chat_id="t").extract(messages)

    assert any(m.type == "user_preference" and m.train_into_adapter for m in memories)
    assert any(m.type == "path_fact" and not m.train_into_adapter for m in memories)
    assert any(m.type == "temporary_status" and not m.train_into_adapter for m in memories)
    assert not any(
        m.type == "hardware_fact" and "/mnt/disk5" in m.content for m in memories
    )


def test_build_adapter_pairs_uses_only_trainable_memories() -> None:
    messages = [
        ChatMessage(role="user", content="Ich will eine tägliche DoRA Memory Pipeline."),
        ChatMessage(role="assistant", content="Der Download ist heute fertig."),
    ]
    memories = MemoryExtractor(chat_id="t").extract(messages)

    pairs = build_adapter_pairs(memories)

    assert pairs
    assert all("Download ist heute fertig" not in pair["output"] for pair in pairs)
    assert any("DoRA" in pair["output"] for pair in pairs)


def test_kernel_corpus_uses_special_memory_tags() -> None:
    memories = MemoryExtractor(chat_id="t").extract(
        [ChatMessage(role="user", content="Ich möchte Auralis mit LoRA personalisieren.")]
    )

    corpus = "\n".join(build_kernel_corpus(memories))

    assert "<|preference|>" in corpus
    assert "<|end|>" in corpus
    assert "LoRA" in corpus


def test_memory_policy_about_temporary_status_is_trainable() -> None:
    memories = MemoryExtractor(chat_id="t").extract(
        [
            ChatMessage(
                role="user",
                content=(
                    "Für User-Chats will ich tägliche Memory-Zusammenfassungen in "
                    "LoRA oder DoRA trainieren, aber temporären Status nicht dauerhaft."
                ),
            )
        ]
    )

    assert any(m.type == "user_preference" and m.train_into_adapter for m in memories)
    assert not any(m.type == "temporary_status" for m in memories)
