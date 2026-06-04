"""Guardrails for the v1 prompt-format bug.

These tests intentionally pin the byte-level output so that any future change
to the chat template is an *explicit* change with a test update, never an
accidental drift between training and inference paths.
"""

from __future__ import annotations

import pytest

from auralis.tokenizer.chat_template import (
    ASSISTANT_OPEN,
    END,
    SYSTEM_OPEN,
    USER_OPEN,
    ChatMessage,
    build_chat_prompt,
    build_inference_prompt,
    build_training_prompt,
)


# --- 1. Training vs. inference paths must be identical where they overlap ---

def test_training_and_inference_agree_on_shared_prefix() -> None:
    """The shared prefix (system + user turns) must be byte-identical."""
    base = [
        {"role": "system", "content": "Du bist Helix."},
        {"role": "user", "content": "Hallo"},
    ]
    inference = build_inference_prompt(base)
    training = build_training_prompt([*base, {"role": "assistant", "content": "Hi!"}])

    shared_prefix_len = len(inference) - len(f"{ASSISTANT_OPEN}\n")
    assert training[:shared_prefix_len] == inference[:shared_prefix_len], (
        f"Shared prefix diverges!\n"
        f"Training : {training[:shared_prefix_len]!r}\n"
        f"Inference: {inference[:shared_prefix_len]!r}"
    )


def test_inference_ends_exactly_with_open_assistant() -> None:
    messages = [{"role": "user", "content": "Hi"}]
    prompt = build_inference_prompt(messages)
    assert prompt.endswith(f"{ASSISTANT_OPEN}\n")


def test_training_includes_end_after_assistant_turn() -> None:
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]
    prompt = build_training_prompt(messages)
    assert prompt.endswith(f"{END}\n")


# --- 2. Default system prompt is auto-injected, consistently ---

def test_default_system_injected_when_missing() -> None:
    messages = [{"role": "user", "content": "Hi"}]
    inf = build_inference_prompt(messages)
    assert inf.startswith(SYSTEM_OPEN + "\n")


def test_explicit_system_is_respected() -> None:
    messages = [
        {"role": "system", "content": "Sprich wie ein Pirat."},
        {"role": "user", "content": "Hi"},
    ]
    inf = build_inference_prompt(messages)
    assert "Sprich wie ein Pirat." in inf
    assert inf.count(SYSTEM_OPEN) == 1


# --- 3. Round-trip: the same message list must produce the same bytes ---

def test_identical_inputs_produce_identical_outputs() -> None:
    messages = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "U2"},
    ]
    a = build_inference_prompt(messages)
    b = build_inference_prompt(messages)
    assert a == b


def test_dataclass_and_dict_inputs_are_equivalent() -> None:
    dict_form = [
        {"role": "user", "content": "Hi"},
    ]
    obj_form = [ChatMessage("user", "Hi")]
    assert build_inference_prompt(dict_form) == build_inference_prompt(obj_form)


# --- 4. Role validation and edge cases ---

def test_training_requires_trailing_assistant() -> None:
    with pytest.raises(ValueError):
        build_training_prompt([{"role": "user", "content": "Hi"}])


def test_inference_requires_trailing_user() -> None:
    with pytest.raises(ValueError):
        build_inference_prompt([
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ])


def test_unknown_role_rejected() -> None:
    with pytest.raises(ValueError):
        build_chat_prompt([{"role": "mystery", "content": "x"}], add_generation_prompt=True)


# --- 5. Pinned byte-level snapshot — v1-bug tripwire ---

def test_pinned_snapshot_inference() -> None:
    """Explicit byte-level pin. Do NOT edit unless you are intentionally
    changing the on-the-wire prompt format — and if you do, retrain.
    """
    prompt = build_inference_prompt(
        [{"role": "user", "content": "Hallo"}],
        default_system="Du bist Helix.",
    )
    expected = (
        f"{SYSTEM_OPEN}\nDu bist Helix.\n{END}\n"
        f"{USER_OPEN}\nHallo\n{END}\n"
        f"{ASSISTANT_OPEN}\n"
    )
    assert prompt == expected, f"Snapshot diverged:\ngot : {prompt!r}\nwant: {expected!r}"


def test_pinned_snapshot_training() -> None:
    prompt = build_training_prompt(
        [
            {"role": "user", "content": "Hallo"},
            {"role": "assistant", "content": "Hi!"},
        ],
        default_system="Du bist Helix.",
    )
    expected = (
        f"{SYSTEM_OPEN}\nDu bist Helix.\n{END}\n"
        f"{USER_OPEN}\nHallo\n{END}\n"
        f"{ASSISTANT_OPEN}\nHi!\n{END}\n"
    )
    assert prompt == expected
