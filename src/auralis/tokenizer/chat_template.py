"""Single source of truth for the Helix v2 chat prompt format.

Why this module exists
----------------------
In Auralis v1 an inconsistency between the training-time prompt (``<|user|>``)
and the inference-time prompt (``User:\\n``) silently degraded inference for
weeks — see ``LESSONS.md`` entry L-001. The v2 rule is:

    One builder. Used by training, inference, eval, and the API.
    Byte-for-byte identical output in every path.

``build_chat_prompt`` below produces the final string that is tokenized.
``build_training_prompt`` and ``build_inference_prompt`` are thin wrappers
that differ *only* in whether a trailing assistant-open tag is appended:

- Training: the full conversation including the assistant's reply is present;
  we optionally append ``<|end|>`` after the assistant turn.
- Inference: we stop right after ``<|assistant|>\\n`` so the model continues.

Any future change to the format belongs here and nowhere else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["system", "user", "assistant"]


# ---------------------------------------------------------------------------
# Special tokens — mirror of helix_v2_tokenizer vocabulary.
# Kept in ONE place. If the tokenizer vocab ever changes, update here and
# fail loudly via tests before retraining anything.
# ---------------------------------------------------------------------------

SYSTEM_OPEN = "<|system|>"
USER_OPEN = "<|user|>"
ASSISTANT_OPEN = "<|assistant|>"
END = "<|end|>"

# Default system prompt when the caller does not provide one.
DEFAULT_SYSTEM = "Du bist Auralis, ein hilfsbereiter KI-Assistent auf Basis des Helix v2 Modells."


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


def _render_turn(role: Role, content: str) -> str:
    """Render a single turn. Newline layout is load-bearing — do not change casually."""
    tag = {"system": SYSTEM_OPEN, "user": USER_OPEN, "assistant": ASSISTANT_OPEN}[role]
    return f"{tag}\n{content}\n{END}\n"


def _normalize(messages: list[ChatMessage] | list[dict[str, str]]) -> list[ChatMessage]:
    out: list[ChatMessage] = []
    for m in messages:
        if isinstance(m, ChatMessage):
            out.append(m)
        else:
            role = m["role"]
            if role not in ("system", "user", "assistant"):
                raise ValueError(f"Unknown role: {role!r}")
            out.append(ChatMessage(role=role, content=m["content"]))  # type: ignore[arg-type]
    return out


def build_chat_prompt(
    messages: list[ChatMessage] | list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    default_system: str = DEFAULT_SYSTEM,
) -> str:
    """Build the exact byte string that gets tokenized.

    Args:
        messages: Conversation turns in order. A system turn is inserted
            automatically (``default_system``) when the first message is not
            already a system turn. This guarantees training and inference
            never diverge due to a missing system prompt.
        add_generation_prompt: When True, append an open assistant tag and
            stop — used at inference. When False, the prompt ends on the
            last supplied turn (training-friendly when the final turn is
            already the assistant reply).

    Returns:
        The complete prompt string.
    """
    msgs = _normalize(messages)
    if not msgs or msgs[0].role != "system":
        msgs = [ChatMessage("system", default_system), *msgs]

    parts: list[str] = [_render_turn(m.role, m.content) for m in msgs]

    if add_generation_prompt:
        parts.append(f"{ASSISTANT_OPEN}\n")
    return "".join(parts)


def build_training_prompt(
    messages: list[ChatMessage] | list[dict[str, str]],
    *,
    default_system: str = DEFAULT_SYSTEM,
) -> str:
    """Build the prompt used during SFT/DPO/ORPO.

    The last turn MUST be an assistant turn (the target). The output is the
    full conversation; the trainer is responsible for masking loss on the
    non-assistant tokens.
    """
    msgs = _normalize(messages)
    if not msgs or msgs[-1].role != "assistant":
        raise ValueError("Training prompt requires the final message to be an assistant turn.")
    return build_chat_prompt(messages, add_generation_prompt=False, default_system=default_system)


def build_inference_prompt(
    messages: list[ChatMessage] | list[dict[str, str]],
    *,
    default_system: str = DEFAULT_SYSTEM,
) -> str:
    """Build the prompt used for generation.

    The last turn MUST be a user turn. We append the open assistant tag so the
    model continues into the reply.
    """
    msgs = _normalize(messages)
    if not msgs or msgs[-1].role != "user":
        raise ValueError("Inference prompt requires the final message to be a user turn.")
    return build_chat_prompt(messages, add_generation_prompt=True, default_system=default_system)


__all__ = [
    "ASSISTANT_OPEN",
    "ChatMessage",
    "DEFAULT_SYSTEM",
    "END",
    "Role",
    "SYSTEM_OPEN",
    "USER_OPEN",
    "build_chat_prompt",
    "build_inference_prompt",
    "build_training_prompt",
]
