"""Torch-dependent scoring primitives for learning detection.

Kept deliberately small and isolated so the rest of the adaptive package stays
torch-free and unit-testable. The key signal is the **teacher-forced margin**:
how much more likely the model thinks the *correct* continuation is versus a
*wrong* one, given the same prompt. That is a far lower-variance "does it know
this fact" signal than free-form generation, and it needs no sampler.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

import torch


@contextmanager
def _eval_mode(model: torch.nn.Module):
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if was_training:
            model.train()


def continuation_nll(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    continuation_ids: Sequence[int],
    device: torch.device | str = "cuda",
) -> float:
    """Mean next-token NLL (nats) of ``continuation_ids`` given ``prompt_ids``.

    Uses label masking: prompt positions are set to ``-100`` so only the
    continuation tokens contribute. Reads ``loss_main`` (pure LM CE, never the
    MTP term). Lower = the model finds the continuation more likely.
    """
    if not continuation_ids:
        raise ValueError("continuation_ids must be non-empty")
    full = list(prompt_ids) + list(continuation_ids)
    input_ids = torch.tensor([full], dtype=torch.long, device=device)
    labels = torch.full((1, len(full)), -100, dtype=torch.long, device=device)
    labels[0, len(prompt_ids):] = torch.tensor(continuation_ids, dtype=torch.long, device=device)
    with _eval_mode(model):
        out = model(input_ids=input_ids, labels=labels)
    loss = out.get("loss_main")
    if loss is None:
        loss = out["loss"]
    return float(loss.detach().cpu())


def margin(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    correct_ids: Sequence[int],
    wrong_ids: Sequence[int],
    device: torch.device | str = "cuda",
) -> dict[str, float]:
    """Contrastive margin = NLL(wrong) - NLL(correct).

    Positive => the model prefers the correct continuation (good). Returns the
    margin plus both raw NLLs for diagnostics.
    """
    nll_correct = continuation_nll(model, prompt_ids, correct_ids, device)
    nll_wrong = continuation_nll(model, prompt_ids, wrong_ids, device)
    return {
        "margin": nll_wrong - nll_correct,
        "nll_correct": nll_correct,
        "nll_wrong": nll_wrong,
    }


def canary_loss(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Deterministic loss on a fixed batch (denoised training signal).

    The batch is captured once at run start and re-evaluated every eval step, so
    its loss curve is free of the sampling noise that makes the live train loss
    hard to read.
    """
    device = next(model.parameters()).device
    with _eval_mode(model):
        out = model(input_ids=input_ids.to(device), labels=labels.to(device))
    loss = out.get("loss_main") or out["loss"]
    return float(loss.detach().cpu())


def greedy_generate(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    max_new_tokens: int = 64,
    eos_id: int | None = None,
    device: torch.device | str = "cuda",
) -> list[int]:
    """Simple greedy decode (no KV cache; recomputes each step).

    Slow but correct and dependency-free — fine at eval cadence for short
    answers used to feed the free-form frozen gate. Not a production sampler.
    """
    ids = list(prompt_ids)
    with _eval_mode(model):
        for _ in range(max_new_tokens):
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            logits = model(input_ids=inp)["logits"]
            nxt = int(logits[0, -1].argmax().item())
            ids.append(nxt)
            if eos_id is not None and nxt == eos_id:
                break
    return ids[len(prompt_ids):]


__all__ = ["continuation_nll", "margin", "canary_loss", "greedy_generate"]
