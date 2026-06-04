"""Diagnose early Helix checkpoints beyond simple greedy QA probes.

This script is intentionally small and read-only. It loads a checkpoint,
prints/saves:

- greedy and sampled generations for fixed German/English prompts
- top-k next-token candidates for short probes
- contrastive continuation margins: NLL(wrong) - NLL(correct)
- repetition statistics for generated text

It is meant for early foundation checkpoints where free-form QA is still too
raw, but latent preferences may already be visible in logits.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sentencepiece as spm
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


def _maybe_enable_mamba_kernel() -> bool:
    if os.environ.get("AURALIS_USE_MAMBA_KERNEL", "") == "1":
        return True
    if not torch.cuda.is_available():
        return False
    try:
        import mamba_ssm  # noqa: F401
    except Exception:
        return False
    os.environ["AURALIS_USE_MAMBA_KERNEL"] = "1"
    return True


_KERNEL_ACTIVE = _maybe_enable_mamba_kernel()

from auralis.model import build_model  # noqa: E402


GEN_PROMPTS = [
    ("de_plain_berlin", "Berlin ist eine Stadt"),
    ("de_plain_wissen", "Die Hauptstadt von Deutschland ist"),
    ("de_qa_capital", "Frage: Was ist die Hauptstadt von Deutschland?\nAntwort:"),
    ("de_instruction", "Schreibe einen einfachen deutschen Satz über Wasser:\n"),
    ("en_plain_capital", "The capital of Germany is"),
    ("en_instruction", "Write one simple sentence about water:\n"),
]

TOPK_PROMPTS = [
    ("de_capital_next", "Die Hauptstadt von Deutschland ist"),
    ("de_faust_next", "Faust wurde geschrieben von"),
    ("de_water_next", "Wasser ist bei Raumtemperatur"),
    ("en_capital_next", "The capital of Germany is"),
]

MARGIN_PROBES = [
    ("capital_berlin_vs_bonn", "Die Hauptstadt von Deutschland ist", " Berlin.", " Bonn."),
    ("faust_goethe_vs_hitler", "Faust wurde geschrieben von", " Goethe.", " Hitler."),
    ("water_liquid_vs_metal", "Wasser ist bei Raumtemperatur", " fluessig.", " ein Metall."),
    ("en_capital_berlin_vs_bonn", "The capital of Germany is", " Berlin.", " Bonn."),
]


@dataclass
class GenerationResult:
    id: str
    prompt: str
    mode: str
    text: str
    token_ids: list[int]
    repetition_ratio: float


@dataclass
class TopKResult:
    id: str
    prompt: str
    tokens: list[dict[str, Any]]


@dataclass
class MarginResult:
    id: str
    prompt: str
    correct: str
    wrong: str
    correct_nll: float
    wrong_nll: float
    margin_wrong_minus_correct: float


def repetition_ratio(text: str, ngram: int = 3) -> float:
    words = re.findall(r"\w+", text.lower())
    if len(words) < ngram * 2:
        return 0.0
    grams = [tuple(words[i : i + ngram]) for i in range(len(words) - ngram + 1)]
    return 1.0 - len(set(grams)) / max(1, len(grams))


def load_model(args: argparse.Namespace):
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    model = build_model(args.model_config).to(device)
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = {k.replace("_orig_mod.", ""): v for k, v in payload["model"].items()}
    missing, extra = model.load_state_dict(state, strict=False)
    if missing or extra:
        raise RuntimeError(
            f"state mismatch missing={len(missing)} extra={len(extra)} "
            f"first_missing={missing[:4]} first_extra={extra[:4]}"
        )
    model.eval()
    return model, device, payload


def generate(
    model,
    sp: spm.SentencePieceProcessor,
    device: torch.device,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> tuple[str, list[int]]:
    ids = sp.EncodeAsIds(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    new_ids: list[int] = []
    eos = sp.EncodeAsIds("<|end|>")[-1]
    with torch.no_grad():
        for _ in range(max_new_tokens):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(input_ids=x)["logits"][0, -1].float()
            if temperature <= 0:
                next_id = int(torch.argmax(logits).item())
            else:
                logits = logits / max(temperature, 1e-6)
                if top_k > 0:
                    vals, idx = torch.topk(logits, min(top_k, logits.numel()))
                    probs = torch.softmax(vals, dim=-1)
                    next_id = int(idx[torch.multinomial(probs, 1)].item())
                else:
                    probs = torch.softmax(logits, dim=-1)
                    next_id = int(torch.multinomial(probs, 1).item())
            if next_id == eos:
                break
            new_ids.append(next_id)
            x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
    return sp.DecodeIds(new_ids), new_ids


def topk(model, sp: spm.SentencePieceProcessor, device: torch.device, prompt: str, k: int) -> list[dict[str, Any]]:
    ids = sp.EncodeAsIds(prompt)
    x = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(input_ids=x)["logits"][0, -1].float()
    probs = torch.softmax(logits, dim=-1)
    vals, idx = torch.topk(probs, k)
    out = []
    for p, token_id in zip(vals.tolist(), idx.tolist()):
        piece = sp.IdToPiece(int(token_id))
        out.append({"id": int(token_id), "piece": piece, "decoded": sp.DecodeIds([int(token_id)]), "prob": float(p)})
    return out


def continuation_nll(
    model,
    sp: spm.SentencePieceProcessor,
    device: torch.device,
    prompt: str,
    continuation: str,
) -> float:
    prompt_ids = sp.EncodeAsIds(prompt)
    full_ids = sp.EncodeAsIds(prompt + continuation)
    cont_start = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        cont_start += 1
    if cont_start >= len(full_ids):
        return float("inf")
    x = torch.tensor([full_ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor(full_ids[1:], dtype=torch.long, device=device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits = model(input_ids=x)["logits"][0].float()
    logp = torch.log_softmax(logits, dim=-1)
    # Token t in full_ids is predicted at logits[t-1].
    losses = []
    for pos in range(max(1, cont_start), len(full_ids)):
        losses.append(float(-logp[pos - 1, int(y[pos - 1])].item()))
    return sum(losses) / max(1, len(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--top-k", type=int, default=12)
    args = parser.parse_args()

    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    model, device, payload = load_model(args)
    step = payload.get("state", {}).get("step")
    generations: list[GenerationResult] = []
    for probe_id, prompt in GEN_PROMPTS:
        for mode, temp in [("greedy", 0.0), ("sample_t0.8_k40", 0.8)]:
            text, ids = generate(
                model,
                sp,
                device,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=temp,
                top_k=40,
            )
            generations.append(
                GenerationResult(
                    id=probe_id,
                    prompt=prompt,
                    mode=mode,
                    text=text,
                    token_ids=ids,
                    repetition_ratio=round(repetition_ratio(text), 4),
                )
            )

    topks = [TopKResult(pid, prompt, topk(model, sp, device, prompt, args.top_k)) for pid, prompt in TOPK_PROMPTS]
    margins: list[MarginResult] = []
    for pid, prompt, correct, wrong in MARGIN_PROBES:
        c = continuation_nll(model, sp, device, prompt, correct)
        w = continuation_nll(model, sp, device, prompt, wrong)
        margins.append(MarginResult(pid, prompt, correct, wrong, c, w, w - c))

    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint),
        "step": step,
        "device": str(device),
        "mamba_backend": "mamba_ssm" if _KERNEL_ACTIVE else "native",
        "generations": [asdict(x) for x in generations],
        "topk": [asdict(x) for x in topks],
        "margins": [asdict(x) for x in margins],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = args.output.with_suffix(".md")
    lines = [
        f"# Checkpoint Generation Diagnosis",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- step: `{step}`",
        f"- backend: `{report['mamba_backend']}`",
        "",
        "## Generations",
        "",
    ]
    for g in generations:
        lines.append(f"### {g.id} / {g.mode} / rep={g.repetition_ratio:.3f}")
        lines.append(f"Prompt: `{g.prompt}`")
        lines.append("")
        lines.append(g.text.replace("\n", " ").strip()[:800] or "(empty)")
        lines.append("")
    lines.extend(["## Top-K Next Tokens", ""])
    for tk in topks:
        lines.append(f"### {tk.id}")
        lines.append(f"Prompt: `{tk.prompt}`")
        for t in tk.tokens:
            lines.append(f"- `{t['piece']}` -> `{t['decoded']}` p={t['prob']:.4f}")
        lines.append("")
    lines.extend(["## Contrastive Margins", "", "Positive margin means correct continuation is preferred.", ""])
    for m in margins:
        lines.append(
            f"- **{m.id}** margin={m.margin_wrong_minus_correct:.4f} "
            f"correct_nll={m.correct_nll:.4f} wrong_nll={m.wrong_nll:.4f}"
        )
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {md}")


if __name__ == "__main__":
    main()
