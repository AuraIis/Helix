"""Run fixed capability probes against a Helix checkpoint.

This is an early-training learning-curve tool, not a leaderboard benchmark.
It answers: "Is the model becoming less noisy and more capable?" by tracking
small, stable probes for German facts, arithmetic, code, instruction following,
and garbage/hallucination regressions.

Examples:

    python scripts/eval/run_capability_probes.py --dry --tag dry

    python scripts/eval/run_capability_probes.py \\
        --model-config configs/model/helix_v2_mid_500m_smart.yaml \\
        --checkpoint checkpoints/pretrain_clean_v2_500m/step_1000.pt \\
        --tag clean_v2_500m_step_1000
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))


DEFAULT_PROBES = REPO / "eval" / "capability_probes_clean_v2.yaml"
DEFAULT_RESULTS_DIR = REPO / "eval" / "results" / "capability"
GARBAGE_MARKERS = [
    "<html",
    "<a href",
    "href=",
    "_end_of_the_data",
    "_user-data",
    "<|im_start|>",
    "<|im_end|>",
]


@dataclass
class Probe:
    id: str
    category: str
    prompt: str
    expect_any: list[str] = field(default_factory=list)
    expect_all: list[str] = field(default_factory=list)
    forbid_any: list[str] = field(default_factory=list)
    regex: str | None = None
    max_new_tokens: int | None = None


@dataclass
class ProbeResult:
    id: str
    category: str
    score: float
    prompt: str
    matched: list[str]
    forbidden: list[str]
    garbage: list[str]
    repetition_ratio: float
    answer: str


def _norm(text: str) -> str:
    return text.lower().replace("ß", "ss")


def _contains(haystack: str, needle: str) -> bool:
    return _norm(needle) in _norm(haystack)


def repetition_ratio(text: str, ngram: int = 3) -> float:
    words = re.findall(r"\w+", _norm(text))
    if len(words) < ngram * 2:
        return 0.0
    grams = [tuple(words[i : i + ngram]) for i in range(len(words) - ngram + 1)]
    if not grams:
        return 0.0
    return 1.0 - (len(set(grams)) / len(grams))


def score_answer(answer: str, probe: Probe) -> ProbeResult:
    matched: list[str] = []
    forbidden: list[str] = []
    garbage = [m for m in GARBAGE_MARKERS if _contains(answer, m)]

    score_parts: list[float] = []
    if probe.expect_any:
        any_hits = [kw for kw in probe.expect_any if _contains(answer, kw)]
        matched.extend(any_hits)
        score_parts.append(1.0 if any_hits else 0.0)
    if probe.expect_all:
        all_hits = [kw for kw in probe.expect_all if _contains(answer, kw)]
        matched.extend(all_hits)
        score_parts.append(len(all_hits) / max(1, len(probe.expect_all)))
    if probe.regex:
        hit = re.search(probe.regex, answer, flags=re.IGNORECASE | re.MULTILINE)
        score_parts.append(1.0 if hit else 0.0)
        if hit:
            matched.append(probe.regex)

    base = sum(score_parts) / len(score_parts) if score_parts else 0.0
    forbidden = [kw for kw in probe.forbid_any if _contains(answer, kw)]
    rep = repetition_ratio(answer)

    penalty = 0.0
    if forbidden:
        penalty += 0.4
    if garbage:
        penalty += 0.3
    if rep > 0.25:
        penalty += min(0.3, rep)

    return ProbeResult(
        id=probe.id,
        category=probe.category,
        score=max(0.0, min(1.0, base - penalty)),
        prompt=probe.prompt,
        matched=sorted(set(matched)),
        forbidden=forbidden,
        garbage=garbage,
        repetition_ratio=round(rep, 4),
        answer=answer,
    )


def load_probes(path: Path) -> tuple[list[Probe], dict[str, Any]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    defaults = data.get("defaults", {}) or {}
    probes = [Probe(**{k: v for k, v in p.items() if k in Probe.__dataclass_fields__}) for p in data["probes"]]
    return probes, defaults


def _maybe_enable_mamba_kernel(device: str) -> bool:
    if os.environ.get("AURALIS_USE_MAMBA_KERNEL", "") == "1":
        return True
    if device != "cuda":
        return False
    try:
        import mamba_ssm  # noqa: F401
    except Exception:
        return False
    os.environ["AURALIS_USE_MAMBA_KERNEL"] = "1"
    return True


def build_generator(args: argparse.Namespace):
    import sentencepiece as spm
    import torch

    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    if str(device) == "auto":
        device = torch.device("cpu")
    _maybe_enable_mamba_kernel(device.type)

    from auralis.model import build_model

    model = build_model(args.model_config).to(device)
    if args.checkpoint:
        payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
        state = {k.replace("_orig_mod.", ""): v for k, v in payload["model"].items()}
        missing, extra = model.load_state_dict(state, strict=False)
        if missing or extra:
            raise RuntimeError(
                f"checkpoint state mismatch: missing={len(missing)} extra={len(extra)}; "
                f"first_missing={missing[:3]} first_extra={extra[:3]}"
            )
    model.eval()
    sp = spm.SentencePieceProcessor(model_file=str(args.tokenizer))
    end_ids = sp.EncodeAsIds("<|end|>")
    eos = end_ids[-1] if end_ids else sp.eos_id()

    def generate(prompt: str, max_new_tokens: int) -> str:
        ids = sp.EncodeAsIds(prompt)
        x = torch.tensor([ids], dtype=torch.long, device=device)
        new_ids: list[int] = []
        with torch.no_grad():
            for _ in range(max_new_tokens):
                autocast = (
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else torch.no_grad()
                )
                with autocast:
                    out = model(input_ids=x)
                next_id = int(out["logits"][0, -1].argmax().item())
                if next_id == eos:
                    break
                new_ids.append(next_id)
                x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        return sp.DecodeIds(new_ids).strip()

    return generate


def run(args: argparse.Namespace) -> dict[str, Any]:
    probes, defaults = load_probes(args.probes)
    max_default = int(args.max_new_tokens or defaults.get("max_new_tokens", 48))
    prompt_style = str(defaults.get("prompt_style", "plain_qa"))

    if args.dry:
        def generator(prompt: str, max_new_tokens: int) -> str:
            if "Hauptstadt von Deutschland" in prompt:
                return "Berlin."
            if "17 + 25" in prompt:
                return "42"
            if "Python-Zeile" in prompt:
                return "print('Hallo')"
            return "Ich weiss es noch nicht."
    else:
        generator = build_generator(args)

    results: list[ProbeResult] = []
    for probe in probes:
        prompt = probe.prompt
        if prompt_style == "chat":
            from auralis.tokenizer.chat_template import build_inference_prompt

            prompt = build_inference_prompt(
                [{"role": "user", "content": probe.prompt}],
                default_system=(
                    "Du bist Auralis, ein hilfreicher deutscher KI-Assistent. "
                    "Antworte korrekt, knapp und ehrlich. Wenn etwas unsicher oder erfunden ist, sage das deutlich."
                ),
            )
        answer = generator(prompt, int(probe.max_new_tokens or max_default))
        results.append(score_answer(answer, probe))

    by_cat: dict[str, list[float]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r.score)
    aggregate = sum(r.score for r in results) / max(1, len(results))
    report: dict[str, Any] = {
        "tag": args.tag,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "model_config": str(args.model_config) if args.model_config else None,
        "probe_file": str(args.probes),
        "aggregate_score": aggregate,
        "by_category": {k: sum(v) / len(v) for k, v in sorted(by_cat.items())},
        "num_probes": len(results),
        "results": [asdict(r) for r in results],
    }
    return report


def write_report(report: dict[str, Any], results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / f"{report['tag']}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = out.with_suffix(".md")
    lines = [
        f"# Capability Probes: {report['tag']}",
        "",
        f"- aggregate: {report['aggregate_score'] * 100:.1f}%",
        f"- probes: {report['num_probes']}",
        f"- checkpoint: `{report['checkpoint']}`",
        "",
        "## Categories",
        "",
    ]
    for cat, score in report["by_category"].items():
        lines.append(f"- {cat}: {score * 100:.1f}%")
    lines.extend(["", "## Answers", ""])
    for r in report["results"]:
        lines.append(f"### {r['id']} ({r['score'] * 100:.0f}%)")
        lines.append(f"Prompt: `{r['prompt']}`")
        lines.append("")
        lines.append(r["answer"].replace("\n", " ").strip()[:500])
        if r["forbidden"] or r["garbage"]:
            lines.append(f"\nFlags: forbidden={r['forbidden']} garbage={r['garbage']}")
        lines.append("")
    md.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--model-config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--tokenizer", type=Path, default=REPO / "tokenizer" / "helix_v2_tokenizer.model")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()

    if not args.dry and (args.model_config is None or args.checkpoint is None):
        raise SystemExit("--model-config and --checkpoint are required unless --dry is used")

    report = run(args)
    out = write_report(report, args.results_dir)
    print(f"Aggregate: {report['aggregate_score'] * 100:.1f}% over {report['num_probes']} probes")
    for cat, score in report["by_category"].items():
        print(f"  {cat:20s} {score * 100:5.1f}%")
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
