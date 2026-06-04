#!/usr/bin/env python3
"""Evaluate frozen target/retention response probes and check prompt leaks.

This script is intentionally stricter than the old aggregate gates:

- target and retention are scored separately
- promotion requires target pass AND retention pass
- a single retention failure makes the checkpoint not promotable
- eval prompt hashes can be checked against training JSONL user prompts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


HELIX_USER_RE = re.compile(r"<\|user\|>\n(.*?)\n<\|end\|>", re.DOTALL)


@dataclass
class FrozenProbe:
    id: str
    split: str
    category: str
    prompt: str
    semantic_must: list[str] = field(default_factory=list)
    semantic_any_groups: list[list[str]] = field(default_factory=list)
    semantic_forbid: list[str] = field(default_factory=list)
    semantic_polarity: str | None = None


def normalize(text: str) -> str:
    text = text.replace("ÃŸ", "ss")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.casefold()
    replacements = {
        "österreich": "oesterreich",
        "osterreich": "oesterreich",
        "münchen": "muenchen",
        "munchen": "muenchen",
        "zürich": "zuerich",
        "zurich": "zuerich",
        "verläss": "verlaess",
        "weiß": "weiss",
        "grün": "gruen",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prompt_hash(text: str) -> str:
    return hashlib.blake2b(normalize(text).encode("utf-8"), digest_size=16).hexdigest()


def starts_with_polarity(text: str) -> str | None:
    norm = normalize(text)
    if re.match(r"^(ja|jawohl|richtig|stimmt)\b", norm):
        return "yes"
    if re.match(r"^(nein|nicht|falsch)\b", norm):
        return "no"
    return None


def contains_all(norm_answer: str, words: list[str]) -> bool:
    return all(normalize(word) in norm_answer for word in words)


def load_probes(path: Path) -> list[FrozenProbe]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    probes: list[FrozenProbe] = []
    for raw in data.get("probes", []):
        probes.append(
            FrozenProbe(
                id=str(raw["id"]),
                split=str(raw.get("split", "target")),
                category=str(raw.get("category", "unknown")),
                prompt=str(raw["prompt"]),
                semantic_must=list(raw.get("semantic_must") or []),
                semantic_any_groups=[list(group) for group in raw.get("semantic_any_groups") or []],
                semantic_forbid=list(raw.get("semantic_forbid") or raw.get("forbid_any") or []),
                semantic_polarity=raw.get("semantic_polarity"),
            )
        )
    if not probes:
        raise SystemExit(f"no probes found in {path}")
    bad = [p.id for p in probes if p.split not in {"target", "retention"}]
    if bad:
        raise SystemExit(f"unknown split in probes: {bad}")
    return probes


def result_items(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "results" in data:
        return data["results"]
    if "summary" in data and "results" in data["summary"]:
        return data["summary"]["results"]
    raise SystemExit(f"{path} is not a probe result JSON")


def evaluate_answer(probe: FrozenProbe, answer: str) -> dict[str, Any]:
    norm_answer = normalize(answer)
    issues: list[str] = []
    actual_polarity = starts_with_polarity(answer)
    if probe.semantic_polarity and actual_polarity and actual_polarity != probe.semantic_polarity:
        issues.append(f"wrong_polarity:{actual_polarity}_expected_{probe.semantic_polarity}")
    elif probe.semantic_polarity and actual_polarity is None and re.match(
        r"^(ist|war|hat|gilt|soll|sollte)\b", normalize(probe.prompt)
    ):
        issues.append(f"missing_explicit_polarity:{probe.semantic_polarity}")
    for word in probe.semantic_must:
        if normalize(word) not in norm_answer:
            issues.append(f"missing:{word}")
    if probe.semantic_any_groups and not any(contains_all(norm_answer, group) for group in probe.semantic_any_groups):
        issues.append("missing_any_group:" + "|".join("+".join(group) for group in probe.semantic_any_groups))
    for phrase in probe.semantic_forbid:
        if normalize(phrase) in norm_answer:
            issues.append(f"forbidden:{phrase}")
    return {
        "id": probe.id,
        "split": probe.split,
        "category": probe.category,
        "prompt": probe.prompt,
        "answer": answer,
        "semantic_score": 1.0 if not issues else 0.0,
        "issues": issues,
        "expected_polarity": probe.semantic_polarity,
        "actual_polarity": actual_polarity,
    }


def evaluate(probe_file: Path, input_json: Path) -> dict[str, Any]:
    probes = {probe.id: probe for probe in load_probes(probe_file)}
    out: list[dict[str, Any]] = []
    for item in result_items(input_json):
        probe = probes.get(str(item.get("id", "")))
        if not probe:
            continue
        out.append(evaluate_answer(probe, str(item.get("answer", ""))))
    missing = sorted(set(probes) - {item["id"] for item in out})
    if missing:
        raise SystemExit(f"missing result ids: {missing}")
    by_split: dict[str, list[dict[str, Any]]] = {"target": [], "retention": []}
    for item in out:
        by_split[item["split"]].append(item)
    split_summary: dict[str, Any] = {}
    for split, items in by_split.items():
        passed = sum(1 for item in items if item["semantic_score"] == 1.0)
        split_summary[split] = {
            "passed": passed,
            "total": len(items),
            "score": passed / len(items) if items else 0.0,
            "failures": [item for item in items if item["semantic_score"] != 1.0],
        }
    promotable = (
        split_summary["target"]["passed"] == split_summary["target"]["total"]
        and split_summary["retention"]["passed"] == split_summary["retention"]["total"]
    )
    return {
        "probe_file": str(probe_file),
        "source": str(input_json),
        "promotable": promotable,
        "summary": split_summary,
        "results": out,
    }


def extract_train_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = str(item.get("text") or item.get("prompt") or "")
            match = HELIX_USER_RE.search(text)
            if match:
                prompts.append(match.group(1).strip())
            elif item.get("prompt"):
                prompts.append(str(item["prompt"]).strip())
    return prompts


def check_leaks(probe_file: Path, train_files: list[Path]) -> dict[str, Any]:
    probes = load_probes(probe_file)
    eval_hashes = {prompt_hash(probe.prompt): probe for probe in probes}
    collisions: list[dict[str, str]] = []
    train_prompt_count = 0
    for path in train_files:
        for prompt in extract_train_prompts(path):
            train_prompt_count += 1
            h = prompt_hash(prompt)
            if h in eval_hashes:
                collisions.append(
                    {
                        "probe_id": eval_hashes[h].id,
                        "probe_prompt": eval_hashes[h].prompt,
                        "train_file": str(path),
                        "train_prompt": prompt,
                        "hash": h,
                    }
                )
    return {
        "probe_file": str(probe_file),
        "train_files": [str(path) for path in train_files],
        "eval_prompts": len(probes),
        "train_prompts": train_prompt_count,
        "collisions": collisions,
        "passed": not collisions,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# Frozen Response Gate: {Path(payload['source']).name}",
        "",
        f"- promotable: {payload['promotable']}",
        "",
        "## Split Summary",
        "",
    ]
    for split in ["target", "retention"]:
        summary = payload["summary"][split]
        lines.append(f"- {split}: {summary['passed']} / {summary['total']} ({summary['score']:.3f})")
    lines.extend(["", "## Failures", ""])
    for split in ["target", "retention"]:
        failures = payload["summary"][split]["failures"]
        lines.append(f"### {split}")
        if not failures:
            lines.append("")
            lines.append("None.")
            lines.append("")
            continue
        for item in failures:
            lines.append(f"- `{item['id']}` issues={item['issues']}")
            lines.append(f"  - prompt: {item['prompt']}")
            lines.append(f"  - answer: {item['answer']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--input", type=Path, help="Capability result JSON to evaluate.")
    ap.add_argument("--output-json", type=Path)
    ap.add_argument("--output-md", type=Path)
    ap.add_argument("--check-train", nargs="*", type=Path, default=[])
    ap.add_argument("--leak-json", type=Path)
    args = ap.parse_args()

    if args.check_train:
        leak = check_leaks(args.probes, args.check_train)
        if args.leak_json:
            args.leak_json.parent.mkdir(parents=True, exist_ok=True)
            args.leak_json.write_text(json.dumps(leak, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(leak, ensure_ascii=False, indent=2))
        if not leak["passed"]:
            return 2

    if args.input:
        payload = evaluate(args.probes, args.input)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        if args.output_md:
            args.output_md.parent.mkdir(parents=True, exist_ok=True)
            write_markdown(args.output_md, payload)
        return 0 if payload["promotable"] else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
