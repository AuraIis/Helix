#!/usr/bin/env python3
"""Rule-based semantic diagnostics for response-fix capability probe outputs.

The normal capability gate is intentionally simple and keyword based. This
script adds a second pass for the fragile German response-fix probes where a
model can mention the right entity but answer with the wrong polarity, e.g.
"Nein. Wien ist die Hauptstadt ...".
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any


EXPECTED: dict[str, dict[str, Any]] = {
    "yes_wien_capital": {
        "polarity": "yes",
        "must": ["wien", "oesterreich"],
        "forbid": ["nein"],
    },
    "yes_bern_capital": {
        "polarity": "yes",
        "must": ["bern", "schweiz"],
        "forbid": ["nein"],
    },
    "no_bonn_current": {
        "polarity": "no",
        "must": ["berlin"],
        "forbid": ["ja", "bonn ist die hauptstadt", "bonn ist heute die hauptstadt"],
    },
    "no_hamburg_bavaria": {
        "polarity": "no",
        "must": ["muenchen"],
        "forbid": ["ja", "hamburg ist die hauptstadt"],
    },
    "goethe_not_mein_kampf_short": {
        "polarity": "no",
        "must": ["hitler"],
        "forbid": ["ja"],
    },
    "goethe_faust_author": {
        "polarity": None,
        "must": ["goethe"],
        "forbid": ["faust wurde geschrieben als faust", "koenig", "alten rom"],
    },
    "water_not_element": {
        "polarity": "no",
        "must": ["verbindung"],
        "any_must": [["wasserstoff", "sauerstoff"], ["h2o"]],
        "forbid": ["ja", "wasser ist ein chemisches element"],
    },
    "oxygen_definition": {
        "polarity": None,
        "must": ["sauerstoff", "element"],
        "forbid": [],
    },
    "photosynthesis_definition": {
        "polarity": None,
        "must": ["pflanzen", "licht"],
        "any_must": [["zucker"], ["glucose"], ["sauerstoff"]],
        "forbid": [],
    },
    "unknown_source_behavior": {
        "polarity": None,
        "must": [],
        "any_must": [
            ["nicht sicher"],
            ["keine sichere"],
            ["keine verlaessliche"],
            ["nicht belegt"],
            ["keine details erfinden"],
            ["sage", "unsicher"],
        ],
        "forbid": ["ich suche nach einer antwort", "ich suche eine antwort", "keine ahnung"],
    },
    "unknown_planet_behavior": {
        "polarity": None,
        "must": [],
        "any_must": [["erfunden"], ["keine"], ["nicht verlaesslich"], ["keine verlaessliche"], ["unbekannt"]],
        "forbid": ["qorblax ist eine farbe"],
    },
    "computer_definition": {
        "polarity": None,
        "must": ["maschine", "daten"],
        "any_must": [["programme"], ["verarbeitet"], ["rechnet"]],
        "forbid": [],
    },
}


def normalize(text: str) -> str:
    text = text.replace("ß", "ss")
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    replacements = {
        "osterreich": "oesterreich",
        "munchen": "muenchen",
        "verlass": "verlaess",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def starts_with_polarity(text: str) -> str | None:
    norm = normalize(text)
    if re.match(r"^(ja|jawohl|richtig|stimmt)\b", norm):
        return "yes"
    if re.match(r"^(nein|nicht|falsch)\b", norm):
        return "no"
    return None


def prompt_requires_explicit_polarity(prompt: str, expected_polarity: str | None) -> bool:
    if not expected_polarity:
        return False
    norm = normalize(prompt)
    return bool(
        re.match(
            r"^(ist|sind|war|waren|hat|haben|kann|koennen|sollte|soll|stammt|gilt|darf)\b",
            norm,
        )
    )


def contains_all(norm_answer: str, words: list[str]) -> bool:
    return all(normalize(word) in norm_answer for word in words)


def evaluate_one(result: dict[str, Any]) -> dict[str, Any]:
    probe_id = result.get("id", "")
    answer = result.get("answer", "")
    spec = EXPECTED.get(probe_id)
    if not spec:
        return {
            "id": probe_id,
            "semantic_score": None,
            "issues": ["unknown_probe_id"],
            "answer": answer,
        }

    norm_answer = normalize(answer)
    prompt = result.get("prompt", "")
    issues: list[str] = []

    expected_polarity = spec.get("polarity")
    actual_polarity = starts_with_polarity(answer)
    if expected_polarity and actual_polarity and actual_polarity != expected_polarity:
        issues.append(f"wrong_polarity:{actual_polarity}_expected_{expected_polarity}")
    elif expected_polarity and actual_polarity is None and prompt_requires_explicit_polarity(prompt, expected_polarity):
        issues.append(f"missing_explicit_polarity:{expected_polarity}")

    for word in spec.get("must", []):
        if normalize(word) not in norm_answer:
            issues.append(f"missing:{word}")

    any_must = spec.get("any_must", [])
    if any_must and not any(contains_all(norm_answer, group) for group in any_must):
        issues.append(
            "missing_any_group:"
            + "|".join("+".join(group) for group in any_must)
        )

    for phrase in spec.get("forbid", []):
        if normalize(phrase) in norm_answer:
            issues.append(f"forbidden:{phrase}")

    semantic_score = 1.0 if not issues else 0.0
    return {
        "id": probe_id,
        "category": result.get("category"),
        "keyword_score": result.get("score"),
        "semantic_score": semantic_score,
        "expected_polarity": expected_polarity,
        "actual_polarity": actual_polarity,
        "issues": issues,
        "answer": answer,
    }


def load_results(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "results" in data:
        return data["results"]
    if "summary" in data and "results" in data["summary"]:
        return data["summary"]["results"]
    raise ValueError(f"{path} does not look like a capability probe result JSON")


def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [item for item in items if item["semantic_score"] is not None]
    passed = sum(1 for item in scored if item["semantic_score"] == 1.0)
    by_category: dict[str, list[float]] = {}
    for item in scored:
        by_category.setdefault(item.get("category") or "uncategorized", []).append(float(item["semantic_score"]))
    return {
        "semantic_score": passed / len(scored) if scored else 0.0,
        "passed": passed,
        "total": len(scored),
        "by_category": {
            key: sum(values) / len(values)
            for key, values in sorted(by_category.items())
        },
        "failures": [item for item in scored if item["semantic_score"] != 1.0],
    }


def write_markdown(path: Path, source: Path, summary: dict[str, Any]) -> None:
    lines = [
        f"# Semantic Response Gate: {source.name}",
        "",
        f"- semantic_score: {summary['semantic_score']:.3f}",
        f"- passed: {summary['passed']} / {summary['total']}",
        "",
        "## By Category",
        "",
    ]
    for category, score in summary["by_category"].items():
        lines.append(f"- {category}: {score:.3f}")
    lines.extend(["", "## Failures", ""])
    for item in summary["failures"]:
        issues = ", ".join(item["issues"])
        lines.append(f"### {item['id']}")
        lines.append(f"- keyword_score: {item['keyword_score']}")
        lines.append(f"- issues: {issues}")
        lines.append(f"- answer: {item['answer']}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    items = [evaluate_one(result) for result in load_results(args.input)]
    summary = summarize(items)
    payload = {
        "source": str(args.input),
        "summary": summary,
        "results": items,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown(args.output_md, args.input, summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
