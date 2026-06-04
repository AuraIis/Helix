#!/usr/bin/env python3
"""Source-disjoint bridge data for Photosynthese and Faust failures.

The v8-safe checkpoint and the first hybrid run still fail on two concepts:

- Photosynthese sometimes generates the memorized bad phrase "Licht aus Licht".
- Faust sometimes falls back to uncertainty instead of naming Goethe.

This builder creates non-exact paraphrase anchors for those concepts and mixes
them with the stable v8 data. Exact prompts from the known v2-v6 gates are
excluded so those gates remain useful diagnostics for this experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import yaml

REPO = Path(__file__).resolve().parents[2]
SYSTEM_DE = (
    "Du bist Auralis, ein hilfreicher deutscher KI-Assistent. "
    "Antworte korrekt, knapp und ehrlich. Wenn etwas unsicher oder erfunden ist, sage das deutlich."
)
BASE_TRAIN = REPO / "data/training/sft_response_fix_de_v8_stable_mix/core_train.helix.jsonl"
BASE_VAL = REPO / "data/training/sft_response_fix_de_v8_stable_mix/val.helix.jsonl"
GATE_FILES = [
    REPO / "eval/sft_response_fix_chat_gate_v2.yaml",
    REPO / "eval/sft_response_fix_chat_gate_v3_holdout.yaml",
    REPO / "eval/sft_response_fix_chat_gate_v4_fresh_holdout.yaml",
    REPO / "eval/sft_response_fix_chat_gate_v5_fresh_holdout.yaml",
    REPO / "eval/sft_response_fix_chat_gate_v6_fresh_holdout.yaml",
]


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def key(text: str) -> str:
    return hashlib.blake2b(norm_text(text).encode("utf-8"), digest_size=16).hexdigest()


def render(user: str, assistant: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_DE}\n<|end|>\n"
        f"<|user|>\n{user.strip()}\n<|end|>\n"
        f"<|assistant|>\n{assistant.strip()}\n<|end|>\n"
    )


def row(user: str, assistant: str, category: str, block: str, family: str) -> dict:
    return {
        "text": render(user, assistant),
        "source": f"synthetic_response_fix_de_v12_photo_faust_bridge/{block}",
        "category": category,
        "block": block,
        "family": family,
        "quality_variant": "sft_response_fix_de_v12_photo_faust_bridge",
    }


def load_gate_prompts() -> set[str]:
    prompts: set[str] = set()
    for path in GATE_FILES:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for probe in data.get("probes", []):
            prompts.add(norm_text(str(probe["prompt"])))
    return prompts


def photo_prompts() -> list[str]:
    starts = [
        "Erklaere kurz",
        "Beschreibe knapp",
        "Sag in einem Satz",
        "Was bedeutet",
        "Wie funktioniert",
        "Was machen Pflanzen durch",
        "Warum brauchen Pflanzen Licht fuer",
        "Wozu dient",
        "Welche Rolle spielt Licht bei",
        "Was entsteht typischerweise bei",
    ]
    subjects = [
        "Photosynthese",
        "die Photosynthese",
        "den Photosynthese-Prozess",
        "die Lichtreaktion von Pflanzen",
        "die Zuckerbildung in Pflanzen",
    ]
    endings = [
        "?",
        " einfach.",
        " fuer Kinder.",
        " sachlich.",
        " ohne Fachjargon.",
    ]
    out = [f"{a} {b}{c}" for a in starts for b in subjects for c in endings]
    out.extend(
        [
            "Wie stellen Pflanzen mithilfe von Licht Nahrung her?",
            "Wie bilden gruenen Pflanzen Zucker?",
            "Warum setzen Pflanzen bei Licht Sauerstoff frei?",
            "Was ist der Zusammenhang zwischen Licht, Zucker und Sauerstoff bei Pflanzen?",
            "Welche Stoffe nutzen Pflanzen, um Zucker herzustellen?",
            "Nenne das Ergebnis der Photosynthese bei Pflanzen.",
            "Was passiert in Blaettern, wenn Pflanzen Licht aufnehmen?",
            "Wie kann man Photosynthese biologisch richtig zusammenfassen?",
        ]
    )
    return out


def faust_prompts() -> list[str]:
    starts = [
        "Von wem stammt",
        "Wer schrieb",
        "Welcher Dichter schrieb",
        "Welcher Autor verfasste",
        "Zu welchem Schriftsteller gehoert",
        "Nenne den Schriftsteller hinter",
        "Wer gilt als Autor von",
        "Aus wessen Feder stammt",
    ]
    subjects = [
        "Faust",
        "Goethes Faust",
        "das Drama Faust",
        "Faust I",
        "die Tragodie Faust",
    ]
    endings = ["?", " in einem kurzen Satz.", " knapp.", " eindeutig."]
    out = [f"{a} {b}{c}" for a in starts for b in subjects for c in endings]
    out.extend(
        [
            "Ist Faust ein Werk von Johann Wolfgang von Goethe?",
            "Ordne Faust dem richtigen deutschen Dichter zu.",
            "Welcher klassische deutsche Autor ist mit Faust verbunden?",
            "Wer ist literarisch mit dem Werk Faust verbunden?",
        ]
    )
    return out


def bridge_rows(gate_prompts: set[str]) -> list[dict]:
    photo_answers = [
        "Bei der Photosynthese nutzen Pflanzen Licht, um aus Wasser und Kohlendioxid Zucker zu bilden; dabei wird Sauerstoff frei.",
        "Photosynthese bedeutet: Pflanzen verwenden Lichtenergie, bilden Zucker und geben Sauerstoff ab.",
        "Pflanzen stellen bei der Photosynthese mit Licht Zucker her und setzen Sauerstoff frei.",
        "Durch Photosynthese wandeln Pflanzen Lichtenergie in chemische Energie in Form von Zucker um; Sauerstoff entsteht dabei ebenfalls.",
    ]
    faust_answers = [
        "Faust wurde von Johann Wolfgang von Goethe geschrieben.",
        "Der Autor von Faust ist Johann Wolfgang von Goethe.",
        "Faust stammt von Johann Wolfgang von Goethe.",
        "Das Werk Faust gehoert zu Johann Wolfgang von Goethe.",
    ]
    keepalive = [
        ("Ist Wien die Hauptstadt von Oesterreich?", "Ja. Wien ist die Hauptstadt von Oesterreich.", "facts_de", "polarity_keepalive", "polarity_keepalive"),
        ("Ist Bern die Hauptstadt der Schweiz?", "Ja. Bern ist die Hauptstadt der Schweiz.", "facts_de", "polarity_keepalive", "polarity_keepalive"),
        ("Ist Bonn aktuell Deutschlands Hauptstadt?", "Nein. Heute ist Berlin die Hauptstadt von Deutschland.", "hallucination_guard", "capital_keepalive", "capital_keepalive"),
        ("Ist Hamburg Bayerns Hauptstadt?", "Nein. Die Hauptstadt von Bayern ist Muenchen.", "hallucination_guard", "capital_keepalive", "capital_keepalive"),
        ("Ist Wasser ein Element?", "Nein. Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff.", "qa_de", "science_keepalive", "science_keepalive"),
        ("Wie antwortest du bei erfundenen Details?", "Ich sage, dass ich keine verlaessliche Information habe, und erfinde keine Details.", "honesty", "honesty_keepalive", "honesty_keepalive"),
    ]
    out: list[dict] = []
    for i, prompt in enumerate(photo_prompts()):
        if norm_text(prompt) not in gate_prompts:
            out.append(row(prompt, photo_answers[i % len(photo_answers)], "qa_de", "photosynthesis_bridge", "photosynthesis_bridge"))
    for i, prompt in enumerate(faust_prompts()):
        if norm_text(prompt) not in gate_prompts:
            out.append(row(prompt, faust_answers[i % len(faust_answers)], "facts_de", "faust_goethe_bridge", "faust_goethe_bridge"))
    for prompt, answer, category, block, family in keepalive:
        if norm_text(prompt) not in gate_prompts:
            out.append(row(prompt, answer, category, block, family))
    return out


def load_jsonl(path: Path, source_prefix: str) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            family = item.get("family") or item.get("block") or item.get("category") or "unknown"
            item["source"] = item.get("source", source_prefix)
            item["family"] = f"{source_prefix}:{family}"
            item["block"] = item.get("block") or family
            rows.append(item)
    return rows


def dedupe(items: Iterable[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in items:
        k = key(item["text"])
        if k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


def write_jsonl(path: Path, items: list[dict]) -> int:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(items)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=REPO / "data/training/sft_response_fix_de_v12_photo_faust_bridge")
    ap.add_argument("--seed", type=int, default=20260529)
    args = ap.parse_args()

    gate_prompts = load_gate_prompts()
    patch = bridge_rows(gate_prompts)
    train = load_jsonl(BASE_TRAIN, "sft_response_fix_de_v8_stable_mix")
    val = load_jsonl(BASE_VAL, "sft_response_fix_de_v8_stable_mix")
    train = dedupe(train + patch)
    random.Random(args.seed).shuffle(train)
    random.Random(args.seed + 1).shuffle(val)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_n = write_jsonl(args.output_dir / "core_train.helix.jsonl", train)
    val_n = write_jsonl(args.output_dir / "val.helix.jsonl", val)
    manifest = {
        "variant": "sft_response_fix_de_v12_photo_faust_bridge",
        "goal": "Add source-disjoint Photosynthese/Faust bridge anchors while preserving v8 stability.",
        "train_records": train_n,
        "val_records": val_n,
        "patch_records": len(patch),
        "exact_gate_prompts_excluded": True,
        "train_categories": dict(Counter(x.get("category", "unknown") for x in train).most_common()),
        "patch_categories": dict(Counter(x.get("category", "unknown") for x in patch).most_common()),
        "inputs": [str(BASE_TRAIN.relative_to(REPO)), "synthetic v12 bridge rows"],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
