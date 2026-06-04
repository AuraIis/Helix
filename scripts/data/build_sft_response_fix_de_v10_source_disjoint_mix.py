#!/usr/bin/env python3
"""Build a broader source-disjoint German response repair SFT mix.

This is the next step after v8/v9 showed that tiny patch-SFTs chase gates.
The mix deliberately combines:

- a bounded keepalive slice from v8 stable repair data,
- non-code German SFT records from clean_de_v1,
- hand-written source-disjoint concept anchors for geography, science,
  literature and honesty.

Exact prompts from the current gate files are avoided. Promotion must use fresh
holdouts, not the examples written here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SYSTEM_DE = (
    "Du bist Auralis, ein hilfreicher deutscher KI-Assistent. "
    "Antworte korrekt, knapp und ehrlich. Wenn etwas unsicher oder erfunden ist, sage das deutlich."
)

KEEP_CATEGORIES = {
    "factual_qa",
    "factual_correction",
    "concept_explain",
    "technical_explanation",
    "general_assistant",
    "step_by_step_reason",
    "honest_refusal",
    "clarify_awareness",
    "format_following",
    "rewrite",
    "translation",
    "math_reasoning",
    "math_word_problem",
}
DROP_CATEGORIES = {
    "code_explain",
    "code_refactoring",
    "code_debug_fix",
    "coding_bugfix",
    "coding_generation",
    "coding_explanation",
    "code_implementation",
    "code_general",
}
BAD_TEXT = [
    "```",
    "python",
    "javascript",
    "typescript",
    "function",
    "class ",
    "def ",
    "import ",
    "stack trace",
    "traceback",
    "html",
    "http://",
    "https://",
]


def render(user: str, assistant: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_DE}\n<|end|>\n"
        f"<|user|>\n{user.strip()}\n<|end|>\n"
        f"<|assistant|>\n{assistant.strip()}\n<|end|>\n"
    )


def row(user: str, assistant: str, category: str, block: str, family: str | None = None) -> dict:
    return {
        "text": render(user, assistant),
        "source": f"synthetic_response_fix_de_v10_source_disjoint/{block}",
        "category": category,
        "block": block,
        "family": family or block,
        "quality_variant": "sft_response_fix_de_v10_source_disjoint_mix",
    }


def norm_key(text: str) -> str:
    return hashlib.blake2b(re.sub(r"\s+", " ", text.lower()).encode("utf-8"), digest_size=16).hexdigest()


def assistant_text(text: str) -> str:
    m = re.search(r"<\|assistant\|>\n(.*?)\n<\|end\|>\n?$", text, flags=re.S)
    return m.group(1).strip() if m else text


def keep_clean_sft(rec: dict) -> bool:
    cat = str(rec.get("category", ""))
    if cat in DROP_CATEGORIES:
        return False
    if KEEP_CATEGORIES and cat not in KEEP_CATEGORIES:
        return False
    text = str(rec.get("text", ""))
    low = text.lower()
    if any(x in low for x in BAD_TEXT):
        return False
    answer = assistant_text(text)
    if not (40 <= len(answer) <= 850):
        return False
    if answer.count("\n") > 8:
        return False
    return True


def load_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            out.append(json.loads(line))
            if limit and len(out) >= limit:
                break
    return out


def synthetic_rows() -> list[dict]:
    geo = [
        ("Welche Stadt ist heute Regierungssitz und Hauptstadt Deutschlands?", "Die Hauptstadt Deutschlands ist Berlin. Bonn war frueher Hauptstadt, ist es heute aber nicht mehr."),
        ("Wenn jemand Bonn als aktuelle deutsche Hauptstadt nennt, wie korrigierst du das?", "Das ist heute nicht korrekt. Die Hauptstadt Deutschlands ist Berlin; Bonn war frueher Hauptstadt."),
        ("Welche Hauptstadt gehoert zu Bayern?", "Die Hauptstadt von Bayern ist Muenchen, nicht Hamburg."),
        ("Welche Stadt ist die Hauptstadt Oesterreichs?", "Die Hauptstadt von Oesterreich ist Wien."),
        ("Welche Stadt ist die Hauptstadt der Schweiz?", "Die Hauptstadt der Schweiz ist Bern."),
        ("Ist Zuerich die Hauptstadt der Schweiz?", "Nein. Die Hauptstadt der Schweiz ist Bern."),
    ]
    science = [
        ("Erklaere Photosynthese sehr knapp.", "Photosynthese ist der Prozess, bei dem Pflanzen mit Licht aus Wasser und Kohlendioxid Zucker bilden und Sauerstoff freisetzen."),
        ("Was passiert in Pflanzen bei Photosynthese?", "Pflanzen nutzen Lichtenergie, um Zucker herzustellen; dabei wird Sauerstoff abgegeben."),
        ("Welche Produkte entstehen bei der Photosynthese?", "Bei der Photosynthese entstehen vor allem Zucker und Sauerstoff."),
        ("Warum brauchen Pflanzen Licht fuer Photosynthese?", "Licht liefert die Energie, mit der Pflanzen Zucker bilden und Sauerstoff freisetzen."),
        ("Ist Wasser ein Element?", "Nein. Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff."),
        ("Was ist Sauerstoff?", "Sauerstoff ist ein chemisches Element und ein Gas, das ein wichtiger Bestandteil der Luft ist."),
    ]
    literature = [
        ("Von wem stammt Faust?", "Faust stammt von Johann Wolfgang von Goethe."),
        ("Nenne den Autor des Werks Faust.", "Der Autor von Faust ist Johann Wolfgang von Goethe."),
        ("Hat Goethe Mein Kampf geschrieben?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe."),
        ("Ist Faust ein KI-Assistent?", "Nein. Faust ist ein literarisches Werk von Johann Wolfgang von Goethe."),
    ]
    honesty = [
        ("Was machst du, wenn eine Frage eine erfundene Person enthaelt?", "Ich sage, dass ich keine verlaessliche Information habe, und erfinde keine Details."),
        ("Welche Farbe hatte der erfundene Planet Norvax?", "Dazu habe ich keine verlaessliche Information. Norvax klingt erfunden oder unklar."),
        ("Kennst du sichere Fakten ueber den Ort Belmora?", "Ich habe keine verlaesslichen Informationen zu Belmora. Ich sollte keine Details erfinden."),
        ("Wie antwortest du, wenn du keine Quelle und kein sicheres Wissen hast?", "Ich sage offen, dass ich unsicher bin oder keine verlaessliche Information habe."),
        ("Solltest du eine unbekannte Tatsache einfach plausibel ergaenzen?", "Nein. Ich sollte keine Details erfinden, sondern Unsicherheit klar benennen."),
    ]
    shape = [
        ("Antworte knapp: Was ist ein Computer?", "Ein Computer ist eine Maschine, die Daten verarbeitet und Programme ausfuehrt."),
        ("Erklaere kurz und direkt, was eine Maschine ist.", "Eine Maschine ist ein technisches System, das Arbeit verrichtet oder Prozesse unterstuetzt."),
        ("Formuliere eine unsichere Antwort korrekt.", "Ich bin mir nicht sicher und habe dazu keine verlaessliche Information."),
    ]

    out: list[dict] = []
    out.extend(row(q, a, "facts_de", "geo_anchor", "geo") for q, a in geo)
    out.extend(row(q, a, "qa_de", "science_anchor", "science") for q, a in science)
    out.extend(row(q, a, "facts_de", "literature_anchor", "literature") for q, a in literature)
    out.extend(row(q, a, "honesty", "honesty_anchor", "honesty") for q, a in honesty)
    out.extend(row(q, a, "instruction_de", "answer_shape_anchor", "answer_shape") for q, a in shape)
    return out


def dedupe(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in items:
        k = norm_key(str(item["text"]))
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
    ap.add_argument("--output-dir", type=Path, default=REPO / "data/training/sft_response_fix_de_v10_source_disjoint_mix")
    ap.add_argument("--seed", type=int, default=20260529)
    ap.add_argument("--clean-limit", type=int, default=650)
    ap.add_argument("--keepalive-limit", type=int, default=220)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    clean = [x for x in load_jsonl(REPO / "data/training/sft_clean_de_v1/train.helix.jsonl") if keep_clean_sft(x)]
    by_cat: dict[str, list[dict]] = {}
    for item in clean:
        by_cat.setdefault(str(item.get("category", "unknown")), []).append(item)
    clean_sample: list[dict] = []
    for cat, rows in sorted(by_cat.items()):
        rng.shuffle(rows)
        quota = max(12, args.clean_limit // max(1, len(by_cat)))
        clean_sample.extend(rows[:quota])
    rng.shuffle(clean_sample)
    clean_sample = clean_sample[: args.clean_limit]
    for item in clean_sample:
        item["source"] = f"clean_de_v1/{item.get('category', 'unknown')}"
        item["family"] = f"clean_de_v1:{item.get('category', 'unknown')}"
        item["block"] = str(item.get("category", "clean_de_v1"))
        item["quality_variant"] = "sft_response_fix_de_v10_source_disjoint_mix"

    keepalive = load_jsonl(REPO / "data/training/sft_response_fix_de_v8_stable_mix/core_train.helix.jsonl")
    rng.shuffle(keepalive)
    keepalive = keepalive[: args.keepalive_limit]
    for item in keepalive:
        item["family"] = f"v8_keepalive:{item.get('family') or item.get('block') or item.get('category', 'unknown')}"
        item["quality_variant"] = "sft_response_fix_de_v10_source_disjoint_mix"

    items = dedupe(synthetic_rows() + clean_sample + keepalive)
    rng.shuffle(items)

    val_source = [x for x in load_jsonl(REPO / "data/training/sft_clean_de_v1/val.helix.jsonl") if keep_clean_sft(x)]
    rng.shuffle(val_source)
    val = dedupe(synthetic_rows() + val_source[:80])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_n = write_jsonl(args.output_dir / "core_train.helix.jsonl", items)
    val_n = write_jsonl(args.output_dir / "val.helix.jsonl", val)
    manifest = {
        "variant": "sft_response_fix_de_v10_source_disjoint_mix",
        "goal": "Broader non-code German repair mix before larger SFT; monitored via learning neuro map.",
        "train_records": train_n,
        "val_records": val_n,
        "train_categories": dict(Counter(x.get("category", "unknown") for x in items).most_common()),
        "train_families": len(set(x.get("family", "unknown") for x in items)),
        "sources": {
            "clean_de_v1": len(clean_sample),
            "v8_keepalive": len(keepalive),
            "synthetic_anchors": len(synthetic_rows()),
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
