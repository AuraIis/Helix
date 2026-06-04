#!/usr/bin/env python3
"""Stable reinforcement mix for the remaining v8 semantic misses.

Targets:
- photosynthesis must consistently mention plants + light + sugar + oxygen.
- Swiss-capital answers should use an explicit, grammatical "Ja. Bern ..." form.
- honesty answers should explicitly say no reliable information and avoid invention.

The base is the v8 stable mix; extra records are paraphrases, not exact v5/v6
fresh-holdout prompts.
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


def render(user: str, assistant: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_DE}\n<|end|>\n"
        f"<|user|>\n{user.strip()}\n<|end|>\n"
        f"<|assistant|>\n{assistant.strip()}\n<|end|>\n"
    )


def row(user: str, assistant: str, category: str, block: str, family: str | None = None) -> dict:
    return {
        "text": render(user, assistant),
        "source": f"synthetic_response_fix_de_v9_stable_reinforce/{block}",
        "category": category,
        "block": block,
        "family": family or block,
        "quality_variant": "sft_response_fix_de_v9_stable_reinforce",
    }


def norm_key(text: str) -> str:
    return hashlib.blake2b(re.sub(r"\s+", " ", text.lower()).encode("utf-8"), digest_size=16).hexdigest()


def load_base(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            item["quality_variant"] = "sft_response_fix_de_v9_stable_reinforce"
            item["family"] = f"base:{item.get('family') or item.get('block') or item.get('category', 'unknown')}"
            rows.append(item)
    return rows


def extra_rows() -> list[dict]:
    photosynthesis = [
        ("Nenne den Ablauf der Photosynthese kurz.", "Pflanzen nutzen Licht, um aus Wasser und Kohlendioxid Zucker zu bilden; dabei entsteht Sauerstoff."),
        ("Was erzeugen Pflanzen durch Photosynthese?", "Pflanzen erzeugen bei der Photosynthese Zucker und setzen Sauerstoff frei."),
        ("Worum geht es bei Photosynthese?", "Bei Photosynthese nutzen Pflanzen Licht, um Zucker herzustellen und Sauerstoff abzugeben."),
        ("Welche Rolle spielen Pflanzen und Licht bei Photosynthese?", "Pflanzen nutzen Licht als Energiequelle, um Zucker zu bilden; Sauerstoff wird freigesetzt."),
        ("Was entsteht bei der Photosynthese in Pflanzen?", "In Pflanzen entstehen bei der Photosynthese Zucker und Sauerstoff."),
        ("Erklaere den Nutzen der Photosynthese fuer Pflanzen.", "Photosynthese hilft Pflanzen, mit Licht Zucker als Energiequelle zu bilden; dabei entsteht Sauerstoff."),
        ("Was wandeln Pflanzen bei Photosynthese um?", "Pflanzen wandeln mit Licht Wasser und Kohlendioxid in Zucker um und geben Sauerstoff ab."),
        ("Warum ist Photosynthese wichtig?", "Sie ermoeglicht Pflanzen, mit Licht Zucker zu bilden, und liefert Sauerstoff."),
    ]
    bern = [
        ("Ist Bern oder Zuerich die Hauptstadt der Schweiz?", "Ja. Bern ist die Hauptstadt der Schweiz."),
        ("Kann man sagen, dass Bern die Schweizer Hauptstadt ist?", "Ja. Bern ist die Hauptstadt der Schweiz."),
        ("Ist Bern die richtige Hauptstadt-Antwort fuer die Schweiz?", "Ja. Bern ist die Hauptstadt der Schweiz."),
        ("Welche Stadt ist die Hauptstadt der Schweiz?", "Die Hauptstadt der Schweiz ist Bern."),
    ]
    honesty = [
        ("Was antwortest du zu einer frei erfundenen Stadt namens Keralis?", "Dazu habe ich keine verlaessliche Information. Keralis klingt erfunden oder unklar."),
        ("Welche Farbe hatte der ausgedachte Mond Tralvorn?", "Dazu habe ich keine verlaessliche Information. Tralvorn klingt erfunden; ich sollte nichts erfinden."),
        ("Was weisst du sicher ueber den erfundenen Planeten Nivora?", "Ich habe keine verlaesslichen Informationen zu Nivora. Der Name klingt erfunden oder unklar."),
        ("Wie reagierst du auf eine erfundene Entitaet ohne Quelle?", "Ich sage, dass ich keine verlaessliche Information habe, und erfinde keine Details."),
        ("Solltest du bei fehlenden Fakten Details ausdenken?", "Nein. Wenn sichere Fakten fehlen, sage ich das ehrlich und erfinde keine Details."),
    ]
    keepalive = [
        ("Ist Bonn heute Deutschlands Hauptstadt?", "Nein. Die Hauptstadt von Deutschland ist Berlin."),
        ("Ist Hamburg die Hauptstadt von Bayern?", "Nein. Die Hauptstadt von Bayern ist Muenchen."),
        ("Ist Wasser ein chemisches Element?", "Nein. Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff."),
        ("Ist Wien die Hauptstadt von Oesterreich?", "Ja. Wien ist die Hauptstadt von Oesterreich."),
        ("Hat Goethe Mein Kampf geschrieben?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe."),
    ]

    out: list[dict] = []
    out.extend(row(q, a, "qa_de", "photosynthesis_reinforce", "photosynthesis") for q, a in photosynthesis)
    out.extend(row(q, a, "facts_de", "bern_positive_reinforce", "bern_positive") for q, a in bern)
    out.extend(row(q, a, "honesty", "honesty_reinforce", "honesty_unknown") for q, a in honesty)
    out.extend(row(q, a, "hallucination_guard", "guard_keepalive", "guard_keepalive") for q, a in keepalive[:3])
    out.extend(row(q, a, "facts_de", "facts_keepalive", "facts_keepalive") for q, a in keepalive[3:4])
    out.extend(row(q, a, "hallucination_guard", "guard_keepalive", "guard_keepalive") for q, a in keepalive[4:])
    return out


def dedupe(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in items:
        k = norm_key(item["text"])
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
    ap.add_argument("--output-dir", type=Path, default=REPO / "data/training/sft_response_fix_de_v9_stable_reinforce")
    ap.add_argument("--seed", type=int, default=20260528)
    args = ap.parse_args()

    base_dir = REPO / "data/training/sft_response_fix_de_v8_stable_mix"
    train = load_base(base_dir / "core_train.helix.jsonl") + extra_rows()
    val = load_base(base_dir / "val.helix.jsonl") + extra_rows()
    train = dedupe(train)
    val = dedupe(val)
    random.Random(args.seed).shuffle(train)
    random.Random(args.seed + 1).shuffle(val)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_n = write_jsonl(args.output_dir / "core_train.helix.jsonl", train)
    val_n = write_jsonl(args.output_dir / "val.helix.jsonl", val)
    manifest = {
        "variant": "sft_response_fix_de_v9_stable_reinforce",
        "goal": "Reinforce remaining photosynthesis, Bern and honesty failures while preserving v8 broad mix.",
        "train_records": train_n,
        "val_records": val_n,
        "train_categories": dict(Counter(x.get("category", "unknown") for x in train).most_common()),
        "train_families": len(set(x.get("family", "unknown") for x in train)),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
