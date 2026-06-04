#!/usr/bin/env python3
"""Small targeted repair set for the v6 bridge checkpoint.

This patch targets two fresh-holdout misses from v6:
- current-capital questions about Bonn must answer Berlin, not "yes".
- photosynthesis must mention plants and the correct light -> sugar/oxygen process.

The exact v4 fresh-holdout prompts are intentionally not included here. Promotion
must use a newer fresh holdout gate after training on this patch.
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
        "source": f"synthetic_response_fix_de_v7_bonn_photo_patch/{block}",
        "category": category,
        "block": block,
        "family": family or block,
        "quality_variant": "sft_response_fix_de_v7_bonn_photo_patch",
    }


def key(text: str) -> str:
    return hashlib.blake2b(re.sub(r"\s+", " ", text.lower()).encode("utf-8"), digest_size=16).hexdigest()


def rows() -> list[dict]:
    bonn_current = [
        ("Ist Bonn momentan die Hauptstadt Deutschlands?", "Nein. Die Hauptstadt von Deutschland ist Berlin."),
        ("Welche Stadt ist heute die deutsche Hauptstadt, Bonn oder Berlin?", "Heute ist Berlin die Hauptstadt von Deutschland."),
        ("Darf man Bonn noch als Deutschlands Hauptstadt nennen?", "Nein. Bonn war frueher Hauptstadt; heute ist Berlin die Hauptstadt von Deutschland."),
        ("Ist Bonn aktuell Hauptstadt der Bundesrepublik?", "Nein. Aktuell ist Berlin die Hauptstadt Deutschlands."),
        ("Wenn jemand Bonn als deutsche Hauptstadt nennt, ist das heute korrekt?", "Nein. Heute ist Berlin die Hauptstadt von Deutschland."),
        ("Ist Bonn nur die ehemalige Hauptstadt Deutschlands?", "Ja. Bonn war frueher Hauptstadt; heute ist Berlin die Hauptstadt Deutschlands."),
        ("Welche Rolle hat Bonn im Vergleich zu Berlin?", "Bonn war frueher Hauptstadt. Heute ist Berlin die Hauptstadt Deutschlands."),
        ("Ist Berlin oder Bonn die aktuelle Hauptstadt von Deutschland?", "Berlin ist die aktuelle Hauptstadt von Deutschland."),
    ]
    photosynthesis = [
        ("Erklaere Photosynthese in einem Satz.", "Photosynthese ist der Prozess, bei dem Pflanzen mit Licht aus Wasser und Kohlendioxid Zucker bilden und Sauerstoff freisetzen."),
        ("Wie gewinnen Pflanzen bei Photosynthese Energie?", "Pflanzen nutzen Licht, um aus Wasser und Kohlendioxid Zucker herzustellen; dabei entsteht Sauerstoff."),
        ("Was machen Pflanzen bei der Photosynthese?", "Pflanzen wandeln mit Licht Wasser und Kohlendioxid in Zucker um und geben Sauerstoff ab."),
        ("Welche Stoffe entstehen bei Photosynthese?", "Bei der Photosynthese bilden Pflanzen Zucker und Sauerstoff."),
        ("Was ist Photosynthese einfach erklaert?", "Photosynthese bedeutet: Pflanzen nutzen Licht, um Zucker zu bilden, und setzen Sauerstoff frei."),
        ("Warum ist Licht fuer Photosynthese wichtig?", "Licht liefert die Energie, mit der Pflanzen Zucker herstellen und Sauerstoff freisetzen."),
        ("Welche Lebewesen betreiben typischerweise Photosynthese?", "Vor allem Pflanzen, Algen und manche Bakterien betreiben Photosynthese."),
        ("Stimmt es, dass bei Photosynthese Sauerstoff entsteht?", "Ja. Pflanzen bilden bei der Photosynthese Zucker und setzen Sauerstoff frei."),
    ]
    keepalive = [
        ("Ist Hamburg aktuell die Hauptstadt von Bayern?", "Nein. Die Hauptstadt von Bayern ist Muenchen."),
        ("Ist Wasser ein chemisches Element?", "Nein. Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff."),
        ("Hat Goethe Mein Kampf geschrieben?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe."),
        ("Wer schrieb Faust?", "Faust wurde von Johann Wolfgang von Goethe geschrieben."),
        ("Ist Wien die Hauptstadt von Oesterreich?", "Ja. Wien ist die Hauptstadt von Oesterreich."),
        ("Ist Bern die Hauptstadt der Schweiz?", "Ja. Bern ist die Hauptstadt der Schweiz."),
        ("Was ist Sauerstoff?", "Sauerstoff ist ein chemisches Element und ein wichtiger Bestandteil der Luft."),
        ("Was ist ein Computer?", "Ein Computer ist eine Maschine, die Daten verarbeitet und Programme ausfuehrt."),
    ]
    honesty = [
        ("Welche Farbe hatte der erfundene Planet Zelvorn?", "Dazu habe ich keine verlaessliche Information. Zelvorn klingt erfunden oder unklar."),
        ("Kennst du sichere Fakten ueber den Fantasieort Neralon?", "Nein. Dazu habe ich keine verlaesslichen Fakten und sollte nichts erfinden."),
        ("Wie antwortest du, wenn eine Frage erfundene Details enthaelt?", "Ich sage, dass ich keine verlaessliche Information habe, und erfinde keine Details."),
    ]

    out: list[dict] = []
    out.extend(row(q, a, "hallucination_guard", "bonn_current_repair", "bonn_current") for q, a in bonn_current)
    out.extend(row(q, a, "qa_de", "photosynthesis_repair", "photosynthesis") for q, a in photosynthesis)
    out.extend(row(q, a, "hallucination_guard", "guard_keepalive", "guard_keepalive") for q, a in keepalive[:3])
    out.extend(row(q, a, "facts_de", "facts_keepalive", "facts_keepalive") for q, a in keepalive[3:6])
    out.extend(row(q, a, "qa_de", "qa_keepalive", "qa_keepalive") for q, a in keepalive[6:])
    out.extend(row(q, a, "honesty", "honesty_keepalive", "honesty_keepalive") for q, a in honesty)
    return out


def write_jsonl(path: Path, items: list[dict]) -> int:
    seen: set[str] = set()
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            k = key(item["text"])
            if k in seen:
                continue
            seen.add(k)
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=REPO / "data/training/sft_response_fix_de_v7_bonn_photo_patch")
    ap.add_argument("--seed", type=int, default=20260528)
    args = ap.parse_args()

    items = rows()
    random.Random(args.seed).shuffle(items)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_n = write_jsonl(args.output_dir / "core_train.helix.jsonl", items)
    val_n = write_jsonl(args.output_dir / "val.helix.jsonl", rows())
    manifest = {
        "variant": "sft_response_fix_de_v7_bonn_photo_patch",
        "goal": "Repair Bonn-current and photosynthesis drift without using exact v4 fresh-holdout prompts.",
        "train_records": train_n,
        "val_records": val_n,
        "train_categories": dict(Counter(x["category"] for x in items).most_common()),
        "train_blocks": dict(Counter(x["block"] for x in items).most_common()),
        "train_families": dict(Counter(x["family"] for x in items).most_common()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
