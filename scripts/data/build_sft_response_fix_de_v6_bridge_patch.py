#!/usr/bin/env python3
"""Bridge patch for the v5 guard-balanced candidate.

This intentionally absorbs the old v3-holdout failure forms into training.
Promotion must therefore use a fresh holdout gate after this patch.
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


def row(user: str, assistant: str, category: str, block: str) -> dict:
    return {
        "text": render(user, assistant),
        "source": f"synthetic_response_fix_de_v6_bridge_patch/{block}",
        "category": category,
        "block": block,
        "family": block,
        "quality_variant": "sft_response_fix_de_v6_bridge_patch",
    }


def key(text: str) -> str:
    return hashlib.blake2b(re.sub(r"\s+", " ", text.lower()).encode("utf-8"), digest_size=16).hexdigest()


def rows() -> list[dict]:
    positive = [
        ("Gilt Wien als Hauptstadt von Oesterreich?", "Ja. Wien ist die Hauptstadt von Oesterreich."),
        ("Gilt Wien heute als Oesterreichs Hauptstadt?", "Ja. Wien ist die Hauptstadt von Oesterreich."),
        ("Kann man bestaetigen, dass Wien die Hauptstadt von Oesterreich ist?", "Ja. Wien ist die Hauptstadt von Oesterreich."),
        ("Welche Antwort ist korrekt: Ist Bern die Schweizer Hauptstadt?", "Ja. Bern ist die Hauptstadt der Schweiz."),
        ("Ist Bern die Schweizer Hauptstadt?", "Ja. Bern ist die Hauptstadt der Schweiz."),
        ("Kann man bestaetigen, dass Bern die Hauptstadt der Schweiz ist?", "Ja. Bern ist die Hauptstadt der Schweiz."),
        ("Welche Stadt ist die Hauptstadt der Schweiz?", "Die Hauptstadt der Schweiz ist Bern."),
        ("Welche Stadt ist Oesterreichs Hauptstadt?", "Die Hauptstadt von Oesterreich ist Wien."),
    ]
    honesty = [
        ("Wie solltest du antworten, wenn du etwas nicht sicher weisst?", "Ich sage, dass ich unsicher bin, und erfinde keine Details."),
        ("Was tust du bei fehlender sicherer Information?", "Ich sage, dass ich keine sichere Information habe, und erfinde keine Details."),
        ("Welche Farbe hatte der ausgedachte Planet Qorblax?", "Dazu habe ich keine verlaessliche Information. Qorblax klingt erfunden oder unklar."),
        ("Welche Farbe hatte der erfundene Planet Qorblax?", "Dazu habe ich keine verlaessliche Information. Qorblax klingt erfunden oder unklar."),
        ("Kennst du sichere Fakten ueber Qorblax?", "Ich kenne keine verlaesslichen Fakten ueber Qorblax und sollte nichts erfinden."),
        ("Ist Qorblax ein sicher bekannter Planet?", "Nein. Qorblax klingt erfunden oder unklar; ich habe keine verlaessliche Information dazu."),
    ]
    guard = [
        ("Ist Bonn heute die Hauptstadt von Deutschland?", "Nein. Die Hauptstadt von Deutschland ist Berlin."),
        ("Ist die deutsche Hauptstadt zurzeit Bonn?", "Nein. Die Hauptstadt von Deutschland ist Berlin."),
        ("Ist Hamburg die Hauptstadt von Bayern?", "Nein. Die Hauptstadt von Bayern ist Muenchen."),
        ("Liegt die Landeshauptstadt Bayerns in Hamburg?", "Nein. Die Hauptstadt von Bayern ist Muenchen."),
        ("Ist Wasser ein chemisches Element?", "Nein. Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff."),
        ("Hat Goethe Mein Kampf verfasst?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe."),
    ]
    qa = [
        ("Definiere Photosynthese kurz.", "Photosynthese ist der Prozess, bei dem Pflanzen mit Licht Zucker und Sauerstoff bilden."),
        ("Was ist Sauerstoff kurz?", "Sauerstoff ist ein chemisches Element und ein wichtiger Bestandteil der Luft."),
        ("Was ist ein Computer einfach gesagt?", "Ein Computer ist eine Maschine, die Daten verarbeitet und Programme ausfuehrt."),
    ]
    out = []
    out.extend(row(q, a, "facts_de", "positive_bridge") for q, a in positive)
    out.extend(row(q, a, "honesty", "honesty_bridge") for q, a in honesty)
    out.extend(row(q, a, "hallucination_guard", "guard_keepalive") for q, a in guard)
    out.extend(row(q, a, "qa_de", "qa_keepalive") for q, a in qa)
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
    ap.add_argument("--output-dir", type=Path, default=REPO / "data/training/sft_response_fix_de_v6_bridge_patch")
    ap.add_argument("--seed", type=int, default=20260528)
    args = ap.parse_args()
    items = rows()
    random.Random(args.seed).shuffle(items)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_n = write_jsonl(args.output_dir / "core_train.helix.jsonl", items)
    val_n = write_jsonl(args.output_dir / "val.helix.jsonl", rows())
    manifest = {
        "variant": "sft_response_fix_de_v6_bridge_patch",
        "goal": "Absorb old holdout-v3 failures, then require fresh holdout gate.",
        "train_records": train_n,
        "val_records": val_n,
        "train_categories": dict(Counter(x["category"] for x in items).most_common()),
        "train_blocks": dict(Counter(x["block"] for x in items).most_common()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
