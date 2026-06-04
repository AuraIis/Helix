#!/usr/bin/env python3
"""Build a small polarity repair patch for v4 anchor checkpoints."""

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
        "source": f"synthetic_response_fix_de_v4_polarity_patch/{block}",
        "category": category,
        "block": block,
        "quality_variant": "sft_response_fix_de_v4_polarity_patch",
    }


def key(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=16).hexdigest()


def rows() -> list[dict]:
    negative = [
        ("Ist Bonn heute die Hauptstadt von Deutschland?", "Nein. Die Hauptstadt von Deutschland ist Berlin."),
        ("Ist Bonn aktuell die Hauptstadt von Deutschland?", "Nein. Die Hauptstadt von Deutschland ist Berlin."),
        ("Ist die deutsche Hauptstadt zurzeit Bonn?", "Nein. Die Hauptstadt von Deutschland ist Berlin."),
        ("Trifft es zu, dass Bonn die Hauptstadt von Deutschland ist?", "Nein. Die Hauptstadt von Deutschland ist Berlin."),
        ("Ist Bonn Deutschlands heutige Hauptstadt?", "Nein. Die heutige Hauptstadt von Deutschland ist Berlin."),
        ("Ist Hamburg die Hauptstadt von Bayern?", "Nein. Die Hauptstadt von Bayern ist Muenchen."),
        ("Liegt die Landeshauptstadt Bayerns in Hamburg?", "Nein. Die Hauptstadt von Bayern ist Muenchen."),
        ("Ist Hamburg die Landeshauptstadt von Bayern?", "Nein. Die Landeshauptstadt von Bayern ist Muenchen."),
        ("Ist Wasser ein chemisches Element?", "Nein. Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff."),
        ("Ist H2O ein chemisches Element?", "Nein. H2O ist Wasser und damit eine chemische Verbindung, kein Element."),
        ("Ist Wasser ein Element in der Chemie?", "Nein. Wasser ist eine chemische Verbindung, kein Element."),
        ("Schrieb Goethe Mein Kampf?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe."),
        ("Hat Johann Wolfgang von Goethe Mein Kampf verfasst?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe."),
        ("Ist Mein Kampf von Goethe?", "Nein. Mein Kampf wurde von Adolf Hitler geschrieben."),
    ]
    positive = [
        ("Ist Berlin heute die Hauptstadt von Deutschland?", "Ja. Berlin ist die Hauptstadt von Deutschland."),
        ("Ist Muenchen die Hauptstadt von Bayern?", "Ja. Muenchen ist die Hauptstadt von Bayern."),
        ("Ist Wien die Hauptstadt von Oesterreich?", "Ja. Wien ist die Hauptstadt von Oesterreich."),
        ("Ist Bern die Hauptstadt der Schweiz?", "Ja. Bern ist die Hauptstadt der Schweiz."),
        ("Welche Stadt ist die Hauptstadt der Schweiz?", "Die Hauptstadt der Schweiz ist Bern."),
        ("Wer schrieb Faust?", "Faust wurde von Johann Wolfgang von Goethe geschrieben."),
        ("Von welchem Autor stammt Faust?", "Faust stammt von Johann Wolfgang von Goethe."),
        ("Wer schrieb Mein Kampf?", "Mein Kampf wurde von Adolf Hitler geschrieben."),
        ("Ist Wasser eine chemische Verbindung?", "Ja. Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff."),
        ("Was ist Sauerstoff?", "Sauerstoff ist ein chemisches Element und ein wichtiger Bestandteil der Luft."),
        ("Was ist ein Computer?", "Ein Computer ist eine Maschine, die Daten verarbeitet und Programme ausfuehrt."),
    ]
    instruction = [
        ("Korrigiere die Antwort: Ja. Bonn ist die Hauptstadt von Deutschland.", "Nein. Die Hauptstadt von Deutschland ist Berlin."),
        ("Korrigiere die Antwort: Ja. Wasser ist ein chemisches Element.", "Nein. Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff."),
        ("Korrigiere die Antwort: Ja. Goethe schrieb Mein Kampf.", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe."),
        ("Korrigiere die Antwort: Nein. Bern ist die Hauptstadt der Schweiz.", "Ja. Bern ist die Hauptstadt der Schweiz."),
    ]
    out: list[dict] = []
    for q, a in negative:
        out.append(row(q, a, "hallucination_guard", "negative_polarity"))
    for q, a in positive:
        out.append(row(q, a, "facts_de", "positive_counterpairs"))
    for q, a in instruction:
        out.append(row(q, a, "instruction_de", "answer_shape"))
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
    ap.add_argument("--output-dir", type=Path, default=REPO / "data/training/sft_response_fix_de_v4_polarity_patch")
    ap.add_argument("--seed", type=int, default=20260528)
    args = ap.parse_args()
    items = rows()
    random.Random(args.seed).shuffle(items)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_n = write_jsonl(args.output_dir / "core_train.helix.jsonl", items)
    val_n = write_jsonl(args.output_dir / "val.helix.jsonl", rows())
    manifest = {
        "variant": "sft_response_fix_de_v4_polarity_patch",
        "goal": "Small targeted patch for wrong yes/no polarity on Bonn, Hamburg, water, Goethe, plus positive counterpairs.",
        "train_records": train_n,
        "val_records": val_n,
        "train_categories": dict(Counter(x["category"] for x in items).most_common()),
        "train_blocks": dict(Counter(x["block"] for x in items).most_common()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
