#!/usr/bin/env python3
"""Small final balance patch for v4 polarity runs."""

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
        "source": f"synthetic_response_fix_de_v4_balance_patch/{block}",
        "category": category,
        "block": block,
        "quality_variant": "sft_response_fix_de_v4_balance_patch",
    }


def key(text: str) -> str:
    return hashlib.blake2b(re.sub(r"\s+", " ", text.lower()).encode("utf-8"), digest_size=16).hexdigest()


def rows() -> list[dict]:
    positive = [
        ("Ist Wien wirklich die Hauptstadt von Oesterreich?", "Ja. Wien ist die Hauptstadt von Oesterreich."),
        ("Ist Wien nach heutigem Stand Oesterreichs Hauptstadt?", "Ja. Wien ist die Hauptstadt von Oesterreich."),
        ("Kann man die Hauptstadt von Oesterreich Wien nennen?", "Ja. Die Hauptstadt von Oesterreich ist Wien."),
        ("Ist Bern wirklich die Hauptstadt der Schweiz?", "Ja. Bern ist die Hauptstadt der Schweiz."),
        ("Kann man die Schweizer Hauptstadt Bern nennen?", "Ja. Die Hauptstadt der Schweiz ist Bern."),
        ("Ist Bern nach heutigem Stand die Hauptstadt der Schweiz?", "Ja. Bern ist die Hauptstadt der Schweiz."),
        ("Welche Stadt ist die Schweizer Hauptstadt, Bern oder Paris?", "Die Hauptstadt der Schweiz ist Bern, nicht Paris."),
        ("Welche Stadt ist Oesterreichs Hauptstadt, Wien oder Salzburg?", "Die Hauptstadt von Oesterreich ist Wien, nicht Salzburg."),
    ]
    honesty = [
        ("Welche Farbe hatte ein erfundener Planet namens Qorblax?", "Dazu habe ich keine verlaessliche Information. Qorblax klingt erfunden oder unklar."),
        ("Kennst du sichere Fakten ueber Qorblax?", "Ich kenne keine verlaesslichen Fakten ueber Qorblax und sollte nichts erfinden."),
        ("Was sagst du, wenn Qorblax nur ausgedacht ist?", "Dann sage ich, dass Qorblax erfunden oder unklar klingt und ich keine verlaesslichen Fakten habe."),
        ("Wie reagierst du auf eine unsichere Frage?", "Ich sage, dass ich unsicher bin, und erfinde keine Details."),
        ("Solltest du bei Unsicherheit einfach raten?", "Nein. Ich sollte Unsicherheit klar sagen und nicht raten."),
    ]
    guard_keep = [
        ("Ist Bonn heute die Hauptstadt von Deutschland?", "Nein. Die Hauptstadt von Deutschland ist Berlin."),
        ("Ist Wasser ein chemisches Element?", "Nein. Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff."),
        ("Schrieb Goethe Mein Kampf?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe."),
    ]
    out = []
    out.extend(row(q, a, "facts_de", "positive_polarity") for q, a in positive)
    out.extend(row(q, a, "honesty", "honesty_repair") for q, a in honesty)
    out.extend(row(q, a, "hallucination_guard", "guard_keepalive") for q, a in guard_keep)
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
    ap.add_argument("--output-dir", type=Path, default=REPO / "data/training/sft_response_fix_de_v4_balance_patch")
    ap.add_argument("--seed", type=int, default=20260528)
    args = ap.parse_args()
    items = rows()
    random.Random(args.seed).shuffle(items)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_n = write_jsonl(args.output_dir / "core_train.helix.jsonl", items)
    val_n = write_jsonl(args.output_dir / "val.helix.jsonl", rows())
    manifest = {
        "variant": "sft_response_fix_de_v4_balance_patch",
        "goal": "Final small patch for positive yes polarity and honesty while keeping repaired guards alive.",
        "train_records": train_n,
        "val_records": val_n,
        "train_categories": dict(Counter(x["category"] for x in items).most_common()),
        "train_blocks": dict(Counter(x["block"] for x in items).most_common()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
