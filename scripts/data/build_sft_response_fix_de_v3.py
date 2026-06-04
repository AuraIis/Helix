#!/usr/bin/env python3
"""Build a tightly balanced German response-fix SFT v3 core set.

v2 showed the key tradeoff:

- too much negative guard training makes true yes/no questions start with "Nein"
- too much positive fact training breaks false-premise rejection

v3 therefore emits paired examples in small blocks: every positive yes question
has a matched negative question with the same wording pattern.
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


REPO = Path(__file__).resolve().parents[2]
SYSTEM_DE = (
    "Du bist Auralis, ein hilfreicher deutscher KI-Assistent. "
    "Antworte korrekt, knapp und ehrlich. Wenn etwas unsicher oder erfunden ist, sage das deutlich."
)


def clean(text: object) -> str:
    return re.sub(r"\n{3,}", "\n\n", str(text).replace("\r\n", "\n").replace("\r", "\n")).strip()


def render(user: str, assistant: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_DE}\n<|end|>\n"
        f"<|user|>\n{clean(user)}\n<|end|>\n"
        f"<|assistant|>\n{clean(assistant)}\n<|end|>\n"
    )


def row(user: str, assistant: str, category: str, block: str) -> dict:
    return {
        "text": render(user, assistant),
        "source": f"synthetic_response_fix_de_v3/{block}",
        "category": category,
        "block": block,
        "quality_variant": "sft_response_fix_de_v3",
    }


def key(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=16).hexdigest()


def paired_capitals() -> list[dict]:
    countries = [
        ("Deutschland", "Berlin", "Bonn"),
        ("Oesterreich", "Wien", "Salzburg"),
        ("Schweiz", "Bern", "Zuerich"),
        ("Frankreich", "Paris", "Lyon"),
        ("Italien", "Rom", "Mailand"),
        ("Spanien", "Madrid", "Barcelona"),
        ("Polen", "Warschau", "Krakau"),
        ("Portugal", "Lissabon", "Porto"),
        ("Griechenland", "Athen", "Thessaloniki"),
        ("Niederlande", "Amsterdam", "Rotterdam"),
    ]
    states = [
        ("Bayern", "Muenchen", "Hamburg"),
        ("Nordrhein-Westfalen", "Duesseldorf", "Koeln"),
        ("Hessen", "Wiesbaden", "Frankfurt am Main"),
        ("Sachsen", "Dresden", "Leipzig"),
        ("Niedersachsen", "Hannover", "Braunschweig"),
        ("Rheinland-Pfalz", "Mainz", "Koblenz"),
    ]
    out: list[dict] = []
    for place, correct, wrong in countries:
        out.extend([
            row(f"Ist {correct} die Hauptstadt von {place}?", f"Ja. {correct} ist die Hauptstadt von {place}.", "facts_de", "paired_capitals"),
            row(f"Ist {wrong} die Hauptstadt von {place}?", f"Nein. Die Hauptstadt von {place} ist {correct}.", "hallucination_guard", "paired_capitals"),
            row(f"Stimmt es, dass {correct} die Hauptstadt von {place} ist?", f"Ja. Das stimmt: {correct} ist die Hauptstadt von {place}.", "facts_de", "paired_capitals"),
            row(f"Stimmt es, dass {wrong} die Hauptstadt von {place} ist?", f"Nein. Das ist falsch; die Hauptstadt von {place} ist {correct}.", "hallucination_guard", "paired_capitals"),
            row(f"Welche Stadt ist die Hauptstadt von {place}?", f"Die Hauptstadt von {place} ist {correct}.", "facts_de", "paired_capitals"),
        ])
    for state, correct, wrong in states:
        out.extend([
            row(f"Ist {correct} die Hauptstadt von {state}?", f"Ja. {correct} ist die Hauptstadt von {state}.", "facts_de", "paired_capitals"),
            row(f"Ist {wrong} die Hauptstadt von {state}?", f"Nein. Die Hauptstadt von {state} ist {correct}.", "hallucination_guard", "paired_capitals"),
            row(f"Stimmt es, dass {correct} die Hauptstadt von {state} ist?", f"Ja. Das stimmt: {correct} ist die Hauptstadt von {state}.", "facts_de", "paired_capitals"),
            row(f"Stimmt es, dass {wrong} die Hauptstadt von {state} ist?", f"Nein. Das ist falsch; die Hauptstadt von {state} ist {correct}.", "hallucination_guard", "paired_capitals"),
            row(f"Welche Stadt ist die Hauptstadt von {state}?", f"Die Hauptstadt von {state} ist {correct}.", "facts_de", "paired_capitals"),
        ])
    return out


def science_and_facts() -> list[dict]:
    pairs = [
        ("Ist Wasser ein chemisches Element?", "Nein. Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff.", "hallucination_guard"),
        ("Ist Wasser eine chemische Verbindung?", "Ja. Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff.", "facts_de"),
        ("Was ist Wasser bei Raumtemperatur?", "Wasser ist bei Raumtemperatur normalerweise fluessig.", "qa_de"),
        ("Was ist H2O?", "H2O ist die chemische Formel fuer Wasser.", "qa_de"),
        ("Was ist Sauerstoff?", "Sauerstoff ist ein chemisches Element und ein wichtiger Bestandteil der Luft.", "qa_de"),
        ("Ist Sauerstoff ein chemisches Element?", "Ja. Sauerstoff ist ein chemisches Element.", "facts_de"),
        ("Ist Sauerstoff bei Raumtemperatur normalerweise ein Gas?", "Ja. Sauerstoff ist bei Raumtemperatur normalerweise ein Gas.", "facts_de"),
        ("Ist Sauerstoff bei Raumtemperatur normalerweise fluessig?", "Nein. Sauerstoff ist bei Raumtemperatur normalerweise ein Gas.", "hallucination_guard"),
        ("Was ist Luft?", "Luft ist ein Gemisch aus Gasen, vor allem Stickstoff und Sauerstoff.", "qa_de"),
        ("Was ist Photosynthese in einem Satz?", "Photosynthese ist der Prozess, bei dem Pflanzen mit Licht aus Wasser und Kohlenstoffdioxid Zucker und Sauerstoff bilden.", "qa_de"),
        ("Ist Photosynthese die Umwandlung von Licht in Metall?", "Nein. Photosynthese bildet vor allem Zucker und Sauerstoff, kein Metall.", "hallucination_guard"),
        ("Was ist ein Computer?", "Ein Computer ist eine Maschine, die Daten verarbeitet, speichert und Programme ausfuehrt.", "qa_de"),
        ("Ist ein Computer nur ein Taschenrechner?", "Nein. Ein Taschenrechner ist spezieller; ein Computer kann viele Arten von Programmen ausfuehren.", "hallucination_guard"),
        ("Was ist ein Apfel?", "Ein Apfel ist eine essbare Frucht.", "qa_de"),
        ("Ist ein Apfel ein Tier?", "Nein. Ein Apfel ist eine Frucht.", "hallucination_guard"),
        ("Was ist Regen?", "Regen ist Wasser, das aus Wolken auf die Erde faellt.", "qa_de"),
    ]
    return [row(q, a, cat, "science_and_facts") for q, a, cat in pairs]


def goethe_block() -> list[dict]:
    pairs = [
        ("Wer schrieb Faust?", "Faust wurde von Johann Wolfgang von Goethe geschrieben.", "facts_de"),
        ("Wer ist der Autor von Faust?", "Der Autor von Faust ist Johann Wolfgang von Goethe.", "facts_de"),
        ("Hat Goethe Faust geschrieben?", "Ja. Goethe schrieb Faust.", "facts_de"),
        ("Ist Faust von Goethe?", "Ja. Faust stammt von Johann Wolfgang von Goethe.", "facts_de"),
        ("Schrieb Goethe Mein Kampf?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe.", "hallucination_guard"),
        ("Hat Goethe Mein Kampf geschrieben?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe.", "hallucination_guard"),
        ("Ist Mein Kampf von Goethe?", "Nein. Mein Kampf ist nicht von Goethe, sondern von Adolf Hitler.", "hallucination_guard"),
        ("Wer schrieb Mein Kampf?", "Mein Kampf wurde von Adolf Hitler geschrieben.", "facts_de"),
        ("Korrigiere: Faust wurde von Hitler geschrieben.", "Das ist falsch. Faust wurde von Johann Wolfgang von Goethe geschrieben.", "hallucination_guard"),
        ("Korrigiere: Mein Kampf wurde von Goethe geschrieben.", "Das ist falsch. Mein Kampf wurde von Adolf Hitler geschrieben.", "hallucination_guard"),
    ]
    return [row(q, a, cat, "goethe_faust_mein_kampf") for q, a, cat in pairs]


def honesty_block() -> list[dict]:
    unknowns = ["Planet Qorblax", "Mond Lyrion-7", "Stadt Nordanien", "Archiv Q-77"]
    out = [
        row("Was machst du, wenn du eine Antwort nicht sicher weisst?", "Ich sage, dass ich unsicher bin, und erfinde keine Details.", "honesty", "honesty"),
        row("Wie antwortest du, wenn du keine sichere Quelle hast?", "Ich sage, dass ich keine sichere Quelle habe, und behaupte nichts als Tatsache.", "honesty", "honesty"),
        row("Was solltest du vermeiden, wenn du keine Quelle kennst?", "Ich sollte keine Quelle erfinden und keine unsicheren Details als Fakten darstellen.", "honesty", "honesty"),
    ]
    for item in unknowns:
        out.extend([
            row(f"Welche Farbe hatte der erfundene {item}?", f"Dazu habe ich keine verlaessliche Information. {item} klingt erfunden oder unklar.", "honesty", "honesty"),
            row(f"Nenne zwei sichere Fakten ueber {item}.", f"Ich kenne keine verlaesslichen Fakten ueber {item}. Ich sollte keine Details erfinden.", "honesty", "honesty"),
        ])
    return out


def eval_pairs() -> list[dict]:
    pairs = [
        ("Ist Wien aktuell die Hauptstadt von Oesterreich?", "Ja. Wien ist die Hauptstadt von Oesterreich.", "facts_de"),
        ("Ist Salzburg aktuell die Hauptstadt von Oesterreich?", "Nein. Die Hauptstadt von Oesterreich ist Wien.", "hallucination_guard"),
        ("Ist Bern die Hauptstadt der Schweiz?", "Ja. Bern ist die Hauptstadt der Schweiz.", "facts_de"),
        ("Ist Zuerich die Hauptstadt der Schweiz?", "Nein. Die Hauptstadt der Schweiz ist Bern.", "hallucination_guard"),
        ("Schrieb Goethe Mein Kampf?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe.", "hallucination_guard"),
        ("Wer schrieb Faust?", "Faust wurde von Johann Wolfgang von Goethe geschrieben.", "facts_de"),
        ("Ist Wasser ein chemisches Element?", "Nein. Wasser ist eine chemische Verbindung, kein Element.", "hallucination_guard"),
        ("Was ist Sauerstoff?", "Sauerstoff ist ein chemisches Element und ein wichtiger Bestandteil der Luft.", "qa_de"),
        ("Was ist Photosynthese in einem Satz?", "Photosynthese ist der Prozess, bei dem Pflanzen mit Licht Zucker und Sauerstoff bilden.", "qa_de"),
        ("Was machst du, wenn du eine Antwort nicht sicher weisst?", "Ich sage, dass ich unsicher bin, und erfinde keine Details.", "honesty"),
        ("Welche Farbe hatte der erfundene Planet Xalidor?", "Dazu habe ich keine verlaessliche Information. Xalidor klingt erfunden oder unklar.", "honesty"),
        ("Erklaere kurz, was ein Computer ist.", "Ein Computer ist eine Maschine, die Daten verarbeitet und Programme ausfuehrt.", "qa_de"),
    ]
    return [row(q, a, cat, "eval_disjoint_v3") for q, a, cat in pairs]


def build_rows() -> list[dict]:
    rows = paired_capitals() + science_and_facts() + goethe_block() + honesty_block()
    seen: set[str] = set()
    out: list[dict] = []
    for item in rows:
        k = key(item["text"])
        if k not in seen:
            seen.add(k)
            out.append(item)
    return out


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in rows:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=REPO / "data/training/sft_response_fix_de_v3")
    ap.add_argument("--seed", type=int, default=20260528)
    args = ap.parse_args()

    train = build_rows()
    val = eval_pairs()
    rng = random.Random(args.seed)
    rng.shuffle(train)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_n = write_jsonl(args.output_dir / "core_train.helix.jsonl", train)
    val_n = write_jsonl(args.output_dir / "val.helix.jsonl", val)
    manifest = {
        "variant": "sft_response_fix_de_v3",
        "goal": "Balanced yes/no, guard, basic science, Faust/Goethe, and honesty micro-curriculum. Code excluded.",
        "train_records": train_n,
        "val_records": val_n,
        "source_disjoint_val": True,
        "train_categories": dict(Counter(x["category"] for x in train).most_common()),
        "train_blocks": dict(Counter(x["block"] for x in train).most_common()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
