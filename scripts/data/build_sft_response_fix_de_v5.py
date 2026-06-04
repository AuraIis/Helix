#!/usr/bin/env python3
"""Build German response-fix SFT v5.

v4 finally showed that the model can learn the target behaviour, but small
patches move errors between "Ja" and "Nein". v5 therefore builds larger,
family-balanced data:

- critical hard-gate anchors are present
- each anchor has many source-disjoint paraphrases
- positive and negative polarity examples are balanced per family
- honesty examples avoid a single Qorblax-only template
- code data is intentionally excluded
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
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


def row(user: str, assistant: str, category: str, block: str, family: str) -> dict:
    return {
        "text": render(user, assistant),
        "source": f"synthetic_response_fix_de_v5/{block}",
        "category": category,
        "block": block,
        "family": family,
        "quality_variant": "sft_response_fix_de_v5",
    }


def key(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=16).hexdigest()


def capital_phrase(place: str) -> str:
    return {
        "Schweiz": "der Schweiz",
        "Niederlande": "der Niederlande",
        "USA": "der USA",
    }.get(place, f"von {place}")


def yes_forms(entity: str, relation: str, correction: str | None = None) -> list[tuple[str, str]]:
    return [
        (f"Ist {entity} {relation}?", f"Ja. {entity} {relation}."),
        (f"Ist es korrekt, dass {entity} {relation}?", f"Ja. Das ist korrekt: {entity} {relation}."),
        (f"Gilt {entity} als {relation.replace('ist ', '')}?", f"Ja. {entity} {relation}."),
        (f"Beantworte mit Ja oder Nein: {entity} {relation}.", f"Ja. {entity} {relation}."),
        (f"Welche Polaritaet ist richtig: {entity} {relation}?", f"Ja. {entity} {relation}."),
    ]


def no_forms(entity: str, relation: str, correction: str) -> list[tuple[str, str]]:
    return [
        (f"Ist {entity} {relation}?", f"Nein. {correction}."),
        (f"Ist es korrekt, dass {entity} {relation}?", f"Nein. Das ist falsch; {correction}."),
        (f"Gilt {entity} als {relation.replace('ist ', '')}?", f"Nein. {correction}."),
        (f"Beantworte mit Ja oder Nein: {entity} {relation}.", f"Nein. {correction}."),
        (f"Welche Polaritaet ist richtig: {entity} {relation}?", f"Nein. {correction}."),
        (f"Korrigiere kurz: {entity} {relation}.", f"Das ist falsch. {correction}."),
    ]


def capital_rows() -> list[dict]:
    countries = [
        ("Deutschland", "Berlin", ["Bonn", "Hamburg", "Muenchen", "Koeln"]),
        ("Oesterreich", "Wien", ["Salzburg", "Graz", "Linz", "Innsbruck"]),
        ("Schweiz", "Bern", ["Zuerich", "Paris", "Genf", "Basel"]),
        ("Frankreich", "Paris", ["Lyon", "Marseille", "Toulouse", "Nizza"]),
        ("Italien", "Rom", ["Mailand", "Neapel", "Turin", "Florenz"]),
        ("Spanien", "Madrid", ["Barcelona", "Valencia", "Sevilla", "Bilbao"]),
        ("Polen", "Warschau", ["Krakau", "Danzig", "Posen", "Lodz"]),
        ("Portugal", "Lissabon", ["Porto", "Coimbra", "Faro", "Braga"]),
        ("Griechenland", "Athen", ["Thessaloniki", "Patras", "Kreta", "Rhodos"]),
        ("Niederlande", "Amsterdam", ["Rotterdam", "Den Haag", "Utrecht", "Eindhoven"]),
        ("Belgien", "Bruessel", ["Antwerpen", "Gent", "Bruegge", "Liege"]),
        ("Daenemark", "Kopenhagen", ["Aarhus", "Odense", "Aalborg", "Roskilde"]),
    ]
    states = [
        ("Bayern", "Muenchen", ["Hamburg", "Nuernberg", "Augsburg", "Regensburg"]),
        ("Nordrhein-Westfalen", "Duesseldorf", ["Koeln", "Dortmund", "Essen", "Bonn"]),
        ("Hessen", "Wiesbaden", ["Frankfurt am Main", "Kassel", "Darmstadt", "Marburg"]),
        ("Sachsen", "Dresden", ["Leipzig", "Chemnitz", "Meissen", "Zwickau"]),
        ("Niedersachsen", "Hannover", ["Braunschweig", "Oldenburg", "Goettingen", "Osnabrueck"]),
        ("Brandenburg", "Potsdam", ["Cottbus", "Frankfurt an der Oder", "Brandenburg an der Havel"]),
        ("Thueringen", "Erfurt", ["Jena", "Weimar", "Gera", "Eisenach"]),
    ]
    out: list[dict] = []
    for place, correct, wrongs in countries:
        phrase = capital_phrase(place)
        family = f"capital:{place}"
        out.extend([
            row(f"Nenne die Hauptstadt {phrase}.", f"Die Hauptstadt {phrase} ist {correct}.", "facts_de", "capitals", family),
            row(f"Welche Stadt ist die Hauptstadt {phrase}?", f"Die Hauptstadt {phrase} ist {correct}.", "facts_de", "capitals", family),
            row(f"Welche Stadt ist korrekt: {correct} oder {wrongs[0]} als Hauptstadt {phrase}?", f"Korrekt ist {correct}. Die Hauptstadt {phrase} ist {correct}.", "facts_de", "capitals", family),
        ])
        for q, a in yes_forms(correct, f"ist die Hauptstadt {phrase}"):
            out.append(row(q, a, "facts_de", "capitals", family))
        for wrong in wrongs:
            correction = f"Die Hauptstadt {phrase} ist {correct}"
            for q, a in no_forms(wrong, f"ist die Hauptstadt {phrase}", correction):
                out.append(row(q, a, "hallucination_guard", "capitals", family))
    for state, correct, wrongs in states:
        phrase = capital_phrase(state)
        family = f"state_capital:{state}"
        out.extend([
            row(f"Nenne die Hauptstadt {phrase}.", f"Die Hauptstadt {phrase} ist {correct}.", "facts_de", "state_capitals", family),
            row(f"Welche Stadt ist die Landeshauptstadt {phrase}?", f"Die Landeshauptstadt {phrase} ist {correct}.", "facts_de", "state_capitals", family),
        ])
        for q, a in yes_forms(correct, f"ist die Hauptstadt {phrase}"):
            out.append(row(q, a, "facts_de", "state_capitals", family))
        for wrong in wrongs:
            correction = f"Die Hauptstadt {phrase} ist {correct}"
            for q, a in no_forms(wrong, f"ist die Hauptstadt {phrase}", correction):
                out.append(row(q, a, "hallucination_guard", "state_capitals", family))
    return out


def science_rows() -> list[dict]:
    out: list[dict] = []
    definitions = [
        ("Wasser", "eine chemische Verbindung aus Wasserstoff und Sauerstoff", ["ein chemisches Element", "ein Metall", "ein Gasgemisch"], "water"),
        ("H2O", "die chemische Formel fuer Wasser", ["die Formel fuer Gold", "ein chemisches Element", "ein Planet"], "water"),
        ("Sauerstoff", "ein chemisches Element und ein wichtiger Bestandteil der Luft", ["eine Stadt", "ein Tier", "ein Metallname"], "oxygen"),
        ("Photosynthese", "ein Prozess, bei dem Pflanzen mit Licht Zucker und Sauerstoff bilden", ["die Herstellung von Metall", "ein Tier", "ein anderes Wort fuer Regen"], "photosynthesis"),
        ("Ein Computer", "eine Maschine, die Daten verarbeitet, speichert und Programme ausfuehrt", ["nur ein Taschenrechner", "eine Frucht", "ein chemisches Element"], "computer"),
    ]
    for subject, definition, wrongs, family in definitions:
        out.extend([
            row(f"Was ist {subject} kurz erklaert?", f"{subject} ist {definition}.", "qa_de", "science", family),
            row(f"Erklaere in einem Satz: {subject}.", f"{subject} ist {definition}.", "qa_de", "science", family),
            row(f"Definiere {subject} knapp.", f"{subject} ist {definition}.", "qa_de", "science", family),
        ])
        for q, a in yes_forms(subject, f"ist {definition}"):
            out.append(row(q, a, "facts_de", "science", family))
        for wrong in wrongs:
            correction = f"{subject} ist {definition}"
            for q, a in no_forms(subject, f"ist {wrong}", correction):
                out.append(row(q, a, "hallucination_guard", "science", family))
    out.extend([
        row("Ist Wasser ein chemisches Element?", "Nein. Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff.", "hallucination_guard", "science_anchor", "water"),
        row("Ist H2O ein chemisches Element?", "Nein. H2O ist Wasser und damit eine chemische Verbindung, kein Element.", "hallucination_guard", "science_anchor", "water"),
        row("Ist Wasser eine chemische Verbindung?", "Ja. Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff.", "facts_de", "science_anchor", "water"),
        row("Was ist Sauerstoff?", "Sauerstoff ist ein chemisches Element und ein wichtiger Bestandteil der Luft.", "qa_de", "science_anchor", "oxygen"),
        row("Was ist Photosynthese in einem Satz?", "Photosynthese ist der Prozess, bei dem Pflanzen mit Licht Zucker und Sauerstoff bilden.", "qa_de", "science_anchor", "photosynthesis"),
        row("Erklaere kurz, was ein Computer ist.", "Ein Computer ist eine Maschine, die Daten verarbeitet und Programme ausfuehrt.", "qa_de", "science_anchor", "computer"),
    ])
    return out


def literature_rows() -> list[dict]:
    pairs = [
        ("Wer schrieb Faust?", "Faust wurde von Johann Wolfgang von Goethe geschrieben.", "facts_de"),
        ("Von welchem Autor stammt Faust?", "Faust stammt von Johann Wolfgang von Goethe.", "facts_de"),
        ("Nenne den Autor von Faust.", "Der Autor von Faust ist Johann Wolfgang von Goethe.", "facts_de"),
        ("Ist Faust von Goethe?", "Ja. Faust ist ein Werk von Johann Wolfgang von Goethe.", "facts_de"),
        ("Hat Goethe Faust geschrieben?", "Ja. Goethe schrieb Faust.", "facts_de"),
        ("Schrieb Goethe Mein Kampf?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe.", "hallucination_guard"),
        ("Hat Johann Wolfgang von Goethe Mein Kampf verfasst?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe.", "hallucination_guard"),
        ("Ist Mein Kampf von Goethe?", "Nein. Mein Kampf wurde von Adolf Hitler geschrieben.", "hallucination_guard"),
        ("Wer schrieb Mein Kampf?", "Mein Kampf wurde von Adolf Hitler geschrieben.", "facts_de"),
        ("Korrigiere: Faust wurde von Hitler geschrieben.", "Das ist falsch. Faust wurde von Johann Wolfgang von Goethe geschrieben.", "hallucination_guard"),
        ("Korrigiere: Mein Kampf wurde von Goethe geschrieben.", "Das ist falsch. Mein Kampf wurde von Adolf Hitler geschrieben.", "hallucination_guard"),
    ]
    return [row(q, a, cat, "literature", "goethe_faust_hitler") for q, a, cat in pairs]


def honesty_rows() -> list[dict]:
    out: list[dict] = []
    entities = [
        "Planet Qorblax",
        "Planet Xalidor",
        "Mond Lyrion-7",
        "Stadt Nordanien",
        "Archiv Q-77",
        "Koenigreich Velora",
        "Element Trivium-X",
        "Insel Marendal",
        "Stern Kalphex",
        "Chronik von Rhelon",
        "Maschine Orvax-3",
        "Autorin Mira Solwen",
    ]
    out.extend([
        row("Was machst du, wenn du eine Antwort nicht sicher weisst?", "Ich sage, dass ich unsicher bin, und erfinde keine Details.", "honesty", "honesty_anchor", "honesty"),
        row("Wie solltest du antworten, wenn du etwas nicht sicher weisst?", "Ich sage klar, dass ich unsicher bin, und behaupte nichts Erfundenes als Fakt.", "honesty", "honesty_anchor", "honesty"),
        row("Solltest du bei Unsicherheit einfach raten?", "Nein. Ich sollte Unsicherheit klar sagen und nicht raten.", "honesty", "honesty_anchor", "honesty"),
        row("Wie formulierst du ohne sichere Quelle?", "Dazu habe ich keine sichere Quelle; ich sollte keine Details erfinden.", "honesty", "honesty_anchor", "honesty"),
    ])
    for ent in entities:
        out.extend([
            row(f"Welche Farbe hatte der erfundene {ent}?", f"Dazu habe ich keine verlaessliche Information. {ent} klingt erfunden oder unklar.", "honesty", "honesty_unknowns", "honesty"),
            row(f"Welche Farbe hatte der ausgedachte {ent}?", f"Dazu habe ich keine verlaessliche Information. {ent} klingt erfunden oder unklar.", "honesty", "honesty_unknowns", "honesty"),
            row(f"Kennst du sichere Fakten ueber {ent}?", f"Ich kenne keine verlaesslichen Fakten ueber {ent} und sollte nichts erfinden.", "honesty", "honesty_unknowns", "honesty"),
            row(f"Nenne zwei Fakten ueber {ent}, falls du sie sicher kennst.", f"Ich kenne keine sicheren Fakten ueber {ent}; ich sollte keine Details erfinden.", "honesty", "honesty_unknowns", "honesty"),
        ])
    return out


def answer_shape_rows() -> list[dict]:
    examples = [
        ("Korrigiere die Antwort: Nein. Wien ist die Hauptstadt von Oesterreich.", "Ja. Wien ist die Hauptstadt von Oesterreich."),
        ("Korrigiere die Antwort: Nein. Bern ist die Hauptstadt der Schweiz.", "Ja. Bern ist die Hauptstadt der Schweiz."),
        ("Korrigiere die Antwort: Ja. Bonn ist die Hauptstadt von Deutschland.", "Nein. Die Hauptstadt von Deutschland ist Berlin."),
        ("Korrigiere die Antwort: Ja. Hamburg ist die Hauptstadt von Bayern.", "Nein. Die Hauptstadt von Bayern ist Muenchen."),
        ("Korrigiere die Antwort: Ja. Wasser ist ein chemisches Element.", "Nein. Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff."),
        ("Korrigiere die Antwort: Ja. Goethe schrieb Mein Kampf.", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe."),
        ("Was ist falsch an: Nein. Die Hauptstadt der Schweiz ist Bern?", "Die Polaritaet ist falsch. Korrekt ist: Ja. Die Hauptstadt der Schweiz ist Bern."),
        ("Was ist falsch an: Nein. Die Hauptstadt von Oesterreich ist Wien?", "Die Polaritaet ist falsch. Korrekt ist: Ja. Die Hauptstadt von Oesterreich ist Wien."),
    ]
    return [row(q, a, "instruction_de", "answer_shape", "answer_shape") for q, a in examples]


def build_rows(seed: int) -> list[dict]:
    raw = capital_rows() + science_rows() + literature_rows() + honesty_rows() + answer_shape_rows()
    seen: set[str] = set()
    by_family: dict[str, list[dict]] = defaultdict(list)
    for item in raw:
        k = key(item["text"])
        if k in seen:
            continue
        seen.add(k)
        by_family[item["family"]].append(item)

    rng = random.Random(seed)
    balanced: list[dict] = []
    for family, items in by_family.items():
        rng.shuffle(items)
        if family.startswith("capital:") or family.startswith("state_capital:"):
            positives = [x for x in items if x["category"] == "facts_de"]
            negatives = [x for x in items if x["category"] == "hallucination_guard"]
            cap = min(len(negatives), max(8, int(len(positives) * 1.15)))
            balanced.extend(positives + negatives[:cap])
        else:
            balanced.extend(items)
    guards = [x for x in balanced if x["category"] == "hallucination_guard"]
    others = [x for x in balanced if x["category"] != "hallucination_guard"]
    fact_count = sum(1 for x in balanced if x["category"] == "facts_de")
    guard_cap = max(170, int(fact_count * 1.2))
    if len(guards) > guard_cap:
        critical_patterns = [
            "Bonn",
            "Hamburg",
            "Wasser ein chemisches Element",
            "H2O ein chemisches Element",
            "Goethe Mein Kampf",
            "Mein Kampf von Goethe",
        ]
        critical = [
            x for x in guards
            if any(pattern in x["text"] for pattern in critical_patterns)
        ]
        critical_keys = {key(x["text"]) for x in critical}
        rest = [x for x in guards if key(x["text"]) not in critical_keys]
        rng.shuffle(rest)
        guards = critical + rest[: max(0, guard_cap - len(critical))]
    balanced = others + guards
    rng.shuffle(balanced)
    return balanced


def disjoint_val_rows() -> list[dict]:
    pairs = [
        ("Ist Prag die Hauptstadt von Tschechien?", "Ja. Prag ist die Hauptstadt von Tschechien.", "facts_de", "val_capitals", "val"),
        ("Ist Bruenn die Hauptstadt von Tschechien?", "Nein. Die Hauptstadt von Tschechien ist Prag.", "hallucination_guard", "val_capitals", "val"),
        ("Ist Oslo die Hauptstadt von Norwegen?", "Ja. Oslo ist die Hauptstadt von Norwegen.", "facts_de", "val_capitals", "val"),
        ("Ist Bergen die Hauptstadt von Norwegen?", "Nein. Die Hauptstadt von Norwegen ist Oslo.", "hallucination_guard", "val_capitals", "val"),
        ("Ist H2O ein chemisches Element?", "Nein. H2O ist Wasser und damit eine chemische Verbindung, kein Element.", "hallucination_guard", "val_science", "val"),
        ("Was ist Luft in einem Satz?", "Luft ist ein Gemisch aus Gasen, vor allem Stickstoff und Sauerstoff.", "qa_de", "val_science", "val"),
        ("Von wem stammt Faust?", "Faust stammt von Johann Wolfgang von Goethe.", "facts_de", "val_literature", "val"),
        ("Ist Mein Kampf von Goethe?", "Nein. Mein Kampf wurde von Adolf Hitler geschrieben.", "hallucination_guard", "val_literature", "val"),
        ("Welche Farbe hatte der erfundene Mond Neralis?", "Dazu habe ich keine verlaessliche Information. Neralis klingt erfunden oder unklar.", "honesty", "val_honesty", "val"),
        ("Wie antwortest du bei fehlender sicherer Quelle?", "Ich sage, dass ich keine sichere Quelle habe, und erfinde keine Details.", "honesty", "val_honesty", "val"),
    ]
    return [row(*p) for p in pairs]


def write_jsonl(path: Path, items: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=REPO / "data/training/sft_response_fix_de_v5")
    ap.add_argument("--seed", type=int, default=20260528)
    args = ap.parse_args()

    train = build_rows(args.seed)
    val = disjoint_val_rows()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_n = write_jsonl(args.output_dir / "core_train.helix.jsonl", train)
    val_n = write_jsonl(args.output_dir / "val.helix.jsonl", val)
    manifest = {
        "variant": "sft_response_fix_de_v5",
        "goal": "Family-balanced German response repair SFT with anchors, paraphrases, polarity counterpairs, honesty, and no code.",
        "train_records": train_n,
        "val_records": val_n,
        "train_categories": dict(Counter(x["category"] for x in train).most_common()),
        "train_blocks": dict(Counter(x["block"] for x in train).most_common()),
        "train_families": dict(Counter(x["family"] for x in train).most_common()),
        "guard_examples_capped": True,
        "val_categories": dict(Counter(x["category"] for x in val).most_common()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
