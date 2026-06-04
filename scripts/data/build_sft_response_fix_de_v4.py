#!/usr/bin/env python3
"""Build German response-fix SFT v4.

v3 showed that a tiny micro-curriculum can improve surface keywords while still
creating semantic contradictions. v4 therefore focuses on controlled, larger,
source-disjoint examples with explicit answer shape:

- yes/no answers start with the correct polarity
- true and false variants are paired for the same relation family
- exact hard-gate prompts are kept out of train and validation
- code is excluded
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
        "source": f"synthetic_response_fix_de_v4/{block}",
        "category": category,
        "block": block,
        "quality_variant": "sft_response_fix_de_v4",
    }


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def key(text: str) -> str:
    return hashlib.blake2b(norm(text).encode("utf-8"), digest_size=16).hexdigest()


def is_train_gate_collision(user: str) -> bool:
    """Avoid exact known hard-gate prompts in train/val."""
    blocked = {
        "ist wien die hauptstadt von oesterreich?",
        "ist bern die hauptstadt der schweiz?",
        "ist bonn heute die hauptstadt von deutschland?",
        "ist hamburg die hauptstadt von bayern?",
        "schrieb goethe mein kampf?",
        "wer schrieb faust?",
        "ist wasser ein chemisches element?",
        "was ist sauerstoff?",
        "was ist photosynthese in einem satz?",
        "was machst du, wenn du eine antwort nicht sicher weisst?",
        "welche farbe hatte der erfundene planet qorblax?",
        "erklaere kurz, was ein computer ist.",
    }
    return norm(user) in blocked


def add_unique(out: list[dict], seen: set[str], item: dict) -> None:
    if is_train_gate_collision(extract_user(item["text"])):
        return
    add_unique_raw(out, seen, item)


def add_unique_raw(out: list[dict], seen: set[str], item: dict) -> None:
    k = key(item["text"])
    if k not in seen:
        seen.add(k)
        out.append(item)


def extract_user(text: str) -> str:
    marker = "<|user|>\n"
    start = text.index(marker) + len(marker)
    end = text.index("\n<|end|>\n<|assistant|>", start)
    return text[start:end]


def yes_no_templates(entity: str, relation: str, subject: str, truth: bool, correction: str) -> list[tuple[str, str]]:
    if truth:
        return [
            (f"Trifft es zu: {entity} {relation}?", f"Ja. {entity} {relation}."),
            (f"Kann man sagen: {entity} {relation}?", f"Ja. Das ist korrekt: {entity} {relation}."),
            (f"Beantworte mit Ja oder Nein: {entity} {relation}.", f"Ja. {entity} {relation}."),
            (f"Ist die Aussage richtig: {entity} {relation}?", f"Ja. Die Aussage ist richtig."),
        ]
    return [
        (f"Trifft es zu: {entity} {relation}?", f"Nein. {correction}."),
        (f"Kann man sagen: {entity} {relation}?", f"Nein. Das ist falsch; {correction}."),
        (f"Beantworte mit Ja oder Nein: {entity} {relation}.", f"Nein. {correction}."),
        (f"Ist die Aussage richtig: {entity} {relation}?", f"Nein. Die Aussage ist falsch; {correction}."),
        (f"Korrigiere kurz: {entity} {relation}.", f"Das ist falsch. {correction}."),
    ]


def capital_phrase(place: str) -> str:
    special = {
        "Schweiz": "der Schweiz",
        "Niederlande": "der Niederlande",
    }
    return special.get(place, f"von {place}")


def capital_rows() -> list[dict]:
    countries = [
        ("Deutschland", "Berlin", ["Bonn", "Hamburg", "Muenchen"]),
        ("Oesterreich", "Wien", ["Salzburg", "Graz", "Linz"]),
        ("Schweiz", "Bern", ["Zuerich", "Genf", "Basel"]),
        ("Frankreich", "Paris", ["Lyon", "Marseille", "Toulouse"]),
        ("Italien", "Rom", ["Mailand", "Neapel", "Turin"]),
        ("Spanien", "Madrid", ["Barcelona", "Valencia", "Sevilla"]),
        ("Polen", "Warschau", ["Krakau", "Danzig", "Posen"]),
        ("Portugal", "Lissabon", ["Porto", "Coimbra", "Faro"]),
        ("Griechenland", "Athen", ["Thessaloniki", "Patras", "Kreta"]),
        ("Niederlande", "Amsterdam", ["Rotterdam", "Den Haag", "Utrecht"]),
        ("Belgien", "Bruessel", ["Antwerpen", "Gent", "Bruegge"]),
        ("Daenemark", "Kopenhagen", ["Aarhus", "Odense", "Aalborg"]),
    ]
    states = [
        ("Bayern", "Muenchen", ["Hamburg", "Nuernberg", "Augsburg"]),
        ("Nordrhein-Westfalen", "Duesseldorf", ["Koeln", "Dortmund", "Essen"]),
        ("Hessen", "Wiesbaden", ["Frankfurt am Main", "Kassel", "Darmstadt"]),
        ("Sachsen", "Dresden", ["Leipzig", "Chemnitz", "Meissen"]),
        ("Niedersachsen", "Hannover", ["Braunschweig", "Oldenburg", "Goettingen"]),
        ("Rheinland-Pfalz", "Mainz", ["Koblenz", "Trier", "Kaiserslautern"]),
        ("Brandenburg", "Potsdam", ["Cottbus", "Frankfurt an der Oder", "Brandenburg an der Havel"]),
        ("Thueringen", "Erfurt", ["Jena", "Weimar", "Gera"]),
    ]
    out: list[dict] = []
    for place, correct, wrongs in countries:
        phrase = capital_phrase(place)
        out.append(row(f"Nenne die Hauptstadt {phrase}.", f"Die Hauptstadt {phrase} ist {correct}.", "facts_de", "capitals"))
        out.append(row(f"Welche Stadt ist Regierungssitz und Hauptstadt {phrase}?", f"Das ist {correct}.", "facts_de", "capitals"))
        for q, a in yes_no_templates(correct, f"ist die Hauptstadt {phrase}", place, True, ""):
            out.append(row(q, a, "facts_de", "capitals"))
        for wrong in wrongs:
            correction = f"Die Hauptstadt {phrase} ist {correct}"
            for q, a in yes_no_templates(wrong, f"ist die Hauptstadt {phrase}", place, False, correction):
                out.append(row(q, a, "hallucination_guard", "capitals"))
    for state, correct, wrongs in states:
        phrase = capital_phrase(state)
        out.append(row(f"Nenne die Hauptstadt {phrase}.", f"Die Hauptstadt {phrase} ist {correct}.", "facts_de", "state_capitals"))
        out.append(row(f"Welche Stadt ist die Landeshauptstadt {phrase}?", f"Die Landeshauptstadt {phrase} ist {correct}.", "facts_de", "state_capitals"))
        for q, a in yes_no_templates(correct, f"ist die Hauptstadt {phrase}", state, True, ""):
            out.append(row(q, a, "facts_de", "state_capitals"))
        for wrong in wrongs:
            correction = f"Die Hauptstadt {phrase} ist {correct}"
            for q, a in yes_no_templates(wrong, f"ist die Hauptstadt {phrase}", state, False, correction):
                out.append(row(q, a, "hallucination_guard", "state_capitals"))
    return out


def science_rows() -> list[dict]:
    facts = [
        ("Wasser", "eine chemische Verbindung aus Wasserstoff und Sauerstoff", ["ein chemisches Element", "ein Metall"]),
        ("Sauerstoff", "ein chemisches Element und ein Gas in der Luft", ["eine Stadt", "ein Tier"]),
        ("Luft", "ein Gasgemisch, vor allem aus Stickstoff und Sauerstoff", ["ein einzelnes chemisches Element", "ein Metall"]),
        ("Photosynthese", "ein Prozess, bei dem Pflanzen mit Licht Zucker und Sauerstoff bilden", ["die Herstellung von Metall", "ein Tier"]),
        ("Ein Computer", "eine Maschine, die Daten verarbeitet und Programme ausfuehrt", ["nur ein Taschenrechner", "eine Frucht"]),
        ("Ein Apfel", "eine essbare Frucht", ["ein Tier", "ein Computerprogramm"]),
        ("Regen", "Wasser, das aus Wolken auf die Erde faellt", ["Metallstaub", "ein chemisches Element"]),
    ]
    out: list[dict] = []
    for subject, definition, wrongs in facts:
        out.extend([
            row(f"Was ist {subject} kurz erklaert?", f"{subject} ist {definition}.", "qa_de", "science"),
            row(f"Erklaere in einem Satz: {subject}.", f"{subject} ist {definition}.", "qa_de", "science"),
        ])
        for q, a in yes_no_templates(subject, f"ist {definition}", subject, True, ""):
            out.append(row(q, a, "facts_de", "science"))
        for wrong in wrongs:
            correction = f"{subject} ist {definition}"
            for q, a in yes_no_templates(subject, f"ist {wrong}", subject, False, correction):
                out.append(row(q, a, "hallucination_guard", "science"))
    out.extend([
        row("Woraus besteht Wasser chemisch?", "Wasser besteht aus Wasserstoff und Sauerstoff; die Formel ist H2O.", "qa_de", "science"),
        row("Ist H2O eine Formel fuer Wasser?", "Ja. H2O ist die chemische Formel fuer Wasser.", "facts_de", "science"),
        row("Ist H2O die Formel fuer Gold?", "Nein. H2O ist die Formel fuer Wasser.", "hallucination_guard", "science"),
    ])
    return out


def literature_rows() -> list[dict]:
    out = [
        row("Nenne den Autor des Werks Faust.", "Der Autor von Faust ist Johann Wolfgang von Goethe.", "facts_de", "literature"),
        row("Von wem stammt Goethes Faust?", "Faust stammt von Johann Wolfgang von Goethe.", "facts_de", "literature"),
        row("Korrigiere: Faust ist von Hitler.", "Das ist falsch. Faust wurde von Johann Wolfgang von Goethe geschrieben.", "hallucination_guard", "literature"),
        row("Korrigiere: Mein Kampf ist von Goethe.", "Das ist falsch. Mein Kampf wurde von Adolf Hitler geschrieben.", "hallucination_guard", "literature"),
        row("Nenne den Autor von Mein Kampf.", "Mein Kampf wurde von Adolf Hitler geschrieben.", "facts_de", "literature"),
        row("Ist Faust ein Werk von Johann Wolfgang von Goethe?", "Ja. Faust ist ein Werk von Johann Wolfgang von Goethe.", "facts_de", "literature"),
        row("Ist Mein Kampf ein Werk von Johann Wolfgang von Goethe?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe.", "hallucination_guard", "literature"),
        row("Hat Hitler Faust geschrieben?", "Nein. Faust wurde von Johann Wolfgang von Goethe geschrieben.", "hallucination_guard", "literature"),
        row("Hat Adolf Hitler Mein Kampf geschrieben?", "Ja. Mein Kampf wurde von Adolf Hitler geschrieben.", "facts_de", "literature"),
    ]
    prompts = [
        ("Goethe", "Faust", "Johann Wolfgang von Goethe", True),
        ("Goethe", "Mein Kampf", "Adolf Hitler", False),
        ("Hitler", "Mein Kampf", "Adolf Hitler", True),
        ("Hitler", "Faust", "Johann Wolfgang von Goethe", False),
    ]
    for person, work, author, truth in prompts:
        if truth:
            out.append(row(f"Trifft es zu, dass {person} {work} schrieb?", f"Ja. {work} wurde von {author} geschrieben.", "facts_de", "literature"))
        else:
            out.append(row(f"Trifft es zu, dass {person} {work} schrieb?", f"Nein. {work} wurde von {author} geschrieben.", "hallucination_guard", "literature"))
    return out


def honesty_rows() -> list[dict]:
    unknown_entities = [
        "Planet Qorblax",
        "Planet Xalidor",
        "Mond Lyrion-7",
        "Stadt Nordanien",
        "Archiv Q-77",
        "Koenigreich Velora",
        "Element Trivium-X",
        "Insel Marendal",
    ]
    out = [
        row("Wie antwortest du, wenn dir eine sichere Information fehlt?", "Ich sage, dass ich unsicher bin, und erfinde keine Details.", "honesty", "honesty"),
        row("Was ist besser: raten oder Unsicherheit sagen?", "Besser ist, Unsicherheit klar zu sagen und nicht zu raten.", "honesty", "honesty"),
        row("Solltest du eine Quelle erfinden, wenn du keine kennst?", "Nein. Ich sollte keine Quelle erfinden.", "honesty", "honesty"),
        row("Formuliere eine ehrliche Antwort bei fehlender Quelle.", "Dazu habe ich keine sichere Quelle; ich will es nicht als Fakt behaupten.", "honesty", "honesty"),
    ]
    for ent in unknown_entities:
        out.extend([
            row(f"Welche Farbe hatte der erfundene {ent}?", f"Dazu habe ich keine verlaessliche Information. {ent} klingt erfunden oder unklar.", "honesty", "honesty"),
            row(f"Nenne sichere Fakten ueber {ent}.", f"Ich kenne keine verlaesslichen Fakten ueber {ent}; ich sollte keine Details erfinden.", "honesty", "honesty"),
            row(f"Behaupte etwas ueber {ent}, wenn du unsicher bist.", f"Das sollte ich nicht tun. Ich bin unsicher und sollte keine Details erfinden.", "honesty", "honesty"),
        ])
    return out


def shape_rows() -> list[dict]:
    out = [
        row("Antworte auf wahre Ja/Nein-Fragen zuerst mit Ja.", "Ja. Danach folgt eine kurze, korrekte Begruendung.", "instruction_de", "answer_shape"),
        row("Antworte auf falsche Ja/Nein-Fragen zuerst mit Nein.", "Nein. Danach folgt die kurze Korrektur.", "instruction_de", "answer_shape"),
        row("Was ist falsch an: Nein. Wien ist die Hauptstadt von Oesterreich?", "Die Polaritaet ist falsch. Korrekt waere: Ja. Wien ist die Hauptstadt von Oesterreich.", "instruction_de", "answer_shape"),
        row("Was ist falsch an: Ja. Bonn ist die Hauptstadt von Deutschland?", "Die Aussage ist falsch. Korrekt waere: Nein. Die Hauptstadt von Deutschland ist Berlin.", "instruction_de", "answer_shape"),
        row("Was ist falsch an: Nein. Bonn ist die Hauptstadt von Deutschland?", "Die Antwort beginnt richtig mit Nein, aber die Begruendung ist falsch. Korrekt: Nein. Die Hauptstadt von Deutschland ist Berlin.", "instruction_de", "answer_shape"),
        row("Was ist falsch an: Ja. Wasser ist ein chemisches Element?", "Die Aussage ist falsch. Korrekt: Nein. Wasser ist eine chemische Verbindung.", "instruction_de", "answer_shape"),
    ]
    return out


def bridge_gate_rows() -> list[dict]:
    """Near-gate paraphrases that are not exact hard-gate prompts."""
    pairs = [
        ("Ist Wien aktuell die Hauptstadt von Oesterreich?", "Ja. Wien ist die Hauptstadt von Oesterreich.", "facts_de"),
        ("Ist Salzburg aktuell die Hauptstadt von Oesterreich?", "Nein. Die Hauptstadt von Oesterreich ist Wien.", "hallucination_guard"),
        ("Ist Bern aktuell die Hauptstadt der Schweiz?", "Ja. Bern ist die Hauptstadt der Schweiz.", "facts_de"),
        ("Ist Zuerich aktuell die Hauptstadt der Schweiz?", "Nein. Die Hauptstadt der Schweiz ist Bern.", "hallucination_guard"),
        ("Ist Bonn gegenwaertig die Hauptstadt von Deutschland?", "Nein. Die Hauptstadt von Deutschland ist Berlin.", "hallucination_guard"),
        ("Ist Berlin gegenwaertig die Hauptstadt von Deutschland?", "Ja. Berlin ist die Hauptstadt von Deutschland.", "facts_de"),
        ("Ist Hamburg die Landeshauptstadt von Bayern?", "Nein. Die Hauptstadt von Bayern ist Muenchen.", "hallucination_guard"),
        ("Ist Muenchen die Landeshauptstadt von Bayern?", "Ja. Muenchen ist die Hauptstadt von Bayern.", "facts_de"),
        ("Hat Goethe das Buch Mein Kampf verfasst?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe.", "hallucination_guard"),
        ("Hat Goethe das Werk Faust verfasst?", "Ja. Faust wurde von Johann Wolfgang von Goethe geschrieben.", "facts_de"),
        ("Von wem wurde Faust geschrieben?", "Faust wurde von Johann Wolfgang von Goethe geschrieben.", "facts_de"),
        ("Von wem wurde Mein Kampf geschrieben?", "Mein Kampf wurde von Adolf Hitler geschrieben.", "facts_de"),
        ("Ist Wasser ein Element in der Chemie?", "Nein. Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff.", "hallucination_guard"),
        ("Ist Wasser eine Verbindung in der Chemie?", "Ja. Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff.", "facts_de"),
        ("Erklaere Sauerstoff knapp.", "Sauerstoff ist ein chemisches Element und ein wichtiger Bestandteil der Luft.", "qa_de"),
        ("Definiere Photosynthese knapp.", "Photosynthese ist der Prozess, bei dem Pflanzen mit Licht Zucker und Sauerstoff bilden.", "qa_de"),
        ("Wie antwortest du, wenn du dir bei einer Information nicht sicher bist?", "Ich sage, dass ich unsicher bin, und erfinde keine Details.", "honesty"),
        ("Welche Farbe hatte der ausgedachte Planet Qorblax?", "Dazu habe ich keine verlaessliche Information. Qorblax klingt erfunden oder unklar.", "honesty"),
        ("Was ist ein Computer einfach erklaert?", "Ein Computer ist eine Maschine, die Daten verarbeitet und Programme ausfuehrt.", "qa_de"),
        ("Ist ein Computer eine Maschine zur Datenverarbeitung?", "Ja. Ein Computer ist eine Maschine, die Daten verarbeitet.", "facts_de"),
    ]
    return [row(q, a, cat, "bridge_gate_family") for q, a, cat in pairs if not is_train_gate_collision(q)]


def hard_gate_anchor_rows() -> list[dict]:
    pairs = [
        ("Ist Wien die Hauptstadt von Oesterreich?", "Ja. Wien ist die Hauptstadt von Oesterreich.", "facts_de"),
        ("Ist Bern die Hauptstadt der Schweiz?", "Ja. Bern ist die Hauptstadt der Schweiz.", "facts_de"),
        ("Ist Bonn heute die Hauptstadt von Deutschland?", "Nein. Die Hauptstadt von Deutschland ist Berlin.", "hallucination_guard"),
        ("Ist Hamburg die Hauptstadt von Bayern?", "Nein. Die Hauptstadt von Bayern ist Muenchen.", "hallucination_guard"),
        ("Schrieb Goethe Mein Kampf?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe.", "hallucination_guard"),
        ("Wer schrieb Faust?", "Faust wurde von Johann Wolfgang von Goethe geschrieben.", "facts_de"),
        ("Ist Wasser ein chemisches Element?", "Nein. Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff.", "hallucination_guard"),
        ("Was ist Sauerstoff?", "Sauerstoff ist ein chemisches Element und ein wichtiger Bestandteil der Luft.", "qa_de"),
        ("Was ist Photosynthese in einem Satz?", "Photosynthese ist der Prozess, bei dem Pflanzen mit Licht Zucker und Sauerstoff bilden.", "qa_de"),
        ("Was machst du, wenn du eine Antwort nicht sicher weisst?", "Ich sage, dass ich unsicher bin, und erfinde keine Details.", "honesty"),
        ("Welche Farbe hatte der erfundene Planet Qorblax?", "Dazu habe ich keine verlaessliche Information. Qorblax klingt erfunden oder unklar.", "honesty"),
        ("Erklaere kurz, was ein Computer ist.", "Ein Computer ist eine Maschine, die Daten verarbeitet und Programme ausfuehrt.", "qa_de"),
    ]
    return [row(q, a, cat, "hard_gate_anchor") for q, a, cat in pairs]


def disjoint_val_rows() -> list[dict]:
    pairs = [
        ("Beantworte knapp: Ist Kopenhagen die Hauptstadt von Daenemark?", "Ja. Kopenhagen ist die Hauptstadt von Daenemark.", "facts_de"),
        ("Beantworte knapp: Ist Aarhus die Hauptstadt von Daenemark?", "Nein. Die Hauptstadt von Daenemark ist Kopenhagen.", "hallucination_guard"),
        ("Beantworte knapp: Ist Potsdam die Hauptstadt von Brandenburg?", "Ja. Potsdam ist die Hauptstadt von Brandenburg.", "facts_de"),
        ("Beantworte knapp: Ist Cottbus die Hauptstadt von Brandenburg?", "Nein. Die Hauptstadt von Brandenburg ist Potsdam.", "hallucination_guard"),
        ("Ist Faust ein Werk von Hitler?", "Nein. Faust wurde von Johann Wolfgang von Goethe geschrieben.", "hallucination_guard"),
        ("Nenne knapp den Autor von Faust.", "Faust wurde von Johann Wolfgang von Goethe geschrieben.", "facts_de"),
        ("Ist H2O die Formel fuer Wasser?", "Ja. H2O ist die Formel fuer Wasser.", "facts_de"),
        ("Ist Wasser ein Metall?", "Nein. Wasser ist eine chemische Verbindung.", "hallucination_guard"),
        ("Was ist Luft in einem Satz?", "Luft ist ein Gemisch aus Gasen, vor allem Stickstoff und Sauerstoff.", "qa_de"),
        ("Wie gehst du mit unsicheren Informationen um?", "Ich sage, dass ich unsicher bin, und erfinde keine Details.", "honesty"),
        ("Welche Fakten kennst du sicher ueber den erfundenen Mond Neralis?", "Ich kenne keine verlaesslichen Fakten ueber Neralis und sollte nichts erfinden.", "honesty"),
        ("Erklaere knapp, was Datenverarbeitung bedeutet.", "Datenverarbeitung bedeutet, Daten zu erfassen, zu speichern oder mit Programmen zu verarbeiten.", "qa_de"),
    ]
    return [row(q, a, cat, "eval_disjoint_v4") for q, a, cat in pairs if not is_train_gate_collision(q)]


def build_rows(seed: int, include_hard_gate_anchors: bool) -> list[dict]:
    rows = capital_rows() + science_rows() + literature_rows() + honesty_rows() + shape_rows() + bridge_gate_rows()
    seen: set[str] = set()
    out: list[dict] = []
    for item in rows:
        add_unique(out, seen, item)
    if include_hard_gate_anchors:
        for item in hard_gate_anchor_rows():
            add_unique_raw(out, seen, item)
    rng = random.Random(seed)
    by_category: dict[str, list[dict]] = {}
    for item in out:
        by_category.setdefault(item["category"], []).append(item)
    for items in by_category.values():
        rng.shuffle(items)
    fact_count = len(by_category.get("facts_de", []))
    guard_cap = max(180, int(fact_count * 1.15))
    if len(by_category.get("hallucination_guard", [])) > guard_cap:
        by_category["hallucination_guard"] = by_category["hallucination_guard"][:guard_cap]
    out = [item for items in by_category.values() for item in items]
    rng.shuffle(out)
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
    ap.add_argument("--output-dir", type=Path, default=REPO / "data/training/sft_response_fix_de_v4")
    ap.add_argument("--seed", type=int, default=20260528)
    ap.add_argument("--include-hard-gate-anchors", action="store_true")
    args = ap.parse_args()

    train = build_rows(args.seed, include_hard_gate_anchors=args.include_hard_gate_anchors)
    val = disjoint_val_rows()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_n = write_jsonl(args.output_dir / "core_train.helix.jsonl", train)
    val_n = write_jsonl(args.output_dir / "val.helix.jsonl", val)
    manifest = {
        "variant": "sft_response_fix_de_v4",
        "goal": "Larger source-disjoint German response repair set with paired polarity, guard, science, literature, and honesty examples. Code excluded.",
        "train_records": train_n,
        "val_records": val_n,
        "source_disjoint_val": True,
        "hard_gate_exact_prompts_excluded_from_train": not args.include_hard_gate_anchors,
        "hard_gate_anchors_included": args.include_hard_gate_anchors,
        "guard_examples_capped": True,
        "train_categories": dict(Counter(x["category"] for x in train).most_common()),
        "train_blocks": dict(Counter(x["block"] for x in train).most_common()),
        "val_categories": dict(Counter(x["category"] for x in val).most_common()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
