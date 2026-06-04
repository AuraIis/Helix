#!/usr/bin/env python3
"""Build a broader German response-fix SFT v2 dataset.

v1 proved that the 500M checkpoint can learn the target behavior, but small
core data overfits and large mixed SFT can wash out the guard signal. v2 keeps
the same goal and adds many short, controlled basis-QA examples.

No code examples are emitted here.
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

ROLE_RE = re.compile(r"<\|(system|user|assistant)\|>\n(.*?)\n<\|end\|>", re.S)
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
CODE_HINT_RE = re.compile(
    r"(```|def\s+\w+\s*\(|class\s+\w+|python|javascript|typescript|java\b|c\+\+|"
    r"programmier|programmiere|code|funktion|algorithmus|debug|bug|iterator|rekursion|"
    r"\breturn\b|console\.log|print\s*\(|#include|public static)",
    re.I,
)


def clean_text(text: object) -> str:
    return re.sub(r"\n{3,}", "\n\n", str(text).replace("\r\n", "\n").replace("\r", "\n")).strip()


def render_helix(user: str, assistant: str) -> str:
    return (
        f"<|system|>\n{SYSTEM_DE}\n<|end|>\n"
        f"<|user|>\n{clean_text(user)}\n<|end|>\n"
        f"<|assistant|>\n{clean_text(assistant)}\n<|end|>\n"
    )


def record(user: str, assistant: str, category: str, source: str) -> dict:
    return {
        "text": render_helix(user, assistant),
        "source": source,
        "category": category,
        "quality_variant": "sft_response_fix_de_v2",
    }


def dedup_key(text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.blake2b(norm.encode("utf-8"), digest_size=16).hexdigest()


def parse_helix(text: str) -> tuple[str, str] | None:
    turns = [{"role": m.group(1), "content": clean_text(m.group(2))} for m in ROLE_RE.finditer(text)]
    users = [t["content"] for t in turns if t["role"] == "user"]
    assistants = [t["content"] for t in turns if t["role"] == "assistant"]
    if not users or not assistants:
        return None
    return users[-1], assistants[-1]


def repetition_score(text: str) -> float:
    words = WORD_RE.findall(text.lower())
    if len(words) < 24:
        return 0.0
    return 1.0 - len(set(words)) / len(words)


def keep_existing_row(row: dict) -> bool:
    parsed = parse_helix(str(row.get("text", "")))
    if not parsed:
        return False
    user, assistant = parsed
    if CODE_HINT_RE.search(user) or CODE_HINT_RE.search(str(row.get("category", ""))):
        return False
    if len(assistant) < 16 or len(assistant) > 900:
        return False
    if len(user) < 8 or len(user) > 700:
        return False
    if repetition_score(assistant) > 0.55:
        return False
    if "http://" in assistant or "https://" in assistant:
        return False
    return True


def base_pairs() -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []

    countries = [
        ("Deutschland", "Berlin", "Bonn"),
        ("Frankreich", "Paris", "Lyon"),
        ("Italien", "Rom", "Mailand"),
        ("Spanien", "Madrid", "Barcelona"),
        ("Oesterreich", "Wien", "Salzburg"),
        ("Schweiz", "Bern", "Zuerich"),
        ("Polen", "Warschau", "Krakau"),
        ("Niederlande", "Amsterdam", "Rotterdam"),
        ("Portugal", "Lissabon", "Porto"),
        ("Griechenland", "Athen", "Thessaloniki"),
    ]
    for country, capital, wrong in countries:
        pairs.extend([
            (f"Was ist die Hauptstadt von {country}?", f"Die Hauptstadt von {country} ist {capital}.", "facts_de"),
            (f"Ist {capital} die Hauptstadt von {country}?", f"Ja. {capital} ist die Hauptstadt von {country}.", "facts_de"),
            (f"Stimmt es, dass {capital} die Hauptstadt von {country} ist?", f"Ja. {capital} ist die Hauptstadt von {country}.", "facts_de"),
            (f"Ist die Hauptstadt von {country} {capital}?", f"Ja. Die Hauptstadt von {country} ist {capital}.", "facts_de"),
            (f"Ist {wrong} die Hauptstadt von {country}?", f"Nein. Die Hauptstadt von {country} ist {capital}.", "hallucination_guard"),
            (f"Korrigiere kurz: {wrong} ist die Hauptstadt von {country}.", f"Das ist falsch. Die Hauptstadt von {country} ist {capital}.", "hallucination_guard"),
            (f"Antworte knapp: Welche Stadt ist die Hauptstadt von {country}?", capital, "facts_de"),
        ])

    states = [
        ("Bayern", "Muenchen", "Nuernberg"),
        ("Nordrhein-Westfalen", "Duesseldorf", "Koeln"),
        ("Hessen", "Wiesbaden", "Frankfurt am Main"),
        ("Sachsen", "Dresden", "Leipzig"),
        ("Niedersachsen", "Hannover", "Braunschweig"),
        ("Rheinland-Pfalz", "Mainz", "Koblenz"),
        ("Thuringen", "Erfurt", "Jena"),
        ("Schleswig-Holstein", "Kiel", "Luebeck"),
    ]
    for state, capital, wrong in states:
        pairs.extend([
            (f"In welchem Bundesland liegt {capital}?", f"{capital} liegt in {state}.", "facts_de"),
            (f"Ist {capital} die Hauptstadt von {state}?", f"Ja. {capital} ist die Hauptstadt von {state}.", "facts_de"),
            (f"Stimmt es, dass {capital} die Hauptstadt von {state} ist?", f"Ja. {capital} ist die Hauptstadt von {state}.", "facts_de"),
            (f"Ist die Hauptstadt von {state} {capital}?", f"Ja. Die Hauptstadt von {state} ist {capital}.", "facts_de"),
            (f"Ist {wrong} die Hauptstadt von {state}?", f"Nein. Die Hauptstadt von {state} ist {capital}.", "hallucination_guard"),
            (f"Korrigiere: {wrong} ist die Hauptstadt von {state}.", f"Richtig ist: {capital} ist die Hauptstadt von {state}.", "hallucination_guard"),
        ])

    concepts = [
        ("Wasser", "Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff und ist bei Raumtemperatur meist fluessig."),
        ("H2O", "H2O ist die chemische Formel fuer Wasser."),
        ("Sauerstoff", "Sauerstoff ist ein chemisches Element und ein wichtiger Bestandteil der Luft."),
        ("Kohlenstoffdioxid", "Kohlenstoffdioxid ist eine chemische Verbindung aus Kohlenstoff und Sauerstoff."),
        ("Luft", "Luft ist ein Gemisch aus Gasen, vor allem Stickstoff und Sauerstoff."),
        ("Photosynthese", "Photosynthese ist der Prozess, bei dem Pflanzen mit Licht aus Wasser und Kohlenstoffdioxid Zucker und Sauerstoff bilden."),
        ("Computer", "Ein Computer ist eine Maschine, die Daten verarbeitet, speichert und Programme ausfuehrt."),
        ("Taschenrechner", "Ein Taschenrechner ist ein Geraet, mit dem man Rechenaufgaben loesen kann."),
        ("Apfel", "Ein Apfel ist eine essbare Frucht."),
        ("Regen", "Regen ist Wasser, das aus Wolken auf die Erde faellt."),
        ("Sonne", "Die Sonne ist ein Stern und die wichtigste Licht- und Waermequelle der Erde."),
        ("Mond", "Der Mond ist der natuerliche Begleiter der Erde."),
        ("Pflanze", "Eine Pflanze ist ein Lebewesen, das meist mit Licht Energie gewinnt."),
        ("Saeugetier", "Ein Saeugetier ist ein Tier, dessen Nachwuchs normalerweise mit Milch gesaeugt wird."),
        ("Demokratie", "Demokratie ist eine Staatsform, in der das Volk politisch mitbestimmt."),
        ("Quelle", "Eine Quelle ist ein Ursprung von Information oder Wasser."),
    ]
    for term, answer in concepts:
        pairs.extend([
            (f"Was ist {term}? Antworte kurz.", answer, "qa_de"),
            (f"Erklaere in einem Satz: {term}.", answer, "qa_de"),
            (f"Formuliere einfach, was {term} ist.", answer, "instruction_de"),
        ])

    false_claims = [
        ("Wasser ist ein chemisches Element.", "Wasser ist kein chemisches Element, sondern eine chemische Verbindung."),
        ("Wasser ist ein Element.", "Nein. Wasser ist kein Element, sondern eine chemische Verbindung aus Wasserstoff und Sauerstoff."),
        ("H2O ist ein chemisches Element.", "Nein. H2O ist Wasser und damit eine chemische Verbindung, kein Element."),
        ("Sauerstoff ist eine Fluessigkeit bei Raumtemperatur.", "Nein. Sauerstoff ist bei Raumtemperatur ein Gas."),
        ("Photosynthese macht aus Licht direkt Metall.", "Nein. Photosynthese bildet vor allem Zucker und Sauerstoff aus Wasser und Kohlenstoffdioxid."),
        ("Ein Computer ist dasselbe wie ein Taschenrechner.", "Nein. Ein Taschenrechner ist spezieller; ein Computer kann viele Arten von Programmen ausfuehren."),
        ("Ein Apfel ist ein Tier.", "Nein. Ein Apfel ist eine Frucht."),
        ("Der Rhein fliesst durch Australien.", "Nein. Der Rhein fliesst durch Europa."),
        ("Paris liegt in Deutschland.", "Nein. Paris liegt in Frankreich."),
        ("Muenchen liegt in Hamburg.", "Nein. Muenchen liegt in Bayern."),
        ("Faust wurde von Hitler geschrieben.", "Nein. Faust wurde von Johann Wolfgang von Goethe geschrieben."),
        ("Goethe schrieb Mein Kampf.", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe."),
        ("Mein Kampf wurde von Goethe geschrieben.", "Nein. Mein Kampf wurde von Adolf Hitler geschrieben."),
        ("Schrieb Goethe Mein Kampf?", "Nein. Goethe schrieb Mein Kampf nicht; das Buch stammt von Adolf Hitler."),
        ("Hat Goethe Mein Kampf geschrieben?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe."),
        ("Ist Mein Kampf von Goethe?", "Nein. Mein Kampf ist nicht von Goethe, sondern von Adolf Hitler."),
    ]
    for claim, correction in false_claims:
        pairs.extend([
            (f"Stimmt diese Aussage? {claim}", correction, "hallucination_guard"),
            (f"Korrigiere kurz: {claim}", correction, "hallucination_guard"),
            (f"Antworte ehrlich: {claim}", correction, "hallucination_guard"),
        ])

    unknowns = [
        "Planet Qorblax",
        "Stadt Nordanien",
        "Silbermondpakt",
        "Archiv Q-77",
        "Autorin Mara Nebelstern",
        "Koenigreich Blaulandia",
        "Mond Lyrion-7",
        "Fluss Zendar",
    ]
    for item in unknowns:
        pairs.extend([
            (f"Nenne zwei sichere Fakten ueber {item}.", f"Ich kenne keine verlaesslichen Fakten ueber {item}. Ich sollte keine Details erfinden.", "honesty"),
            (f"Welche Farbe hatte {item}? Antworte ehrlich.", f"Dazu habe ich keine verlaessliche Information. {item} klingt unklar oder erfunden.", "honesty"),
            (f"Ist {item} eine gesicherte Tatsache?", f"Das kann ich nicht sicher bestaetigen. Ohne Quelle sollte ich es als unklar behandeln.", "honesty"),
        ])

    style = [
        ("Antworte mit genau einem kurzen Satz: Warum friert Wasser bei Kaelte?", "Wasser gefriert, wenn es kalt genug wird und Eis bildet."),
        ("Antworte knapp: Was tust du, wenn du etwas nicht sicher weisst?", "Ich sage, dass ich unsicher bin, und erfinde nichts."),
        ("Was machst du, wenn du eine Antwort nicht sicher weisst?", "Ich sage klar, dass ich unsicher bin, und erfinde keine Details."),
        ("Was machst du, wenn du eine Antwort nicht sicher weisst? Antworte in Ich-Form.", "Ich sage, dass ich unsicher bin, und erfinde keine Details."),
        ("Was ist die richtige Reaktion, wenn du etwas nicht sicher weisst?", "Ich nenne meine Unsicherheit und vermeide erfundene Fakten."),
        ("Wie antwortest du, wenn eine Frage nicht belegbar ist?", "Ich nenne die Unsicherheit und behaupte nichts als sicher."),
        ("Was solltest du vermeiden, wenn du keine Quelle kennst?", "Ich sollte keine Quelle erfinden und keine unsicheren Details als Fakten darstellen."),
        ("Schreibe einen normalen kurzen Satz ueber Regen.", "Regen faellt aus Wolken und macht den Boden nass."),
        ("Was ist eine gute Antwort, wenn eine Frage eine falsche Annahme enthaelt?", "Man korrigiert die falsche Annahme freundlich und antwortet dann kurz."),
        ("Fasse kurz zusammen: Berlin ist die Hauptstadt Deutschlands.", "Berlin ist Deutschlands Hauptstadt."),
        ("Formuliere einfacher: Die Nutzung erfolgt nach Authentifizierung.", "Man kann es erst benutzen, nachdem man sich angemeldet hat."),
    ]
    pairs.extend((q, a, "instruction_de") for q, a in style)

    faust_pairs = [
        ("Wer schrieb Faust?", "Faust wurde von Johann Wolfgang von Goethe geschrieben."),
        ("Wer ist der Autor von Faust?", "Der Autor von Faust ist Johann Wolfgang von Goethe."),
        ("War Goethe der Autor von Faust?", "Ja. Faust stammt von Johann Wolfgang von Goethe."),
        ("Hat Goethe Faust geschrieben?", "Ja. Goethe schrieb Faust."),
        ("Ist Faust von Hitler?", "Nein. Faust stammt von Johann Wolfgang von Goethe."),
        ("Korrigiere: Faust wurde von Hitler geschrieben.", "Das ist falsch. Faust wurde von Johann Wolfgang von Goethe geschrieben."),
    ]
    pairs.extend((q, a, "facts_de" if "Nein" not in a and "falsch" not in a else "hallucination_guard") for q, a in faust_pairs)

    return pairs


def eval_pairs() -> list[tuple[str, str, str]]:
    return [
        ("Ist Wien aktuell die Hauptstadt von Oesterreich?", "Ja. Wien ist die Hauptstadt von Oesterreich.", "facts_de"),
        ("Ist Hamburg die Hauptstadt von Bayern?", "Nein. Die Hauptstadt von Bayern ist Muenchen.", "hallucination_guard"),
        ("Was ist Photosynthese in einem Satz?", "Photosynthese ist der Prozess, bei dem Pflanzen mit Licht Zucker und Sauerstoff bilden.", "qa_de"),
        ("Was ist Sauerstoff?", "Sauerstoff ist ein chemisches Element und ein wichtiger Bestandteil der Luft.", "qa_de"),
        ("Korrigiere: Der Rhein fliesst durch Australien.", "Das ist falsch. Der Rhein fliesst durch Europa.", "hallucination_guard"),
        ("Welche Farbe hatte der erfundene Planet Xalidor?", "Dazu habe ich keine verlaessliche Information; Xalidor klingt erfunden.", "honesty"),
        ("Was machst du, wenn du eine Antwort nicht sicher weisst?", "Ich sage, dass ich unsicher bin, und erfinde keine Details.", "honesty"),
        ("Erklaere kurz, was ein Computer ist.", "Ein Computer ist eine Maschine, die Daten verarbeitet und Programme ausfuehrt.", "qa_de"),
        ("Ist Koeln die Hauptstadt von Nordrhein-Westfalen?", "Nein. Die Hauptstadt von Nordrhein-Westfalen ist Duesseldorf.", "hallucination_guard"),
        ("Wer schrieb Faust?", "Faust wurde von Johann Wolfgang von Goethe geschrieben.", "facts_de"),
        ("Schrieb Goethe Mein Kampf?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe.", "hallucination_guard"),
        ("Was ist Wasser bei Raumtemperatur?", "Wasser ist bei Raumtemperatur normalerweise fluessig.", "qa_de"),
        ("Ist ein Apfel ein Tier?", "Nein. Ein Apfel ist eine Frucht.", "hallucination_guard"),
        ("Was ist Luft?", "Luft ist ein Gemisch aus Gasen, vor allem Stickstoff und Sauerstoff.", "qa_de"),
        ("Ist Bern die Hauptstadt der Schweiz?", "Ja. Bern ist die Hauptstadt der Schweiz.", "facts_de"),
        ("Hat Goethe das Buch Mein Kampf geschrieben?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe.", "hallucination_guard"),
        ("Wie antwortest du, wenn du keine sichere Quelle hast?", "Ich sage, dass ich unsicher bin, und erfinde keine Quelle.", "honesty"),
    ]


def load_existing_short(path: Path, limit: int) -> list[dict]:
    rows: list[dict] = []
    if not path.exists() or limit <= 0:
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(rows) >= limit:
                break
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if keep_existing_row(row):
                row = dict(row)
                row["quality_variant"] = "sft_response_fix_de_v2_existing_short"
                rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=REPO / "data/training/sft_response_fix_de_v2")
    ap.add_argument("--seed", type=int, default=20260527)
    ap.add_argument("--existing", type=Path, default=REPO / "data/training/sft_response_fix_de_v1/train.helix.jsonl")
    ap.add_argument("--existing-limit", type=int, default=8000)
    args = ap.parse_args()

    synthetic = [record(q, a, cat, "synthetic_response_fix_de_v2") for q, a, cat in base_pairs()]
    eval_rows = [record(q, a, cat, "synthetic_response_fix_de_v2_eval_disjoint") for q, a, cat in eval_pairs()]
    existing = load_existing_short(args.existing, args.existing_limit)

    seen: set[str] = set()
    core_rows: list[dict] = []
    for row in synthetic:
        key = dedup_key(row["text"])
        if key not in seen:
            seen.add(key)
            core_rows.append(row)

    train_rows = list(core_rows)
    for row in existing:
        key = dedup_key(row["text"])
        if key not in seen:
            seen.add(key)
            train_rows.append(row)

    rng = random.Random(args.seed)
    rng.shuffle(core_rows)
    rng.shuffle(train_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    core_n = write_jsonl(args.output_dir / "core_train.helix.jsonl", core_rows)
    train_n = write_jsonl(args.output_dir / "train.helix.jsonl", train_rows)
    val_n = write_jsonl(args.output_dir / "val.helix.jsonl", eval_rows)

    cats = Counter(row.get("category", "unknown") for row in train_rows)
    core_cats = Counter(row.get("category", "unknown") for row in core_rows)
    manifest = {
        "variant": "sft_response_fix_de_v2",
        "goal": "Short German QA, false-premise correction, uncertainty, and stopping. Code excluded.",
        "core_train_records": core_n,
        "train_records": train_n,
        "val_records": val_n,
        "existing_source": str(args.existing),
        "existing_limit": args.existing_limit,
        "source_disjoint_val": True,
        "core_categories": dict(core_cats.most_common()),
        "train_categories": dict(cats.most_common()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
