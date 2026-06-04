#!/usr/bin/env python3
"""Generate a small deterministic German pretraining booster corpus.

This is not a replacement for web/Wikipedia/code corpora. It is a clean
capability booster with plain documents, not chat/SFT turns:

- arithmetic with worked calculations
- stable German/world facts
- Python/code mini-lessons
- short reasoning and correction snippets
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]

FACTS = [
    ("Berlin", "die Hauptstadt Deutschlands", "Berlin ist ein Stadtstaat und zugleich ein Bundesland."),
    ("Paris", "die Hauptstadt Frankreichs", "Paris liegt an der Seine."),
    ("Wien", "die Hauptstadt Österreichs", "Wien liegt an der Donau."),
    ("Bern", "die Bundesstadt der Schweiz", "Die Schweiz hat keine Hauptstadt im engen verfassungsrechtlichen Sinn."),
    ("Goethe", "der Autor von Faust", "Johann Wolfgang von Goethe war ein deutscher Dichter."),
    ("Schiller", "der Autor von Die Räuber", "Friedrich Schiller veröffentlichte das Drama 1781."),
    ("Wasser", "eine chemische Verbindung aus Wasserstoff und Sauerstoff", "Die Formel von Wasser ist H2O."),
    ("Kohlendioxid", "eine chemische Verbindung aus Kohlenstoff und Sauerstoff", "Die Formel von Kohlendioxid ist CO2."),
    ("das Grundgesetz", "die Verfassung der Bundesrepublik Deutschland", "Es trat am 23. Mai 1949 in Kraft."),
    ("die Berliner Mauer", "ein Symbol der deutschen Teilung", "Sie fiel am 9. November 1989."),
]

PY_CONCEPTS = [
    ("Liste", "eine geordnete veränderbare Sammlung", "werte = [1, 2, 3]"),
    ("Dictionary", "eine Zuordnung von Schlüsseln zu Werten", "alter = {'Ada': 36}"),
    ("Funktion", "ein wiederverwendbarer Block von Anweisungen", "def addiere(a, b):\n    return a + b"),
    ("Schleife", "eine Wiederholung von Anweisungen", "for zahl in range(3):\n    print(zahl)"),
    ("Bedingung", "eine Verzweigung nach Wahrheit eines Ausdrucks", "if temperatur > 30:\n    print('warm')"),
]


@dataclass
class Manifest:
    output_file: str
    documents: int = 0
    categories: Counter = field(default_factory=Counter)


def arithmetic_docs(rng: random.Random, n: int) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    ops = ["+", "-", "*"]
    for _ in range(n):
        a = rng.randint(2, 99_999)
        b = rng.randint(2, 9_999)
        op = rng.choice(ops)
        if op == "+":
            result = a + b
            text = f"Rechenbeispiel: {a} + {b} = {result}. Man addiert zuerst {a} und danach {b}."
        elif op == "-":
            if b > a:
                a, b = b, a
            result = a - b
            text = f"Rechenbeispiel: {a} - {b} = {result}. Subtraktion bedeutet, dass {b} von {a} abgezogen wird."
        else:
            result = a * b
            text = f"Rechenbeispiel: {a} * {b} = {result}. Multiplikation ist wiederholte Addition."
        docs.append(("math", text))
    return docs


def fraction_docs(rng: random.Random, n: int) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for _ in range(n):
        denom = rng.choice([2, 3, 4, 5, 6, 8, 10])
        base = denom * rng.randint(2, 5_000)
        num = rng.randint(1, denom - 1)
        part = base // denom * num
        text = (
            f"Bruchbeispiel: {num}/{denom} von {base} ist {part}. "
            f"Ein {denom}tel von {base} ist {base // denom}; davon nimmt man {num} Teile."
        )
        docs.append(("math_fraction", text))
    return docs


def doc_key(text: str) -> str:
    normalized = " ".join(text.strip().lower().split())
    return hashlib.blake2b(normalized.encode("utf-8"), digest_size=16).hexdigest()


def add_unique(
    target: list[tuple[str, str]],
    seen: set[str],
    rows: list[tuple[str, str]],
    *,
    limit: int,
) -> None:
    for category, text in rows:
        if len(target) >= limit:
            return
        one_line = " ".join(text.strip().replace("\r", "").splitlines())
        key = doc_key(one_line)
        if key in seen:
            continue
        seen.add(key)
        target.append((category, one_line))


def fact_docs(rng: random.Random, n: int) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for _ in range(n):
        subject, fact, extra = rng.choice(FACTS)
        style = rng.randint(0, 2)
        if style == 0:
            text = f"{subject}: {subject} ist {fact}. {extra}"
        elif style == 1:
            text = f"Faktennotiz: {subject} wird korrekt mit folgender Aussage verbunden: {fact}. {extra}"
        else:
            text = f"Korrekturwissen: Wenn eine Aussage {subject} falsch einordnet, ist wichtig: {subject} ist {fact}. {extra}"
        docs.append(("facts", text))
    return docs


def code_docs(rng: random.Random, n: int) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for _ in range(n):
        name, desc, code = rng.choice(PY_CONCEPTS)
        text = (
            f"Python-Notiz: Eine {name} ist {desc}. Beispiel:\n"
            f"```python\n{code}\n```\n"
            "Der Code sollte klein, lesbar und eindeutig sein."
        )
        docs.append(("code", text))
    return docs


def reasoning_docs(rng: random.Random, n: int) -> list[tuple[str, str]]:
    docs: list[tuple[str, str]] = []
    for _ in range(n):
        subject, fact, extra = rng.choice(FACTS)
        wrong = rng.choice(["Frankfurt", "Hitler", "1949", "Wasserfarbe", "England"])
        text = (
            f"Prüfnotiz: Eine Behauptung wie '{subject} bedeutet {wrong}' muss geprüft werden. "
            f"Die stabile Zuordnung lautet: {subject} ist {fact}. {extra}"
        )
        docs.append(("correction", text))
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=REPO / "data" / "training" / "pretrain_booster_de_v1.txt")
    parser.add_argument("--documents", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260509)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    docs: list[tuple[str, str]] = []
    seen: set[str] = set()
    add_unique(docs, seen, arithmetic_docs(rng, args.documents * 30 // 100), limit=args.documents)
    add_unique(docs, seen, fraction_docs(rng, args.documents * 15 // 100), limit=args.documents)
    add_unique(docs, seen, fact_docs(rng, args.documents * 25 // 100), limit=args.documents)
    add_unique(docs, seen, code_docs(rng, args.documents * 15 // 100), limit=args.documents)
    add_unique(docs, seen, reasoning_docs(rng, args.documents * 15 // 100), limit=args.documents)

    # Static fact/code templates are intentionally capped by exact de-dup. Fill
    # the remaining budget with arithmetic variants, which have a large
    # combinatorial space and do not teach repeated canned prose.
    while len(docs) < args.documents:
        need = args.documents - len(docs)
        add_unique(docs, seen, arithmetic_docs(rng, max(need, 10_000)), limit=args.documents)
    rng.shuffle(docs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest = Manifest(output_file=str(args.output))
    with args.output.open("w", encoding="utf-8", newline="\n") as fh:
        for category, text in docs:
            fh.write(text + "\n")
            manifest.documents += 1
            manifest.categories[category] += 1

    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_payload = {
        "output_file": manifest.output_file,
        "documents": manifest.documents,
        "categories": dict(manifest.categories.most_common()),
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output} ({manifest.documents:,} docs)")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
