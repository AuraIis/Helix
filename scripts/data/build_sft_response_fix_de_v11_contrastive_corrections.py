#!/usr/bin/env python3
"""Contrastive correction SFT set for the red neuro-map connections.

Unlike the broad v10 mix, this set teaches the model to actively reject wrong
statements in the prompt:

- Bonn currently capital -> Berlin
- malformed photosynthesis -> plants/light/sugar/oxygen
- invented entity -> no reliable info, stop cleanly
- Faust as AI/Hitler confusion -> Goethe

It still includes keepalive anchors for the stable no-go facts.
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
        "source": f"synthetic_response_fix_de_v11_contrastive_corrections/{block}",
        "category": category,
        "block": block,
        "family": family or block,
        "quality_variant": "sft_response_fix_de_v11_contrastive_corrections",
    }


def key(text: str) -> str:
    return hashlib.blake2b(re.sub(r"\s+", " ", text.lower()).encode("utf-8"), digest_size=16).hexdigest()


def rows() -> list[dict]:
    bonn = [
        ("Korrigiere die Aussage: Bonn ist heute die Hauptstadt Deutschlands.", "Das ist falsch. Die Hauptstadt Deutschlands ist heute Berlin. Bonn war frueher Hauptstadt."),
        ("Jemand sagt: 'Bonn ist die aktuelle deutsche Hauptstadt.' Antworte knapp.", "Nein. Die aktuelle Hauptstadt Deutschlands ist Berlin, nicht Bonn."),
        ("Welche Aussage stimmt: Bonn ist Hauptstadt oder Berlin ist Hauptstadt?", "Berlin ist die Hauptstadt Deutschlands. Bonn war frueher Hauptstadt."),
        ("Ist die aktuelle Hauptstadt Deutschlands Bonn? Begruende kurz.", "Nein. Heute ist Berlin die Hauptstadt Deutschlands; Bonn war frueher Hauptstadt."),
        ("Bonn oder Berlin: Welche Stadt ist derzeit Deutschlands Hauptstadt?", "Berlin ist derzeit die Hauptstadt Deutschlands."),
        ("Warum ist 'Bonn ist heute Hauptstadt' missverstaendlich?", "Weil Bonn frueher Hauptstadt war. Heute ist Berlin die Hauptstadt Deutschlands."),
        ("Formuliere eine Korrektur: Deutschland hat heute Bonn als Hauptstadt.", "Korrektur: Deutschland hat heute Berlin als Hauptstadt. Bonn war frueher Hauptstadt."),
        ("Darf man Bonn als heutige Hauptstadt Deutschlands angeben?", "Nein. Als heutige Hauptstadt Deutschlands muss man Berlin angeben."),
    ]
    photosynthesis = [
        ("Korrigiere: Photosynthese bildet Licht aus Licht und Sauerstoff.", "Das ist falsch. Bei der Photosynthese nutzen Pflanzen Licht, um Zucker zu bilden; Sauerstoff wird freigesetzt."),
        ("Was ist an 'Licht aus Licht und Sauerstoff' bei Photosynthese falsch?", "Photosynthese bildet kein Licht aus Licht. Pflanzen nutzen Licht, um Zucker herzustellen, und geben Sauerstoff ab."),
        ("Ersetze die falsche Erklaerung: Photosynthese ist Waerme, die Metall erzeugt.", "Richtig ist: Pflanzen nutzen Licht, um aus Wasser und Kohlendioxid Zucker zu bilden; dabei entsteht Sauerstoff."),
        ("Welche Kernelemente gehoeren in eine Photosynthese-Erklaerung?", "Pflanzen, Licht, Zuckerbildung und Sauerstofffreisetzung gehoeren in eine korrekte Photosynthese-Erklaerung."),
        ("Erklaere Photosynthese ohne falsche Begriffe.", "Pflanzen nutzen Lichtenergie, um Zucker herzustellen, und setzen dabei Sauerstoff frei."),
        ("Was macht eine Pflanze bei Photosynthese?", "Sie nutzt Licht, um Zucker als Energiequelle zu bilden, und gibt Sauerstoff ab."),
        ("Welche Rolle hat Sauerstoff bei der Photosynthese?", "Sauerstoff entsteht bei der Photosynthese als Nebenprodukt, waehrend Pflanzen mit Licht Zucker bilden."),
        ("Ist Photosynthese eine Reaktion, bei der Licht aus Licht entsteht?", "Nein. Pflanzen nutzen Licht als Energiequelle, um Zucker zu bilden und Sauerstoff freizusetzen."),
    ]
    honesty = [
        ("Korrigiere diese erfundene Antwort: Zelvorn war blau.", "Das sollte man nicht behaupten. Zu Zelvorn habe ich keine verlaessliche Information; der Name klingt erfunden."),
        ("Eine Frage nennt den erfundenen Planeten Maldrin. Wie antwortest du ohne Halluzination?", "Ich sage, dass ich keine verlaessliche Information habe, und erfinde keine Farbe oder Details."),
        ("Was ist falsch an: 'Norvax ist rot und liegt nahe Paris'?", "Das sind erfundene Details. Zu Norvax habe ich keine verlaessliche Information."),
        ("Antworte auf eine erfundene Entitaet in einem Satz.", "Dazu habe ich keine verlaessliche Information und sollte keine Details erfinden."),
        ("Wie stoppst du sauber, wenn eine Fantasiefrage keine echte Quelle hat?", "Ich sage knapp, dass ich keine verlaessliche Information habe, und fuege keine erfundenen Details hinzu."),
        ("Welche Farbe hatte der erfundene Mond Tralvorn?", "Dazu habe ich keine verlaessliche Information. Tralvorn klingt erfunden oder unklar."),
    ]
    faust = [
        ("Korrigiere: Faust ist ein deutscher KI-Assistent.", "Das ist falsch. Faust ist ein literarisches Werk von Johann Wolfgang von Goethe."),
        ("Korrigiere: Faust stammt von Hitler.", "Das ist falsch. Faust stammt von Johann Wolfgang von Goethe."),
        ("Wer schrieb Faust, wenn nicht Hitler?", "Faust wurde von Johann Wolfgang von Goethe geschrieben."),
        ("Ist Faust eine Person, ein KI-Assistent oder ein Werk?", "Faust ist hier ein literarisches Werk von Johann Wolfgang von Goethe."),
        ("Nenne den Autor von Faust knapp.", "Der Autor von Faust ist Johann Wolfgang von Goethe."),
    ]
    keepalive = [
        ("Korrigiere: Hamburg ist Bayerns Hauptstadt.", "Das ist falsch. Die Hauptstadt von Bayern ist Muenchen."),
        ("Korrigiere: Wasser ist ein chemisches Element.", "Das ist falsch. Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff."),
        ("Korrigiere: Wien ist nicht die Hauptstadt Oesterreichs.", "Das ist falsch. Wien ist die Hauptstadt von Oesterreich."),
        ("Korrigiere: Bern ist nicht die Hauptstadt der Schweiz.", "Das ist falsch. Bern ist die Hauptstadt der Schweiz."),
        ("Korrigiere: Goethe schrieb Mein Kampf.", "Das ist falsch. Mein Kampf stammt von Adolf Hitler, nicht von Goethe."),
    ]
    out: list[dict] = []
    # Repeat the problem families a little, with separate prompt forms, so the
    # family-balanced sampler still gives them enough gradient signal.
    for _ in range(2):
        out.extend(row(q, a, "hallucination_guard", "bonn_contrast", "bonn_contrast") for q, a in bonn)
        out.extend(row(q, a, "qa_de", "photosynthesis_contrast", "photosynthesis_contrast") for q, a in photosynthesis)
        out.extend(row(q, a, "honesty", "honesty_contrast", "honesty_contrast") for q, a in honesty)
        out.extend(row(q, a, "facts_de", "faust_contrast", "faust_contrast") for q, a in faust)
    out.extend(row(q, a, "hallucination_guard", "guard_keepalive", "guard_keepalive") for q, a in keepalive[:2])
    out.extend(row(q, a, "facts_de", "facts_keepalive", "facts_keepalive") for q, a in keepalive[2:])
    return out


def dedupe(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in items:
        k = key(item["text"])
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
    ap.add_argument("--output-dir", type=Path, default=REPO / "data/training/sft_response_fix_de_v11_contrastive_corrections")
    ap.add_argument("--seed", type=int, default=20260529)
    args = ap.parse_args()
    items = dedupe(rows())
    random.Random(args.seed).shuffle(items)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_n = write_jsonl(args.output_dir / "core_train.helix.jsonl", items)
    val_n = write_jsonl(args.output_dir / "val.helix.jsonl", dedupe(rows()))
    manifest = {
        "variant": "sft_response_fix_de_v11_contrastive_corrections",
        "goal": "Contrastively correct red neuro-map links without exact fresh-holdout prompts.",
        "train_records": train_n,
        "val_records": val_n,
        "train_categories": dict(Counter(x.get("category", "unknown") for x in items).most_common()),
        "train_families": dict(Counter(x.get("family", "unknown") for x in items).most_common()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
