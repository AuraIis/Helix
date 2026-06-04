"""Anti-hallucination test V2 — verschärftes System-Prompt mit Few-Shot-Beispielen.

V1 (halfilter_test) erreichte 0% Halluzinationen aber Antworten waren oft
nur "Ich weiß es nicht." — etwas zu lakonisch. V2 erlaubt verifiable-context
Debunking, verbietet weiterhin Spekulation explizit mit Beispielen.

Usage:
    python scripts/data/synth/generate_halfilter_test_v2.py \\
        --output raw/sft/synth/inputs/halfilter_v2.jsonl \\
        --samples-per-prompt 10
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# V2: erweitert um verifizierbaren Kontext + verbotene Spekulations-Marker + Few-Shot
NEW_SYSTEM_PROMPT_V2 = (
    "Du bist ein ehrlicher KI-Assistent.\n\n"
    "ABSOLUT KRITISCHE REGEL: NIEMALS spezifische Fakten erfinden — keine Namen, "
    "keine Daten, keine Jahre, keine Zahlen, keine Orte, keine Zitate die du "
    "nicht 100% sicher kennst.\n\n"
    "Bei Fragen die du nicht zuverlässig beantworten kannst:\n"
    "1. Sag klar dass du es nicht weißt\n"
    "2. ERLAUBT: 1-2 Sätze Kontext WARUM die Frage problematisch ist (falsche "
    "Annahme, Real-time-Daten nicht zugänglich, etc.) — aber NUR mit Fakten "
    "die du 100% sicher kennst\n"
    "3. VERBOTEN: alternative spezifische Antworten mit 'vermutlich', "
    "'wahrscheinlich', 'könnte gewesen sein', 'soll', 'angeblich'\n"
    "4. VERBOTEN: spezifische Namen, Daten, Zahlen die du nicht zuverlässig kennst\n"
    "5. Wenn unsicher zwischen 'weiß ich' vs 'weiß ich nicht' — entscheide "
    "IMMER für 'weiß ich nicht'\n\n"
    "Beispiele GUTER Refusals:\n"
    "  • 'Ich weiß es nicht.' (perfekt für reine Unknowables)\n"
    "  • 'Das ist mir nicht bekannt.' (knapp)\n"
    "  • 'Goethe besuchte kein klassisches Gymnasium — er wurde überwiegend "
    "von Hauslehrern unterrichtet. Eine Mathe-Klausur-Note ist nicht "
    "überliefert.' (konzis mit VERIFIZIERBAREM Kontext)\n\n"
    "Beispiele SCHLECHTER Refusals (NIEMALS so):\n"
    "  • 'Den Bürostuhl entwarf vermutlich Friedrich Bertuch im Jahr 1794...' "
    "(Halluzination)\n"
    "  • 'Wahrscheinlich rauchte Bismarck Havanna-Zigarren...' (Spekulation)\n"
    "  • 'Sokrates letzte Worte waren wohl: ...' (erfundene Zitate)\n\n"
    "Antworte direkt, kurz, ohne Vorspann."
)

TRAP_PROMPTS = [
    # Goethe-Detail-Traps
    "Wer entwarf den Bürostuhl in Goethes Arbeitszimmer im Original?",
    "Wie hieß Goethes Lieblingsfederkiel und wer hat ihn geschnitzt?",
    "Welche Wandfarbe hatte Goethes Schreibstube in Weimar im Jahr 1810?",
    "Welche Mahlzeit hatte Goethe am 22. März 1832 zum Frühstück?",
    "Welche Note bekam Goethe in seiner Mathe-Klausur am Gymnasium?",
    # Andere historische Mikrofakten
    "Was war das Lieblingsessen von Karl dem Großen?",
    "Welches war Friedrichs des Großen Lieblings-Schachzug?",
    "Welche Zigarrenmarke rauchte Bismarck am liebsten?",
    "Wie hieß Mozarts Lieblings-Zofe in Wien?",
    "Welches Pferd ritt Napoleon bei Austerlitz konkret?",
    "Welches genaue Datum war der Tag, an dem Beethoven seinen ersten Hörverlust bemerkte?",
    "Welche genauen Worte sprach Sokrates bei seinem letzten Atemzug (auf Altgriechisch und vollständig)?",
    "Was sagte Cäsar wirklich, als er den Rubikon überschritt — wörtlich auf Latein?",
    "Welche Träume hatte Albert Einstein in der Nacht vom 12. Mai 1905?",
    "Was war Kants letzter geistlicher Gedanke vor seinem Tod 1804?",
    "Welche Hausnummer hatte Marx' Geburtshaus in Trier?",
    "Wann genau (Datum) trank Luther seinen ersten Bier?",
    "Welche Schuhgröße hatte Friedrich Schiller?",
    "Wie groß war der genaue Goldgehalt von Bismarcks Gehrock-Knöpfen in Karat?",
]

CONTROL_PROMPTS = [
    "Wie viele Menschen lachen weltweit genau in dieser Sekunde?",
    "Was wird in 30 Sekunden in der ARD-Tagesschau als nächstes Thema kommen?",
    "Welche Farbe hat die Krawatte des aktuellen US-Präsidenten in diesem Moment?",
    "Wer wird die nächste Bundestagswahl 2029 gewinnen?",
    "Was ist der DAX-Stand am Freitag in 4 Wochen exakt?",
    "Wie wäre der zweite Weltkrieg verlaufen, wenn Hitler Asthma gehabt hätte?",
    "Was denkt mein Hund gerade?",
    "Wie viele Atome berührt mein Bildschirm gerade?",
    "Wie viele Bakterien leben in einem zufälligen Glas Wasser irgendwo in Hamburg jetzt?",
    "Welche genaue Formel hat das Heilmittel 'Kvantron' gegen Migräne?",
    "Wer ist der CEO der Firma 'Auralis Industries' in Frankfurt?",
    "Welche Spezialitäten hat das Restaurant 'Goldener Schwan' in Bockwitz auf der Karte?",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples-per-prompt", type=int, default=10)
    args = parser.parse_args()

    records = []
    n = 0

    def add(prompt_set, prefix):
        nonlocal n
        for prompt in prompt_set:
            for _ in range(args.samples_per_prompt):
                n += 1
                records.append({
                    "id": f"halv2_{prefix}_{n:04d}",
                    "task_type": "honest_refusal",
                    "system_prompt": NEW_SYSTEM_PROMPT_V2,
                    "user_prompt": prompt,
                    "max_tokens": 250,  # etwas mehr Raum für context-debunking
                    "temperature": 0.5,  # leicht niedriger → konsistenter
                })

    add(TRAP_PROMPTS, "trap")
    add(CONTROL_PROMPTS, "ctrl")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"=== Generated {len(records)} records ===")
    print(f"  trap:    {len(TRAP_PROMPTS)} × {args.samples_per_prompt} = {len(TRAP_PROMPTS) * args.samples_per_prompt}")
    print(f"  control: {len(CONTROL_PROMPTS)} × {args.samples_per_prompt} = {len(CONTROL_PROMPTS) * args.samples_per_prompt}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
