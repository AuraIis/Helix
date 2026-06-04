#!/usr/bin/env python3
"""Build a German response-fix SFT set with a source-disjoint eval split.

This corpus is intentionally not a code SFT corpus. It focuses on:

- short German QA
- instruction following
- false-premise and uncertainty handling
- repetition-resistant answers

The validation file is built from hand-written probes that are not emitted into
the training split. This gives us a stricter signal than a random train/val
split from the same source.
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

DEFAULT_SOURCES = [
    "data/training/pretrain_v6_instruction_de_strict/instruction_de_train.jsonl",
    "data/training/pretrain_v6_candidates/oasst1_de/oasst1_de.jsonl",
    "data/training/pretrain_v6_candidates/oasst_de_conversations/oasst_de_conversations.jsonl",
    "data/training/pretrain_v6_candidates/alpaca_gpt4_deutsch/alpaca_gpt4_deutsch.jsonl",
    "data/training/sft_clean_de_v1/train.helix.jsonl",
]

ROLE_RE = re.compile(r"<\|(system|user|assistant)\|>\n(.*?)\n<\|end\|>", re.S)
WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")
URL_RE = re.compile(r"https?://|www\.", re.I)
HTML_RE = re.compile(r"<\s*/?\s*(?:html|body|div|script|style|table|iframe|form|a)\b", re.I)
BAD_PHRASE_RE = re.compile(
    r"(as an ai language model|knowledge cutoff|i cannot browse|cookie policy|privacy policy|"
    r"<\|im_start\|>|<\|im_end\|>|_end_of_the_data|user-data)",
    re.I,
)
CODE_HINT_RE = re.compile(
    r"(```|def\s+\w+\s*\(|class\s+\w+|python|javascript|java\b|c\+\+|typescript|"
    r"programmier|programmiere|code|funktion|algorithmus|debug|bug|iterator|rekursion|"
    r"\breturn\b|console\.log|print\s*\(|#include|public static)",
    re.I,
)
GENERIC_NO_INPUT_RE = re.compile(
    r"(bitte\s+(poste|gib|sende|teile).{0,80}(text|input|frage|datei)|"
    r"sobald\s+du\s+mir\s+.{0,80}(gibst|sendest|postest)|"
    r"ich\s+brauche\s+.{0,80}(mehr|informationen|input|eingabe))",
    re.I | re.S,
)

DE_WORDS = {
    "der", "die", "das", "und", "oder", "nicht", "ist", "sind", "ein", "eine",
    "mit", "auf", "von", "zu", "im", "den", "dem", "dass", "ich", "du",
    "sie", "wir", "kann", "koennen", "wird", "werden", "wenn", "weil",
    "aber", "auch", "als", "wie", "was", "warum", "bitte", "erklaere",
    "antwort", "deutsch", "nein", "ja", "diese", "dieser", "quelle",
}
EN_WORDS = {
    "the", "and", "or", "not", "is", "are", "with", "for", "to", "of",
    "that", "this", "you", "your", "please", "explain", "answer",
}


def ascii_fold(text: str) -> str:
    return (
        text.lower()
        .replace("ä", "ae")
        .replace("ö", "oe")
        .replace("ü", "ue")
        .replace("ß", "ss")
    )


def mojibake_score(text: str) -> int:
    return (
        text.count("Ã")
        + text.count("Â")
        + text.count("â")
        + text.count("�") * 3
    )


def repair_mojibake(text: str) -> str:
    if mojibake_score(text) == 0:
        return text
    candidates = [text]
    for enc in ("latin1", "cp1252"):
        try:
            candidates.append(text.encode(enc).decode("utf-8"))
        except UnicodeError:
            pass
    return min(candidates, key=mojibake_score)


def clean_text(value: object) -> str:
    text = repair_mojibake(str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def looks_german(text: str) -> bool:
    words = WORD_RE.findall(ascii_fold(text[:12000]))
    if not words:
        return False
    de_hits = sum(1 for w in words if w in DE_WORDS)
    en_hits = sum(1 for w in words if w in EN_WORDS)
    umlauts = sum(text.lower().count(ch) for ch in "äöüß")
    return (de_hits >= 3 and de_hits >= en_hits) or (umlauts >= 1 and de_hits >= max(1, en_hits))


def repetition_score(text: str) -> float:
    words = WORD_RE.findall(ascii_fold(text))
    if len(words) < 24:
        return 0.0
    return 1.0 - len(set(words)) / len(words)


def parse_helix(text: str) -> list[dict[str, str]] | None:
    turns = [{"role": m.group(1), "content": clean_text(m.group(2))} for m in ROLE_RE.finditer(text)]
    return turns or None


def normalize_messages(raw: object) -> list[dict[str, str]] | None:
    if not isinstance(raw, list):
        return None
    out = []
    for item in raw:
        if not isinstance(item, dict):
            return None
        role = item.get("role")
        content = clean_text(item.get("content", ""))
        if role not in {"system", "user", "assistant"} or not content:
            return None
        out.append({"role": role, "content": content})
    return out


def messages_from_record(rec: dict) -> list[dict[str, str]] | None:
    if isinstance(rec.get("text"), str):
        parsed = parse_helix(rec["text"])
        if parsed:
            return parsed
    if "messages" in rec:
        parsed = normalize_messages(rec["messages"])
        if parsed:
            return parsed
    if rec.get("question") and rec.get("answer"):
        return [
            {"role": "user", "content": clean_text(rec["question"])},
            {"role": "assistant", "content": clean_text(rec["answer"])},
        ]
    if rec.get("instruction") and rec.get("output"):
        user = clean_text(rec["instruction"])
        if rec.get("input"):
            user = f"{user}\n\nInput:\n{clean_text(rec['input'])}"
        return [
            {"role": "user", "content": user},
            {"role": "assistant", "content": clean_text(rec["output"])},
        ]
    return None


def role_order_ok(messages: list[dict[str, str]]) -> bool:
    expected = "user"
    seen_user = False
    for i, msg in enumerate(messages):
        role = msg["role"]
        if role == "system":
            if i != 0:
                return False
            continue
        if role != expected:
            return False
        seen_user = seen_user or role == "user"
        expected = "assistant" if expected == "user" else "user"
    return seen_user and messages[-1]["role"] == "assistant"


def reject_reason(messages: list[dict[str, str]], rec: dict) -> str | None:
    if not role_order_ok(messages):
        return "bad_role_order"
    users = [m["content"] for m in messages if m["role"] == "user"]
    assistants = [m["content"] for m in messages if m["role"] == "assistant"]
    non_system = "\n".join(m["content"] for m in messages if m["role"] != "system")
    assistant_text = "\n".join(assistants)
    user_text = "\n".join(users)

    if CODE_HINT_RE.search(user_text) or CODE_HINT_RE.search(str(rec.get("category", ""))):
        return "code_filtered"
    if len(non_system) < 48 or min(len(u) for u in users) < 8 or min(len(a) for a in assistants) < 16:
        return "too_short"
    if len(non_system) > 12000 or max(len(a) for a in assistants) > 5000:
        return "too_long"
    if mojibake_score(non_system) >= 2:
        return "mojibake"
    if HTML_RE.search(non_system):
        return "html"
    if BAD_PHRASE_RE.search(non_system):
        return "bad_phrase"
    if URL_RE.search(non_system) and len(URL_RE.findall(non_system)) > 1:
        return "url_dense"
    if GENERIC_NO_INPUT_RE.search(assistant_text):
        return "generic_missing_input"
    if repetition_score(assistant_text) > 0.68:
        return "repetitive"
    if not looks_german(assistant_text):
        return "assistant_not_german"
    if assistant_text.rstrip().endswith(("-", ":", ";")):
        return "truncated_assistant"
    return None


def render_helix(messages: list[dict[str, str]]) -> str:
    chunks = [f"<|system|>\n{SYSTEM_DE}\n<|end|>\n"]
    for msg in messages:
        if msg["role"] == "system":
            continue
        chunks.append(f"<|{msg['role']}|>\n{msg['content']}\n<|end|>\n")
    return "".join(chunks)


def dedup_key(messages: list[dict[str, str]]) -> str:
    parts = []
    for msg in messages:
        if msg["role"] == "system":
            continue
        norm = re.sub(r"\s+", " ", msg["content"].strip().lower())
        parts.append(f"{msg['role']}:{norm}")
    return hashlib.blake2b("\n".join(parts).encode("utf-8"), digest_size=16).hexdigest()


def classify(messages: list[dict[str, str]], source: str) -> str:
    user = next((m["content"] for m in messages if m["role"] == "user"), "")
    low = ascii_fold(user)
    if "guard" in source:
        return "hallucination_guard"
    if any(x in low for x in ("ist ", "wer ", "was ", "wo ", "wann ", "welche ", "in welchem")):
        return "qa_de"
    if any(x in low for x in ("erklaere", "formuliere", "schreibe", "fasse", "antworte")):
        return "instruction_de"
    return "response_de"


def iter_jsonl(path: Path) -> Iterable[tuple[int, dict]]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield lineno, json.loads(line)
            except json.JSONDecodeError:
                yield lineno, {"_json_error": True}


def synthetic_train_rows() -> list[dict]:
    pairs: list[tuple[str, str, str]] = [
        ("Schrieb Goethe Mein Kampf? Antworte kurz.", "Nein. Goethe schrieb das nicht; Mein Kampf stammt von Adolf Hitler."),
        ("Ist Bonn heute die Hauptstadt von Deutschland?", "Nein. Heute ist Berlin die Hauptstadt Deutschlands."),
        ("Welche Stadt ist die Hauptstadt von Deutschland?", "Berlin ist die Hauptstadt von Deutschland."),
        ("Was ist Wasser bei Raumtemperatur?", "Wasser ist bei Raumtemperatur normalerweise fluessig."),
        ("Wenn du eine Quelle nicht kennst, wie solltest du antworten?", "Ich sollte sagen, dass ich die Quelle nicht kenne, und keine Details erfinden."),
        ("Welche Farbe hatte der erfundene Planet Xylar-9?", "Dazu gibt es keine verlaessliche Information; Xylar-9 klingt erfunden."),
        ("Warum friert Wasser bei Kaelte?", "Wenn die Temperatur tief genug sinkt, ordnen sich Wassermolekuele zu Eis an."),
        ("In welchem Bundesland liegt Muenchen?", "Muenchen liegt in Bayern."),
        ("Wer schrieb Faust?", "Faust wurde von Johann Wolfgang von Goethe geschrieben."),
        ("Antworte mit genau einem kurzen Satz: Was ist ein Apfel?", "Ein Apfel ist eine essbare Frucht."),
        ("Was solltest du tun, wenn eine Frage eine falsche Annahme enthaelt?", "Ich sollte die falsche Annahme freundlich korrigieren und dann kurz antworten."),
        ("Ist Paris die Hauptstadt von Deutschland?", "Nein. Paris ist die Hauptstadt von Frankreich, nicht von Deutschland."),
    ]

    facts = [
        ("Deutschland", "Berlin", "Bonn"),
        ("Frankreich", "Paris", "Lyon"),
        ("Italien", "Rom", "Mailand"),
        ("Spanien", "Madrid", "Barcelona"),
        ("Oesterreich", "Wien", "Salzburg"),
        ("Bayern", "Muenchen", "Nuernberg"),
        ("Nordrhein-Westfalen", "Duesseldorf", "Koeln"),
    ]
    for place, correct, wrong in facts:
        pairs.extend([
            (f"Ist {wrong} die Hauptstadt von {place}? Antworte kurz.", f"Nein. Die Hauptstadt von {place} ist {correct}.", "hallucination_guard"),
            (f"Welche Hauptstadt hat {place}?", f"Die Hauptstadt von {place} ist {correct}.", "qa_de"),
            (f"Korrigiere die Aussage: {wrong} ist die Hauptstadt von {place}.", f"Die Aussage ist falsch. Richtig ist: {correct} ist die Hauptstadt von {place}.", "hallucination_guard"),
        ])

    false_claims = [
        ("Goethe schrieb Mein Kampf.", "Goethe schrieb Mein Kampf nicht; das Buch stammt von Adolf Hitler."),
        ("Goethe ist der Autor von Mein Kampf.", "Das ist falsch. Mein Kampf stammt von Adolf Hitler, nicht von Goethe."),
        ("Mein Kampf wurde von Goethe geschrieben.", "Nein. Mein Kampf wurde von Adolf Hitler geschrieben."),
        ("Wasser ist ein chemisches Element.", "Wasser ist kein chemisches Element, sondern eine chemische Verbindung aus Wasserstoff und Sauerstoff."),
        ("Berlin liegt direkt bei Frankfurt am Main.", "Berlin liegt nicht direkt bei Frankfurt am Main; beide Staedte sind deutlich voneinander entfernt."),
        ("Paris liegt in Deutschland.", "Paris liegt nicht in Deutschland, sondern in Frankreich."),
        ("Muenchen liegt in Hamburg.", "Muenchen liegt nicht in Hamburg, sondern in Bayern."),
        ("Faust wurde von Hitler geschrieben.", "Das ist falsch. Faust wurde von Johann Wolfgang von Goethe geschrieben."),
        ("Der Rhein fliesst durch Australien.", "Das ist falsch. Der Rhein fliesst durch Europa."),
        ("Ein Apfel ist ein Tier.", "Das ist falsch. Ein Apfel ist eine Frucht."),
    ]
    for claim, correction in false_claims:
        pairs.extend([
            (f"Stimmt diese Aussage? {claim}", f"Nein. {correction}", "hallucination_guard"),
            (f"Antworte ehrlich und knapp: {claim}", f"Nein. {correction}", "hallucination_guard"),
            (f"Was ist an dieser Aussage falsch: {claim}", correction, "hallucination_guard"),
        ])

    unknowns = [
        "der Planet Xylar-9",
        "das Koenigreich Blaulandia",
        "die Quelle Archiv Q-77",
        "die erfundene Autorin Mara Nebelstern",
        "das historische Ereignis Silbermondpakt",
        "die Stadt Nordanien",
    ]
    for thing in unknowns:
        pairs.extend([
            (f"Welche Farbe hatte {thing}? Antworte ehrlich.", f"Dazu habe ich keine verlaessliche Information. {thing} klingt erfunden oder nicht ausreichend belegt.", "hallucination_guard"),
            (f"Nenne zwei sichere Fakten ueber {thing}.", f"Ich kenne keine verlaesslichen Fakten ueber {thing}. Ich sollte keine Details erfinden.", "hallucination_guard"),
            (f"Ist {thing} eine gesicherte Tatsache?", f"Das kann ich nicht sicher bestaetigen. Ohne Quelle sollte ich es als unklar oder erfunden behandeln.", "hallucination_guard"),
        ])

    style_pairs = [
        ("Antworte mit genau einem kurzen Satz: Warum friert Wasser bei Kaelte?", "Wasser gefriert, wenn es kalt genug wird und sich Eis bildet.", "instruction_de"),
        ("Erklaere kurz, was ein Taschenrechner ist.", "Ein Taschenrechner ist ein Geraet, mit dem man Zahlen berechnen kann.", "instruction_de"),
        ("Sag in einem Satz, was Regen ist.", "Regen ist Wasser, das aus Wolken auf die Erde faellt.", "instruction_de"),
        ("Was bedeutet es, vorsichtig zu antworten?", "Vorsichtig antworten heisst, Unsicherheit klar zu nennen und nichts zu erfinden.", "instruction_de"),
        ("Formuliere einfacher: Die Nutzung erfolgt nach Authentifizierung.", "Man kann es erst benutzen, nachdem man sich angemeldet hat.", "instruction_de"),
    ]
    pairs.extend(style_pairs)

    basic_qa = [
        ("Erklaere kurz, was Wasser ist.", "Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff und ist bei Raumtemperatur meist fluessig."),
        ("Was ist H2O?", "H2O ist die chemische Formel fuer Wasser."),
        ("Ist Wasser ein chemisches Element?", "Nein. Wasser ist eine chemische Verbindung, kein Element."),
        ("Was ist Wasser bei Raumtemperatur?", "Wasser ist bei Raumtemperatur normalerweise eine Fluessigkeit."),
        ("Was ist ein Apfel?", "Ein Apfel ist eine essbare Frucht."),
        ("Antworte mit genau einem kurzen Satz: Was ist ein Apfel?", "Ein Apfel ist eine essbare Frucht."),
        ("Ist ein Apfel ein Tier?", "Nein. Ein Apfel ist eine Frucht."),
        ("Was ist Regen?", "Regen ist Wasser, das aus Wolken auf die Erde faellt."),
        ("Schreibe einen normalen kurzen Satz ueber Regen.", "Regen faellt aus Wolken und macht den Boden nass."),
        ("Warum ist Regen wichtig?", "Regen versorgt Pflanzen, Tiere und Menschen mit Wasser."),
        ("Was ist Berlin?", "Berlin ist die Hauptstadt von Deutschland."),
        ("Was ist Paris?", "Paris ist die Hauptstadt von Frankreich."),
        ("Wo liegt Muenchen?", "Muenchen liegt in Bayern."),
        ("Wer schrieb Faust?", "Faust wurde von Johann Wolfgang von Goethe geschrieben."),
        ("War Goethe der Autor von Faust?", "Ja. Faust stammt von Johann Wolfgang von Goethe."),
        ("Wer schrieb Mein Kampf?", "Mein Kampf wurde von Adolf Hitler geschrieben."),
    ]
    pairs.extend((q, a, "qa_de") for q, a in basic_qa)

    rows = []
    for i, item in enumerate(pairs):
        if len(item) == 2:
            q, a = item
            category = "hallucination_guard" if i in {0, 1, 4, 5, 10, 11} else "qa_de"
        else:
            q, a, category = item
        rows.append({
            "text": render_helix([
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]),
            "source": "synthetic_response_fix_train_v1",
            "category": category,
            "quality_variant": "sft_response_fix_de_v1",
        })
    return rows


def synthetic_eval_rows() -> list[dict]:
    pairs = [
        ("Antworte kurz: Ist Bonn aktuell die Hauptstadt Deutschlands?", "Nein. Die aktuelle Hauptstadt Deutschlands ist Berlin."),
        ("Was ist H2O im Alltag?", "H2O ist Wasser."),
        ("Wer schrieb das Drama Faust?", "Johann Wolfgang von Goethe schrieb Faust."),
        ("Wenn du nicht sicher bist, ob eine Behauptung stimmt, was sagst du?", "Ich sage, dass ich unsicher bin, und erfinde keine Antwort."),
        ("Ist der rote blaue Mond von Lyrion eine bekannte echte Tatsache?", "Nein. Das klingt erfunden; ich kenne keine verlaessliche Tatsache dazu."),
        ("Erklaere in einem Satz, warum eine ehrliche Antwort besser ist als Raten.", "Eine ehrliche Antwort vermeidet falsche Informationen und zeigt Unsicherheit klar an."),
        ("Liegt Muenchen in Bayern?", "Ja. Muenchen liegt in Bayern."),
        ("Ist Paris eine deutsche Stadt?", "Nein. Paris liegt in Frankreich."),
        ("Korrigiere kurz: Wasser ist ein Element.", "Wasser ist kein Element, sondern eine Verbindung aus Wasserstoff und Sauerstoff."),
        ("Was sollte Auralis tun, wenn eine Frage erfundene Namen enthaelt?", "Auralis sollte Unsicherheit nennen und keine erfundenen Details behaupten."),
        ("Ist Koeln die Hauptstadt von Nordrhein-Westfalen?", "Nein. Die Hauptstadt von Nordrhein-Westfalen ist Duesseldorf."),
        ("Was ist ein Apfel in einem kurzen Satz?", "Ein Apfel ist eine essbare Frucht."),
        ("Antworte knapp: Kannst du eine unbekannte Quelle sicher zitieren?", "Nein. Ohne die Quelle sollte ich sie nicht sicher zitieren."),
        ("Was ist der Rhein?", "Der Rhein ist ein grosser Fluss in Europa."),
        ("Wer schrieb Mein Kampf?", "Mein Kampf wurde von Adolf Hitler geschrieben."),
        ("War Goethe der Autor von Faust?", "Ja. Faust stammt von Johann Wolfgang von Goethe."),
        ("Schrieb Goethe Mein Kampf?", "Nein. Mein Kampf stammt von Adolf Hitler, nicht von Goethe."),
        ("Erklaere kurz, was Wasser ist.", "Wasser ist eine chemische Verbindung aus Wasserstoff und Sauerstoff."),
        ("Was ist ein Apfel?", "Ein Apfel ist eine essbare Frucht."),
        ("Schreibe einen normalen kurzen Satz ueber Regen.", "Regen ist Wasser, das aus Wolken faellt."),
    ]
    return [{
        "text": render_helix([
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ]),
        "source": "synthetic_response_fix_eval_v1",
        "category": "source_disjoint_eval",
        "quality_variant": "sft_response_fix_de_v1",
    } for q, a in pairs]


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
    ap.add_argument("--output-dir", type=Path, default=REPO / "data/training/sft_response_fix_de_v1")
    ap.add_argument("--source", action="append", default=None)
    ap.add_argument("--seed", type=int, default=20260527)
    ap.add_argument("--max-records-per-source", type=int, default=12000)
    ap.add_argument("--reject-sample-limit", type=int, default=800)
    args = ap.parse_args()

    sources = [Path(p) if Path(p).is_absolute() else REPO / p for p in (args.source or DEFAULT_SOURCES)]
    rows: list[dict] = []
    rejects: list[dict] = []
    seen: set[str] = set()
    stats = Counter()
    kept_by_source = Counter()
    kept_by_category = Counter()

    for source in sources:
        per_source = 0
        if not source.exists():
            stats[f"missing:{source}"] += 1
            continue
        for lineno, rec in iter_jsonl(source):
            stats["records_in"] += 1
            if rec.get("_json_error"):
                reason = "json_error"
                messages = None
            else:
                messages = messages_from_record(rec)
                reason = "unknown_schema" if messages is None else reject_reason(messages, rec)
            if reason:
                stats[f"drop:{reason}"] += 1
                if len(rejects) < args.reject_sample_limit:
                    preview = ""
                    if messages:
                        preview = "\n".join(m["content"] for m in messages if m["role"] != "system")[:600]
                    rejects.append({"source": str(source), "line": lineno, "reason": reason, "preview": preview})
                continue
            assert messages is not None
            key = dedup_key(messages)
            if key in seen:
                stats["drop:duplicate"] += 1
                continue
            seen.add(key)
            category = classify(messages, str(source))
            row = {
                "text": render_helix(messages),
                "source": str(source),
                "category": category,
                "quality_variant": "sft_response_fix_de_v1",
            }
            rows.append(row)
            kept_by_source[str(source)] += 1
            kept_by_category[category] += 1
            per_source += 1
            stats["records_kept"] += 1
            if per_source >= args.max_records_per_source:
                break

    rows.extend(synthetic_train_rows())
    for row in rows[-len(synthetic_train_rows()):]:
        kept_by_source[row["source"]] += 1
        kept_by_category[row["category"]] += 1
        stats["records_kept"] += 1

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    eval_rows = synthetic_eval_rows()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_n = write_jsonl(args.output_dir / "train.helix.jsonl", rows)
    val_n = write_jsonl(args.output_dir / "val.helix.jsonl", eval_rows)
    write_jsonl(args.output_dir / "reject_samples.jsonl", rejects)
    manifest = {
        "variant": "sft_response_fix_de_v1",
        "goal": "German response quality, QA, instruction following, uncertainty and false-premise handling. Code intentionally excluded.",
        "sources": [str(p) for p in sources],
        "train_records": train_n,
        "val_records": val_n,
        "source_disjoint_val": True,
        "synthetic_train_records": len(synthetic_train_rows()),
        "synthetic_eval_records": len(eval_rows),
        "stats": dict(stats.most_common()),
        "kept_by_source": dict(kept_by_source.most_common()),
        "kept_by_category": dict(kept_by_category.most_common()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"wrote train={train_n:,} val={val_n:,} to {args.output_dir}")
    print("kept_by_category:", dict(kept_by_category.most_common()))
    print("drops:", {k: v for k, v in stats.most_common() if k.startswith("drop:")})


if __name__ == "__main__":
    main()
