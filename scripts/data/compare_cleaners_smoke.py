#!/usr/bin/env python3
"""Compare cleaning strategies on representative small documents.

This is a smoke benchmark, not a production pipeline. Each sample has an
expected keep/drop decision plus required and forbidden fragments. The report
therefore answers a practical question: which chain best preserves useful prose
while rejecting the kinds of garbage that damaged the previous base run?

Optional dependencies are used when installed:

- trafilatura: HTML article extraction
- beautifulsoup4/lxml: simple HTML text extraction baseline
- datatrove: availability check for future web-scale orchestration
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.data.structure_clean_pretrain import clean_document, normalize_text  # noqa: E402


HTML_HINT_RE = re.compile(r"<\s*(?:html|body|article|main|section|div|p|h1|nav|footer)\b", re.I)


@dataclass(frozen=True)
class Sample:
    id: str
    kind: str
    text: str
    should_keep: bool
    must_contain: tuple[str, ...] = ()
    forbid_any: tuple[str, ...] = ()
    note: str = ""


SAMPLES = [
    Sample(
        id="good_photosynthesis",
        kind="clean_prose",
        should_keep=True,
        must_contain=("photosynthese", "chloroplasten", "calvin"),
        forbid_any=("cookie", "newsletter", "<html"),
        text=(
            "Die Photosynthese ist ein fundamentaler biochemischer Prozess, durch den Pflanzen, "
            "Algen und bestimmte Bakterien Lichtenergie in chemische Energie umwandeln. Dieser "
            "Vorgang findet primaer in den Chloroplasten statt, spezialisierten Organellen, die "
            "das Pigment Chlorophyll enthalten. Die Gesamtreaktion laesst sich vereinfacht durch "
            "die Gleichung 6CO2 + 6H2O + Lichtenergie -> C6H12O6 + 6O2 beschreiben. In der "
            "Lichtreaktion wird Wasser gespalten und Energie in Form von ATP und NADPH gespeichert. "
            "Darauf folgt der Calvin-Zyklus, in dem Kohlendioxid fixiert wird, um Glukose "
            "aufzubauen. Ohne diesen Prozess gaebe es keinen Sauerstoff in der Erdatmosphaere."
        ),
    ),
    Sample(
        id="html_article",
        kind="html",
        should_keep=True,
        must_contain=("berlin", "hauptstadt", "bundestag"),
        forbid_any=("cookie", "newsletter", "login", "<nav", "<footer"),
        text="""
        <html><body><nav>Home Login Datenschutz</nav><article>
        <h1>Berlin</h1><p>Berlin ist die Hauptstadt Deutschlands und zugleich ein eigenes Bundesland.
        Die Stadt ist ein politisches, kulturelles und wissenschaftliches Zentrum.</p>
        <p>Zu den bekannten Einrichtungen zaehlen der Bundestag, zahlreiche Universitaeten und Museen.
        Auch die Geschichte der Stadt spiegelt politische, soziale und kulturelle Umbrueche wider.</p>
        </article><footer>Cookie Policy Newsletter Subscribe</footer></body></html>
        """,
    ),
    Sample(
        id="html_boilerplate_only",
        kind="html_garbage",
        should_keep=False,
        forbid_any=("cookie", "subscribe", "privacy"),
        text=(
            "<html><body><nav>Home Login</nav><main>Accept all cookies. Privacy policy. "
            "Subscribe to our newsletter. Advertisement.</main><footer>All rights reserved.</footer></body></html>"
        ),
    ),
    Sample(
        id="boilerplate_repeat",
        kind="garbage",
        should_keep=False,
        forbid_any=("accept all cookies", "newsletter"),
        text="Accept all cookies. Subscribe to our newsletter. " * 24,
    ),
    Sample(
        id="mojibake_long",
        kind="encoding",
        should_keep=True,
        must_contain=("photosynthese", "pflanzen", "sauerstoff"),
        forbid_any=("Ã", "Â", "�"),
        text=(
            "Die Photosynthese ist ein Prozess, der in Pflanzen stattfindet. "
            "Sie ermÃ¶glicht die Umwandlung von Lichtenergie in chemische Energie. "
            "Dabei entsteht Sauerstoff, der fÃ¼r viele Lebewesen wichtig ist. "
            "In den Chloroplasten absorbiert Chlorophyll Licht und startet Reaktionen, "
            "bei denen Wasser gespalten wird. Anschliessend wird Kohlendioxid gebunden "
            "und in energiereiche Kohlenhydrate ueberfuehrt. Dieser Ablauf ist fuer "
            "Oekosysteme zentral, weil er Biomasse und Sauerstoff bereitstellt."
        ),
    ),
    Sample(
        id="list_noise",
        kind="mixed_structure",
        should_keep=True,
        must_contain=("wasser", "raumtemperatur", "fluessigkeit"),
        forbid_any=("home", "kontakt", "teilen"),
        text=(
            "- Home\n- Kontakt\n1. Teilen\n"
            "Wasser ist bei Raumtemperatur eine Fluessigkeit. Es besteht aus Molekuelen mit zwei "
            "Wasserstoffatomen und einem Sauerstoffatom. In der Natur kommt Wasser in Fluessen, "
            "Seen, Meeren, Wolken und Gletschern vor. Fuer Lebewesen ist Wasser notwendig, weil "
            "viele biologische Prozesse in waessriger Umgebung ablaufen."
        ),
    ),
    Sample(
        id="qa_for_pretrain",
        kind="qa_like",
        should_keep=False,
        forbid_any=("frage:", "antwort:", "###"),
        note="Base pretraining prose should not be dominated by SFT/QA template text.",
        text=(
            "### Aufgabe: Was ist Berlin?\n### Antwort: Berlin ist die Hauptstadt Deutschlands.\n" * 12
        ),
    ),
    Sample(
        id="english_mixed",
        kind="language_mixed",
        should_keep=False,
        forbid_any=("the model", "training data"),
        text=(
            "The model should learn from high quality training data. The article explains how "
            "data cleaning improves language model quality. It is written mostly in English and "
            "should not enter the German prose bucket. The text continues with general information "
            "about preprocessing, filtering, deduplication and dataset construction."
        ),
    ),
    Sample(
        id="short_fact",
        kind="too_short",
        should_keep=False,
        text="Berlin ist die Hauptstadt Deutschlands.",
    ),
    Sample(
        id="repetitive_good_words",
        kind="repetitive",
        should_keep=False,
        forbid_any=("berlin berlin",),
        text=("Berlin ist eine Stadt. " * 80),
    ),
    Sample(
        id="name_list",
        kind="list_like",
        should_keep=False,
        forbid_any=("schriftsteller", "philosoph", "regisseur"),
        text=(
            "Anna Beispiel (1920-1980), deutsche Schriftstellerin. Bernd Beispiel (1931-1999), "
            "deutscher Philosoph. Carla Beispiel (1944-2010), oesterreichische Regisseurin. "
            "Daniel Beispiel (1950-2015), Schweizer Journalist. Eva Beispiel (1961-2020), "
            "deutsche Sachbuchautorin. Franz Beispiel (1970-2021), deutscher Lyriker. "
            "Gisela Beispiel (1980-), deutsche Kulturwissenschaftlerin. " * 4
        ),
    ),
]


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _norm(text: str) -> str:
    return text.lower().replace("ß", "ss")


def auralis_clean(text: str) -> dict:
    doc, reason = clean_document(
        text,
        min_words=35,
        min_score=0.5,
        target_paragraph_chars=360,
        max_paragraph_chars=760,
    )
    if doc is None:
        return {"ok": False, "drop_reason": reason, "text": ""}
    return {
        "ok": True,
        "score": doc.score,
        "paragraphs": len(doc.paragraphs),
        "metrics": doc.metrics,
        "text": doc.text,
    }


def trafilatura_clean(text: str) -> dict:
    try:
        import trafilatura
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "text": ""}
    extracted = trafilatura.extract(text, include_comments=False, include_tables=False) or ""
    return {"ok": bool(extracted.strip()), "text": normalize_text(extracted)}


def bs4_clean(text: str) -> dict:
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        return {"ok": False, "error": repr(exc), "text": ""}
    soup = BeautifulSoup(text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    extracted = soup.get_text(" ")
    return {"ok": bool(extracted.strip()), "text": normalize_text(extracted)}


def plain_normalize(text: str) -> dict:
    cleaned = normalize_text(text)
    return {"ok": bool(cleaned), "text": cleaned}


def chain_clean(first: Callable[[str], dict], text: str) -> dict:
    first_result = first(text)
    first_text = first_result.get("text", "")
    if not first_text:
        return {
            "ok": False,
            "stage1_ok": first_result.get("ok", False),
            "drop_reason": "stage1_empty",
            "text": "",
        }
    second = auralis_clean(first_text)
    second["stage1_ok"] = first_result.get("ok", False)
    return second


def smart_extract_then_auralis(text: str) -> dict:
    if HTML_HINT_RE.search(text):
        first = trafilatura_clean(text)
        if not first.get("text"):
            first = bs4_clean(text)
        return chain_clean(lambda _: first, text)
    return auralis_clean(text)


def summarize_text(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    return text[:limit] + ("..." if len(text) > limit else "")


def judge_result(sample: Sample, result: dict) -> dict:
    text = result.get("text", "")
    text_norm = _norm(text)
    ok = bool(result.get("ok"))
    required_hits = [term for term in sample.must_contain if _norm(term) in text_norm]
    forbidden_hits = [term for term in sample.forbid_any if _norm(term) in text_norm]

    score = 0.0
    decision_correct = ok == sample.should_keep
    if decision_correct:
        score += 0.5
    if sample.should_keep:
        if ok and sample.must_contain:
            score += 0.35 * (len(required_hits) / len(sample.must_contain))
        elif ok:
            score += 0.35
        if ok and not forbidden_hits:
            score += 0.15
    else:
        if not ok:
            score += 0.5
        elif not forbidden_hits:
            score += 0.1

    return {
        "score": round(min(1.0, score), 4),
        "decision_correct": decision_correct,
        "required_hits": required_hits,
        "forbidden_hits": forbidden_hits,
    }


def run() -> dict:
    cleaners: dict[str, Callable[[str], dict]] = {
        "plain_normalize": plain_normalize,
        "bs4": bs4_clean,
        "trafilatura": trafilatura_clean,
        "auralis_structure": auralis_clean,
        "bs4_then_auralis": lambda text: chain_clean(bs4_clean, text),
        "trafilatura_then_auralis": lambda text: chain_clean(trafilatura_clean, text),
        "smart_extract_then_auralis": smart_extract_then_auralis,
    }
    aggregate: dict[str, dict] = {
        name: {"score": 0.0, "decision_correct": 0, "false_keep": 0, "false_drop": 0}
        for name in cleaners
    }
    report = {
        "versions": {
            "datatrove": _version("datatrove"),
            "trafilatura": _version("trafilatura"),
            "beautifulsoup4": _version("beautifulsoup4"),
        },
        "samples": [],
        "aggregate": {},
    }
    drop_reasons: dict[str, defaultdict[str, int]] = {name: defaultdict(int) for name in cleaners}

    for sample in SAMPLES:
        row = {
            "id": sample.id,
            "kind": sample.kind,
            "should_keep": sample.should_keep,
            "note": sample.note,
            "results": {},
        }
        for name, cleaner in cleaners.items():
            result = cleaner(sample.text)
            judgement = judge_result(sample, result)
            ok = bool(result.get("ok"))
            aggregate[name]["score"] += judgement["score"]
            aggregate[name]["decision_correct"] += int(judgement["decision_correct"])
            aggregate[name]["false_keep"] += int(ok and not sample.should_keep)
            aggregate[name]["false_drop"] += int((not ok) and sample.should_keep)
            if not ok:
                drop_reasons[name][str(result.get("drop_reason", "unknown"))] += 1

            compact = {k: v for k, v in result.items() if k != "text"}
            compact.update(judgement)
            compact["preview"] = summarize_text(result.get("text", ""))
            row["results"][name] = compact
        report["samples"].append(row)

    total = len(SAMPLES)
    ranking = []
    for name, stats in aggregate.items():
        avg = stats["score"] / total
        report["aggregate"][name] = {
            "avg_score": round(avg, 4),
            "decision_accuracy": round(stats["decision_correct"] / total, 4),
            "false_keep": stats["false_keep"],
            "false_drop": stats["false_drop"],
            "drop_reasons": dict(sorted(drop_reasons[name].items())),
        }
        ranking.append((avg, name))
    report["ranking"] = [
        {"strategy": name, "avg_score": round(score, 4)}
        for score, name in sorted(ranking, reverse=True)
    ]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=REPO / "data" / "eval" / "cleaner_compare_smoke.json")
    args = parser.parse_args()
    report = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["versions"], indent=2, ensure_ascii=False))
    print(json.dumps(report["ranking"], indent=2, ensure_ascii=False))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
