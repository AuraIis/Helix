"""Tests for the structure-aware prose cleaning pass."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.data.structure_clean_pretrain import clean_document, normalize_text


def test_normalize_removes_html_urls_and_repairs_mojibake() -> None:
    raw = "<div>Die Photosynthese ist ein Prozess.</div> Mehr unter https://example.org Ã¼ber Pflanzen."
    text = normalize_text(raw)
    assert "<div>" not in text
    assert "https://example.org" not in text
    assert "über Pflanzen" in text


def test_clean_document_builds_structured_paragraphs() -> None:
    raw = (
        "Home\n"
        "Die Photosynthese ist ein fundamentaler biochemischer Prozess, durch den Pflanzen Lichtenergie "
        "in chemische Energie umwandeln. Dieser Vorgang findet hauptsächlich in den Chloroplasten statt, "
        "die das Pigment Chlorophyll enthalten. Die Lichtreaktion spaltet Wasser und speichert Energie "
        "in Form von ATP und NADPH. Im Calvin-Zyklus wird Kohlendioxid fixiert und Glukose aufgebaut. "
        "Ohne diesen Prozess gäbe es keinen Sauerstoff in der Erdatmosphäre, was die Entwicklung "
        "komplexen Lebens stark beeinflusst hat."
    )
    doc, reason = clean_document(
        raw,
        min_words=40,
        min_score=0.5,
        target_paragraph_chars=180,
        max_paragraph_chars=320,
    )
    assert reason is None
    assert doc is not None
    assert "Home" not in doc.text
    assert "Photosynthese" in doc.text
    assert doc.score >= 0.5
    assert doc.metrics["sentences"] >= 4


def test_clean_document_rejects_boilerplate() -> None:
    raw = "Accept all cookies. Privacy policy. Subscribe to our newsletter. " * 20
    doc, reason = clean_document(
        raw,
        min_words=20,
        min_score=0.5,
        target_paragraph_chars=180,
        max_paragraph_chars=320,
    )
    assert doc is None
    assert reason == "boilerplate"


def test_clean_document_rejects_disambiguation_list_article() -> None:
    raw = (
        "Königsberg heißen folgende geographische Objekte: Königsberg in Bayern, Stadt im "
        "Landkreis Haßberge; Kongsberg, Stadt in Norwegen; Chojna, Stadt in Polen; "
        "Nová Baňa, Stadt in der Slowakei; Königsberg, Ortsteil einer Gemeinde; "
        "Königsberg, historischer Name verschiedener Siedlungen. Weitere Bedeutungen sind "
        "Berge, Burgen, Fluren, Familiennamen und literarische Titel."
    )
    doc, reason = clean_document(
        raw,
        min_words=30,
        min_score=0.5,
        target_paragraph_chars=180,
        max_paragraph_chars=320,
    )
    assert doc is None
    assert reason == "list_article"
