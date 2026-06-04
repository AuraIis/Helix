from __future__ import annotations

from scripts.eval.run_capability_probes import Probe, repetition_ratio, score_answer


def test_capability_probe_scores_expected_keyword():
    probe = Probe(
        id="fact",
        category="facts",
        prompt="Frage",
        expect_any=["Berlin"],
        forbid_any=["Frankfurt"],
    )
    result = score_answer("Die Hauptstadt ist Berlin.", probe)
    assert result.score == 1.0
    assert result.matched == ["Berlin"]
    assert result.forbidden == []


def test_capability_probe_penalizes_forbidden_and_garbage():
    probe = Probe(
        id="bad",
        category="facts",
        prompt="Frage",
        expect_any=["Goethe"],
        forbid_any=["Mein Kampf"],
    )
    result = score_answer("Goethe schrieb Mein Kampf <a href='x'>", probe)
    assert result.score < 1.0
    assert "Mein Kampf" in result.forbidden
    assert result.garbage


def test_repetition_ratio_detects_looping_text():
    text = "Berlin ist Berlin ist Berlin ist Berlin ist Berlin ist"
    assert repetition_ratio(text) > 0.25
