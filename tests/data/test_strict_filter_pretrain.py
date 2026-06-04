from __future__ import annotations

from argparse import Namespace

from scripts.data.strict_filter_pretrain import normalize, reject_reason


def args(**overrides):
    base = dict(
        language="german",
        profile="prose",
        min_chars=80,
        max_chars=80_000,
        max_urls=1,
        min_alpha=0.45,
        max_symbol=0.28,
        max_repetition=0.62,
        v3_structure_filters=True,
        max_list_score=8,
        max_name_list_hits=30,
        max_ocr_hits=2,
        max_bibliography_hits=3,
        max_speaker_labels=18,
        max_digit_ratio=0.16,
        max_ngram_repetition=0.28,
        drop_web_boilerplate=False,
        max_web_boilerplate_hits=1,
        drop_old_ocr=False,
        max_old_ocr_hits=8,
    )
    base.update(overrides)
    return Namespace(**base)


GOOD_PROSE = (
    "Die Photosynthese ist ein grundlegender biologischer Prozess, bei dem Pflanzen Lichtenergie "
    "in chemische Energie umwandeln. Dabei wird Wasser gespalten und Kohlendioxid in organische "
    "Verbindungen eingebaut. Dieser Vorgang findet in den Chloroplasten statt und bildet eine "
    "wichtige Grundlage vieler Nahrungsketten. Außerdem beeinflusst er den Sauerstoffgehalt der "
    "Atmosphäre und damit die Entwicklung komplexen Lebens."
)


def test_clean_v3_keeps_normal_german_prose():
    assert reject_reason(normalize(GOOD_PROSE), args()) is None


def test_clean_v3_drops_table_of_contents():
    raw = (
        "CONTENTS PAGE From Heine's Buch der Lieder .... 149 Im wunderschönen Monat Mai .... 150 "
        "Aus Flügeln des Gesanges .... 151 Ich grolle nicht .... 154 Postscript — In praise of "
        "Robert Burns .... 181"
    )
    assert reject_reason(normalize(raw), args()) == "toc_or_index"


def test_clean_v3_drops_name_catalogue():
    raw = (
        "Aa Bertus Aafjes (1914–1993), NL Jeppe Aakjær (1866–1930), DK Johannes Aal (1500–1551), "
        "Hans Aanrud (1863–1953), Emil Aarestrup (1800–1856), Ivar Aasen (1813–1896), Jacob Abbott "
        "(1803–1879), Edward Abbey (1927–1989), Max Abe (1865–1949), Martha Abicht (1878–1941), "
        "Peter Abrahams (1919–2017), Dannie Abse (1923–2014), Edmond About (1828–1885), Chinua "
        "Achebe (1930–2013), Ilse Aichinger (1921–2016), Hans Fallada (1893–1947), Goethe (1749–1832)."
    )
    assert reject_reason(normalize(raw), args()) == "name_catalogue"


def test_clean_v3_math_profile_keeps_symbolic_problem():
    raw = (
        "Problem: Points A, B, C, and D lie on a line, in that order. If AB=2 units, BC=5 units "
        "and AD=14 units, what is the ratio of AC to BD? Solution: AC = AB + BC = 7. Since "
        "AB + BC + CD = AD, we get CD = 7 and BD = 12. Therefore the ratio is 7/12."
    )
    assert reject_reason(normalize(raw), args(language="english", profile="math", min_chars=60)) is None


def test_clean_v3_booster_profile_keeps_short_math_booster():
    raw = "Rechenbeispiel: 15207 - 9088 = 6119. Subtraktion bedeutet, dass 9088 von 15207 abgezogen wird."
    assert reject_reason(normalize(raw), args(profile="booster", min_chars=40)) is None


def test_clean_v3_drops_chat_markers_from_base():
    raw = "<|system|>\nDu bist Auralis.\n<|end|>\n<|user|>\nHallo\n<|end|>"
    assert reject_reason(normalize(raw), args(min_chars=20)) == "chat_marker"


def test_clean_v3_drops_dialogue_script_fragments():
    raw = (
        "PROMETHEUS. Merkur! MERKUR. Was willst du? PROMETHEUS. Das Schicksal. "
        "MERKUR. Jupiter ruft. PROMETHEUS. Ich bleibe. MINERVA. Sei ruhig. "
        "PROMETHEUS. Freiheit! MERKUR. Gehorche. PROMETHEUS. Nein. "
        "MERKUR. So sprich. PROMETHEUS. Ich schaffe Menschen. MINERVA. Genug. "
        "PROMETHEUS. Noch nicht. MERKUR. Jupiter wartet. PROMETHEUS. Dann warte er. "
        "MERKUR. PROMETHEUS. MINERVA. PROMETHEUS. MERKUR. PROMETHEUS."
    )
    assert reject_reason(normalize(raw), args()) == "dialogue_script"


def test_clean_v31_drops_web_boilerplate():
    raw = (
        "Verfassen Sie die erste Bewertung zu diesem Artikel. "
        "Zwei Klicks für mehr Datenschutz: Beim Laden der Seite werden noch keine "
        "Daten an Dritte übertragen. Erst wenn du auf den Teilen-Button klickst, "
        "kannst du deine Empfehlung an Facebook, Google+ oder Twitter senden. "
        "Danach folgt ein kurzer normaler Produkttext mit vielen deutschen Wörtern."
    )
    assert reject_reason(normalize(raw), args(drop_web_boilerplate=True)) == "web_boilerplate"


def test_clean_v31_drops_strong_old_ocr():
    raw = (
        "Amtsblatt der Gesetzsammlung. I. Abtheilung. Das Verzeichniß der Akten "
        "muß bei dem Civilprozeß vollständig sein. Ueber das Verfahren wird "
        "thatsächlich entschieden; die be stimmten Regeln sind ge stellt und "
        "das ver fahr wird ent schieden. Ferner wird die Zustellungsurkunde "
        "im Verwaltungsgericht und im alten Behördenstil erläutert."
    )
    assert reject_reason(normalize(raw), args(drop_old_ocr=True, max_old_ocr_hits=5)) == "old_ocr_or_fraktur"
