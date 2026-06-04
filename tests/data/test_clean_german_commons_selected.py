import gzip
import json
from argparse import Namespace

from scripts.data.clean_german_commons_selected import (
    clean_split,
    combine_texts,
    metadata_reject_reason,
    run,
    source_paths,
)


GOOD_TEXT = (
    "Die Photosynthese ist ein grundlegender Prozess, bei dem Pflanzen Lichtenergie in chemische Energie "
    "umwandeln. Dabei wird Wasser gespalten und Kohlendioxid in organische Verbindungen eingebaut. "
    "Dieser Vorgang findet in den Chloroplasten statt und liefert die Grundlage fuer viele Nahrungsketten. "
    "Er beeinflusst ausserdem den Sauerstoffgehalt der Atmosphaere und damit die Entwicklung komplexen Lebens."
)


def test_metadata_reject_reason_applies_source_filters():
    assert metadata_reject_reason({"ocr_score": 80}, {"min_ocr_score": 90}) == "low_ocr_score"
    assert metadata_reject_reason({"perplexity": 900}, {"max_perplexity": 700}) == "high_perplexity"
    assert metadata_reject_reason({"ocr_score": 95, "perplexity": 300}, {"min_ocr_score": 90}) is None


def test_source_paths_match_downloader_layout(tmp_path):
    input_path, output_jsonl, output_text = source_paths(tmp_path / "raw", tmp_path / "clean", "web", "wikipedia")

    assert input_path.as_posix().endswith("raw/web/web__wikipedia.jsonl.gz")
    assert output_jsonl.as_posix().endswith("clean/web/web__wikipedia.clean.jsonl")
    assert output_text.as_posix().endswith("clean/web/web__wikipedia.clean.txt")


def test_clean_split_reads_jsonl_gz_text_field_only(tmp_path):
    input_path = tmp_path / "raw" / "web" / "web__wikipedia.jsonl.gz"
    input_path.parent.mkdir(parents=True)
    rows = [
        {
            "text": GOOD_TEXT,
            "source": "Wikipedia",
            "subset": "Web",
            "license": "cc-by-sa-4.0",
            "num_tokens": 80,
            "perplexity": 120,
            "ocr_score": 99,
        },
        {"text": "Accept all cookies. Privacy policy. Subscribe to our newsletter. " * 8},
    ]
    with gzip.open(input_path, "wt", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = clean_split(
        input_path=input_path,
        output_jsonl=tmp_path / "clean" / "web" / "web__wikipedia.clean.jsonl",
        output_text=tmp_path / "clean" / "web" / "web__wikipedia.clean.txt",
        config="web",
        split="wikipedia",
        global_filters={"min_words": 30, "min_quality_score": 0.4, "min_language_signal": 0.03},
        source_filters={"min_ocr_score": 90, "max_perplexity": 700},
        max_docs=0,
        flush_every=0,
    )

    assert manifest.docs_in == 2
    assert manifest.docs_written == 1
    out_text = (tmp_path / "clean" / "web" / "web__wikipedia.clean.txt").read_text(encoding="utf-8")
    assert "Photosynthese" in out_text
    assert '"text"' not in out_text


def test_combine_texts_concatenates_split_outputs(tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("eins\n", encoding="utf-8")
    second.write_text("zwei\n", encoding="utf-8")

    class Manifest:
        output_text: str

        def __init__(self, path):
            self.output_text = str(path)

    summary = combine_texts([Manifest(first), Manifest(second)], tmp_path / "combined.txt")

    assert summary["docs"] == 2
    assert (tmp_path / "combined.txt").read_text(encoding="utf-8") == "eins\nzwei\n"


def test_run_can_skip_missing_sources(tmp_path):
    plan = {
        "dataset": "coral-nlp/german-commons",
        "global_filters": {"min_words": 30, "min_quality_score": 0.4, "min_language_signal": 0.03},
        "take_first": [{"config": "web", "split": "wikipedia"}, {"config": "legal", "split": "eurlex"}],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    input_path = tmp_path / "raw" / "web" / "web__wikipedia.jsonl.gz"
    input_path.parent.mkdir(parents=True)
    with gzip.open(input_path, "wt", encoding="utf-8") as fh:
        fh.write(json.dumps({"text": GOOD_TEXT}, ensure_ascii=False) + "\n")

    summary = run(
        Namespace(
            plan=plan_path,
            input_root=tmp_path / "raw",
            output_root=tmp_path / "clean",
            combined_text=tmp_path / "clean" / "combined.txt",
            manifest=tmp_path / "clean" / "manifest.json",
            include_special=False,
            include_hard_filter=False,
            skip_missing=True,
            max_docs_per_split=0,
            flush_every=0,
        )
    )

    assert summary["docs_in"] == 1
    assert summary["docs_written"] == 1
