import json

from scripts.data.download_german_commons_selected import load_targets, row_payload, safe_slug


def test_load_targets_uses_take_first_by_default(tmp_path):
    plan = {
        "take_first": [{"config": "web", "split": "wikipedia"}],
        "small_specialty": [{"config": "cultural", "split": "wikiquote"}],
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")

    targets = load_targets(path, include_special=False)

    assert targets == [{"config": "web", "split": "wikipedia"}]


def test_row_payload_keeps_training_fields():
    row = {
        "id": "doc1",
        "source": "Wikipedia",
        "subset": 6,
        "text": "Ein sauberer deutscher Text.",
        "license": ["cc-by-sa-4.0"],
        "num_tokens": 12,
        "perplexity": 123.4,
        "ocr_score": 100,
        "extra": "ignored",
    }

    payload = row_payload(row, "web", "wikipedia")

    assert payload["text"] == row["text"]
    assert payload["config"] == "web"
    assert payload["split"] == "wikipedia"
    assert "extra" not in payload


def test_safe_slug_removes_path_separators():
    assert safe_slug("web", "wiki/discussions") == "web__wiki__discussions"
