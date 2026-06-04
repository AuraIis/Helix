from pathlib import Path

from scripts.data.dataset_market_app import (
    DatasetMarket,
    _choose_split,
    _column_profile,
    _row_text,
    _sample_quality,
    _score,
    _trim_json_value,
)


def test_score_prefers_base_corpus_over_sft_template_dataset():
    corpus = {
        "id": "org/german-clean-web-corpus",
        "downloads": 100_000,
        "likes": 500,
        "tags": ["language:de", "license:cc-by-4.0", "text", "corpus", "pretrain"],
    }
    chat = {
        "id": "org/german-chat-sft-dpo",
        "downloads": 100_000,
        "likes": 500,
        "tags": ["language:de", "license:cc-by-4.0", "instruction", "chat", "dpo"],
    }

    corpus_card = _score(corpus, goal="base_pretrain", language="de")
    chat_card = _score(chat, goal="base_pretrain", language="de")

    assert corpus_card.score > chat_card.score
    assert corpus_card.score_label in {"strong", "usable"}
    assert any("post-training" in warning for warning in chat_card.warnings)


def test_plan_mix_downweights_warning_heavy_dataset():
    market = DatasetMarket(Path("I:/KI/Auralis_datasets"))
    strong = _score(
        {
            "id": "org/german-pretrain",
            "downloads": 10_000,
            "likes": 100,
            "tags": ["language:de", "license:cc-by-4.0", "corpus", "pretrain"],
        },
        goal="base_pretrain",
        language="de",
    )
    risky = _score(
        {
            "id": "org/german-unknown-license-chat",
            "downloads": 10_000,
            "likes": 100,
            "tags": ["language:de", "license:unknown", "chat", "sft"],
        },
        goal="base_pretrain",
        language="de",
    )

    mix = market.plan_mix([strong.__dict__, risky.__dict__], 1_000_000, "base_pretrain")

    assert mix["items"][0]["id"] == strong.id
    assert mix["items"][0]["weight"] > mix["items"][1]["weight"]
    assert sum(item["target_tokens"] for item in mix["items"]) <= 1_000_000


def test_pipeline_uses_assembled_text_input_not_raw_dataset_directory():
    market = DatasetMarket(Path("I:/KI/Auralis_datasets"))
    card = _score(
        {
            "id": "org/german-math",
            "downloads": 1000,
            "likes": 20,
            "tags": ["language:de", "license:mit", "math", "problem"],
        },
        goal="math",
        language="de",
    )
    mix = market.plan_mix([card.__dict__], 100_000, "math")

    pipeline = market.pipeline([card.__dict__], mix)
    commands = "\n".join(pipeline["commands"])

    assert ".assembled.txt" in commands
    assert "--min-language-signal 0.0" in commands
    assert "raw_dir" in pipeline["manifest"]["datasets"][0]
    assert pipeline["manifest"]["datasets"][0]["assembled_text"].endswith(".assembled.txt")


def test_sample_quality_rewards_clean_german_longform_and_penalizes_html():
    clean = [
        "Die Photosynthese ist ein biologischer Prozess, bei dem Pflanzen Lichtenergie in chemische Energie umwandeln. "
        "Der Vorgang ist fuer das Leben auf der Erde wichtig und wird in vielen Lehrtexten beschrieben."
        for _ in range(5)
    ]
    noisy = [
        "<html><div>Subscribe to our newsletter</div><script>javascript</script> http://example.com cookie banner"
        for _ in range(5)
    ]

    clean_report = _sample_quality(clean, goal="base_pretrain", language="de")
    noisy_report = _sample_quality(noisy, goal="base_pretrain", language="de")

    assert clean_report["score_delta"] > noisy_report["score_delta"]
    assert clean_report["estimated_keep_rate"] > noisy_report["estimated_keep_rate"]
    assert noisy_report["warnings"]


def test_sample_quality_penalizes_ocr_character_noise():
    ocr = [
        "D-er D^MfHe UWMkmg im Jahre 1866. Aiisbrnches der Feindseligkeiten und Htadtvermal"
        for _ in range(6)
    ]

    report = _sample_quality(ocr, goal="base_pretrain", language="de")

    assert report["score_delta"] < 0
    assert any("OCR" in warning for warning in report["warnings"])
    assert report["estimated_keep_rate"] < 0.86


def test_search_filters_exclude_bad_terms_without_network():
    market = DatasetMarket()

    class FakeApi:
        def list_datasets(self, **_kwargs):
            return [
                {
                    "id": "org/german-clean-corpus",
                    "downloads": 100,
                    "likes": 10,
                    "tags": ["language:de", "license:mit", "corpus", "pretrain"],
                },
                {
                    "id": "org/german-chat-sft",
                    "downloads": 1000,
                    "likes": 50,
                    "tags": ["language:de", "license:mit", "chat", "sft"],
                },
                {
                    "id": "org/german-audio-asr",
                    "downloads": 1000,
                    "likes": 50,
                    "tags": ["language:de", "license:mit", "audio", "asr"],
                },
            ]

    market._api = FakeApi()

    result = market.search(
        query="german",
        language="de",
        goal="base_pretrain",
        limit=10,
        min_score=0,
        license_mode="open",
        language_mode="strict",
        exclude_terms="sft, chat, audio, asr",
    )

    assert [item["id"] for item in result["items"]] == ["org/german-clean-corpus"]
    assert result["fetched"] == 3


def test_choose_split_prefers_default_train_then_first_train():
    splits = [{"config": "web", "split": "wikipedia"}, {"config": "default", "split": "train"}]
    assert _choose_split(splits) == {"config": "default", "split": "train"}
    assert _choose_split(splits, "web", "wikipedia") == {"config": "web", "split": "wikipedia"}


def test_column_profile_summarizes_rows():
    rows = [
        {"text": "Hallo Welt", "num_tokens": 2, "license": ["mit"]},
        {"text": "Noch ein Text", "num_tokens": 3, "license": []},
    ]

    profile = {item["name"]: item for item in _column_profile(rows)}

    assert profile["text"]["non_empty"] == 2
    assert profile["num_tokens"]["types"] == ["int"]
    assert profile["license"]["non_empty"] == 1


def test_preview_text_is_bounded_before_returning_to_browser():
    huge = "x" * 20_000

    assert len(_row_text({"text": huge})) < 5_000
    assert len(_trim_json_value({"nested": {"text": huge}})["nested"]["text"]) < 3_000
