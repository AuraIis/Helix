from scripts.data.audit_german_commons import classify, markdown


def test_classify_flags_ocr_noise_for_hard_filter():
    report = {
        "split": "blbooks",
        "sample_count": 12,
        "cleaner_keep_rate": 0.8,
        "sample_quality": {
            "estimated_keep_rate": 0.7,
            "warnings": ["OCR or character-noise hints in samples: 9"],
        },
        "median_ocr_score": 70,
        "median_perplexity": 900,
    }

    decision, reason = classify(report)

    assert decision == "hard_filter"
    assert "OCR" in reason


def test_markdown_renders_ranked_table():
    payload = {
        "dataset": "coral-nlp/german-commons",
        "generated_at": "now",
        "samples_per_split": 2,
        "splits": [
            {
                "decision": "take",
                "config": "web",
                "split": "wikipedia",
                "cleaner_keep_rate": 1.0,
                "sample_quality": {"estimated_keep_rate": 0.9},
                "median_ocr_score": 100,
                "median_perplexity": 400,
                "decision_reason": "good",
            }
        ],
    }

    text = markdown(payload)

    assert "German Commons Audit" in text
    assert "| take | web | wikipedia |" in text
