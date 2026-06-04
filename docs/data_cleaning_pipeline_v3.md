# Data Cleaning Pipeline V3

Goal: keep only text that is useful for base pretraining and shape prose into
stable, readable paragraphs before tokenization.

Current production candidate after the rescue work is the v4 boosted mix:

- mix builder: `scripts/data/build_pretrain_mix_v4_boosted.py`
- tokenized output: `tokenized/pretrain_mix_v4_boosted/`
- 500M canary config: `configs/training/pretrain_mix_v4_boosted_500m.yaml`

Next 1B-data candidate is the v3.2/v5 pipeline:

- cleaner: `scripts/data/run_clean_v32_pretrain.py`
- mix builder: `scripts/data/build_pretrain_mix_v5_boosted.py`
- clean output: `data/training/pretrain_clean_v32/`
- boosted mix output: `data/training/pretrain_mix_v5_boosted/`
- purpose: revalidate clean-v3.1, add strict-filtered FineWeb2-DE,
  FineWeb-Edu, and DCLM-Edu, then add small QA/math/chat/reddit boosters.
- Knowledge-DNA is disabled by default in v5 until the ablation has a clear
  positive signal.

Older names like `pretrain_clean_v2` and `pretrain_clean_v3/v31` are lineage
steps, not the final recommended training target by themselves.

## Recommended Stages

1. Extract raw text
   - HTML/PDF/web dumps become one document per record.
   - Keep source manifests and byte counts.

2. Fast reject
   - Remove empty, tiny, URL-heavy, symbol-heavy, HTML-heavy, mojibake, and
     obvious boilerplate documents.
   - Existing tool: `scripts/data/strict_filter_pretrain.py`.

3. Structure clean
   - Convert surviving prose into clean paragraphs.
   - Remove navigation/list/header leftovers.
   - Score each document and write JSONL for auditing plus one-line text for
     training.
   - Tool: `scripts/data/structure_clean_pretrain.py`.
   - Current smoke-test winner: route HTML through extraction first, then run
     the Auralis structure pass:
     `trafilatura -> auralis` with `bs4 -> auralis` fallback for HTML, and
     direct `auralis` for already extracted plain text.

4. Deduplicate
   - Use exact document hashes locally.
   - For larger corpora, add MinHash/exact-substring dedupe with Datatrove.

5. Quality audit
   - Track kept/dropped ratios, repetition, token efficiency, samples, and
     source contribution.
   - Existing tool: `scripts/data/quality_scores.py`.

6. Mix and tokenize
   - Mix source-balanced text with a representative validation tail.
   - Existing tools: `scripts/data/build_pretrain_mix_v2.py`,
     `scripts/data/tokenize_for_pretraining.py`.

## Example

```bash
python scripts/data/strict_filter_pretrain.py \
  --input data/training/pretrain_clean_v2/german.raw.txt \
  --output data/training/pretrain_clean_v2/german.strict.txt \
  --language german

python scripts/data/structure_clean_pretrain.py \
  --input data/training/pretrain_clean_v2/german.strict.txt \
  --output-jsonl data/training/pretrain_clean_v3/german.structured.jsonl \
  --output-text data/training/pretrain_clean_v3/german.structured.txt \
  --min-words 80 \
  --min-score 0.62
```

Compare strategies before scaling up:

```bash
python scripts/data/compare_cleaners_smoke.py \
  --output data/eval/cleaner_compare_smoke.json
```

German Commons selected-source flow:

```bash
python scripts/data/download_german_commons_selected.py \
  --plan configs/data/german_commons_clean_plan_v1.json \
  --output-root I:/KI/Auralis_datasets/german_commons_selected_raw \
  --log-every 5000

python scripts/data/clean_german_commons_selected.py \
  --plan configs/data/german_commons_clean_plan_v1.json \
  --input-root I:/KI/Auralis_datasets/german_commons_selected_raw \
  --output-root I:/KI/Auralis_datasets/german_commons_selected_clean \
  --combined-text I:/KI/Auralis_datasets/german_commons_selected_clean/german_commons_selected.clean.txt \
  --manifest I:/KI/Auralis_datasets/german_commons_selected_clean/manifest.json
```

For a quick smoke while a large split is still downloading, add
`--skip-missing --max-docs-per-split 500`.

Clean-v3.2 smoke/full run:

```bash
python scripts/data/run_clean_v32_pretrain.py \
  --clean-v31-dir data/training/pretrain_clean_v31 \
  --raw-1b-root data/pretrain_1b_sources_v1/raw \
  --output-dir data/training/pretrain_clean_v32 \
  --val-tail-bytes 120000000

python scripts/data/build_pretrain_mix_v5_boosted.py \
  --base data/training/pretrain_clean_v32/mix_full.txt \
  --out-dir data/training/pretrain_mix_v5_boosted
```

The JSONL keeps audit structure:

```json
{"text":"Absatz 1...\n\nAbsatz 2...","paragraphs":["Absatz 1...","Absatz 2..."],"quality_score":0.91}
```

The text output keeps the current tokenizer contract:

```text
Absatz 1... Absatz 2...
```

## Why This Shape

Datatrove is the right large-scale orchestrator for web-scale extraction,
filtering, and deduplication. For this repo, the first practical win is a
local structure pass: it catches the exact failure mode we saw in the old run,
where the model learned web fragments, boilerplate, bad formatting, and noisy
associations instead of clean prose.
