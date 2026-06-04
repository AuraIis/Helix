# Auralis Dataset Market

Local MVP for discovering public Hugging Face datasets, scoring them for an
Auralis training goal, planning a weighted data mix, and generating a first
download/cleaning plan for `I:\KI\Auralis_datasets`.

## Run

```powershell
python scripts\data\dataset_market_app.py --host 127.0.0.1 --port 8765 --output-root I:\KI\Auralis_datasets
```

Open:

```text
http://127.0.0.1:8765
```

## Current Workflow

1. Search Hugging Face datasets by query, language, goal, and result limit.
2. Filter candidates before ranking:
   - minimum score
   - permissive-license-only mode
   - strict or soft language matching
   - include terms that must be present
   - exclude terms such as `sft`, `chat`, `asr`, `audio`, `image`, `embed`
   - quality, low-risk, download, or like sorting
3. Score candidates with metadata signals:
   - downloads and likes
   - license risk
   - language match
   - goal-specific hints for base pretraining, math, code, or SFT
   - warnings for gated/private/unknown-license/template-heavy datasets
4. Select promising datasets.
5. Generate a weighted token mix.
6. Generate a local PowerShell-oriented pipeline plan:
   - create `market_raw` and `market_clean`
   - download each selected HF dataset
   - assemble/export downloaded data into one UTF-8 text file
   - run `structure_clean_pretrain.py` with the route-specific thresholds
7. Run `KI pruefen` for a single dataset:
   - fetch fresh metadata, README/card data, and visible repository files
   - try Hugging Face first-row samples with timeout
   - fall back to local streaming samples in a short-lived subprocess
   - detect HTML/boilerplate, instruction templates, weak German signal, short
     examples, media-heavy repos, and OCR/character-noise patterns
   - produce an AI score, verdict, estimated keep rate, warnings, and examples
8. Run `Preview` for a single dataset:
   - list visible configs/splits
   - inspect first rows without downloading the full dataset
   - profile columns, types, and non-empty examples
   - list matching Parquet shards and visible size
   - run the lightweight sample-quality check on preview rows

## Cleaning Routes

`auralis_structure`:
General German text cleaning with structure repair and normal language signal.

`extract_then_auralis`:
For web/crawl-like corpora that likely need extraction before the Auralis
structure cleaner.

`math_structure_min_language`:
For math corpora where formulas and mixed notation should not be rejected by a
strict German-language signal.

`code_preserve_structure`:
For code/programming data where line structure matters and language filtering is
relaxed.

## Next Steps

The MVP intentionally does not blindly execute downloads yet. The next useful
upgrade is a full dataset profiler that downloads a bounded shard from selected
files, runs the real cleaner on it, and reports measured junk rate, duplicate
rate, average document length, language mix, and usable token yield before a full
download is approved.
