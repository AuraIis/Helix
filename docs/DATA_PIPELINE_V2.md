# Data Pipeline v2

Status: **partially implemented**.

This document is the operational guide for the curated corpus wave after the
initial canary infrastructure work. The emphasis is:

- explicit source budgets
- explicit keep / replace / acquire decisions
- lightweight local quality filtering
- no silent source substitution

## Implemented now

The following pieces are now in the repo:

- [`configs/data/curated_40b_mix.yaml`](I:/AuralisV2/configs/data/curated_40b_mix.yaml)  
  locked 40B source plan
- [`scripts/data/prepare_curated_corpus.py`](I:/AuralisV2/scripts/data/prepare_curated_corpus.py)  
  inventory + missing-source report + optional delegated downloads
- [`scripts/data/filter_quality.py`](I:/AuralisV2/scripts/data/filter_quality.py)  
  local quality pass for already-downloaded text files
- [`scripts/data/mix_corpora.py`](I:/AuralisV2/scripts/data/mix_corpora.py)  
  deterministic source-budgeted mixer
- existing per-source downloaders:
  - [`download_english.py`](I:/AuralisV2/scripts/data/download_english.py)
  - [`download_german.py`](I:/AuralisV2/scripts/data/download_german.py)
  - [`download_code.py`](I:/AuralisV2/scripts/data/download_code.py)

The download scripts now accept per-source token overrides so the curated 40B
plan can drive them directly.

## Target corpus

The curated target is **40B tokens post-filter**:

- English: **30B**
- German: **8B**
- Code: **2B**

Current policy is encoded in
[`curated_40b_mix.yaml`](I:/AuralisV2/configs/data/curated_40b_mix.yaml).

## Workflow

```text
prepare -> download -> quality-filter -> mix -> tokenize -> contamination-check
```

### 1. Prepare / inventory

```bash
python scripts/data/prepare_curated_corpus.py
```

Outputs:

- `data/eval/curated_corpus_report.md`
- `data/eval/curated_corpus_report.json`

This tells us exactly which planned sources already exist on the NAS and which
still need to be fetched.

### 2. Download missing sources

```bash
python scripts/data/prepare_curated_corpus.py --download-missing
```

This delegates to the existing source downloaders and passes exact token
budgets from the curated plan.

### 3. Quality-filter raw sources

Example:

```bash
python scripts/data/filter_quality.py ^
  --input "\\<HOST>\Auralis\AuralisV2\raw\english\fineweb2_en.txt" ^
  --output "\\<HOST>\Auralis\AuralisV2\cleaned\fineweb2_en.filtered.txt" ^
  --language english
```

Current filters:

- length bounds
- URL density
- symbol density
- boilerplate patterns
- obvious mojibake
- extreme repetition

### 4. Mix the curated corpus

```bash
python scripts/data/mix_corpora.py
```

Default output:

- `data/training/curated_40b/english.txt`
- `data/training/curated_40b/german.txt`
- `data/training/curated_40b/code.txt`
- `data/training/curated_40b/mix_manifest.json`

The mixer is intentionally strict: if a required source is missing or too
small, it fails instead of improvising.

### 5. Tokenize

After mixing, run the existing tokenisation path on the final mixed files.

### 6. Contamination gate

Run the existing
[`contamination_check.py`](I:/AuralisV2/scripts/data/contamination_check.py)
before launch.

## Keep / replace / drop summary

### Keep

- FineWeb-Edu
- Wikipedia EN
- smaller OpenMath allocation
- OpenWebMath only as a **small** structured top-up

### Replace

- `starcoderdata.txt` -> replace with `the_stack_v2.txt`
- `cleaned/german.txt` as sole German truth -> replace with explicit German
  sources

### Acquire

- FineWeb2 EN
- FineWeb2 DE
- German Commons
- Wikipedia DE
- Dolma
- The Stack v2

### Drop as main inputs

- hidden merged German blob as final source of truth
- OpenWebMath as a major code component
- any source not explicitly listed in the curated plan

## Not implemented yet

These are still future work, not current repo reality:

- full DataTrove-backed executor graph
- GPU-heavy dedup / semantic dedup
- Qwen-teacher synthetic data pipeline
- automatic benchmark decontamination beyond the current baseline checker

Those can layer on top of this simpler deterministic pipeline later.
