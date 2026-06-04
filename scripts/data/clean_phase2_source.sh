#!/bin/bash
# Clean a single Phase-2 raw source: assemble + filter_quality + manifest.
#
# Per-source config table (language + filter args) lives at the top of this
# file. Adding a new source means: append the case to the dispatch.
#
# Output goes to /staging/cleaned/<source>.filtered.txt plus a .manifest.json.
#
# Usage (in the auralis-downloader container):
#   bash /staging/scripts/clean_phase2_source.sh fineweb_10bt
#   bash /staging/scripts/clean_phase2_source.sh smollm_python_edu
#   bash /staging/scripts/clean_phase2_source.sh the_stack_v2_python

set -euo pipefail

SOURCE="${1:-}"
if [ -z "$SOURCE" ]; then
    echo "Usage: $0 <source>" >&2
    echo "Available: fineweb_10bt smollm_python_edu the_stack_v2_python the_stack_v2_js_ts the_stack_v2_rust_go" >&2
    exit 1
fi

STAGING="${STAGING:-/staging}"
RAW_FILE="${STAGING}/raw/${SOURCE}/${SOURCE}.txt"
PRE_DIR="${STAGING}/cleaned/_pre_filter"
PRE_FILE="${PRE_DIR}/${SOURCE}.assembled.txt"
OUT_FILE="${STAGING}/cleaned/${SOURCE}.filtered.txt"
LOG_FILE="${STAGING}/logs/clean_${SOURCE}.log"
SCRIPTS="${STAGING}/scripts"

FILTER_PY="${SCRIPTS}/data/filter_quality.py"
ASSEMBLE_PY="${SCRIPTS}/data/assemble_for_filter.py"
if [ ! -f "$FILTER_PY" ] || [ ! -f "$ASSEMBLE_PY" ]; then
    echo "ERROR: cleaner scripts not found in ${SCRIPTS}/data/" >&2
    exit 1
fi

mkdir -p "$PRE_DIR" "$(dirname "$OUT_FILE")" "$(dirname "$LOG_FILE")"

# Per-source config: language + assemble-mode + filter-extra-args
case "$SOURCE" in
    fineweb_10bt)
        LANGUAGE=english
        MODE=text
        FILTER_EXTRA=""
        ;;
    smollm_python_edu)
        LANGUAGE=code
        MODE=code
        # python-edu is highly-curated synthetic content — relax repetition
        # threshold (formal code patterns repeat by design)
        FILTER_EXTRA="--max-repetition 0.85"
        ;;
    the_stack_v2_python|the_stack_v2_js_ts|the_stack_v2_rust_go)
        LANGUAGE=code
        MODE=code
        FILTER_EXTRA="--max-repetition 0.80"
        ;;
    fineweb2_de|wikipedia_de)
        LANGUAGE=german
        MODE=text
        FILTER_EXTRA=""
        ;;
    *)
        echo "Unknown source: $SOURCE" >&2
        exit 1
        ;;
esac

echo "=== clean ${SOURCE} ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ===" | tee -a "$LOG_FILE"
echo "  raw:    $RAW_FILE" | tee -a "$LOG_FILE"
echo "  pre:    $PRE_FILE" | tee -a "$LOG_FILE"
echo "  out:    $OUT_FILE" | tee -a "$LOG_FILE"
echo "  lang:   $LANGUAGE" | tee -a "$LOG_FILE"
echo "  mode:   $MODE" | tee -a "$LOG_FILE"
echo "  extra:  $FILTER_EXTRA" | tee -a "$LOG_FILE"

if [ ! -f "$RAW_FILE" ]; then
    echo "ERROR: raw file not found: $RAW_FILE" | tee -a "$LOG_FILE" >&2
    exit 2
fi

echo "--- stage 1: assemble ---" | tee -a "$LOG_FILE"
python "$ASSEMBLE_PY" --input "$RAW_FILE" --output "$PRE_FILE" --mode "$MODE" 2>&1 | tee -a "$LOG_FILE"

echo "--- stage 2: filter_quality ---" | tee -a "$LOG_FILE"
# shellcheck disable=SC2086  # we WANT word splitting on FILTER_EXTRA
python "$FILTER_PY" --input "$PRE_FILE" --output "$OUT_FILE" --language "$LANGUAGE" $FILTER_EXTRA 2>&1 | tee -a "$LOG_FILE"

echo "--- stage 3: cleanup pre-filter intermediate ---" | tee -a "$LOG_FILE"
# Save disk space — pre-filter file is reproducible, no need to keep
rm -f "$PRE_FILE" "${PRE_FILE}.manifest.json"

echo "=== done ${SOURCE} ($(date -u +%Y-%m-%dT%H:%M:%SZ)) ===" | tee -a "$LOG_FILE"
