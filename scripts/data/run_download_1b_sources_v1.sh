#!/usr/bin/env sh
set -eu

ROOT="/disk5v2data/data/pretrain_1b_sources_v1"
LOG_DIR="/disk5v2data/logs"
LOG="$LOG_DIR/download_1b_sources_v1.log"
CFG="/workspace/v2data/configs/data_paths_1b_samples_container.yaml"

mkdir -p "$ROOT" "$LOG_DIR" "$ROOT/_hf_cache"
export HF_HOME="$ROOT/_hf_cache/home"
export HF_DATASETS_CACHE="$ROOT/_hf_cache/datasets"
export HF_HUB_CACHE="$ROOT/_hf_cache/hub"
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTHONUNBUFFERED=1

{
  echo "=== download_1b_sources_v1 started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  echo "ROOT=$ROOT"
  echo "HF_HOME=$HF_HOME"
  echo
} >> "$LOG"

run_step() {
  name="$1"
  shift
  {
    echo
    echo "=== [$name] started $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    "$@"
    echo "=== [$name] finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  } >> "$LOG" 2>&1
}

run_step "fineweb_edu+dclm_edu" \
  python /workspace/v2data/scripts/data/download_english.py \
    --config "$CFG" \
    --sources fineweb_edu dclm_edu \
    --required-free-gb 100

run_step "fineweb2_de" \
  python /workspace/v2data/scripts/data/download_german.py \
    --config "$CFG" \
    --sources fineweb2_de \
    --target-tokens-override fineweb2_de=6_000_000_000 \
    --required-free-gb 100

{
  echo
  echo "=== download_1b_sources_v1 finished $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
  find "$ROOT/raw" -maxdepth 3 -type f -printf "%TY-%Tm-%Td %TH:%TM %s %p\n" 2>/dev/null | sort
} >> "$LOG" 2>&1
