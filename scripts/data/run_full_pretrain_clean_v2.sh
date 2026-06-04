#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

mkdir -p data/training/pretrain_clean_v2 logs

log_step() {
  printf '\n[%s] %s\n' "$(date -Is)" "$*"
}

log_step "strict filter: german_commons"
python scripts/data/strict_filter_pretrain.py \
  --input cleaned/german_commons.filtered.txt \
  --output data/training/pretrain_clean_v2/german_commons.strict.txt \
  --language german

log_step "strict filter: german"
python scripts/data/strict_filter_pretrain.py \
  --input cleaned/german.txt \
  --output data/training/pretrain_clean_v2/german.strict.txt \
  --language german

log_step "strict filter: wikipedia_de"
python scripts/data/strict_filter_pretrain.py \
  --input cleaned/wikipedia_de.filtered.txt \
  --output data/training/pretrain_clean_v2/wikipedia_de.strict.txt \
  --language german

log_step "strict filter: openmath"
python scripts/data/strict_filter_pretrain.py \
  --input cleaned/openmath.filtered.txt \
  --output data/training/pretrain_clean_v2/openmath.strict.txt \
  --language english

log_step "generate 1M deterministic booster"
python scripts/data/generate_pretrain_booster.py \
  --documents 1000000 \
  --output data/training/pretrain_booster_de_v1m.txt

log_step "assemble full mix"
python scripts/data/build_pretrain_mix_v2.py \
  --output data/training/pretrain_clean_v2/mix_full.txt

log_step "done"
