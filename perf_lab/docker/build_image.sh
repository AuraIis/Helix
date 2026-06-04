#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."
docker build \
  -f perf_lab/docker/Dockerfile \
  --build-arg INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}" \
  --build-arg MAX_JOBS="${MAX_JOBS:-4}" \
  -t auralis-perf-lab:cu130 \
  .
