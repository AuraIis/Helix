#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --name auralis-perf-lab \
  -v "${REPO_ROOT}:/workspace/v2data" \
  -v /data/cache_perf_lab:/cache \
  -w /workspace/v2data \
  auralis-perf-lab:cu130 \
  bash

