#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
ROWS="${ROWS:-8192}"
DIM="${DIM:-1280}"
DTYPE="${DTYPE:-bf16}"

docker run --rm \
  --gpus all \
  --ipc=host \
  --name auralis-perf-lab-rmsnorm-smoke \
  -v "${REPO_ROOT}:/workspace/v2data" \
  -v /data/cache_perf_lab:/cache \
  -w /workspace/v2data \
  -e TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}" \
  -e MAX_JOBS="${MAX_JOBS:-4}" \
  -e AURALIS_PERF_BUILD_DIR=/workspace/v2data/perf_lab/.build/rmsnorm \
  auralis-perf-lab:cu130 \
  python perf_lab/benchmarks/bench_rmsnorm.py \
    --dtype "${DTYPE}" \
    --rows "${ROWS}" \
    --dim "${DIM}" \
    --warmup "${WARMUP:-3}" \
    --iters "${ITERS:-10}"

