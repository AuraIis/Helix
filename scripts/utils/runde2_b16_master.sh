#!/usr/bin/env bash
# Canary Runde 2 — batch=16 ablation runner.
#
# Runs baseline_b16 -> de_heavy_b16 -> code_heavy_b16 sequentially with the
# CUDA-kernels enabled. Stops on the first failure.
set -u

REPO=/workspace/v2data
cd "$REPO" || { echo "FATAL: cannot cd $REPO"; exit 1; }

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export AURALIS_USE_CUDA_KERNELS=1
export TRITON_OVERRIDE_ARCH=sm89

LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/runde2_b16_master.log"
TRAIN="$REPO/scripts/pretrain/train_phase1.py"
VARIANTS=(baseline de_heavy code_heavy)
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "=== RUNDE 2 b16 MASTER START $(ts) ===" | tee -a "$MASTER_LOG"
for variant in "${VARIANTS[@]}"; do
    config="$REPO/configs/training/canary_runde2_${variant}_b16.yaml"
    var_log="$LOG_DIR/runde2_${variant}_b16.log"
    echo "=== [$variant] START $(ts) ===" | tee -a "$MASTER_LOG"
    if [[ ! -f "$config" ]]; then
        echo "  ABORT: config missing" | tee -a "$MASTER_LOG"; exit 2
    fi
    # Idempotent skip if step_5000 ckpt already there.
    if [[ -f "$REPO/checkpoints/canary_runde2_${variant}_b16/step_5000.pt" ]]; then
        echo "  SKIP: already finished" | tee -a "$MASTER_LOG"
        echo "=== [$variant] DONE skip $(ts) ===" | tee -a "$MASTER_LOG"
        continue
    fi
    python "$TRAIN" --config "$config" >"$var_log" 2>&1
    rc=$?
    echo "=== [$variant] DONE exit=$rc $(ts) ===" | tee -a "$MASTER_LOG"
    if [[ $rc -ne 0 ]]; then
        echo "  ABORT: $variant failed (rc=$rc), see $var_log" | tee -a "$MASTER_LOG"
        exit $rc
    fi
done
echo "=== ALL DONE $(ts) ===" | tee -a "$MASTER_LOG"
