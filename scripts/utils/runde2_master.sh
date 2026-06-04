#!/usr/bin/env bash
# Canary Runde 2 — sequential master runner.
#
# Runs baseline -> de_heavy -> code_heavy and stops on the first failure.
# Forces cwd to /workspace/v2data/ so configs and code resolve to the
# canonical AuralisV2 source-of-truth (not the stale /workspace/auralis_v2
# v1 copy which caused the earlier abort).
set -u

REPO=/workspace/v2data
cd "$REPO" || { echo "FATAL: cannot cd $REPO"; exit 1; }

LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/runde2_master.log"
export PYTHONUNBUFFERED=1
export AURALIS_USE_CUDA_KERNELS=1
export TRITON_OVERRIDE_ARCH=sm89
TRAIN="$REPO/scripts/pretrain/train_phase1.py"
DASHBOARD="$REPO/scripts/eval/regression_dashboard.py"
VARIANTS=(baseline de_heavy code_heavy)
ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "=== RUNDE 2 MASTER START $(ts) ===" | tee -a "$MASTER_LOG"
for variant in "${VARIANTS[@]}"; do
    config="$REPO/configs/training/canary_runde2_${variant}.yaml"
    var_log="$LOG_DIR/runde2_${variant}.log"
    echo "=== [$variant] START $(ts) ===" | tee -a "$MASTER_LOG"
    if [[ ! -f "$config" ]]; then
        echo "  ABORT: config $config not found" | tee -a "$MASTER_LOG"
        exit 2
    fi
    if [[ "$variant" == "baseline" ]]        && [[ -f "$REPO/checkpoints/canary_runde2_baseline/step_5000.pt" ]]        && grep -q 'eval @ step 5000' "$LOG_DIR/runde2_baseline.log" 2>/dev/null; then
        echo "  SKIP: baseline already finished" | tee -a "$MASTER_LOG"
        echo "=== [$variant] DONE exit=skip $(ts) ===" | tee -a "$MASTER_LOG"
        continue
    fi
    python "$TRAIN" --config "$config" >"$var_log" 2>&1
    rc=$?
    echo "=== [$variant] DONE exit=$rc $(ts) ===" | tee -a "$MASTER_LOG"
    if [[ $rc -ne 0 ]]; then
        echo "  ABORT: variant $variant failed (rc=$rc)" | tee -a "$MASTER_LOG"
        echo "  see $var_log" | tee -a "$MASTER_LOG"
        exit $rc
    fi
done
echo "=== regression dashboards $(ts) ===" | tee -a "$MASTER_LOG"
if [[ -f "$DASHBOARD" ]]; then
    python "$DASHBOARD" --runs canary_runde2_baseline canary_runde2_de_heavy canary_runde2_code_heavy >>"$LOG_DIR/runde2_dashboard.log" 2>&1
    echo "  dashboard exit=$?" | tee -a "$MASTER_LOG"
fi
echo "=== ALL DONE $(ts) ===" | tee -a "$MASTER_LOG"
