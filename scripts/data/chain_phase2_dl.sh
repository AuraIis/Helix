#!/bin/bash
# Chain-Downloader: wartet bis fineweb_10bt fertig ist, dann startet smollm und
# anschließend the_stack_v2_python. Soll im Container als detached job laufen.
#
# Nutzt L-016-trick: [d]ownload statt download in pgrep, damit das Wait-Loop
# nicht sich selbst matcht.

set -uo pipefail

LOG=/staging/logs/chain.log
exec >> "$LOG" 2>&1

echo "=== chain start: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

wait_for() {
    local src="$1"
    while pgrep -f "[d]ownload_phase2_pretrain.py --source ${src}" > /dev/null; do
        sleep 60
    done
}

run() {
    local src="$1"
    echo "=== ${src} START: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    export HF_TOKEN="$(cat /root/.hf_token)"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export PHASE2_RAW_ROOT=/staging/raw
    python /staging/scripts/data/download_phase2_pretrain.py --source "${src}" \
        > /staging/logs/dl_${src}.log 2>&1
    local rc=$?
    echo "=== ${src} END (rc=${rc}): $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
    return $rc
}

# Step 1: wait for already-running fineweb
echo ">> waiting for fineweb_10bt..."
wait_for fineweb_10bt
echo ">> fineweb_10bt finished"

# Step 2: smollm
run smollm_python_edu || echo ">> smollm failed, continuing anyway"

# Step 3: the_stack_v2_python
run the_stack_v2_python || echo ">> the_stack_v2_python failed"

echo "=== chain end: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
