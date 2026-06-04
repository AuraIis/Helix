#!/bin/bash
# Phase-2 DE chain: fineweb-2 deu_Latn -> wikipedia DE, with auto-clean overlap.
#
# Sequencing (≈3 h total):
#   1. Start fineweb2_de DL (largest, ≈70 min)
#   2. When done: clean fineweb2_de (≈40 min) + start wikipedia_de DL in parallel (≈10 min)
#   3. When wiki DL done: clean wikipedia_de (≈10 min)
#
# Runs detached in the auralis-downloader container.

set -uo pipefail

LOG=/staging/logs/chain_de.log
exec >> "$LOG" 2>&1

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "============================================================"
echo "=== chain_de start: $(ts) ==="
echo "============================================================"

# Codex 3rd-pass P2-2: PID-aware helper instead of pgrep-only wait.
declare -A DL_PIDS=()

start_dl_bg() {
    local src="$1"
    echo "--- DL ${src} START: $(ts) ---"
    export HF_TOKEN="$(cat /root/.hf_token)"
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export PHASE2_RAW_ROOT=/staging/raw
    nohup python /staging/scripts/data/download_phase2_pretrain.py --source "${src}" \
        > /staging/logs/dl_${src}.log 2>&1 &
    local pid=$!
    DL_PIDS[$src]=$pid
    echo "    ${src} PID ${pid}"
}

wait_dl_bg() {
    local src="$1"
    local pid="${DL_PIDS[$src]}"
    if [ -z "$pid" ]; then
        echo "--- DL ${src} WAIT FAILED: no PID recorded ---"
        return 99
    fi
    wait "$pid"
    local rc=$?
    unset 'DL_PIDS[$src]'
    if [ $rc -ne 0 ]; then
        echo "--- DL ${src} FAILED (rc=${rc}): $(ts) ---"
        return $rc
    fi
    echo "--- DL ${src} END: $(ts) ---"
    return 0
}

verify_dl_output() {
    local src="$1"
    local raw="/staging/raw/${src}/${src}.txt"
    local manifest="/staging/raw/${src}/manifest.json"
    if [ ! -s "$raw" ]; then
        echo "--- DL ${src} VERIFY FAILED: ${raw} missing or empty ---"
        return 2
    fi
    if [ ! -f "$manifest" ]; then
        echo "--- DL ${src} VERIFY FAILED: ${manifest} missing ---"
        return 3
    fi
    return 0
}

clean_one() {
    local src="$1"
    local out="/staging/cleaned/${src}.filtered.txt"
    if [ -f "${out}.manifest.json" ] && [ -s "$out" ]; then
        echo "--- CLEAN ${src} SKIP (already done): $(ts) ---"
        return 0
    fi
    if ! verify_dl_output "$src"; then
        echo "--- CLEAN ${src} SKIP (upstream DL output incomplete) ---"
        return 1
    fi
    echo "--- CLEAN ${src} START: $(ts) ---"
    if bash /staging/scripts/clean_phase2_source.sh "${src}"; then
        echo "--- CLEAN ${src} END: $(ts) ---"
    else
        echo "--- CLEAN ${src} FAILED (rc=$?): $(ts) ---"
        return 1
    fi
}

# Step 1: fineweb-2 DE DL (alone, max bandwidth)
echo ">> step 1: fineweb2_de DL"
mkdir -p /staging/raw/fineweb2_de
start_dl_bg fineweb2_de
wait_dl_bg fineweb2_de || { echo "FATAL: fineweb2_de DL failed, aborting chain"; exit 1; }
echo ">> fineweb2_de DL finished at $(ts)"

# Step 2: parallel — clean fineweb2_de + DL wikipedia_de
echo ">> step 2a: start wikipedia_de DL (background, small)"
mkdir -p /staging/raw/wikipedia_de
start_dl_bg wikipedia_de

echo ">> step 2b: clean fineweb2_de (parallel)"
clean_one fineweb2_de || echo "WARN: clean fineweb2_de failed"

echo ">> step 3: wait wikipedia_de"
wait_dl_bg wikipedia_de || echo "WARN: wikipedia_de DL exited non-zero, downstream clean will be skipped"
echo ">> wikipedia_de DL finished at $(ts)"

# Step 4: clean wikipedia_de
echo ">> step 4: clean wikipedia_de"
clean_one wikipedia_de || echo "WARN: clean wikipedia_de failed"

echo "============================================================"
echo "=== chain_de end: $(ts) ==="
echo "============================================================"

echo ">> SUMMARY"
echo "raw sizes:"
du -sh /staging/raw/fineweb2_de /staging/raw/wikipedia_de 2>/dev/null
echo "cleaned sizes:"
du -sh /staging/cleaned/fineweb2_de.filtered.txt /staging/cleaned/wikipedia_de.filtered.txt 2>/dev/null
echo
for m in /staging/cleaned/fineweb2_de.filtered.txt.manifest.json /staging/cleaned/wikipedia_de.filtered.txt.manifest.json; do
    [ -f "$m" ] || continue
    echo "--- $m ---"
    cat "$m"
done
