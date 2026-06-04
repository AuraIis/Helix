#!/bin/bash
# Master chain for Phase-2 corpus prep:
#   wait for already-running download → clean → next download → clean → ...
#
# Runs detached in the auralis-downloader container. Safe to leave for hours.
# Logs progress to /staging/logs/master_chain.log
#
# Each stage is idempotent: if cleaned/<src>.filtered.txt already exists with
# matching manifest, the clean stage is skipped on re-run.

set -uo pipefail

LOG=/staging/logs/master_chain.log
exec >> "$LOG" 2>&1

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }

echo "============================================================"
echo "=== master_chain start: $(ts) ==="
echo "============================================================"

wait_for_dl() {
    local src="$1"
    while pgrep -f "[d]ownload_phase2_pretrain.py --source ${src}" > /dev/null; do
        sleep 60
    done
}

# --- DL helpers (Codex P4 + 3rd-pass P2-2) ----------------------------------
# Two flavours, both PID-aware:
#   start_dl_bg   — fork, store PID in DL_PIDS[$src], return immediately so
#                   the caller can run cleaning in parallel.
#   wait_dl_bg    — `wait` on the previously stored PID and surface its exit
#                   code. Pairs with start_dl_bg.
#   start_dl      — convenience: start_dl_bg + wait_dl_bg in one call.
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

start_dl() {
    start_dl_bg "$1" || return $?
    wait_dl_bg "$1"
}

# Verify a download produced non-empty output. Returns 0 only if the
# raw .txt exists, has size > 0, and the manifest exists. Used to gate
# the clean step (Codex P4).
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
    # Codex P4: don't clean if the upstream DL didn't actually produce data.
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

# Step 1: wait for the in-flight fineweb_10bt (already started outside the
# chain, so we use pgrep — no PID to wait on. Verify_dl_output below catches
# any incomplete output before we clean.)
echo ">> step 1: wait fineweb_10bt"
wait_for_dl fineweb_10bt
echo ">> fineweb_10bt finished at $(ts)"

# Step 2: clean fineweb (in parallel with starting smollm DL)
echo ">> step 2a: start smollm DL (background)"
start_dl_bg smollm_python_edu

echo ">> step 2b: clean fineweb_10bt (parallel to smollm DL)"
clean_one fineweb_10bt || echo "WARN: clean fineweb_10bt failed, continuing"

echo ">> step 3: wait smollm_python_edu"
wait_dl_bg smollm_python_edu || echo "WARN: smollm DL exited non-zero, downstream clean will be skipped"

# Step 4: clean smollm in parallel with stack_v2 DL
echo ">> step 4a: start the_stack_v2_python DL (background)"
start_dl_bg the_stack_v2_python

echo ">> step 4b: clean smollm_python_edu (parallel to stack_v2 DL)"
clean_one smollm_python_edu || echo "WARN: clean smollm_python_edu failed, continuing"

echo ">> step 5: wait the_stack_v2_python"
wait_dl_bg the_stack_v2_python || echo "WARN: stack_v2 DL exited non-zero, downstream clean will be skipped"

echo ">> step 6: clean the_stack_v2_python"
clean_one the_stack_v2_python || echo "WARN: clean the_stack_v2_python failed, continuing"

echo "============================================================"
echo "=== master_chain end: $(ts) ==="
echo "============================================================"

echo ">> SUMMARY"
echo "raw sizes:"
du -sh /staging/raw/* 2>/dev/null
echo "cleaned sizes:"
du -sh /staging/cleaned/* 2>/dev/null
echo
echo "manifests:"
for m in /staging/cleaned/*.manifest.json; do
    [ -f "$m" ] || continue
    echo "--- $m ---"
    cat "$m"
done
