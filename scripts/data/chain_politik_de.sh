#!/bin/bash
# Politik-DE chain: pull every public source we have a parser for, then
# clean each into the standard Auralis filtered.txt format.
#
# Order is by size (smallest first). bundestag_mdb is already done from
# the smoke test; chain skips it if the manifest exists.
#
# Runs detached in auralis-downloader. Logs to /staging/politik_de/logs/.

set -uo pipefail

LOG=/staging/politik_de/logs/chain_politik.log
exec >> "$LOG" 2>&1

ts() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "============================================================"
echo "=== chain_politik start: $(ts) ==="
echo "============================================================"

run_dl() {
    local src="$1"
    local manifest="/staging/politik_de/raw/${src}/manifest.json"
    if [ -f "$manifest" ]; then
        echo "--- DL ${src} SKIP (manifest exists): $(ts) ---"
        return 0
    fi
    echo "--- DL ${src} START: $(ts) ---"
    export POLITIK_RAW_ROOT=/staging/politik_de/raw
    if python /staging/politik_de/download_politik_de.py --source "${src}" \
        > /staging/politik_de/logs/dl_${src}.log 2>&1; then
        echo "--- DL ${src} END: $(ts) ---"
    else
        echo "--- DL ${src} FAILED (rc=$?): $(ts) ---"
        return 1
    fi
}

# Order: smallest/cheapest first
echo ">> 1. bundestag_mdb (4.6k politicians, ~1 min)"
run_dl bundestag_mdb || true

echo ">> 2. europarl_meps (~720 MEPs, <1 min)"
run_dl europarl_meps || true

echo ">> 3. lobbyregister_de (~5k entries, ~5 min)"
run_dl lobbyregister_de || true

echo ">> 4. abgeordnetenwatch (5k Q&A, ~5 min)"
run_dl abgeordnetenwatch || true

echo ">> 5. bundestag_protokolle (200 protocols, ~30 min — biggest)"
run_dl bundestag_protokolle || true

echo "============================================================"
echo "=== chain_politik end: $(ts) ==="
echo "============================================================"

echo ">> SUMMARY"
du -sh /staging/politik_de/raw/* 2>/dev/null
echo
for m in /staging/politik_de/raw/*/manifest.json; do
    [ -f "$m" ] || continue
    echo "--- $m ---"
    cat "$m"
done
