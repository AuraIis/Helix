#!/bin/bash
# Wrapper that reads HF token from file and launches a download in the
# auralis-downloader container with PHASE2_RAW_ROOT pointing to /staging/raw
# (the disk7 bind-mount).
#
# Usage (on host):
#   /mnt/disk7/Auralis/phase2_corpus/scripts/start_phase2_dl.sh <source>
# where <source> is one of:
#   fineweb_10bt | smollm_python_edu | the_stack_v2_python
#   the_stack_v2_js_ts | the_stack_v2_rust_go

set -euo pipefail

SOURCE="${1:-}"
if [ -z "$SOURCE" ]; then
    echo "Usage: $0 <source>"
    exit 1
fi

CONTAINER="auralis-downloader"
LOG="/staging/logs/dl_${SOURCE}.log"

# Token comes from a file inside the container (mounted/copied separately,
# never on the command line so it does not appear in process args).
docker exec -d "$CONTAINER" bash -c "
  export HF_TOKEN=\"\$(cat /root/.hf_token)\"
  export HUGGING_FACE_HUB_TOKEN=\"\$HF_TOKEN\"
  export PHASE2_RAW_ROOT=/staging/raw
  cd /staging
  exec python /staging/scripts/data/download_phase2_pretrain.py --source ${SOURCE} > ${LOG} 2>&1
"

echo "Started ${SOURCE} -> /mnt/disk7/Auralis/phase2_corpus/logs/dl_${SOURCE}.log"
