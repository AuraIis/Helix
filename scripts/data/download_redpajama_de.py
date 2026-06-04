"""Download a byte-capped German slice of RedPajama-V2 (head_middle).

RPv2's HF dataset is script-based, which `datasets` 4.x no longer loads. So we go
direct: read the per-(lang,snapshot,partition) listing from the HF repo, then pull
the document shards (.json.gz) from Together's CDN, extract `raw_content`, apply the
same light clean as download_german.py, and stop at a byte budget. We never touch
the full 270TB — only the shards we consume up to --target-gb.

Listing : https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2/resolve/main/listings/de-<snap>-head_middle.txt
Shards  : https://data.together.xyz/redpajama-data-v2/v1.0.0/documents/<line>.json.gz
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data._common import (  # noqa: E402
    atomic_text_writer,
    check_free_space,
    clean_text,
    load_paths,
    now_iso,
    resolve,
)

LISTING_URL = ("https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2/"
               "resolve/main/listings/de-{snap}-head_middle.txt")
SHARD_URL = "https://data.together.xyz/redpajama-data-v2/v1.0.0/documents/{path}.json.gz"


def fetch(url: str, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "auralis-rpv2/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshots", nargs="+", default=["2023-14"],
                    help="CC snapshots to pull German head_middle from")
    ap.add_argument("--target-gb", type=float, default=50.0, help="cleaned-text byte budget")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--progress-every", type=int, default=50)
    args = ap.parse_args()

    cfg = load_paths()
    filters = cfg["filters"]["german"]
    min_len = int(filters["min_length"])
    max_len = int(filters["max_length"])
    out = args.output or (resolve(cfg, "raw", "german") / "redpajama_de.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    check_free_space(out.parent, args.target_gb + 5.0)

    target_bytes = int(args.target_gb * 1e9)
    written_bytes = 0
    docs = 0
    shards_done = 0
    shards_failed = 0
    dropped = {"too_short": 0, "too_long": 0, "empty": 0}
    t0 = time.monotonic()

    print(f"RPv2-de head_middle, snapshots={args.snapshots}, target={args.target_gb}GB -> {out}", flush=True)
    with atomic_text_writer(out) as fh:
        stop = False
        for snap in args.snapshots:
            if stop:
                break
            listing = fetch(LISTING_URL.format(snap=snap)).decode("utf-8", "replace").splitlines()
            print(f"  snapshot {snap}: {len(listing):,} shards", flush=True)
            for shard_path in listing:
                shard_path = shard_path.strip()
                if not shard_path:
                    continue
                try:
                    blob = fetch(SHARD_URL.format(path=shard_path))
                except Exception as e:  # noqa: BLE001
                    shards_failed += 1
                    if shards_failed <= 10:
                        print(f"  WARN shard fail {shard_path}: {type(e).__name__}", flush=True)
                    continue
                try:
                    with gzip.open(io.BytesIO(blob), "rt", encoding="utf-8", errors="replace") as gz:
                        for raw in gz:
                            try:
                                obj = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            text = obj.get("raw_content") or ""
                            if not text:
                                dropped["empty"] += 1
                                continue
                            if len(text) < min_len:
                                dropped["too_short"] += 1
                                continue
                            if len(text) > max_len:
                                dropped["too_long"] += 1
                                continue
                            line = clean_text(text).replace("\n", " ").replace("\r", " ")
                            fh.write(line + "\n")
                            written_bytes += len(line.encode("utf-8")) + 1
                            docs += 1
                except Exception as e:  # noqa: BLE001
                    shards_failed += 1
                    continue
                shards_done += 1
                if args.progress_every and shards_done % args.progress_every == 0:
                    rate = written_bytes / max(1e-9, time.monotonic() - t0) / 1e6
                    print(f"  shards {shards_done:,} | docs {docs:,} | "
                          f"{written_bytes/1e9:.2f} GB | {rate:.1f} MB/s", flush=True)
                if written_bytes >= target_bytes:
                    stop = True
                    break

    manifest = {
        "source": "RedPajama-Data-V2 (de, head_middle)",
        "snapshots": args.snapshots,
        "output_file": str(out),
        "target_gb": args.target_gb,
        "shards_downloaded": shards_done,
        "shards_failed": shards_failed,
        "docs_written": docs,
        "bytes_written": written_bytes,
        "dropped": dropped,
        "min_length": min_len,
        "max_length": max_len,
        "finished_at": now_iso(),
    }
    out.with_suffix(out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"DONE docs={docs:,} bytes={written_bytes/1e9:.2f}GB shards={shards_done:,} failed={shards_failed}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
