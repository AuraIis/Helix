"""Download a byte-capped German slice of HPLT2.0_cleaned (deu_Latn).

HPLT2.0_cleaned is OPEN (no gating) and parquet-based, so the standard `datasets`
streaming loader works (unlike RedPajama-V2's script loader). It is more heavily
pre-cleaned than RedPajama, so it is our preferred "high-quality more-German"
source: we stream it, light-clean, and stop at a byte budget. Once edu-scored it
competes with the other sources by score, so better HPLT docs naturally displace
weaker docs from the kept pool.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target-gb", type=float, default=50.0)
    ap.add_argument("--skip-docs", type=int, default=0)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--progress-every", type=int, default=50_000)
    args = ap.parse_args()

    from datasets import load_dataset

    cfg = load_paths()
    filters = cfg["filters"]["german"]
    min_len = int(filters["min_length"])
    max_len = int(filters["max_length"])
    out = args.output or (resolve(cfg, "raw", "german") / "hplt_de.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    check_free_space(out.parent, args.target_gb + 5.0)

    print(f"streaming HPLT2.0_cleaned deu_Latn, skip={args.skip_docs:,}, "
          f"target={args.target_gb}GB -> {out}", flush=True)
    ds = load_dataset("HPLT/HPLT2.0_cleaned", name="deu_Latn", split="train", streaming=True)
    if args.skip_docs > 0:
        ds = ds.skip(args.skip_docs)

    target_bytes = int(args.target_gb * 1e9)
    written = 0
    docs = 0
    seen = 0
    dropped = {"too_short": 0, "too_long": 0, "empty": 0}
    with atomic_text_writer(out) as fh:
        for ex in ds:
            seen += 1
            text = ex.get("text") or ""
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
            written += len(line.encode("utf-8")) + 1
            docs += 1
            if args.progress_every and docs % args.progress_every == 0:
                print(f"  written {docs:,} docs / {written/1e9:.2f} GB (seen {seen:,})", flush=True)
            if written >= target_bytes:
                break

    manifest = {
        "input_dataset": "HPLT/HPLT2.0_cleaned:deu_Latn",
        "output_file": str(out),
        "skip_docs": args.skip_docs,
        "target_gb": args.target_gb,
        "stream_docs_seen": seen,
        "docs_written": docs,
        "bytes_written": written,
        "dropped": dropped,
        "min_length": min_len,
        "max_length": max_len,
        "finished_at": now_iso(),
    }
    out.with_suffix(out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"DONE docs={docs:,} bytes={written/1e9:.2f}GB", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
