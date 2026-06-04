"""Download a SECOND, non-overlapping fineweb2-de slice.

The first slice (raw/german/fineweb2_de.txt, ~4.1M docs / 12.5GB) was taken from
the START of the HuggingFaceFW/fineweb-2 `deu_Latn` stream. To grow the unique
German pool without overlap, this skips past the already-consumed docs and writes
the NEXT chunk, applying the same light download-time clean as download_german.py.
The heavy structure-clean + edu-scoring happen as separate downstream steps.
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
    ap.add_argument("--skip-docs", type=int, default=5_000_000,
                    help="raw stream docs to skip (must exceed what slice 1 consumed) to avoid overlap")
    ap.add_argument("--target-gb", type=float, default=30.0, help="bytes of cleaned text to write")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--progress-every", type=int, default=50_000)
    args = ap.parse_args()

    from datasets import load_dataset

    cfg = load_paths()
    filters = cfg["filters"]["german"]
    min_len = int(filters["min_length"])
    max_len = int(filters["max_length"])
    out = args.output or (resolve(cfg, "raw", "german") / "fineweb2_de_v2.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    check_free_space(out.parent, args.target_gb + 5.0)

    print(f"streaming HuggingFaceFW/fineweb-2 deu_Latn, skip={args.skip_docs:,}, "
          f"target={args.target_gb}GB -> {out}", flush=True)
    ds = load_dataset("HuggingFaceFW/fineweb-2", name="deu_Latn", split="train", streaming=True)
    if args.skip_docs > 0:
        ds = ds.skip(args.skip_docs)

    target_bytes = int(args.target_gb * 1e9)
    written_bytes = 0
    docs = 0
    seen = 0
    dropped = {"too_short": 0, "too_long": 0}
    with atomic_text_writer(out) as fh:
        for ex in ds:
            seen += 1
            text = ex.get("text", "") or ""
            if len(text) < min_len:
                dropped["too_short"] += 1
                continue
            if len(text) > max_len:
                dropped["too_long"] += 1
                continue
            line = clean_text(text)
            fh.write(line + "\n")
            written_bytes += len(line.encode("utf-8")) + 1
            docs += 1
            if args.progress_every and docs % args.progress_every == 0:
                print(f"  written {docs:,} docs / {written_bytes/1e9:.2f} GB "
                      f"(stream seen {seen:,})", flush=True)
            if written_bytes >= target_bytes:
                break

    manifest = {
        "input_dataset": "HuggingFaceFW/fineweb-2:deu_Latn",
        "output_file": str(out),
        "skip_docs": args.skip_docs,
        "target_gb": args.target_gb,
        "stream_docs_seen": seen,
        "docs_written": docs,
        "bytes_written": written_bytes,
        "dropped": dropped,
        "min_length": min_len,
        "max_length": max_len,
        "finished_at": now_iso(),
    }
    mpath = out.with_suffix(out.suffix + ".manifest.json")
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"DONE docs={docs:,} bytes={written_bytes/1e9:.2f}GB skipped={args.skip_docs:,}", flush=True)
    print(f"manifest: {mpath}", flush=True)
    # Hard exit: skip interpreter finalization so HF streaming's background
    # prefetch threads don't crash on teardown after the early break.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
