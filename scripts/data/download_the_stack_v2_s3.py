"""Download the_stack_v2 with Software Heritage S3 content-hydration.

Background:
    bigcode/the-stack-v2 stores **pointers** (blob_id) to source files in
    parquet metadata. The actual content lives in the SoftwareHeritage S3
    bucket (anonymous read access, no AWS account needed) compressed with
    gzip. This script:

    1. Streams metadata records from the_stack_v2 parquets, per language.
    2. For each record, fetches gzip-content from S3 in parallel
       (ThreadPoolExecutor) and decodes with the row's src_encoding.
    3. Filters by min_stars / length / vendor-or-generated.
    4. Writes one document per line to ``raw/code/the_stack_v2.txt``,
       wrapped with ``<|code|>[lang]\\n...\\n<|endcode|>`` like the
       existing ``download_code.py`` does for starcoderdata.

Why this is its own script (vs. extending download_code.py):
    The S3-hydration loop is fundamentally different from a streaming
    text iterator (single-threaded), and ThreadPoolExecutor is needed
    to make the wall-clock acceptable. Keeping it separate avoids
    polluting the existing single-source loop.

Usage (inside the container):
    python scripts/data/download_the_stack_v2_s3.py \\
        --config configs/data_paths_phase2.yaml \\
        --target-tokens 2_000_000_000 \\
        --workers 16
"""
from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from pathlib import Path
from typing import Any, Iterator

import boto3
import smart_open
from botocore import UNSIGNED
from botocore.client import Config
from datasets import load_dataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.data._common import (  # noqa: E402
    DownloadStats,
    atomic_text_writer,
    check_free_space,
    load_paths,
    now_iso,
    resolve,
)

CODE_BYTES_PER_TOKEN = 3.5

# the_stack_v2 layout: data/{LinguistName}/train-*.parquet
LANG_FOLDERS = {
    "python":     "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "rust":       "Rust",
    "cpp":        "C++",
    "go":         "Go",
    "java":       "Java",
    "shell":      "Shell",
    "sql":        "SQL",
}

# Per-language token-budget shares. Same shares as download_code.py
# starcoderdata to keep the 2 sources comparable.
LANG_SHARES: dict[str, float] = {
    "python":     0.30,
    "javascript": 0.20,
    "typescript": 0.10,
    "rust":       0.10,
    "cpp":        0.10,
    "go":         0.08,
    "java":       0.07,
    "shell":      0.03,
    "sql":        0.02,
}


def make_s3_client():
    """Anonymous S3 client for the softwareheritage bucket (no AWS account)."""
    return boto3.client("s3", config=Config(signature_version=UNSIGNED))


def fetch_content(s3_client, blob_id: str, src_encoding: str | None) -> str | None:
    """Fetch gzip-stored file content from softwareheritage S3.

    Returns None on any error (missing blob, decode failure, network glitch).
    Caller filters None out without aborting the pipeline.
    """
    try:
        s3_url = f"s3://softwareheritage/content/{blob_id}"
        with smart_open.open(
            s3_url,
            "rb",
            compression=".gz",
            transport_params={"client": s3_client},
        ) as fin:
            raw = fin.read()
        return raw.decode(src_encoding or "utf-8", errors="replace")
    except Exception:
        return None


def stream_language(lang_folder: str, min_stars: int) -> Iterator[dict[str, Any]]:
    """Stream filtered metadata records for one language sub-folder."""
    ds = load_dataset(
        "bigcode/the-stack-v2",
        data_dir=f"data/{lang_folder}",
        split="train",
        streaming=True,
    )
    for ex in ds:
        if not ex.get("blob_id"):
            continue
        stars = (
            ex.get("star_events_count", 0)
            or ex.get("max_stars_count", 0)
            or 0
        )
        if stars < min_stars:
            continue
        if ex.get("is_vendor", False) or ex.get("is_generated", False):
            continue
        yield ex


def download_lang(
    lang: str,
    lang_folder: str,
    target_bytes: int,
    out_writer,
    s3_client,
    filters: dict[str, Any],
    workers: int,
) -> dict[str, int]:
    """Pull files for one language until target_bytes written.

    Pipeline keeps ``workers * 4`` futures in flight at any time, so the
    S3-fetch latency overlaps with metadata streaming and disk writes.
    """
    min_len = filters["min_length"]
    max_len = filters["max_length"]
    min_stars = filters["min_stars"]

    bytes_written = 0
    files_written = 0
    files_filtered = 0
    blob_fetch_failed = 0

    pbar = tqdm(desc=lang_folder, unit=" files", total=None)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        meta_iter = stream_language(lang_folder, min_stars)
        in_flight: deque = deque()

        def submit_one() -> bool:
            try:
                rec = next(meta_iter)
            except StopIteration:
                return False
            future = pool.submit(
                fetch_content,
                s3_client,
                rec["blob_id"],
                rec.get("src_encoding"),
            )
            in_flight.append((future, rec))
            return True

        # Prime the pipeline.
        prime_count = workers * 4
        for _ in range(prime_count):
            if not submit_one():
                break

        while in_flight and bytes_written < target_bytes:
            future, rec = in_flight.popleft()
            content = future.result()
            # Keep pipeline full unless source exhausted.
            submit_one()

            if content is None:
                blob_fetch_failed += 1
                continue
            if len(content) < min_len or len(content) > max_len:
                files_filtered += 1
                continue

            wrapped = (
                f"<|code|>[{lang}]\n{content}\n<|endcode|>".replace("\r\n", "\n")
            )
            out_writer.write(wrapped + "\n")
            sz = len(wrapped.encode("utf-8")) + 1
            bytes_written += sz
            files_written += 1

            if files_written % 50 == 0:
                pbar.update(50)
                pbar.set_postfix(
                    MB=f"{bytes_written/1e6:.0f}",
                    pct=f"{100 * bytes_written / max(target_bytes,1):.1f}",
                    miss=blob_fetch_failed,
                    filt=files_filtered,
                )

        # Cancel anything still in flight (we hit the byte budget).
        for future, _ in in_flight:
            future.cancel()
        pbar.close()

    return {
        "bytes": bytes_written,
        "files": files_written,
        "filtered": files_filtered,
        "fetch_failed": blob_fetch_failed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--target-tokens",
        type=lambda s: int(s.replace("_", "")),
        default=2_000_000_000,
        help="Total token budget across all languages (default 2B).",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--required-free-gb", type=float, default=10.0)
    parser.add_argument(
        "--output-name",
        default="the_stack_v2.txt",
        help="Output filename inside raw/code (default: the_stack_v2.txt).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=sorted(LANG_FOLDERS),
        default=list(LANG_FOLDERS),
        help="Subset of languages to download, useful for smoke tests.",
    )
    args = parser.parse_args()

    cfg = load_paths(args.config) if args.config else load_paths()
    out_dir = resolve(cfg, "raw", "code")
    out_dir.mkdir(parents=True, exist_ok=True)
    check_free_space(out_dir, args.required_free_gb)

    output_path = out_dir / args.output_name
    target_bytes = int(args.target_tokens * CODE_BYTES_PER_TOKEN)
    filters = cfg["filters"]["code"]
    s3_client = make_s3_client()

    stats = DownloadStats(
        source="the_stack_v2_s3",
        output_file=str(output_path),
        target_tokens=args.target_tokens,
        estimated_bytes_per_token=CODE_BYTES_PER_TOKEN,
        started_at=now_iso(),
        filters_applied={
            **filters,
            "lang_shares": LANG_SHARES,
            "workers": args.workers,
            "languages": args.languages,
        },
    )

    print(f"Output:  {output_path}")
    print(f"Target:  {target_bytes/1e9:.2f} GB ({args.target_tokens/1e9:.2f}B tokens)")
    print(f"Workers: {args.workers}", flush=True)

    with atomic_text_writer(output_path) as fh:
        selected_share_total = sum(LANG_SHARES[lang] for lang in args.languages)
        for lang in args.languages:
            share = LANG_SHARES[lang] / selected_share_total
            lang_folder = LANG_FOLDERS[lang]
            lang_target_bytes = int(target_bytes * share)
            print(
                f"\n=== {lang} ({lang_folder}) → {lang_target_bytes/1e6:.0f} MB target ===",
                flush=True,
            )
            result = download_lang(
                lang=lang,
                lang_folder=lang_folder,
                target_bytes=lang_target_bytes,
                out_writer=fh,
                s3_client=s3_client,
                filters=filters,
                workers=args.workers,
            )
            stats.final_bytes += result["bytes"]
            stats.final_docs += result["files"]
            stats.filtered_total += result["filtered"]
            for k, v in result.items():
                if k in ("filtered", "fetch_failed"):
                    stats.filtered_reasons[f"{lang}:{k}"] = v
            print(
                f"  {lang}: {result['files']:>7,} files | "
                f"{result['bytes']/1e6:>7.0f} MB | "
                f"filt={result['filtered']:,} miss={result['fetch_failed']:,}",
                flush=True,
            )

    stats.finished_at = now_iso()
    stats.write_manifest(output_path.with_suffix(".txt.manifest.json"))

    print()
    print(f"=== DONE ===")
    print(f"  output: {output_path}")
    print(f"  size:   {stats.final_bytes/1e9:.2f} GB")
    print(f"  files:  {stats.final_docs:,}")


if __name__ == "__main__":
    main()
