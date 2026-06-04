#!/usr/bin/env python3
"""Download selected coral-nlp/german-commons source splits to local disk.

The script writes one compressed JSONL file per source split. It is deliberately
sequential and manifest-driven so long downloads can be monitored and completed
files are not downloaded again on restart.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from pathlib import Path
from typing import Any


DATASET_ID = "coral-nlp/german-commons"


def configure_hf_cache(output_root: Path) -> None:
    cache = output_root / "_hf_cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache / "home"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(cache / "datasets"))
    os.environ.setdefault("HF_HUB_CACHE", str(cache / "hub"))


def load_targets(plan_path: Path, include_special: bool) -> list[dict[str, Any]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    targets = list(plan.get("take_first", []))
    if include_special:
        targets.extend(plan.get("small_specialty", []))
    return targets


def safe_slug(config: str, split: str) -> str:
    return f"{config}__{split}".replace("/", "__").replace("\\", "__")


def row_payload(row: dict[str, Any], config: str, split: str) -> dict[str, Any]:
    return {
        "config": config,
        "split": split,
        "id": row.get("id"),
        "source": row.get("source"),
        "subset": row.get("subset"),
        "text": row.get("text") or "",
        "license": row.get("license"),
        "num_tokens": row.get("num_tokens"),
        "perplexity": row.get("perplexity"),
        "ocr_score": row.get("ocr_score"),
    }


def download_split(
    *,
    config: str,
    split: str,
    output_root: Path,
    max_docs: int,
    log_every: int,
) -> dict[str, Any]:
    from datasets import load_dataset

    slug = safe_slug(config, split)
    out_dir = output_root / config
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / f"{slug}.jsonl.gz"
    tmp_path = out_dir / f"{slug}.jsonl.gz.tmp"
    manifest_path = out_dir / f"{slug}.manifest.json"

    if final_path.exists() and manifest_path.exists():
        print(f"[skip] {config}/{split} already exists: {final_path}", flush=True)
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    if tmp_path.exists():
        tmp_path.unlink()

    print(f"[start] {config}/{split} -> {final_path}", flush=True)
    t0 = time.time()
    docs = 0
    bytes_text = 0
    tokens = 0
    skipped_empty = 0

    stream = load_dataset(DATASET_ID, name=config, split=split, streaming=True)
    with gzip.open(tmp_path, "wt", encoding="utf-8", newline="\n", compresslevel=5) as fh:
        for row in stream:
            payload = row_payload(row, config, split)
            text = payload["text"]
            if not text.strip():
                skipped_empty += 1
                continue
            fh.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            docs += 1
            bytes_text += len(text.encode("utf-8", errors="ignore"))
            if isinstance(payload.get("num_tokens"), int):
                tokens += int(payload["num_tokens"])
            if docs % log_every == 0:
                elapsed = max(1.0, time.time() - t0)
                mb = tmp_path.stat().st_size / (1024 * 1024)
                print(
                    f"[progress] {config}/{split} docs={docs:,} gz_mb={mb:.1f} docs_s={docs / elapsed:.1f}",
                    flush=True,
                )
            if max_docs and docs >= max_docs:
                break

    tmp_path.replace(final_path)
    elapsed = time.time() - t0
    manifest = {
        "dataset": DATASET_ID,
        "config": config,
        "split": split,
        "path": str(final_path),
        "docs": docs,
        "tokens_from_metadata": tokens,
        "text_bytes": bytes_text,
        "gzip_bytes": final_path.stat().st_size,
        "skipped_empty": skipped_empty,
        "max_docs": max_docs,
        "elapsed_seconds": round(elapsed, 2),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[done] {config}/{split} docs={docs:,} gzip_gb={manifest['gzip_bytes'] / 1e9:.3f} elapsed_s={elapsed:.1f}",
        flush=True,
    )
    return manifest


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    configure_hf_cache(output_root)
    targets = load_targets(args.plan, args.include_special)
    manifests = []
    for target in targets:
        manifests.append(
            download_split(
                config=target["config"],
                split=target["split"],
                output_root=output_root,
                max_docs=args.max_docs_per_split,
                log_every=args.log_every,
            )
        )
    summary = {
        "dataset": DATASET_ID,
        "output_root": str(output_root),
        "targets": len(targets),
        "docs": sum(item.get("docs", 0) for item in manifests),
        "tokens_from_metadata": sum(item.get("tokens_from_metadata", 0) for item in manifests),
        "gzip_bytes": sum(item.get("gzip_bytes", 0) for item in manifests),
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "splits": manifests,
    }
    (output_root / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[summary] docs={summary['docs']:,} gzip_gb={summary['gzip_bytes'] / 1e9:.3f}", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=Path("configs/data/german_commons_clean_plan_v1.json"))
    parser.add_argument("--output-root", type=Path, default=Path("I:/KI/Auralis_datasets/german_commons_selected_raw"))
    parser.add_argument("--include-special", action="store_true")
    parser.add_argument("--max-docs-per-split", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10_000)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
