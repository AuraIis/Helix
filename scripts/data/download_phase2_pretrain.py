"""Phase-2 Pretrain Korpus Downloads.

Lädt drei Pflicht-Datasets nach /mnt/disk7/Auralis/phase2_corpus/raw/:
- bigcode/the-stack-v2-dedup [Python subset]   → ersetzt fehlende the_stack_v2-Quelle
- HuggingFaceTB/smollm-corpus [python-edu]     → synthetisches Edu-Python (high quality)
- HuggingFaceFW/fineweb [sample-10BT]          → ersetzt fehlende 8B-EN-Web-Tokens

Jeder Download läuft als eigener Prozess. Resume-safe via huggingface_hub-Cache.
Output: einzelne .txt-Files (one document per line, geführt durch Newlines)
        plus manifest.json mit document_count + bytes.

Usage:
    python download_phase2_pretrain.py --source the_stack_v2_python
    python download_phase2_pretrain.py --source smollm_python_edu
    python download_phase2_pretrain.py --source fineweb_10bt

Run alle drei parallel im background:
    nohup python download_phase2_pretrain.py --source the_stack_v2_python > logs/dl_stack_py.log 2>&1 &
    nohup python download_phase2_pretrain.py --source smollm_python_edu  > logs/dl_smollm.log   2>&1 &
    nohup python download_phase2_pretrain.py --source fineweb_10bt       > logs/dl_fineweb.log  2>&1 &
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


# --- ROOT: where the downloads land. -----------------------------------------
# The container runs with /staging mounted to disk7, the host sees /mnt/disk7
# directly. Either way we MUST resolve to a real bind-mount — never to a path
# that only exists in the container overlay-fs.
#
# Codex 3rd-pass P2-1: the chain scripts set PHASE2_RAW_ROOT=/staging/raw
# themselves, which previously bypassed the auto-detect mount-check. Fix:
# the mount-check applies regardless of how the path was chosen — any path
# under /staging requires /staging to be a real mount.
_explicit = os.environ.get("PHASE2_RAW_ROOT")
if _explicit:
    ROOT = Path(_explicit)
elif Path("/staging").is_mount():
    ROOT = Path("/staging/raw")
else:
    ROOT = Path("/mnt/disk7/Auralis/phase2_corpus/raw")


def _must_be_mounted(p: Path, anchor_dir: str) -> None:
    """Ensure that ``anchor_dir`` is a real mount-point if ``p`` lives under
    it. Empty anchor_dir or a path that does not match means: nothing to
    enforce here (other guards apply).
    """
    if not anchor_dir:
        return
    anchor = Path(anchor_dir)
    try:
        p.resolve().relative_to(anchor.resolve())
    except ValueError:
        return  # path is not under anchor_dir
    if not anchor.is_mount():
        raise SystemExit(
            f"FATAL: refusing to use {p} because {anchor_dir} is not a real "
            f"mount-point. The path may resolve into the container overlay-fs "
            f"and silently consume container disk. Either bind-mount disk7 "
            f"to {anchor_dir}, or set PHASE2_RAW_ROOT to a path on a real mount."
        )


# Always enforce the /staging mount-check, even when the operator passed the
# path explicitly via PHASE2_RAW_ROOT — explicit doesn't mean "ignore safety".
_must_be_mounted(ROOT, "/staging")
if not ROOT.parent.exists():
    raise SystemExit(
        f"FATAL: ROOT.parent does not exist: {ROOT.parent} — refusing to write to "
        f"a path that may resolve into the container overlay. Set PHASE2_RAW_ROOT "
        f"to a path on a real mount, or bind-mount disk7 to /staging."
    )

# --- HF-Cache: derived from ROOT, never hard-coded. --------------------------
# Previous version pinned the cache to /mnt/disk7/... and called os.makedirs()
# at import time. Inside the container that path is typically not a mount,
# so the cache silently materialised in the overlay-fs and ate container disk
# (Codex P2 2nd-pass). Fix: derive cache path from the validated ROOT (which
# IS guaranteed to live on a real mount), or honour PHASE2_CACHE_ROOT if set.
_cache_root_env = os.environ.get("PHASE2_CACHE_ROOT")
if _cache_root_env:
    CACHE_ROOT = Path(_cache_root_env)
else:
    CACHE_ROOT = ROOT.parent / ".hf_cache"
# Same mount-guard as for ROOT — applies whether the path was explicit or
# auto-derived (Codex 3rd-pass P2-1).
_must_be_mounted(CACHE_ROOT, "/staging")
if not CACHE_ROOT.parent.exists():
    raise SystemExit(
        f"FATAL: CACHE_ROOT.parent does not exist: {CACHE_ROOT.parent}. "
        f"Refusing to materialise HF cache in a path that may resolve into "
        f"the container overlay-fs."
    )
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(CACHE_ROOT)
os.environ["HF_DATASETS_CACHE"] = str(CACHE_ROOT / "datasets")
os.environ["HF_HUB_CACHE"] = str(CACHE_ROOT / "hub")

# Import datasets AFTER the env vars are set — datasets reads them at import.
from datasets import load_dataset  # noqa: E402

print(f"PHASE2_RAW_ROOT = {ROOT}", flush=True)
print(f"PHASE2_CACHE_ROOT = {CACHE_ROOT}", flush=True)


def write_jsonl_to_text(out_file: Path, ds_iter, content_field: str, max_bytes: int | None = None):
    """Stream documents to a single .txt file, one document per line.

    Each document is written as: <content>\\n<\\n>  (separator: empty line).
    """
    bytes_written = 0
    docs = 0
    skipped = 0
    t0 = time.time()
    last_log = t0

    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8", buffering=1024 * 1024) as f:
        for ex in ds_iter:
            text = ex.get(content_field) or ""
            if not isinstance(text, str) or len(text) < 50:
                skipped += 1
                continue
            # Strip null-bytes & low-control chars (L-008)
            text = "".join(c for c in text if c == "\n" or c == "\t" or ord(c) >= 0x20)
            if not text.strip():
                skipped += 1
                continue
            line = text.replace("\0", "") + "\n"
            f.write(line + "\n")  # blank line as document separator
            bytes_written += len(line.encode("utf-8")) + 1
            docs += 1
            if time.time() - last_log > 30:
                rate = bytes_written / max(time.time() - t0, 0.01) / 1e6
                print(f"  [{out_file.name}] {docs:,} docs, {bytes_written/1e9:.2f} GB, {rate:.1f} MB/s", flush=True)
                last_log = time.time()
            if max_bytes and bytes_written >= max_bytes:
                print(f"  [{out_file.name}] reached max_bytes cap — stopping", flush=True)
                break
    return docs, bytes_written, skipped


def manifest(out_dir: Path, src: str, docs: int, bytes_out: int, started: float, skipped: int = 0, **extra):
    info = {
        "source": src,
        "documents": docs,
        "bytes_written": bytes_out,
        "skipped": skipped,
        "elapsed_seconds": time.time() - started,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(started)),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }
    (out_dir / "manifest.json").write_text(json.dumps(info, indent=2))


def _dl_stack_v2_lang(out_subdir: str, languages: list[str], max_bytes: int):
    """Code downloader using ``bigcode/starcoderdata`` (parquet+content, non-gated).

    History of this function (chosen for the explanatory value, the alternatives
    we tried and why they failed):

    * ``bigcode/the-stack-v2-dedup`` — gated, but parquet contains only metadata
      (``blob_id``, ``path``, ``language`` …). Real content lives in S3 via
      Software Heritage; would need 7M+ HTTP requests per parquet. Too slow.
    * ``codeparrot/github-code-clean`` — script-based loader (``github-code-clean.py``);
      ``datasets`` v4+ rejects script loaders (see L-011).
    * ``bigcode/starcoderdata`` — **what we use**: pure parquet, ``content`` field
      present, organised by language (``python/``, ``go/``, ``rust/`` …),
      non-gated. Phase-1 already used a streaming subset of the same dataset.

    Language tokens are lower-case in starcoderdata, but we accept the original
    capitalisation for backward compatibility and lower-case them here.
    """
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    out_dir = ROOT / out_subdir
    out_file = out_dir / f"{out_subdir}.txt"
    print(f"=== bigcode/starcoderdata ({'+'.join(languages)}) -> {out_file} ===", flush=True)
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    repo_files = api.list_repo_files(repo_id="bigcode/starcoderdata", repo_type="dataset")

    total_docs = 0
    total_bytes = 0
    total_skipped = 0
    parquet_count = 0

    with out_file.open("w", encoding="utf-8", buffering=1024 * 1024) as f:
        for lang in languages:
            lang_lc = lang.lower()
            lang_files = sorted(p for p in repo_files if p.startswith(f"{lang_lc}/") and p.endswith(".parquet"))
            print(f"  --- {lang} ({len(lang_files)} parquet files) ---", flush=True)
            if not lang_files:
                print(f"  WARN: no parquet files for language {lang_lc!r}", flush=True)
                continue

            for pf in lang_files:
                if total_bytes >= max_bytes:
                    break
                parquet_count += 1
                print(f"  [{parquet_count}/{len(lang_files)}] downloading {pf} ...", flush=True)
                local_pf = hf_hub_download(
                    repo_id="bigcode/starcoderdata",
                    filename=pf,
                    repo_type="dataset",
                )
                pf_size_gb = Path(local_pf).stat().st_size / 1e9
                print(f"  [{parquet_count}] {pf_size_gb:.2f} GB on disk, iterating...", flush=True)

                pq_file = pq.ParquetFile(local_pf)
                for batch in pq_file.iter_batches(batch_size=2000, columns=["content"]):
                    rows = batch.to_pylist()
                    for row in rows:
                        text = row.get("content") or ""
                        if not isinstance(text, str) or len(text) < 50:
                            total_skipped += 1
                            continue
                        text = "".join(c for c in text if c == "\n" or c == "\t" or ord(c) >= 0x20)
                        if not text.strip():
                            total_skipped += 1
                            continue
                        line = text.replace("\0", "") + "\n"
                        f.write(line + "\n")
                        total_bytes += len(line.encode("utf-8")) + 1
                        total_docs += 1
                        if total_docs % 10000 == 0:
                            rate = total_bytes / max(time.time() - t0, 0.01) / 1e6
                            print(f"  [{out_file.name}] {total_docs:,} docs, {total_bytes/1e9:.2f} GB, {rate:.1f} MB/s, skipped {total_skipped:,}", flush=True)
                        if total_bytes >= max_bytes:
                            break
                    if total_bytes >= max_bytes:
                        break

                # Free disk: remove the consumed parquet
                try:
                    Path(local_pf).unlink()
                except Exception:
                    pass

            if total_bytes >= max_bytes:
                break

    manifest(out_dir, f"bigcode/starcoderdata [{'+'.join(languages)}]",
             total_docs, total_bytes, t0, total_skipped, languages=languages)
    print(f"=== DONE: {total_docs:,} docs, {total_bytes/1e9:.1f} GB in {(time.time()-t0)/60:.1f} min ===")


def dl_the_stack_v2_python():
    """Python files from the-stack-v2-dedup. Cap 60 GB raw."""
    _dl_stack_v2_lang("the_stack_v2_python", ["Python"], max_bytes=60 * 1024**3)


def dl_the_stack_v2_js_ts():
    """JavaScript + TypeScript. Cap 30 GB."""
    _dl_stack_v2_lang("the_stack_v2_js_ts", ["JavaScript", "TypeScript"], max_bytes=30 * 1024**3)


def dl_the_stack_v2_rust_go():
    """Rust + Go. Cap 15 GB."""
    _dl_stack_v2_lang("the_stack_v2_rust_go", ["Rust", "Go"], max_bytes=15 * 1024**3)


def dl_smollm_python_edu():
    """SmolLM corpus python-edu subset. ~4 GB raw, ~1 B tokens."""
    out_dir = ROOT / "smollm_python_edu"
    out_file = out_dir / "smollm_python_edu.txt"
    print(f"=== smollm-corpus python-edu -> {out_file} ===", flush=True)
    t0 = time.time()
    ds = load_dataset(
        "HuggingFaceTB/smollm-corpus",
        "python-edu",
        split="train",
        streaming=True,
    )
    docs, bytes_out, skipped = write_jsonl_to_text(out_file, ds, content_field="text")
    manifest(out_dir, "HuggingFaceTB/smollm-corpus [python-edu]", docs, bytes_out, t0, skipped)
    print(f"=== DONE: {docs:,} docs, {bytes_out/1e9:.1f} GB in {(time.time()-t0)/60:.1f} min ===")


def dl_fineweb_10bt():
    """FineWeb sample-10BT. ~50 GB raw, ~10 B tokens."""
    out_dir = ROOT / "fineweb_10bt"
    out_file = out_dir / "fineweb_10bt.txt"
    print(f"=== fineweb sample-10BT -> {out_file} ===", flush=True)
    t0 = time.time()
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    docs, bytes_out, skipped = write_jsonl_to_text(out_file, ds, content_field="text")
    manifest(out_dir, "HuggingFaceFW/fineweb [sample-10BT]", docs, bytes_out, t0, skipped)
    print(f"=== DONE: {docs:,} docs, {bytes_out/1e9:.1f} GB in {(time.time()-t0)/60:.1f} min ===")


def _dl_parquet_corpus(out_subdir: str, repo_id: str, file_prefix: str,
                      content_field: str, max_bytes: int, label: str | None = None):
    """Generic parquet-corpus downloader using hf_hub_download per file.

    Used for non-streaming, non-script-loader datasets where each parquet is a
    self-contained chunk (fineweb-2, wikipedia, starcoderdata, …).
    """
    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    out_dir = ROOT / out_subdir
    out_file = out_dir / f"{out_subdir}.txt"
    label = label or f"{repo_id} [{file_prefix}]"
    print(f"=== {label} -> {out_file} ===", flush=True)
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()
    repo_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    target_files = sorted(p for p in repo_files
                          if p.startswith(file_prefix) and p.endswith(".parquet"))
    print(f"  {len(target_files)} parquet files matching prefix {file_prefix!r}", flush=True)

    total_docs = 0
    total_bytes = 0
    total_skipped = 0

    with out_file.open("w", encoding="utf-8", buffering=1024 * 1024) as f:
        for i, pf in enumerate(target_files, 1):
            if total_bytes >= max_bytes:
                break
            print(f"  [{i}/{len(target_files)}] downloading {pf} ...", flush=True)
            local_pf = hf_hub_download(repo_id=repo_id, filename=pf, repo_type="dataset")
            pf_size_gb = Path(local_pf).stat().st_size / 1e9
            print(f"  [{i}] {pf_size_gb:.2f} GB on disk, iterating...", flush=True)

            pq_file = pq.ParquetFile(local_pf)
            for batch in pq_file.iter_batches(batch_size=2000, columns=[content_field]):
                rows = batch.to_pylist()
                for row in rows:
                    text = row.get(content_field) or ""
                    if not isinstance(text, str) or len(text) < 100:
                        total_skipped += 1
                        continue
                    text = "".join(c for c in text if c == "\n" or c == "\t" or ord(c) >= 0x20)
                    if not text.strip():
                        total_skipped += 1
                        continue
                    line = text.replace("\0", "") + "\n"
                    f.write(line + "\n")
                    total_bytes += len(line.encode("utf-8")) + 1
                    total_docs += 1
                    if total_docs % 10000 == 0:
                        rate = total_bytes / max(time.time() - t0, 0.01) / 1e6
                        print(f"  [{out_file.name}] {total_docs:,} docs, {total_bytes/1e9:.2f} GB, {rate:.1f} MB/s, skipped {total_skipped:,}", flush=True)
                    if total_bytes >= max_bytes:
                        break
                if total_bytes >= max_bytes:
                    break

            try:
                Path(local_pf).unlink()
            except Exception:
                pass

    manifest(out_dir, label, total_docs, total_bytes, t0, total_skipped, repo_id=repo_id, file_prefix=file_prefix)
    print(f"=== DONE: {total_docs:,} docs, {total_bytes/1e9:.1f} GB in {(time.time()-t0)/60:.1f} min ===")


def dl_fineweb2_de():
    """FineWeb-2 deu_Latn: ~10-12 B tokens of cleaned German web. Cap 50 GB."""
    _dl_parquet_corpus(
        out_subdir="fineweb2_de",
        repo_id="HuggingFaceFW/fineweb-2",
        file_prefix="data/deu_Latn/train/",
        content_field="text",
        max_bytes=50 * 1024**3,
        label="HuggingFaceFW/fineweb-2 [deu_Latn/train]",
    )


def dl_wikipedia_de():
    """German Wikipedia 2023-11-01 snapshot: clean encyclopedia content. Full ~15 GB."""
    _dl_parquet_corpus(
        out_subdir="wikipedia_de",
        repo_id="wikimedia/wikipedia",
        file_prefix="20231101.de/",
        content_field="text",
        max_bytes=20 * 1024**3,
        label="wikimedia/wikipedia [20231101.de]",
    )


SOURCES = {
    "the_stack_v2_python": dl_the_stack_v2_python,
    "the_stack_v2_js_ts": dl_the_stack_v2_js_ts,
    "the_stack_v2_rust_go": dl_the_stack_v2_rust_go,
    "smollm_python_edu": dl_smollm_python_edu,
    "fineweb_10bt": dl_fineweb_10bt,
    "fineweb2_de": dl_fineweb2_de,
    "wikipedia_de": dl_wikipedia_de,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=sorted(SOURCES.keys()), required=True)
    args = parser.parse_args()
    SOURCES[args.source]()


if __name__ == "__main__":
    main()
