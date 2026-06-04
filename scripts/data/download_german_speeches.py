"""Download the German Political Speeches Corpus (Adrien Barbaresi).

Source:
    http://adrien.barbaresi.eu/corpora/speeches/

What it is:
    Plenary-session and political speeches from German federal institutions
    (Bundestag, Bundespräsident, Bundesregierung, Bundesrat). Formal,
    high-register German that is rare in web-crawled corpora like fineweb2_de.

Output:
    raw/german/political_speeches.txt              (one document per line)
    raw/german/political_speeches.txt.manifest.json

Usage (from inside the container):
    python scripts/data/download_german_speeches.py
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import re
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.data._common import atomic_text_writer, now_iso  # noqa: E402

# Stable archive of the corpus. Downloads as a single zip with XML files,
# one speech per file. Total ~250 MB compressed.
DEFAULT_URL = "http://adrien.barbaresi.eu/corpora/speeches/German-political-speeches.zip"
DEFAULT_OUT = Path("raw/german/political_speeches.txt")


@dataclass
class Manifest:
    source: str
    url: str
    output_file: str
    started_at: str
    finished_at: str = ""
    elapsed_seconds: float = 0.0
    archive_size_bytes: int = 0
    files_in_archive: int = 0
    documents_written: int = 0
    bytes_written: int = 0
    notes: list = field(default_factory=list)


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def _xml_to_text(xml_bytes: bytes) -> str:
    """Strip XML tags and normalize whitespace.

    The corpus' XML schema is simple — strip tags is good enough; we don't
    need a real XML parser and avoid the lxml dependency."""
    text = xml_bytes.decode("utf-8", errors="replace")
    text = _TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _download(url: str, dest: Path) -> int:
    print(f"  downloading {url} -> {dest}", flush=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    bytes_written = 0
    with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            bytes_written += len(chunk)
            if bytes_written % (32 * 1024 * 1024) == 0:
                print(f"    {bytes_written / 1e6:.1f} MB", flush=True)
    tmp.rename(dest)
    print(f"  download done: {bytes_written / 1e6:.1f} MB", flush=True)
    return bytes_written


def _process_zip(archive: Path, out_path: Path, manifest: Manifest) -> None:
    """Iterate XML files in the zip; emit one cleaned document per line."""
    print(f"  extracting + cleaning XML -> {out_path}", flush=True)
    with zipfile.ZipFile(archive) as zf, atomic_text_writer(out_path) as fout:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if not info.filename.lower().endswith((".xml", ".txt")):
                continue
            manifest.files_in_archive += 1
            with zf.open(info) as fh:
                raw = fh.read()
            if info.filename.lower().endswith(".xml"):
                text = _xml_to_text(raw)
            else:
                text = raw.decode("utf-8", errors="replace")
                text = _WS_RE.sub(" ", text).strip()
            if len(text) < 200:               # skip tiny/empty stubs
                continue
            line = text + "\n"
            fout.write(line)
            manifest.documents_written += 1
            manifest.bytes_written += len(line.encode("utf-8"))
            if manifest.documents_written % 5000 == 0:
                print(f"    {manifest.documents_written} docs, "
                      f"{manifest.bytes_written / 1e6:.1f} MB", flush=True)


def _process_tar(archive: Path, out_path: Path, manifest: Manifest) -> None:
    """Same as _process_zip but for .tar.gz archives (fallback shape)."""
    print(f"  extracting + cleaning tar -> {out_path}", flush=True)
    with tarfile.open(archive, "r:*") as tf, atomic_text_writer(out_path) as fout:
        for member in tf:
            if not member.isfile():
                continue
            if not member.name.lower().endswith((".xml", ".txt")):
                continue
            manifest.files_in_archive += 1
            fh = tf.extractfile(member)
            if fh is None:
                continue
            raw = fh.read()
            if member.name.lower().endswith(".xml"):
                text = _xml_to_text(raw)
            else:
                text = raw.decode("utf-8", errors="replace")
                text = _WS_RE.sub(" ", text).strip()
            if len(text) < 200:
                continue
            line = text + "\n"
            fout.write(line)
            manifest.documents_written += 1
            manifest.bytes_written += len(line.encode("utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--keep-archive", action="store_true",
                   help="Keep the raw archive after extraction (default: delete).")
    args = p.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    archive = args.output.parent / Path(args.url).name
    manifest = Manifest(
        source="german_political_speeches",
        url=args.url,
        output_file=str(args.output),
        started_at=now_iso(),
    )
    t0 = time.time()
    try:
        manifest.archive_size_bytes = _download(args.url, archive)

        if archive.suffix == ".zip":
            _process_zip(archive, args.output, manifest)
        elif ".tar" in archive.suffixes or archive.suffix in (".tgz", ".tar"):
            _process_tar(archive, args.output, manifest)
        else:
            raise SystemExit(f"unsupported archive type: {archive.name}")

        if not args.keep_archive:
            archive.unlink()
            manifest.notes.append("archive deleted post-extract")
    except Exception as e:                    # noqa: BLE001
        manifest.notes.append(f"FAILED: {type(e).__name__}: {e}")
        raise
    finally:
        manifest.finished_at = now_iso()
        manifest.elapsed_seconds = round(time.time() - t0, 1)
        manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
        manifest_path.write_text(
            json.dumps(asdict(manifest), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  manifest: {manifest_path}")
    print(f"\ndone: {manifest.documents_written} docs, "
          f"{manifest.bytes_written / 1e6:.1f} MB, "
          f"{manifest.elapsed_seconds:.0f}s")


if __name__ == "__main__":
    main()
