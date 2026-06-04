"""Assemble downloaded raw .txt (multi-line docs separated by blank lines) into
the one-doc-per-line format that ``filter_quality.py`` expects.

The Phase-1 raw cleaning pipeline reads files where each *line* is one document.
``download_phase2_pretrain.py`` writes blocks like::

    line 1 of doc A
    line 2 of doc A
    <blank line>
    line 1 of doc B
    <blank line>

This converter has two modes:

* ``--mode text``  — join all internal newlines with a single space.
  Use for natural-language web text (fineweb, smollm-edu cosmopedia, etc.).

* ``--mode code``  — preserve newlines inside a doc. Each doc becomes ONE
  output line by escaping internal newlines as the literal string ``\\n``.
  ``filter_quality.py`` will see protected boundary markers, so we prepend
  ``<file_sep>`` between docs (after writing them line by line, no need to
  collapse — instead emit ``<file_sep>`` as its own line, then the doc's
  raw lines, so ``filter_quality.py`` can keep its line-level filtering and
  ``preserve_newlines=True`` for the ``code`` language).

Modes summarised:

* ``text``: one line per doc, internal newlines → space
* ``code``: line-per-line, with a ``<file_sep>`` line preceding each doc

Usage:
    python assemble_for_filter.py --input raw/fineweb_10bt/fineweb_10bt.txt \\
        --output cleaned/_pre_filter/fineweb_10bt.assembled.txt --mode text
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def assemble_text(input_path: Path, output_path: Path) -> dict:
    """Multi-line docs (blank-line separated) -> one-line-per-doc."""
    docs = 0
    bytes_in = input_path.stat().st_size
    bytes_out = 0
    t0 = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    buf: list[str] = []

    def flush(out_fh) -> int:
        if not buf:
            return 0
        # Join all collected lines with single space, collapse multiple spaces
        text = " ".join(b.strip() for b in buf if b.strip())
        if not text:
            return 0
        out_fh.write(text + "\n")
        return len(text) + 1

    with output_path.open("w", encoding="utf-8", buffering=1024 * 1024) as out_fh, \
         input_path.open("r", encoding="utf-8", errors="replace") as in_fh:
        for line in in_fh:
            stripped = line.rstrip("\n").rstrip("\r")
            if stripped == "":
                # Blank line = doc separator
                written = flush(out_fh)
                if written:
                    bytes_out += written
                    docs += 1
                buf.clear()
            else:
                buf.append(stripped)
        # Flush trailing doc (if file does not end with blank line)
        written = flush(out_fh)
        if written:
            bytes_out += written
            docs += 1

    return {
        "mode": "text",
        "input_file": str(input_path),
        "output_file": str(output_path),
        "bytes_in": bytes_in,
        "bytes_out": bytes_out,
        "documents": docs,
        "elapsed_seconds": time.time() - t0,
    }


def assemble_code(input_path: Path, output_path: Path) -> dict:
    """Multi-line docs -> still multi-line, but with <file_sep> markers between
    docs so the line-level filter knows where one ends and another begins."""
    docs = 0
    bytes_in = input_path.stat().st_size
    bytes_out = 0
    t0 = time.time()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    in_doc = False
    with output_path.open("w", encoding="utf-8", buffering=1024 * 1024) as out_fh, \
         input_path.open("r", encoding="utf-8", errors="replace") as in_fh:
        for line in in_fh:
            stripped_nl = line.rstrip("\n").rstrip("\r")
            if stripped_nl == "":
                if in_doc:
                    in_doc = False
                    docs += 1
                continue
            if not in_doc:
                # Start of a new doc — emit a marker line first
                out_fh.write("<file_sep>\n")
                bytes_out += len("<file_sep>\n")
                in_doc = True
            out_fh.write(stripped_nl + "\n")
            bytes_out += len(stripped_nl) + 1
        if in_doc:
            docs += 1

    return {
        "mode": "code",
        "input_file": str(input_path),
        "output_file": str(output_path),
        "bytes_in": bytes_in,
        "bytes_out": bytes_out,
        "documents": docs,
        "elapsed_seconds": time.time() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["text", "code"], required=True)
    args = parser.parse_args()

    if not args.input.exists():
        sys.exit(f"input not found: {args.input}")

    if args.mode == "text":
        info = assemble_text(args.input, args.output)
    else:
        info = assemble_code(args.input, args.output)

    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(info, indent=2), encoding="utf-8")

    print(f"=== assemble done ===")
    print(f"  mode: {info['mode']}")
    print(f"  in:   {info['bytes_in']/1e9:.2f} GB ({args.input})")
    print(f"  out:  {info['bytes_out']/1e9:.2f} GB, {info['documents']:,} docs ({args.output})")
    print(f"  time: {info['elapsed_seconds']/60:.1f} min")


if __name__ == "__main__":
    main()
