#!/usr/bin/env python3
"""Build clean-v3 pretraining text sources and mix.

This orchestrates strict_filter_pretrain.py with source-specific profiles:

- prose for German Commons, legacy German, and Wikipedia
- math for OpenMath
- booster for tiny synthetic arithmetic examples

Outputs are written under data/training/pretrain_clean_v3 by default.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]

SOURCES = [
    {
        "name": "german_commons",
        "input": "data/training/pretrain_clean_v2/german_commons.strict.txt",
        "output": "german_commons.v3.txt",
        "language": "german",
        "profile": "prose",
        "extra": ["--max-chars", "30000", "--max-list-score", "8", "--max-name-list-hits", "24"],
    },
    {
        "name": "german_legacy",
        "input": "data/training/pretrain_clean_v2/german.strict.txt",
        "output": "german_legacy.v3.txt",
        "language": "german",
        "profile": "prose",
        "extra": ["--max-chars", "30000", "--max-list-score", "8", "--max-name-list-hits", "24"],
    },
    {
        "name": "wikipedia_de",
        "input": "data/training/pretrain_clean_v2/wikipedia_de.strict.txt",
        "output": "wikipedia_de.v3.txt",
        "language": "german",
        "profile": "prose",
        "extra": ["--max-chars", "35000", "--max-list-score", "10", "--max-name-list-hits", "30"],
    },
    {
        "name": "openmath",
        "input": "data/training/pretrain_clean_v2/openmath.strict.txt",
        "output": "openmath.v3.txt",
        "language": "english",
        "profile": "math",
        "extra": [
            "--min-chars",
            "80",
            "--max-symbol",
            "0.45",
            "--max-urls",
            "2",
            "--max-repetition",
            "0.72",
        ],
    },
    {
        "name": "booster",
        "input": "data/training/pretrain_booster_de_v1m.txt",
        "output": "booster.v3.txt",
        "language": "german",
        "profile": "booster",
        "extra": [
            "--min-chars",
            "40",
            "--max-symbol",
            "0.45",
            "--max-urls",
            "0",
            "--max-repetition",
            "0.85",
        ],
    },
]


def run(cmd: list[str], dry_run: bool) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True, cwd=REPO)


def read_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(output_dir: Path, mix_manifest: Path) -> None:
    rows = []
    for item in SOURCES:
        out = output_dir / item["output"]
        manifest = read_manifest(out.with_suffix(out.suffix + ".manifest.json"))
        keep_rate = manifest["lines_written"] / max(manifest["lines_in"], 1)
        byte_rate = manifest["bytes_written"] / max(manifest["bytes_in"], 1)
        rows.append(
            {
                "name": item["name"],
                "input_file": manifest["input_file"],
                "output_file": manifest["output_file"],
                "lines_in": manifest["lines_in"],
                "lines_written": manifest["lines_written"],
                "bytes_in": manifest["bytes_in"],
                "bytes_written": manifest["bytes_written"],
                "keep_rate": round(keep_rate, 4),
                "byte_keep_rate": round(byte_rate, 4),
                "dropped": manifest["dropped"],
            }
        )

    summary = {
        "variant": "pretrain_clean_v3",
        "sources": rows,
        "mix_manifest": str(mix_manifest),
        "mix": read_manifest(mix_manifest) if mix_manifest.exists() else None,
    }
    summary_path = output_dir / "manifest.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Pretrain Clean-v3 Summary\n"]
    md.append("| Source | Docs In | Docs Out | Keep | GB In | GB Out | Top Drops |")
    md.append("|---|---:|---:|---:|---:|---:|---|")
    for row in rows:
        top = ", ".join(f"{k}: {v:,}" for k, v in list(row["dropped"].items())[:5])
        md.append(
            f"| `{row['name']}` | {row['lines_in']:,} | {row['lines_written']:,} | "
            f"{row['keep_rate']*100:.1f}% | {row['bytes_in']/1e9:.2f} | "
            f"{row['bytes_written']/1e9:.2f} | {top or '-'} |"
        )
    if summary["mix"]:
        mix = summary["mix"]
        md.append("\n## Mix\n")
        md.append(f"- Documents: {mix['documents']:,}")
        md.append(f"- Bytes: {mix['bytes_written']/1e9:.2f} GB")
        md.append(f"- Validation tail: {mix['val_tail_bytes']/1e6:.1f} MB / {mix['val_tail_documents']:,} docs")
    md.append("\n## Notes\n")
    md.append("- Prose sources use clean-v3 structural filters for TOC/index/list/OCR/catalogue fragments.")
    md.append("- OpenMath uses a math profile so symbolic examples are preserved.")
    md.append("- Booster remains separate and should be used as a small upsampled specialty source, not dominant base text.")
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO / "data/training/pretrain_clean_v3")
    parser.add_argument("--mix-output", type=Path, default=None)
    parser.add_argument("--val-tail-bytes", type=int, default=80_000_000)
    parser.add_argument("--chunk-lines", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument("--max-docs-per-source", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    filter_script = REPO / "scripts/data/strict_filter_pretrain.py"
    for item in SOURCES:
        inp = REPO / item["input"]
        out = output_dir / item["output"]
        if not inp.is_file():
            raise SystemExit(f"missing source for {item['name']}: {inp}")
        cmd = [
            sys.executable,
            str(filter_script),
            "--input",
            str(inp),
            "--output",
            str(out),
            "--language",
            item["language"],
            "--profile",
            item["profile"],
            "--v3-structure-filters",
            "--seed",
            str(args.seed),
        ]
        if args.max_docs_per_source:
            cmd += ["--max-docs", str(args.max_docs_per_source)]
        cmd += item["extra"]
        run(cmd, args.dry_run)

    mix_output = args.mix_output or (output_dir / "mix_full.txt")
    mix_script = REPO / "scripts/data/build_pretrain_mix_v2.py"
    mix_cmd = [
        sys.executable,
        str(mix_script),
        "--output",
        str(mix_output),
        "--val-tail-bytes",
        str(args.val_tail_bytes),
        "--chunk-lines",
        str(args.chunk_lines),
        "--seed",
        str(args.seed),
    ]
    for item in SOURCES:
        mix_cmd += ["--source", str(output_dir / item["output"])]
    run(mix_cmd, args.dry_run)

    mix_manifest = mix_output.with_suffix(mix_output.suffix + ".manifest.json")
    if not args.dry_run:
        write_summary(output_dir, mix_manifest)
        print(f"wrote {output_dir / 'manifest.json'}")
        print(f"wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
