#!/usr/bin/env python3
"""Polish clean-v3 into clean-v3.1 before tokenization.

clean-v3 removed the dangerous structural junk. This pass is deliberately
narrower: it keeps the already-good sources, but removes remaining web
boilerplate from legacy German text and harder OCR/old-print fragments from
the German prose sources.
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
        "input": "german_commons.v3.txt",
        "output": "german_commons.v31.txt",
        "language": "german",
        "profile": "prose",
        "extra": [
            "--max-chars",
            "30000",
            "--max-list-score",
            "8",
            "--max-name-list-hits",
            "24",
            "--drop-old-ocr",
            "--max-old-ocr-hits",
            "10",
        ],
    },
    {
        "name": "german_legacy",
        "input": "german_legacy.v3.txt",
        "output": "german_legacy.v31.txt",
        "language": "german",
        "profile": "prose",
        "extra": [
            "--max-chars",
            "30000",
            "--max-list-score",
            "8",
            "--max-name-list-hits",
            "24",
            "--drop-web-boilerplate",
            "--drop-old-ocr",
            "--max-old-ocr-hits",
            "12",
        ],
    },
    {
        "name": "wikipedia_de",
        "input": "wikipedia_de.v3.txt",
        "output": "wikipedia_de.v31.txt",
        "language": "german",
        "profile": "prose",
        "extra": [
            "--max-chars",
            "35000",
            "--max-list-score",
            "10",
            "--max-name-list-hits",
            "30",
            "--drop-web-boilerplate",
        ],
    },
    {
        "name": "openmath",
        "input": "openmath.v3.txt",
        "output": "openmath.v31.txt",
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
        "input": "booster.v3.txt",
        "output": "booster.v31.txt",
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
        "variant": "pretrain_clean_v31",
        "sources": rows,
        "mix_manifest": str(mix_manifest),
        "mix": read_manifest(mix_manifest) if mix_manifest.exists() else None,
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Pretrain Clean-v3.1 Summary\n"]
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
    md.append("- v3.1 starts from clean-v3 and removes remaining web boilerplate plus hard OCR/old-print fragments.")
    md.append("- OpenMath and Booster are revalidated with their domain profiles, not prose filters.")
    md.append("- This is intended as the final text source before tokenizer conversion.")
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=REPO / "data/training/pretrain_clean_v3")
    parser.add_argument("--output-dir", type=Path, default=REPO / "data/training/pretrain_clean_v31")
    parser.add_argument("--mix-output", type=Path, default=None)
    parser.add_argument("--val-tail-bytes", type=int, default=80_000_000)
    parser.add_argument("--chunk-lines", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260513)
    parser.add_argument("--max-docs-per-source", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_dir = args.input_dir if args.input_dir.is_absolute() else REPO / args.input_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    filter_script = REPO / "scripts/data/strict_filter_pretrain.py"
    for item in SOURCES:
        inp = input_dir / item["input"]
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
