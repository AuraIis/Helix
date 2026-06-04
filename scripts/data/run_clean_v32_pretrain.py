#!/usr/bin/env python3
"""Build pretrain clean-v3.2 from clean-v3.1 plus new 1B raw sources.

This pass keeps the active v4/500M run untouched. It revalidates clean-v3.1,
adds the downloaded FineWeb/DCLM sources, and applies the stricter filters that
caught the remaining shop, adult, casino, URL, TOC, and boilerplate fragments.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]

BASE_SOURCES = [
    {
        "name": "german_commons_v31",
        "input_base": "clean_v31",
        "input": "german_commons.v31.txt",
        "output": "german_commons.v32.txt",
        "language": "german",
        "profile": "prose",
        "extra": [
            "--min-chars",
            "260",
            "--max-chars",
            "28000",
            "--max-urls",
            "0",
            "--max-list-score",
            "7",
            "--max-name-list-hits",
            "20",
            "--drop-web-boilerplate",
            "--drop-commercial-boilerplate",
            "--drop-adult-gambling-spam",
            "--drop-old-ocr",
            "--max-old-ocr-hits",
            "9",
        ],
    },
    {
        "name": "german_legacy_v31",
        "input_base": "clean_v31",
        "input": "german_legacy.v31.txt",
        "output": "german_legacy.v32.txt",
        "language": "german",
        "profile": "prose",
        "extra": [
            "--min-chars",
            "260",
            "--max-chars",
            "26000",
            "--max-urls",
            "0",
            "--max-list-score",
            "7",
            "--max-name-list-hits",
            "20",
            "--drop-web-boilerplate",
            "--drop-commercial-boilerplate",
            "--drop-adult-gambling-spam",
            "--drop-old-ocr",
            "--max-old-ocr-hits",
            "9",
        ],
    },
    {
        "name": "wikipedia_de_v31",
        "input_base": "clean_v31",
        "input": "wikipedia_de.v31.txt",
        "output": "wikipedia_de.v32.txt",
        "language": "german",
        "profile": "prose",
        "extra": [
            "--min-chars",
            "220",
            "--max-chars",
            "35000",
            "--max-urls",
            "1",
            "--max-list-score",
            "9",
            "--max-name-list-hits",
            "26",
            "--drop-web-boilerplate",
            "--drop-commercial-boilerplate",
            "--drop-adult-gambling-spam",
        ],
    },
    {
        "name": "openmath_v31",
        "input_base": "clean_v31",
        "input": "openmath.v31.txt",
        "output": "openmath.v32.txt",
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
        "name": "booster_v31",
        "input_base": "clean_v31",
        "input": "booster.v31.txt",
        "output": "booster.v32.txt",
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

RAW_1B_SOURCES = [
    {
        "name": "fineweb2_de_raw",
        "input_base": "raw_1b",
        "input": "german/fineweb2_de.txt",
        "output": "fineweb2_de.v32.txt",
        "language": "german",
        "profile": "prose",
        "extra": [
            "--min-chars",
            "320",
            "--max-chars",
            "32000",
            "--max-urls",
            "0",
            "--max-list-score",
            "6",
            "--max-name-list-hits",
            "18",
            "--max-bibliography-hits",
            "2",
            "--max-ocr-hits",
            "2",
            "--max-repetition",
            "0.58",
            "--drop-web-boilerplate",
            "--drop-commercial-boilerplate",
            "--drop-adult-gambling-spam",
            "--drop-old-ocr",
            "--max-old-ocr-hits",
            "8",
        ],
    },
    {
        "name": "fineweb_edu_raw",
        "input_base": "raw_1b",
        "input": "english/fineweb_edu.txt",
        "output": "fineweb_edu.v32.txt",
        "language": "english",
        "profile": "prose",
        "extra": [
            "--min-chars",
            "320",
            "--max-chars",
            "42000",
            "--max-urls",
            "1",
            "--max-list-score",
            "7",
            "--max-name-list-hits",
            "20",
            "--max-bibliography-hits",
            "3",
            "--max-ocr-hits",
            "3",
            "--max-repetition",
            "0.60",
            "--drop-web-boilerplate",
            "--drop-commercial-boilerplate",
            "--drop-adult-gambling-spam",
        ],
    },
    {
        "name": "dclm_edu_raw",
        "input_base": "raw_1b",
        "input": "english/dclm_edu.txt",
        "output": "dclm_edu.v32.txt",
        "language": "english",
        "profile": "prose",
        "extra": [
            "--min-chars",
            "320",
            "--max-chars",
            "42000",
            "--max-urls",
            "1",
            "--max-list-score",
            "7",
            "--max-name-list-hits",
            "20",
            "--max-bibliography-hits",
            "3",
            "--max-ocr-hits",
            "3",
            "--max-repetition",
            "0.60",
            "--drop-web-boilerplate",
            "--drop-commercial-boilerplate",
            "--drop-adult-gambling-spam",
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


def source_input_path(item: dict, clean_v31_dir: Path, raw_1b_root: Path) -> Path:
    base = clean_v31_dir if item["input_base"] == "clean_v31" else raw_1b_root
    return base / item["input"]


def write_summary(output_dir: Path, sources: list[dict], mix_manifest: Path) -> None:
    rows = []
    for item in sources:
        out = output_dir / item["output"]
        manifest = read_manifest(out.with_suffix(out.suffix + ".manifest.json"))
        keep_rate = manifest["lines_written"] / max(manifest["lines_in"], 1)
        byte_rate = manifest["bytes_written"] / max(manifest["bytes_in"], 1)
        rows.append(
            {
                "name": item["name"],
                "input_base": item["input_base"],
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
        "variant": "pretrain_clean_v32",
        "sources": rows,
        "mix_manifest": str(mix_manifest),
        "mix": read_manifest(mix_manifest) if mix_manifest.exists() else None,
        "notes": [
            "v3.2 revalidates clean-v3.1 and adds strict-filtered FineWeb/DCLM 1B sources.",
            "Commercial/shop/adult/gambling boilerplate filters are enabled for prose.",
            "This does not alter the active v4 boosted 500M run.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md = ["# Pretrain Clean-v3.2 Summary\n"]
    md.append("| Source | Base | Docs In | Docs Out | Keep | GB In | GB Out | Top Drops |")
    md.append("|---|---|---:|---:|---:|---:|---:|---|")
    for row in rows:
        top = ", ".join(f"{k}: {v:,}" for k, v in list(row["dropped"].items())[:5])
        md.append(
            f"| `{row['name']}` | {row['input_base']} | {row['lines_in']:,} | "
            f"{row['lines_written']:,} | {row['keep_rate']*100:.1f}% | "
            f"{row['bytes_in']/1e9:.2f} | {row['bytes_written']/1e9:.2f} | {top or '-'} |"
        )
    if summary["mix"]:
        mix = summary["mix"]
        md.append("\n## Mix\n")
        md.append(f"- Documents: {mix['documents']:,}")
        md.append(f"- Bytes: {mix['bytes_written']/1e9:.2f} GB")
        md.append(f"- Validation tail: {mix['val_tail_bytes']/1e6:.1f} MB / {mix['val_tail_documents']:,} docs")
    md.append("\n## Notes\n")
    md.extend(f"- {note}" for note in summary["notes"])
    (output_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-v31-dir", type=Path, default=REPO / "data/training/pretrain_clean_v31")
    parser.add_argument("--raw-1b-root", type=Path, default=REPO / "data/pretrain_1b_sources_v1/raw")
    parser.add_argument("--output-dir", type=Path, default=REPO / "data/training/pretrain_clean_v32")
    parser.add_argument("--mix-output", type=Path, default=None)
    parser.add_argument("--val-tail-bytes", type=int, default=120_000_000)
    parser.add_argument("--chunk-lines", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--max-docs-per-source", type=int, default=0)
    parser.add_argument("--skip-raw-1b", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    clean_v31_dir = args.clean_v31_dir if args.clean_v31_dir.is_absolute() else REPO / args.clean_v31_dir
    raw_1b_root = args.raw_1b_root if args.raw_1b_root.is_absolute() else REPO / args.raw_1b_root
    output_dir = args.output_dir if args.output_dir.is_absolute() else REPO / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = list(BASE_SOURCES)
    if not args.skip_raw_1b:
        sources.extend(RAW_1B_SOURCES)

    filter_script = REPO / "scripts/data/strict_filter_pretrain.py"
    for item in sources:
        inp = source_input_path(item, clean_v31_dir, raw_1b_root)
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
    for item in sources:
        mix_cmd += ["--source", str(output_dir / item["output"])]
    run(mix_cmd, args.dry_run)

    mix_manifest = mix_output.with_suffix(mix_output.suffix + ".manifest.json")
    if not args.dry_run:
        write_summary(output_dir, sources, mix_manifest)
        print(f"wrote {output_dir / 'manifest.json'}")
        print(f"wrote {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
