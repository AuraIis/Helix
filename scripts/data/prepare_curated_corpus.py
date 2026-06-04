"""Inspect and prepare the curated 40B corpus plan.

This script does three things:

1. reads a source-budgeted corpus plan (default: ``configs/data/curated_40b_mix.yaml``)
2. reports which sources already exist locally and which still need to be fetched
3. optionally downloads the missing sources by delegating to the existing
   ``download_english.py`` / ``download_german.py`` / ``download_code.py`` scripts

It is intentionally orchestration-only. The heavy lifting remains in the
source-specific download scripts we already trust.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.data._common import load_paths  # noqa: E402


@dataclass
class SourceStatus:
    language: str
    name: str
    status: str
    target_tokens: int
    local_path: str
    exists: bool
    size_bytes: int
    notes: str


SCRIPT_BY_LANGUAGE = {
    "english": REPO / "scripts" / "data" / "download_english.py",
    "german": REPO / "scripts" / "data" / "download_german.py",
    "code": REPO / "scripts" / "data" / "download_code.py",
}


DOWNLOAD_NAME_MAP = {
    "fineweb_edu": "fineweb_edu",
    "fineweb2_en": "fineweb2_en",
    "wikipedia_en": "wikipedia_en",
    "dolma": "dolma",
    "openmath": "openmath",
    "german_commons": "german_commons",
    "wikipedia_de": "wikipedia_de",
    "oscar_de": "oscar_de",
    "the_stack_v2": "the_stack_v2",
    "open_web_math": "open_web_math",
}


def _iter_sources(plan: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    for language in ("english", "german", "code"):
        for source in plan[language]["sources"]:
            rows.append((language, source))
    return rows


def _build_statuses(plan: dict[str, Any], data_root: Path) -> list[SourceStatus]:
    statuses: list[SourceStatus] = []
    for language, source in _iter_sources(plan):
        path = data_root / source["local_path"]
        exists = path.is_file()
        size_bytes = path.stat().st_size if exists else 0
        statuses.append(
            SourceStatus(
                language=language,
                name=source["name"],
                status=source["status"],
                target_tokens=int(source["target_tokens"]),
                local_path=str(path),
                exists=exists,
                size_bytes=size_bytes,
                notes=str(source.get("notes", "")),
            )
        )
    return statuses


def _write_report(statuses: list[SourceStatus], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Curated Corpus Preparation Report",
        "",
        "| Language | Source | Policy | Exists | GB | Target Tokens | Notes |",
        "|---|---|---|:-:|--:|--:|---|",
    ]
    for row in statuses:
        exists = "yes" if row.exists else "no"
        lines.append(
            f"| {row.language} | `{row.name}` | {row.status} | {exists} | "
            f"{row.size_bytes/1e9:.2f} | {row.target_tokens/1e9:.2f}B | {row.notes} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _download_missing(statuses: list[SourceStatus], config_path: Path) -> None:
    missing_by_language: dict[str, list[SourceStatus]] = {"english": [], "german": [], "code": []}
    for row in statuses:
        if row.exists:
            continue
        if row.status not in {"acquire", "keep", "keep_small"}:
            continue
        missing_by_language[row.language].append(row)

    for language, rows in missing_by_language.items():
        if not rows:
            continue
        script = SCRIPT_BY_LANGUAGE[language]
        source_names = [DOWNLOAD_NAME_MAP[row.name] for row in rows if row.name in DOWNLOAD_NAME_MAP]
        if not source_names:
            continue
        overrides = [f"{row.name}={row.target_tokens}" for row in rows]
        cmd = [
            sys.executable,
            str(script),
            "--config",
            str(config_path),
            "--sources",
            *source_names,
            "--target-tokens-override",
            *overrides,
        ]
        subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO / "configs" / "data_paths.yaml",
        help="Path to data_paths.yaml.",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=REPO / "configs" / "data" / "curated_40b_mix.yaml",
        help="Curated corpus plan YAML.",
    )
    parser.add_argument(
        "--report-md",
        type=Path,
        default=REPO / "data" / "eval" / "curated_corpus_report.md",
        help="Markdown report output path.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=REPO / "data" / "eval" / "curated_corpus_report.json",
        help="JSON report output path.",
    )
    parser.add_argument(
        "--download-missing",
        action="store_true",
        help="Fetch missing sources through the existing download scripts.",
    )
    args = parser.parse_args()

    cfg = load_paths(args.config)
    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    statuses = _build_statuses(plan, Path(cfg["_data_root"]))

    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps([asdict(row) for row in statuses], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_report(statuses, args.report_md)

    if args.download_missing:
        _download_missing(statuses, args.config)

    print(f"wrote {args.report_md}")
    print(f"wrote {args.report_json}")


if __name__ == "__main__":
    main()
