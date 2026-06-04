"""Create a final source-budgeted corpus from explicit source files.

This is the "mixer" step for the curated pipeline:

- read the curated source plan
- copy only the requested token-equivalent budget from each source
- emit one mixed file per language (english / german / code)
- write a manifest showing exactly how much each source contributed

The mixer is deliberately conservative:
- if a required source file is missing -> fail
- if a source is too small for its target budget -> fail by default
- no silent substitution
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Let the script run via `python scripts/data/mix_corpora.py` without an
# editable install: make the repo root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data._common import atomic_text_writer, load_paths, now_iso

REPO = Path(__file__).resolve().parents[2]


@dataclass
class SourceContribution:
    name: str
    local_path: str
    target_tokens: int
    target_bytes: int
    lines_written: int = 0
    bytes_written: int = 0


@dataclass
class LanguageManifest:
    language: str
    output_file: str
    started_at: str
    finished_at: str = ""
    target_tokens: int = 0
    target_bytes: int = 0
    bytes_written: int = 0
    lines_written: int = 0
    contributions: list[SourceContribution] = field(default_factory=list)


def _copy_budget(
    input_path: Path,
    output_fh,
    target_bytes: int,
) -> tuple[int, int]:
    """Copy up to ``target_bytes`` from ``input_path`` as whole lines."""
    lines_written = 0
    bytes_written = 0
    with input_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            encoded_len = len(line.encode("utf-8"))
            if bytes_written + encoded_len > target_bytes and lines_written > 0:
                break
            output_fh.write(line)
            lines_written += 1
            bytes_written += encoded_len
            if bytes_written >= target_bytes:
                break
    return lines_written, bytes_written


def _mix_language(
    language: str,
    section: dict[str, Any],
    data_root: Path,
    output_dir: Path,
    allow_shortfall: bool,
) -> LanguageManifest:
    bytes_per_token = float(section["estimated_bytes_per_token"])
    target_tokens = int(section["target_tokens"])
    target_bytes = int(target_tokens * bytes_per_token)
    output_path = output_dir / f"{language}.txt"
    manifest = LanguageManifest(
        language=language,
        output_file=str(output_path),
        started_at=now_iso(),
        target_tokens=target_tokens,
        target_bytes=target_bytes,
    )

    with atomic_text_writer(output_path) as out_fh:
        for source in section["sources"]:
            local_path = data_root / source["local_path"]
            source_target_tokens = int(source["target_tokens"])
            source_target_bytes = int(source_target_tokens * bytes_per_token)
            contribution = SourceContribution(
                name=str(source["name"]),
                local_path=str(local_path),
                target_tokens=source_target_tokens,
                target_bytes=source_target_bytes,
            )

            if not local_path.is_file():
                # Under --allow-shortfall we treat a missing source as a
                # zero-contribution entry and keep going. Without it, this
                # is a hard error (the strict default).
                if allow_shortfall:
                    contribution.lines_written = 0
                    contribution.bytes_written = 0
                    manifest.contributions.append(contribution)
                    print(f"  warn: missing source {local_path} — skipping (allow-shortfall)")
                    continue
                raise FileNotFoundError(local_path)

            lines_written, bytes_written = _copy_budget(local_path, out_fh, source_target_bytes)
            contribution.lines_written = lines_written
            contribution.bytes_written = bytes_written
            if bytes_written < source_target_bytes and not allow_shortfall:
                raise RuntimeError(
                    f"Source {source['name']} was too small: wrote {bytes_written} bytes, "
                    f"needed {source_target_bytes}."
                )
            manifest.lines_written += lines_written
            manifest.bytes_written += bytes_written
            manifest.contributions.append(contribution)

    manifest.finished_at = now_iso()
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("configs/data_paths.yaml"),
        help="Path to data_paths.yaml.",
    )
    parser.add_argument(
        "--plan",
        type=Path,
        default=Path("configs/data/curated_40b_mix.yaml"),
        help="Mix plan YAML.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO / "data" / "training" / "curated_40b",
        help="Directory for the mixed english/german/code txt files.",
    )
    parser.add_argument(
        "--allow-shortfall",
        action="store_true",
        help="Allow a source to contribute less than its target budget.",
    )
    args = parser.parse_args()

    cfg = load_paths(args.data_config)
    plan = yaml.safe_load(args.plan.read_text(encoding="utf-8"))
    data_root = Path(cfg["_data_root"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifests: list[LanguageManifest] = []
    for language in ("english", "german", "code"):
        manifests.append(
            _mix_language(
                language=language,
                section=plan[language],
                data_root=data_root,
                output_dir=args.output_dir,
                allow_shortfall=args.allow_shortfall,
            )
        )

    manifest_path = args.output_dir / "mix_manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(item) for item in manifests], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
