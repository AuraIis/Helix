#!/usr/bin/env python3
"""Build stable train/holdout manifests for source-disjoint data experiments.

The script does not copy text into a new corpus. It records source, line number,
hash, byte size, and split assignment so downstream builders can assemble train
and validation sets without taking a tail split of a final mixed file.

Examples:
  python scripts/data/build_source_disjoint_manifest.py \
    --source oasst_de data/raw/oasst_de.jsonl \
    --source codeparrot data/training/code_public/codeparrot.txt \
    --out data/manifests/pretrain_v6_manifest.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", str(text or "").replace("\x00", " ")).strip().lower()


def sha256_text(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8", errors="replace")).hexdigest()


def stable_bucket(key: str, mod: int) -> int:
    digest = hashlib.sha256(key.encode("utf-8", errors="replace")).digest()
    return int.from_bytes(digest[:8], "big") % mod


def json_text(obj: dict[str, Any], text_fields: list[str]) -> str:
    parts: list[str] = []
    for field in text_fields:
        value: Any = obj
        for part in field.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break
        if value:
            parts.append(str(value))
    if parts:
        return "\n".join(parts)
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def json_group_key(obj: dict[str, Any], group_fields: list[str], fallback: str) -> str:
    parts: list[str] = []
    for field in group_fields:
        value: Any = obj
        for part in field.split("."):
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break
        if value:
            parts.append(str(value))
    return "|".join(parts) if parts else fallback


def iter_records(
    source_name: str,
    path: Path,
    *,
    text_fields: list[str],
    group_fields: list[str],
) -> Any:
    suffix = path.suffix.lower()
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            obj: dict[str, Any] | None = None
            if suffix == ".jsonl":
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    obj = parsed
            text = json_text(obj, text_fields) if obj is not None else raw
            norm = normalize_text(text)
            if not norm:
                continue
            content_hash = sha256_text(norm)
            fallback_key = f"{source_name}:{content_hash}"
            group_key = json_group_key(obj, group_fields, fallback_key) if obj is not None else fallback_key
            yield {
                "source": source_name,
                "path": str(path),
                "line_no": line_no,
                "bytes": len(raw.encode("utf-8", errors="replace")),
                "sha256": content_hash,
                "group_key": group_key,
            }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        required=True,
        help="Source name and input file. May be repeated.",
    )
    parser.add_argument("--out", required=True, help="Output manifest JSONL path.")
    parser.add_argument("--holdout-mod", type=int, default=1000, help="Hash modulo for split buckets.")
    parser.add_argument(
        "--holdout-buckets",
        default="0,1,2,3,4",
        help="Comma-separated buckets assigned to holdout. Default is 0.5%% with mod=1000.",
    )
    parser.add_argument(
        "--text-field",
        action="append",
        default=[],
        help="JSONL text field to hash, e.g. text, question, answer, content. Repeatable.",
    )
    parser.add_argument(
        "--group-field",
        action="append",
        default=[],
        help="JSONL field used for split grouping, e.g. repo_name, message_tree_id. Repeatable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    holdout_buckets = {int(x) for x in args.holdout_buckets.split(",") if x.strip()}
    if args.holdout_mod <= 0:
        raise SystemExit("--holdout-mod must be > 0")
    if any(bucket < 0 or bucket >= args.holdout_mod for bucket in holdout_buckets):
        raise SystemExit("holdout bucket out of range")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    text_fields = args.text_field or ["text", "content", "question", "answer", "prompt", "completion"]
    group_fields = args.group_field or ["repo_name", "message_tree_id", "id", "title"]

    counts: dict[str, dict[str, int]] = {}
    with out.open("w", encoding="utf-8") as fh:
        for source_name, raw_path in args.source:
            path = Path(raw_path)
            if not path.is_file():
                raise SystemExit(f"missing source file: {path}")
            counts.setdefault(source_name, {"train": 0, "holdout": 0})
            for rec in iter_records(source_name, path, text_fields=text_fields, group_fields=group_fields):
                key = f"{source_name}:{rec['group_key']}"
                bucket = stable_bucket(key, args.holdout_mod)
                split = "holdout" if bucket in holdout_buckets else "train"
                rec["split"] = split
                rec["bucket"] = bucket
                fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                counts[source_name][split] += 1

    print(json.dumps({"out": str(out), "counts": counts}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

