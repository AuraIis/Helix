#!/usr/bin/env python3
"""Stable mixed repair data for the response-fix SFT series.

v7 showed that a tiny repair-only set can improve keyword probes while harming
semantic polarity. This builder mixes the broad v5 family-balanced set with the
v6 bridge and v7 Bonn/photosynthesis repairs, then tags records by source family
so the trainer can sample across families instead of overfitting one patch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = [
    REPO / "data/training/sft_response_fix_de_v5/core_train.helix.jsonl",
    REPO / "data/training/sft_response_fix_de_v6_bridge_patch/core_train.helix.jsonl",
    REPO / "data/training/sft_response_fix_de_v7_bonn_photo_patch/core_train.helix.jsonl",
]
DEFAULT_VAL_INPUTS = [
    REPO / "data/training/sft_response_fix_de_v5/val.helix.jsonl",
    REPO / "data/training/sft_response_fix_de_v6_bridge_patch/val.helix.jsonl",
    REPO / "data/training/sft_response_fix_de_v7_bonn_photo_patch/val.helix.jsonl",
]


def norm_key(text: str) -> str:
    return hashlib.blake2b(re.sub(r"\s+", " ", text.lower()).encode("utf-8"), digest_size=16).hexdigest()


def load_jsonl(path: Path, source_prefix: str) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            item = json.loads(line)
            src = item.get("source", source_prefix)
            family = item.get("family") or item.get("block") or item.get("category") or "unknown"
            item["source"] = src
            item["family"] = f"{source_prefix}:{family}"
            item["block"] = item.get("block") or family
            item["quality_variant"] = "sft_response_fix_de_v8_stable_mix"
            rows.append(item)
    return rows


def dedupe(items: list[dict]) -> list[dict]:
    seen: set[str] = set()
    out: list[dict] = []
    for item in items:
        k = norm_key(item["text"])
        if k in seen:
            continue
        seen.add(k)
        out.append(item)
    return out


def write_jsonl(path: Path, items: list[dict]) -> int:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for item in items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(items)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=REPO / "data/training/sft_response_fix_de_v8_stable_mix")
    ap.add_argument("--seed", type=int, default=20260528)
    args = ap.parse_args()

    train: list[dict] = []
    for path in DEFAULT_INPUTS:
        train.extend(load_jsonl(path, path.parent.name))
    val: list[dict] = []
    for path in DEFAULT_VAL_INPUTS:
        val.extend(load_jsonl(path, path.parent.name))

    train = dedupe(train)
    val = dedupe(val)
    random.Random(args.seed).shuffle(train)
    random.Random(args.seed + 1).shuffle(val)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_n = write_jsonl(args.output_dir / "core_train.helix.jsonl", train)
    val_n = write_jsonl(args.output_dir / "val.helix.jsonl", val)
    manifest = {
        "variant": "sft_response_fix_de_v8_stable_mix",
        "goal": "Stabilize v6 with broad v5 data plus v6/v7 targeted repairs.",
        "train_records": train_n,
        "val_records": val_n,
        "train_categories": dict(Counter(x.get("category", "unknown") for x in train).most_common()),
        "train_families": len(set(x.get("family", "unknown") for x in train)),
        "inputs": [str(p.relative_to(REPO)) for p in DEFAULT_INPUTS],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
