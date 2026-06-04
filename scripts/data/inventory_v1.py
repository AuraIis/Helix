"""Inventory of reusable Auralis v1 data on ``I:/Auralis/NEWGPT/data``.

For every ``*.jsonl`` / ``*.txt`` below ``v1_data_root`` we record:

- relative path
- byte size
- sample count (line count for jsonl; approximate)
- detected fields on the first line (for jsonl)
- first two samples (truncated) for a quick eyeball

Output: ``data/eval/v1_inventory.json`` (in-repo, small, checked in).

Use this as an input to the Phase 3 SFT mixture planner. v1 data quality
ratings (from the user) are kept in ``LESSONS.md`` / ``project_v1_datasets``
memory — do not re-derive them here.

Run::

    python scripts/data/inventory_v1.py --limit 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.data._common import load_paths


def _peek_jsonl(path: Path, max_samples: int = 2) -> dict[str, Any]:
    n_lines = 0
    first_fields: list[str] = []
    samples: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for i, raw in enumerate(fh):
                n_lines += 1
                if i == 0:
                    try:
                        obj = json.loads(raw)
                        if isinstance(obj, dict):
                            first_fields = sorted(obj.keys())
                    except json.JSONDecodeError:
                        pass
                if i < max_samples:
                    samples.append(raw.strip()[:300])
    except OSError as e:
        return {"error": str(e), "lines": 0, "fields": [], "samples": []}
    return {"lines": n_lines, "fields": first_fields, "samples": samples}


def _peek_text(path: Path, max_samples: int = 2) -> dict[str, Any]:
    n_lines = 0
    samples: list[str] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for i, raw in enumerate(fh):
                n_lines += 1
                if i < max_samples:
                    samples.append(raw.strip()[:300])
    except OSError as e:
        return {"error": str(e), "lines": 0, "samples": []}
    return {"lines": n_lines, "samples": samples}


def inventory(root: Path, limit: int | None) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    checked = 0
    for pattern in ("**/*.jsonl", "**/*.txt"):
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            if limit is not None and checked >= limit:
                break
            rel = path.relative_to(root).as_posix()
            size_mb = path.stat().st_size / 1024**2
            entry: dict[str, Any] = {
                "path": rel,
                "size_mb": round(size_mb, 2),
                "kind": path.suffix.lstrip("."),
            }
            if size_mb < 500:
                if path.suffix == ".jsonl":
                    entry.update(_peek_jsonl(path))
                else:
                    entry.update(_peek_text(path))
            else:
                entry["note"] = "skipped deep peek (file >500MB)"
            files.append(entry)
            checked += 1

    totals = {
        "files": len(files),
        "total_size_gb": round(sum(f["size_mb"] for f in files) / 1024, 2),
        "total_lines": sum(f.get("lines", 0) for f in files),
    }
    return {"root": str(root), "totals": totals, "files": files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory reusable Auralis v1 data.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "eval" / "v1_inventory.json")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max files to scan (useful for quick runs).")
    args = parser.parse_args()

    cfg = load_paths(args.config) if args.config else load_paths()
    v1_root = Path(cfg["v1_data_root"])
    if not v1_root.exists():
        sys.exit(f"v1_data_root does not exist: {v1_root}")

    report = inventory(v1_root, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"  files={report['totals']['files']}  "
          f"size={report['totals']['total_size_gb']} GB  "
          f"lines={report['totals']['total_lines']:,}")


if __name__ == "__main__":
    main()
