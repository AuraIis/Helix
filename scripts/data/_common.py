"""Shared helpers for data download / preparation scripts.

Every download script uses the same helpers to:

- resolve paths against ``configs/data_paths.yaml`` (no hardcoded paths),
- write atomically (``*.tmp`` → rename), so a killed download never leaves a
  partially-written file that looks complete to later steps,
- emit a JSON manifest next to each output describing filters used and the
  final doc/byte counts.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Windows default stdout is cp1252 and cannot print non-Latin1 glyphs
# (arrows, tqdm block chars, German umlauts on some systems). Reconfigure
# once at import time so every data script is safe.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass

# Repo root = two levels up from this file (scripts/data/_common.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "data_paths.yaml"


def load_paths(config_path: Path | str = DEFAULT_CONFIG) -> dict[str, Any]:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root = Path(cfg["data_root"])
    cfg["_data_root"] = root
    return cfg


def resolve(cfg: dict[str, Any], *keys: str) -> Path:
    """Join ``data_root`` with a nested key path from the config.

    Example: ``resolve(cfg, "raw", "english")`` → ``<data_root>/raw/english``.
    """
    node: Any = cfg
    for k in keys:
        node = node[k]
    return Path(cfg["_data_root"]) / node


@dataclass
class DownloadStats:
    source: str
    output_file: str
    target_tokens: int
    estimated_bytes_per_token: float
    final_docs: int = 0
    final_bytes: int = 0
    filtered_total: int = 0
    filtered_reasons: dict[str, int] = field(default_factory=dict)
    started_at: str = ""
    finished_at: str = ""
    filters_applied: dict[str, Any] = field(default_factory=dict)

    def target_bytes(self) -> int:
        return int(self.target_tokens * self.estimated_bytes_per_token)

    def write_manifest(self, manifest_path: Path) -> None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(asdict(self), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


@contextmanager
def atomic_text_writer(path: Path, encoding: str = "utf-8"):
    """Write to ``path`` atomically via a sibling ``.tmp`` file.

    The temp file is renamed over the target only on clean exit. If the caller
    throws, the temp file is removed so the next run can start fresh.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    f = tmp.open("w", encoding=encoding, newline="\n")
    try:
        yield f
        f.close()
        # os.replace is atomic on same filesystem, including over existing file
        os.replace(tmp, path)
    except BaseException:
        try:
            f.close()
        finally:
            if tmp.exists():
                tmp.unlink()
        raise


def clean_text(text: str) -> str:
    """Normalise whitespace for pretraining corpus lines."""
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def check_free_space(path: Path, required_gb: float) -> None:
    """Abort early if the target disk does not have enough free space."""
    usage = shutil.disk_usage(path if path.exists() else path.parent)
    free_gb = usage.free / 1024**3
    if free_gb < required_gb:
        sys.exit(
            f"Not enough free space at {path}: "
            f"{free_gb:.1f} GB free, need {required_gb:.1f} GB. "
            "Free up space or switch data_root."
        )
