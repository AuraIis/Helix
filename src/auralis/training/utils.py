"""Small helpers for training scripts.

Kept deliberately lean — anything that belongs on Trainer or Dataset lives
there. This module just has "glue" (YAML loading, seeding, preflight).
"""

from __future__ import annotations

import random
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def resolve_gradient_checkpointing(model: Any, training_cfg: dict[str, Any]) -> bool:
    """Resolve the effective gradient-checkpointing flag for a run.

    ``training.gradient_checkpointing`` is treated as an explicit override
    when present, including ``false``. If the training config omits the key,
    we fall back to the model config default.
    """
    if "gradient_checkpointing" in training_cfg:
        return bool(training_cfg["gradient_checkpointing"])
    advanced = getattr(getattr(model, "config", None), "advanced", None)
    return bool(getattr(advanced, "gradient_checkpointing", False))


def apply_gradient_checkpointing(model: Any, enabled: bool) -> None:
    """Apply the resolved checkpointing mode to the model explicitly."""
    if enabled:
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        return
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()


def load_yaml(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def set_seed(seed: int) -> None:
    """Seed Python, numpy and torch (CPU + CUDA) deterministically."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def preflight_check(
    *,
    data_dir: Path,
    required_data_files: list[str],
    checkpoint_dir: Path,
    required_free_gb: float,
    require_cuda: bool,
) -> None:
    """Abort with a clear message if the environment is not ready.

    Verifies:
      - all required tokenized *.bin files exist under ``data_dir``
      - ``checkpoint_dir`` is creatable + has at least ``required_free_gb`` free
      - a CUDA device is available when ``require_cuda`` is True
    """
    errs: list[str] = []

    for name in required_data_files:
        p = data_dir / name
        if not p.is_file():
            errs.append(f"missing data file: {p}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(checkpoint_dir).free / 1024**3
    if free_gb < required_free_gb:
        errs.append(f"{checkpoint_dir} has {free_gb:.1f} GB free, need {required_free_gb:.1f}")

    if require_cuda and not torch.cuda.is_available():
        errs.append("CUDA device not available (require_cuda=True)")

    if errs:
        print("\n=== PREFLIGHT FAILED ===", file=sys.stderr)
        for e in errs:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)


__all__ = [
    "apply_gradient_checkpointing",
    "load_yaml",
    "preflight_check",
    "resolve_gradient_checkpointing",
    "set_seed",
]
