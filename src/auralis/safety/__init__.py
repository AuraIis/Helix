"""Auralis safety layer.

S0 — the immutable safety constitution and its fail-closed loader. This is the
hard shell of the self-learning safety design (``docs/SELF_LEARNING_SAFETY_LAYER.md``):
the rules live in a hash-pinned JSON file *outside* the learnable weights, so a
self-modifying model cannot change, freeze, or skip-load them.

Decision/integrity logic here is pure-Python and torch-free, on purpose: it must
be testable without a GPU and must not depend on a heavy import that could fail
and take the safety check down with it.
"""

from __future__ import annotations

from .constitution import (
    Constitution,
    ConstitutionError,
    DEFAULT_CONSTITUTION_PATH,
    EXPECTED_SCHEMA,
    EXPECTED_VERSION,
    ForbiddenAction,
    HardNoCategory,
    PINNED_SHA256_V1,
    assert_safety_available,
    compute_sha256,
    load_constitution,
)

__all__ = [
    "Constitution",
    "ConstitutionError",
    "DEFAULT_CONSTITUTION_PATH",
    "EXPECTED_SCHEMA",
    "EXPECTED_VERSION",
    "ForbiddenAction",
    "HardNoCategory",
    "PINNED_SHA256_V1",
    "assert_safety_available",
    "compute_sha256",
    "load_constitution",
]
