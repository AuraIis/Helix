"""Fail-closed loader for the immutable safety constitution.

This is **S0** of the self-learning safety layer (``docs/SELF_LEARNING_SAFETY_LAYER.md``):
the hard shell that must stand *before* the self-learning loop is ever armed.

The constitution (``configs/safety/constitution.v1.json``) is the LAW. This loader
guarantees the three properties the design demands of it, so a self-modifying
model can neither change, freeze, nor skip-load its own rules:

- **mandatory-load / fail-closed** — every entry point calls :func:`load_constitution`
  (or :func:`assert_safety_available`) and *that call raises* if the file is
  missing, malformed, empty, or tampered. No rules loaded => the caller refuses
  to run. There is deliberately no "continue without a constitution" path.
- **integrity / hash-pinning** — the SHA-256 of the file bytes is pinned in code
  (:data:`PINNED_SHA256_V1`). Any byte change flips the hash and the load fails.
  The model cannot edit the JSON without breaking this pin.
- **owner-only change** — re-pinning requires editing this constant by hand, which
  is the deliberate versioning ritual from ``AURALIS_SAFETY_POLICY.md`` §7. The
  model has no write access to either this file or the JSON.

Everything here is torch-free and stdlib-only so it is unit-testable without a GPU
and has no heavy import that could fail and accidentally take the safety layer
down with it. Fail-closed means: when in doubt, raise.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConstitutionError(Exception):
    """Raised whenever the constitution cannot be trusted.

    Any instance of this exception is a *fail-closed* signal: the caller must
    refuse to proceed (no inference, no learning, no actions). It is never
    something to catch-and-continue inside the safety path.
    """


# SHA-256 of ``configs/safety/constitution.v1.json`` (raw file bytes).
# Re-pin ONLY together with a deliberate, owner-reviewed edit of that file
# (AURALIS_SAFETY_POLICY.md §7). To recompute after an intended change:
#     python -c "from auralis.safety.constitution import compute_sha256, \
#         DEFAULT_CONSTITUTION_PATH as p; print(compute_sha256(p))"
PINNED_SHA256_V1 = "6e99e2b0db719dd71d47e78aa52821f2a497fa6c241a5e03f95bfe9d0dd8d13a"

EXPECTED_SCHEMA = "auralis.safety.constitution"
EXPECTED_VERSION = "1"

# repo_root/configs/safety/constitution.v1.json, resolved from this file's
# location (src/auralis/safety/constitution.py -> parents[3] is the repo root).
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONSTITUTION_PATH = _REPO_ROOT / "configs" / "safety" / "constitution.v1.json"


def compute_sha256(path: str | Path) -> str:
    """Return the hex SHA-256 of the raw bytes of ``path``.

    Hashing the raw bytes (not the parsed JSON) is deliberate: *any* change —
    whitespace, key order, a flipped flag — changes the hash, which is exactly
    the tamper-evidence we want.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class HardNoCategory:
    """One non-negotiable refusal category (HN-1..HN-5). Refused in both modes."""

    id: str
    name: str
    summary: str
    refused_in: tuple[str, ...]


@dataclass(frozen=True)
class ForbiddenAction:
    """One self-action the system may never take (FA-1..FA-6)."""

    id: str
    action: str
    summary: str


@dataclass(frozen=True)
class Constitution:
    """A loaded, integrity-verified constitution.

    Construct it only through :func:`load_constitution` — a hand-built instance
    has not passed the fail-closed checks and must not be trusted.
    """

    version: str
    sha256: str
    hard_no: tuple[HardNoCategory, ...]
    forbidden_actions: tuple[ForbiddenAction, ...]
    raw: dict[str, Any]

    def hard_no_ids(self) -> frozenset[str]:
        return frozenset(c.id for c in self.hard_no)

    def forbidden_action_ids(self) -> frozenset[str]:
        return frozenset(a.id for a in self.forbidden_actions)

    def is_forbidden_action(self, action_id: str) -> bool:
        """True if ``action_id`` is on the forbidden self-action list.

        Note: this only answers membership for a *known* id. The runtime
        action-gate (S1) is default-deny over the whole capability surface;
        this helper is a building block for it, not the gate itself.
        """
        return action_id in self.forbidden_action_ids()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConstitutionError(message)


def load_constitution(
    path: str | Path | None = None,
    expected_sha256: str | None = None,
) -> Constitution:
    """Load and verify the constitution, fail-closed.

    Raises :class:`ConstitutionError` (and loads nothing) if the file is
    missing, unreadable, not valid JSON, the wrong schema/version, structurally
    incomplete, empty of rules, or its hash does not match the pin. A successful
    return is the *only* signal that the safety law is present and intact.

    Args:
        path: constitution file; defaults to :data:`DEFAULT_CONSTITUTION_PATH`.
        expected_sha256: hash to verify against; defaults to the in-code pin
            :data:`PINNED_SHA256_V1`. Pass an explicit value only in tests or
            when intentionally loading a different pinned version.
    """
    path = Path(path) if path is not None else DEFAULT_CONSTITUTION_PATH
    expected = expected_sha256 if expected_sha256 is not None else PINNED_SHA256_V1

    # 1) mandatory-load: the file must exist and be readable.
    try:
        raw_bytes = path.read_bytes()
    except FileNotFoundError as exc:
        raise ConstitutionError(
            f"safety constitution missing at {path} — refusing to run (fail-closed)"
        ) from exc
    except OSError as exc:
        raise ConstitutionError(
            f"safety constitution at {path} could not be read: {exc} (fail-closed)"
        ) from exc

    # 2) integrity: hash the bytes before trusting any content.
    digest = hashlib.sha256(raw_bytes).hexdigest()
    _require(
        digest == expected,
        f"safety constitution hash mismatch at {path}: "
        f"expected {expected}, got {digest} — tampered or wrong version "
        f"(fail-closed)",
    )

    # 3) parse only after the hash checks out.
    try:
        data = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConstitutionError(
            f"safety constitution at {path} is not valid UTF-8 JSON: {exc} "
            f"(fail-closed)"
        ) from exc
    _require(isinstance(data, dict), "constitution root must be a JSON object")

    # 4) schema / version must be exactly what this loader understands.
    _require(
        data.get("schema") == EXPECTED_SCHEMA,
        f"unexpected constitution schema {data.get('schema')!r} "
        f"(want {EXPECTED_SCHEMA!r})",
    )
    _require(
        data.get("version") == EXPECTED_VERSION,
        f"unexpected constitution version {data.get('version')!r} "
        f"(want {EXPECTED_VERSION!r})",
    )

    # 5) structural completeness — a constitution with no Hard-No rules is
    #    meaningless and must be refused, not silently accepted as "all allowed".
    hard_no = _parse_hard_no(data.get("hard_no"))
    forbidden = _parse_forbidden_actions(data.get("forbidden_self_actions"))
    _require(len(hard_no) > 0, "constitution has no hard_no categories (empty law)")
    _require(
        len(forbidden) > 0,
        "constitution has no forbidden_self_actions (empty action law)",
    )

    return Constitution(
        version=EXPECTED_VERSION,
        sha256=digest,
        hard_no=hard_no,
        forbidden_actions=forbidden,
        raw=data,
    )


def _parse_hard_no(value: Any) -> tuple[HardNoCategory, ...]:
    _require(isinstance(value, list), "hard_no must be a list")
    out: list[HardNoCategory] = []
    for i, item in enumerate(value):
        _require(isinstance(item, dict), f"hard_no[{i}] must be an object")
        for key in ("id", "name", "summary"):
            _require(
                isinstance(item.get(key), str) and item[key],
                f"hard_no[{i}].{key} must be a non-empty string",
            )
        refused = item.get("refused_in", ["default", "owner"])
        _require(
            isinstance(refused, list) and all(isinstance(x, str) for x in refused),
            f"hard_no[{i}].refused_in must be a list of strings",
        )
        out.append(
            HardNoCategory(
                id=item["id"],
                name=item["name"],
                summary=item["summary"],
                refused_in=tuple(refused),
            )
        )
    return tuple(out)


def _parse_forbidden_actions(value: Any) -> tuple[ForbiddenAction, ...]:
    _require(isinstance(value, list), "forbidden_self_actions must be a list")
    out: list[ForbiddenAction] = []
    for i, item in enumerate(value):
        _require(isinstance(item, dict), f"forbidden_self_actions[{i}] must be an object")
        for key in ("id", "action", "summary"):
            _require(
                isinstance(item.get(key), str) and item[key],
                f"forbidden_self_actions[{i}].{key} must be a non-empty string",
            )
        out.append(
            ForbiddenAction(
                id=item["id"], action=item["action"], summary=item["summary"]
            )
        )
    return tuple(out)


def assert_safety_available(
    path: str | Path | None = None,
    expected_sha256: str | None = None,
) -> Constitution:
    """Fail-closed entry point: return the verified constitution or raise.

    Call this once at the start of any inference, learning, or action runtime.
    Its raising *is* the kill-switch: if the safety law is not present and
    intact, nothing downstream should run.
    """
    return load_constitution(path=path, expected_sha256=expected_sha256)


__all__ = [
    "ConstitutionError",
    "PINNED_SHA256_V1",
    "DEFAULT_CONSTITUTION_PATH",
    "EXPECTED_SCHEMA",
    "EXPECTED_VERSION",
    "compute_sha256",
    "HardNoCategory",
    "ForbiddenAction",
    "Constitution",
    "load_constitution",
    "assert_safety_available",
]
