"""Unit tests for the fail-closed safety constitution loader (S0).

The headline guarantee under test: a tampered, missing, empty, or wrong-version
constitution makes the loader *raise* instead of returning — so a self-modifying
model cannot change, freeze, or skip-load its own rules and keep running.

All torch-free, no GPU needed.
"""

import json
import tempfile
import unittest
from pathlib import Path

from auralis.safety import (
    Constitution,
    ConstitutionError,
    DEFAULT_CONSTITUTION_PATH,
    PINNED_SHA256_V1,
    assert_safety_available,
    compute_sha256,
    load_constitution,
)


def _write(tmp: Path, name: str, content: bytes) -> Path:
    p = tmp / name
    p.write_bytes(content)
    return p


class TestRealConstitution(unittest.TestCase):
    """The shipped constitution must load against the in-code pin."""

    def test_default_loads_and_matches_pin(self):
        const = load_constitution()
        self.assertIsInstance(const, Constitution)
        self.assertEqual(const.version, "1")
        self.assertEqual(const.sha256, PINNED_SHA256_V1)

    def test_pin_matches_file_on_disk(self):
        # If this fails, the file was edited without re-pinning the hash.
        self.assertEqual(compute_sha256(DEFAULT_CONSTITUTION_PATH), PINNED_SHA256_V1)

    def test_exposes_hard_no_and_forbidden_actions(self):
        const = load_constitution()
        self.assertEqual(
            const.hard_no_ids(),
            frozenset({"HN-1", "HN-2", "HN-3", "HN-4", "HN-5"}),
        )
        self.assertTrue(const.is_forbidden_action("FA-1"))
        self.assertFalse(const.is_forbidden_action("FA-999"))
        # Hard-No applies in both modes, including owner mode.
        for cat in const.hard_no:
            self.assertIn("owner", cat.refused_in)

    def test_assert_safety_available_returns_constitution(self):
        self.assertIsInstance(assert_safety_available(), Constitution)


class TestFailClosed(unittest.TestCase):
    """Every untrustworthy state must raise, never silently pass."""

    def test_missing_file_raises(self):
        with tempfile.TemporaryDirectory() as d:
            missing = Path(d) / "nope.json"
            with self.assertRaises(ConstitutionError):
                load_constitution(path=missing)

    def test_tampered_byte_breaks_hash(self):
        # The core tamper test: change one byte of the real file -> hash mismatch.
        original = DEFAULT_CONSTITUTION_PATH.read_bytes()
        tampered = original.replace(b"HN-1", b"HN-9", 1)
        self.assertNotEqual(tampered, original)
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), "constitution.v1.json", tampered)
            # Verified against the in-code pin -> must fail closed.
            with self.assertRaises(ConstitutionError):
                load_constitution(path=p)

    def test_wrong_expected_hash_raises(self):
        with self.assertRaises(ConstitutionError):
            load_constitution(expected_sha256="0" * 64)

    def test_malformed_json_raises(self):
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), "c.json", b"{not valid json")
            with self.assertRaises(ConstitutionError):
                load_constitution(path=p, expected_sha256=compute_sha256(p))

    def test_empty_hard_no_raises(self):
        body = json.dumps(
            {
                "schema": "auralis.safety.constitution",
                "version": "1",
                "hard_no": [],
                "forbidden_self_actions": [
                    {"id": "FA-1", "action": "x", "summary": "y"}
                ],
            }
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), "c.json", body)
            with self.assertRaises(ConstitutionError):
                load_constitution(path=p, expected_sha256=compute_sha256(p))

    def test_empty_forbidden_actions_raises(self):
        body = json.dumps(
            {
                "schema": "auralis.safety.constitution",
                "version": "1",
                "hard_no": [{"id": "HN-1", "name": "n", "summary": "s"}],
                "forbidden_self_actions": [],
            }
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), "c.json", body)
            with self.assertRaises(ConstitutionError):
                load_constitution(path=p, expected_sha256=compute_sha256(p))

    def test_wrong_schema_raises(self):
        body = json.dumps(
            {
                "schema": "something.else",
                "version": "1",
                "hard_no": [{"id": "HN-1", "name": "n", "summary": "s"}],
                "forbidden_self_actions": [{"id": "FA-1", "action": "x", "summary": "y"}],
            }
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), "c.json", body)
            with self.assertRaises(ConstitutionError):
                load_constitution(path=p, expected_sha256=compute_sha256(p))

    def test_wrong_version_raises(self):
        body = json.dumps(
            {
                "schema": "auralis.safety.constitution",
                "version": "99",
                "hard_no": [{"id": "HN-1", "name": "n", "summary": "s"}],
                "forbidden_self_actions": [{"id": "FA-1", "action": "x", "summary": "y"}],
            }
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), "c.json", body)
            with self.assertRaises(ConstitutionError):
                load_constitution(path=p, expected_sha256=compute_sha256(p))


class TestWellFormedAlternate(unittest.TestCase):
    """A structurally valid, self-consistently pinned file loads fine."""

    def test_minimal_valid_constitution_loads(self):
        body = json.dumps(
            {
                "schema": "auralis.safety.constitution",
                "version": "1",
                "hard_no": [
                    {"id": "HN-1", "name": "n", "summary": "s", "refused_in": ["default", "owner"]}
                ],
                "forbidden_self_actions": [{"id": "FA-1", "action": "x", "summary": "y"}],
            }
        ).encode("utf-8")
        with tempfile.TemporaryDirectory() as d:
            p = _write(Path(d), "c.json", body)
            const = load_constitution(path=p, expected_sha256=compute_sha256(p))
            self.assertEqual(const.hard_no_ids(), frozenset({"HN-1"}))
            self.assertTrue(const.is_forbidden_action("FA-1"))


if __name__ == "__main__":
    unittest.main()
