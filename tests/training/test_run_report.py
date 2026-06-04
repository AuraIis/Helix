"""Run-report (MANIFEST.yaml) write/merge tests."""

from __future__ import annotations

from pathlib import Path

import yaml

from auralis.training.run_report import write_end_manifest, write_start_manifest
from auralis.training.trainer import RunMetadata, TrainerState


def test_start_then_end_manifest_merges_fields(tmp_path: Path):
    path = tmp_path / "MANIFEST.yaml"
    md = RunMetadata(
        git_sha="abc123",
        config_sha16="1234567890abcdef",
        hostname="testhost",
        torch_version="2.7.0",
        dtype="bf16",
    )
    write_start_manifest(
        path=path,
        config={"training": {"total_steps": 100}},
        metadata=md,
        backend_summary={"summary": {"mamba:native": 6}, "per_layer": []},
    )
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert doc["status"] == "started"
    assert doc["metadata"]["git_sha"] == "abc123"

    state = TrainerState(step=42, best_val_loss=1.5, tokens_seen=123)
    write_end_manifest(
        path=path,
        state=state,
        exit_reason="completed",
        health_summary={"n_alerts": 2},
    )
    doc2 = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert doc2["status"] == "finished"
    assert doc2["exit_reason"] == "completed"
    # Start-manifest fields are preserved
    assert doc2["metadata"]["git_sha"] == "abc123"
    assert doc2["final_state"]["step"] == 42
    assert doc2["health"]["n_alerts"] == 2


def test_end_manifest_works_without_start(tmp_path: Path):
    """If the trainer crashes before writing start-manifest, end-write must
    still produce a usable doc."""
    path = tmp_path / "MANIFEST.yaml"
    state = TrainerState(step=5)
    write_end_manifest(path=path, state=state, exit_reason="crashed_early")
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert doc["status"] == "finished"
    assert doc["exit_reason"] == "crashed_early"
