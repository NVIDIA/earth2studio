import json
from pathlib import Path

import pytest

from od_experiments.earth2_001.provenance import (
    compute_sha256,
    create_provenance_manifest,
    get_git_info,
)


def test_git_info_structure():
    info = get_git_info()
    assert "commit_sha" in info
    assert "branch" in info
    assert "dirty" in info


def test_provenance_manifest_creation(tmp_path):
    config = {"experiment_id": "TEST-EXP", "token": "secret12345"}
    dummy_file = tmp_path / "test.txt"
    dummy_file.write_text("Hello Origin Datum")
    digest = compute_sha256(dummy_file)

    artifacts = [
        {"path": str(dummy_file), "sha256": digest, "bytes": dummy_file.stat().st_size}
    ]

    manifest = create_provenance_manifest(
        experiment_id="TEST-EXP",
        configuration=config,
        artifacts=artifacts,
        execution_outcome="EXECUTION_SUCCESS",
        validation_status="NOT_VALIDATED",
        classification="SYNTHETIC_SMOKE_TEST",
        output_dir=tmp_path,
    )

    assert manifest["experiment_id"] == "TEST-EXP"
    assert manifest["scientific_integrity"]["execution_outcome"] == "EXECUTION_SUCCESS"
    assert manifest["scientific_integrity"]["experiment_classification"] == "SYNTHETIC_SMOKE_TEST"
    assert manifest["configuration"]["token"] == "[REDACTED]"
    assert (tmp_path / "provenance_manifest.json").exists()

    saved_manifest = json.loads((tmp_path / "provenance_manifest.json").read_text())
    assert saved_manifest["artifacts"][0]["sha256"] == digest
