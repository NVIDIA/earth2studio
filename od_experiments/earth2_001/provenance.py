import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from od_experiments.earth2_001.security import scrub_dict


def compute_sha256(filepath: str | Path) -> str:
    """Calculates the SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def get_git_info(repo_root: str | Path | None = None) -> dict[str, Any]:
    """Retrieves current Git commit SHA, branch, and dirty status."""
    cwd = repo_root or Path(__file__).resolve().parents[2]
    try:
        commit_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        commit_sha = "unknown"

    try:
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        branch = "unknown"

    try:
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(status_out)
    except Exception:
        dirty = False

    return {
        "commit_sha": commit_sha,
        "branch": branch,
        "dirty": dirty,
    }


def create_provenance_manifest(
    experiment_id: str,
    configuration: dict[str, Any],
    artifacts: list[dict[str, Any]],
    execution_outcome: str = "EXECUTION_SUCCESS",
    validation_status: str = "NOT_VALIDATED",
    classification: str = "SYNTHETIC_SMOKE_TEST",  # "SYNTHETIC_SMOKE_TEST" or "GENUINE_ATMOSPHERIC_EXPERIMENT"
    model_type: str = "BASELINE_PERSISTENCE",  # "BASELINE_PERSISTENCE" or "NEURAL_PROGNOSTIC"
    data_source_type: str = "SYNTHETIC_RANDOM",  # "SYNTHETIC_RANDOM" or "HISTORICAL_GFS"
    verification_reference: str | None = None,
    error_message: str | None = None,
    execution_time_seconds: float | None = None,
    device: str = "cpu",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Generates a comprehensive provenance manifest in JSON format."""
    import torch

    import earth2studio

    manifest = {
        "manifest_version": "2.0.0",
        "experiment_id": experiment_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scientific_integrity": {
            "experiment_classification": classification,
            "model_type": model_type,
            "data_source_type": data_source_type,
            "execution_outcome": execution_outcome,
            "validation_status": validation_status,
            "verification_reference": verification_reference,
        },
        "error_message": error_message,
        "execution_time_seconds": execution_time_seconds,
        "git": get_git_info(),
        "environment": {
            "os": sys.platform,
            "python_version": sys.version.split()[0],
            "torch_version": getattr(torch, "__version__", "unknown"),
            "earth2studio_version": getattr(earth2studio, "__version__", "unknown"),
            "device": device,
        },
        "configuration": scrub_dict(configuration),
        "artifacts": artifacts,
    }

    if output_dir:
        out_path = Path(output_dir) / "provenance_manifest.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return manifest
