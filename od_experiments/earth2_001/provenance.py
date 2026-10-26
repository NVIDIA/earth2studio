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


def get_git_info(repo_root: Optional[str | Path] = None) -> Dict[str, Any]:
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
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        branch = "unknown"

    try:
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=cwd, text=True, stderr=subprocess.DEVNULL
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
    configuration: Dict[str, Any],
    artifacts: List[Dict[str, Any]],
    outcome: str = "SUCCESS",
    error_message: Optional[str] = None,
    execution_time_seconds: Optional[float] = None,
    device: str = "cpu",
    output_dir: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Generates a comprehensive provenance manifest in JSON format."""
    import earth2studio
    import torch

    manifest = {
        "manifest_version": "1.0.0",
        "experiment_id": experiment_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "outcome": outcome,
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
