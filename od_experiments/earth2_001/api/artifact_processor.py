"""Artifact processing and provenance hashing for NVIDIA-hosted FourCastNet responses.

Safely handles:
- JSON metadata responses
- Binary TAR archives containing .npy arrays
- Path traversal prevention for archive extractions
- SHA-256 calculation for all artifacts
- Provenance manifest enrichment
"""

import hashlib
import io
import json
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from od_experiments.earth2_001.api.fourcastnet_hosted import ForecastRequest, ForecastResponse


@dataclass
class ProcessedArtifact:
    """Metadata regarding a saved forecast artifact."""

    filename: str
    filepath: str
    sha256: str
    size_bytes: int
    artifact_type: str  # 'json_response', 'numpy_array', 'tar_archive'
    metadata: Dict[str, Any]


class ArtifactSecurityError(Exception):
    """Raised when an artifact violates security boundaries."""


def calculate_sha256(data: bytes) -> str:
    """Calculate deterministic SHA-256 hex digest."""
    return hashlib.sha256(data).hexdigest()


def calculate_file_sha256(filepath: Path) -> str:
    """Calculate SHA-256 hex digest for a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


class ForecastArtifactProcessor:
    """Validates and persists responses from FourCastNet hosted API."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_response(
        self,
        request: ForecastRequest,
        response: ForecastResponse,
        test_id: str = "T01",
    ) -> List[ProcessedArtifact]:
        """Process response bytes, validate format, and write artifacts safely."""
        artifacts: List[ProcessedArtifact] = []

        # 1. Always record the verified response payload
        resp_json_name = f"{test_id.lower()}_response.json"
        resp_json_path = self.output_dir / resp_json_name

        payload_to_write = {
            "test_id": test_id,
            "status_code": response.status_code,
            "req_id": response.req_id,
            "nvcf_status": response.nvcf_status,
            "content_type": response.content_type,
            "duration_seconds": round(response.duration_seconds, 4),
            "endpoint": response.endpoint,
            "request_parameters": request.to_dict(),
            "raw_body_preview": response.raw_data.decode("utf-8", errors="replace")[:1000],
            "parsed_json": response.json_payload,
        }

        json_bytes = json.dumps(payload_to_write, indent=2).encode("utf-8")
        resp_json_path.write_bytes(json_bytes)
        artifacts.append(
            ProcessedArtifact(
                filename=resp_json_name,
                filepath=str(resp_json_path),
                sha256=calculate_sha256(json_bytes),
                size_bytes=len(json_bytes),
                artifact_type="json_response",
                metadata={"test_id": test_id, "nvcf_reqid": response.req_id},
            )
        )

        # 2. If the response is a TAR archive, inspect and extract securely
        if "tar" in response.content_type or (response.raw_data[:2] == b"\x1f\x8b" or b"ustar" in response.raw_data[:512]):
            try:
                tar_bytes = io.BytesIO(response.raw_data)
                with tarfile.open(fileobj=tar_bytes, mode="r:*") as tar:
                    for member in tar.getmembers():
                        # Security check: Prevent path traversal (Zip Slip / Tar Slip)
                        norm_name = os.path.normpath(member.name)
                        if norm_name.startswith("/") or ".." in norm_name.split(os.path.sep):
                            raise ArtifactSecurityError(f"Unsafe TAR member path detected: {member.name}")

                        f = tar.extractfile(member)
                        if f is not None:
                            member_bytes = f.read()
                            target_file = self.output_dir / f"{test_id.lower()}_{os.path.basename(member.name)}"
                            target_file.write_bytes(member_bytes)

                            meta: Dict[str, Any] = {"tar_member": member.name}
                            if target_file.suffix == ".npy":
                                try:
                                    arr = np.load(str(target_file))
                                    meta["array_shape"] = list(arr.shape)
                                    meta["dtype"] = str(arr.dtype)
                                    meta["has_nan"] = bool(np.isnan(arr).any())
                                except Exception as e:
                                    logger.warning(f"Could not inspect .npy array: {e}")

                            artifacts.append(
                                ProcessedArtifact(
                                    filename=target_file.name,
                                    filepath=str(target_file),
                                    sha256=calculate_sha256(member_bytes),
                                    size_bytes=len(member_bytes),
                                    artifact_type="numpy_array" if target_file.suffix == ".npy" else "tar_extracted",
                                    metadata=meta,
                                )
                            )
            except tarfile.TarError as e:
                logger.info(f"Response data is not a valid tar archive: {e}")

        return artifacts
