# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np


class StorageType(str, Enum):
    """Storage type for inference results."""

    SERVER = "server"
    S3 = "s3"


class RequestStatus(Enum):
    """Status values for inference requests."""

    # Initial statuses
    ACCEPTED = "accepted"
    QUEUED = "queued"

    # Processing statuses
    RUNNING = "running"

    # Final statuses
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PENDING_RESULTS = "pending_results"


@dataclass
class InferenceRequest:
    """Main inference request payload."""

    parameters: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary suitable for JSON request body (e.g. {"parameters": ...}).
        """
        # use custom JSON encoder first to convert objects that are normally not serializable
        parameters = json.loads(
            json.dumps(self.parameters, default=InferenceRequest.json_serial)
        )
        return {"parameters": parameters}

    @staticmethod
    def json_serial(obj: Any) -> Any:
        """
        JSON serializer for objects not serializable by default json code.

        Parameters
        ----------
        obj : Any
            Object to serialize (numpy array, datetime, timedelta, etc.).

        Returns
        -------
        Any
            JSON-serializable representation of the object.
        """
        if isinstance(obj, np.ndarray):
            return list(obj)
        elif isinstance(obj, (datetime, date)):
            return obj.isoformat()
        elif isinstance(obj, timedelta):
            return obj.total_seconds()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.datetime64):
            return str(obj)
        elif isinstance(obj, np.timedelta64):
            return obj / np.timedelta64(1, "s")  # seconds as float
        else:
            raise TypeError(
                f"Object of type {type(obj).__name__} is not JSON serializable"
            )


@dataclass
class InferenceRequestResponse:
    """Response when submitting an inference request."""

    execution_id: str
    status: RequestStatus
    message: str
    timestamp: datetime

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceRequestResponse":
        """
        Build an InferenceRequestResponse from an API response dict.

        Parameters
        ----------
        data : dict[str, Any]
            API response dictionary containing execution_id, status, message, timestamp.

        Returns
        -------
        InferenceRequestResponse
            Constructed response instance.
        """
        return cls(
            execution_id=data["execution_id"],
            status=RequestStatus(data["status"]),
            message=data["message"],
            timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")),
        )


@dataclass
class ProgressInfo:
    """Progress information for inference requests."""

    progress: str
    current_step: int
    total_steps: int


@dataclass
class InferenceRequestStatus:
    """Status information for an inference request."""

    execution_id: str
    status: RequestStatus
    progress: ProgressInfo | None
    error_message: str | None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceRequestStatus":
        """
        Build an InferenceRequestStatus from an API response dict.

        Parameters
        ----------
        data : dict[str, Any]
            API response dictionary containing execution_id, status, progress, error_message.

        Returns
        -------
        InferenceRequestStatus
            Constructed status instance.
        """
        progress_data = data.get("progress")
        progress = (
            ProgressInfo(
                progress=progress_data["progress"],
                current_step=progress_data["current_step"],
                total_steps=progress_data["total_steps"],
            )
            if progress_data is not None
            else None
        )

        return cls(
            execution_id=data["execution_id"],
            status=RequestStatus(data["status"]),
            progress=progress,
            error_message=data.get("error_message"),
        )


@dataclass
class OutputFile:
    """Information about an output file."""

    path: str
    size: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OutputFile":
        """
        Build an OutputFile from an API response dict.

        Parameters
        ----------
        data : dict[str, Any]
            API response dictionary containing path and size.

        Returns
        -------
        OutputFile
            Constructed output file instance.
        """
        return cls(path=data["path"], size=data["size"])


@dataclass
class InferenceRequestResults:
    """Results of a completed inference request."""

    request_id: str
    status: RequestStatus
    output_files: list[OutputFile]
    completion_time: datetime | None
    execution_time_seconds: float | None = None
    storage_type: StorageType = StorageType.SERVER
    signed_url: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceRequestResults":
        """
        Build InferenceRequestResults from an API response dict.

        Parameters
        ----------
        data : dict[str, Any]
            API response dictionary containing request_id, status, output_files,
            completion_time, and optional execution_time_seconds, storage_type, signed_url.

        Returns
        -------
        InferenceRequestResults
            Constructed results instance.
        """
        output_files = [
            OutputFile.from_dict(file_data) for file_data in data["output_files"]
        ]
        completion_timestamp = data.get("completion_time")
        completion_time = (
            datetime.fromisoformat(completion_timestamp.replace("Z", "+00:00"))
            if completion_timestamp is not None
            else None
        )

        return cls(
            request_id=data["request_id"],
            status=RequestStatus(data["status"]),
            output_files=output_files,
            completion_time=completion_time,
            execution_time_seconds=data.get("execution_time_seconds"),
            storage_type=StorageType(
                data.get("storage_type", StorageType.SERVER.value)
            ),
            signed_url=data.get("signed_url"),
        )

    def result_paths(self) -> list[str]:
        """
        Get paths to result data files found in the request response.

        Returns
        -------
        list[str]
            List of result paths (e.g. zarr roots or .nc files). Append to the
            server results URL to download or open.
        """
        # locate the root of zarr store
        zarr_paths = {
            f.path
            for f in self.output_files
            if (".zarr/" in f.path) or f.path.endswith(".zarr")
        }
        zarr_paths_sorted = sorted(
            {path[: path.find(".zarr") + len(".zarr")] for path in zarr_paths}
        )
        netcdf_paths = [f.path for f in self.output_files if f.path.endswith(".nc")]
        return zarr_paths_sorted + netcdf_paths


@dataclass
class HealthStatus:
    """Health status response."""

    status: str
    timestamp: datetime

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HealthStatus":
        """
        Build a HealthStatus from an API response dict.

        Parameters
        ----------
        data : dict[str, Any]
            API response dictionary containing status and timestamp.

        Returns
        -------
        HealthStatus
            Constructed health status instance.
        """
        return cls(
            status=data["status"],
            timestamp=datetime.fromisoformat(data["timestamp"].replace("Z", "+00:00")),
        )
