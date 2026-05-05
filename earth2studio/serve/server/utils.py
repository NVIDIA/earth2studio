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

import asyncio
import logging
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import aiofiles  # type: ignore[import-untyped]
from fastapi import HTTPException

from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)

try:
    import redis  # type: ignore[import-untyped]
    from rq import Queue
except ImportError:
    OptionalDependencyFailure("serve")
    redis = None
    Queue = None

from earth2studio.serve.server.config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# Redis Key Functions
# =============================================================================


def get_inference_request_zip_key(request_id: str) -> str:
    """Get Redis key for storing zip file path for a request."""
    return f"inference_request_zips:{request_id}:zip_file"


def get_inference_request_output_path_key(request_id: str) -> str:
    """Get Redis key for storing output path for a request."""
    return f"inference_request:{request_id}:output_path"


def get_inference_request_metadata_key(request_id: str) -> str:
    """Get Redis key for storing pending metadata (before object storage upload)."""
    return f"inference_request:{request_id}:pending_metadata"


def get_results_zip_dir_key(request_id: str) -> str:
    """Get Redis key for storing results zip directory path."""
    return f"inference_request:{request_id}:results_zip_dir"


def get_signed_url_key(request_id: str) -> str:
    """Get Redis key for storing signed URL."""
    return f"inference_request:{request_id}:signed_url"


def parse_azure_blob_container_url(url: str) -> tuple[str, str]:
    """
    Parse an HTTPS Azure Blob container URL into storage account and container name.

    Parameters
    ----------
    url : str
        ``https://<account>.blob.core.windows.net/<container>`` with optional extra path
        segments after the container.

    Returns
    -------
    tuple[str, str]
        ``(account_name, container_name)``.

    Raises
    ------
    ValueError
        If the URL is not a valid Azure Blob container HTTPS URL.
    """
    trimmed = url.strip()
    parsed = urlparse(trimmed)
    if parsed.scheme.lower() != "https":
        raise ValueError("container_url must use https")
    host = (parsed.hostname or "").lower()
    suffix = ".blob.core.windows.net"
    if not host.endswith(suffix):
        raise ValueError("container_url host must be <account>.blob.core.windows.net")
    account = host[: -len(suffix)]
    if not account:
        raise ValueError("container_url is missing a storage account name")
    parts = [p for p in parsed.path.strip("/").split("/") if p]
    if not parts:
        raise ValueError(
            "container_url must include a container name in the path "
            "(e.g. https://acct.blob.core.windows.net/mycontainer)"
        )
    return account, parts[0]


# =============================================================================
# File Streaming Utilities
# =============================================================================
def parse_range_header(
    range_header: str | None, file_size: int
) -> tuple[int, int, int, int]:
    """
    Parse Range header and return start, end, content_length, and status_code.

    Args:
        range_header: The Range header value from the request, or None
        file_size: Total size of the file in bytes

    Returns:
        Tuple of (start, end, content_length, status_code)

    Raises:
        HTTPException: If the range is invalid (416 status)
    """
    start = 0
    end = file_size - 1
    status_code = 200

    if not range_header:
        return (start, end, file_size, status_code)

    # Parse Range header: "bytes=start-end" or "bytes=start-" or "bytes=-suffix"
    if not range_header.startswith("bytes="):
        raise HTTPException(
            status_code=416,
            detail={
                "error": "Range Not Satisfiable",
                "details": "Only byte ranges are supported",
            },
        )

    range_spec = range_header[6:]  # Remove "bytes=" prefix
    ranges = range_spec.split(",")

    # For now, only handle single range (first range)
    if len(ranges) > 1:
        logger.warning(f"Multiple ranges requested, using first: {range_header}")

    range_part = ranges[0].strip()
    if "-" not in range_part:
        raise HTTPException(
            status_code=416,
            detail={
                "error": "Range Not Satisfiable",
                "details": "Invalid range format",
            },
        )

    start_str, end_str = range_part.split("-", 1)

    if start_str:
        start = int(start_str)
        if end_str:
            end = int(end_str)
        else:
            end = file_size - 1
    else:
        # Suffix range: "-suffix" means last N bytes
        suffix = int(end_str)
        start = max(0, file_size - suffix)
        end = file_size - 1

    # Validate range
    if start < 0 or start >= file_size or end < start or end >= file_size:
        raise HTTPException(
            status_code=416,
            headers={
                "Content-Range": f"bytes */{file_size}",
            },
            detail={
                "error": "Range Not Satisfiable",
                "details": f"Requested range {start}-{end} is invalid for file size {file_size}",
            },
        )

    status_code = 206  # Partial Content
    content_length = end - start + 1
    return (start, end, content_length, status_code)


async def create_file_stream(
    file_path: Path, start: int, content_length: int, file_description: str = "file"
) -> AsyncGenerator[bytes, None]:
    """
    Create an async generator that streams a file with optional range support.

    Args:
        file_path: Path to the file to stream
        start: Starting byte position (0 for full file, or range start)
        content_length: Number of bytes to stream
        file_description: Description for error logging

    Yields:
        Bytes chunks from the file
    """
    try:
        chunk_size = 1048576  # 1MB chunks for better performance
        async with aiofiles.open(file_path, "rb") as f:
            # Skip leading bytes for range requests. Prefer sequential reads over seek so
            # we stay compatible across aiofiles versions (seek may be sync or async).
            if start > 0:
                remaining_skip = start
                while remaining_skip > 0:
                    n = min(chunk_size, remaining_skip)
                    data = await f.read(n)
                    if not data:
                        break
                    remaining_skip -= len(data)

            remaining = content_length
            while remaining > 0:
                read_size = min(chunk_size, remaining)
                chunk = await f.read(read_size)
                if not chunk:
                    break
                yield chunk
                remaining -= len(chunk)
                await asyncio.sleep(0)
    except Exception:
        logger.exception(f"Error streaming {file_description} {file_path}")
        raise


# =============================================================================
# Pipeline Stage Utilities
# =============================================================================


Stage = Literal["inference", "result_zip", "object_storage", "geocatalog_ingestion"]


@check_optional_dependencies()
def queue_next_stage(
    redis_client: redis.Redis,
    current_stage: Stage,
    workflow_name: str,
    execution_id: str,
    output_path_str: str,
    results_zip_dir_str: str | None = None,
) -> str | None:
    """
    Queue the next pipeline stage based on configuration.

    Pipeline flow:
    - If result_zip_enabled: inference -> result_zip -> object_storage (if enabled) -> [geocatalog_ingestion (if request had geo_catalog_url)] -> finalize
    - If not result_zip_enabled: inference -> object_storage (if enabled) -> [geocatalog_ingestion (if request had geo_catalog_url)] -> finalize

    Args:
        redis_client: Redis client for queue connection
        current_stage: The stage that just completed ("inference", "result_zip", "object_storage", "geocatalog_ingestion")
        workflow_name: Name of the workflow
        execution_id: Execution ID of the workflow
        output_path_str: Path to the output files
        results_zip_dir_str: Directory for result zips (required when queuing result_zip)

    Returns:
        The RQ job ID if queued successfully, None on error
    """
    config = get_config()

    # Determine next stage based on current stage and config
    args: tuple[Any, ...]
    if current_stage == "inference":
        # Always go to result_zip stage (for manifest building), but control zip creation via config
        next_queue = "result_zip"
        next_func = "earth2studio.serve.server.cpu_worker.process_result_zip"
        # Pass create_zip parameter based on config
        # Disable zip creation when object storage is enabled (files uploaded directly)
        create_zip = (
            config.paths.result_zip_enabled and not config.object_storage.enabled
        )
        args = (
            workflow_name,
            execution_id,
            output_path_str,
            results_zip_dir_str,
            create_zip,
        )

    elif current_stage == "result_zip":
        if config.object_storage.enabled:
            next_queue = "object_storage"
            next_func = (
                "earth2studio.serve.server.cpu_worker.process_object_storage_upload"
            )
            args = (workflow_name, execution_id, output_path_str)
        else:
            next_queue = "finalize_metadata"
            next_func = "earth2studio.serve.server.cpu_worker.process_finalize_metadata"
            args = (workflow_name, execution_id)

    elif current_stage == "object_storage":
        if config.object_storage.storage_type == "azure":
            next_queue = "geocatalog_ingestion"
            next_func = "azure_planetary_computer.geocatalog_ingestion.process_geocatalog_ingestion"
            args = (workflow_name, execution_id)
        else:
            next_queue = "finalize_metadata"
            next_func = "earth2studio.serve.server.cpu_worker.process_finalize_metadata"
            args = (workflow_name, execution_id)

    elif current_stage == "geocatalog_ingestion":
        next_queue = "finalize_metadata"
        next_func = "earth2studio.serve.server.cpu_worker.process_finalize_metadata"
        args = (workflow_name, execution_id)

    else:
        logger.error(f"Unknown stage: {current_stage}")
        return None

    # Queue the job
    try:
        queue = Queue(
            next_queue,
            connection=redis_client,
            default_timeout=config.queue.default_timeout,
        )
        job = queue.enqueue(
            next_func,
            *args,
            job_id=f"{next_queue}_{workflow_name}_{execution_id}",
            job_timeout=config.queue.job_timeout,
        )
        logger.info(
            f"Queued {next_queue} for {workflow_name}:{execution_id} with RQ job ID: {job.id}"
        )
        return job.id
    except Exception as e:
        logger.error(
            f"Failed to queue {next_queue} for {workflow_name}:{execution_id}: {e}"
        )
        return None
