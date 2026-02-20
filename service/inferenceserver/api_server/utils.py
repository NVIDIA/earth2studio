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

"""
Common utilities for the API Server.

This module contains:
- Redis key generation functions
- Pipeline stage queuing utilities
"""

import logging
from typing import Any, Literal

import redis  # type: ignore[import-untyped]
from rq import Queue

from api_server.config import get_config

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


# =============================================================================
# Pipeline Stage Utilities
# =============================================================================

Stage = Literal["inference", "result_zip", "object_storage"]


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
    - If result_zip_enabled: inference -> result_zip -> object_storage (if enabled) -> finalize
    - If not result_zip_enabled: inference -> object_storage (if enabled) -> finalize

    Args:
        redis_client: Redis client for queue connection
        current_stage: The stage that just completed ("inference", "result_zip", "object_storage")
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
        next_func = "api_server.cpu_worker.process_result_zip"
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
            next_func = "api_server.cpu_worker.process_object_storage_upload"
            args = (workflow_name, execution_id, output_path_str)
        else:
            next_queue = "finalize_metadata"
            next_func = "api_server.cpu_worker.process_finalize_metadata"
            args = (workflow_name, execution_id)

    elif current_stage == "object_storage":
        next_queue = "finalize_metadata"
        next_func = "api_server.cpu_worker.process_finalize_metadata"
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
