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
Earth2Studio RQ Worker

This module contains the worker functions for processing inference requests
using Redis Queue (RQ). Workers run the actual inference computations.

"""

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis  # type: ignore[import-untyped]

# Import configuration
from api_server.config import get_config, get_config_manager

# Import queue_next_stage utility
from api_server.utils import queue_next_stage

# Import workflow registry
from api_server.workflow import WorkflowStatus, workflow_registry

# Get configuration
config = get_config()
config_manager = get_config_manager()

# Configure logging
config_manager.setup_logging()
logger = logging.getLogger(__name__)

# Path configuration from config
DEFAULT_OUTPUT_DIR = Path(config.paths.default_output_dir)
RESULTS_ZIP_DIR = Path(config.paths.results_zip_dir)

# Model registry for caching loaded models
model_registry: dict[str, Any] = {}

# Redis client for worker
redis_client = redis.Redis(
    host=config.redis.host,
    port=config.redis.port,
    db=config.redis.db,
    password=config.redis.password,
    decode_responses=config.redis.decode_responses,
    socket_connect_timeout=config.redis.socket_connect_timeout,
    socket_timeout=config.redis.socket_timeout,
)

# Register custom workflows in the worker process
try:
    from api_server.workflow import register_all_workflows

    register_all_workflows(redis_client)
    logger.info("Custom workflows registered successfully in worker process")
except ImportError:
    logger.warning(
        "Workflow registration module not found in worker, skipping custom workflow registration"
    )
except Exception as e:
    logger.error(f"Failed to register custom workflows in worker: {e}")
    # Don't raise - worker can still handle other tasks


def get_output_path(
    io_config: dict[str, Any] | None,
    timestamp: str,
    workflow_type: str,
    request_id: str,
) -> Path:
    """Generate output path for workflow results"""
    # Create timestamp-based subdirectory
    timestamp_str = timestamp.replace(":", "").replace("Z", "").replace("+", "")
    output_dir = DEFAULT_OUTPUT_DIR / f"{workflow_type}_{timestamp_str}_{request_id}"
    output_dir.mkdir(parents=True, exist_ok=True)
    # Default file name
    backend_type = io_config.get("backend_type", "zarr") if io_config else "zarr"
    return output_dir / f"forecast.{backend_type}"


def run_custom_workflow(
    workflow_name: str, execution_id: str, parameters: dict[str, Any]
) -> Any:
    """RQ Worker function to run custom workflow"""
    # Create logger adapter with execution_id for automatic log prefixing
    log = logging.LoggerAdapter(logger, {"execution_id": execution_id})

    log.info(f"Starting custom workflow {workflow_name}")

    # Get workflow class from registry
    workflow_class = workflow_registry.get_workflow_class(workflow_name)
    if not workflow_class:
        raise ValueError(f"Custom workflow '{workflow_name}' not found in registry")

    # Create workflow instance for execution
    custom_workflow = workflow_registry.get(workflow_name, redis_client=redis_client)

    try:
        start_timestamp = time.time()
        start_time = datetime.now(timezone.utc).isoformat()
        updates = {
            "status": WorkflowStatus.RUNNING,
            "start_time": start_time,
        }
        workflow_class._update_execution_data(
            redis_client, workflow_name, execution_id, updates
        )
        log.info(f"Executing workflow {workflow_name} with execution ID {execution_id}")

        result = custom_workflow.run(parameters, execution_id)

        # Update status to PENDING_RESULTS and record execution time so far
        execution_time_seconds = time.time() - start_timestamp
        updates = {
            "status": WorkflowStatus.PENDING_RESULTS,
            "execution_time_seconds": execution_time_seconds,
        }
        workflow_class._update_execution_data(
            redis_client, workflow_name, execution_id, updates
        )
        log.info(f"Workflow {workflow_name} execution {execution_id} pending results")

        # Queue next pipeline stage (determined by configuration)
        output_path = custom_workflow.get_output_path(execution_id)
        job_id = queue_next_stage(
            redis_client=redis_client,
            current_stage="inference",
            workflow_name=workflow_name,
            execution_id=execution_id,
            output_path_str=str(output_path),
            results_zip_dir_str=str(RESULTS_ZIP_DIR),
        )
        if not job_id:
            error_msg = f"Failed to queue next pipeline stage for {workflow_name}:{execution_id}"
            log.error(error_msg)
            try:
                updates = {
                    "status": WorkflowStatus.FAILED,
                    "end_time": datetime.now(timezone.utc).isoformat(),
                    "error_message": error_msg,
                }
                workflow_class._update_execution_data(
                    redis_client, workflow_name, execution_id, updates
                )
            except Exception:
                log.exception("Failed to update workflow status after queue failure")
            raise RuntimeError(error_msg)

        logger.info(
            f"Queued next stage for {workflow_name}:{execution_id} with RQ job ID: {job_id}"
        )
        return result

    except Exception as e:
        log.exception(
            f"Custom workflow {workflow_name} execution {execution_id} failed"
        )
        # Update workflow status to failed if possible
        try:
            updates = {
                "status": WorkflowStatus.FAILED,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "error_message": str(e),
            }
            workflow_class._update_execution_data(
                redis_client, workflow_name, execution_id, updates
            )
            log.info(
                f"Workflow {workflow_name} execution {execution_id} failed with error: {str(e)}"
            )
        except Exception:
            log.exception("Failed to update workflow status after failure")
        raise e  # Re-raise for RQ to handle
