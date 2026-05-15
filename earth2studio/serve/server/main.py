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
Earth2Studio REST API server.

Provides FastAPI endpoints for workflow execution, status, and result retrieval,
with Redis and RQ for job queuing and Prometheus metrics.
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from earth2studio.utils.imports import OptionalDependencyError

try:
    import aiofiles  # type: ignore[import-untyped]
    import redis as redis_sync  # type: ignore[import-untyped]  # For RQ (synchronous)
    import uvicorn
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response, StreamingResponse
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    from pydantic import BaseModel, Field
    from rq import Queue
except ImportError as e:
    raise OptionalDependencyError(
        "serve", "earth2studio.serve.server.main", e, e.__traceback__
    )

from earth2studio.serve.server.config import (
    get_config,
    get_config_manager,
)
from earth2studio.serve.server.dependencies import (
    AsyncRedis,
    InferenceQueue,
    SyncRedis,
)
from earth2studio.serve.server.health import check_all_services
from earth2studio.serve.server.redis_factory import (
    create_async_redis_client,
    create_sync_redis_client,
)
from earth2studio.serve.server.utils import (
    create_file_stream,
    get_inference_request_output_path_key,
    get_inference_request_zip_key,
    parse_range_header,
)
from earth2studio.serve.server.workflow import (
    WorkflowRegistry,
    WorkflowResult,
    WorkflowStatus,
)

# Get configuration
config = get_config()
config_manager = get_config_manager()

# Configure logging
config_manager.setup_logging()


def check_admission_control(sync_redis: redis_sync.Redis) -> None:
    """
    Check if the inference request can be admitted based on queue sizes.

    Checks all relevant queues (inference, result zip, object storage,
    and finalize metadata) to ensure none are at capacity before allowing
    a new request to be enqueued.

    Parameters
    ----------
    sync_redis : redis.Redis
        Synchronous Redis client.

    Raises
    ------
    HTTPException
        429 if any queue is full (service temporarily unavailable; retry later).
        500 if queue status cannot be determined.
    """
    queue_names = [
        config.queue.name,
        config.queue.result_zip_queue_name,
        config.queue.object_storage_queue_name,
        config.queue.geocatalog_ingestion_queue_name,
        config.queue.finalize_metadata_queue_name,
    ]
    for queue_name in queue_names:
        try:
            current_queue_size = sync_redis.llen(f"rq:queue:{queue_name}")
        except Exception as e:
            logger.error(f"Failed to get queue length for {queue_name}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: unable to check queue '{queue_name}' status.",
            )
        if current_queue_size >= config.queue.max_size:
            logger.error(
                f"Queue '{queue_name}' is full ({current_queue_size}/{config.queue.max_size})."
            )
            raise HTTPException(
                status_code=429,
                detail=f"Service temporarily unavailable. Queue '{queue_name}' is full ({current_queue_size}/{config.queue.max_size}). Please try again later.",
            )


def get_queue_position(queue: Queue | None, job_id: str) -> int | None:
    """
    Get the position of a job in the queue.

    Uses RQ's Queue.job_ids property, which correctly returns job IDs
    including custom job_id values set during enqueue.

    Parameters
    ----------
    queue : Queue | None
        The RQ inference queue, or None if unavailable.
    job_id : str
        The ID of the job to find.

    Returns
    -------
    int or None
        Position in queue (0-indexed), or None if not found (e.g. picked up by worker).
    """
    if queue is None:
        return None
    try:
        job_ids = queue.job_ids

        logger.debug(f"Queue has {len(job_ids)} jobs. Looking for job_id: '{job_id}'")

        if job_id in job_ids:
            position = job_ids.index(job_id)
            logger.debug(f"Job {job_id} found at position {position}")
            return position

        logger.debug(f"Job '{job_id}' not found in queue (likely picked up by worker)")
        return None
    except Exception as e:
        logger.warning(f"Failed to get queue position for job {job_id}: {e}")
        return None


# Path configuration from config
DEFAULT_OUTPUT_DIR = Path(config.paths.default_output_dir)
RESULTS_ZIP_DIR = Path(config.paths.results_zip_dir)

# Model registry for caching loaded models
model_registry: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for FastAPI application startup and shutdown.

    On startup: connects to Redis (async and sync), initializes the RQ inference
    queue, and registers custom workflows. On shutdown: closes Redis connections.
    """
    # Startup
    async_client = None
    sync_client = None
    try:
        async_client = create_async_redis_client()
        await async_client.ping()

        sync_client = create_sync_redis_client()
        sync_client.ping()

        queue = Queue(
            config.queue.name,
            connection=sync_client,
            default_timeout=config.queue.default_timeout,
        )

        # Store on app.state for dependency injection
        app.state.redis_client = async_client
        app.state.redis_sync_client = sync_client
        app.state.inference_queue = queue

        logger.info(f"Connected to Redis at {config.redis.host}:{config.redis.port}")
        logger.info(
            f"RQ inference queue initialized with max size: {config.queue.max_size}"
        )

        # Register custom workflows
        try:
            from earth2studio.serve.server.workflow import register_all_workflows

            register_all_workflows(sync_client)
            logger.info("Custom workflows registered successfully")
        except ImportError:
            logger.warning(
                "Workflow registration module not found, skipping custom workflow registration"
            )
        except Exception:
            logger.exception("Failed to register custom workflows")

    except Exception:
        logger.exception("Failed to connect to Redis or initialize RQ")
        if async_client:
            await async_client.close()
        if sync_client:
            sync_client.close()
        raise

    # Application is running
    yield

    # Shutdown
    if hasattr(app.state, "redis_client") and app.state.redis_client:
        await app.state.redis_client.close()
        logger.info("Async Redis connection closed")
    if hasattr(app.state, "redis_sync_client") and app.state.redis_sync_client:
        app.state.redis_sync_client.close()
        logger.info("Sync Redis connection closed")


# Create FastAPI app
app = FastAPI(
    title=config.server.title,
    description=config.server.description,
    version=config.server.version,
    docs_url=config.server.docs_url,
    redoc_url=config.server.redoc_url,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors.allow_origins,
    allow_credentials=config.cors.allow_credentials,
    allow_methods=config.cors.allow_methods,
    allow_headers=config.cors.allow_headers,
)


# Workflow execution models are defined below


@app.get("/health")
@app.get("/readiness")
async def health_check(sync_redis: SyncRedis) -> dict[str, str]:
    """
    Health and readiness check endpoint.

    Runs in-process health checks for all services and returns overall health.

    Returns
    -------
    dict
        Keys ``status`` (e.g. "healthy"/"unhealthy") and ``timestamp`` (ISO format).

    Raises
    ------
    HTTPException
        503 if status is unhealthy; 500 if the check fails.
    """
    try:
        result = await asyncio.to_thread(check_all_services, redis_client=sync_redis)
        overall_status = "healthy" if result.healthy else "unhealthy"

        response_data = {
            "status": overall_status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if overall_status == "unhealthy":
            raise HTTPException(status_code=503, detail=response_data)

        return response_data

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


# Azure ML uses this to see if the container is alive
@app.get("/liveness")
def liveness() -> dict[str, str]:
    """
    Liveness probe for Azure ML.

    Returns
    -------
    dict
        ``status`` key set to "alive" to indicate the container is running.
    """
    return {"status": "alive"}


@app.get("/metrics")
async def get_metrics() -> Response:
    """
    Expose Prometheus metrics.

    Returns
    -------
    Response
        Response with Prometheus text format and appropriate content type.

    Raises
    ------
    HTTPException
        500 if metrics generation fails.
    """
    try:
        metrics_data = generate_latest()
        return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logger.exception("Failed to generate metrics")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to generate metrics",
                "details": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


@app.get("/v1/infer/workflows")
async def list_workflows() -> dict[str, dict[str, str]]:
    """
    List all available workflows.

    Returns
    -------
    dict
        Single key ``workflows`` mapping workflow name to description.
    """
    workflows = WorkflowRegistry.instance().list_workflows(exposed_only=True)
    return {"workflows": workflows}


@app.get("/v1/infer/workflows/{workflow_name}/schema")
async def get_workflow_schema(workflow_name: str) -> dict[str, Any]:
    """
    Get the OpenAPI JSON schema for a workflow's parameters.

    For Workflow subclasses the schema comes from WorkflowParameters; for
    Earth2Workflow subclasses it is generated from the __call__ signature via
    AutoParameters.

    Parameters
    ----------
    workflow_name : str
        Name of the workflow.

    Returns
    -------
    dict
        JSON schema for the workflow parameters (OpenAPI-compatible).

    Raises
    ------
    HTTPException
        404 if workflow not found; 500 if schema generation fails.
    """
    # Check if workflow exists and is exposed
    workflow_class = WorkflowRegistry.instance().get_workflow_class(workflow_name)
    if not workflow_class:
        raise HTTPException(
            status_code=404, detail=f"Workflow '{workflow_name}' not found"
        )
    if not WorkflowRegistry.instance().is_workflow_exposed(workflow_name):
        raise HTTPException(
            status_code=404, detail=f"Workflow '{workflow_name}' is not exposed"
        )

    try:
        # Get the Parameters class from the workflow
        # Both Workflow and Earth2Workflow have a Parameters attribute
        # For Workflow subclasses, this is explicitly defined
        # For Earth2Workflow subclasses, this is auto-generated by AutoParameters metaclass
        parameters_class = workflow_class.Parameters

        # Generate JSON Schema using Pydantic's built-in method
        # This produces a valid JSON Schema (draft 2020-12) compatible with OpenAPI 3.1
        schema = parameters_class.model_json_schema()

        # Return the JSON Schema directly for OpenAPI compatibility
        return schema

    except Exception as e:
        logger.exception(f"Failed to generate schema for workflow {workflow_name}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to generate schema",
                "details": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


class WorkflowExecutionRequest(BaseModel):
    """
    Request body for workflow execution.

    Attributes
    ----------
    parameters : dict
        Workflow-specific parameters (validated by the workflow's parameter class).
    """

    model_config = {"extra": "forbid"}

    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Workflow parameters"
    )


class WorkflowExecutionResponse(BaseModel):
    """
    Response after submitting a workflow execution.

    Attributes
    ----------
    workflow_name : str
        Name of the workflow.
    execution_id : str
        Unique execution identifier.
    status : str
        Current status (e.g. QUEUED).
    position : int or None
        Queue position (0-indexed); set when status is QUEUED.
    message : str
        Human-readable message.
    timestamp : str
        ISO format timestamp.
    """

    workflow_name: str
    execution_id: str
    status: str
    position: int | None = (
        None  # Queue position (0-indexed), only set when status is QUEUED
    )
    message: str
    timestamp: str


@app.post("/v1/infer", response_model=WorkflowExecutionResponse)
async def execute_default_workflow(
    request: WorkflowExecutionRequest,
    sync_redis: SyncRedis,
    queue: InferenceQueue,
) -> WorkflowExecutionResponse:
    """
    Enqueue the single exposed workflow for execution.

    This endpoint is only valid when exactly one workflow is exposed (same notion
    of *exposed* as ``GET /v1/infer/workflows``). If zero or more than one workflow
    is exposed, the request fails.

    Parameters
    ----------
    request : WorkflowExecutionRequest
        Request body containing workflow parameters for the selected workflow.
    sync_redis : SyncRedis
        Synchronous Redis client (injected).
    queue : InferenceQueue
        RQ inference queue (injected).

    Returns
    -------
    WorkflowExecutionResponse
        Execution ID, status (QUEUED), queue position, and message.

    Raises
    ------
    HTTPException
        503 if no exposed workflows are registered; 409 if more than one exposed
        workflow; otherwise same status codes as ``POST /v1/infer/{workflow_name}``.
    """
    exposed = WorkflowRegistry.instance().list_workflows(exposed_only=True)
    n_exposed = len(exposed)
    if n_exposed == 0:
        raise HTTPException(
            status_code=503,
            detail="No exposed workflows are available.",
        )
    if n_exposed > 1:
        names = ", ".join(exposed)
        raise HTTPException(
            status_code=409,
            detail=(
                "POST /v1/infer requires exactly one exposed workflow; "
                f"found {n_exposed}: {names}."
            ),
        )
    workflow_name = next(iter(exposed))
    return await execute_workflow(workflow_name, request, sync_redis, queue)


@app.post("/v1/infer/{workflow_name}", response_model=WorkflowExecutionResponse)
async def execute_workflow(
    workflow_name: str,
    request: WorkflowExecutionRequest,
    sync_redis: SyncRedis,
    queue: InferenceQueue,
) -> WorkflowExecutionResponse:
    """
    Enqueue a custom workflow for execution.

    Validates parameters, performs admission control, persists execution data
    in Redis, and enqueues the job to the RQ inference queue.

    Parameters
    ----------
    workflow_name : str
        Name of the registered workflow.
    request : WorkflowExecutionRequest
        Request body containing workflow parameters.
    sync_redis : SyncRedis
        Synchronous Redis client (injected).
    queue : InferenceQueue
        RQ inference queue (injected).

    Returns
    -------
    WorkflowExecutionResponse
        Execution ID, status (QUEUED), queue position, and message.

    Raises
    ------
    HTTPException
        404 if workflow not found; 422 if parameters invalid; 429 if queues full;
        503 if Redis/queue not initialized; 500 on enqueue failure.
    """
    # Check if workflow exists and is exposed
    custom_workflow_class = WorkflowRegistry.instance().get_workflow_class(
        workflow_name
    )
    if not custom_workflow_class:
        raise HTTPException(
            status_code=404, detail=f"Workflow '{workflow_name}' not found"
        )
    if not WorkflowRegistry.instance().is_workflow_exposed(workflow_name):
        raise HTTPException(
            status_code=404, detail=f"Workflow '{workflow_name}' is not exposed"
        )

    # Validate parameters early to provide immediate feedback using classmethod
    try:
        validated_params = custom_workflow_class.validate_parameters(request.parameters)
        validated_params_dict = validated_params.model_dump()
    except ValueError as e:
        logger.error(f"Parameter validation failed for workflow {workflow_name}: {e}")
        raise HTTPException(
            status_code=422,
            detail=f"Invalid parameters for workflow '{workflow_name}': {str(e)}",
        )
    except Exception as e:
        logger.exception(
            f"Unexpected error validating parameters for workflow {workflow_name}"
        )
        raise HTTPException(
            status_code=400, detail=f"Error validating parameters: {str(e)}"
        )

    # Admission control: check all queue sizes before enqueuing
    check_admission_control(sync_redis)

    # Generate execution ID
    execution_id = f"exec_{int(time.time())}_{uuid.uuid4().hex[:8]}"

    try:
        # Save initial execution data using classmethod
        execution_data = WorkflowResult(
            workflow_name=workflow_name,
            execution_id=execution_id,
            status=WorkflowStatus.QUEUED,
            start_time=datetime.now(timezone.utc).isoformat(),
            metadata={"parameters": validated_params_dict},
        )
        custom_workflow_class._save_execution_data(
            sync_redis, workflow_name, execution_id, execution_data
        )

        job = queue.enqueue(
            "earth2studio.serve.server.worker.run_custom_workflow",
            workflow_name,
            execution_id,
            validated_params_dict,
            job_id=f"{workflow_name}_{execution_id}",
            job_timeout=config.queue.job_timeout,
        )

        try:
            queue_length = sync_redis.llen(f"rq:queue:{config.queue.name}")
            queue_position = queue_length
        except Exception as e:
            logger.error(f"Failed to get queue length from Redis: {e}")
            raise HTTPException(
                status_code=500,
                detail="Internal server error: unable to determine queue position.",
            )

        logger.info(
            f"Queued workflow {workflow_name} execution {execution_id} with RQ job ID: {job.id}, position: {queue_position}"
        )

        return WorkflowExecutionResponse(
            workflow_name=workflow_name,
            execution_id=execution_id,
            status=WorkflowStatus.QUEUED,
            position=queue_position,
            message=f"Workflow '{workflow_name}' queued for execution",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            f"Failed to enqueue workflow {workflow_name} execution {execution_id}"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to enqueue workflow execution: {str(e)}"
        )


@app.get(
    "/v1/infer/{workflow_name}/{execution_id}/status", response_model=WorkflowResult
)
async def get_workflow_status(
    workflow_name: str,
    execution_id: str,
    sync_redis: SyncRedis,
    queue: InferenceQueue,
) -> WorkflowResult:
    """
    Get the status of a workflow execution.

    Parameters
    ----------
    workflow_name : str
        Name of the workflow.
    execution_id : str
        Unique execution identifier.
    sync_redis : SyncRedis
        Synchronous Redis client (injected).
    queue : InferenceQueue
        RQ inference queue (injected).

    Returns
    -------
    WorkflowResult
        Current status, progress, timestamps, and optional queue position.

    Raises
    ------
    HTTPException
        404 if workflow or execution not found; 500 on server error.
    """
    log = logger.bind(execution_id=execution_id)

    custom_workflow_class = WorkflowRegistry.instance().get_workflow_class(
        workflow_name
    )
    if not custom_workflow_class:
        raise HTTPException(
            status_code=404, detail=f"Custom workflow '{workflow_name}' not found"
        )
    if not WorkflowRegistry.instance().is_workflow_exposed(workflow_name):
        raise HTTPException(
            status_code=404, detail=f"Workflow '{workflow_name}' is not exposed"
        )

    try:
        result = custom_workflow_class._get_execution_data(
            sync_redis, workflow_name, execution_id
        )

        if result.status == WorkflowStatus.QUEUED:
            job_id = f"{workflow_name}_{execution_id}"
            queue_position = get_queue_position(queue, job_id)

            if queue_position is None:
                log.info(
                    f"Job {job_id} has status QUEUED but not found in queue - worker likely picked it up. "
                    f"Treating as RUNNING."
                )
                result.status = WorkflowStatus.RUNNING
                result.position = None
            else:
                result.position = queue_position

        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        log.exception(f"Failed to get status for {workflow_name}:{execution_id}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get workflow status: {str(e)}"
        )


@app.get("/v1/infer/{workflow_name}/{execution_id}/results", response_model=None)
async def get_workflow_results(
    workflow_name: str,
    execution_id: str,
    sync_redis: SyncRedis,
) -> dict[str, Any] | StreamingResponse:
    """
    Get result metadata (e.g. metadata.json) for a completed workflow execution.

    Parameters
    ----------
    workflow_name : str
        Name of the workflow.
    execution_id : str
        Unique execution identifier.

    Returns
    -------
    dict
        JSON metadata (e.g. from metadata_{workflow_name}:{execution_id}.json).

    Raises
    ------
    HTTPException
        202 if still queued/running/pending; 400 if expired or bad request;
        404 if workflow, execution, or metadata file not found, or if execution failed/cancelled.
    """
    log = logger.bind(execution_id=execution_id)

    # Check if workflow exists and is exposed
    custom_workflow_class = WorkflowRegistry.instance().get_workflow_class(
        workflow_name
    )
    if not custom_workflow_class:
        raise HTTPException(
            status_code=404, detail=f"Custom workflow '{workflow_name}' not found"
        )
    if not WorkflowRegistry.instance().is_workflow_exposed(workflow_name):
        raise HTTPException(
            status_code=404, detail=f"Workflow '{workflow_name}' is not exposed"
        )

    # Check workflow status first
    try:
        result = custom_workflow_class._get_execution_data(
            sync_redis, workflow_name, execution_id
        )
        if result.status == WorkflowStatus.EXPIRED:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Results expired",
                    "details": f"Results for workflow execution {execution_id} have expired and are no longer available",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        if (
            result.status == WorkflowStatus.PENDING_RESULTS
            or result.status == WorkflowStatus.QUEUED
            or result.status == WorkflowStatus.RUNNING
        ):
            raise HTTPException(
                status_code=202,
                detail={
                    "message": f"Workflow execution {execution_id} is still {result.status}",
                    "status": result.status,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        if result.status == WorkflowStatus.FAILED:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Workflow execution failed",
                    "details": f"Workflow execution {execution_id} failed and has no results",
                    "error_message": result.error_message,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
        if result.status == WorkflowStatus.CANCELLED:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Workflow execution cancelled",
                    "details": f"Workflow execution {execution_id} cancelled and has no results",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Get metadata file
    try:
        request_id = f"{workflow_name}:{execution_id}"
        metadata_filename = f"metadata_{request_id}.json"
        metadata_path = RESULTS_ZIP_DIR / metadata_filename

        if not metadata_path.exists():
            log.error(f"Metadata file not found: {metadata_path}")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Metadata file not found",
                    "details": f"The metadata file for execution {execution_id} could not be found",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        async with aiofiles.open(metadata_path, "r") as f:
            metadata_content = await f.read()
            metadata_json = json.loads(metadata_content)

        return metadata_json

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to get results for {workflow_name}:{execution_id}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to retrieve results",
                "details": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


@app.get("/v1/infer/{workflow_name}/{execution_id}/results/{filepath:path}")
async def get_workflow_result_file(
    request: Request,
    workflow_name: str,
    execution_id: str,
    filepath: str,
    sync_redis: SyncRedis,
    async_redis: AsyncRedis,
) -> StreamingResponse:
    """
    Stream a specific file from the workflow execution results.

    If filepath equals the request ID (workflow_name:execution_id), streams the
    results zip file. Otherwise serves a file from the output directory (path
    normalized and checked to prevent directory traversal).

    Parameters
    ----------
    workflow_name : str
        Name of the workflow.
    execution_id : str
        Execution identifier.
    filepath : str
        Relative path within the output directory, or the request ID for the zip.
    sync_redis : SyncRedis
        Synchronous Redis client (injected).
    async_redis : AsyncRedis
        Async Redis client (injected).

    Returns
    -------
    StreamingResponse
        File or zip contents with appropriate headers.

    Raises
    ------
    HTTPException
        403 on path traversal attempt; 404 if workflow, execution, file, or zip
        not found or results not completed; 500 on error.
    """
    # Check if workflow exists and is exposed
    custom_workflow_class = WorkflowRegistry.instance().get_workflow_class(
        workflow_name
    )
    if not custom_workflow_class:
        raise HTTPException(
            status_code=404, detail=f"Custom workflow '{workflow_name}' not found"
        )
    if not WorkflowRegistry.instance().is_workflow_exposed(workflow_name):
        raise HTTPException(
            status_code=404, detail=f"Workflow '{workflow_name}' is not exposed"
        )

    # Check workflow status first
    try:
        result = custom_workflow_class._get_execution_data(
            sync_redis, workflow_name, execution_id
        )
        if result.status != WorkflowStatus.COMPLETED:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Results not available",
                    "details": f"Workflow execution {execution_id} is {result.status}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        # Special case: if filepath matches the request_id format, return the zip file
        request_id = f"{workflow_name}:{execution_id}"
        if filepath == request_id:
            zip_key = get_inference_request_zip_key(request_id)
            zip_filename = await async_redis.get(zip_key)

            if not zip_filename:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "Zip file not found",
                        "details": f"No zip file associated with execution {execution_id}",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

            zip_file_path = RESULTS_ZIP_DIR / zip_filename
            if not zip_file_path.exists():
                logger.error(f"Zip file not found: {zip_file_path}")
                raise HTTPException(
                    status_code=404,
                    detail={
                        "error": "Zip file not found on disk",
                        "details": f"The zip file {zip_filename} could not be found",
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    },
                )

            zip_file_size = zip_file_path.stat().st_size
            range_header = request.headers.get("Range")
            start, end, content_length, range_status = parse_range_header(
                range_header, zip_file_size
            )
            stream_generator = create_file_stream(
                zip_file_path, start, content_length, "zip file"
            )

            headers = {
                "Content-Disposition": f'attachment; filename="{zip_filename}"',
                "Content-Length": str(content_length),
                "Accept-Ranges": "bytes",
            }
            if range_status == 206:
                headers["Content-Range"] = f"bytes {start}-{end}/{zip_file_size}"

            return StreamingResponse(
                stream_generator,
                media_type="application/zip",
                headers=headers,
                status_code=range_status,
            )

        # Regular case: get file from output directory

        # Get output directory for this execution
        request_id = f"{workflow_name}:{execution_id}"
        output_path_key = get_inference_request_output_path_key(request_id)
        output_dir_str = await async_redis.get(output_path_key)

        if output_dir_str:
            output_dir = Path(output_dir_str)
        else:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Output directory not found",
                    "details": f"No output directory associated with execution {execution_id}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        # Handle case where filepath includes the directory name (as seen in zip manifest)
        # The manifest includes the execution ID prefix (e.g., "exec_123/summary.txt")
        # but the file is located at output_dir/summary.txt
        if filepath.startswith(f"{output_dir.name}/"):
            filepath = filepath[len(output_dir.name) + 1 :]

        # Security: Normalize the filepath and ensure it's within output_dir
        requested_path = (output_dir / filepath).resolve()
        logger.debug(f"Requested path: {requested_path}")
        # Check that the resolved path is actually within the output directory
        try:
            requested_path.relative_to(output_dir.resolve())
        except ValueError:
            # Path is outside output directory - potential directory traversal attack
            logger.warning(
                f"Attempted directory traversal: {filepath} for {workflow_name}:{execution_id}"
            )
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Access denied",
                    "details": "The requested file path is not within the results directory",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        # Check if file exists
        if not requested_path.exists():
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "File not found",
                    "details": f"The file '{filepath}' does not exist in the results",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        # Check if it's actually a file (not a directory)
        if not requested_path.is_file():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Not a file",
                    "details": f"'{filepath}' is not a file",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )

        # Determine media type based on file extension
        import mimetypes

        media_type, _ = mimetypes.guess_type(requested_path)
        if media_type is None:
            media_type = "application/octet-stream"

        file_size = requested_path.stat().st_size
        range_header = request.headers.get("Range")
        start, end, content_length, range_status = parse_range_header(
            range_header, file_size
        )
        stream_generator = create_file_stream(
            requested_path, start, content_length, "file"
        )

        headers = {
            "Content-Disposition": f'attachment; filename="{requested_path.name}"',
            "Content-Length": str(content_length),
            "Accept-Ranges": "bytes",
        }
        if range_status == 206:
            headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"

        return StreamingResponse(
            stream_generator,
            media_type=media_type,
            headers=headers,
            status_code=range_status,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            f"Failed to get file {filepath} for {workflow_name}:{execution_id}"
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to retrieve file",
                "details": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=config.server.host,  # noqa: S104
        port=config.server.port,
        workers=config.server.workers,
    )
