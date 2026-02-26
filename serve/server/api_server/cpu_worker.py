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
CPU Worker Functions

This module contains CPU-intensive functions that can be offloaded from the main worker process.

"""

__all__ = [
    "create_results_zip",
    "process_result_zip",
    "process_object_storage_upload",
    "process_finalize_metadata",
]

import json
import logging
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis  # type: ignore[import-untyped]

# Import configuration
from api_server.config import get_config, get_config_manager
from api_server.utils import (
    get_inference_request_metadata_key,
    get_inference_request_output_path_key,
    get_inference_request_zip_key,
    get_results_zip_dir_key,
    get_signed_url_key,
    queue_next_stage,
)
from api_server.workflow import (
    WorkflowResult,
    WorkflowStatus,
    workflow_registry,
)

# Get configuration
config = get_config()
config_manager = get_config_manager()

# Configure logging
config_manager.setup_logging()
logger = logging.getLogger(__name__)

# Redis client for CPU worker
redis_client = redis.Redis(
    host=config.redis.host,
    port=config.redis.port,
    db=config.redis.db,
    password=config.redis.password,
    decode_responses=config.redis.decode_responses,
    socket_connect_timeout=config.redis.socket_connect_timeout,
    socket_timeout=config.redis.socket_timeout,
)

# Register custom workflows in the CPU worker process
try:
    from api_server.workflow import register_all_workflows

    register_all_workflows(redis_client)
    logger.info("Custom workflows registered successfully in CPU worker process")
except ImportError:
    logger.warning(
        "Workflow registration module not found in CPU worker, skipping custom workflow registration"
    )
except Exception as e:
    logger.error(f"Failed to register custom workflows in CPU worker: {e}")
    # Don't raise - worker can still handle other tasks


def fail_workflow(
    workflow_name: str,
    execution_id: str,
    error_message: str,
) -> dict[str, Any]:
    """
    Mark a workflow as failed and return an error dict.

    This is a common helper function used across CPU worker functions to handle
    workflow failures consistently.

    Args:
        workflow_name: Name of the workflow
        execution_id: Execution ID of the workflow
        error_message: Error message describing the failure

    Returns:
        Dict with success=False and error message
    """
    logger.error(error_message)
    try:
        workflow_class = workflow_registry.get_workflow_class(workflow_name)
        if workflow_class:
            updates = {
                "status": WorkflowStatus.FAILED,
                "end_time": datetime.now(timezone.utc).isoformat(),
                "error_message": error_message,
            }
            workflow_class._update_execution_data(
                redis_client, workflow_name, execution_id, updates
            )
    except Exception:
        logger.exception("Failed to update workflow status")
    return {"success": False, "error": error_message}


@dataclass
class FileManifestEntry:
    """Entry in the file manifest describing a file in the zip"""

    path: str
    size: int


@dataclass
class ResultMetadata:
    """Metadata structure for results zip files"""

    request_id: str
    status: str
    completion_time: str | None
    execution_time_seconds: float | None
    workflow_type: str | None = None  # For legacy inference requests
    workflow_name: str | None = None  # For custom workflows
    created_at: str | None = None
    peak_memory_usage: str | None = None
    device: str | None = None
    zip_created_at: str = ""
    parameters: dict[str, Any] | None = None
    output_files: list[FileManifestEntry] = field(default_factory=list)
    # Object storage fields
    storage_type: str = "server"  # "server", "s3", etc.
    signed_url: str | None = None
    remote_path: str | None = None  # Remote path in object storage

    @classmethod
    def from_workflow_result(
        cls,
        workflow_result: WorkflowResult,
        request_id: str,
        file_manifest: list[FileManifestEntry],
        zip_created_at: str,
    ) -> "ResultMetadata":
        """
        Create ResultMetadata from a WorkflowResult (custom workflows).

        Args:
            workflow_result: WorkflowResult instance from custom workflow
            request_id: Request ID (execution_id from workflow)
            file_manifest: List of files in the zip
            zip_created_at: Timestamp when zip was created

        Returns:
            ResultMetadata instance
        """
        workflow_metadata = workflow_result.metadata or {}
        return cls(
            request_id=workflow_result.execution_id or request_id,
            status=workflow_result.status or "completed",
            completion_time=workflow_result.end_time,
            execution_time_seconds=workflow_result.execution_time_seconds,
            workflow_name=workflow_result.workflow_name,
            created_at=workflow_result.start_time,
            peak_memory_usage=workflow_metadata.get("peak_memory_usage"),
            device=workflow_metadata.get("device"),
            zip_created_at=zip_created_at,
            parameters=workflow_metadata.get("parameters"),
            output_files=file_manifest,
        )

    @classmethod
    def from_legacy_dict(
        cls,
        inference_request: dict[str, Any],
        request_id: str,
        file_manifest: list[FileManifestEntry],
        zip_created_at: str,
    ) -> "ResultMetadata":
        """
        Create ResultMetadata from a legacy inference request dict.

        Args:
            inference_request: Legacy inference request dictionary
            request_id: Request ID
            file_manifest: List of files in the zip
            zip_created_at: Timestamp when zip was created

        Returns:
            ResultMetadata instance
        """
        return cls(
            request_id=request_id,
            status=inference_request.get("status", "completed"),
            completion_time=inference_request.get("completion_time"),
            execution_time_seconds=inference_request.get("execution_time_seconds"),
            workflow_type=inference_request.get("type"),
            created_at=inference_request.get("created_at"),
            peak_memory_usage=inference_request.get("peak_memory_usage"),
            device=inference_request.get("device"),
            zip_created_at=zip_created_at,
            parameters=inference_request.get("request"),
            output_files=file_manifest,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        result = {
            "request_id": self.request_id,
            "status": self.status,
            "completion_time": self.completion_time,
            "execution_time_seconds": self.execution_time_seconds,
            "created_at": self.created_at,
            "peak_memory_usage": self.peak_memory_usage,
            "device": self.device,
            "zip_created_at": self.zip_created_at,
            "parameters": self.parameters,
            "output_files": [
                {"path": f.path, "size": f.size} for f in self.output_files
            ],
            "storage_type": self.storage_type,
        }

        # Add either workflow_type or workflow_name
        if self.workflow_name:
            result["workflow_name"] = self.workflow_name
        if self.workflow_type:
            result["workflow_type"] = self.workflow_type

        # Add object storage fields if present
        if self.signed_url:
            result["signed_url"] = self.signed_url
        if self.remote_path:
            result["remote_path"] = self.remote_path

        return result


def build_file_manifest(
    output_path: Path, arc_prefix: str = ""
) -> list[FileManifestEntry]:
    """
    Build a file manifest by recursively scanning output files.

    Args:
        output_path: Path to output files (file or directory)
        arc_prefix: Prefix for archive paths (used for nested directories)

    Returns:
        List of FileManifestEntry objects describing all files
    """
    file_manifest: list[FileManifestEntry] = []

    if not output_path.exists():
        return file_manifest

    if output_path.is_file():
        arc_path = (
            f"{arc_prefix}/{output_path.name}" if arc_prefix else output_path.name
        )
        file_manifest.append(
            FileManifestEntry(path=arc_path, size=output_path.stat().st_size)
        )
    elif output_path.is_dir():
        dir_arc_path = (
            f"{arc_prefix}/{output_path.name}" if arc_prefix else output_path.name
        )
        for item in output_path.iterdir():
            if item.is_file():
                item_arc_path = f"{dir_arc_path}/{item.name}"
                file_manifest.append(
                    FileManifestEntry(path=item_arc_path, size=item.stat().st_size)
                )
            elif item.is_dir():
                # Recursively scan subdirectories
                file_manifest.extend(build_file_manifest(item, dir_arc_path))

    return file_manifest


def create_results_zip(
    request_id: str,
    output_path: Path,
    inference_request: dict[str, Any] | WorkflowResult,
    results_zip_dir: Path,
    redis_client: redis.Redis,
    create_zip: bool = True,
) -> str | None:
    """Create zip file containing inference results and store it in the results directory

    Args:
        request_id: Request ID (for legacy) or execution_id (for custom workflows)
        output_path: Path to output files
        inference_request: Dict containing request data (legacy) or WorkflowResult (for custom workflows)
        results_zip_dir: Directory to store result zips
        redis_client: Redis client instance
        create_zip: If False, only build file manifest without creating zip file
    """
    # Create logger adapter with execution_id for automatic log prefixing
    log = logging.LoggerAdapter(logger, {"execution_id": request_id})

    try:
        zip_filename: str | None = f"{request_id}"
        zip_path: Path | None = None

        # Build file manifest (always done)
        file_manifest = build_file_manifest(output_path)

        if not output_path.exists():
            logger.warning(f"Output path {output_path} does not exist")

        # Create zip file only if requested
        if create_zip:
            if zip_filename is None:
                zip_filename = f"{request_id}"
            zip_path = results_zip_dir / zip_filename
            logger.info(f"Creating results zip file: {zip_path}")

            with zipfile.ZipFile(
                zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6
            ) as zip_file:
                # Add output files - handle both file and directory outputs
                if output_path.exists():
                    if output_path.is_file():
                        zip_file.write(output_path, output_path.name)
                    elif output_path.is_dir():
                        # Add all files from manifest to zip
                        for entry in file_manifest:
                            # Convert arc_path back to actual file path
                            # arc_path is like "dirname/subdir/file.txt", we need the actual path
                            rel_path = (
                                entry.path.replace(output_path.name + "/", "", 1)
                                if entry.path.startswith(output_path.name + "/")
                                else entry.path
                            )
                            actual_path = (
                                output_path / rel_path
                                if entry.path.startswith(output_path.name)
                                else output_path.parent / entry.path
                            )
                            if actual_path.exists() and actual_path.is_file():
                                zip_file.write(actual_path, entry.path)

            # Add zip file itself to the manifest
            if zip_path is not None and zip_filename is not None:
                zip_size = zip_path.stat().st_size
                file_manifest.append(
                    FileManifestEntry(path=zip_filename, size=zip_size)
                )

                logger.info(f"Created zip file {zip_filename} ({zip_size} bytes)")
        else:
            logger.info(
                f"Skipping zip creation for {request_id}, manifest has {len(file_manifest)} files"
            )
            zip_filename = None  # No zip file created

        # Create metadata using factory methods
        zip_created_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        if isinstance(inference_request, WorkflowResult):
            # Custom workflow (WorkflowResult)
            metadata = ResultMetadata.from_workflow_result(
                workflow_result=inference_request,
                request_id=request_id,
                file_manifest=file_manifest,
                zip_created_at=zip_created_at,
            )
        elif isinstance(inference_request, dict):
            # Legacy inference request (dict)
            metadata = ResultMetadata.from_legacy_dict(
                inference_request=inference_request,
                request_id=request_id,
                file_manifest=file_manifest,
                zip_created_at=zip_created_at,
            )
        else:
            raise TypeError(
                f"inference_request must be Dict or WorkflowResult, got {type(inference_request)}"
            )

        # Store metadata in Redis for the object storage worker to finalize
        metadata_key = get_inference_request_metadata_key(request_id)
        redis_client.setex(
            metadata_key,
            config.redis.retention_ttl,
            json.dumps(metadata.to_dict()),
        )
        # Store results_zip_dir path for the object storage worker
        results_zip_dir_key = get_results_zip_dir_key(request_id)
        redis_client.setex(
            results_zip_dir_key,
            config.redis.retention_ttl,
            str(results_zip_dir),
        )
        logger.info(f"Stored pending metadata in Redis for {request_id}")

        # Store zip filename association in Redis (only if zip was created)
        if zip_filename:
            zip_key = get_inference_request_zip_key(request_id)
            redis_client.setex(zip_key, config.redis.retention_ttl, zip_filename)

        # Store output path association in Redis
        output_path_key = get_inference_request_output_path_key(request_id)
        redis_client.setex(output_path_key, 86400, str(output_path))  # 24 hours

        return zip_filename

    except Exception as e:
        log.error(f"Failed to create results zip: {e}")
        return None


def process_result_zip(
    workflow_name: str,
    execution_id: str,
    output_path_str: str,
    results_zip_dir_str: str,
    create_zip: bool = True,
) -> str | None:
    """
    RQ Worker function to process zip file creation for workflow results.

    This function is intended to be executed by the CPU worker from the result_zip_queue.
    It builds the file manifest and optionally creates the zip file.

    Args:
        workflow_name: Name of the workflow
        execution_id: Execution ID of the workflow
        output_path_str: Path to the output files (as string for serialization)
        results_zip_dir_str: Directory to store result zips (as string for serialization)
        create_zip: If False, only build file manifest without creating zip file

    Returns:
        The zip filename if created, "manifest_only" if skipped, None on error
    """
    output_path = Path(output_path_str)
    results_zip_dir = Path(results_zip_dir_str)

    request_id = f"{workflow_name}:{execution_id}"
    logger.info(f"Processing result zip for {request_id} (create_zip={create_zip})")

    try:
        # Get workflow class from registry
        workflow_class = workflow_registry.get_workflow_class(workflow_name)
        if not workflow_class:
            raise ValueError(f"Workflow '{workflow_name}' not found in registry")

        # Get execution data from Redis for metadata
        execution_data = workflow_class._get_execution_data(
            redis_client, workflow_name, execution_id
        )

        # Create the zip file (or just build manifest if create_zip=False)
        zip_filename = create_results_zip(
            request_id,
            output_path,
            execution_data,
            results_zip_dir,
            redis_client,
            create_zip=create_zip,
        )

        # Success if zip was created OR if we intentionally skipped zip creation
        if zip_filename or not create_zip:
            if zip_filename:
                logger.info(
                    f"Zip file created for {workflow_name}:{execution_id}, queuing next stage"
                )
            else:
                logger.info(
                    f"File manifest built for {workflow_name}:{execution_id} (zip skipped), queuing next stage"
                )

            # Queue next pipeline stage
            job_id = queue_next_stage(
                redis_client=redis_client,
                current_stage="result_zip",
                workflow_name=workflow_name,
                execution_id=execution_id,
                output_path_str=output_path_str,
                results_zip_dir_str=results_zip_dir_str,
            )
            if not job_id:
                fail_workflow(
                    workflow_name, execution_id, "Failed to queue next pipeline stage"
                )
                return None

            return zip_filename if zip_filename else "manifest_only"
        else:
            fail_workflow(
                workflow_name, execution_id, "Failed to create results zip file"
            )
            return None

    except Exception as e:
        logger.exception(f"Failed to process zip for {request_id}")
        fail_workflow(workflow_name, execution_id, f"Zip creation failed: {str(e)}")
        raise


def process_object_storage_upload(
    workflow_name: str,
    execution_id: str,
    output_path_str: str,
) -> dict[str, Any] | None:
    """
    RQ Worker function to finalize metadata and optionally upload workflow results to object storage.

    This function is intended to be executed by the CPU worker from the object_storage_queue.
    It always creates the final metadata file, and optionally uploads to S3 and generates
    a CloudFront signed URL if object storage is enabled.

    Args:
        workflow_name: Name of the workflow
        execution_id: Execution ID of the workflow
        output_path_str: Path to the output files (as string for serialization)

    Returns:
        Dict containing result info, None on critical failure
    """
    output_path = Path(output_path_str)
    request_id = f"{workflow_name}:{execution_id}"

    logger.info(f"Processing object storage worker for {request_id}")

    try:
        # Initialize result tracking
        signed_url = None
        remote_prefix = None
        upload_result = None
        storage_type = "server"  # Default to "server" when object storage is disabled

        # Upload to object storage if enabled

        if config.object_storage.enabled and config.object_storage.bucket:
            from api_server.object_storage import (
                MSCObjectStorage,
                ObjectStorageError,
            )

            # Check output path exists
            if not output_path.exists():
                return fail_workflow(
                    workflow_name,
                    execution_id,
                    f"Output path does not exist: {output_path}",
                )

            # Create S3 storage instance
            storage_kwargs = {
                "bucket": config.object_storage.bucket,
                "region": config.object_storage.region,
                "use_transfer_acceleration": config.object_storage.use_transfer_acceleration,
                "max_concurrency": config.object_storage.max_concurrency,
                "multipart_chunksize": config.object_storage.multipart_chunksize,
                "use_rust_client": config.object_storage.use_rust_client,
            }

            # Add optional credentials
            if (
                config.object_storage.access_key_id
                and config.object_storage.secret_access_key
            ):
                storage_kwargs["access_key_id"] = config.object_storage.access_key_id
                storage_kwargs["secret_access_key"] = (
                    config.object_storage.secret_access_key
                )
            if config.object_storage.session_token:
                storage_kwargs["session_token"] = config.object_storage.session_token
            if config.object_storage.endpoint_url:
                storage_kwargs["endpoint_url"] = config.object_storage.endpoint_url

            # Add CloudFront configuration for signed URLs
            if config.object_storage.cloudfront_domain:
                storage_kwargs["cloudfront_domain"] = (
                    config.object_storage.cloudfront_domain
                )
            if config.object_storage.cloudfront_key_pair_id:
                storage_kwargs["cloudfront_key_pair_id"] = (
                    config.object_storage.cloudfront_key_pair_id
                )
            if config.object_storage.cloudfront_private_key:
                storage_kwargs["cloudfront_private_key"] = (
                    config.object_storage.cloudfront_private_key
                )

            try:
                storage = MSCObjectStorage(**storage_kwargs)
            except Exception as e:
                return fail_workflow(
                    workflow_name,
                    execution_id,
                    f"Failed to create MSC storage client: {e}",
                )

            # Construct remote prefix: {base_prefix}/{workflow_name}/{execution_id}
            remote_prefix = (
                f"{config.object_storage.prefix}/{workflow_name}/{execution_id}"
            )

            # Upload the output directory
            logger.info(
                f"Uploading {output_path} to s3://{config.object_storage.bucket}/{remote_prefix}"
            )

            try:
                upload_result = storage.upload_directory(
                    output_path,
                    remote_prefix,
                    recursive=True,
                    overwrite=True,
                )
            except Exception as e:
                return fail_workflow(
                    workflow_name,
                    execution_id,
                    f"Object storage upload failed for {request_id}: {e}",
                )

            if not upload_result.success:
                return fail_workflow(
                    workflow_name,
                    execution_id,
                    f"Failed to upload to object storage: {upload_result.errors}",
                )

            storage_type = "s3"
            logger.info(
                f"Successfully uploaded {upload_result.files_uploaded} files "
                f"({upload_result.total_bytes} bytes) to {upload_result.destination}"
            )

            # Generate signed URL if CloudFront is configured
            cloudfront_configured = all(
                [
                    config.object_storage.cloudfront_domain,
                    config.object_storage.cloudfront_key_pair_id,
                    config.object_storage.cloudfront_private_key,
                ]
            )

            if not cloudfront_configured:
                logger.info("CloudFront not configured, skipping signed URL generation")
            else:
                try:
                    signed_url_path = f"{remote_prefix}/*"
                    signed_url = storage.generate_signed_url(
                        signed_url_path,
                        expires_in=config.object_storage.signed_url_expires_in,
                    )
                    logger.info(f"Generated signed URL for {request_id}")

                    # Store signed URL in Redis
                    signed_url_key = get_signed_url_key(request_id)
                    redis_client.setex(
                        signed_url_key,
                        config.object_storage.signed_url_expires_in,
                        signed_url,
                    )
                except ObjectStorageError as e:
                    return fail_workflow(
                        workflow_name,
                        execution_id,
                        f"Failed to generate signed URL: {e}",
                    )
        else:
            logger.info("Object storage not enabled, skipping upload")

        # Store object storage info in Redis for the finalize worker
        storage_info = {
            "storage_type": storage_type,
        }
        if storage_type == "s3" and remote_prefix:
            storage_info["remote_path"] = (
                f"s3://{config.object_storage.bucket}/{remote_prefix}"
            )
        if signed_url:
            storage_info["signed_url"] = signed_url

        storage_info_key = f"inference_request:{request_id}:storage_info"
        redis_client.setex(
            storage_info_key,
            config.redis.retention_ttl,
            json.dumps(storage_info),
        )

        # Queue next pipeline stage (finalize metadata)
        job_id = queue_next_stage(
            redis_client=redis_client,
            current_stage="object_storage",
            workflow_name=workflow_name,
            execution_id=execution_id,
            output_path_str=output_path_str,
        )
        if not job_id:
            return fail_workflow(
                workflow_name,
                execution_id,
                f"Failed to queue next pipeline stage for {workflow_name}:{execution_id}",
            )

        logger.info(
            f"Queued next stage for {workflow_name}:{execution_id} "
            f"with RQ job ID: {job_id}"
        )

        # Build result
        result = {
            "success": True,
            "storage_type": storage_type,
            "signed_url": signed_url,
        }

        if upload_result and storage_type == "s3":
            result["files_uploaded"] = upload_result.files_uploaded
            result["total_bytes"] = upload_result.total_bytes
            result["destination"] = upload_result.destination
            result["remote_prefix"] = remote_prefix

        logger.info(f"Object storage worker completed for {request_id}")
        return result

    except Exception as e:
        logger.exception(f"Failed in object storage worker for {request_id}")
        return fail_workflow(
            workflow_name,
            execution_id,
            f"Object storage worker failed for {request_id}: {str(e)}",
        )


def process_finalize_metadata(
    workflow_name: str,
    execution_id: str,
) -> dict[str, Any] | None:
    """
    RQ Worker function to finalize metadata file and set workflow status to COMPLETED.

    This function is intended to be executed by the CPU worker from the finalize_metadata_queue.
    It retrieves pending metadata and storage info from Redis, writes the final metadata file,
    and updates the workflow status to COMPLETED.

    Args:
        workflow_name: Name of the workflow
        execution_id: Execution ID of the workflow

    Returns:
        Dict containing result info, None on critical failure
    """
    from api_server.workflow import WorkflowStatus, workflow_registry

    request_id = f"{workflow_name}:{execution_id}"
    logger.info(f"Processing finalize metadata for {request_id}")

    # Retrieve pending metadata and storage info from Redis
    metadata_key = get_inference_request_metadata_key(request_id)
    results_zip_dir_key = get_results_zip_dir_key(request_id)
    storage_info_key = f"inference_request:{request_id}:storage_info"

    pending_metadata_json = redis_client.get(metadata_key)
    results_zip_dir_str = redis_client.get(results_zip_dir_key)
    storage_info_json = redis_client.get(storage_info_key)

    if not pending_metadata_json or not results_zip_dir_str:
        logger.error(f"Pending metadata not found in Redis for {request_id}")
        return fail_workflow(
            workflow_name,
            execution_id,
            f"Pending metadata not found in Redis for {request_id}",
        )

    try:
        # Parse data from Redis
        metadata_dict = json.loads(pending_metadata_json)
        results_zip_dir = Path(results_zip_dir_str)

        # Apply storage info if available
        if storage_info_json:
            storage_info = json.loads(storage_info_json)
            metadata_dict["storage_type"] = storage_info.get("storage_type", "server")
            if storage_info.get("remote_path"):
                metadata_dict["remote_path"] = storage_info["remote_path"]
            if storage_info.get("signed_url"):
                metadata_dict["signed_url"] = storage_info["signed_url"]
        else:
            # No storage info means object storage was skipped or failed
            metadata_dict["storage_type"] = "server"

        # Write final metadata file
        metadata_filename = f"metadata_{request_id}.json"
        metadata_path = results_zip_dir / metadata_filename
        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        logger.info(f"Created final metadata file: {metadata_path}")

        # Update workflow status to COMPLETED
        workflow_class = workflow_registry.get_workflow_class(workflow_name)
        if not workflow_class:
            return fail_workflow(
                workflow_name,
                execution_id,
                f"Workflow class not found for {workflow_name}",
            )

        updates = {
            "status": WorkflowStatus.COMPLETED,
            "end_time": datetime.now(timezone.utc).isoformat(),
        }
        workflow_class._update_execution_data(
            redis_client, workflow_name, execution_id, updates
        )
        logger.info(
            f"Workflow {workflow_name} execution {execution_id} status set to COMPLETED"
        )

        # Clean up Redis keys
        redis_client.delete(metadata_key)
        redis_client.delete(results_zip_dir_key)
        if storage_info_json:
            redis_client.delete(storage_info_key)

        logger.info(f"Finalize metadata completed for {request_id}")
        return {
            "success": True,
            "metadata_path": str(metadata_path),
            "storage_type": metadata_dict.get("storage_type", "server"),
        }

    except Exception as e:
        logger.exception(f"Failed to finalize metadata for {request_id}")
        return fail_workflow(
            workflow_name, execution_id, f"Metadata finalization failed: {str(e)}"
        )
