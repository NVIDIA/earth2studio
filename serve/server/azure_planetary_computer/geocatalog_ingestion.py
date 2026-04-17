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
import logging
from typing import Any

from azure_planetary_computer.pc_client import (
    PLANETARY_COMPUTER_CLIENT_WORKFLOWS,
    PlanetaryComputerClient,
)
from earth2studio.serve.server.config import get_config
from earth2studio.serve.server.cpu_worker import (
    fail_workflow,
    redis_client,
)
from earth2studio.serve.server.utils import (
    get_inference_request_metadata_key,
    queue_next_stage,
)

logger = logging.getLogger(__name__)


def _merge_geocatalog_ids_into_storage_info(
    request_id: str,
    collection_id: str,
    stac_feature_id: str,
) -> None:
    """Persist GeoCatalog collection and STAC feature IDs into Redis ``storage_info`` for finalize."""
    key = f"inference_request:{request_id}:storage_info"
    raw = redis_client.get(key)
    if not raw:
        logger.warning(
            "storage_info missing for %s; cannot attach GeoCatalog IDs", request_id
        )
        return
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Invalid storage_info JSON for %s", request_id)
        return
    info["geocatalog_collection_id"] = collection_id
    info["geocatalog_stac_feature_id"] = stac_feature_id
    config = get_config()
    redis_client.setex(key, config.redis.retention_ttl, json.dumps(info))


def process_geocatalog_ingestion(
    workflow_name: str,
    execution_id: str,
) -> dict[str, Any] | None:
    """Trigger GeoCatalog / Planetary Computer ingestion when the workflow request included ``geo_catalog_url``.

    Intended to run from the geocatalog_ingestion_queue after process_object_storage_upload.
    Reads storage info and parameters from Redis and calls the Planetary Computer client to
    create a STAC feature for the uploaded netcdf blob.

    Parameters
    ----------
    workflow_name : str
        Name of the workflow
    execution_id : str
        Execution ID of the workflow

    Returns
    -------
    dict[str, Any] | None
        Result dict with ``success``; on successful ingestion also
        ``geocatalog_collection_id`` and ``geocatalog_stac_feature_id`` when available.
        Those values are merged into Redis ``storage_info`` and written to the final
        workflow metadata file by ``process_finalize_metadata``. None on critical failure.
    """
    request_id = f"{workflow_name}:{execution_id}"
    logger.info(f"Processing geocatalog ingestion for {request_id}")

    try:
        storage_info_key = f"inference_request:{request_id}:storage_info"
        metadata_key = get_inference_request_metadata_key(request_id)
        storage_info_json = redis_client.get(storage_info_key)
        pending_metadata_json = redis_client.get(metadata_key)

        if not storage_info_json or not pending_metadata_json:
            logger.warning(
                f"Storage info or pending metadata missing for {request_id}, skipping geocatalog ingestion"
            )
            job_id = queue_next_stage(
                redis_client=redis_client,
                current_stage="geocatalog_ingestion",
                workflow_name=workflow_name,
                execution_id=execution_id,
                output_path_str="",
            )
            if not job_id:
                return fail_workflow(
                    workflow_name,
                    execution_id,
                    f"Failed to queue finalize_metadata for {request_id}",
                )
            return {
                "success": True,
                "skipped": True,
                "reason": "missing storage/metadata",
            }

        storage_info = json.loads(storage_info_json)
        metadata_dict = json.loads(pending_metadata_json)
        parameters = metadata_dict.get("parameters") or {}
        geocatalog_url = storage_info.get("geo_catalog_url") or parameters.get(
            "geo_catalog_url"
        )
        if not geocatalog_url:
            logger.warning(
                f"geo_catalog_url missing for {request_id}, skipping geocatalog ingestion"
            )
            job_id = queue_next_stage(
                redis_client=redis_client,
                current_stage="geocatalog_ingestion",
                workflow_name=workflow_name,
                execution_id=execution_id,
                output_path_str="",
            )
            if not job_id:
                return fail_workflow(
                    workflow_name,
                    execution_id,
                    f"Failed to queue finalize_metadata for {request_id}",
                )
            return {
                "success": True,
                "skipped": True,
                "reason": "geo_catalog_url not set",
            }

        blob_url = storage_info.get("blob_url")

        if not blob_url:
            logger.warning(
                f"No blob_url in storage info for {request_id} (e.g. not Azure or no .nc/.zarr "
                f"dataset), skipping geocatalog ingestion"
            )
            job_id = queue_next_stage(
                redis_client=redis_client,
                current_stage="geocatalog_ingestion",
                workflow_name=workflow_name,
                execution_id=execution_id,
                output_path_str="",
            )
            if not job_id:
                return fail_workflow(
                    workflow_name,
                    execution_id,
                    f"Failed to queue finalize_metadata for {request_id}",
                )
            return {"success": True, "skipped": True, "reason": "no blob_url"}

        logger.info(f"Blob URL: {blob_url}")
        if workflow_name not in PLANETARY_COMPUTER_CLIENT_WORKFLOWS:
            logger.info(
                f"Workflow {workflow_name} not supported by Planetary Computer client, skipping ingestion for {request_id}"
            )
            job_id = queue_next_stage(
                redis_client=redis_client,
                current_stage="geocatalog_ingestion",
                workflow_name=workflow_name,
                execution_id=execution_id,
                output_path_str="",
            )
            if not job_id:
                return fail_workflow(
                    workflow_name,
                    execution_id,
                    f"Failed to queue finalize_metadata for {request_id}",
                )
            return {
                "success": True,
                "skipped": True,
                "reason": "workflow not supported",
            }

        geocatalog_collection_id_out: str | None = None
        geocatalog_stac_feature_id_out: str | None = None
        try:
            pc_client = PlanetaryComputerClient(workflow_name=workflow_name)
            collection_id_param = parameters.get("collection_id")
            resolved_collection_id, stac_feature_id = pc_client.create_feature(
                geocatalog_url=geocatalog_url,
                collection_id=collection_id_param,
                parameters=parameters,
                blob_url=blob_url,
            )
            geocatalog_collection_id_out = resolved_collection_id
            geocatalog_stac_feature_id_out = stac_feature_id
            _merge_geocatalog_ids_into_storage_info(
                request_id,
                resolved_collection_id,
                stac_feature_id,
            )
            logger.info(
                f"GeoCatalog ingestion completed for {request_id} "
                f"(collection_id={resolved_collection_id}, feature_id={stac_feature_id})"
            )
        except Exception as e:
            # Log but do not fail the pipeline; finalize_metadata should still run
            logger.exception(
                f"GeoCatalog ingestion failed for {request_id}: {e}. Queuing finalize_metadata anyway."
            )

        job_id = queue_next_stage(
            redis_client=redis_client,
            current_stage="geocatalog_ingestion",
            workflow_name=workflow_name,
            execution_id=execution_id,
            output_path_str="",
        )
        if not job_id:
            return fail_workflow(
                workflow_name,
                execution_id,
                f"Failed to queue next pipeline stage for {request_id}",
            )
        logger.info(
            f"Queued finalize_metadata for {workflow_name}:{execution_id} with RQ job ID: {job_id}"
        )
        result: dict[str, Any] = {"success": True}
        if geocatalog_collection_id_out is not None:
            result["geocatalog_collection_id"] = geocatalog_collection_id_out
        if geocatalog_stac_feature_id_out is not None:
            result["geocatalog_stac_feature_id"] = geocatalog_stac_feature_id_out
        return result

    except Exception as e:
        logger.exception(f"Failed in geocatalog ingestion for {request_id}")
        return fail_workflow(
            workflow_name,
            execution_id,
            f"Geocatalog ingestion failed for {request_id}: {str(e)}",
        )
