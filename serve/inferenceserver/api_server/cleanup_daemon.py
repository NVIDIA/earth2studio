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
Cleanup Daemon for API Server

This daemon runs as a separate process and periodically cleans up expired
inference results based on the configured TTL (time-to-live).
"""

import json
import logging
import shutil
import signal
import sys
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis  # type: ignore[import-untyped]

from api_server.config import get_config, get_config_manager
from api_server.workflow import WorkflowStatus

# Get configuration
config = get_config()
config_manager = get_config_manager()

# Configure logging
config_manager.setup_logging()
logger = logging.getLogger(__name__)

# Path configuration from config
DEFAULT_OUTPUT_DIR = Path(config.paths.default_output_dir)
RESULTS_ZIP_DIR = Path(config.paths.results_zip_dir)

# Global flag for graceful shutdown
shutdown_flag = False


def signal_handler(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully"""
    global shutdown_flag
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_flag = True


def _delete_result_files(
    redis_client: redis.Redis,
    result_id: str,
    search_id: str,
    workflow_name: str | None = None,
) -> None:
    """
    Delete files for an inference request or workflow execution.

    Args:
        redis_client: Redis client
        result_id: ID to use for zip file lookup (request_id or combined_id)
        search_id: ID to search for in directory names
        workflow_name: Optional workflow name for workflow-specific subdirectory checking
    """
    # Delete zip file
    zip_key = f"inference_request_zips:{result_id}:zip_file"
    zip_filename = redis_client.get(zip_key)
    if zip_filename:
        zip_path = RESULTS_ZIP_DIR / zip_filename
        if zip_path.exists():
            zip_path.unlink()
            logger.info(f"Deleted zip file: {zip_path}")

    # Delete metadata file (metadata_{request_id}.json)
    for item in RESULTS_ZIP_DIR.iterdir():
        if (
            item.is_file()
            and item.name.startswith("metadata")
            and search_id in item.name
        ):
            item.unlink()
            logger.info(f"Deleted metadata file: {item}")

    # Delete raw results directory (search for directories containing search_id)
    for item in DEFAULT_OUTPUT_DIR.iterdir():
        if item.is_dir() and search_id in item.name:
            shutil.rmtree(item)
            logger.info(f"Deleted raw results directory: {item}")

    # Also check subdirectories recursively for results
    for item in RESULTS_ZIP_DIR.iterdir():
        if item.is_dir() and search_id in item.name:
            shutil.rmtree(item)
            logger.info(f"Deleted results from zip dir: {item}")

    # Check in workflow-specific subdirectories if workflow_name is provided
    if workflow_name:
        workflow_dir = DEFAULT_OUTPUT_DIR / workflow_name
        if workflow_dir.exists():
            for item in workflow_dir.iterdir():
                if item.is_dir() and search_id in item.name:
                    shutil.rmtree(item)
                    logger.info(f"Deleted workflow results: {item}")


def _process_expired_key(
    redis_client: redis.Redis,
    key: str,
    current_time: datetime,
    results_ttl_hours: float,
    expected_status: str,
    get_end_time_field: str,
    delete_files_func: Callable[[redis.Redis, str], None],
    log_prefix: str,
) -> bool:
    """
    Process a single Redis key to check if it's expired and clean it up.

    Args:
        redis_client: Redis client
        key: Redis key to process
        current_time: Current timestamp for age calculation
        results_ttl_hours: TTL threshold in hours
        expected_status: Expected status value for completed items
        get_end_time_field: Primary field name for end time
        delete_files_func: Function to call for deleting files
        log_prefix: Prefix for log messages

    Returns:
        True if item was cleaned up, False otherwise
    """
    data_json = redis_client.get(key)
    if not data_json:
        return False

    data = json.loads(data_json)
    status = data.get("status")

    # Only check completed items
    if status != expected_status:
        return False

    # Check for end_time
    end_time_str = data.get(get_end_time_field) or data.get("completion_time")
    if not end_time_str:
        return False

    # Parse the timestamp
    try:
        end_time = datetime.fromisoformat(end_time_str.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        logger.warning(f"Invalid timestamp format for {key}: {end_time_str}")
        return False

    # Calculate age
    age_hours = (current_time - end_time).total_seconds() / 3600

    if age_hours > results_ttl_hours:
        logger.info(
            f"Cleaning up expired {log_prefix}: {key} (age: {age_hours:.2f} hours)"
        )

        # Delete files using the provided function
        delete_files_func(redis_client, key)

        # Update status to expired
        data["status"] = WorkflowStatus.EXPIRED
        redis_client.setex(
            key,
            config.redis.retention_ttl,
            json.dumps(data),
        )

        return True

    return False


def cleanup_expired_results(redis_client: redis.Redis) -> None:
    """
    Check Redis for expired results and clean them up.

    This function:
    1. Checks Redis for completed inference requests older than TTL
    2. Deletes both raw result files and zip files
    3. Updates the status to "expired"
    """
    results_ttl_hours = config.server.results_ttl_hours

    logger.info("Running cleanup watchdog check...")
    current_time = datetime.now(timezone.utc)

    cleaned_count = 0

    # Process custom workflow executions
    workflow_keys = redis_client.keys("workflow_execution:*")
    logger.info(f"Found {len(workflow_keys)} workflow execution keys")

    for key in workflow_keys:
        try:
            # Extract workflow_name and execution_id from key
            # Key format: workflow_execution:{workflow_name}:{execution_id}
            parts = key.split(":")
            if len(parts) >= 3:
                workflow_name = parts[1]
                execution_id = parts[2]
                combined_id = f"{workflow_name}:{execution_id}"

                # Create delete function with computed IDs
                def delete_func(
                    rc: redis.Redis,
                    k: str,
                    _combined_id: str = combined_id,
                    _execution_id: str = execution_id,
                    _workflow_name: str = workflow_name,
                ) -> None:
                    _delete_result_files(
                        rc,
                        result_id=_combined_id,
                        search_id=_execution_id,
                        workflow_name=_workflow_name,
                    )

                if _process_expired_key(
                    redis_client=redis_client,
                    key=key,
                    current_time=current_time,
                    results_ttl_hours=results_ttl_hours,
                    expected_status=WorkflowStatus.COMPLETED,
                    get_end_time_field="end_time",
                    delete_files_func=delete_func,
                    log_prefix="workflow",
                ):
                    cleaned_count += 1
        except Exception as e:
            logger.exception(f"Error processing workflow key {key} for cleanup: {e}")
            continue

    if cleaned_count > 0:
        logger.info(f"Cleanup watchdog: cleaned up {cleaned_count} expired result(s)")
    else:
        logger.info("Cleanup watchdog: no expired results found")


def main() -> None:
    """Main daemon loop"""
    global shutdown_flag

    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    logger.info("Starting cleanup daemon...")
    logger.info(f"Results TTL: {config.server.results_ttl_hours} hours")
    logger.info(f"Cleanup interval: {config.server.cleanup_watchdog_sec} seconds")

    # Connect to Redis
    try:
        redis_client = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            db=config.redis.db,
            password=config.redis.password,
            decode_responses=config.redis.decode_responses,
            socket_connect_timeout=config.redis.socket_connect_timeout,
            socket_timeout=config.redis.socket_timeout,
        )
        # Test connection
        redis_client.ping()
        logger.info(f"Connected to Redis at {config.redis.host}:{config.redis.port}")
    except Exception:
        logger.exception("Failed to connect to Redis")
        sys.exit(1)

    # Main daemon loop
    try:
        while not shutdown_flag:
            try:
                cleanup_expired_results(redis_client)
            except Exception as e:
                logger.exception(f"Error in cleanup cycle: {e}")

            # Sleep for the configured interval, checking shutdown flag periodically
            sleep_remaining = config.server.cleanup_watchdog_sec
            while sleep_remaining > 0 and not shutdown_flag:
                sleep_time = min(5, sleep_remaining)  # Check every 5 seconds
                time.sleep(sleep_time)
                sleep_remaining -= sleep_time

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    finally:
        logger.info("Cleanup daemon shutting down...")
        redis_client.close()
        logger.info("Cleanup daemon stopped")


if __name__ == "__main__":
    main()
