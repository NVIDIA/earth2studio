#!/bin/bash
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

# Earth2Studio Multiple Workers Startup Script
# This script starts multiple API workers, and RQ workers

# Get the directory where this script is located.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# A way to override the config location while starting up the endpoint.
CONFIG_DIR=${CONFIG_DIR:-"${SCRIPT_DIR}/../api_server/conf"}
CONFIG_FILE="$CONFIG_DIR/config.yaml"

# Function to read config values from YAML using Python
read_config() {
    python3 -c "
import yaml
import sys
try:
    with open('$CONFIG_FILE', 'r') as f:
        config = yaml.safe_load(f)
        # Navigate to the config value
        value = config
        for key in '$1'.split('.'):
            value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                sys.exit(1)
        print(value)
except Exception as e:
    sys.exit(1)
" 2>/dev/null
}

# Read configuration values with fallbacks
# Command line arguments take precedence, then config file, then defaults
if [ -f "$CONFIG_FILE" ]; then
    CONFIG_NUM_WORKERS=$(read_config "server.workers")
    CONFIG_API_PORT=$(read_config "server.port")
    CONFIG_REDIS_HOST=$(read_config "redis.host")
    CONFIG_RQ_NUM_WORKERS=$(read_config "worker.num_workers")
    CONFIG_ZIP_NUM_WORKERS=$(read_config "worker.zip_num_workers")
    CONFIG_OBJSTORE_NUM_WORKERS=$(read_config "worker.objstore_num_workers")
    CONFIG_FINALIZE_NUM_WORKERS=$(read_config "worker.finalize_num_workers")
    CONFIG_PERSISTENT_WORKER=$(read_config "worker.persistent")
fi

NUM_WORKERS=${1:-${CONFIG_NUM_WORKERS:-4}}  # Default to 4 workers
API_PORT=${2:-${SERVER_PORT:-${CONFIG_API_PORT:-8000}}}  # Check env var first, then config file, then default port
REDIS_HOST=${3:-${CONFIG_REDIS_HOST:-localhost}}  # Default Redis host
NUM_RQ_WORKERS=${4:-${CONFIG_RQ_NUM_WORKERS:-1}}  # Default to 1 RQ workers
NUM_ZIP_WORKERS=${5:-${CONFIG_ZIP_NUM_WORKERS:-1}}  # Default to 1 workers for result_zip queue
NUM_OBJSTORE_WORKERS=${CONFIG_OBJSTORE_NUM_WORKERS:-1}  # Default to 1 object storage worker
NUM_FINALIZE_WORKERS=${CONFIG_FINALIZE_NUM_WORKERS:-1}  # Default to 1 finalize metadata worker
PERSISTENT_WORKER=${CONFIG_PERSISTENT_WORKER:-false}

echo "Starting Earth2Studio with $NUM_WORKERS API workers, $NUM_RQ_WORKERS RQ workers, $NUM_ZIP_WORKERS zip workers, $NUM_OBJSTORE_WORKERS object storage workers, and $NUM_FINALIZE_WORKERS finalize workers on port $API_PORT..."
echo "Configuration: Redis=$REDIS_HOST, Persistent Worker=$PERSISTENT_WORKER"

# Function to cleanup on exit
cleanup() {
    echo "Shutting down all workers..."

    # Stop cleanup daemon
    if pgrep -f "python.*cleanup_daemon" > /dev/null; then
        echo "Stopping cleanup daemon..."
        pkill -f "python.*cleanup_daemon"
    fi

    # Stop all API workers
    if pgrep -f "uvicorn.*api_server.main:app" > /dev/null; then
        echo "Stopping API workers..."
        pkill -f "uvicorn.*api_server.main:app"
    fi

    # Stop all RQ inference workers
    if pgrep -f "rq.*worker.*inference" > /dev/null; then
        echo "Stopping RQ inference workers..."
        pkill -f "rq.*worker.*inference"
    fi

    # Stop all zip workers (result_zip queue)
    if pgrep -f "rq.*worker.*result_zip" > /dev/null; then
        echo "Stopping zip workers (result_zip)..."
        pkill -f "rq.*worker.*result_zip"
    fi

    # Stop all object storage workers
    if pgrep -f "rq.*worker.*object_storage" > /dev/null; then
        echo "Stopping object storage workers..."
        pkill -f "rq.*worker.*object_storage"
    fi

    # Stop all finalize metadata workers
    if pgrep -f "rq.*worker.*finalize_metadata" > /dev/null; then
        echo "Stopping finalize metadata workers..."
        pkill -f "rq.*worker.*finalize_metadata"
    fi

    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Set environmental variable to signal API server environment
export EARTH2STUDIO_API_ACTIVE=1

# Start multiple workers using uvicorn with extended timeouts for large file downloads
uvicorn api_server.main:app --host 0.0.0.0 --port $API_PORT --workers $NUM_WORKERS --loop asyncio --timeout-keep-alive 300 --timeout-graceful-shutdown 30 &
UVICORN_PID=$!

# Start RQ workers
if [ "$PERSISTENT_WORKER" = "True" ] || [ "$PERSISTENT_WORKER" = "true" ]; then
    RQ_WORKER_CMD="rq worker -w rq.worker.SimpleWorker inference"
    WORKER_MODE="with SimpleWorker (persistent mode)"
else
    RQ_WORKER_CMD="rq worker inference"
    WORKER_MODE="(standard mode)"
fi

echo "Starting $NUM_RQ_WORKERS RQ workers $WORKER_MODE..."
RQ_WORKER_PIDS=()
for i in $(seq 1 $NUM_RQ_WORKERS); do
    $RQ_WORKER_CMD &
    RQ_WORKER_PIDS+=($!)
    echo "Started RQ worker $i $WORKER_MODE (PID: $!)"
done

# Start zip workers for result_zip queue
echo "Starting $NUM_ZIP_WORKERS zip workers for result_zip queue..."
ZIP_WORKER_PIDS=()
for i in $(seq 1 $NUM_ZIP_WORKERS); do
    rq worker -w rq.worker.SimpleWorker result_zip &
    ZIP_WORKER_PIDS+=($!)
    echo "Started zip worker $i for result_zip queue (PID: $!)"
done

# Start object storage workers
echo "Starting $NUM_OBJSTORE_WORKERS object storage workers..."
OBJSTORE_WORKER_PIDS=()
for i in $(seq 1 $NUM_OBJSTORE_WORKERS); do
    rq worker -w rq.worker.SimpleWorker object_storage &
    OBJSTORE_WORKER_PIDS+=($!)
    echo "Started object storage worker $i (PID: $!)"
done

# Start finalize metadata workers
echo "Starting $NUM_FINALIZE_WORKERS finalize metadata workers..."
FINALIZE_WORKER_PIDS=()
for i in $(seq 1 $NUM_FINALIZE_WORKERS); do
    rq worker -w rq.worker.SimpleWorker finalize_metadata &
    FINALIZE_WORKER_PIDS+=($!)
    echo "Started finalize metadata worker $i (PID: $!)"
done

# Start cleanup daemon
python -m api_server.cleanup_daemon &
CLEANUP_DAEMON_PID=$!
echo "Started cleanup daemon (PID: $CLEANUP_DAEMON_PID)"

# Wait for workers to start
sleep 5

# Check if API workers are running
API_WORKER_COUNT=$(pgrep -f "uvicorn.*api_server.main:app" | wc -l)
if [ "$API_WORKER_COUNT" -eq 0 ]; then
    echo "Failed to start API workers..."
    exit 1
fi

# Check if RQ workers are running (includes both inference and result_zip workers)
RQ_WORKER_COUNT=$(pgrep -f "rq.*worker.*inference" | wc -l)
if [ "$RQ_WORKER_COUNT" -eq 0 ]; then
    echo "Failed to start RQ inference workers..."
    exit 1
fi

# Check if zip workers are running
ZIP_WORKER_COUNT=$(pgrep -f "rq.*worker.*result_zip" | wc -l)
if [ "$ZIP_WORKER_COUNT" -eq 0 ]; then
    echo "Failed to start zip workers for result_zip queue..."
    exit 1
fi

# Check if object storage workers are running
OBJSTORE_WORKER_COUNT=$(pgrep -f "rq.*worker.*object_storage" | wc -l)
if [ "$OBJSTORE_WORKER_COUNT" -eq 0 ]; then
    echo "Failed to start object storage workers..."
    exit 1
fi

# Check if finalize metadata workers are running
FINALIZE_WORKER_COUNT=$(pgrep -f "rq.*worker.*finalize_metadata" | wc -l)
if [ "$FINALIZE_WORKER_COUNT" -eq 0 ]; then
    echo "Failed to start finalize metadata workers..."
    exit 1
fi

# Check if cleanup daemon is running
CLEANUP_DAEMON_COUNT=$(pgrep -f "python.*cleanup_daemon" | wc -l)
if [ "$CLEANUP_DAEMON_COUNT" -eq 0 ]; then
    echo "Failed to start cleanup daemon..."
    exit 1
fi

echo "All services started successfully!"
echo "Uvicorn PID: $UVICORN_PID"
echo "RQ Worker PIDs: ${RQ_WORKER_PIDS[*]}"
echo "Zip Worker PIDs: ${ZIP_WORKER_PIDS[*]}"
echo "Object Storage Worker PIDs: ${OBJSTORE_WORKER_PIDS[*]}"
echo "Finalize Metadata Worker PIDs: ${FINALIZE_WORKER_PIDS[*]}"
echo "Cleanup Daemon PID: $CLEANUP_DAEMON_PID"
echo "Active API workers: $API_WORKER_COUNT"
echo "Active RQ inference workers: $RQ_WORKER_COUNT"
echo "Active zip workers: $ZIP_WORKER_COUNT"
echo "Active object storage workers: $OBJSTORE_WORKER_COUNT"
echo "Active finalize metadata workers: $FINALIZE_WORKER_COUNT"
echo "API available at http://localhost:$API_PORT"
echo "API docs at http://localhost:$API_PORT/docs"

# Invoke example_user_workflow to warm up the server
# Wait for health check to pass before invoking warmup workflow
echo ""
echo "Waiting for health check to pass..."
MAX_HEALTH_RETRIES=30
HEALTH_RETRY_INTERVAL=2
for i in $(seq 1 $MAX_HEALTH_RETRIES); do
    if curl -s "http://localhost:$API_PORT/health" | grep -q '"status":"healthy"'; then
        echo "Health check passed!"
        break
    fi
    if [ $i -eq $MAX_HEALTH_RETRIES ]; then
        echo "Warning: Health check did not pass after $MAX_HEALTH_RETRIES attempts"
        break
    fi
    echo "Health check attempt $i/$MAX_HEALTH_RETRIES failed, retrying in ${HEALTH_RETRY_INTERVAL}s..."
    sleep $HEALTH_RETRY_INTERVAL
done

# Invoke example_user_workflow to warm up the server
echo "Invoking example_user_workflow to warm up the server..."
curl -s -X POST "http://localhost:$API_PORT/v1/infer/example_user_workflow" \
    -H "Content-Type: application/json" \
    -d '{"parameters": {"task_name": "warmup", "num_iterations": 1, "delay_seconds": 0.1, "generate_output": false}}' \
    && echo "" && echo "Example workflow invoked successfully!" \
    || echo "Warning: Failed to invoke example workflow"
