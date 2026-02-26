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

# Status tracking variables
REDIS_RUNNING=0
API_RUNNING=0
RQ_RUNNING=0
CLEANUP_RUNNING=0

echo "Services Status"
echo "============================"

# Check Redis status
echo "Redis:"
if pgrep -x "redis-server" > /dev/null; then
    REDIS_PID=$(pgrep -x redis-server)
    echo "  Status: Running (PID: $REDIS_PID)"

    # Check Redis connection
    if redis-cli ping > /dev/null 2>&1; then
        echo "  Connection: OK"
        REDIS_RUNNING=1

        # Get Redis info
        REDIS_INFO=$(redis-cli info server 2>/dev/null | grep -E "(redis_version|uptime_in_seconds|connected_clients|used_memory_human)")
        if [ -n "$REDIS_INFO" ]; then
            echo "  Info:"
            echo "$REDIS_INFO" | sed 's/^/    /'
        fi
    else
        echo "  Connection: Failed"
    fi
else
    echo "  Status: Not running"
fi

echo ""

# Check API workers status
echo "API Workers:"
API_WORKERS=$(pgrep -f "uvicorn.*api_server.main:app")
if [ -n "$API_WORKERS" ]; then
    WORKER_COUNT=$(echo "$API_WORKERS" | wc -l)
    echo "  Status: Running ($WORKER_COUNT workers)"
    echo "  PIDs: $API_WORKERS"
    API_RUNNING=1
else
    echo "  Status: Not running"
fi

echo ""

# Check RQ workers status per queue
RQ_QUEUES=("inference" "result_zip" "object_storage" "finalize_metadata")
RQ_ALL_OK=1

echo "RQ Workers:"
for QUEUE_NAME in "${RQ_QUEUES[@]}"; do
    QUEUE_WORKERS=$(pgrep -f "rq.*worker.*${QUEUE_NAME}" 2>/dev/null)
    if [ -n "$QUEUE_WORKERS" ]; then
        QUEUE_WORKER_COUNT=$(echo "$QUEUE_WORKERS" | wc -l)
        QUEUE_WORKER_PIDS=$(echo "$QUEUE_WORKERS" | tr '\n' ' ')
        echo "  ${QUEUE_NAME}:"
        echo "    Workers: $QUEUE_WORKER_COUNT (PIDs: ${QUEUE_WORKER_PIDS})"
        # Show queue depth if Redis is reachable
        if command -v redis-cli > /dev/null && redis-cli ping > /dev/null 2>&1; then
            QUEUE_DEPTH=$(redis-cli llen "rq:queue:${QUEUE_NAME}" 2>/dev/null)
            if [ -n "$QUEUE_DEPTH" ]; then
                echo "    Queue depth: $QUEUE_DEPTH jobs pending"
            fi
        fi
    else
        echo "  ${QUEUE_NAME}:"
        echo "    Workers: Not running"
        RQ_ALL_OK=0
    fi
done

if [ $RQ_ALL_OK -eq 1 ]; then
    # Verify Redis connectivity for RQ
    if command -v redis-cli > /dev/null && redis-cli ping > /dev/null 2>&1; then
        RQ_RUNNING=1
        # Show total registered RQ workers
        WORKER_INFO=$(redis-cli keys "rq:worker:*" 2>/dev/null | wc -l)
        if [ -n "$WORKER_INFO" ] && [ "$WORKER_INFO" -gt 0 ]; then
            echo "  Registered workers (total): $WORKER_INFO"
        fi
    else
        echo "  Redis Connection: Failed"
    fi
fi

echo ""

# Check Cleanup Daemon status
echo "Cleanup Daemon:"
CLEANUP_DAEMON=$(pgrep -f "python.*cleanup_daemon")
if [ -n "$CLEANUP_DAEMON" ]; then
    echo "  Status: Running"
    echo "  PID: $CLEANUP_DAEMON"
    CLEANUP_RUNNING=1
else
    echo "  Status: Not running"
fi

echo ""

# Calculate overall status as conjunction of all service statuses
if [ $REDIS_RUNNING -eq 1 ] && [ $API_RUNNING -eq 1 ] && [ $RQ_RUNNING -eq 1 ] && [ $CLEANUP_RUNNING -eq 1 ]; then
    OVERALL_STATUS=0  # All services running
else
    OVERALL_STATUS=1  # At least one service failed
fi

# Summary
echo "Summary:"
echo "========"

if [ $REDIS_RUNNING -eq 1 ]; then
    echo "✓ Redis: Running"
else
    echo "✗ Redis: Failed"
fi

if [ $API_RUNNING -eq 1 ]; then
    echo "✓ API Workers: Running"
else
    echo "✗ API Workers: Failed"
fi

if [ $RQ_RUNNING -eq 1 ]; then
    echo "✓ RQ Workers: All queues running"
else
    echo "✗ RQ Workers: One or more queue workers not running"
    for QUEUE_NAME in "${RQ_QUEUES[@]}"; do
        if pgrep -f "rq.*worker.*${QUEUE_NAME}" > /dev/null 2>&1; then
            echo "    ✓ ${QUEUE_NAME}: Running"
        else
            echo "    ✗ ${QUEUE_NAME}: Not running"
        fi
    done
fi

if [ $CLEANUP_RUNNING -eq 1 ]; then
    echo "✓ Cleanup Daemon: Running"
else
    echo "✗ Cleanup Daemon: Failed"
fi

echo ""


if [ $OVERALL_STATUS -eq 0 ]; then
    echo "Overall Status: All services running ✓"
    exit 0
else
    echo "Overall Status: One or more services failed ✗"
    exit 1
fi
