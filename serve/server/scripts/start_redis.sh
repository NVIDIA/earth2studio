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

# Earth2Studio Redis Startup Script
# This script starts Redis locally as a process (not Docker)

echo "Starting Redis for Earth2Studio API..."

# Check if Redis is already running
if pgrep -x "redis-server" > /dev/null; then
    echo "Redis is already running (PID: $(pgrep -x redis-server))"
    echo "Redis is accessible on localhost:6379"
    exit 0
fi

# Create data directory if it doesn't exist
REDIS_DATA_DIR="./redis-data"
mkdir -p "$REDIS_DATA_DIR"

# Start Redis server with our configuration
echo "Starting Redis server with persistence enabled..."
redis-server ./scripts/redis.conf --dir "$REDIS_DATA_DIR"

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
for i in {1..30}; do
    if redis-cli ping > /dev/null 2>&1; then
        echo "Redis is ready!"
        echo "Redis is running on localhost:6379"
        echo "Data is persisted to $REDIS_DATA_DIR"
        echo "Redis PID: $(pgrep -x redis-server)"
        exit 0
    fi
    sleep 1
done

echo "Error: Redis failed to start within 30 seconds"
exit 1
