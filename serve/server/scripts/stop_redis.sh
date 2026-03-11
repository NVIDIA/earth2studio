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

# Earth2Studio Redis Stop Script
# This script stops the local Redis process

echo "Stopping Redis for Earth2Studio API..."

# Check if Redis is running
if pgrep -x "redis-server" > /dev/null; then
    echo "Stopping Redis server..."
    REDIS_PID=$(pgrep -x redis-server)
    kill "$REDIS_PID"

    # Wait for Redis to stop
    for i in {1..10}; do
        if ! pgrep -x "redis-server" > /dev/null; then
            echo "Redis stopped successfully (PID: $REDIS_PID)"
            exit 0
        fi
        sleep 1
    done

    # Force kill if still running
    if pgrep -x "redis-server" > /dev/null; then
        echo "Force killing Redis..."
        kill -9 "$REDIS_PID"
        echo "Redis force stopped"
    fi
else
    echo "Redis is not running"
fi
