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

echo "Stopping API Server..."

# Stop API workers
echo "Stopping API workers..."
pkill -f "uvicorn.*api_server.main:app"

# Stop RQ inference workers
echo "Stopping RQ inference workers..."
pkill -f "rq.*worker.*inference"

# Stop zip workers (result_zip queue)
echo "Stopping zip workers..."
pkill -f "rq.*worker.*result_zip"

# Stop object storage workers
echo "Stopping object storage workers..."
pkill -f "rq.*worker.*object_storage"

# Stop finalize metadata workers
echo "Stopping finalize metadata workers..."
pkill -f "rq.*worker.*finalize_metadata"

# Stop cleanup daemon
echo "Stopping cleanup daemon..."
pkill -f "python.*cleanup_daemon"

echo "All services stopped."
exit 0
