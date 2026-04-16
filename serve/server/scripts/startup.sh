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

set -euo pipefail

# Set EARTH2STUDIO_MODEL_CACHE to use AZUREML_MODEL_DIR if available
if [ -n "${AZUREML_MODEL_DIR:-}" ]; then
    echo "AZUREML_MODEL_DIR: $AZUREML_MODEL_DIR"
    export EARTH2STUDIO_MODEL_CACHE="$AZUREML_MODEL_DIR/${EARTH2STUDIO_MODEL_SUBPATH:-e2s_fcn3_stormscope}"
    echo "--------------------------------"
    echo "EARTH2STUDIO_MODEL_CACHE: $EARTH2STUDIO_MODEL_CACHE"
    ls -la $EARTH2STUDIO_MODEL_CACHE && echo "--------------------------------"
fi

# Use CONFIG_DIR/SCRIPT_DIR from env if set (e.g. in Docker); else resolve from script location
SCRIPT_DIR="${SCRIPT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
SERVE_SERVER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export SCRIPT_DIR
export CONFIG_DIR="${CONFIG_DIR:-$SCRIPT_DIR/../conf}"
export WORKFLOW_DIR="${WORKFLOW_DIR:-}"

# PYTHONPATH (repo root + serve/server for azure_planetary_computer.*) is set in
# scripts/start_api_server.sh before uvicorn and RQ workers start.

cd "$SERVE_SERVER_DIR"
make start-redis
make start-api-server
sleep infinity
