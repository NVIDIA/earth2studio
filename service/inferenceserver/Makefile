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

# Earth2Studio API Makefile
# Provides easy commands for managing Redis and API services

.PHONY: container-earth2studio-scicomp

# Show service status
status:
	@./scripts/status.sh

# Start Redis only
start-redis:
	@echo "Starting Redis..."
	@./scripts/start_redis.sh

# Stop Redis only
stop-redis:
	@echo "Stopping Redis..."
	@./scripts/stop_redis.sh

# Start API workers
start-api-server:
	@echo "Starting API workers..."
	@if [ -f "../../.venv/bin/activate" ]; then \
		. ../../.venv/bin/activate && ./scripts/start_api_server.sh; \
	else \
		./scripts/start_api_server.sh; \
	fi

# Stop API workers
stop-api-server:
	@echo "Stopping API workers..."
	@./scripts/stop_api_server.sh

# Clean up data and logs
clean:
	@echo "Cleaning up Redis data and logs..."
	@rm -rf redis-data/*.rdb redis-data/*.aof
	@echo "Cleanup complete"

# Install dependencies
install:
	@echo "Installing dependencies..."
	@pip install -r requirements.txt

# Run tests
# integration_test:
#	@echo "Running API tests..."
#	@python inferenceserver/test/integration/test_api.py
