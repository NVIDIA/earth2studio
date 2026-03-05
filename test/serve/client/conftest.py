#!/usr/bin/env python3
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
Pytest configuration for earth2studio.serve.client tests.

This conftest.py ensures that the serve/client directory is on the Python path
so that earth2studio.serve.client can be imported when running tests from the repository root
with: uv run -m pytest test/serve/client

Integration tests require a running Earth2Studio API server and the
EARTH2STUDIO_API_URL environment variable to be set.
"""

import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

# Add serve/client to Python path so earth2studio.serve.client can be imported
_client_path = Path(__file__).resolve().parent.parent.parent.parent / "serve" / "client"
if str(_client_path) not in sys.path:
    sys.path.insert(0, str(_client_path))

# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config: Any) -> None:
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "integration: mark test as an integration test requiring API server",
    )


# =============================================================================
# Integration Test Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def api_url() -> str | None:
    """
    Get API URL from environment for integration tests.

    Returns None if not set (integration tests will be skipped).
    """
    return os.getenv("EARTH2STUDIO_API_URL")


@pytest.fixture(scope="session")
def check_api_availability(api_url: str | None) -> Iterator[str]:
    """
    Check if API is available before running integration tests.

    This fixture should be used by integration tests to ensure
    the API server is running and accessible.
    """
    if not api_url:
        pytest.skip(
            "EARTH2STUDIO_API_URL environment variable not set. "
            "Set it to run integration tests: export EARTH2STUDIO_API_URL=http://localhost:8000"
        )

    # Verify API is reachable
    try:
        import requests  # type: ignore[import-untyped]

        response = requests.get(f"{api_url}/health", timeout=10)
        if response.status_code != 200:
            pytest.skip(f"API health check failed: {response.status_code}")
    except Exception as e:
        pytest.skip(f"Cannot connect to API at {api_url}: {e}")

    assert api_url is not None  # narrow type for mypy after skip
    yield api_url


# =============================================================================
# Unit Test Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_health_response() -> dict[str, Any]:
    """Fixture providing a mock health check response"""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00+00:00",
    }


@pytest.fixture
def mock_inference_response() -> dict[str, Any]:
    """Fixture providing a mock inference request response"""
    return {
        "execution_id": "exec_test_123",
        "status": "accepted",
        "message": "Request accepted for processing",
        "timestamp": "2024-01-01T00:00:00+00:00",
    }


@pytest.fixture
def mock_status_response() -> dict[str, Any]:
    """Fixture providing a mock status response"""
    return {
        "execution_id": "exec_test_123",
        "status": "running",
        "progress": {
            "progress": "Processing step 5 of 10",
            "current_step": 5,
            "total_steps": 10,
        },
        "error_message": None,
    }


@pytest.fixture
def mock_results_response() -> dict[str, Any]:
    """Fixture providing a mock results response"""
    return {
        "request_id": "exec_test_123",
        "status": "completed",
        "output_files": [
            {"path": "output.zarr", "size": 1024000},
            {"path": "metadata.json", "size": 512},
        ],
        "completion_time": "2024-01-01T00:10:00+00:00",
        "execution_time_seconds": 600.0,
    }
