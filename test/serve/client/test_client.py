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
Unit tests for Earth2StudioClient.
"""

import io
import json
from datetime import datetime
from unittest.mock import Mock, patch

import pytest
import requests  # type: ignore[import-untyped]

from earth2studio.serve.client.client import Earth2StudioClient
from earth2studio.serve.client.exceptions import (
    BadRequestError,
    Earth2StudioAPIError,
    InferenceRequestNotFoundError,
    InternalServerError,
    RequestTimeoutError,
)
from earth2studio.serve.client.exceptions import (
    ConnectionError as ClientConnectionError,
)
from earth2studio.serve.client.models import (
    HealthStatus,
    InferenceRequest,
    InferenceRequestResponse,
    InferenceRequestResults,
    InferenceRequestStatus,
    OutputFile,
    RequestStatus,
)


class TestEarth2StudioClientInitialization:
    """Test Earth2StudioClient initialization"""

    def test_client_initialization(self) -> None:
        """Test client initialization with default and custom parameters"""
        # Test defaults
        client = Earth2StudioClient()
        assert client.base_url == "http://localhost:8000"
        assert client.workflow_name == "deterministic_earth2_workflow"
        assert client.timeout == 30.0
        assert client.session is not None

        # Test custom params and trailing slash stripping
        client = Earth2StudioClient(
            base_url="https://api.example.com:8080/",
            workflow_name="custom_workflow",
            timeout=60.0,
        )
        assert client.base_url == "https://api.example.com:8080"
        assert client.workflow_name == "custom_workflow"
        assert client.timeout == 60.0


class TestEarth2StudioClientHealthCheck:
    """Test health_check method"""

    def test_health_check(self) -> None:
        """Test health check success and error handling"""
        client = Earth2StudioClient()

        # Test success
        mock_response = {
            "status": "healthy",
            "timestamp": "2024-01-01T00:00:00+00:00",
        }
        with patch.object(client, "_make_request", return_value=mock_response):
            health = client.health_check()
            assert isinstance(health, HealthStatus)
            assert health.status == "healthy"

        # Test error
        with patch.object(
            client, "_make_request", side_effect=Earth2StudioAPIError("API error")
        ):
            with pytest.raises(Earth2StudioAPIError, match="API error"):
                client.health_check()


class TestEarth2StudioClientInferenceRequest:
    """Test submit_inference_request method"""

    def test_submit_inference_request(self) -> None:
        """Test inference request submission with success and error cases"""
        client = Earth2StudioClient()
        request = InferenceRequest(
            parameters={"forecast_times": ["2024-01-01T00:00:00"]}
        )

        # Test success
        mock_response = {
            "execution_id": "exec_123",
            "status": "accepted",
            "message": "Request accepted",
            "timestamp": "2024-01-01T00:00:00+00:00",
        }
        with patch.object(client, "_make_request", return_value=mock_response):
            response = client.submit_inference_request(request)
            assert isinstance(response, InferenceRequestResponse)
            assert response.execution_id == "exec_123"
            assert response.status == RequestStatus.ACCEPTED

        # Test bad request error
        with patch.object(
            client,
            "_make_request",
            side_effect=BadRequestError("Invalid parameters"),
        ):
            with pytest.raises(BadRequestError, match="Invalid parameters"):
                client.submit_inference_request(request)


class TestEarth2StudioClientGetRequestStatus:
    """Test get_request_status method"""

    def test_get_request_status(self) -> None:
        """Test status retrieval for various states"""
        client = Earth2StudioClient()

        # Test running status with progress
        mock_response = {
            "execution_id": "exec_123",
            "status": "running",
            "progress": {
                "progress": "Processing step 5 of 10",
                "current_step": 5,
                "total_steps": 10,
            },
            "error_message": None,
        }
        with patch.object(client, "_make_request", return_value=mock_response):
            status = client.get_request_status("exec_123")
            assert isinstance(status, InferenceRequestStatus)
            assert status.status == RequestStatus.RUNNING
            assert status.progress.current_step == 5

        # Test failed status
        mock_response["status"] = "failed"
        mock_response["error_message"] = "Data source unavailable"
        mock_response["progress"] = None
        with patch.object(client, "_make_request", return_value=mock_response):
            status = client.get_request_status("exec_123")
            assert status.status == RequestStatus.FAILED
            assert status.error_message == "Data source unavailable"

        # Test not found
        with patch.object(
            client,
            "_make_request",
            side_effect=InferenceRequestNotFoundError("Request not found"),
        ):
            with pytest.raises(InferenceRequestNotFoundError):
                client.get_request_status("nonexistent_id")


class TestEarth2StudioClientGetRequestResults:
    """Test get_request_results method"""

    def test_get_request_results(self) -> None:
        """Test results retrieval in various scenarios"""
        client = Earth2StudioClient()

        # Test successful results
        mock_response_data = {
            "request_id": "exec_123",
            "status": "completed",
            "output_files": [
                {"path": "output.zarr", "size": 1024},
                {"path": "metadata.json", "size": 256},
            ],
            "completion_time": "2024-01-01T00:10:00+00:00",
            "execution_time_seconds": 600.0,
        }
        mock_http_response = Mock()
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = mock_response_data

        with patch.object(client, "_make_request", return_value=mock_http_response):
            results = client.get_request_results("exec_123")
            assert isinstance(results, InferenceRequestResults)
            assert results.status == RequestStatus.COMPLETED
            assert len(results.output_files) == 2

        # Test still processing (202)
        mock_http_response.status_code = 202
        mock_http_response.json.return_value = {"message": "Still processing"}
        with patch.object(client, "_make_request", return_value=mock_http_response):
            with pytest.raises(Earth2StudioAPIError, match="[Ss]till processing"):
                client.get_request_results("exec_123")

        # Test custom timeout
        mock_http_response.status_code = 200
        mock_http_response.json.return_value = mock_response_data
        with patch.object(
            client, "_make_request", return_value=mock_http_response
        ) as mock_req:
            client.get_request_results("exec_123", timeout=120.0)
            assert mock_req.call_args[1]["timeout"] == 120.0


class TestEarth2StudioClientWaitForCompletion:
    """Test wait_for_completion method"""

    def test_wait_for_completion(self) -> None:
        """Test waiting for completion with various outcomes"""
        client = Earth2StudioClient()

        # Test success after polling
        status_responses = [
            InferenceRequestStatus(
                execution_id="exec_123",
                status=RequestStatus.RUNNING,
                progress=None,
                error_message=None,
            ),
            InferenceRequestStatus(
                execution_id="exec_123",
                status=RequestStatus.COMPLETED,
                progress=None,
                error_message=None,
            ),
        ]
        mock_results = InferenceRequestResults(
            request_id="exec_123",
            status=RequestStatus.COMPLETED,
            output_files=[],
            completion_time=datetime.now(),
        )
        with (
            patch.object(client, "get_request_status", side_effect=status_responses),
            patch.object(client, "get_request_results", return_value=mock_results),
            patch("time.sleep"),
        ):
            results = client.wait_for_completion("exec_123", poll_interval=1.0)
            assert results.status == RequestStatus.COMPLETED

        # Test timeout
        status_response = InferenceRequestStatus(
            execution_id="exec_123",
            status=RequestStatus.RUNNING,
            progress=None,
            error_message=None,
        )
        with (
            patch.object(client, "get_request_status", return_value=status_response),
            patch("time.sleep"),
            patch("time.time", side_effect=[0, 0, 5, 11]),
        ):
            with pytest.raises(RequestTimeoutError, match="did not complete within"):
                client.wait_for_completion("exec_123", poll_interval=1.0, timeout=10.0)

        # Test failed request
        status_response.status = RequestStatus.FAILED
        status_response.error_message = "Data source error"
        with patch.object(client, "get_request_status", return_value=status_response):
            with pytest.raises(Earth2StudioAPIError, match="failed"):
                client.wait_for_completion("exec_123")

        # Test cancelled request
        status_response.status = RequestStatus.CANCELLED
        with patch.object(client, "get_request_status", return_value=status_response):
            with pytest.raises(Earth2StudioAPIError, match="was cancelled"):
                client.wait_for_completion("exec_123")


class TestEarth2StudioClientDownloadResult:
    """Test download_result and related methods"""

    def test_download_result(self) -> None:
        """Test downloading result files with custom timeout"""
        client = Earth2StudioClient(workflow_name="test_workflow")

        result = InferenceRequestResults(
            request_id="exec_123",
            status=RequestStatus.COMPLETED,
            output_files=[OutputFile(path="output.zarr", size=1024)],
            completion_time=datetime.now(),
        )

        # Test result_root_path
        path = client.result_root_path(result)
        assert path == "/v1/infer/test_workflow/exec_123/results/"

        # Test download
        mock_http_response = Mock()
        mock_http_response.content = b"mock file content"

        with patch.object(
            client, "_make_request", return_value=mock_http_response
        ) as mock_req:
            data = client.download_result(result, "output.zarr", timeout=300.0)
            assert isinstance(data, io.BytesIO)
            assert data.read() == b"mock file content"
            assert mock_req.call_args[1]["timeout"] == 300.0


class TestEarth2StudioClientMakeRequest:
    """Test _make_request method"""

    def test_make_request_success_and_errors(self) -> None:
        """Test HTTP request handling for various scenarios"""
        client = Earth2StudioClient()

        # Test successful request
        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.return_value = {"result": "success"}
        with patch.object(client.session, "request", return_value=mock_response):
            result = client._make_request("GET", "/test")
            assert result == {"result": "success"}

        # Test timeout
        with patch.object(
            client.session, "request", side_effect=requests.exceptions.Timeout()
        ):
            with pytest.raises(RequestTimeoutError, match="timed out"):
                client._make_request("GET", "/test")

        # Test connection error
        with patch.object(
            client.session,
            "request",
            side_effect=requests.exceptions.ConnectionError("Connection failed"),
        ):
            with pytest.raises(ClientConnectionError, match="Failed to connect"):
                client._make_request("GET", "/test")

    def test_make_request_http_errors(self) -> None:
        """Test HTTP error status code handling"""
        client = Earth2StudioClient()

        # Test 400 Bad Request
        mock_response = Mock()
        mock_response.ok = False
        mock_response.status_code = 400
        mock_response.json.return_value = {"error": "Invalid parameters"}
        with patch.object(client.session, "request", return_value=mock_response):
            with pytest.raises(BadRequestError, match="Invalid parameters"):
                client._make_request("POST", "/test")

        # Test 404 Not Found
        mock_response.status_code = 404
        mock_response.json.return_value = {"error": "Not found"}
        with patch.object(client.session, "request", return_value=mock_response):
            with pytest.raises(InferenceRequestNotFoundError, match="Not found"):
                client._make_request("GET", "/test")

        # Test 500 Internal Server Error
        mock_response.status_code = 500
        mock_response.json.return_value = {"error": "Internal server error"}
        with patch.object(client.session, "request", return_value=mock_response):
            with pytest.raises(InternalServerError, match="Internal server error"):
                client._make_request("GET", "/test")

        # Test invalid JSON response
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        with patch.object(client.session, "request", return_value=mock_response):
            with pytest.raises(Earth2StudioAPIError, match="Invalid JSON response"):
                client._make_request("GET", "/test")

    def test_make_request_return_response_and_stream(self) -> None:
        """Test returning raw response and streaming"""
        client = Earth2StudioClient()

        mock_response = Mock()
        mock_response.ok = True
        mock_response.status_code = 200
        mock_response.content = b"streaming content"

        # Test return_response
        with patch.object(client.session, "request", return_value=mock_response):
            result = client._make_request("GET", "/test", return_response=True)
            assert result is mock_response

        # Test stream with custom timeout
        with patch.object(
            client.session, "request", return_value=mock_response
        ) as mock_req:
            client._make_request(
                "GET", "/test", return_response=True, stream=True, timeout=120.0
            )
            assert mock_req.call_args[1]["stream"] is True
            assert mock_req.call_args[1]["timeout"] == 120.0


class TestEarth2StudioClientContextManager:
    """Test context manager protocol"""

    def test_context_manager(self) -> None:
        """Test using client as context manager"""
        with Earth2StudioClient() as client:
            assert isinstance(client, Earth2StudioClient)

        # Test that session is closed on exit
        client = Earth2StudioClient()
        with patch.object(client.session, "close") as mock_close:
            with client:
                pass
            mock_close.assert_called_once()


class TestEarth2StudioClientParseErrorResponse:
    """Test _parse_error_response method"""

    def test_parse_error_response(self) -> None:
        """Test parsing error responses with valid and invalid JSON"""
        client = Earth2StudioClient()

        # Test valid JSON
        mock_response = Mock()
        mock_response.json.return_value = {"error": "Test error", "details": "Details"}
        result = client._parse_error_response(mock_response)
        assert result == {"error": "Test error", "details": "Details"}

        # Test invalid JSON with text
        mock_response.status_code = 500
        mock_response.text = "Plain text error"
        mock_response.json.side_effect = json.JSONDecodeError("Invalid", "", 0)
        result = client._parse_error_response(mock_response)
        assert result == {"error": "Plain text error"}

        # Test invalid JSON with empty text
        mock_response.text = ""
        result = client._parse_error_response(mock_response)
        assert "HTTP 500 error" in result["error"]
