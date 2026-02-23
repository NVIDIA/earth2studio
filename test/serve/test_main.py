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
Unit tests for api_server.main module.

Tests the FastAPI endpoints including the workflow schema endpoint.
"""

import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import Field

# Set API environment variable before importing main
os.environ["EARTH2STUDIO_API_ACTIVE"] = "1"

# Patch FastAPI route creation to handle union return types
# This fixes the issue where FastAPI can't handle dict[str, Any] | StreamingResponse
# ruff: noqa: E402
import fastapi  # type: ignore[import-untyped]
import fastapi.routing  # type: ignore[import-untyped]
from fastapi.exceptions import FastAPIError  # type: ignore[import-untyped]

_original_route_init = fastapi.routing.APIRoute.__init__


def _patched_route_init(self, *args, **kwargs):
    """Patched APIRoute.__init__ that handles union return types with StreamingResponse"""
    import inspect

    from starlette.responses import StreamingResponse

    # Check if the endpoint has a union return type with StreamingResponse
    endpoint = kwargs.get("endpoint") or (args[1] if len(args) > 1 else None)
    if endpoint:
        sig = inspect.signature(endpoint)
        return_annotation = sig.return_annotation
        if (
            hasattr(return_annotation, "__args__")
            and StreamingResponse in return_annotation.__args__
        ):
            # Set response_model=None to skip response model generation
            kwargs["response_model"] = None

    try:
        return _original_route_init(self, *args, **kwargs)
    except FastAPIError as e:
        if "Invalid args for response field" in str(e):
            # Retry with response_model=None
            kwargs["response_model"] = None
            return _original_route_init(self, *args, **kwargs)
        raise


# Apply the patch before any imports of api_server.main
fastapi.routing.APIRoute.__init__ = _patched_route_init


class TestWorkflowSchemaEndpoint:
    """Tests for GET /v1/workflows/{workflow_name}/schema endpoint"""

    @pytest.fixture
    def mock_workflow_class(self):
        """Create a mock workflow class with Parameters"""
        from api_server.workflow import Workflow, WorkflowParameters

        class TestWorkflowParameters(WorkflowParameters):
            """Test parameters for unit testing"""

            forecast_times: list[str] = Field(
                default=["2024-01-01T00:00:00"],
                description="List of forecast initialization times",
            )
            nsteps: int = Field(
                default=10,
                ge=1,
                le=100,
                description="Number of forecast steps",
            )
            model_type: str = Field(
                default="fcn",
                description="Model type to use",
            )

        class TestWorkflow(Workflow):
            name = "test_workflow"
            description = "Test workflow for unit testing"
            Parameters = TestWorkflowParameters

            @classmethod
            def validate_parameters(
                cls, parameters: dict[str, Any] | TestWorkflowParameters
            ) -> TestWorkflowParameters:
                return TestWorkflowParameters.validate(parameters)

            def run(self, parameters, execution_id):
                return {"status": "success"}

        return TestWorkflow

    @pytest.fixture
    def client(self, mock_workflow_class):
        """Create test client with mocked dependencies"""
        # Mock Redis and other dependencies at their source locations
        with (
            patch("redis.asyncio.Redis") as mock_async_redis,
            patch("redis.Redis") as mock_sync_redis,
            patch("rq.Queue"),
            patch("api_server.workflow.register_all_workflows"),
        ):
            # Setup mock async Redis
            mock_async_instance = MagicMock()
            mock_async_instance.ping = AsyncMock(return_value=True)
            mock_async_instance.close = AsyncMock()
            mock_async_instance.get = AsyncMock(return_value=None)
            mock_async_redis.return_value = mock_async_instance

            # Setup mock sync Redis
            mock_sync_instance = MagicMock()
            mock_sync_instance.ping = MagicMock(return_value=True)
            mock_sync_instance.close = MagicMock()
            mock_sync_redis.return_value = mock_sync_instance

            from api_server.main import app
            from api_server.workflow import workflow_registry

            # Register the test workflow
            workflow_registry._workflows["test_workflow"] = mock_workflow_class

            with TestClient(app, raise_server_exceptions=False) as client:
                yield client

            # Cleanup
            if "test_workflow" in workflow_registry._workflows:
                del workflow_registry._workflows["test_workflow"]

    def test_schema_endpoint_returns_valid_json_schema(
        self, client, mock_workflow_class
    ):
        """Test that schema endpoint returns valid JSON Schema"""
        response = client.get("/v1/workflows/test_workflow/schema")

        assert response.status_code == 200
        schema = response.json()

        # Verify it's a valid JSON Schema structure
        assert "type" in schema
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "title" in schema

    def test_schema_endpoint_includes_all_parameters(self, client, mock_workflow_class):
        """Test that schema includes all defined parameters"""
        response = client.get("/v1/workflows/test_workflow/schema")

        assert response.status_code == 200
        schema = response.json()

        properties = schema["properties"]

        # Check all parameters are present
        assert "forecast_times" in properties
        assert "nsteps" in properties
        assert "model_type" in properties

    def test_schema_endpoint_includes_field_metadata(self, client, mock_workflow_class):
        """Test that schema includes field descriptions and constraints"""
        response = client.get("/v1/workflows/test_workflow/schema")

        assert response.status_code == 200
        schema = response.json()

        properties = schema["properties"]

        # Check forecast_times has correct type
        assert properties["forecast_times"]["type"] == "array"
        assert "description" in properties["forecast_times"]

        # Check nsteps has constraints
        nsteps = properties["nsteps"]
        assert nsteps["type"] == "integer"
        assert "description" in nsteps
        # Pydantic uses "minimum" and "maximum" for ge/le constraints
        assert nsteps.get("minimum") == 1 or nsteps.get("exclusiveMinimum") == 0
        assert nsteps.get("maximum") == 100 or nsteps.get("exclusiveMaximum") == 101

    def test_schema_endpoint_includes_defaults(self, client, mock_workflow_class):
        """Test that schema includes default values"""
        response = client.get("/v1/workflows/test_workflow/schema")

        assert response.status_code == 200
        schema = response.json()

        properties = schema["properties"]

        # Check defaults are included
        assert properties["forecast_times"]["default"] == ["2024-01-01T00:00:00"]
        assert properties["nsteps"]["default"] == 10
        assert properties["model_type"]["default"] == "fcn"

    def test_schema_endpoint_workflow_not_found(self, client):
        """Test that 404 is returned for non-existent workflow"""
        response = client.get("/v1/workflows/nonexistent_workflow/schema")

        assert response.status_code == 404
        error = response.json()
        assert "not found" in error["detail"].lower()

    def test_schema_endpoint_openapi_compatibility(self, client, mock_workflow_class):
        """Test that returned schema is OpenAPI 3.1 compatible (JSON Schema draft 2020-12)"""
        response = client.get("/v1/workflows/test_workflow/schema")

        assert response.status_code == 200
        schema = response.json()

        # OpenAPI 3.1 / JSON Schema required fields
        assert "type" in schema
        assert "properties" in schema

        # Schema should be directly usable (no wrapper objects)
        assert "workflow_name" not in schema
        assert "parameters_schema" not in schema


class TestWorkflowParameterValidation:
    """Test workflow parameter validation at submission time"""

    @pytest.fixture
    def client(self):
        """Create test client"""
        from typing import Any, Literal
        from unittest.mock import MagicMock, patch

        from api_server.main import app
        from api_server.workflow import (
            Workflow,
            WorkflowParameters,
            workflow_registry,
        )
        from pydantic import Field

        # Create test workflow parameter classes with Literal validation
        class TestDeterministicParams(WorkflowParameters):
            forecast_times: list[str] = Field(default=["2024-01-01T00:00:00"])
            nsteps: int = Field(default=6, ge=1, le=100)
            model_type: Literal["dlwp", "fcn"] = Field(default="fcn")
            data_source: Literal["gfs"] = Field(default="gfs")
            output_format: Literal["zarr"] = Field(default="zarr")
            plot_variable: Literal["t2m", "msl", "u10m", "v10m", "tcwv", "z500"] = (
                Field(default="t2m")
            )

        class TestDiagnosticParams(WorkflowParameters):
            forecast_times: list[str] = Field(default=["2024-01-01T00:00:00"])
            nsteps: int = Field(default=6, ge=1, le=100)
            prognostic_model_type: Literal["dlwp", "fcn"] = Field(default="fcn")
            diagnostic_model_type: Literal["precipitation_afno"] = Field(
                default="precipitation_afno"
            )
            data_source: Literal["gfs"] = Field(default="gfs")
            plot_variable: Literal[
                "tp", "t2m", "msl", "u10m", "v10m", "tcwv", "z500"
            ] = Field(default="tp")

        # Create mock workflow classes
        class TestDeterministicWorkflow(Workflow):
            name = "deterministic_workflow"
            description = "Test deterministic workflow"
            Parameters = TestDeterministicParams

            @classmethod
            def validate_parameters(
                cls, parameters: dict[str, Any] | TestDeterministicParams
            ) -> TestDeterministicParams:
                try:
                    return TestDeterministicParams.validate(parameters)
                except Exception as e:
                    raise ValueError(f"Invalid parameters: {e}") from e

            def run(self, parameters, execution_id):
                return {"status": "success"}

        class TestDiagnosticWorkflow(Workflow):
            name = "diagnostic_workflow"
            description = "Test diagnostic workflow"
            Parameters = TestDiagnosticParams

            @classmethod
            def validate_parameters(
                cls, parameters: dict[str, Any] | TestDiagnosticParams
            ) -> TestDiagnosticParams:
                try:
                    return TestDiagnosticParams.validate(parameters)
                except Exception as e:
                    raise ValueError(f"Invalid parameters: {e}") from e

            def run(self, parameters, execution_id):
                return {"status": "success"}

        # Register test workflows directly in the registry
        workflow_registry._workflows["deterministic_workflow"] = (
            TestDeterministicWorkflow
        )
        workflow_registry._workflows["diagnostic_workflow"] = TestDiagnosticWorkflow

        # Create mocks
        mock_redis = MagicMock()
        mock_queue = MagicMock()
        mock_queue.__len__ = MagicMock(return_value=0)
        mock_queue.max_size = 100

        with patch("api_server.main.redis_sync_client", mock_redis):
            with patch("api_server.main.inference_queue", mock_queue):
                test_client = TestClient(app)
                yield test_client, mock_queue, mock_redis

    def test_valid_deterministic_workflow_parameters(self, client):
        """Test that valid parameters for deterministic workflow are accepted"""
        test_client, mock_queue, mock_redis = client

        # Setup mock job
        mock_job = MagicMock()
        mock_job.id = "test_job"
        mock_queue.enqueue = MagicMock(return_value=mock_job)
        mock_redis.llen = MagicMock(return_value=0)

        response = test_client.post(
            "/v1/infer/deterministic_workflow",
            json={
                "parameters": {
                    "forecast_times": ["2024-01-01T00:00:00"],
                    "nsteps": 10,
                    "model_type": "fcn",
                    "data_source": "gfs",
                    "output_format": "zarr",
                    "plot_variable": "t2m",
                }
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"

    def test_invalid_model_type_rejected(self, client):
        """Test that invalid model_type is rejected at submission"""
        test_client, _, _ = client

        response = test_client.post(
            "/v1/infer/deterministic_workflow",
            json={
                "parameters": {
                    "forecast_times": ["2024-01-01T00:00:00"],
                    "nsteps": 10,
                    "model_type": "invalid_model",  # Invalid value
                    "data_source": "gfs",
                }
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert "Invalid parameters" in data["detail"]
        assert "model_type" in data["detail"]

    def test_invalid_data_source_rejected(self, client):
        """Test that invalid data_source is rejected at submission"""
        test_client, _, _ = client

        response = test_client.post(
            "/v1/infer/deterministic_workflow",
            json={
                "parameters": {
                    "forecast_times": ["2024-01-01T00:00:00"],
                    "nsteps": 10,
                    "model_type": "fcn",
                    "data_source": "invalid_source",  # Invalid value
                }
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert "Invalid parameters" in data["detail"]
        assert "data_source" in data["detail"]

    def test_invalid_output_format_rejected(self, client):
        """Test that invalid output_format is rejected at submission"""
        test_client, _, _ = client

        response = test_client.post(
            "/v1/infer/deterministic_workflow",
            json={
                "parameters": {
                    "forecast_times": ["2024-01-01T00:00:00"],
                    "nsteps": 10,
                    "model_type": "fcn",
                    "data_source": "gfs",
                    "output_format": "invalid_format",  # Invalid value
                }
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert "Invalid parameters" in data["detail"]
        assert "output_format" in data["detail"]

    def test_invalid_plot_variable_rejected(self, client):
        """Test that invalid plot_variable is rejected at submission"""
        test_client, _, _ = client

        response = test_client.post(
            "/v1/infer/deterministic_workflow",
            json={
                "parameters": {
                    "forecast_times": ["2024-01-01T00:00:00"],
                    "nsteps": 10,
                    "model_type": "fcn",
                    "data_source": "gfs",
                    "plot_variable": "invalid_variable",  # Invalid value
                }
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert "Invalid parameters" in data["detail"]
        assert "plot_variable" in data["detail"]

    def test_invalid_nsteps_range_rejected(self, client):
        """Test that nsteps outside valid range is rejected"""
        test_client, _, _ = client

        # Test nsteps too low
        response = test_client.post(
            "/v1/infer/deterministic_workflow",
            json={
                "parameters": {
                    "forecast_times": ["2024-01-01T00:00:00"],
                    "nsteps": 0,  # Below minimum of 1
                    "model_type": "fcn",
                }
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert "Invalid parameters" in data["detail"]

        # Test nsteps too high
        response = test_client.post(
            "/v1/infer/deterministic_workflow",
            json={
                "parameters": {
                    "forecast_times": ["2024-01-01T00:00:00"],
                    "nsteps": 101,  # Above maximum of 100
                    "model_type": "fcn",
                }
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert "Invalid parameters" in data["detail"]

    def test_diagnostic_workflow_invalid_model_types_rejected(self, client):
        """Test that invalid model types for diagnostic workflow are rejected"""
        test_client, _, _ = client

        # Test invalid prognostic model
        response = test_client.post(
            "/v1/infer/diagnostic_workflow",
            json={
                "parameters": {
                    "forecast_times": ["2024-01-01T00:00:00"],
                    "nsteps": 10,
                    "prognostic_model_type": "invalid_prognostic",  # Invalid
                    "diagnostic_model_type": "precipitation_afno",
                }
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert "Invalid parameters" in data["detail"]
        assert "prognostic_model_type" in data["detail"]

        # Test invalid diagnostic model
        response = test_client.post(
            "/v1/infer/diagnostic_workflow",
            json={
                "parameters": {
                    "forecast_times": ["2024-01-01T00:00:00"],
                    "nsteps": 10,
                    "prognostic_model_type": "fcn",
                    "diagnostic_model_type": "invalid_diagnostic",  # Invalid
                }
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert "Invalid parameters" in data["detail"]
        assert "diagnostic_model_type" in data["detail"]

    def test_valid_diagnostic_workflow_parameters(self, client):
        """Test that valid parameters for diagnostic workflow are accepted"""
        test_client, mock_queue, mock_redis = client

        # Setup mock job
        mock_job = MagicMock()
        mock_job.id = "test_job"
        mock_queue.enqueue = MagicMock(return_value=mock_job)
        mock_redis.llen = MagicMock(return_value=0)

        response = test_client.post(
            "/v1/infer/diagnostic_workflow",
            json={
                "parameters": {
                    "forecast_times": ["2024-01-01T00:00:00"],
                    "nsteps": 10,
                    "prognostic_model_type": "fcn",
                    "diagnostic_model_type": "precipitation_afno",
                    "data_source": "gfs",
                    "plot_variable": "tp",
                }
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"

    def test_multiple_invalid_parameters_rejected(self, client):
        """Test that multiple invalid parameters are caught"""
        test_client, _, _ = client

        response = test_client.post(
            "/v1/infer/deterministic_workflow",
            json={
                "parameters": {
                    "forecast_times": ["2024-01-01T00:00:00"],
                    "nsteps": 101,  # Invalid: too high
                    "model_type": "invalid_model",  # Invalid value
                    "data_source": "invalid_source",  # Invalid value
                    "output_format": "invalid_format",  # Invalid value
                }
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert "Invalid parameters" in data["detail"]

    def test_valid_forecast_times_iso8601_accepted(self, client):
        """Test that valid ISO 8601 forecast_times are accepted"""
        test_client, mock_queue, mock_redis = client

        # Setup mock job
        mock_job = MagicMock()
        mock_job.id = "test_job"
        mock_queue.enqueue = MagicMock(return_value=mock_job)
        mock_redis.llen = MagicMock(return_value=0)

        # Test various valid ISO 8601 formats
        valid_times = [
            ["2024-01-01T00:00:00"],
            ["2024-01-01T00:00:00Z"],
            ["2024-01-01T00:00:00+00:00"],
            ["2024-01-01T00:00:00.123456"],
            ["2024-01-01 00:00:00"],  # Space separator also accepted
            ["2024-01-01T00:00:00", "2024-01-02T00:00:00", "2024-01-03T00:00:00"],
        ]

        for times in valid_times:
            response = test_client.post(
                "/v1/infer/deterministic_workflow",
                json={
                    "parameters": {
                        "forecast_times": times,
                        "nsteps": 10,
                        "model_type": "fcn",
                    }
                },
            )

            assert response.status_code == 200, f"Failed for times: {times}"
            data = response.json()
            assert data["status"] == "queued"

    def test_invalid_forecast_times_rejected(self, client):
        """Test that invalid forecast_times formats are rejected"""
        test_client, _, _ = client

        # Test various invalid formats
        invalid_times = [
            (["2024-01-01"], "date without time"),
            (["01/01/2024"], "wrong date format"),
            (["not-a-date"], "completely invalid"),
            (["2024-13-01T00:00:00"], "invalid month"),
            (["2024-01-32T00:00:00"], "invalid day"),
            (["2024-01-01T25:00:00"], "invalid hour"),
        ]

        for times, description in invalid_times:
            response = test_client.post(
                "/v1/infer/deterministic_workflow",
                json={
                    "parameters": {
                        "forecast_times": times,
                        "nsteps": 10,
                        "model_type": "fcn",
                    }
                },
            )

            assert (
                response.status_code == 422
            ), f"Expected 422 for {description}, got {response.status_code}"
            data = response.json()
            assert (
                "Invalid parameters" in data["detail"]
            ), f"Expected validation error for {description}"
            assert (
                "forecast_times" in data["detail"]
            ), f"Expected forecast_times in error for {description}"

    def test_forecast_times_mixed_valid_invalid_rejected(self, client):
        """Test that forecast_times with one invalid value fails"""
        test_client, _, _ = client

        response = test_client.post(
            "/v1/infer/deterministic_workflow",
            json={
                "parameters": {
                    "forecast_times": [
                        "2024-01-01T00:00:00",  # Valid
                        "not-a-date",  # Invalid
                        "2024-01-03T00:00:00",  # Valid
                    ],
                    "nsteps": 10,
                    "model_type": "fcn",
                }
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert "Invalid parameters" in data["detail"]
        assert "forecast_times" in data["detail"]

    def test_forecast_times_wrong_type_rejected(self, client):
        """Test that forecast_times with non-string values are rejected"""
        test_client, _, _ = client

        # Test with integer
        response = test_client.post(
            "/v1/infer/deterministic_workflow",
            json={
                "parameters": {
                    "forecast_times": [12345],  # Not a string
                    "nsteps": 10,
                    "model_type": "fcn",
                }
            },
        )

        assert response.status_code == 422
        data = response.json()
        assert "Invalid parameters" in data["detail"]
        assert "forecast_times" in data["detail"]


class TestWorkflowExecutionRequest:
    """Tests for WorkflowExecutionRequest validation"""

    def test_valid_request_with_parameters(self):
        """Test that valid request with parameters field is accepted"""
        from api_server.main import WorkflowExecutionRequest

        # Valid request
        request = WorkflowExecutionRequest(
            parameters={"field1": "value1", "field2": 123}
        )
        assert request.parameters == {"field1": "value1", "field2": 123}

    def test_valid_request_with_empty_parameters(self):
        """Test that request with empty parameters dict is accepted"""
        from api_server.main import WorkflowExecutionRequest

        # Empty parameters is valid
        request = WorkflowExecutionRequest(parameters={})
        assert request.parameters == {}

    def test_valid_request_without_parameters_uses_default(self):
        """Test that request without parameters field uses default empty dict"""
        from api_server.main import WorkflowExecutionRequest

        # No parameters provided - should use default
        request = WorkflowExecutionRequest()
        assert request.parameters == {}

    def test_invalid_request_with_extra_fields(self):
        """Test that request with unknown fields is rejected"""
        from api_server.main import WorkflowExecutionRequest
        from pydantic import ValidationError

        # Request with unknown field should be rejected
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            WorkflowExecutionRequest(
                parameters={"valid": "param"}, unknown_field="should_fail"
            )

    def test_invalid_request_with_multiple_extra_fields(self):
        """Test that request with multiple unknown fields is rejected"""
        from api_server.main import WorkflowExecutionRequest
        from pydantic import ValidationError

        # Multiple unknown fields
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            WorkflowExecutionRequest(
                parameters={"valid": "param"},
                extra1="bad",
                extra2="also_bad",
                extra3="still_bad",
            )

    def test_invalid_request_flat_structure(self):
        """Test that flat request structure (without parameters wrapper) is rejected"""
        from api_server.main import WorkflowExecutionRequest
        from pydantic import ValidationError

        # Flat structure - fields that should be inside 'parameters'
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            WorkflowExecutionRequest(
                forecast_times=["2024-01-01"],
                nsteps=6,
                model_type="fcn",
            )

    def test_invalid_request_mixed_structure(self):
        """Test that mixed structure (parameters + extra fields) is rejected"""
        from api_server.main import WorkflowExecutionRequest
        from pydantic import ValidationError

        # Mixed: valid parameters field + invalid extra fields
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            WorkflowExecutionRequest(
                parameters={"forecast_times": ["2024-01-01"], "nsteps": 6},
                workflow_type="deterministic",  # This should not be at this level
                device="auto",  # This should not be at this level
            )

    def test_parameters_field_type_validation(self):
        """Test that parameters field must be a dict"""
        from api_server.main import WorkflowExecutionRequest
        from pydantic import ValidationError

        # parameters must be a dict, not a string
        with pytest.raises(ValidationError):
            WorkflowExecutionRequest(parameters="not_a_dict")

        # parameters must be a dict, not a list
        with pytest.raises(ValidationError):
            WorkflowExecutionRequest(parameters=["not", "a", "dict"])

        # parameters must be a dict, not None
        with pytest.raises(ValidationError):
            WorkflowExecutionRequest(parameters=None)

    def test_valid_request_serialization(self):
        """Test that valid request can be serialized to dict"""
        from api_server.main import WorkflowExecutionRequest

        request = WorkflowExecutionRequest(
            parameters={"forecast_times": ["2024-01-01"], "nsteps": 10}
        )

        # Serialize to dict
        data = request.model_dump()
        assert isinstance(data, dict)
        assert data == {"parameters": {"forecast_times": ["2024-01-01"], "nsteps": 10}}

    def test_valid_request_with_nested_parameters(self):
        """Test that request with nested parameter structures is accepted"""
        from api_server.main import WorkflowExecutionRequest

        # Complex nested parameters
        request = WorkflowExecutionRequest(
            parameters={
                "forecast_times": ["2024-01-01", "2024-01-02"],
                "nsteps": 10,
                "config": {
                    "model": {"type": "fcn", "version": "1.0"},
                    "data": {"source": "gfs", "cache": True},
                },
                "options": ["opt1", "opt2", "opt3"],
            }
        )

        assert request.parameters["nsteps"] == 10
        assert request.parameters["config"]["model"]["type"] == "fcn"
        assert len(request.parameters["options"]) == 3


class TestQueuePosition:
    """Tests for queue position functionality"""

    @pytest.fixture
    def mock_queue(self):
        """Create a mock RQ queue with job IDs"""
        from unittest.mock import MagicMock

        queue = MagicMock()
        queue.job_ids = [
            "workflow1_exec_001",
            "workflow2_exec_002",
            "workflow1_exec_003",
            "workflow3_exec_004",
        ]
        return queue

    def test_get_queue_position_first_in_queue(self, mock_queue):
        """Test getting position for first job in queue"""
        from unittest.mock import patch

        from api_server.main import get_queue_position

        with patch("api_server.main.inference_queue", mock_queue):
            position = get_queue_position("workflow1_exec_001")
            assert position == 0  # 0-indexed: first job is at position 0

    def test_get_queue_position_middle_of_queue(self, mock_queue):
        """Test getting position for job in middle of queue"""
        from unittest.mock import patch

        from api_server.main import get_queue_position

        with patch("api_server.main.inference_queue", mock_queue):
            position = get_queue_position("workflow1_exec_003")
            assert position == 2  # 0-indexed: third job is at position 2

    def test_get_queue_position_last_in_queue(self, mock_queue):
        """Test getting position for last job in queue"""
        from unittest.mock import patch

        from api_server.main import get_queue_position

        with patch("api_server.main.inference_queue", mock_queue):
            position = get_queue_position("workflow3_exec_004")
            assert position == 3  # 0-indexed: fourth job is at position 3

    def test_get_queue_position_not_in_queue(self, mock_queue):
        """Test getting position for job not in queue returns None"""
        from unittest.mock import patch

        from api_server.main import get_queue_position

        with patch("api_server.main.inference_queue", mock_queue):
            position = get_queue_position("nonexistent_job")
            assert position is None

    def test_get_queue_position_no_queue(self):
        """Test getting position when queue is not initialized returns None"""
        from unittest.mock import patch

        from api_server.main import get_queue_position

        with patch("api_server.main.inference_queue", None):
            position = get_queue_position("any_job")
            assert position is None

    def test_get_queue_position_empty_queue(self):
        """Test getting position when queue is empty"""
        from unittest.mock import MagicMock, patch

        from api_server.main import get_queue_position

        empty_queue = MagicMock()
        empty_queue.job_ids = []

        with patch("api_server.main.inference_queue", empty_queue):
            position = get_queue_position("any_job")
            assert position is None

    def test_get_queue_position_exception_handling(self):
        """Test that exceptions in get_queue_position are handled gracefully"""
        from unittest.mock import MagicMock, PropertyMock, patch

        from api_server.main import get_queue_position

        queue_with_error = MagicMock()
        type(queue_with_error).job_ids = PropertyMock(
            side_effect=Exception("Queue connection error")
        )

        with patch("api_server.main.inference_queue", queue_with_error):
            position = get_queue_position("any_job")
            assert position is None


class TestWorkflowExecutionWithQueuePosition:
    """Tests for workflow execution endpoint with queue position"""

    @pytest.fixture
    def mock_workflow_class(self):
        """Create a mock workflow class"""
        from api_server.workflow import Workflow, WorkflowParameters

        class TestWorkflowParameters(WorkflowParameters):
            """Test parameters"""

            test_param: str = "default"

        class TestWorkflow(Workflow):
            name = "test_workflow"
            description = "Test workflow"
            Parameters = TestWorkflowParameters

            @classmethod
            def validate_parameters(cls, parameters):
                return TestWorkflowParameters.validate(parameters)

            def run(self, parameters, execution_id):
                return {"status": "success"}

        return TestWorkflow

    @pytest.fixture
    def client(self, mock_workflow_class):
        """Create test client with mocked dependencies"""
        with (
            patch("redis.asyncio.Redis") as mock_async_redis,
            patch("redis.Redis") as mock_sync_redis,
            patch("rq.Queue") as mock_queue_class,
            patch("api_server.workflow.register_all_workflows"),
        ):
            # Setup mock async Redis
            mock_async_instance = MagicMock()
            mock_async_instance.ping = AsyncMock(return_value=True)
            mock_async_instance.close = AsyncMock()
            mock_async_instance.get = AsyncMock(return_value=None)
            mock_async_redis.return_value = mock_async_instance

            # Setup mock sync Redis
            mock_sync_instance = MagicMock()
            mock_sync_instance.ping = MagicMock(return_value=True)
            mock_sync_instance.close = MagicMock()
            mock_sync_instance.setex = MagicMock()
            mock_sync_redis.return_value = mock_sync_instance

            # Setup mock queue
            mock_queue = MagicMock()
            mock_queue.__len__ = MagicMock(return_value=0)
            mock_queue.job_ids = []
            mock_queue_class.return_value = mock_queue

            from api_server.main import app
            from api_server.workflow import workflow_registry

            # Register the test workflow
            workflow_registry._workflows["test_workflow"] = mock_workflow_class

            with TestClient(app, raise_server_exceptions=False) as client:
                yield client, mock_queue, mock_sync_instance

            # Cleanup
            if "test_workflow" in workflow_registry._workflows:
                del workflow_registry._workflows["test_workflow"]

    def test_execute_workflow_includes_queue_position(self, client):
        """Test that POST /v1/infer/{workflow_name} includes queue position"""
        from unittest.mock import patch

        test_client, mock_queue, mock_redis = client

        # Setup mock job
        mock_job = MagicMock()
        mock_job.id = "test_workflow_exec_123"

        # Before enqueue: queue size is 0
        mock_queue.__len__ = MagicMock(return_value=0)
        mock_queue.enqueue = MagicMock(return_value=mock_job)

        # Redis llen returns 0 (the queue size before the job is fully committed)
        mock_redis.llen = MagicMock(return_value=0)

        # Patch the inference_queue at module level
        with patch("api_server.main.inference_queue", mock_queue):
            response = test_client.post(
                "/v1/infer/test_workflow", json={"parameters": {"test_param": "value"}}
            )

        assert response.status_code == 200
        data = response.json()

        # Check that position is included in response
        assert "position" in data
        assert data["position"] == 0  # Redis llen returns 0 (first job, position 0)
        assert data["status"] == "queued"

    def test_execute_workflow_position_with_existing_jobs(self, client):
        """Test that position is correct when there are existing jobs in queue"""
        from unittest.mock import patch

        test_client, mock_queue, mock_redis = client

        # Setup mock job
        mock_job = MagicMock()
        mock_job.id = "test_workflow_exec_456"

        # Before enqueue: queue has 2 existing jobs
        mock_queue.__len__ = MagicMock(return_value=2)
        mock_queue.enqueue = MagicMock(return_value=mock_job)

        # Redis llen returns 2 (the queue size before the job is fully committed)
        mock_redis.llen = MagicMock(return_value=2)

        # Patch the inference_queue at module level
        with patch("api_server.main.inference_queue", mock_queue):
            response = test_client.post(
                "/v1/infer/test_workflow", json={"parameters": {"test_param": "value"}}
            )

        assert response.status_code == 200
        data = response.json()

        # Check that position is correct
        assert "position" in data
        assert data["position"] == 2  # Redis llen returns 2 (third job, position 2)
        assert data["status"] == "queued"

    def test_execute_workflow_position_with_five_existing_jobs(self, client):
        """Test that position is correct with multiple existing jobs in queue"""
        from unittest.mock import patch

        test_client, mock_queue, mock_redis = client

        # Setup mock job
        mock_job = MagicMock()
        mock_job.id = "test_workflow_exec_789"
        mock_queue.enqueue = MagicMock(return_value=mock_job)

        # Before enqueue: queue has 5 existing jobs
        mock_queue.__len__ = MagicMock(return_value=5)

        # Redis llen returns 5 (the queue size before the job is fully committed)
        mock_redis.llen = MagicMock(return_value=5)

        # Patch the inference_queue at module level
        with patch("api_server.main.inference_queue", mock_queue):
            response = test_client.post(
                "/v1/infer/test_workflow", json={"parameters": {"test_param": "value"}}
            )

        assert response.status_code == 200
        data = response.json()

        # Check that position is correct
        assert "position" in data
        assert data["position"] == 5  # Redis llen returns 5 (sixth job, position 5)
        assert data["status"] == "queued"

    def test_get_workflow_status_includes_position_when_queued(self, client):
        """Test that GET /v1/infer/{workflow_name}/{execution_id}/status includes position when status is queued"""
        test_client, mock_queue, mock_redis = client

        # Setup queue with multiple jobs
        execution_id = "exec_456"
        job_id = f"test_workflow_{execution_id}"

        # Set up mock queue job_ids to include our job at position 1
        mock_queue.job_ids = ["other_job_1", job_id, "other_job_2"]

        # Mock Redis response for queued execution
        import json
        from datetime import datetime, timezone
        from unittest.mock import patch

        execution_data = {
            "workflow_name": "test_workflow",
            "execution_id": execution_id,
            "status": "queued",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
        }
        mock_redis.get = MagicMock(return_value=json.dumps(execution_data).encode())

        # Patch the inference_queue for get_queue_position to use
        with patch("api_server.main.inference_queue", mock_queue):
            response = test_client.get(f"/v1/infer/test_workflow/{execution_id}/status")

        assert response.status_code == 200
        data = response.json()

        # Check that position is included and correct
        assert "position" in data
        assert data["position"] == 1  # Second in queue (0-indexed, so position 1)
        assert data["status"] == "queued"

    def test_get_workflow_status_queued_but_not_in_queue(self, client):
        """Test race condition: status is QUEUED but job not in queue (worker picked it up)"""
        test_client, mock_queue, mock_redis = client

        execution_id = "exec_race"

        # Set up mock queue with empty job_ids (job already picked up by worker)
        mock_queue.job_ids = []

        # Mock Redis response with QUEUED status (worker hasn't updated it yet)
        import json
        from datetime import datetime, timezone
        from unittest.mock import patch

        execution_data = {
            "workflow_name": "test_workflow",
            "execution_id": execution_id,
            "status": "queued",  # Still queued in Redis
            "start_time": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
        }
        mock_redis.get = MagicMock(return_value=json.dumps(execution_data).encode())

        # Patch the inference_queue for get_queue_position to use
        with patch("api_server.main.inference_queue", mock_queue):
            response = test_client.get(f"/v1/infer/test_workflow/{execution_id}/status")

        assert response.status_code == 200
        data = response.json()

        # Should be treated as RUNNING since job is not in queue
        assert data["status"] == "running"
        # Position should be None for running status
        if "position" in data:
            assert data["position"] is None

    def test_get_workflow_status_no_position_when_running(self, client):
        """Test that position is None when workflow is running (not queued)"""
        test_client, mock_queue, mock_redis = client

        execution_id = "exec_789"

        # Mock Redis response for running execution
        import json
        from datetime import datetime, timezone

        execution_data = {
            "workflow_name": "test_workflow",
            "execution_id": execution_id,
            "status": "running",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
        }
        mock_redis.get = MagicMock(return_value=json.dumps(execution_data).encode())

        response = test_client.get(f"/v1/infer/test_workflow/{execution_id}/status")

        assert response.status_code == 200
        data = response.json()

        # Position should be None or not included for non-queued status
        # (The model allows None, so it could be None or absent)
        if "position" in data:
            assert data["position"] is None
        assert data["status"] == "running"

    def test_get_workflow_status_no_position_when_completed(self, client):
        """Test that position is None when workflow is completed"""
        test_client, mock_queue, mock_redis = client

        execution_id = "exec_999"

        # Mock Redis response for completed execution
        import json
        from datetime import datetime, timezone

        execution_data = {
            "workflow_name": "test_workflow",
            "execution_id": execution_id,
            "status": "completed",
            "start_time": datetime.now(timezone.utc).isoformat(),
            "end_time": datetime.now(timezone.utc).isoformat(),
            "metadata": {},
        }
        mock_redis.get = MagicMock(return_value=json.dumps(execution_data).encode())

        response = test_client.get(f"/v1/infer/test_workflow/{execution_id}/status")

        assert response.status_code == 200
        data = response.json()

        # Position should be None for completed workflows
        if "position" in data:
            assert data["position"] is None
        assert data["status"] == "completed"
