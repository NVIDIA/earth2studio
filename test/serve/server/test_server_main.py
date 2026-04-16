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

import json
import os
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from pydantic import Field

# Set API environment variable before importing main (DANGER!!! REMOVE THIS)
os.environ["EARTH2STUDIO_API_ACTIVE"] = "1"
# test/serve/server/conftest.py clears EXPOSED_WORKFLOWS; keep parity if this file is
# imported without that conftest (e.g. dynamic imports).
os.environ.pop("EXPOSED_WORKFLOWS", None)

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


# Apply the patch before any imports of earth2studio.serve.server.main
fastapi.routing.APIRoute.__init__ = _patched_route_init


class TestWorkflowSchemaEndpoint:
    """Tests for GET /v1/workflows/{workflow_name}/schema endpoint"""

    @pytest.fixture
    def mock_workflow_class(self):
        """Create a mock workflow class with Parameters"""
        from earth2studio.serve.server.workflow import Workflow, WorkflowParameters

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
            patch("earth2studio.serve.server.workflow.register_all_workflows"),
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

            from earth2studio.serve.server.main import app
            from earth2studio.serve.server.workflow import workflow_registry

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

        from pydantic import Field

        from earth2studio.serve.server.main import app
        from earth2studio.serve.server.workflow import (
            Workflow,
            WorkflowParameters,
            workflow_registry,
        )

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

        with patch("earth2studio.serve.server.main.redis_sync_client", mock_redis):
            with patch("earth2studio.serve.server.main.inference_queue", mock_queue):
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
        from earth2studio.serve.server.main import WorkflowExecutionRequest

        # Valid request
        request = WorkflowExecutionRequest(
            parameters={"field1": "value1", "field2": 123}
        )
        assert request.parameters == {"field1": "value1", "field2": 123}

    def test_valid_request_with_empty_parameters(self):
        """Test that request with empty parameters dict is accepted"""
        from earth2studio.serve.server.main import WorkflowExecutionRequest

        # Empty parameters is valid
        request = WorkflowExecutionRequest(parameters={})
        assert request.parameters == {}

    def test_valid_request_without_parameters_uses_default(self):
        """Test that request without parameters field uses default empty dict"""
        from earth2studio.serve.server.main import WorkflowExecutionRequest

        # No parameters provided - should use default
        request = WorkflowExecutionRequest()
        assert request.parameters == {}

    def test_invalid_request_with_extra_fields(self):
        """Test that request with unknown fields is rejected"""
        from pydantic import ValidationError

        from earth2studio.serve.server.main import WorkflowExecutionRequest

        # Request with unknown field should be rejected
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            WorkflowExecutionRequest(
                parameters={"valid": "param"}, unknown_field="should_fail"
            )

    def test_invalid_request_with_multiple_extra_fields(self):
        """Test that request with multiple unknown fields is rejected"""
        from pydantic import ValidationError

        from earth2studio.serve.server.main import WorkflowExecutionRequest

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
        from pydantic import ValidationError

        from earth2studio.serve.server.main import WorkflowExecutionRequest

        # Flat structure - fields that should be inside 'parameters'
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            WorkflowExecutionRequest(
                forecast_times=["2024-01-01"],
                nsteps=6,
                model_type="fcn",
            )

    def test_invalid_request_mixed_structure(self):
        """Test that mixed structure (parameters + extra fields) is rejected"""
        from pydantic import ValidationError

        from earth2studio.serve.server.main import WorkflowExecutionRequest

        # Mixed: valid parameters field + invalid extra fields
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            WorkflowExecutionRequest(
                parameters={"forecast_times": ["2024-01-01"], "nsteps": 6},
                workflow_type="deterministic",  # This should not be at this level
                device="auto",  # This should not be at this level
            )

    def test_parameters_field_type_validation(self):
        """Test that parameters field must be a dict"""
        from pydantic import ValidationError

        from earth2studio.serve.server.main import WorkflowExecutionRequest

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
        from earth2studio.serve.server.main import WorkflowExecutionRequest

        request = WorkflowExecutionRequest(
            parameters={"forecast_times": ["2024-01-01"], "nsteps": 10}
        )

        # Serialize to dict
        data = request.model_dump()
        assert isinstance(data, dict)
        assert data == {"parameters": {"forecast_times": ["2024-01-01"], "nsteps": 10}}

    def test_valid_request_with_nested_parameters(self):
        """Test that request with nested parameter structures is accepted"""
        from earth2studio.serve.server.main import WorkflowExecutionRequest

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

        from earth2studio.serve.server.main import get_queue_position

        with patch("earth2studio.serve.server.main.inference_queue", mock_queue):
            position = get_queue_position("workflow1_exec_001")
            assert position == 0  # 0-indexed: first job is at position 0

    def test_get_queue_position_middle_of_queue(self, mock_queue):
        """Test getting position for job in middle of queue"""
        from unittest.mock import patch

        from earth2studio.serve.server.main import get_queue_position

        with patch("earth2studio.serve.server.main.inference_queue", mock_queue):
            position = get_queue_position("workflow1_exec_003")
            assert position == 2  # 0-indexed: third job is at position 2

    def test_get_queue_position_last_in_queue(self, mock_queue):
        """Test getting position for last job in queue"""
        from unittest.mock import patch

        from earth2studio.serve.server.main import get_queue_position

        with patch("earth2studio.serve.server.main.inference_queue", mock_queue):
            position = get_queue_position("workflow3_exec_004")
            assert position == 3  # 0-indexed: fourth job is at position 3

    def test_get_queue_position_not_in_queue(self, mock_queue):
        """Test getting position for job not in queue returns None"""
        from unittest.mock import patch

        from earth2studio.serve.server.main import get_queue_position

        with patch("earth2studio.serve.server.main.inference_queue", mock_queue):
            position = get_queue_position("nonexistent_job")
            assert position is None

    def test_get_queue_position_no_queue(self):
        """Test getting position when queue is not initialized returns None"""
        from unittest.mock import patch

        from earth2studio.serve.server.main import get_queue_position

        with patch("earth2studio.serve.server.main.inference_queue", None):
            position = get_queue_position("any_job")
            assert position is None

    def test_get_queue_position_empty_queue(self):
        """Test getting position when queue is empty"""
        from unittest.mock import MagicMock, patch

        from earth2studio.serve.server.main import get_queue_position

        empty_queue = MagicMock()
        empty_queue.job_ids = []

        with patch("earth2studio.serve.server.main.inference_queue", empty_queue):
            position = get_queue_position("any_job")
            assert position is None

    def test_get_queue_position_exception_handling(self):
        """Test that exceptions in get_queue_position are handled gracefully"""
        from unittest.mock import MagicMock, PropertyMock, patch

        from earth2studio.serve.server.main import get_queue_position

        queue_with_error = MagicMock()
        type(queue_with_error).job_ids = PropertyMock(
            side_effect=Exception("Queue connection error")
        )

        with patch("earth2studio.serve.server.main.inference_queue", queue_with_error):
            position = get_queue_position("any_job")
            assert position is None


class TestWorkflowExecutionWithQueuePosition:
    """Tests for workflow execution endpoint with queue position"""

    @pytest.fixture
    def mock_workflow_class(self):
        """Create a mock workflow class"""
        from earth2studio.serve.server.workflow import Workflow, WorkflowParameters

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
            patch("earth2studio.serve.server.workflow.register_all_workflows"),
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

            from earth2studio.serve.server.main import app
            from earth2studio.serve.server.workflow import workflow_registry

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
        with patch("earth2studio.serve.server.main.inference_queue", mock_queue):
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
        with patch("earth2studio.serve.server.main.inference_queue", mock_queue):
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
        with patch("earth2studio.serve.server.main.inference_queue", mock_queue):
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
        with patch("earth2studio.serve.server.main.inference_queue", mock_queue):
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
        with patch("earth2studio.serve.server.main.inference_queue", mock_queue):
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


class TestHealthAndProbes:
    """Tests for /health, /readiness, /liveness, and /metrics endpoints."""

    @pytest.fixture
    def client_with_mocks(self):
        """Client with mocked Redis and RQ for probe endpoints."""
        with (
            patch("redis.asyncio.Redis") as mock_async_redis,
            patch("redis.Redis") as mock_sync_redis,
            patch("rq.Queue"),
            patch("earth2studio.serve.server.workflow.register_all_workflows"),
        ):
            mock_async_instance = MagicMock()
            mock_async_instance.ping = AsyncMock(return_value=True)
            mock_async_instance.close = AsyncMock()
            mock_async_redis.return_value = mock_async_instance

            mock_sync_instance = MagicMock()
            mock_sync_instance.ping = MagicMock(return_value=True)
            mock_sync_instance.close = MagicMock()
            mock_sync_redis.return_value = mock_sync_instance

            from earth2studio.serve.server.main import app

            with TestClient(app, raise_server_exceptions=False) as c:
                yield c

    def test_liveness_returns_alive(self, client_with_mocks):
        """GET /liveness returns status alive."""
        response = client_with_mocks.get("/liveness")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}

    def test_health_healthy_when_script_returns_zero(self, client_with_mocks):
        """GET /health and /readiness return healthy when status script exits 0."""
        with patch(
            "earth2studio.serve.server.main.asyncio.create_subprocess_exec"
        ) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = 0
            mock_proc.communicate = AsyncMock(return_value=(b"", b""))
            mock_exec.return_value = mock_proc

            for path in ("/health", "/readiness"):
                response = client_with_mocks.get(path)
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert "timestamp" in data

    def test_health_unhealthy_when_script_returns_nonzero(self, client_with_mocks):
        """GET /health returns 503 when status script exits non-zero."""
        with patch(
            "earth2studio.serve.server.main.asyncio.create_subprocess_exec"
        ) as mock_exec:
            mock_proc = MagicMock()
            mock_proc.returncode = 1
            mock_proc.communicate = AsyncMock(return_value=(b"", b"stderr"))
            mock_exec.return_value = mock_proc

            response = client_with_mocks.get("/health")
            assert response.status_code == 503
            data = response.json()
            assert data["detail"]["status"] == "unhealthy"

    def test_metrics_returns_prometheus_format(self, client_with_mocks):
        """GET /metrics returns Prometheus text format."""
        with patch("earth2studio.serve.server.main.generate_latest") as mock_gen:
            mock_gen.return_value = (
                b"# HELP mock_metric Mock\n# TYPE mock_metric gauge\nmock_metric 1.0\n"
            )

            response = client_with_mocks.get("/metrics")
            assert response.status_code == 200
            assert "mock_metric" in response.text
            assert "text/plain" in response.headers.get("content-type", "")


class TestListWorkflows:
    """Tests for GET /v1/workflows and /v1/infer/workflows."""

    @pytest.fixture
    def client_with_workflows(self):
        """Client with a registered workflow for list_workflows."""
        with (
            patch("redis.asyncio.Redis") as mock_async_redis,
            patch("redis.Redis") as mock_sync_redis,
            patch("rq.Queue"),
            patch("earth2studio.serve.server.workflow.register_all_workflows"),
        ):
            mock_async_instance = MagicMock()
            mock_async_instance.ping = AsyncMock(return_value=True)
            mock_async_instance.close = AsyncMock()
            mock_async_redis.return_value = mock_async_instance
            mock_sync_instance = MagicMock()
            mock_sync_instance.ping = MagicMock(return_value=True)
            mock_sync_instance.close = MagicMock()
            mock_sync_redis.return_value = mock_sync_instance

            from earth2studio.serve.server.main import app
            from earth2studio.serve.server.workflow import workflow_registry

            class FakeWorkflow:
                name = "fake_wf"
                description = "Fake workflow for list test"

            workflow_registry._workflows["fake_wf"] = FakeWorkflow

            with TestClient(app, raise_server_exceptions=False) as c:
                yield c

            if "fake_wf" in workflow_registry._workflows:
                del workflow_registry._workflows["fake_wf"]

    def test_list_workflows_returns_registry(self, client_with_workflows):
        """GET /v1/workflows returns workflows dict."""
        response = client_with_workflows.get("/v1/workflows")
        assert response.status_code == 200
        data = response.json()
        assert "workflows" in data
        assert "fake_wf" in data["workflows"]
        assert data["workflows"]["fake_wf"] == "Fake workflow for list test"

    def test_list_workflows_infer_path(self, client_with_workflows):
        """GET /v1/infer/workflows returns same structure."""
        response = client_with_workflows.get("/v1/infer/workflows")
        assert response.status_code == 200
        data = response.json()
        assert "workflows" in data


class TestAdmissionControl:
    """Tests for check_admission_control (queue full, Redis None)."""

    @pytest.fixture
    def client_for_admission(self):
        """Client with mocked queue and redis for admission tests."""
        from earth2studio.serve.server.workflow import Workflow, WorkflowParameters

        with (
            patch("redis.asyncio.Redis") as mock_async_redis,
            patch("redis.Redis") as mock_sync_redis,
            patch("rq.Queue") as mock_queue_class,
            patch("earth2studio.serve.server.workflow.register_all_workflows"),
        ):
            mock_async_instance = MagicMock()
            mock_async_instance.ping = AsyncMock(return_value=True)
            mock_async_instance.close = AsyncMock()
            mock_async_redis.return_value = mock_async_instance

            mock_sync_instance = MagicMock()
            mock_sync_instance.ping = MagicMock(return_value=True)
            mock_sync_instance.llen = MagicMock(return_value=0)
            mock_sync_redis.return_value = mock_sync_instance

            mock_queue = MagicMock()
            mock_queue.enqueue = MagicMock(return_value=MagicMock(id="job_1"))
            mock_queue_class.return_value = mock_queue

            class AdmitParams(WorkflowParameters):
                x: int = Field(default=1, ge=0)

            class AdmitWorkflow(Workflow):
                name = "admit_wf"
                description = "For admission test"
                Parameters = AdmitParams

                @classmethod
                def validate_parameters(
                    cls, parameters: dict[str, Any] | AdmitParams
                ) -> AdmitParams:
                    return AdmitParams.validate(parameters)

                def run(self, parameters, execution_id):
                    return {"status": "ok"}

            from earth2studio.serve.server.main import app
            from earth2studio.serve.server.workflow import workflow_registry

            workflow_registry._workflows["admit_wf"] = AdmitWorkflow

            with TestClient(app, raise_server_exceptions=False) as c:
                yield c, mock_sync_instance, mock_queue

            if "admit_wf" in workflow_registry._workflows:
                del workflow_registry._workflows["admit_wf"]

    def test_admission_control_redis_none_returns_503(self, client_for_admission):
        """When redis_sync_client is None, execute returns 503."""
        client, mock_sync_redis, mock_queue = client_for_admission
        with patch("earth2studio.serve.server.main.redis_sync_client", None):
            response = client.post(
                "/v1/infer/admit_wf",
                json={"parameters": {}},
            )
        assert response.status_code == 503
        assert "Redis" in response.json().get("detail", "")

    def test_admission_control_queue_full_returns_429(self, client_for_admission):
        """When queue is full, execute returns 429."""
        client, mock_sync_redis, mock_queue = client_for_admission
        mock_sync_redis.llen.return_value = 100
        with patch("earth2studio.serve.server.main.config") as mock_config:
            mock_config.queue.max_size = 10
            mock_config.queue.name = "inference"
            mock_config.queue.result_zip_queue_name = "result_zip"
            mock_config.queue.object_storage_queue_name = "object_storage"
            mock_config.queue.geocatalog_ingestion_queue_name = "geocatalog_ingestion"
            mock_config.queue.finalize_metadata_queue_name = "finalize_metadata"
            response = client.post(
                "/v1/infer/admit_wf",
                json={"parameters": {}},
            )
        assert response.status_code == 429
        assert "full" in response.json().get("detail", "").lower()

    def test_admission_control_llen_raises_returns_500(self, client_for_admission):
        """When redis_sync_client.llen raises, execute returns 500."""
        client, mock_sync_redis, mock_queue = client_for_admission
        mock_sync_redis.llen.side_effect = RuntimeError("Redis connection lost")
        response = client.post(
            "/v1/infer/admit_wf",
            json={"parameters": {}},
        )
        assert response.status_code == 500
        assert "queue" in response.json().get("detail", "").lower()


class TestExecuteWorkflowBranches:
    """Tests for execute_workflow branches (404, 400, 503, 500)."""

    @pytest.fixture
    def client_exec(self):
        """Client with one workflow for execute tests."""
        from earth2studio.serve.server.workflow import Workflow, WorkflowParameters

        with (
            patch("redis.asyncio.Redis") as mock_async_redis,
            patch("redis.Redis") as mock_sync_redis,
            patch("rq.Queue") as mock_queue_class,
            patch("earth2studio.serve.server.workflow.register_all_workflows"),
        ):
            mock_async_instance = MagicMock()
            mock_async_instance.ping = AsyncMock(return_value=True)
            mock_async_instance.close = AsyncMock()
            mock_async_instance.get = AsyncMock(return_value=None)
            mock_async_redis.return_value = mock_async_instance

            mock_sync_instance = MagicMock()
            mock_sync_instance.ping = MagicMock(return_value=True)
            mock_sync_instance.close = MagicMock()
            mock_sync_instance.setex = MagicMock()
            mock_sync_instance.llen = MagicMock(return_value=0)
            mock_sync_redis.return_value = mock_sync_instance

            mock_queue = MagicMock()
            mock_job = MagicMock()
            mock_job.id = "exec_wf_exec_123"
            mock_queue.enqueue = MagicMock(return_value=mock_job)
            mock_queue_class.return_value = mock_queue

            class ExecParams(WorkflowParameters):
                x: int = Field(default=1, ge=0)

            class ExecWorkflow(Workflow):
                name = "exec_wf"
                description = "For execute tests"
                Parameters = ExecParams

                @classmethod
                def validate_parameters(
                    cls, parameters: dict[str, Any] | ExecParams
                ) -> ExecParams:
                    return ExecParams.validate(parameters)

                def run(self, parameters, execution_id):
                    return {"status": "ok"}

            from earth2studio.serve.server.main import app
            from earth2studio.serve.server.workflow import workflow_registry

            workflow_registry._workflows["exec_wf"] = ExecWorkflow
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c, mock_sync_instance, mock_queue
            if "exec_wf" in workflow_registry._workflows:
                del workflow_registry._workflows["exec_wf"]

    def test_execute_workflow_not_found_404(self, client_exec):
        """POST /v1/infer/{workflow_name} with unknown workflow returns 404."""
        client, _, _ = client_exec
        response = client.post(
            "/v1/infer/nonexistent_workflow",
            json={"parameters": {}},
        )
        assert response.status_code == 404
        assert "not found" in response.json().get("detail", "").lower()

    def test_execute_workflow_validate_parameters_unexpected_error_400(
        self, client_exec
    ):
        """When validate_parameters raises non-ValueError, returns 400."""
        client, _, _ = client_exec
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_registry:
            mock_wf = MagicMock()
            mock_wf.validate_parameters.side_effect = AttributeError("no model_dump")
            mock_registry.get_workflow_class.return_value = mock_wf
            response = client.post(
                "/v1/infer/exec_wf",
                json={"parameters": {"x": 1}},
            )
        assert response.status_code == 400
        assert "validating" in response.json().get("detail", "").lower()

    def test_execute_workflow_inference_queue_none_503(self, client_exec):
        """When inference_queue is None, returns 503."""
        client, mock_redis, mock_queue = client_exec
        with patch("earth2studio.serve.server.main.inference_queue", None):
            response = client.post(
                "/v1/infer/exec_wf",
                json={"parameters": {}},
            )
        assert response.status_code == 503
        assert "queue" in response.json().get("detail", "").lower()

    def test_execute_workflow_llen_raises_after_enqueue_500(self, client_exec):
        """When llen raises after enqueue (queue position lookup), returns 500."""
        client, mock_redis, mock_queue = client_exec
        # 5 queue checks in admission control + 1 for queue position lookup
        mock_redis.llen.side_effect = [0, 0, 0, 0, 0, RuntimeError("redis error")]
        with patch("earth2studio.serve.server.main.inference_queue", mock_queue):
            response = client.post(
                "/v1/infer/exec_wf",
                json={"parameters": {}},
            )
        assert response.status_code == 500
        assert "queue position" in response.json().get(
            "detail", ""
        ).lower() or "Internal" in response.json().get("detail", "")

    def test_execute_workflow_enqueue_failure_500(self, client_exec):
        """When enqueue raises, returns 500."""
        client, mock_redis, mock_queue = client_exec
        mock_queue.enqueue.side_effect = RuntimeError("RQ enqueue failed")
        with patch("earth2studio.serve.server.main.inference_queue", mock_queue):
            response = client.post(
                "/v1/infer/exec_wf",
                json={"parameters": {}},
            )
        assert response.status_code == 500
        assert "enqueue" in response.json().get("detail", "").lower()

    def test_default_infer_no_exposed_503(self, client_exec):
        """POST /v1/infer returns 503 when no workflows are exposed."""
        client, _, _ = client_exec
        from earth2studio.serve.server.workflow import workflow_registry

        with patch.object(workflow_registry, "list_workflows", return_value={}):
            r = client.post("/v1/infer", json={"parameters": {}})
        assert r.status_code == 503
        assert "No exposed workflows" in r.json()["detail"]

    def test_default_infer_multiple_exposed_409(self, client_exec):
        """POST /v1/infer returns 409 when more than one workflow is exposed."""
        client, _, _ = client_exec
        from earth2studio.serve.server.workflow import workflow_registry

        with patch.object(
            workflow_registry,
            "list_workflows",
            return_value={"wf_a": "A", "wf_b": "B"},
        ):
            r = client.post("/v1/infer", json={"parameters": {}})
        assert r.status_code == 409
        d = r.json()["detail"]
        assert "exactly one" in d
        assert "wf_a" in d and "wf_b" in d

    def test_default_infer_delegates_to_named_route(self, client_exec):
        """POST /v1/infer matches POST /v1/infer/exec_wf when exactly one workflow is exposed."""
        client, mock_redis, mock_queue = client_exec
        from earth2studio.serve.server.workflow import workflow_registry

        mock_job = MagicMock()
        mock_job.id = "solo_job"
        mock_queue.enqueue = MagicMock(return_value=mock_job)
        mock_redis.llen = MagicMock(return_value=0)
        # Registry may contain other workflows from app init; isolate to one exposed name.
        with patch("earth2studio.serve.server.main.inference_queue", mock_queue):
            with patch.object(
                workflow_registry,
                "list_workflows",
                return_value={"exec_wf": "For execute tests"},
            ):
                r_default = client.post("/v1/infer", json={"parameters": {}})
            r_named = client.post("/v1/infer/exec_wf", json={"parameters": {}})
        assert r_default.status_code == 200
        assert r_named.status_code == 200
        assert r_default.json()["status"] == r_named.json()["status"] == "queued"


class TestHealthMetricsSchemaExceptions:
    """Tests for health, metrics, and schema exception paths."""

    @pytest.fixture
    def client_probes(self):
        """Client for probe endpoints."""
        with (
            patch("redis.asyncio.Redis") as mock_async_redis,
            patch("redis.Redis") as mock_sync_redis,
            patch("rq.Queue"),
            patch("earth2studio.serve.server.workflow.register_all_workflows"),
        ):
            mock_async_instance = MagicMock()
            mock_async_instance.ping = AsyncMock(return_value=True)
            mock_async_instance.close = AsyncMock()
            mock_async_redis.return_value = mock_async_instance
            mock_sync_instance = MagicMock()
            mock_sync_instance.ping = MagicMock(return_value=True)
            mock_sync_instance.close = MagicMock()
            mock_sync_redis.return_value = mock_sync_instance
            from earth2studio.serve.server.main import app

            with TestClient(app, raise_server_exceptions=False) as c:
                yield c

    def test_health_check_subprocess_exception_500(self, client_probes):
        """When create_subprocess_exec raises, health returns 500."""
        with patch(
            "earth2studio.serve.server.main.asyncio.create_subprocess_exec"
        ) as mock_exec:
            mock_exec.side_effect = FileNotFoundError("status.sh not found")
            response = client_probes.get("/health")
        assert response.status_code == 500
        assert "Health check failed" in response.json().get("detail", "")

    def test_metrics_generate_latest_raises_500(self, client_probes):
        """When generate_latest raises, metrics returns 500."""
        with patch("earth2studio.serve.server.main.generate_latest") as mock_gen:
            mock_gen.side_effect = RuntimeError("metrics error")
            response = client_probes.get("/metrics")
        assert response.status_code == 500
        detail = response.json().get("detail", {})
        if isinstance(detail, dict):
            assert "metrics" in str(detail).lower()
        else:
            assert "metrics" in detail.lower()

    def test_schema_generation_exception_500(self, client_probes):
        """When model_json_schema raises, schema endpoint returns 500."""
        from earth2studio.serve.server.workflow import Workflow, WorkflowParameters

        class BadSchemaParams(WorkflowParameters):
            x: int = Field(default=1)

        class BadSchemaWorkflow(Workflow):
            name = "bad_schema_wf"
            description = "Bad schema"
            Parameters = BadSchemaParams

            @classmethod
            def validate_parameters(cls, parameters):
                return BadSchemaParams.validate(parameters)

            def run(self, parameters, execution_id):
                return {"status": "ok"}

        from earth2studio.serve.server.workflow import workflow_registry

        workflow_registry._workflows["bad_schema_wf"] = BadSchemaWorkflow
        try:
            with patch.object(
                BadSchemaParams,
                "model_json_schema",
                side_effect=RuntimeError("schema error"),
            ):
                response = client_probes.get("/v1/workflows/bad_schema_wf/schema")
            assert response.status_code == 500
            detail = response.json().get("detail", {})
            assert "schema" in str(detail).lower()
        finally:
            if "bad_schema_wf" in workflow_registry._workflows:
                del workflow_registry._workflows["bad_schema_wf"]


class TestGetWorkflowStatusBranches:
    """Tests for get_workflow_status branches (404 ValueError, 500 Exception)."""

    @pytest.fixture
    def client_status(self):
        """Client with workflow for status tests."""
        with (
            patch("redis.asyncio.Redis") as mock_async_redis,
            patch("redis.Redis") as mock_sync_redis,
            patch("rq.Queue") as mock_queue_class,
            patch("earth2studio.serve.server.workflow.register_all_workflows"),
        ):
            mock_async_instance = MagicMock()
            mock_async_instance.ping = AsyncMock(return_value=True)
            mock_async_instance.close = AsyncMock()
            mock_async_instance.get = AsyncMock(return_value=None)
            mock_async_redis.return_value = mock_async_instance
            mock_sync_instance = MagicMock()
            mock_sync_instance.ping = MagicMock(return_value=True)
            mock_sync_instance.close = MagicMock()
            mock_sync_redis.return_value = mock_sync_instance
            mock_queue_class.return_value = MagicMock()

            from earth2studio.serve.server.main import app
            from earth2studio.serve.server.workflow import (
                Workflow,
                WorkflowParameters,
                workflow_registry,
            )

            class StatusParams(WorkflowParameters):
                x: int = Field(default=1)

            class StatusWorkflow(Workflow):
                name = "status_wf"
                description = "Status test"
                Parameters = StatusParams

                @classmethod
                def validate_parameters(cls, parameters):
                    return StatusParams.validate(parameters)

                def run(self, parameters, execution_id):
                    return {"status": "ok"}

            workflow_registry._workflows["status_wf"] = StatusWorkflow
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c, mock_sync_instance
            if "status_wf" in workflow_registry._workflows:
                del workflow_registry._workflows["status_wf"]

    def test_get_workflow_status_value_error_404(self, client_status):
        """When _get_execution_data raises ValueError, returns 404."""
        client, mock_redis = client_status
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_registry:
            mock_wf = MagicMock()
            mock_wf._get_execution_data = MagicMock(
                side_effect=ValueError("Execution not found")
            )
            mock_registry.get_workflow_class.return_value = mock_wf
            response = client.get("/v1/infer/status_wf/exec_123/status")
        assert response.status_code == 404

    def test_get_workflow_status_exception_500(self, client_status):
        """When _get_execution_data raises Exception, returns 500."""
        client, mock_redis = client_status
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_registry:
            mock_wf = MagicMock()
            mock_wf._get_execution_data = MagicMock(
                side_effect=RuntimeError("Redis error")
            )
            mock_registry.get_workflow_class.return_value = mock_wf
            response = client.get("/v1/infer/status_wf/exec_123/status")
        assert response.status_code == 500
        assert "status" in response.json().get("detail", "").lower()


class TestGetWorkflowResultsBranches:
    """Tests for get_workflow_results (EXPIRED, 202, FAILED, CANCELLED, 404, 200, 500)."""

    @pytest.fixture
    def client_results(self, tmp_path):
        """Client and tmp dir for results tests."""
        with (
            patch("redis.asyncio.Redis") as mock_async_redis,
            patch("redis.Redis") as mock_sync_redis,
            patch("rq.Queue") as mock_queue_class,
            patch("earth2studio.serve.server.workflow.register_all_workflows"),
        ):
            mock_async_instance = MagicMock()
            mock_async_instance.ping = AsyncMock(return_value=True)
            mock_async_instance.close = AsyncMock()
            mock_async_instance.get = AsyncMock(return_value=None)
            mock_async_redis.return_value = mock_async_instance
            mock_sync_instance = MagicMock()
            mock_sync_instance.ping = MagicMock(return_value=True)
            mock_sync_instance.close = MagicMock()
            mock_sync_redis.return_value = mock_sync_instance
            mock_queue_class.return_value = MagicMock()

            from earth2studio.serve.server.main import app
            from earth2studio.serve.server.workflow import (
                Workflow,
                WorkflowParameters,
                WorkflowStatus,
                workflow_registry,
            )

            class ResultsParams(WorkflowParameters):
                x: int = Field(default=1)

            class ResultsWorkflow(Workflow):
                name = "results_wf"
                description = "Results test"
                Parameters = ResultsParams

                @classmethod
                def validate_parameters(cls, parameters):
                    return ResultsParams.validate(parameters)

                def run(self, parameters, execution_id):
                    return {"status": "ok"}

            workflow_registry._workflows["results_wf"] = ResultsWorkflow
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c, mock_sync_instance, tmp_path, WorkflowStatus
            if "results_wf" in workflow_registry._workflows:
                del workflow_registry._workflows["results_wf"]

    def _make_execution_data(self, status, execution_id="exec_1"):
        from earth2studio.serve.server.workflow import WorkflowResult

        return WorkflowResult(
            workflow_name="results_wf",
            execution_id=execution_id,
            status=status,
            start_time=datetime.now(timezone.utc).isoformat(),
            end_time=(
                datetime.now(timezone.utc).isoformat()
                if status in ("completed", "failed", "expired", "cancelled")
                else None
            ),
            error_message="failed" if status == "failed" else None,
        )

    def test_get_workflow_results_expired_400(self, client_results):
        """When status is EXPIRED, returns 400."""
        client, mock_redis, _, WorkflowStatus = client_results
        exec_data = self._make_execution_data(WorkflowStatus.EXPIRED)
        mock_redis.get.return_value = json.dumps(exec_data.model_dump()).encode()
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            mock_wf = MagicMock()
            mock_wf._get_execution_data = MagicMock(return_value=exec_data)
            mock_reg.get_workflow_class.return_value = mock_wf
            response = client.get("/v1/infer/results_wf/exec_1/results")
        assert response.status_code == 400
        assert "expired" in response.json().get("detail", {}).get("error", "").lower()

    def test_get_workflow_results_still_running_202(self, client_results):
        """When status is RUNNING, returns 202."""
        client, mock_redis, _, WorkflowStatus = client_results
        exec_data = self._make_execution_data(WorkflowStatus.RUNNING)
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            mock_wf = MagicMock()
            mock_wf._get_execution_data = MagicMock(return_value=exec_data)
            mock_reg.get_workflow_class.return_value = mock_wf
            response = client.get("/v1/infer/results_wf/exec_1/results")
        assert response.status_code == 202

    def test_get_workflow_results_failed_404(self, client_results):
        """When status is FAILED, returns 404."""
        client, mock_redis, _, WorkflowStatus = client_results
        exec_data = self._make_execution_data(WorkflowStatus.FAILED)
        exec_data.error_message = "Something failed"
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            mock_wf = MagicMock()
            mock_wf._get_execution_data = MagicMock(return_value=exec_data)
            mock_reg.get_workflow_class.return_value = mock_wf
            response = client.get("/v1/infer/results_wf/exec_1/results")
        assert response.status_code == 404
        assert "failed" in response.json().get("detail", {}).get("error", "").lower()

    def test_get_workflow_results_cancelled_404(self, client_results):
        """When status is CANCELLED, returns 404."""
        client, mock_redis, _, WorkflowStatus = client_results
        exec_data = self._make_execution_data(WorkflowStatus.CANCELLED)
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            mock_wf = MagicMock()
            mock_wf._get_execution_data = MagicMock(return_value=exec_data)
            mock_reg.get_workflow_class.return_value = mock_wf
            response = client.get("/v1/infer/results_wf/exec_1/results")
        assert response.status_code == 404
        assert "cancelled" in response.json().get("detail", {}).get("error", "").lower()

    def test_get_workflow_results_metadata_not_found_404(self, client_results):
        """When status is COMPLETED but metadata file missing, returns 404."""
        client, mock_redis, tmp_path, WorkflowStatus = client_results
        exec_data = self._make_execution_data(WorkflowStatus.COMPLETED)
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            with patch("earth2studio.serve.server.main.RESULTS_ZIP_DIR", tmp_path):
                mock_wf = MagicMock()
                mock_wf._get_execution_data = MagicMock(return_value=exec_data)
                mock_reg.get_workflow_class.return_value = mock_wf
                response = client.get("/v1/infer/results_wf/exec_1/results")
        assert response.status_code == 404
        assert "Metadata" in response.json().get("detail", {}).get("error", "")

    def test_get_workflow_results_success_200(self, client_results):
        """When status is COMPLETED and metadata exists, returns 200 with JSON."""
        client, mock_redis, tmp_path, WorkflowStatus = client_results
        exec_data = self._make_execution_data(WorkflowStatus.COMPLETED)
        metadata_path = tmp_path / "metadata_results_wf:exec_1.json"
        metadata_path.write_text(
            json.dumps({"request_id": "results_wf:exec_1", "status": "completed"})
        )
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            with patch("earth2studio.serve.server.main.RESULTS_ZIP_DIR", tmp_path):
                mock_wf = MagicMock()
                mock_wf._get_execution_data = MagicMock(return_value=exec_data)
                mock_reg.get_workflow_class.return_value = mock_wf
                response = client.get("/v1/infer/results_wf/exec_1/results")
        assert response.status_code == 200
        assert response.json().get("request_id") == "results_wf:exec_1"

    def test_get_workflow_results_value_error_400(self, client_results):
        """When _get_execution_data raises ValueError, returns 400."""
        client, _, _, _ = client_results
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            mock_wf = MagicMock()
            mock_wf._get_execution_data = MagicMock(
                side_effect=ValueError("Invalid key")
            )
            mock_reg.get_workflow_class.return_value = mock_wf
            response = client.get("/v1/infer/results_wf/exec_1/results")
        assert response.status_code == 400

    def test_get_workflow_results_exception_500(self, client_results):
        """When reading metadata raises, returns 500."""
        client, mock_redis, tmp_path, WorkflowStatus = client_results
        exec_data = self._make_execution_data(WorkflowStatus.COMPLETED)
        metadata_path = tmp_path / "metadata_results_wf:exec_1.json"
        metadata_path.write_text("not valid json {{")
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            with patch("earth2studio.serve.server.main.RESULTS_ZIP_DIR", tmp_path):
                mock_wf = MagicMock()
                mock_wf._get_execution_data = MagicMock(return_value=exec_data)
                mock_reg.get_workflow_class.return_value = mock_wf
                response = client.get("/v1/infer/results_wf/exec_1/results")
        assert response.status_code == 500


class TestGetWorkflowResultFileBranches:
    """Tests for get_workflow_result_file (404, 503, 403, 400, 200)."""

    @pytest.fixture
    def client_file(self, tmp_path):
        """Client and tmp dir for result file tests."""
        with (
            patch("redis.asyncio.Redis") as mock_async_redis,
            patch("redis.Redis") as mock_sync_redis,
            patch("rq.Queue") as mock_queue_class,
            patch("earth2studio.serve.server.workflow.register_all_workflows"),
        ):
            mock_async_instance = MagicMock()
            mock_async_instance.ping = AsyncMock(return_value=True)
            mock_async_instance.close = AsyncMock()
            mock_async_instance.get = AsyncMock(return_value=None)
            mock_async_redis.return_value = mock_async_instance
            mock_sync_instance = MagicMock()
            mock_sync_instance.ping = MagicMock(return_value=True)
            mock_sync_instance.close = MagicMock()
            mock_sync_redis.return_value = mock_sync_instance
            mock_queue_class.return_value = MagicMock()

            from earth2studio.serve.server.main import app
            from earth2studio.serve.server.workflow import (
                Workflow,
                WorkflowParameters,
                WorkflowStatus,
                workflow_registry,
            )

            class FileParams(WorkflowParameters):
                x: int = Field(default=1)

            class FileWorkflow(Workflow):
                name = "file_wf"
                description = "File test"
                Parameters = FileParams

                @classmethod
                def validate_parameters(cls, parameters):
                    return FileParams.validate(parameters)

                def run(self, parameters, execution_id):
                    return {"status": "ok"}

            workflow_registry._workflows["file_wf"] = FileWorkflow
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c, mock_async_instance, mock_sync_instance, tmp_path, WorkflowStatus
            if "file_wf" in workflow_registry._workflows:
                del workflow_registry._workflows["file_wf"]

    def _exec_data(self, status=None):
        from earth2studio.serve.server.workflow import (
            WorkflowResult,
            WorkflowStatus,
        )

        if status is None:
            status = WorkflowStatus.COMPLETED
        return WorkflowResult(
            workflow_name="file_wf",
            execution_id="exec_1",
            status=status,
            start_time=datetime.now(timezone.utc).isoformat(),
            end_time=datetime.now(timezone.utc).isoformat(),
        )

    def test_get_workflow_result_file_workflow_not_found_404(self, client_file):
        """When workflow not in registry, returns 404."""
        client, *_ = client_file
        response = client.get("/v1/infer/nonexistent_wf/exec_1/results/out.nc")
        assert response.status_code == 404

    def test_get_workflow_result_file_status_not_completed_404(self, client_file):
        """When status is not COMPLETED, returns 404."""
        client, mock_async, mock_sync, _, WorkflowStatus = client_file
        exec_data = self._exec_data(WorkflowStatus.RUNNING)
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            mock_wf = MagicMock()
            mock_wf._get_execution_data = MagicMock(return_value=exec_data)
            mock_reg.get_workflow_class.return_value = mock_wf
            response = client.get("/v1/infer/file_wf/exec_1/results/out.nc")
        assert response.status_code == 404
        assert (
            "not available"
            in response.json().get("detail", {}).get("error", "").lower()
        )

    def test_get_workflow_result_file_redis_none_503(self, client_file):
        """When redis_client (async) is None for file request, returns 503."""
        client, mock_async, mock_sync, _, WorkflowStatus = client_file
        exec_data = self._exec_data()
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            with patch("earth2studio.serve.server.main.redis_client", None):
                mock_wf = MagicMock()
                mock_wf._get_execution_data = MagicMock(return_value=exec_data)
                mock_reg.get_workflow_class.return_value = mock_wf
                response = client.get("/v1/infer/file_wf/exec_1/results/out.nc")
        assert response.status_code == 503

    def test_get_workflow_result_file_zip_not_in_redis_404(self, client_file):
        """When filepath equals request_id but zip not in Redis, returns 404."""
        client, mock_async, mock_sync, tmp_path, WorkflowStatus = client_file
        exec_data = self._exec_data()
        mock_async.get = AsyncMock(return_value=None)
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            with patch("earth2studio.serve.server.main.redis_client", mock_async):
                with patch("earth2studio.serve.server.main.RESULTS_ZIP_DIR", tmp_path):
                    mock_wf = MagicMock()
                    mock_wf._get_execution_data = MagicMock(return_value=exec_data)
                    mock_reg.get_workflow_class.return_value = mock_wf
                    response = client.get(
                        "/v1/infer/file_wf/exec_1/results/file_wf:exec_1"
                    )
        assert response.status_code == 404
        assert "Zip" in response.json().get("detail", {}).get("error", "")

    def test_get_workflow_result_file_output_dir_not_found_404(self, client_file):
        """When output_path_key not in Redis, returns 404."""
        client, mock_async, mock_sync, tmp_path, WorkflowStatus = client_file
        exec_data = self._exec_data()
        mock_async.get = AsyncMock(return_value=None)
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            with patch("earth2studio.serve.server.main.redis_client", mock_async):
                mock_wf = MagicMock()
                mock_wf._get_execution_data = MagicMock(return_value=exec_data)
                mock_reg.get_workflow_class.return_value = mock_wf
                response = client.get("/v1/infer/file_wf/exec_1/results/somefile.txt")
        assert response.status_code == 404
        assert "Output directory" in response.json().get("detail", {}).get("error", "")

    def test_get_workflow_result_file_path_traversal_403(self, client_file):
        """When filepath escapes output dir, returns 403 (or 404 if path invalid)."""
        client, mock_async, mock_sync, tmp_path, WorkflowStatus = client_file
        exec_data = self._exec_data()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        # Create a file inside output_dir; request it via path that escapes (..)
        inner = output_dir / "inner.txt"
        inner.write_text("data")
        mock_async.get = AsyncMock(return_value=str(output_dir))
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            with patch("earth2studio.serve.server.main.redis_client", mock_async):
                mock_wf = MagicMock()
                mock_wf._get_execution_data = MagicMock(return_value=exec_data)
                mock_reg.get_workflow_class.return_value = mock_wf
                # Path ../output/inner.txt from output_dir resolves to output_dir/inner.txt
                # Use path that clearly escapes: .. with resolved path outside
                response = client.get(
                    "/v1/infer/file_wf/exec_1/results/..%2F..%2F..%2Fetc%2Fpasswd"
                )
        # Path traversal must not succeed: either 403 (access denied) or 404
        assert response.status_code in (403, 404)
        if response.status_code == 403:
            assert "Access denied" in response.json().get("detail", {}).get("error", "")

    def test_get_workflow_result_file_not_found_404(self, client_file):
        """When file does not exist in output dir, returns 404."""
        client, mock_async, mock_sync, tmp_path, WorkflowStatus = client_file
        exec_data = self._exec_data()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        mock_async.get = AsyncMock(return_value=str(output_dir))
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            with patch("earth2studio.serve.server.main.redis_client", mock_async):
                mock_wf = MagicMock()
                mock_wf._get_execution_data = MagicMock(return_value=exec_data)
                mock_reg.get_workflow_class.return_value = mock_wf
                response = client.get(
                    "/v1/infer/file_wf/exec_1/results/nonexistent.txt"
                )
        assert response.status_code == 404
        assert "File not found" in response.json().get("detail", {}).get("error", "")

    def test_get_workflow_result_file_not_a_file_400(self, client_file):
        """When path is a directory, returns 400."""
        client, mock_async, mock_sync, tmp_path, WorkflowStatus = client_file
        exec_data = self._exec_data()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        subdir = output_dir / "subdir"
        subdir.mkdir()
        mock_async.get = AsyncMock(return_value=str(output_dir))
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            with patch("earth2studio.serve.server.main.redis_client", mock_async):
                mock_wf = MagicMock()
                mock_wf._get_execution_data = MagicMock(return_value=exec_data)
                mock_reg.get_workflow_class.return_value = mock_wf
                response = client.get("/v1/infer/file_wf/exec_1/results/subdir")
        assert response.status_code == 400
        assert "Not a file" in response.json().get("detail", {}).get("error", "")

    def test_get_workflow_result_file_stream_file_200(self, client_file):
        """When file exists, returns 200 with streamed content."""
        client, mock_async, mock_sync, tmp_path, WorkflowStatus = client_file
        exec_data = self._exec_data()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        out_file = output_dir / "data.txt"
        out_file.write_text("hello world")
        mock_async.get = AsyncMock(return_value=str(output_dir))
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            with patch("earth2studio.serve.server.main.redis_client", mock_async):
                mock_wf = MagicMock()
                mock_wf._get_execution_data = MagicMock(return_value=exec_data)
                mock_reg.get_workflow_class.return_value = mock_wf
                response = client.get("/v1/infer/file_wf/exec_1/results/data.txt")
        assert response.status_code == 200
        assert response.text == "hello world"
        accept_ranges = response.headers.get("accept-ranges") or response.headers.get(
            "Accept-Ranges", ""
        )
        assert accept_ranges == "bytes"

    def test_get_workflow_result_file_range_returns_206(self, client_file):
        """Range request returns 206 with Content-Range and partial body."""
        client, mock_async, mock_sync, tmp_path, WorkflowStatus = client_file
        exec_data = self._exec_data()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        out_file = output_dir / "data.txt"
        out_file.write_text("hello world")
        mock_async.get = AsyncMock(return_value=str(output_dir))
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            with patch("earth2studio.serve.server.main.redis_client", mock_async):
                mock_wf = MagicMock()
                mock_wf._get_execution_data = MagicMock(return_value=exec_data)
                mock_reg.get_workflow_class.return_value = mock_wf
                response = client.get(
                    "/v1/infer/file_wf/exec_1/results/data.txt",
                    headers={"Range": "bytes=0-4"},
                )
        assert response.status_code == 206
        assert response.content == b"hello"
        accept_ranges = response.headers.get("accept-ranges") or response.headers.get(
            "Accept-Ranges", ""
        )
        assert accept_ranges == "bytes"
        cl = response.headers.get("content-length") or response.headers.get(
            "Content-Length", ""
        )
        assert cl == "5"
        cr = (
            response.headers.get("content-range")
            or response.headers.get("Content-Range")
            or ""
        )
        assert "bytes 0-4/" in cr and cr.rstrip().endswith("/11")


class TestLifespanBranches:
    """Tests for lifespan startup exception branches (lines 236-242)."""

    def test_lifespan_workflow_registration_generic_exception_continues(self):
        """When register_all_workflows raises non-ImportError, app still starts."""
        with (
            patch("redis.asyncio.Redis") as mock_async_redis,
            patch("redis.Redis") as mock_sync_redis,
            patch("rq.Queue"),
            patch(
                "earth2studio.serve.server.workflow.register_all_workflows",
                side_effect=RuntimeError("unexpected registration error"),
            ),
        ):
            mock_async_instance = MagicMock()
            mock_async_instance.ping = AsyncMock(return_value=True)
            mock_async_instance.close = AsyncMock()
            mock_async_redis.return_value = mock_async_instance
            mock_sync_instance = MagicMock()
            mock_sync_instance.ping = MagicMock(return_value=True)
            mock_sync_instance.close = MagicMock()
            mock_sync_redis.return_value = mock_sync_instance

            from earth2studio.serve.server.main import app

            with TestClient(app, raise_server_exceptions=False) as c:
                response = c.get("/liveness")
            assert response.status_code == 200

    def test_lifespan_redis_ping_failure_raises(self):
        """When Redis ping fails, lifespan raises and app fails to start."""
        with (
            patch("redis.asyncio.Redis") as mock_async_redis,
            patch("redis.Redis") as mock_sync_redis,
            patch("rq.Queue"),
            patch("earth2studio.serve.server.workflow.register_all_workflows"),
        ):
            mock_async_instance = MagicMock()
            mock_async_instance.ping = AsyncMock(
                side_effect=ConnectionError("Redis unavailable")
            )
            mock_async_instance.close = AsyncMock()
            mock_async_redis.return_value = mock_async_instance
            mock_sync_instance = MagicMock()
            mock_sync_instance.ping = MagicMock(return_value=True)
            mock_sync_redis.return_value = mock_sync_instance

            from earth2studio.serve.server.main import app

            with pytest.raises(Exception):
                with TestClient(app):
                    pass


class TestHealthCheckWithoutScriptDir:
    """Tests health endpoint when SCRIPT_DIR env var is absent (lines 304-305)."""

    @pytest.fixture
    def client_probes(self):
        """Client for probe endpoints."""
        with (
            patch("redis.asyncio.Redis") as mock_async_redis,
            patch("redis.Redis") as mock_sync_redis,
            patch("rq.Queue"),
            patch("earth2studio.serve.server.workflow.register_all_workflows"),
        ):
            mock_async_instance = MagicMock()
            mock_async_instance.ping = AsyncMock(return_value=True)
            mock_async_instance.close = AsyncMock()
            mock_async_redis.return_value = mock_async_instance
            mock_sync_instance = MagicMock()
            mock_sync_instance.ping = MagicMock(return_value=True)
            mock_sync_instance.close = MagicMock()
            mock_sync_redis.return_value = mock_sync_instance

            from earth2studio.serve.server.main import app

            with TestClient(app, raise_server_exceptions=False) as c:
                yield c

    def test_health_follows_repo_relative_path_when_script_dir_empty(
        self, client_probes
    ):
        """Health check uses repo-relative script path when SCRIPT_DIR is empty/unset."""
        with patch.dict(os.environ, {"SCRIPT_DIR": ""}):
            with patch(
                "earth2studio.serve.server.main.asyncio.create_subprocess_exec"
            ) as mock_exec:
                mock_proc = MagicMock()
                mock_proc.returncode = 0
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                mock_exec.return_value = mock_proc

                response = client_probes.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


class TestNotExposedWorkflowEndpoints:
    """Tests for 404 when workflow is registered but not exposed (lines 427, 539, 671, 743, 876)."""

    @pytest.fixture
    def client_with_workflow(self):
        """Standard client with a registered workflow."""
        with (
            patch("redis.asyncio.Redis") as mock_async_redis,
            patch("redis.Redis") as mock_sync_redis,
            patch("rq.Queue"),
            patch("earth2studio.serve.server.workflow.register_all_workflows"),
        ):
            mock_async_instance = MagicMock()
            mock_async_instance.ping = AsyncMock(return_value=True)
            mock_async_instance.close = AsyncMock()
            mock_async_redis.return_value = mock_async_instance
            mock_sync_instance = MagicMock()
            mock_sync_instance.ping = MagicMock(return_value=True)
            mock_sync_instance.close = MagicMock()
            mock_sync_redis.return_value = mock_sync_instance

            from earth2studio.serve.server.main import app
            from earth2studio.serve.server.workflow import (
                Workflow,
                WorkflowParameters,
                workflow_registry,
            )

            class TestEndpointsParams(WorkflowParameters):
                x: int = Field(default=1)

            class TestEndpointsWf(Workflow):
                name = "test_endpoints_wf"
                description = "Test"
                Parameters = TestEndpointsParams

                @classmethod
                def validate_parameters(cls, parameters):
                    return TestEndpointsParams.validate(parameters)

                def run(self, parameters, execution_id):
                    return {"status": "ok"}

            workflow_registry._workflows["test_endpoints_wf"] = TestEndpointsWf

            with TestClient(app, raise_server_exceptions=False) as c:
                yield c

            if "test_endpoints_wf" in workflow_registry._workflows:
                del workflow_registry._workflows["test_endpoints_wf"]

    def test_schema_not_exposed_404(self, client_with_workflow):
        """get_workflow_schema returns 404 when workflow is not exposed."""
        with patch(
            "earth2studio.serve.server.main.workflow_registry.is_workflow_exposed",
            return_value=False,
        ):
            response = client_with_workflow.get(
                "/v1/workflows/test_endpoints_wf/schema"
            )
        assert response.status_code == 404
        assert "not exposed" in response.json().get("detail", "").lower()

    def test_execute_not_exposed_404(self, client_with_workflow):
        """execute_workflow returns 404 when workflow is not exposed."""
        with patch(
            "earth2studio.serve.server.main.workflow_registry.is_workflow_exposed",
            return_value=False,
        ):
            response = client_with_workflow.post(
                "/v1/infer/test_endpoints_wf", json={"parameters": {}}
            )
        assert response.status_code == 404

    def test_get_status_workflow_not_found_404(self, client_with_workflow):
        """get_workflow_status returns 404 when workflow not found."""
        response = client_with_workflow.get("/v1/infer/nonexistent_wf/exec_1/status")
        assert response.status_code == 404
        assert "not found" in response.json().get("detail", "").lower()

    def test_get_status_not_exposed_404(self, client_with_workflow):
        """get_workflow_status returns 404 when workflow is not exposed."""
        with patch(
            "earth2studio.serve.server.main.workflow_registry.is_workflow_exposed",
            return_value=False,
        ):
            response = client_with_workflow.get(
                "/v1/infer/test_endpoints_wf/exec_1/status"
            )
        assert response.status_code == 404

    def test_get_results_workflow_not_found_404(self, client_with_workflow):
        """get_workflow_results returns 404 when workflow not found."""
        response = client_with_workflow.get("/v1/infer/nonexistent_wf/exec_1/results")
        assert response.status_code == 404

    def test_get_results_not_exposed_404(self, client_with_workflow):
        """get_workflow_results returns 404 when workflow is not exposed."""
        with patch(
            "earth2studio.serve.server.main.workflow_registry.is_workflow_exposed",
            return_value=False,
        ):
            response = client_with_workflow.get(
                "/v1/infer/test_endpoints_wf/exec_1/results"
            )
        assert response.status_code == 404

    def test_get_result_file_not_exposed_404(self, client_with_workflow):
        """get_workflow_result_file returns 404 when workflow is not exposed."""
        with patch(
            "earth2studio.serve.server.main.workflow_registry.is_workflow_exposed",
            return_value=False,
        ):
            response = client_with_workflow.get(
                "/v1/infer/test_endpoints_wf/exec_1/results/file.nc"
            )
        assert response.status_code == 404


class TestExecuteWorkflowAdditionalBranches:
    """Additional coverage for execute_workflow (lines 598, 605-609)."""

    @pytest.fixture
    def client_exec2(self):
        """Client with workflow for additional execute branch tests."""
        from earth2studio.serve.server.workflow import Workflow, WorkflowParameters

        with (
            patch("redis.asyncio.Redis") as mock_async_redis,
            patch("redis.Redis") as mock_sync_redis,
            patch("rq.Queue") as mock_queue_class,
            patch("earth2studio.serve.server.workflow.register_all_workflows"),
        ):
            mock_async_instance = MagicMock()
            mock_async_instance.ping = AsyncMock(return_value=True)
            mock_async_instance.close = AsyncMock()
            mock_async_instance.get = AsyncMock(return_value=None)
            mock_async_redis.return_value = mock_async_instance

            mock_sync_instance = MagicMock()
            mock_sync_instance.ping = MagicMock(return_value=True)
            mock_sync_instance.close = MagicMock()
            mock_sync_instance.setex = MagicMock()
            mock_sync_instance.llen = MagicMock(return_value=0)
            mock_sync_redis.return_value = mock_sync_instance

            mock_queue = MagicMock()
            mock_job = MagicMock()
            mock_job.id = "exec2_wf_exec_123"
            mock_queue.enqueue = MagicMock(return_value=mock_job)
            mock_queue_class.return_value = mock_queue

            class Exec2Params(WorkflowParameters):
                x: int = Field(default=1)

            class Exec2Workflow(Workflow):
                name = "exec2_wf"
                description = "For additional execute tests"
                Parameters = Exec2Params

                @classmethod
                def validate_parameters(cls, parameters):
                    return Exec2Params.validate(parameters)

                def run(self, parameters, execution_id):
                    return {"status": "ok"}

            from earth2studio.serve.server.main import app
            from earth2studio.serve.server.workflow import workflow_registry

            workflow_registry._workflows["exec2_wf"] = Exec2Workflow
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c, mock_sync_instance, mock_queue, Exec2Workflow
            if "exec2_wf" in workflow_registry._workflows:
                del workflow_registry._workflows["exec2_wf"]

    def test_execute_llen_raises_after_enqueue_500(self, client_exec2):
        """When llen raises during queue position lookup, returns 500."""
        client, mock_redis, mock_queue, _ = client_exec2
        # 5 queue checks in admission control + 1 for queue position lookup
        mock_redis.llen.side_effect = [0, 0, 0, 0, 0, RuntimeError("redis error")]
        with patch("earth2studio.serve.server.main.inference_queue", mock_queue):
            response = client.post(
                "/v1/infer/exec2_wf",
                json={"parameters": {}},
            )
        assert response.status_code == 500

    def test_execute_redis_none_during_queue_position_503(self, client_exec2):
        """When redis_sync_client is None at queue position check, returns 503."""
        client, mock_redis, mock_queue, wf_class = client_exec2
        with patch("earth2studio.serve.server.main.check_admission_control"):
            with patch.object(wf_class, "_save_execution_data"):
                with patch("earth2studio.serve.server.main.redis_sync_client", None):
                    with patch(
                        "earth2studio.serve.server.main.inference_queue", mock_queue
                    ):
                        response = client.post(
                            "/v1/infer/exec2_wf",
                            json={"parameters": {}},
                        )
        assert response.status_code == 503
        assert "Redis" in response.json().get("detail", "")


class TestGetWorkflowResultFileAdditionalBranches:
    """Additional tests for get_workflow_result_file (lines 895, 903, 919-952, 986, 1035, 1052, 1062-1066)."""

    @pytest.fixture
    def client_file2(self, tmp_path):
        """Client and tmp dir for additional result file tests."""
        with (
            patch("redis.asyncio.Redis") as mock_async_redis,
            patch("redis.Redis") as mock_sync_redis,
            patch("rq.Queue") as mock_queue_class,
            patch("earth2studio.serve.server.workflow.register_all_workflows"),
        ):
            mock_async_instance = MagicMock()
            mock_async_instance.ping = AsyncMock(return_value=True)
            mock_async_instance.close = AsyncMock()
            mock_async_instance.get = AsyncMock(return_value=None)
            mock_async_redis.return_value = mock_async_instance
            mock_sync_instance = MagicMock()
            mock_sync_instance.ping = MagicMock(return_value=True)
            mock_sync_instance.close = MagicMock()
            mock_sync_redis.return_value = mock_sync_instance
            mock_queue_class.return_value = MagicMock()

            from earth2studio.serve.server.main import app
            from earth2studio.serve.server.workflow import (
                Workflow,
                WorkflowParameters,
                WorkflowStatus,
                workflow_registry,
            )

            class File2Params(WorkflowParameters):
                x: int = Field(default=1)

            class File2Workflow(Workflow):
                name = "file2_wf"
                description = "File2 test"
                Parameters = File2Params

                @classmethod
                def validate_parameters(cls, parameters):
                    return File2Params.validate(parameters)

                def run(self, parameters, execution_id):
                    return {"status": "ok"}

            workflow_registry._workflows["file2_wf"] = File2Workflow
            with TestClient(app, raise_server_exceptions=False) as c:
                yield c, mock_async_instance, tmp_path, WorkflowStatus
            if "file2_wf" in workflow_registry._workflows:
                del workflow_registry._workflows["file2_wf"]

    def _completed_exec_data(self):
        from earth2studio.serve.server.workflow import WorkflowResult, WorkflowStatus

        return WorkflowResult(
            workflow_name="file2_wf",
            execution_id="exec_1",
            status=WorkflowStatus.COMPLETED,
            start_time=datetime.now(timezone.utc).isoformat(),
            end_time=datetime.now(timezone.utc).isoformat(),
        )

    def test_get_result_file_value_error_in_exec_data_404(self, client_file2):
        """When _get_execution_data raises ValueError, returns 404."""
        client, *_ = client_file2
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            mock_wf = MagicMock()
            mock_wf._get_execution_data = MagicMock(
                side_effect=ValueError("Execution not found")
            )
            mock_reg.get_workflow_class.return_value = mock_wf
            response = client.get("/v1/infer/file2_wf/exec_1/results/file.nc")
        assert response.status_code == 404
        assert "Execution not found" in response.json().get("detail", "")

    def test_get_result_file_redis_none_for_zip_path_503(self, client_file2):
        """When filepath == request_id but redis_client is None, returns 503."""
        client, mock_async, tmp_path, _ = client_file2
        exec_data = self._completed_exec_data()
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            with patch("earth2studio.serve.server.main.redis_client", None):
                mock_wf = MagicMock()
                mock_wf._get_execution_data = MagicMock(return_value=exec_data)
                mock_reg.get_workflow_class.return_value = mock_wf
                response = client.get(
                    "/v1/infer/file2_wf/exec_1/results/file2_wf:exec_1"
                )
        assert response.status_code == 503

    def test_get_result_file_zip_not_on_disk_404(self, client_file2):
        """When zip key in Redis but zip file missing from disk, returns 404."""
        client, mock_async, tmp_path, _ = client_file2
        exec_data = self._completed_exec_data()
        mock_async.get = AsyncMock(return_value="missing_zip.zip")
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            with patch("earth2studio.serve.server.main.redis_client", mock_async):
                with patch("earth2studio.serve.server.main.RESULTS_ZIP_DIR", tmp_path):
                    mock_wf = MagicMock()
                    mock_wf._get_execution_data = MagicMock(return_value=exec_data)
                    mock_reg.get_workflow_class.return_value = mock_wf
                    response = client.get(
                        "/v1/infer/file2_wf/exec_1/results/file2_wf:exec_1"
                    )
        assert response.status_code == 404
        assert "disk" in response.json().get("detail", {}).get("error", "").lower()

    def test_get_result_file_zip_stream_success(self, client_file2):
        """When zip file exists on disk, returns 200 with streamed content."""
        client, mock_async, tmp_path, _ = client_file2
        exec_data = self._completed_exec_data()
        zip_file = tmp_path / "results.zip"
        zip_file.write_bytes(b"PK\x03\x04fake_zip_content")
        mock_async.get = AsyncMock(return_value="results.zip")
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            with patch("earth2studio.serve.server.main.redis_client", mock_async):
                with patch("earth2studio.serve.server.main.RESULTS_ZIP_DIR", tmp_path):
                    mock_wf = MagicMock()
                    mock_wf._get_execution_data = MagicMock(return_value=exec_data)
                    mock_reg.get_workflow_class.return_value = mock_wf
                    response = client.get(
                        "/v1/infer/file2_wf/exec_1/results/file2_wf:exec_1"
                    )
        assert response.status_code == 200
        assert 'results.zip"' in response.headers.get("content-disposition", "")

    def test_get_result_file_filepath_with_output_dir_prefix(self, client_file2):
        """When filepath starts with output dir name, the prefix is stripped."""
        client, mock_async, tmp_path, _ = client_file2
        exec_data = self._completed_exec_data()
        output_dir = tmp_path / "exec_1"
        output_dir.mkdir()
        data_file = output_dir / "data.txt"
        data_file.write_text("contents")
        mock_async.get = AsyncMock(return_value=str(output_dir))
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            with patch("earth2studio.serve.server.main.redis_client", mock_async):
                mock_wf = MagicMock()
                mock_wf._get_execution_data = MagicMock(return_value=exec_data)
                mock_reg.get_workflow_class.return_value = mock_wf
                # filepath starts with output dir name (exec_1/data.txt)
                response = client.get(
                    "/v1/infer/file2_wf/exec_1/results/exec_1/data.txt"
                )
        assert response.status_code == 200
        assert response.text == "contents"

    def test_get_result_file_no_mime_type_uses_octet_stream(self, client_file2):
        """Files with no recognized MIME type use application/octet-stream."""
        client, mock_async, tmp_path, _ = client_file2
        exec_data = self._completed_exec_data()
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        unknown_file = output_dir / "data.unknownext99999"
        unknown_file.write_bytes(b"\x00\x01\x02\x03")
        mock_async.get = AsyncMock(return_value=str(output_dir))
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            with patch("earth2studio.serve.server.main.redis_client", mock_async):
                mock_wf = MagicMock()
                mock_wf._get_execution_data = MagicMock(return_value=exec_data)
                mock_reg.get_workflow_class.return_value = mock_wf
                with patch("mimetypes.guess_type", return_value=(None, None)):
                    response = client.get(
                        "/v1/infer/file2_wf/exec_1/results/data.unknownext99999"
                    )
        assert response.status_code == 200
        assert "octet-stream" in response.headers.get("content-type", "")

    def test_get_result_file_generic_exception_500(self, client_file2):
        """When an unexpected exception occurs in the file handler, returns 500."""
        client, mock_async, tmp_path, _ = client_file2
        exec_data = self._completed_exec_data()
        mock_async.get = AsyncMock(side_effect=RuntimeError("unexpected redis error"))
        with patch("earth2studio.serve.server.main.workflow_registry") as mock_reg:
            with patch("earth2studio.serve.server.main.redis_client", mock_async):
                mock_wf = MagicMock()
                mock_wf._get_execution_data = MagicMock(return_value=exec_data)
                mock_reg.get_workflow_class.return_value = mock_wf
                response = client.get(
                    "/v1/infer/file2_wf/exec_1/results/file2_wf:exec_1"
                )
        assert response.status_code == 500
        assert "Failed to retrieve file" in response.json().get("detail", {}).get(
            "error", ""
        )


class TestMainEntrypoint:
    """Test main module entrypoint (covers line 1044)."""

    def test_main_entrypoint_calls_uvicorn(self):
        """When run as __main__, uvicorn.run is called."""
        import runpy

        import earth2studio.serve.server.main as main_module

        with patch("uvicorn.run") as mock_run:
            runpy.run_path(main_module.__file__, run_name="__main__")
            mock_run.assert_called_once()
            call = mock_run.call_args
            # runpy re-executes the module so app is a new instance; check type and args
            assert call[0][0].__class__.__name__ == "FastAPI"
            assert call[1]["host"] == main_module.config.server.host
            assert call[1]["port"] == main_module.config.server.port
