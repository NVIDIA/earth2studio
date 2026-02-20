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
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import redis  # type: ignore[import-untyped]
from api_server.config import get_config  # type: ignore[import-untyped]
from api_server.workflow import (  # type: ignore[import-untyped]
    Workflow,
    WorkflowParameters,
    WorkflowProgress,
    WorkflowRegistry,
    WorkflowResult,
    WorkflowStatus,
    parse_workflow_directories_from_env,
    register_all_workflows,
    workflow_registry,
)
from pydantic import Field, ValidationError  # type: ignore[import-untyped]

# imitate API server environment
os.environ["EARTH2STUDIO_API_ACTIVE"] = "1"


def get_test_config(base_path: str | None = None):  # type: ignore[no-untyped-def]
    """Gets a config with overrides for testing."""
    config = get_config()
    if base_path is None:
        base_path = tempfile.mkdtemp(prefix="e2s_testing_")
    config.paths.default_output_dir = base_path
    config.paths.results_zip_dir = base_path
    return config


@pytest.fixture(autouse=True)
def setup_test_paths():  # type: ignore[no-untyped-def]
    """Automatically patch config paths for all tests in this module."""
    with tempfile.TemporaryDirectory(prefix="e2s_testing_") as tmpdir:
        test_config = get_test_config(tmpdir)
        with (
            patch(
                "api_server.workflow.workflow.get_config",
                return_value=test_config,
            ),
            patch("api_server.config.get_config", return_value=test_config),
            patch("api_server.workflow.workflow.config", test_config),
        ):
            yield test_config


# Test WorkflowStatus Constants
class TestWorkflowStatus:
    """Test WorkflowStatus constants"""

    def test_workflow_status_values(self):
        """Test all workflow status values exist and are strings"""
        assert WorkflowStatus.QUEUED == "queued"
        assert WorkflowStatus.RUNNING == "running"
        assert WorkflowStatus.PENDING_RESULTS == "pending_results"
        assert WorkflowStatus.COMPLETED == "completed"
        assert WorkflowStatus.FAILED == "failed"
        assert WorkflowStatus.CANCELLED == "cancelled"
        assert WorkflowStatus.EXPIRED == "expired"

    def test_workflow_status_constants(self):
        """Test all status constants are present"""
        assert hasattr(WorkflowStatus, "QUEUED")
        assert hasattr(WorkflowStatus, "RUNNING")
        assert hasattr(WorkflowStatus, "PENDING_RESULTS")
        assert hasattr(WorkflowStatus, "COMPLETED")
        assert hasattr(WorkflowStatus, "FAILED")
        assert hasattr(WorkflowStatus, "CANCELLED")
        assert hasattr(WorkflowStatus, "EXPIRED")


# Test WorkflowParameters
class TestWorkflowParameters:
    """Test WorkflowParameters base class"""

    def test_validate_with_dict(self):
        """Test validation with dictionary input"""
        data = {}
        result = WorkflowParameters.validate(data)
        assert isinstance(result, WorkflowParameters)

    def test_validate_with_instance(self):
        """Test validation with WorkflowParameters instance"""
        params = WorkflowParameters()
        result = WorkflowParameters.validate(params)
        assert isinstance(result, WorkflowParameters)

    def test_validate_with_invalid_type(self):
        """Test validation with invalid type raises ValueError"""
        with pytest.raises(ValueError, match="Parameters must be dict or"):
            WorkflowParameters.validate("invalid")

    def test_custom_parameters_validation(self):
        """Test custom parameters with fields"""

        class CustomParams(WorkflowParameters):
            value: int = Field(gt=0, description="Must be positive")
            name: str

        # Valid params
        result = CustomParams.validate({"value": 10, "name": "test"})
        assert result.value == 10
        assert result.name == "test"

        # Invalid params - negative value
        with pytest.raises(ValidationError):
            CustomParams.validate({"value": -1, "name": "test"})

        # Missing required field
        with pytest.raises(ValidationError):
            CustomParams.validate({"value": 10})

    def test_extra_fields_rejected(self):
        """Test that extra/unknown fields are rejected due to extra='forbid'"""

        class StrictParams(WorkflowParameters):
            value: int = Field(default=0)
            name: str = Field(default="default")

        # Valid params - only known fields
        result = StrictParams.validate({"value": 10, "name": "test"})
        assert result.value == 10
        assert result.name == "test"

        # Invalid params - contains unknown field 'unknown_field'
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            StrictParams.validate({"value": 10, "name": "test", "unknown_field": "bad"})

        # Invalid params - multiple unknown fields
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            StrictParams.validate(
                {
                    "value": 10,
                    "name": "test",
                    "extra1": "bad",
                    "extra2": "also bad",
                }
            )

    def test_missing_fields_use_defaults(self):
        """Test that missing optional fields use default values"""

        class ParamsWithDefaults(WorkflowParameters):
            required_field: str
            optional_int: int = Field(default=42)
            optional_str: str = Field(default="default_value")
            optional_bool: bool = Field(default=True)

        # Provide only required field
        result = ParamsWithDefaults.validate({"required_field": "test"})
        assert result.required_field == "test"
        assert result.optional_int == 42
        assert result.optional_str == "default_value"
        assert result.optional_bool is True

        # Override some defaults
        result2 = ParamsWithDefaults.validate(
            {"required_field": "test", "optional_int": 100}
        )
        assert result2.required_field == "test"
        assert result2.optional_int == 100
        assert result2.optional_str == "default_value"  # Still uses default
        assert result2.optional_bool is True  # Still uses default

    def test_field_type_validation(self):
        """Test that field types are validated correctly"""

        class TypedParams(WorkflowParameters):
            int_field: int
            str_field: str
            bool_field: bool
            list_field: list

        # Valid types
        result = TypedParams.validate(
            {
                "int_field": 10,
                "str_field": "text",
                "bool_field": True,
                "list_field": [1, 2, 3],
            }
        )
        assert result.int_field == 10
        assert result.str_field == "text"
        assert result.bool_field is True
        assert result.list_field == [1, 2, 3]

        # Invalid type - string for int field
        with pytest.raises(ValidationError):
            TypedParams.validate(
                {
                    "int_field": "not_an_int",
                    "str_field": "text",
                    "bool_field": True,
                    "list_field": [],
                }
            )

        # Invalid type - int for bool field
        with pytest.raises(ValidationError):
            TypedParams.validate(
                {
                    "int_field": 10,
                    "str_field": "text",
                    "bool_field": "not_a_bool",
                    "list_field": [],
                }
            )

    def test_field_constraints_validation(self):
        """Test that field constraints (gt, ge, lt, le, min_length, etc.) are validated"""

        class ConstrainedParams(WorkflowParameters):
            positive_int: int = Field(gt=0, description="Must be positive")
            bounded_int: int = Field(ge=1, le=100, description="Between 1 and 100")
            min_length_str: str = Field(min_length=3, description="At least 3 chars")
            max_length_str: str = Field(max_length=10, description="At most 10 chars")

        # Valid constrained params
        result = ConstrainedParams.validate(
            {
                "positive_int": 5,
                "bounded_int": 50,
                "min_length_str": "abc",
                "max_length_str": "short",
            }
        )
        assert result.positive_int == 5
        assert result.bounded_int == 50

        # Invalid - positive_int not positive
        with pytest.raises(ValidationError):
            ConstrainedParams.validate(
                {
                    "positive_int": 0,
                    "bounded_int": 50,
                    "min_length_str": "abc",
                    "max_length_str": "short",
                }
            )

        # Invalid - bounded_int out of range
        with pytest.raises(ValidationError):
            ConstrainedParams.validate(
                {
                    "positive_int": 5,
                    "bounded_int": 101,
                    "min_length_str": "abc",
                    "max_length_str": "short",
                }
            )

        # Invalid - min_length_str too short
        with pytest.raises(ValidationError):
            ConstrainedParams.validate(
                {
                    "positive_int": 5,
                    "bounded_int": 50,
                    "min_length_str": "ab",
                    "max_length_str": "short",
                }
            )

        # Invalid - max_length_str too long
        with pytest.raises(ValidationError):
            ConstrainedParams.validate(
                {
                    "positive_int": 5,
                    "bounded_int": 50,
                    "min_length_str": "abc",
                    "max_length_str": "this is way too long",
                }
            )

    def test_list_field_validation(self):
        """Test validation of list fields with specific types"""

        class ListParams(WorkflowParameters):
            str_list: list[str] = Field(default_factory=list)
            int_list: list[int] = Field(default=[1, 2, 3])

        # Valid list params
        result = ListParams.validate(
            {"str_list": ["a", "b", "c"], "int_list": [10, 20, 30]}
        )
        assert result.str_list == ["a", "b", "c"]
        assert result.int_list == [10, 20, 30]

        # Using defaults
        result2 = ListParams.validate({})
        assert result2.str_list == []
        assert result2.int_list == [1, 2, 3]

        # Invalid - wrong type in list
        with pytest.raises(ValidationError):
            ListParams.validate({"str_list": ["a", "b", 123], "int_list": [1, 2, 3]})

    def test_nested_dict_validation(self):
        """Test validation of nested dictionary fields"""

        class NestedParams(WorkflowParameters):
            config: dict[str, Any] = Field(default_factory=dict)
            metadata: dict[str, str] = Field(default_factory=dict)

        # Valid nested params
        result = NestedParams.validate(
            {
                "config": {"key1": "value1", "key2": 123},
                "metadata": {"author": "test", "version": "1.0"},
            }
        )
        assert result.config == {"key1": "value1", "key2": 123}
        assert result.metadata == {"author": "test", "version": "1.0"}

        # Using defaults
        result2 = NestedParams.validate({})
        assert result2.config == {}
        assert result2.metadata == {}

    def test_empty_parameters_validation(self):
        """Test validation with completely empty parameters"""

        class EmptyParams(WorkflowParameters):
            pass

        # Empty params should validate successfully
        result = EmptyParams.validate({})
        assert isinstance(result, EmptyParams)

        # Extra fields should still be rejected even for empty parameter class
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            EmptyParams.validate({"unexpected_field": "value"})

    def test_forecast_times_valid_iso8601(self):
        """Test that valid ISO 8601 forecast_times are accepted"""

        class ForecastParams(WorkflowParameters):
            forecast_times: list[str] = Field(default=["2024-01-01T00:00:00"])

        # Valid ISO 8601 formats
        valid_times = [
            ["2024-01-01T00:00:00"],
            ["2024-01-01T00:00:00Z"],
            ["2024-01-01T00:00:00+00:00"],
            ["2024-01-01T00:00:00.123456"],
            ["2024-12-31T23:59:59"],
            ["2024-01-01T00:00:00", "2024-01-02T00:00:00", "2024-01-03T00:00:00"],
        ]

        for times in valid_times:
            result = ForecastParams.validate({"forecast_times": times})
            assert result.forecast_times == times

    def test_forecast_times_invalid_format(self):
        """Test that invalid ISO 8601 forecast_times are rejected"""

        class ForecastParams(WorkflowParameters):
            forecast_times: list[str] = Field(default=["2024-01-01T00:00:00"])

        # Invalid formats
        invalid_times = [
            (["2024-01-01"], "Missing time component"),
            (["01/01/2024"], "Wrong date format"),
            (["not-a-date"], "Completely invalid"),
            (["2024-13-01T00:00:00"], "Invalid month"),
            (["2024-01-32T00:00:00"], "Invalid day"),
            (["2024-01-01T25:00:00"], "Invalid hour"),
        ]

        for times, description in invalid_times:
            with pytest.raises(
                ValidationError, match="is not a valid ISO 8601 datetime format"
            ):
                ForecastParams.validate({"forecast_times": times})

    def test_forecast_times_space_separator_accepted(self):
        """Test that space separator is accepted (Python's fromisoformat allows it)"""

        class ForecastParams(WorkflowParameters):
            forecast_times: list[str] = Field(default=["2024-01-01T00:00:00"])

        # Space separator is accepted by fromisoformat (though not strict ISO 8601)
        result = ForecastParams.validate({"forecast_times": ["2024-01-01 00:00:00"]})
        assert result.forecast_times == ["2024-01-01 00:00:00"]

    def test_forecast_times_mixed_valid_invalid(self):
        """Test that forecast_times with one invalid value in a list fails validation"""

        class ForecastParams(WorkflowParameters):
            forecast_times: list[str] = Field(default=["2024-01-01T00:00:00"])

        # Mix of valid and invalid - should fail on the invalid one
        with pytest.raises(
            ValidationError, match="forecast_times\\[1\\].*is not a valid ISO 8601"
        ):
            ForecastParams.validate(
                {
                    "forecast_times": [
                        "2024-01-01T00:00:00",  # Valid
                        "invalid-date",  # Invalid
                        "2024-01-03T00:00:00",  # Valid
                    ]
                }
            )

    def test_forecast_times_wrong_type(self):
        """Test that forecast_times with non-string values are rejected"""

        class ForecastParams(WorkflowParameters):
            forecast_times: list[str] = Field(default=["2024-01-01T00:00:00"])

        # Integer in the list
        with pytest.raises(
            ValidationError, match="forecast_times\\[0\\] must be a string"
        ):
            ForecastParams.validate({"forecast_times": [12345]})

        # None in the list
        with pytest.raises(
            ValidationError, match="forecast_times\\[1\\] must be a string"
        ):
            ForecastParams.validate(
                {"forecast_times": ["2024-01-01T00:00:00", None, "2024-01-03T00:00:00"]}
            )

    def test_forecast_times_empty_list(self):
        """Test that empty forecast_times list is allowed (for workflows that have defaults)"""

        class ForecastParams(WorkflowParameters):
            forecast_times: list[str] = Field(default=["2024-01-01T00:00:00"])

        # Empty list should be valid (workflow might handle it with defaults)
        result = ForecastParams.validate({"forecast_times": []})
        assert result.forecast_times == []

    def test_forecast_times_none_allowed(self):
        """Test that None forecast_times is allowed if field is Optional"""

        class ForecastParams(WorkflowParameters):
            forecast_times: list[str] | None = Field(default=None)

        # None should be valid for Optional fields
        result = ForecastParams.validate({"forecast_times": None})
        assert result.forecast_times is None

        # Can also be omitted
        result2 = ForecastParams.validate({})
        assert result2.forecast_times is None

    def test_start_time_valid_iso8601(self):
        """Test that valid ISO 8601 start_time values are accepted"""
        from datetime import datetime

        class StartTimeParams(WorkflowParameters):
            start_time: list[datetime] = Field(
                default_factory=lambda: [datetime(2024, 1, 1, 0)]
            )

        # Valid ISO 8601 string formats (Pydantic will parse to datetime)
        valid_times = [
            ["2024-01-01T00:00:00"],
            ["2024-01-01T00:00:00Z"],
            ["2024-01-01T00:00:00+00:00"],
            ["2024-01-01 00:00:00"],
            ["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
        ]

        for times in valid_times:
            result = StartTimeParams.validate({"start_time": times})
            assert result.start_time is not None
            assert len(result.start_time) == len(times)

    def test_start_time_invalid_format(self):
        """Test that invalid start_time formats are rejected"""
        from datetime import datetime

        class StartTimeParams(WorkflowParameters):
            start_time: list[datetime] = Field(
                default_factory=lambda: [datetime(2024, 1, 1, 0)]
            )

        # Invalid formats
        invalid_times = [
            (["2024-01-01"], "date without time"),
            (["not-a-date"], "completely invalid"),
            (["2024-13-01T00:00:00"], "invalid month"),
        ]

        for times, description in invalid_times:
            with pytest.raises(
                ValidationError, match="is not a valid ISO 8601 datetime format"
            ):
                StartTimeParams.validate({"start_time": times})

    def test_start_time_with_datetime_objects(self):
        """Test that start_time accepts datetime objects directly"""
        from datetime import datetime

        class StartTimeParams(WorkflowParameters):
            start_time: list[datetime] = Field(
                default_factory=lambda: [datetime(2024, 1, 1, 0)]
            )

        # Pass datetime objects directly
        dt1 = datetime(2024, 1, 1, 0)
        dt2 = datetime(2024, 1, 2, 0)

        result = StartTimeParams.validate({"start_time": [dt1, dt2]})
        assert result.start_time == [dt1, dt2]


# Test WorkflowResult
class TestWorkflowResult:
    """Test WorkflowResult model"""

    def test_create_workflow_result(self):
        """Test creating a WorkflowResult"""
        result = WorkflowResult(
            workflow_name="test_workflow",
            execution_id="exec_123",
            status=WorkflowStatus.COMPLETED,
        )
        assert result.workflow_name == "test_workflow"
        assert result.execution_id == "exec_123"
        assert result.status == WorkflowStatus.COMPLETED
        assert result.start_time is None
        assert result.end_time is None
        assert result.execution_time_seconds is None
        assert result.error_message is None
        assert result.metadata == {}

    def test_workflow_result_with_all_fields(self):
        """Test WorkflowResult with all fields populated"""
        now = datetime.now(timezone.utc).isoformat()
        result = WorkflowResult(
            workflow_name="test_workflow",
            execution_id="exec_123",
            status=WorkflowStatus.COMPLETED,
            start_time=now,
            end_time=now,
            execution_time_seconds=10.5,
            error_message=None,
            metadata={"key": "value"},
        )
        assert result.start_time == now
        assert result.end_time == now
        assert result.execution_time_seconds == 10.5
        assert result.metadata == {"key": "value"}

    def test_workflow_result_model_dump(self):
        """Test model_dump serialization"""
        result = WorkflowResult(
            workflow_name="test_workflow",
            execution_id="exec_123",
            status=WorkflowStatus.FAILED,
            error_message="Test error",
        )
        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["workflow_name"] == "test_workflow"
        assert data["execution_id"] == "exec_123"
        assert data["error_message"] == "Test error"


# Test WorkflowProgress
class TestWorkflowProgress:
    """Test WorkflowProgress model"""

    def test_create_workflow_progress(self):
        """Test creating a WorkflowProgress with all fields"""
        progress = WorkflowProgress(
            progress="Processing data...", current_step=5, total_steps=10
        )
        assert progress.progress == "Processing data..."
        assert progress.current_step == 5
        assert progress.total_steps == 10

    def test_workflow_progress_optional_fields(self):
        """Test WorkflowProgress with optional fields"""
        # Only progress message
        progress1 = WorkflowProgress(progress="Starting...")
        assert progress1.progress == "Starting..."
        assert progress1.current_step is None
        assert progress1.total_steps is None

        # Only step counters
        progress2 = WorkflowProgress(current_step=3, total_steps=10)
        assert progress2.progress is None
        assert progress2.current_step == 3
        assert progress2.total_steps == 10

    def test_workflow_progress_model_dump(self):
        """Test model_dump serialization"""
        progress = WorkflowProgress(
            progress="Processing...", current_step=2, total_steps=5
        )
        data = progress.model_dump()
        assert isinstance(data, dict)
        assert data["progress"] == "Processing..."
        assert data["current_step"] == 2
        assert data["total_steps"] == 5

    def test_workflow_progress_model_dump_exclude_none(self):
        """Test model_dump with exclude_none=True"""
        progress = WorkflowProgress(progress="Starting...")
        data = progress.model_dump(exclude_none=True)
        assert data == {"progress": "Starting..."}
        assert "current_step" not in data
        assert "total_steps" not in data

    def test_workflow_progress_with_error_message(self):
        """Test WorkflowProgress with error message"""
        progress = WorkflowProgress(
            progress="Failed!",
            error_message="Connection timeout: Could not reach data source",
        )
        assert progress.progress == "Failed!"
        assert (
            progress.error_message == "Connection timeout: Could not reach data source"
        )
        assert progress.current_step is None
        assert progress.total_steps is None

    def test_workflow_progress_error_message_serialization(self):
        """Test WorkflowProgress with error_message serialization"""
        progress = WorkflowProgress(
            progress="Failed!",
            current_step=3,
            total_steps=10,
            error_message="Network error",
        )
        data = progress.model_dump()
        assert data["progress"] == "Failed!"
        assert data["current_step"] == 3
        assert data["error_message"] == "Network error"

    def test_workflow_progress_custom_subclass(self):
        """Test creating custom WorkflowProgress subclass"""

        class CustomProgress(WorkflowProgress):
            custom_field: str = "default"
            data_processed_gb: float = 0.0

        progress = CustomProgress(
            progress="Processing batch...",
            current_step=1,
            total_steps=5,
            custom_field="custom_value",
            data_processed_gb=150.5,
        )
        assert progress.progress == "Processing batch..."
        assert progress.current_step == 1
        assert progress.custom_field == "custom_value"
        assert progress.data_processed_gb == 150.5

    def test_workflow_result_with_progress(self):
        """Test WorkflowResult with progress field"""
        progress = WorkflowProgress(
            progress="Running...", current_step=3, total_steps=10
        )
        result = WorkflowResult(
            workflow_name="test_workflow",
            execution_id="exec_123",
            status=WorkflowStatus.RUNNING,
            progress=progress,
        )
        assert result.progress is not None
        assert result.progress.progress == "Running..."
        assert result.progress.current_step == 3
        assert result.progress.total_steps == 10

    def test_workflow_result_progress_serialization(self):
        """Test WorkflowResult serialization with progress"""
        progress = WorkflowProgress(
            progress="Processing...", current_step=5, total_steps=10
        )
        result = WorkflowResult(
            workflow_name="test_workflow",
            execution_id="exec_123",
            status=WorkflowStatus.RUNNING,
            progress=progress,
        )
        data = result.model_dump()
        assert "progress" in data
        assert data["progress"]["progress"] == "Processing..."
        assert data["progress"]["current_step"] == 5
        assert data["progress"]["total_steps"] == 10


# Concrete test implementation of Workflow
class testWorkflowImpl(Workflow):
    """Test implementation of Workflow for testing purposes"""

    name = "test_workflow"

    # No __init__ needed - inherits from Workflow base class

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | WorkflowParameters
    ) -> WorkflowParameters:
        return WorkflowParameters.validate(parameters)

    def run(
        self, parameters: dict[str, Any] | WorkflowParameters, execution_id: str
    ) -> dict[str, Any]:
        return {"result": "success", "execution_id": execution_id}


# Additional test workflow classes with fixed names
class MyWorkflow(Workflow):
    """Test workflow with 'my_workflow' name"""

    name = "my_workflow"

    # No __init__ needed - name and description set by registry

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | WorkflowParameters
    ) -> WorkflowParameters:
        return WorkflowParameters.validate(parameters)

    def run(
        self, parameters: dict[str, Any] | WorkflowParameters, execution_id: str
    ) -> dict[str, Any]:
        return {"result": "success", "execution_id": execution_id}


class Workflow1(Workflow):
    """Test workflow 1"""

    name = "workflow1"
    description = "First workflow"

    # No __init__ needed - name and description set by registry

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | WorkflowParameters
    ) -> WorkflowParameters:
        return WorkflowParameters.validate(parameters)

    def run(
        self, parameters: dict[str, Any] | WorkflowParameters, execution_id: str
    ) -> dict[str, Any]:
        return {"result": "success", "execution_id": execution_id}


class Workflow2(Workflow):
    """Test workflow 2"""

    name = "workflow2"
    description = "Second workflow"

    # No __init__ needed - name and description set by registry

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | WorkflowParameters
    ) -> WorkflowParameters:
        return WorkflowParameters.validate(parameters)

    def run(
        self, parameters: dict[str, Any] | WorkflowParameters, execution_id: str
    ) -> dict[str, Any]:
        return {"result": "success", "execution_id": execution_id}


class Workflow3(Workflow):
    """Test workflow 3"""

    name = "workflow3"

    # No __init__ needed - name and description set by registry

    @classmethod
    def validate_parameters(
        cls, parameters: dict[str, Any] | WorkflowParameters
    ) -> WorkflowParameters:
        return WorkflowParameters.validate(parameters)

    def run(
        self, parameters: dict[str, Any] | WorkflowParameters, execution_id: str
    ) -> dict[str, Any]:
        return {"result": "success", "execution_id": execution_id}


# Test Workflow base class
class TestWorkflow:
    """Test Workflow base class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.workflow = testWorkflowImpl()
        # Set name manually for tests (normally set by registry)
        self.workflow.name = "test_workflow"

    def test_workflow_initialization(self):
        """Test workflow initialization"""
        workflow = testWorkflowImpl()
        # Description is empty by default
        assert workflow.description == ""
        assert workflow.redis_client is None
        assert isinstance(workflow.output_dir, Path)

        # Test that name and description can be set after initialization
        workflow.name = "my_workflow"
        workflow.description = "My test workflow"
        assert workflow.name == "my_workflow"
        assert workflow.description == "My test workflow"

    def test_set_redis_client(self):
        """Test setting Redis client"""
        mock_redis = Mock(spec=redis.Redis)
        self.workflow.set_redis_client(mock_redis)
        assert self.workflow.redis_client == mock_redis

    def test_validate_parameters_abstract(self):
        """Test that abstract methods must be implemented"""

        class IncompleteWorkflow(Workflow):
            pass

        with pytest.raises(TypeError):
            IncompleteWorkflow()

    def test_validate_parameters_must_be_implemented(self):
        """Test that validate_parameters must be implemented by subclasses"""

        class NoValidateWorkflow(Workflow):
            def run(self, parameters, execution_id):
                return {"result": "success"}

        with pytest.raises(TypeError):
            NoValidateWorkflow()

    def test_validate_parameters_works(self):
        """Test that validate_parameters works correctly"""
        result = self.workflow.validate_parameters({})
        assert isinstance(result, WorkflowParameters)

    def test_private_get_execution_data_without_redis(self):
        """Test _get_execution_data classmethod works with redis client passed in"""
        # This test is no longer relevant since _get_execution_data is a classmethod
        # that takes redis_client as a parameter, not an instance method
        # We'll test the classmethod directly
        mock_redis = Mock(spec=redis.Redis)
        execution_data = {
            "workflow_name": "test_workflow",
            "execution_id": "exec_123",
            "status": "completed",
        }
        mock_redis.get.return_value = json.dumps(execution_data).encode()

        result = testWorkflowImpl._get_execution_data(
            mock_redis, "test_workflow", "exec_123"
        )
        assert isinstance(result, WorkflowResult)
        assert result.workflow_name == "test_workflow"

    def test_private_get_execution_data_with_redis(self):
        """Test _get_execution_data classmethod with Redis client"""
        mock_redis = Mock(spec=redis.Redis)
        execution_data = {
            "workflow_name": "test_workflow",
            "execution_id": "exec_123",
            "status": "completed",
        }
        mock_redis.get.return_value = json.dumps(execution_data).encode()

        result = testWorkflowImpl._get_execution_data(
            mock_redis, "test_workflow", "exec_123"
        )

        assert isinstance(result, WorkflowResult)
        assert result.workflow_name == "test_workflow"
        assert result.execution_id == "exec_123"
        mock_redis.get.assert_called_once_with(
            "workflow_execution:test_workflow:exec_123"
        )

    def test_private_get_execution_data_not_found(self):
        """Test _get_execution_data when execution is not found"""
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.get.return_value = None

        with pytest.raises(ValueError, match="not found"):
            testWorkflowImpl._get_execution_data(
                mock_redis, "test_workflow", "exec_123"
            )

    def test_private_get_execution_data_redis_error(self):
        """Test _get_execution_data when Redis raises an error"""
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.get.side_effect = Exception("Redis connection error")

        result = testWorkflowImpl._get_execution_data(
            mock_redis, "test_workflow", "exec_123"
        )

        assert result.status == WorkflowStatus.FAILED
        assert "Redis connection error" in result.error_message

    def test_private_save_execution_data(self):
        """Test saving execution data in Redis using classmethod"""
        mock_redis = Mock(spec=redis.Redis)

        result = WorkflowResult(
            workflow_name="test_workflow",
            execution_id="exec_123",
            status=WorkflowStatus.COMPLETED,
        )

        testWorkflowImpl._save_execution_data(
            mock_redis, "test_workflow", "exec_123", result
        )

        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args
        config = get_test_config()
        assert args[0][0] == "workflow_execution:test_workflow:exec_123"
        assert args[0][1] == config.redis.retention_ttl
        data = json.loads(args[0][2])
        assert data["status"] == "completed"

    def test_private_save_execution_data_without_redis(self):
        """Test _save_execution_data classmethod with None redis raises error"""
        result = WorkflowResult(
            workflow_name="test_workflow",
            execution_id="exec_123",
            status=WorkflowStatus.COMPLETED,
        )
        # Since it's now a classmethod, we test that it raises an error if redis operations fail
        mock_redis = Mock(spec=redis.Redis)
        mock_redis.setex.side_effect = Exception("Redis error")

        with pytest.raises(Exception, match="Redis error"):
            testWorkflowImpl._save_execution_data(
                mock_redis, "test_workflow", "exec_123", result
            )

    def test_update_execution_data(self):
        """Test updating execution data"""
        mock_redis = Mock(spec=redis.Redis)
        current_data = {
            "workflow_name": "test_workflow",
            "execution_id": "exec_123",
            "status": "running",
        }
        mock_redis.get.return_value = json.dumps(current_data).encode()

        self.workflow.set_redis_client(mock_redis)
        self.workflow.update_execution_data(
            "exec_123", {"result": "success", "metadata": {"key": "value"}}
        )

        mock_redis.setex.assert_called_once()
        args = mock_redis.setex.call_args
        data = json.loads(args[0][2])
        assert data["status"] == "running"  # Status should be preserved
        assert data["result"] == "success"
        assert data["metadata"]["key"] == "value"

    def test_update_execution_data_prevents_status_override(self):
        """Test that update_execution_data prevents status from being overwritten"""
        mock_redis = Mock(spec=redis.Redis)
        current_data = {
            "workflow_name": "test_workflow",
            "execution_id": "exec_123",
            "status": "running",
        }
        mock_redis.get.return_value = json.dumps(current_data).encode()

        self.workflow.set_redis_client(mock_redis)
        # Try to update status - it should be prevented
        self.workflow.update_execution_data(
            "exec_123", {"status": "completed", "result": "success"}
        )

        args = mock_redis.setex.call_args
        data = json.loads(args[0][2])
        # Status should remain "running" (not changed to "completed")
        assert data["status"] == "running"
        assert data["result"] == "success"

    def test_update_execution_data_with_workflow_progress(self):
        """Test updating execution data with WorkflowProgress object"""
        mock_redis = Mock(spec=redis.Redis)
        current_data = {
            "workflow_name": "test_workflow",
            "execution_id": "exec_123",
            "status": "running",
        }
        mock_redis.get.return_value = json.dumps(current_data).encode()

        self.workflow.set_redis_client(mock_redis)

        # Update with WorkflowProgress object
        progress = WorkflowProgress(
            progress="Processing...", current_step=5, total_steps=10
        )
        self.workflow.update_execution_data("exec_123", progress)

        # Verify Redis was updated with progress field
        args = mock_redis.setex.call_args
        data = json.loads(args[0][2])
        assert "progress" in data
        assert data["progress"]["progress"] == "Processing..."
        assert data["progress"]["current_step"] == 5
        assert data["progress"]["total_steps"] == 10

    def test_update_execution_data_progress_merging(self):
        """Test that WorkflowProgress updates merge with existing progress"""
        mock_redis = Mock(spec=redis.Redis)

        # Initial state with existing progress
        current_data = {
            "workflow_name": "test_workflow",
            "execution_id": "exec_123",
            "status": "running",
            "progress": {"progress": "Step 1", "current_step": 1, "total_steps": 10},
        }
        mock_redis.get.return_value = json.dumps(current_data).encode()

        self.workflow.set_redis_client(mock_redis)

        # Update only the progress message - other fields should be preserved
        progress = WorkflowProgress(progress="Step 2")
        self.workflow.update_execution_data("exec_123", progress)

        args = mock_redis.setex.call_args
        data = json.loads(args[0][2])

        # Progress message should be updated
        assert data["progress"]["progress"] == "Step 2"
        # But current_step and total_steps should be preserved
        assert data["progress"]["current_step"] == 1
        assert data["progress"]["total_steps"] == 10

    def test_update_execution_data_progress_partial_update(self):
        """Test that partial WorkflowProgress updates merge correctly"""
        mock_redis = Mock(spec=redis.Redis)

        # Initial state
        current_data = {
            "workflow_name": "test_workflow",
            "execution_id": "exec_123",
            "status": "running",
            "progress": {
                "progress": "Processing...",
                "current_step": 5,
                "total_steps": 10,
            },
        }
        mock_redis.get.return_value = json.dumps(current_data).encode()

        self.workflow.set_redis_client(mock_redis)

        # Update only current_step
        progress = WorkflowProgress(current_step=6)
        self.workflow.update_execution_data("exec_123", progress)

        args = mock_redis.setex.call_args
        data = json.loads(args[0][2])

        # current_step should be updated
        assert data["progress"]["current_step"] == 6
        # Other fields should be preserved
        assert data["progress"]["progress"] == "Processing..."
        assert data["progress"]["total_steps"] == 10

    def test_update_execution_data_progress_override_fields(self):
        """Test that WorkflowProgress updates can override existing fields"""
        mock_redis = Mock(spec=redis.Redis)

        # Initial state
        current_data = {
            "workflow_name": "test_workflow",
            "execution_id": "exec_123",
            "status": "running",
            "progress": {
                "progress": "Old message",
                "current_step": 1,
                "total_steps": 10,
            },
        }
        mock_redis.get.return_value = json.dumps(current_data).encode()

        self.workflow.set_redis_client(mock_redis)

        # Update all fields
        progress = WorkflowProgress(
            progress="New message", current_step=5, total_steps=20
        )
        self.workflow.update_execution_data("exec_123", progress)

        args = mock_redis.setex.call_args
        data = json.loads(args[0][2])

        # All fields should be updated
        assert data["progress"]["progress"] == "New message"
        assert data["progress"]["current_step"] == 5
        assert data["progress"]["total_steps"] == 20

    def test_update_execution_data_progress_with_no_existing_progress(self):
        """Test updating progress when no progress exists yet"""
        mock_redis = Mock(spec=redis.Redis)

        # Initial state with no progress
        current_data = {
            "workflow_name": "test_workflow",
            "execution_id": "exec_123",
            "status": "running",
        }
        mock_redis.get.return_value = json.dumps(current_data).encode()

        self.workflow.set_redis_client(mock_redis)

        # Set initial progress
        progress = WorkflowProgress(
            progress="Starting...", current_step=0, total_steps=10
        )
        self.workflow.update_execution_data("exec_123", progress)

        args = mock_redis.setex.call_args
        data = json.loads(args[0][2])

        # Progress should be set
        assert "progress" in data
        assert data["progress"]["progress"] == "Starting..."
        assert data["progress"]["current_step"] == 0
        assert data["progress"]["total_steps"] == 10

    def test_update_execution_data_progress_with_error_message(self):
        """Test updating progress with error_message field"""
        mock_redis = Mock(spec=redis.Redis)

        # Initial state
        current_data = {
            "workflow_name": "test_workflow",
            "execution_id": "exec_123",
            "status": "running",
            "progress": {
                "progress": "Processing...",
                "current_step": 5,
                "total_steps": 10,
            },
        }
        mock_redis.get.return_value = json.dumps(current_data).encode()

        self.workflow.set_redis_client(mock_redis)

        # Update with error
        progress = WorkflowProgress(
            progress="Failed!",
            error_message="Connection timeout: Could not reach data source",
        )
        self.workflow.update_execution_data("exec_123", progress)

        args = mock_redis.setex.call_args
        data = json.loads(args[0][2])

        # Progress should be updated with error message
        assert data["progress"]["progress"] == "Failed!"
        assert (
            data["progress"]["error_message"]
            == "Connection timeout: Could not reach data source"
        )
        # Step counters should be preserved from previous state
        assert data["progress"]["current_step"] == 5
        assert data["progress"]["total_steps"] == 10

    def test_get_output_path(self):
        """Test getting output directory path"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.workflow.output_dir = Path(tmpdir)
            path = self.workflow.get_output_path("exec_123")

            assert isinstance(path, Path)
            assert path.is_dir()
            assert "test_workflow" in str(path)
            assert "exec_123" in str(path)
            assert path.exists()

    def test_get_output_path_creates_directory(self):
        """Test that get_output_path creates directory if it doesn't exist"""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.workflow.output_dir = Path(tmpdir)
            path = self.workflow.get_output_path("exec_456")

            # Verify directory was created
            assert path.exists()
            assert path.is_dir()
            # Verify correct path structure
            assert path == Path(tmpdir) / "test_workflow" / "exec_456"


# Test WorkflowRegistry
class TestWorkflowRegistry:
    """Test WorkflowRegistry class"""

    def setup_method(self):
        """Set up test fixtures"""
        self.registry = WorkflowRegistry()

    def test_registry_initialization(self):
        """Test registry initialization"""
        assert len(self.registry._workflows) == 0

    def test_register_workflow(self):
        """Test registering a workflow"""
        self.registry.register(MyWorkflow)

        assert "my_workflow" in self.registry._workflows
        assert self.registry._workflows["my_workflow"] == MyWorkflow

    def test_register_duplicate_workflow(self):
        """Test registering duplicate workflow raises error"""
        self.registry.register(MyWorkflow)
        with pytest.raises(ValueError, match="already registered"):
            self.registry.register(MyWorkflow)

    def test_register_reserved_workflow_name(self):
        """Test that registering a workflow with reserved name 'workflows' raises error"""

        class WorkflowsNamedWorkflow(Workflow):
            """Test workflow with reserved name"""

            name = "workflows"
            description = "Test workflow with reserved name"

            def run(self, parameters: WorkflowParameters) -> WorkflowResult:
                return WorkflowResult(
                    status=WorkflowStatus.COMPLETED,
                    message="Test completed",
                    data={"test": "data"},
                )

        with pytest.raises(ValueError, match="Workflow name 'workflows' is reserved"):
            self.registry.register(WorkflowsNamedWorkflow)

    def test_register_with_instance_raises_error(self):
        """Test that registering an instance (not a class) raises TypeError"""
        workflow_instance = MyWorkflow()
        with pytest.raises(
            TypeError, match="Expected a Workflow class.*got an instance"
        ):
            self.registry.register(workflow_instance)

    def test_register_with_non_workflow_class_raises_error(self):
        """Test that registering a non-Workflow class raises TypeError"""

        class NotAWorkflow:
            pass

        with pytest.raises(TypeError, match="must be a subclass of Workflow"):
            self.registry.register(NotAWorkflow)

    def test_register_with_custom_name(self):
        """Test registering a workflow with a custom name"""

        class MyCustomWorkflow(MyWorkflow):
            name = "custom_workflow"

        self.registry.register(MyCustomWorkflow)

        assert "custom_workflow" in self.registry._workflows
        assert self.registry._workflows["custom_workflow"] == MyCustomWorkflow
        assert self.registry._workflows["custom_workflow"].name == "custom_workflow"

    def test_register_with_custom_description(self):
        """Test registering a workflow with a custom description"""

        class MyCustomWorkflow(MyWorkflow):
            name = "my_custom_workflow"
            description = "This is a custom description"

        self.registry.register(MyCustomWorkflow)

        assert "my_custom_workflow" in self.registry._workflows
        assert (
            self.registry._workflows["my_custom_workflow"].description
            == "This is a custom description"
        )

    def test_register_defaults_to_class_name(self):
        """Test that workflow_name defaults to class name when not provided"""
        self.registry.register(MyWorkflow)

        # Should use the class name "MyWorkflow" as fallback if instance name not accessible
        assert (
            "my_workflow" in self.registry._workflows
            or "MyWorkflow" in self.registry._workflows
        )

    def test_unregister_workflow(self):
        """Test unregistering a workflow"""
        self.registry.register(MyWorkflow)
        self.registry.unregister("my_workflow")

        assert "my_workflow" not in self.registry._workflows

    def test_unregister_nonexistent_workflow(self):
        """Test unregistering nonexistent workflow raises error"""
        with pytest.raises(ValueError, match="not registered"):
            self.registry.unregister("nonexistent")

    def test_get_workflow(self):
        """Test getting a workflow by name"""
        self.registry.register(MyWorkflow)

        retrieved = self.registry.get("my_workflow")
        assert retrieved is not None
        assert isinstance(retrieved, MyWorkflow)
        assert retrieved.name == "my_workflow"

        # Get should return the same cached instance
        retrieved2 = self.registry.get("my_workflow")
        assert retrieved2 is not None
        assert isinstance(retrieved2, MyWorkflow)
        assert retrieved is retrieved2  # Same cached instance

    def test_get_nonexistent_workflow(self):
        """Test getting nonexistent workflow returns None"""
        result = self.registry.get("nonexistent")
        assert result is None

    def test_get_workflow_with_custom_name_and_description(self):
        """Test that workflow instances get the registered name and description"""

        class MyCustomWorkflow(MyWorkflow):
            name = "custom_name"
            description = "Custom description for testing"

        self.registry.register(MyCustomWorkflow)

        retrieved = self.registry.get("custom_name")
        assert retrieved is not None
        assert isinstance(retrieved, MyWorkflow)
        # Verify the instance has the custom registered name and description
        assert retrieved.name == "custom_name"
        assert retrieved.description == "Custom description for testing"

    def test_workflow_instance_caching(self):
        """Test that workflow instances are cached"""
        self.registry.register(MyWorkflow)

        # First get creates and caches instance
        instance1 = self.registry.get("my_workflow")
        assert instance1 is not None

        # Verify the instance is in the cache
        assert "my_workflow" in self.registry._workflow_instances
        assert self.registry._workflow_instances["my_workflow"] is instance1

        # Second get returns the same cached instance
        instance2 = self.registry.get("my_workflow")
        assert instance2 is instance1

    def test_unregister_clears_cache(self):
        """Test that unregistering a workflow clears its cached instance"""
        self.registry.register(MyWorkflow)

        # Get instance to populate cache
        instance = self.registry.get("my_workflow")
        assert instance is not None
        assert "my_workflow" in self.registry._workflow_instances

        # Unregister should clear the cache
        self.registry.unregister("my_workflow")
        assert "my_workflow" not in self.registry._workflow_instances

    def test_list_workflows(self):
        """Test listing all workflows"""
        self.registry.register(Workflow1)
        self.registry.register(Workflow2)

        workflows = self.registry.list_workflows()
        assert len(workflows) == 2
        assert workflows["workflow1"] == "First workflow"
        assert workflows["workflow2"] == "Second workflow"

    def test_list_workflows_empty(self):
        """Test listing workflows when registry is empty"""
        workflows = self.registry.list_workflows()
        assert len(workflows) == 0
        assert isinstance(workflows, dict)

    def test_set_redis_client(self):
        """Test setting Redis client for workflow instances"""
        self.registry.register(Workflow1)
        self.registry.register(Workflow2)

        mock_redis = Mock(spec=redis.Redis)

        # Verify that instances created via get() have redis client set
        instance1 = self.registry.get("workflow1", redis_client=mock_redis)
        instance2 = self.registry.get("workflow2", redis_client=mock_redis)

        assert instance1.redis_client == mock_redis
        assert instance2.redis_client == mock_redis

    def test_discover_and_register_from_directories_nonexistent(self):
        """Test discovering workflows from nonexistent directory"""
        successful, failed = self.registry.discover_and_register_from_directories(
            ["/nonexistent/path"], include_builtin=False
        )
        assert successful == 0
        assert failed == 0

    def test_discover_and_register_from_directories_not_a_dir(self):
        """Test discovering workflows from file instead of directory"""
        with tempfile.NamedTemporaryFile(suffix=".py") as tmpfile:
            successful, failed = self.registry.discover_and_register_from_directories(
                [tmpfile.name], include_builtin=False
            )
            assert successful == 0
            assert failed == 0

    def test_discover_and_register_from_directories_empty(self):
        """Test discovering workflows from empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            successful, failed = self.registry.discover_and_register_from_directories(
                [tmpdir], include_builtin=False
            )
            assert successful == 0
            assert failed == 0

    def test_discover_and_register_from_directories_with_python_file(self):
        """Test discovering workflows from directory with Python file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple workflow file
            workflow_file = Path(tmpdir) / "test_workflow_module.py"
            workflow_file.write_text(
                """
from api_server.workflow import Workflow, WorkflowParameters, workflow_registry

class SimpleWorkflow(Workflow):
    # No __init__ needed - name and description set by registry

    name = "simple"
    description = "Simple test workflow"

    @classmethod
    def validate_parameters(cls, parameters):
        return WorkflowParameters.validate(parameters)

    def run(self, parameters, execution_id):
        return {"result": "success"}

workflow_registry.register(SimpleWorkflow)
"""
            )

            successful, failed = self.registry.discover_and_register_from_directories(
                [tmpdir], include_builtin=False
            )
            from api_server.workflow import workflow_registry

            assert successful == 1
            assert failed == 0
            assert "simple" in workflow_registry.list_workflows().keys()

    def test_discover_and_register_with_invalid_python_file(self):
        """Test discovering workflows from directory with invalid Python file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an invalid Python file
            workflow_file = Path(tmpdir) / "invalid_workflow.py"
            workflow_file.write_text("this is not valid python syntax ][{")

            successful, failed = self.registry.discover_and_register_from_directories(
                [tmpdir], include_builtin=False
            )

            assert successful == 0
            assert failed == 1

    def test_discover_and_register_filters_init_files(self):
        """Test that __init__.py files are filtered out"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create __init__.py
            init_file = Path(tmpdir) / "__init__.py"
            init_file.write_text("# init file")

            successful, failed = self.registry.discover_and_register_from_directories(
                [tmpdir], include_builtin=False
            )

            assert successful == 0
            assert failed == 0

    @patch("api_server.workflow.workflow.Path")
    def test_discover_and_register_with_builtin(self, mock_path):
        """Test discovering workflows with built-in workflows"""
        # Mock the builtin workflows directory to not exist
        mock_builtin_dir = MagicMock()
        mock_builtin_dir.exists.return_value = False
        mock_path.return_value.parent.__truediv__.return_value = mock_builtin_dir

        with tempfile.TemporaryDirectory() as tmpdir:
            successful, failed = self.registry.discover_and_register_from_directories(
                [tmpdir], include_builtin=True
            )

            # Should attempt to check for builtin directory
            assert successful == 0
            assert failed == 0

    def test_auto_register_workflows(self):
        """Test auto-registering workflows"""
        mock_redis = Mock(spec=redis.Redis)

        with (
            patch(
                "api_server.workflow.workflow.parse_workflow_directories_from_env",
                return_value=[],
            ),
            patch.object(
                self.registry,
                "discover_and_register_from_directories",
                return_value=(0, 0),
            ),
        ):
            self.registry.auto_register_workflows(mock_redis)

            # Should set Redis client for all workflows
            for workflow in self.registry._workflows.values():
                assert workflow.redis_client == mock_redis

    def test_auto_register_workflows_with_error(self):
        """Test auto-registering workflows with error"""
        mock_redis = Mock(spec=redis.Redis)

        with patch(
            "api_server.workflow.workflow.parse_workflow_directories_from_env",
            side_effect=Exception("Test error"),
        ):
            with pytest.raises(Exception, match="Test error"):
                self.registry.auto_register_workflows(mock_redis)


# Test helper functions
class TestHelperFunctions:
    """Test helper functions"""

    def test_parse_workflow_directories_from_env_not_set(self):
        """Test parsing when environment variable not set"""
        with patch.dict(os.environ, {}, clear=True):
            dirs = parse_workflow_directories_from_env()
            assert dirs == []

    def test_parse_workflow_directories_from_env_single(self):
        """Test parsing single directory"""
        with patch.dict(os.environ, {"WORKFLOW_DIR": "/path/to/workflows"}):
            dirs = parse_workflow_directories_from_env()
            assert dirs == ["/path/to/workflows"]

    def test_parse_workflow_directories_from_env_comma_separated(self):
        """Test parsing comma-separated directories"""
        with patch.dict(
            os.environ, {"WORKFLOW_DIR": "/path/one,/path/two,/path/three"}
        ):
            dirs = parse_workflow_directories_from_env()
            assert dirs == ["/path/one", "/path/two", "/path/three"]

    def test_parse_workflow_directories_from_env_colon_separated(self):
        """Test parsing colon-separated directories"""
        with patch.dict(
            os.environ, {"WORKFLOW_DIR": "/path/one:/path/two:/path/three"}
        ):
            dirs = parse_workflow_directories_from_env()
            assert dirs == ["/path/one", "/path/two", "/path/three"]

    def test_parse_workflow_directories_from_env_with_whitespace(self):
        """Test parsing with whitespace"""
        with patch.dict(
            os.environ, {"WORKFLOW_DIR": " /path/one , /path/two , /path/three "}
        ):
            dirs = parse_workflow_directories_from_env()
            assert dirs == ["/path/one", "/path/two", "/path/three"]

    def test_parse_workflow_directories_from_env_removes_duplicates(self):
        """Test that duplicates are removed"""
        with patch.dict(os.environ, {"WORKFLOW_DIR": "/path/one,/path/two,/path/one"}):
            dirs = parse_workflow_directories_from_env()
            assert dirs == ["/path/one", "/path/two"]

    def test_register_all_workflows(self):
        """Test register_all_workflows convenience function"""
        mock_redis = Mock(spec=redis.Redis)

        with patch.object(workflow_registry, "auto_register_workflows") as mock_auto:
            register_all_workflows(mock_redis)
            mock_auto.assert_called_once_with(mock_redis)


# Integration tests
class TestWorkflowIntegration:
    """Integration tests for Workflow and WorkflowRegistry"""

    def test_workflow_lifecycle(self):
        """Test complete workflow lifecycle"""
        # Create registry and register workflow class
        registry = WorkflowRegistry()
        registry.register(testWorkflowImpl)

        # Get workflow instance with redis client
        mock_redis = Mock(spec=redis.Redis)
        workflow = registry.get("test_workflow", redis_client=mock_redis)

        # Verify workflow instance has Redis client
        assert workflow.redis_client == mock_redis

        # Test workflow class method for validation
        params = testWorkflowImpl.validate_parameters({})
        assert isinstance(params, WorkflowParameters)

        # Test workflow instance run method
        result = workflow.run(params, "exec_123")
        assert result["result"] == "success"
        assert result["execution_id"] == "exec_123"

    def test_multiple_workflows_in_registry(self):
        """Test multiple workflows in registry"""
        registry = WorkflowRegistry()
        registry.register(Workflow1)
        registry.register(Workflow2)
        registry.register(Workflow3)

        workflows = registry.list_workflows()
        assert len(workflows) == 3

        # Test retrieval - get() returns cached instances
        instance1 = registry.get("workflow1")
        instance2 = registry.get("workflow2")
        instance3 = registry.get("workflow3")

        assert isinstance(instance1, Workflow1)
        assert isinstance(instance2, Workflow2)
        assert isinstance(instance3, Workflow3)

        # Test unregistering
        registry.unregister("workflow2")
        workflows = registry.list_workflows()
        assert len(workflows) == 2
        assert "workflow2" not in workflows
