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
Tests for Earth2Workflow and AutoParameters functionality
"""

import os
from datetime import datetime
from typing import Literal
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

# Set environment variable before importing workflows
os.environ["EARTH2STUDIO_API_ACTIVE"] = "1"

# Try to import earth2studio, skip tests if not available
try:
    from earth2studio.io import IOBackend

    EARTH2STUDIO_AVAILABLE = True
except ImportError:
    EARTH2STUDIO_AVAILABLE = False
    IOBackend = Mock  # Mock for type hints

# ruff: noqa: E402
from api_server.workflow.e2workflow import Earth2Workflow, func_to_model

# Skip all tests in this module if earth2studio is not available
pytestmark = pytest.mark.skipif(
    not EARTH2STUDIO_AVAILABLE, reason="earth2studio not available in test environment"
)


class TestAutoParametersValidation:
    """Test validation of auto-generated parameters from Earth2Workflow"""

    def test_auto_parameters_with_num_steps_validation(self):
        """Test that num_steps parameter gets automatic range validation"""

        # Create a mock Earth2Workflow with num_steps parameter
        class TestWorkflow(Earth2Workflow):
            def __init__(self, model_type: Literal["fcn", "dlwp"] = "fcn"):
                pass

            def __call__(
                self,
                io: IOBackend,
                start_time: list[datetime] = [datetime(2024, 1, 1, 0)],
                num_steps: int = 20,
            ):
                pass

        # Test valid num_steps
        valid_params = {
            "start_time": ["2024-01-01T00:00:00"],
            "num_steps": 50,
        }
        result = TestWorkflow.validate_parameters(valid_params)
        assert result.num_steps == 50

        # Test num_steps too low (below 1)
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            TestWorkflow.validate_parameters(
                {
                    "start_time": ["2024-01-01T00:00:00"],
                    "num_steps": 0,
                }
            )

        # Test num_steps too high (above 1000)
        with pytest.raises(ValidationError, match="less than or equal to 1000"):
            TestWorkflow.validate_parameters(
                {
                    "start_time": ["2024-01-01T00:00:00"],
                    "num_steps": 1001,
                }
            )

    def test_auto_parameters_with_start_time_validation(self):
        """Test that start_time parameter gets ISO 8601 validation"""

        class TestWorkflow(Earth2Workflow):
            def __init__(self):
                pass

            def __call__(
                self,
                io: IOBackend,
                start_time: list[datetime] = [datetime(2024, 1, 1, 0)],
            ):
                pass

        # Test valid ISO 8601 formats
        valid_times = [
            ["2024-01-01T00:00:00"],
            ["2024-01-01T00:00:00Z"],
            ["2024-01-01T00:00:00+00:00"],
            ["2024-01-01 00:00:00"],
        ]

        for times in valid_times:
            result = TestWorkflow.validate_parameters({"start_time": times})
            assert result.start_time is not None

        # Test invalid formats
        invalid_times = [
            ["2024-01-01"],  # Missing time component
            ["not-a-date"],  # Completely invalid
            ["2024-13-01T00:00:00"],  # Invalid month
        ]

        for times in invalid_times:
            with pytest.raises(
                ValidationError, match="is not a valid ISO 8601 datetime format"
            ):
                TestWorkflow.validate_parameters({"start_time": times})

    def test_auto_parameters_with_config_validation(self):
        """Test that __init__ parameters get proper validation"""

        class TestWorkflow(Earth2Workflow):
            def __init__(self, model_type: Literal["fcn", "dlwp"] = "fcn"):
                self.model_type = model_type

            def __call__(self, io: IOBackend):
                pass

        # Valid model_type
        assert (
            TestWorkflow.Config.model_validate({"model_type": "fcn"}).model_type
            == "fcn"
        )
        assert (
            TestWorkflow.Config.model_validate({"model_type": "dlwp"}).model_type
            == "dlwp"
        )

        # Invalid model_type
        with pytest.raises(ValidationError):
            TestWorkflow.Config.model_validate({"model_type": "invalid"})

    def test_auto_parameters_nsteps_variant(self):
        """Test that nsteps (alternate spelling) also gets range validation"""

        class TestWorkflow(Earth2Workflow):
            def __init__(self):
                pass

            def __call__(self, io: IOBackend, nsteps: int = 10):
                pass

        # Valid nsteps
        result = TestWorkflow.validate_parameters({"nsteps": 100})
        assert result.nsteps == 100

        # Invalid nsteps
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            TestWorkflow.validate_parameters({"nsteps": 0})

    def test_func_to_model_with_validation(self):
        """Test func_to_model creates models with proper validation"""

        def test_func(
            start_time: list[datetime] = [datetime(2024, 1, 1, 0)],
            num_steps: int = 20,
        ) -> None:
            pass

        # Create model from function
        TestModel = func_to_model(test_func, model_name="TestModel")

        # Test valid parameters
        result = TestModel.model_validate(
            {"start_time": ["2024-01-01T00:00:00"], "num_steps": 50}
        )
        assert result.num_steps == 50

        # Test num_steps validation
        with pytest.raises(ValidationError):
            TestModel.model_validate({"num_steps": 0})

        # Test start_time validation
        with pytest.raises(ValidationError, match="is not a valid ISO 8601"):
            TestModel.model_validate({"start_time": ["2024-01-01"]})  # Missing time

    def test_auto_parameters_with_defaults(self):
        """Test that default values work correctly with validation"""

        class TestWorkflow(Earth2Workflow):
            def __init__(self):
                pass

            def __call__(
                self,
                io: IOBackend,
                start_time: list[datetime] = [datetime(2024, 1, 1, 0)],
                num_steps: int = 20,
            ):
                pass

        # Use defaults
        result = TestWorkflow.validate_parameters({})
        assert result.num_steps == 20
        assert len(result.start_time) == 1

        # Override defaults
        result2 = TestWorkflow.validate_parameters(
            {"start_time": ["2024-02-01T00:00:00"], "num_steps": 100}
        )
        assert result2.num_steps == 100


class TestEarth2WorkflowIntegration:
    """Integration tests for Earth2Workflow validation"""

    def test_complete_workflow_validation(self):
        """Test complete validation workflow"""

        class CompleteWorkflow(Earth2Workflow):
            def __init__(
                self,
                model_type: Literal["fcn", "dlwp"] = "fcn",
                device: Literal["cpu", "cuda"] = "cpu",
            ):
                self.model_type = model_type
                self.device = device

            def __call__(
                self,
                io: IOBackend,
                start_time: list[datetime] = [datetime(2024, 1, 1, 0)],
                num_steps: int = 20,
                output_frequency: int = 1,
            ):
                pass

        # Test valid complete parameters
        params = {
            "start_time": ["2024-01-01T00:00:00", "2024-01-02T00:00:00"],
            "num_steps": 50,
            "output_frequency": 5,
        }
        result = CompleteWorkflow.validate_parameters(params)
        assert result.num_steps == 50
        assert len(result.start_time) == 2

        # Test valid config
        config = {"model_type": "dlwp", "device": "cuda"}
        result_config = CompleteWorkflow.Config.model_validate(config)
        assert result_config.model_type == "dlwp"
        assert result_config.device == "cuda"

        # Test invalid config
        with pytest.raises(ValidationError):
            CompleteWorkflow.Config.model_validate({"model_type": "invalid"})

        # Test invalid parameters
        with pytest.raises(ValidationError):
            CompleteWorkflow.validate_parameters({"num_steps": -1})

    def test_workflow_with_extra_fields_rejected(self):
        """Test that extra fields are rejected due to strict validation"""

        class TestWorkflow(Earth2Workflow):
            def __init__(self):
                pass

            def __call__(self, io: IOBackend, num_steps: int = 20):
                pass

        # Extra field should be rejected
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            TestWorkflow.validate_parameters({"num_steps": 20, "extra_field": "value"})
