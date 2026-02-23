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

from unittest.mock import MagicMock, Mock, patch

import pytest
import redis
import torch
from api_server.workflow import (
    Earth2Workflow,
    WorkflowParameters,
)
from api_server.workflow.e2workflow import (
    AutoParameters,
    BackendProgress,
    func_to_model,
)

from earth2studio.io import IOBackend
from earth2studio.utils.type import CoordSystem


# Test func_to_model function
class TestFuncToModel:
    """Test func_to_model helper function"""

    def test_function_to_model_conversion(self):
        """Test converting function signature to Pydantic model"""

        def test_func(required: int, optional: str = "default", exclude_me: int = 0):
            pass

        model = func_to_model(test_func, exclude_params={"exclude_me"})

        # Verify model creation and field handling
        assert issubclass(model, WorkflowParameters)
        instance = model(required=5)
        assert instance.required == 5
        assert instance.optional == "default"
        assert not hasattr(instance, "exclude_me")


# Test AutoParameters metaclass
class TestAutoParameters:
    """Test AutoParameters metaclass"""

    def test_auto_generates_parameters_from_call_signature(self):
        """Test that AutoParameters creates Parameters from __call__ signature"""

        class TestWorkflow(metaclass=AutoParameters):
            def __call__(self, io: IOBackend, x: int, y: str = "default"):
                pass

        # Verify Parameters class is created with correct name
        assert hasattr(TestWorkflow, "Parameters")
        assert TestWorkflow.Parameters.__name__ == "TestWorkflowParameters"
        assert issubclass(TestWorkflow.Parameters, WorkflowParameters)

        # Verify io parameter is excluded and other params are included
        params = TestWorkflow.Parameters(x=10)
        assert params.x == 10
        assert params.y == "default"
        assert not hasattr(params, "io")


# Concrete test implementations of Earth2Workflow
class TestEarth2WorkflowImpl(Earth2Workflow):
    """Simple test implementation of Earth2Workflow"""

    def __call__(self, io: IOBackend, x: int = 10, y: str = "default"):
        """Test implementation with simple parameters"""
        # Store call info for testing
        self.called_with = {"io": io, "x": x, "y": y}


class ComplexEarth2WorkflowImpl(Earth2Workflow):
    """Test implementation with more complex parameters"""

    def __call__(
        self,
        io: IOBackend,
        required_param: int,
        optional_param: str = "default",
        num_hours: int = 24,
    ):
        """Test implementation with required and optional parameters"""
        self.called_with = {
            "io": io,
            "required_param": required_param,
            "optional_param": optional_param,
            "num_hours": num_hours,
        }


# Test Earth2Workflow base class
class TestEarth2Workflow:
    """Test Earth2Workflow-specific functionality (base Workflow tests in test_workflow.py)"""

    def setup_method(self):
        """Set up test fixtures"""
        self.workflow = TestEarth2WorkflowImpl()
        self.workflow.name = "test_workflow"

    def test_validate_parameters_uses_auto_generated_parameters(self):
        """Test that validate_parameters works with auto-generated Parameters class"""
        params = TestEarth2WorkflowImpl.validate_parameters({"x": 30, "y": "test"})
        assert isinstance(params, WorkflowParameters)
        assert params.x == 30
        assert params.y == "test"

        # Test with missing required parameter
        with pytest.raises(ValueError):
            ComplexEarth2WorkflowImpl.validate_parameters({"optional_param": "test"})

    @patch("api_server.e2workflow.get_config")
    def test_run_method_with_io_backend_and_progress_updates(
        self, mock_get_config, tmp_path
    ):
        """Test Earth2Workflow.run creates IOBackend and updates progress"""
        mock_config = MagicMock()
        mock_config.paths.output_format = "zarr"
        mock_get_config.return_value = mock_config

        mock_redis = Mock(spec=redis.Redis)
        self.workflow.set_redis_client(mock_redis)

        with (
            patch.object(self.workflow, "update_execution_data") as mock_update,
            patch.object(self.workflow, "get_output_path") as mock_get_path,
            patch("api_server.e2workflow.ZarrBackend") as mock_zarr,
        ):
            mock_get_path.return_value = tmp_path / "test_output"
            mock_io = Mock(spec=IOBackend)
            mock_zarr.return_value = mock_io

            result = self.workflow.run({"x": 100, "y": "test"}, "exec_123")

            # Verify workflow executed with correct parameters
            assert self.workflow.called_with["x"] == 100
            assert self.workflow.called_with["y"] == "test"
            assert result["status"] == "success"

            # Verify progress updates were made
            assert mock_update.call_count >= 2

    @patch("api_server.e2workflow.get_config")
    def test_run_method_handles_workflow_exceptions(self, mock_get_config):
        """Test that run method properly handles exceptions from workflow execution"""
        mock_config = MagicMock()
        mock_config.paths.output_format = "zarr"
        mock_get_config.return_value = mock_config

        class FailingWorkflow(Earth2Workflow):
            def __call__(self, io: IOBackend, x: int = 10):
                raise RuntimeError("Workflow execution failed")

        failing_workflow = FailingWorkflow()
        failing_workflow.name = "failing"
        failing_workflow.set_redis_client(Mock(spec=redis.Redis))

        with (
            patch.object(failing_workflow, "update_execution_data") as mock_update,
            patch.object(failing_workflow, "get_output_path"),
            patch("api_server.e2workflow.ZarrBackend"),
        ):
            with pytest.raises(RuntimeError, match="Workflow execution failed"):
                failing_workflow.run({}, "exec_fail")

            # Verify error was logged in execution data
            error_calls = [
                call
                for call in mock_update.call_args_list
                if "error_message" in call[1]["updates"]
            ]
            assert len(error_calls) > 0


# Test BackendProgress wrapper
class TestBackendProgress:
    """Test BackendProgress wrapper for progress tracking"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_io = Mock(spec=IOBackend)
        self.mock_workflow = Mock(spec=Earth2Workflow)
        self.execution_id = "test_exec"
        self.backend = BackendProgress(
            self.mock_io, self.mock_workflow, self.execution_id
        )

    def test_add_array_initializes_progress_tracking(self):
        """Test that add_array initializes progress tracking when progress_dim present"""
        coords = {"lead_time": [0, 1, 2, 3], "lat": [0, 1]}

        self.backend.add_array(coords, "temperature")

        # Verify progress tracking initialized
        assert self.backend.progress_coords == [0, 1, 2, 3]
        assert self.backend.progress_array == "temperature"
        self.mock_workflow.update_execution_data.assert_called_once_with(
            self.execution_id, updates={"current_step": 0, "total_steps": 4}
        )

        # Only tracks first array
        self.backend.add_array(coords, "pressure")
        assert self.backend.progress_array == "temperature"

    def test_write_updates_progress(self):
        """Test that write updates progress for tracked arrays"""
        # Initialize progress tracking
        init_coords: CoordSystem = {"lead_time": [0, 1, 2], "lat": [0, 1]}
        self.backend.add_array(init_coords, "temperature")
        self.mock_workflow.update_execution_data.reset_mock()

        # Write data at different time steps
        data = torch.randn(1, 2)

        write_coords1: CoordSystem = {"lead_time": [0], "lat": [0, 1]}
        self.backend.write(data, write_coords1, "temperature")
        self.mock_workflow.update_execution_data.assert_called_with(
            self.execution_id, updates={"current_step": 1}
        )

        write_coords2: CoordSystem = {"lead_time": [2], "lat": [0, 1]}
        self.backend.write(data, write_coords2, "temperature")
        self.mock_workflow.update_execution_data.assert_called_with(
            self.execution_id, updates={"current_step": 3}
        )

    def test_passthrough_to_underlying_io(self) -> None:
        """Test that BackendProgress properly wraps underlying IOBackend"""
        # Test method passthrough
        self.mock_io.some_method = Mock(return_value="test_value")
        assert self.backend.some_method() == "test_value"

        # Test attribute passthrough
        self.mock_io.some_attribute = "test_attr"
        assert self.backend.some_attribute == "test_attr"


# Integration tests
class TestEarth2WorkflowIntegration:
    """Integration tests for Earth2Workflow with BackendProgress"""

    @patch("api_server.e2workflow.get_config")
    def test_workflow_with_backend_progress_tracking(self, mock_get_config):
        """Test complete workflow execution with automatic progress tracking"""
        mock_config = MagicMock()
        mock_config.paths.output_format = "zarr"
        mock_get_config.return_value = mock_config

        class TestWorkflow(Earth2Workflow):
            def __call__(self, io: IOBackend, num_steps: int = 3):
                coords: CoordSystem = {
                    "lead_time": list(range(num_steps)),
                    "lat": [0, 1],
                }
                io.add_array(coords, "forecast")
                for i in range(num_steps):
                    data = torch.randn(1, 2)
                    write_coords: CoordSystem = {"lead_time": [i], "lat": [0, 1]}
                    io.write(data, write_coords, "forecast")

        workflow = TestWorkflow()
        workflow.name = "test"
        workflow.set_redis_client(Mock(spec=redis.Redis))

        with (
            patch.object(workflow, "update_execution_data") as mock_update,
            patch.object(workflow, "get_output_path"),
            patch("api_server.e2workflow.ZarrBackend"),
        ):
            result = workflow.run({"num_steps": 3}, "exec_test")

            # Verify successful execution
            assert result["status"] == "success"

            # Verify progress tracking was active
            progress_calls = [
                call
                for call in mock_update.call_args_list
                if "current_step" in call[1]["updates"]
            ]
            assert len(progress_calls) >= 1
