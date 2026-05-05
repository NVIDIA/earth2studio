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

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from earth2studio.serve.server.worker import get_output_path, run_custom_workflow


class TestGetOutputPath:
    """Tests for get_output_path."""

    def test_returns_path_with_default_backend_when_io_config_none(self):
        """When io_config is None, backend_type defaults to zarr."""
        with tempfile.TemporaryDirectory() as tmpdir:
            default_dir = Path(tmpdir)
            with (
                patch(
                    "earth2studio.serve.server.worker.DEFAULT_OUTPUT_DIR",
                    default_dir,
                ),
            ):
                out = get_output_path(
                    io_config=None,
                    timestamp="2024-01-01T12:00:00Z",
                    workflow_type="my_workflow",
                    request_id="req_123",
                )
            assert out.suffix == ".zarr"
            assert out.name == "forecast.zarr"
            assert "my_workflow" in str(out)
            assert "req_123" in str(out)
            assert out.parent.exists()

    def test_uses_backend_type_from_io_config(self):
        """When io_config has backend_type, output file uses that extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            default_dir = Path(tmpdir)
            with (
                patch(
                    "earth2studio.serve.server.worker.DEFAULT_OUTPUT_DIR",
                    default_dir,
                ),
            ):
                out = get_output_path(
                    io_config={"backend_type": "netcdf4"},
                    timestamp="2024-01-01T00:00:00",
                    workflow_type="wf",
                    request_id="r1",
                )
            assert out.suffix == ".netcdf4" or "netcdf4" in out.name
            assert out.name == "forecast.netcdf4"

    def test_normalizes_timestamp_in_path(self):
        """Timestamp is normalized (colons, Z, + removed) in directory name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            default_dir = Path(tmpdir)
            with (
                patch(
                    "earth2studio.serve.server.worker.DEFAULT_OUTPUT_DIR",
                    default_dir,
                ),
            ):
                out = get_output_path(
                    io_config=None,
                    timestamp="2024-01-01T12:00:00+00:00",
                    workflow_type="w",
                    request_id="r",
                )
            # Directory should not contain : or + or Z
            dir_name = out.parent.name
            assert ":" not in dir_name
            assert "Z" not in dir_name
            assert "+" not in dir_name


class TestRunCustomWorkflow:
    """Tests for run_custom_workflow."""

    def test_raises_value_error_when_workflow_not_in_registry(self):
        """Raises ValueError when workflow name is not registered."""
        with patch(
            "earth2studio.serve.server.worker.workflow_registry"
        ) as mock_registry:
            mock_registry.get_workflow_class.return_value = None

            with pytest.raises(ValueError, match="not found in registry"):
                run_custom_workflow(
                    workflow_name="nonexistent",
                    execution_id="exec_1",
                    parameters={},
                )

    def test_raises_value_error_when_workflow_cannot_be_instantiated(self):
        """Raises ValueError when registry.get returns None (instantiation failed)."""
        with patch(
            "earth2studio.serve.server.worker.workflow_registry"
        ) as mock_registry:
            mock_registry.get_workflow_class.return_value = MagicMock()
            mock_registry.get.return_value = None

            with pytest.raises(ValueError, match="could not be instantiated"):
                run_custom_workflow(
                    workflow_name="my_workflow",
                    execution_id="exec_1",
                    parameters={},
                )

    def test_success_returns_result_and_queues_next_stage(self):
        """On success, returns workflow result and queue_next_stage is called."""
        mock_workflow_class = MagicMock()
        mock_workflow = MagicMock()
        mock_workflow.run.return_value = {"result": "ok"}
        with tempfile.TemporaryDirectory() as tmpdir:
            safe_out = Path(tmpdir) / "out"
            safe_results_zip = Path(tmpdir) / "results_zip"
            mock_workflow.get_output_path.return_value = safe_out

            with (
                patch(
                    "earth2studio.serve.server.worker.workflow_registry"
                ) as mock_registry,
                patch(
                    "earth2studio.serve.server.worker.redis_client",
                    MagicMock(),
                ),
                patch(
                    "earth2studio.serve.server.worker.queue_next_stage",
                    return_value="job_123",
                ) as mock_queue,
                patch(
                    "earth2studio.serve.server.worker.RESULTS_ZIP_DIR",
                    safe_results_zip,
                ),
            ):
                mock_registry.get_workflow_class.return_value = mock_workflow_class
                mock_registry.get.return_value = mock_workflow

                result = run_custom_workflow(
                    workflow_name="my_workflow",
                    execution_id="exec_1",
                    parameters={"key": "value"},
                )

            assert result == {"result": "ok"}
            mock_workflow.run.assert_called_once_with({"key": "value"}, "exec_1")
            mock_queue.assert_called_once()
            call_kw = mock_queue.call_args[1]
            assert call_kw["workflow_name"] == "my_workflow"
            assert call_kw["execution_id"] == "exec_1"
            assert call_kw["current_stage"] == "inference"

    def test_raises_runtime_error_when_queue_next_stage_fails(self):
        """When queue_next_stage returns None, raises RuntimeError and updates status."""
        mock_workflow_class = MagicMock()
        mock_workflow = MagicMock()
        mock_workflow.run.return_value = {}
        with tempfile.TemporaryDirectory() as tmpdir:
            safe_out = Path(tmpdir) / "out"
            safe_results_zip = Path(tmpdir) / "results_zip"
            mock_workflow.get_output_path.return_value = safe_out

            with (
                patch(
                    "earth2studio.serve.server.worker.workflow_registry"
                ) as mock_registry,
                patch(
                    "earth2studio.serve.server.worker.redis_client",
                    MagicMock(),
                ),
                patch(
                    "earth2studio.serve.server.worker.queue_next_stage",
                    return_value=None,
                ),
                patch(
                    "earth2studio.serve.server.worker.RESULTS_ZIP_DIR",
                    safe_results_zip,
                ),
            ):
                mock_registry.get_workflow_class.return_value = mock_workflow_class
                mock_registry.get.return_value = mock_workflow

                with pytest.raises(RuntimeError, match="Failed to queue next pipeline"):
                    run_custom_workflow(
                        workflow_name="my_workflow",
                        execution_id="exec_1",
                        parameters={},
                    )

        # Status should have been updated to FAILED
        mock_workflow_class._update_execution_data.assert_called()
        calls = mock_workflow_class._update_execution_data.call_args_list
        # Last call before raise is the one with status FAILED
        fail_calls = [c for c in calls if c[0][3].get("status") == "failed"]
        assert len(fail_calls) >= 1

    def test_on_run_exception_updates_status_and_reraises(self):
        """When workflow.run raises, status is updated to FAILED and exception is re-raised."""
        mock_workflow_class = MagicMock()
        mock_workflow = MagicMock()
        mock_workflow.run.side_effect = ValueError("run failed")

        with (
            patch(
                "earth2studio.serve.server.worker.workflow_registry"
            ) as mock_registry,
            patch(
                "earth2studio.serve.server.worker.redis_client",
                MagicMock(),
            ),
        ):
            mock_registry.get_workflow_class.return_value = mock_workflow_class
            mock_registry.get.return_value = mock_workflow

            with pytest.raises(ValueError, match="run failed"):
                run_custom_workflow(
                    workflow_name="my_workflow",
                    execution_id="exec_1",
                    parameters={},
                )

        mock_workflow_class._update_execution_data.assert_called()
        calls = mock_workflow_class._update_execution_data.call_args_list
        fail_updates = [c for c in calls if c[0][3].get("status") == "failed"]
        assert len(fail_updates) >= 1
