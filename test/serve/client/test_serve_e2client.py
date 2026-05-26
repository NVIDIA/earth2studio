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

from collections import OrderedDict
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.serve.client.e2client import (
    InferenceOutputModel,
    RemoteEarth2Workflow,
    RemoteEarth2WorkflowResult,
    _convert_time_to_lead_time,
)
from earth2studio.serve.client.exceptions import Earth2StudioAPIError
from earth2studio.serve.client.models import (
    InferenceRequestResponse,
    InferenceRequestResults,
    OutputFile,
    RequestStatus,
)


class TestRemoteEarth2WorkflowInitialization:
    """Test RemoteEarth2Workflow initialization"""

    def test_initialization(self) -> None:
        """Test workflow initialization with various device configurations"""
        with patch("earth2studio.serve.client.e2client.Earth2StudioClient"):
            # Test with CUDA available
            with patch("torch.cuda.is_available", return_value=True):
                workflow = RemoteEarth2Workflow(
                    base_url="http://localhost:8000",
                    workflow_name="test_workflow",
                )
                assert workflow.device == "cuda"

            # Test with CUDA unavailable
            with patch("torch.cuda.is_available", return_value=False):
                workflow = RemoteEarth2Workflow(
                    base_url="http://localhost:8000",
                    workflow_name="test_workflow",
                )
                assert workflow.device == "cpu"

            # Test with custom device and xr_args
            xr_args = {"chunks": {"time": 1}}
            workflow = RemoteEarth2Workflow(
                base_url="http://localhost:8000",
                workflow_name="test_workflow",
                device="cpu",
                xr_args=xr_args,
            )
            assert workflow.device == "cpu"
            assert workflow.xr_args == xr_args
            assert workflow.xr_args is not xr_args  # Verify it's a copy

    def test_initialization_with_client_kwargs(self) -> None:
        """Test workflow initialization passes kwargs to client"""
        with patch(
            "earth2studio.serve.client.e2client.Earth2StudioClient"
        ) as mock_client_class:
            RemoteEarth2Workflow(
                base_url="http://localhost:8000",
                workflow_name="test_workflow",
                timeout=60.0,
                max_retries=5,
            )
            mock_client_class.assert_called_once_with(
                base_url="http://localhost:8000",
                workflow_name="test_workflow",
                timeout=60.0,
                max_retries=5,
            )


class TestRemoteEarth2WorkflowCall:
    """Test RemoteEarth2Workflow __call__ method"""

    def test_call_submits_request(self) -> None:
        """Test that calling workflow submits inference request with parameters"""
        with patch(
            "earth2studio.serve.client.e2client.Earth2StudioClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_response = InferenceRequestResponse(
                execution_id="exec_123",
                status=RequestStatus.ACCEPTED,
                message="Request accepted",
                timestamp=datetime.now(),
            )
            mock_client.submit_inference_request.return_value = mock_response

            workflow = RemoteEarth2Workflow(
                base_url="http://localhost:8000",
                workflow_name="test_workflow",
            )

            # Call with parameters
            result = workflow(
                forecast_times=["2024-01-01T00:00:00"],
                nsteps=20,
                model="fcn",
            )

            assert isinstance(result, RemoteEarth2WorkflowResult)
            assert result.execution_id == "exec_123"

            # Verify parameters were passed correctly
            call_args = mock_client.submit_inference_request.call_args
            request = call_args[0][0]
            assert request.parameters["nsteps"] == 20
            assert request.parameters["model"] == "fcn"


class TestRemoteEarth2WorkflowTo:
    """Test RemoteEarth2Workflow to() method"""

    def test_to_device(self) -> None:
        """Test moving workflow to different devices"""
        with patch("earth2studio.serve.client.e2client.Earth2StudioClient"):
            workflow = RemoteEarth2Workflow(
                base_url="http://localhost:8000",
                workflow_name="test_workflow",
                device="cpu",
            )

            # Test with string
            result = workflow.to("cuda")
            assert workflow.device == "cuda"
            assert result is workflow

            # Test with torch.device
            device = torch.device("cpu")
            result = workflow.to(device)
            assert workflow.device == device


class TestRemoteEarth2WorkflowResult:
    """Test RemoteEarth2WorkflowResult class"""

    def test_result_caching(self) -> None:
        """Test that result is cached after first retrieval"""
        with patch(
            "earth2studio.serve.client.e2client.Earth2StudioClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_inference_result = InferenceRequestResults(
                request_id="exec_123",
                status=RequestStatus.COMPLETED,
                output_files=[],
                completion_time=datetime.now(),
            )
            mock_client.wait_for_completion.return_value = mock_inference_result

            workflow = RemoteEarth2Workflow(
                base_url="http://localhost:8000",
                workflow_name="test_workflow",
            )
            result = RemoteEarth2WorkflowResult(workflow, "exec_123")

            # Multiple calls should use cache
            result1 = result._get_result()
            result2 = result._get_result()
            assert result1 is result2
            mock_client.wait_for_completion.assert_called_once()

    def test_as_dataset_formats(self) -> None:
        """Test as_dataset with different output formats"""
        with patch(
            "earth2studio.serve.client.e2client.Earth2StudioClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client.timeout = 30.0  # used by as_dataset for zarr_timeout
            mock_client_class.return_value = mock_client

            workflow = RemoteEarth2Workflow(
                base_url="http://localhost:8000",
                workflow_name="test_workflow",
                xr_args={"decode_coords": False},
            )

            # Test Zarr format
            with patch("xarray.open_zarr") as mock_open_zarr:
                mock_inference_result = InferenceRequestResults(
                    request_id="exec_123",
                    status=RequestStatus.COMPLETED,
                    output_files=[OutputFile(path="output.zarr", size=1024)],
                    completion_time=datetime.now(),
                )
                mock_client.wait_for_completion.return_value = mock_inference_result
                mock_client.result_root_path.return_value = (
                    "/v1/infer/test/exec_123/results/"
                )

                mock_ds = xr.Dataset()
                mock_open_zarr.return_value = mock_ds

                result = RemoteEarth2WorkflowResult(workflow, "exec_123")
                ds = result.as_dataset()

                assert ds is mock_ds
                # Verify xr_args were passed
                assert mock_open_zarr.call_args[1]["decode_coords"] is False

            # Test NetCDF format
            with patch("xarray.open_dataset") as mock_open_dataset:
                mock_inference_result = InferenceRequestResults(
                    request_id="exec_123",
                    status=RequestStatus.COMPLETED,
                    output_files=[OutputFile(path="output.nc", size=512)],
                    completion_time=datetime.now(),
                )
                mock_client.wait_for_completion.return_value = mock_inference_result
                mock_client.download_result.return_value = MagicMock()

                result = RemoteEarth2WorkflowResult(workflow, "exec_123")
                result._result = None  # Reset cache
                ds = result.as_dataset()

                mock_client.download_result.assert_called_once()
                mock_open_dataset.assert_called_once()

    def test_as_dataset_no_outputs(self) -> None:
        """Test as_dataset when no output files are available"""
        with patch(
            "earth2studio.serve.client.e2client.Earth2StudioClient"
        ) as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            mock_inference_result = InferenceRequestResults(
                request_id="exec_123",
                status=RequestStatus.COMPLETED,
                output_files=[],
                completion_time=datetime.now(),
            )
            mock_client.wait_for_completion.return_value = mock_inference_result

            workflow = RemoteEarth2Workflow(
                base_url="http://localhost:8000",
                workflow_name="test_workflow",
            )
            result = RemoteEarth2WorkflowResult(workflow, "exec_123")

            with pytest.raises(
                Earth2StudioAPIError, match="did not return any outputs"
            ):
                result.as_dataset()

    def test_as_data_source_and_model(self) -> None:
        """Test as_data_source and as_model methods"""
        with (
            patch(
                "earth2studio.serve.client.e2client.Earth2StudioClient"
            ) as mock_client_class,
            patch("xarray.open_zarr"),
            patch(
                "earth2studio.serve.client.e2client.InferenceOutputSource"
            ) as mock_ios_class,
        ):
            mock_client = Mock()
            mock_client.timeout = 30.0  # used by as_dataset for zarr_timeout
            mock_client_class.return_value = mock_client

            mock_inference_result = InferenceRequestResults(
                request_id="exec_123",
                status=RequestStatus.COMPLETED,
                output_files=[OutputFile(path="output.zarr", size=1024)],
                completion_time=datetime.now(),
            )
            mock_client.wait_for_completion.return_value = mock_inference_result
            mock_client.result_root_path.return_value = (
                "/v1/infer/test/exec_123/results/"
            )

            workflow = RemoteEarth2Workflow(
                base_url="http://localhost:8000",
                workflow_name="test_workflow",
            )
            result = RemoteEarth2WorkflowResult(workflow, "exec_123")

            # Test as_data_source
            result.as_data_source()
            mock_ios_class.assert_called_once()

            # Test as_model with different iteration coordinates
            model = result.as_model(iter_coord="lead_time")
            assert isinstance(model, InferenceOutputModel)
            assert model.iter_coord == "lead_time"

            model = result.as_model(iter_coord="time")
            assert model.iter_coord == "time"


class TestConvertTimeToLeadTime:
    """Test _convert_time_to_lead_time function"""

    def test_convert_time_to_lead_time(self) -> None:
        """Test converting time coordinate to lead_time"""
        # Create test data - shape should match coord order: (time, lead_time, variable, lat, lon)
        x = torch.randn(3, 1, 4, 5, 6)
        start_time = np.datetime64("2024-01-01T00:00:00")
        times = np.array(
            [
                np.datetime64("2024-01-01T00:00:00"),
                np.datetime64("2024-01-01T06:00:00"),
                np.datetime64("2024-01-01T12:00:00"),
            ]
        )

        coords = OrderedDict(
            {
                "time": times,
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(["u10m", "v10m", "t2m", "tp"]),
                "lat": np.arange(5),
                "lon": np.arange(6),
            }
        )

        x_new, coords_new = _convert_time_to_lead_time(x, coords, start_time)

        # Verify transformation
        assert coords_new["time"].shape == (1,)
        assert coords_new["time"][0] == start_time
        assert coords_new["lead_time"].shape == (3,)
        # Verify lead times are correct
        expected_lead_times = times - start_time
        np.testing.assert_array_equal(coords_new["lead_time"], expected_lead_times)

    def test_convert_time_to_lead_time_preserves_other_coords(self) -> None:
        """Test that other coordinates are preserved"""
        # Shape: (time, lead_time, variable, lat, lon)
        x = torch.randn(2, 1, 3, 4, 5)
        start_time = np.datetime64("2024-01-01T00:00:00")
        times = np.array(
            [
                np.datetime64("2024-01-01T00:00:00"),
                np.datetime64("2024-01-01T06:00:00"),
            ]
        )

        coords = OrderedDict(
            {
                "time": times,
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(["u10m", "v10m", "t2m"]),
                "lat": np.arange(4),
                "lon": np.arange(5),
            }
        )

        _, coords_new = _convert_time_to_lead_time(x, coords, start_time)

        # Verify other coordinates are unchanged
        np.testing.assert_array_equal(coords_new["variable"], coords["variable"])
        np.testing.assert_array_equal(coords_new["lat"], coords["lat"])
        np.testing.assert_array_equal(coords_new["lon"], coords["lon"])


class TestInferenceOutputModel:
    """Test InferenceOutputModel class"""

    @pytest.fixture
    def mock_data_source(self) -> Any:
        """Create a mock InferenceOutputSource"""
        mock_ds = Mock()

        # Create mock DataArray with coordinates
        times = np.array(
            [
                np.datetime64("2024-01-01T00:00:00"),
                np.datetime64("2024-01-01T06:00:00"),
            ]
        )
        variables = np.array(["u10m", "v10m", "t2m"])
        lats = np.arange(10)
        lons = np.arange(20)

        # Create mock coordinate objects that support slicing
        mock_time_coord = Mock()
        mock_time_coord.values = times
        mock_time_coord.__getitem__ = Mock(
            side_effect=lambda key: Mock(values=times[key])
        )

        mock_lat_coord = Mock()
        mock_lat_coord.values = lats

        mock_lon_coord = Mock()
        mock_lon_coord.values = lons

        mock_da = Mock()
        mock_da.coords = {
            "time": mock_time_coord,
            "variable": variables,
            "lat": mock_lat_coord,
            "lon": mock_lon_coord,
        }

        mock_ds.da = mock_da
        return mock_ds

    def test_initialization(self, mock_data_source: Any) -> None:
        """Test InferenceOutputModel initialization with default and custom variables"""
        # Test with default variables
        model = InferenceOutputModel(
            data_source=mock_data_source,
            iter_coord="lead_time",
            device="cpu",
        )
        assert model.iter_coord == "lead_time"
        assert model.device == "cpu"
        np.testing.assert_array_equal(
            model.variables, np.array(["u10m", "v10m", "t2m"])
        )

        # Test with custom variables
        model = InferenceOutputModel(
            data_source=mock_data_source,
            variables=["u10m", "v10m"],
            device="cpu",
        )
        np.testing.assert_array_equal(model.variables, np.array(["u10m", "v10m"]))

    def test_coords(self, mock_data_source: Any) -> None:
        """Test input_coords and output_coords methods"""
        model = InferenceOutputModel(
            data_source=mock_data_source,
            iter_coord="lead_time",
            device="cpu",
        )

        # Test input_coords
        input_coords = model.input_coords()
        assert isinstance(input_coords, OrderedDict)
        assert len(input_coords["time"]) == 0
        assert len(input_coords["variable"]) == 0

        # Test output_coords
        output_coords = model.output_coords(OrderedDict())
        assert isinstance(output_coords, OrderedDict)
        np.testing.assert_array_equal(
            output_coords["variable"], np.array(["u10m", "v10m", "t2m"])
        )
        assert len(output_coords["lat"]) == 10
        assert len(output_coords["lon"]) == 20
        # Time step should be inferred as 6 hours
        assert output_coords["lead_time"][0] == np.timedelta64(6, "h")

    def test_output_coords_single_time_step(self) -> None:
        """Test output_coords with only one time step"""
        mock_ds = Mock()

        # Single time step
        times = np.array([np.datetime64("2024-01-01T00:00:00")])

        # Create mock coordinate that supports slicing
        mock_time_coord = Mock()
        mock_time_coord.values = times
        mock_time_coord.__getitem__ = Mock(
            side_effect=lambda key: Mock(values=times[key])
        )

        mock_lat_coord = Mock()
        mock_lat_coord.values = np.arange(10)

        mock_lon_coord = Mock()
        mock_lon_coord.values = np.arange(20)

        mock_da = Mock()
        mock_da.coords = {
            "time": mock_time_coord,
            "variable": np.array(["u10m"]),
            "lat": mock_lat_coord,
            "lon": mock_lon_coord,
        }
        mock_ds.da = mock_da

        model = InferenceOutputModel(
            data_source=mock_ds,
            iter_coord="lead_time",
            device="cpu",
        )

        output_coords = model.output_coords(OrderedDict())

        # Should use default placeholder of 6 hours
        assert output_coords["lead_time"][0] == np.timedelta64(6, "h")

    def test_to_device(self, mock_data_source: Any) -> None:
        """Test to() method"""
        model = InferenceOutputModel(
            data_source=mock_data_source,
            iter_coord="lead_time",
            device="cpu",
        )

        result = model.to("cuda")
        assert model.device == "cuda"
        assert result is model

    def test_call_method(self, mock_data_source: Any) -> None:
        """Test __call__ method returns first item from iterator"""
        model = InferenceOutputModel(
            data_source=mock_data_source,
            iter_coord="lead_time",
            device="cpu",
        )

        mock_tensor = torch.randn(1, 1, 3, 10, 20)
        mock_coords = OrderedDict({"time": np.array([np.datetime64("2024-01-01")])})

        with patch.object(
            model, "create_iterator", return_value=iter([(mock_tensor, mock_coords)])
        ):
            x, coords = model()
            assert torch.is_tensor(x)
            assert isinstance(coords, OrderedDict)

    def test_create_iterator(self, mock_data_source: Any) -> None:
        """Test create_iterator with and without lead_time conversion"""
        model = InferenceOutputModel(
            data_source=mock_data_source,
            iter_coord="time",
            device="cpu",
        )

        # Mock fetch_data to return different data for each time
        def mock_fetch_data(
            source: Any, time: Any, variable: Any, device: Any
        ) -> tuple[Any, Any]:
            tensor = torch.randn(1, 3, 10, 20)
            coords = OrderedDict(
                {
                    "time": time,
                    "variable": variable,
                    "lat": np.arange(10),
                    "lon": np.arange(20),
                }
            )
            return (tensor, coords)

        with patch(
            "earth2studio.serve.client.e2client.fetch_data",
            side_effect=mock_fetch_data,
        ):
            iterator = model.create_iterator()
            results = list(iterator)

            # Should yield data for both time steps
            assert len(results) == 2
            for x, coords in results:
                assert torch.is_tensor(x)
                assert isinstance(coords, OrderedDict)

    def test_create_iterator_with_lead_time_conversion(
        self, mock_data_source: Any
    ) -> None:
        """Test create_iterator with lead_time coordinate conversion"""
        model = InferenceOutputModel(
            data_source=mock_data_source,
            iter_coord="lead_time",
            device="cpu",
        )

        mock_tensor = torch.randn(1, 3, 10, 20)
        mock_coords = OrderedDict(
            {
                "time": np.array([np.datetime64("2024-01-01T00:00:00")]),
                "variable": np.array(["u10m", "v10m", "t2m"]),
                "lat": np.arange(10),
                "lon": np.arange(20),
                "lead_time": np.array([np.timedelta64(0, "h")]),
            }
        )

        with (
            patch(
                "earth2studio.serve.client.e2client.fetch_data",
                return_value=(mock_tensor, mock_coords),
            ),
            patch(
                "earth2studio.serve.client.e2client._convert_time_to_lead_time",
                return_value=(mock_tensor, mock_coords),
            ) as mock_convert,
        ):
            iterator = model.create_iterator()
            next(iterator)

            # Verify conversion was called
            mock_convert.assert_called_once()


class TestInferenceOutputModelIntegration:
    """Integration tests for InferenceOutputModel"""

    def test_model_mimics_prognostic_interface(self) -> None:
        """Test that InferenceOutputModel provides prognostic model interface"""
        mock_ds = Mock()

        times = np.array([np.datetime64("2024-01-01T00:00:00")])

        # Create mock coordinate that supports slicing
        mock_time_coord = Mock()
        mock_time_coord.values = times
        mock_time_coord.__getitem__ = Mock(
            side_effect=lambda key: Mock(values=times[key])
        )

        mock_lat_coord = Mock()
        mock_lat_coord.values = np.arange(10)

        mock_lon_coord = Mock()
        mock_lon_coord.values = np.arange(20)

        mock_da = Mock()
        mock_da.coords = {
            "time": mock_time_coord,
            "variable": np.array(["u10m"]),
            "lat": mock_lat_coord,
            "lon": mock_lon_coord,
        }
        mock_ds.da = mock_da

        model = InferenceOutputModel(
            data_source=mock_ds,
            iter_coord="lead_time",
            device="cpu",
        )

        # Verify interface methods exist and return correct types
        assert hasattr(model, "input_coords")
        assert hasattr(model, "output_coords")
        assert hasattr(model, "to")
        assert hasattr(model, "create_iterator")
        assert callable(model)
        assert isinstance(model.input_coords(), OrderedDict)
        assert isinstance(model.output_coords(OrderedDict()), OrderedDict)
