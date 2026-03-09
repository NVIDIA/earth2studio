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
from collections.abc import Iterator
from typing import Any, Literal
from urllib.parse import urljoin

import numpy as np
import torch

try:
    import aiohttp
except ImportError:
    aiohttp = None  # type: ignore[assignment]
import xarray as xr

from earth2studio.data import (  # type: ignore[import-untyped]
    InferenceOutputSource,
    fetch_data,
)
from earth2studio.models.auto import AutoModelMixin  # type: ignore[import-untyped]
from earth2studio.models.px.utils import PrognosticMixin  # type: ignore[import-untyped]
from earth2studio.serve.client import fsspec_utils
from earth2studio.serve.client.client import Earth2StudioClient
from earth2studio.serve.client.exceptions import Earth2StudioAPIError
from earth2studio.serve.client.models import (
    InferenceRequest,
    InferenceRequestResults,
    StorageType,
)
from earth2studio.utils.type import CoordSystem  # type: ignore[import-untyped]


class RemoteEarth2Workflow:
    """
    Remote inference workflow client for Earth2Studio API.

    Provides Earth2Studio-compatible interface for running inference on a remote
    Earth2Studio API server. Supports both direct calls and iterator-based access.

    Parameters
    ----------
    base_url : str
        URL of the Earth2Studio API server.
    workflow_name : str
        Name of the workflow to execute on the server.
    device : str or torch.device, optional
        Device for tensor operations (e.g. "cuda", "cpu"). Default from CUDA availability.
    xr_args : dict, optional
        Additional arguments passed to xarray.open_dataset / xarray.open_zarr.
    **client_kwargs : Any
        Additional arguments passed to Earth2StudioClient (e.g. token for authentication).
    """

    def __init__(
        self,
        base_url: str,
        workflow_name: str,
        device: str | torch.device | None = None,
        xr_args: dict[str, Any] | None = None,
        **client_kwargs: Any,
    ) -> None:
        self.base_url = base_url
        self.client = Earth2StudioClient(
            base_url=base_url, workflow_name=workflow_name, **client_kwargs
        )
        self.workflow_name = workflow_name
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.xr_args = xr_args.copy() if xr_args else {}

    def __call__(self, **kwargs: Any) -> "RemoteEarth2WorkflowResult":
        """
        Execute inference request and return result object for accessing outputs.

        Parameters
        ----------
        **kwargs : Any
            Workflow parameters for the inference request.

        Returns
        -------
        RemoteEarth2WorkflowResult
            Result object for accessing inference outputs.
        """
        request = InferenceRequest(parameters=kwargs.copy())
        response = self.client.submit_inference_request(request)
        return RemoteEarth2WorkflowResult(self, response.execution_id)

    def to(self, device: torch.device | str) -> "RemoteEarth2Workflow":
        """
        Move workflow to specified device.

        Parameters
        ----------
        device : torch.device or str
            Target device ("cuda", "cpu", or torch.device).

        Returns
        -------
        RemoteEarth2Workflow
            self for method chaining.
        """
        self.device = device
        return self


class RemoteEarth2WorkflowResult:
    """
    Result object for a remote inference request.

    Provides methods to access inference results as datasets, data sources, or iterators.
    The result is fetched lazily; methods wait for completion if needed.

    Parameters
    ----------
    workflow : RemoteEarth2Workflow
        Parent RemoteEarth2Workflow instance.
    execution_id : str
        Unique identifier for the inference execution.
    """

    def __init__(self, workflow: RemoteEarth2Workflow, execution_id: str) -> None:
        """
        Store the parent workflow and execution id; result is fetched lazily.

        Parameters
        ----------
        workflow : RemoteEarth2Workflow
            Parent workflow instance.
        execution_id : str
            Unique identifier for the inference execution.
        """
        self.workflow = workflow
        self.execution_id = execution_id
        self._result: InferenceRequestResults | None = None

    def _get_result(self) -> InferenceRequestResults:
        """
        Return cached result or wait for completion and cache it.

        Returns
        -------
        InferenceRequestResults
            Cached or newly fetched inference results.
        """
        if self._result is None:
            self._result = self.workflow.client.wait_for_completion(self.execution_id)
        return self._result

    def as_dataset(self) -> xr.Dataset:
        """
        Wait for inference to complete and return output dataset.

        Returns
        -------
        xr.Dataset
            Xarray Dataset containing the inference result data.

        Raises
        ------
        Earth2StudioAPIError
            If the request did not return any outputs.
        ValueError
            If the result file format is not .zarr or .nc.
        """
        request_result = self._get_result()
        result_paths = request_result.result_paths()
        if not result_paths:
            raise Earth2StudioAPIError("The request did not return any outputs.")
        result_path = result_paths[0]

        if result_path.endswith(".zarr"):
            if request_result.storage_type == StorageType.S3:
                # Extract zarr path without execution_id prefix (first path component)
                zarr_path = "/".join(result_path.split("/")[1:])
                mapper = fsspec_utils.get_mapper(request_result, zarr_path)
                ds = xr.open_zarr(mapper, consolidated=True, **self.workflow.xr_args)
            elif request_result.storage_type == StorageType.SERVER:
                result_url = urljoin(
                    self.workflow.base_url + "/",
                    (
                        self.workflow.client.result_root_path(request_result)
                        + result_path
                    ).lstrip("/"),
                )
                # Pass auth token and longer timeout for HTTP requests to zarr store
                xr_kwargs = dict(self.workflow.xr_args)
                storage_options = dict(xr_kwargs.pop("storage_options", {}))
                if self.workflow.client.token:
                    headers = dict(storage_options.get("headers", {}))
                    headers["Authorization"] = f"Bearer {self.workflow.client.token}"
                    storage_options["headers"] = headers
                # Use at least 300s timeout for zarr reads (fsspec/aiohttp)
                zarr_timeout = max(300.0, self.workflow.client.timeout)
                client_kwargs = dict(storage_options.get("client_kwargs", {}))
                if aiohttp is not None and "timeout" not in client_kwargs:
                    client_kwargs["timeout"] = aiohttp.ClientTimeout(total=zarr_timeout)
                storage_options["client_kwargs"] = client_kwargs
                ds = xr.open_zarr(
                    result_url,
                    consolidated=True,
                    storage_options=storage_options or None,
                    **xr_kwargs,
                )
            else:
                raise ValueError(
                    f"Unsupported storage type: {request_result.storage_type}"
                )
        elif result_path.endswith(".nc"):
            # TODO: support OpenDAP in the future for remote NetCDF4 access?
            result_data = self.workflow.client.download_result(
                request_result, result_path
            )
            ds = xr.open_dataset(result_data, engine="netcdf4", **self.workflow.xr_args)
        else:
            raise ValueError(
                f"Unsupported result file format: {result_path!r}. "
                "Only .zarr and .nc are supported for as_dataset()."
            )

        return ds

    def as_data_source(self) -> InferenceOutputSource:
        """
        Wait for inference to complete and return as DataSource.

        Returns
        -------
        DataSource
            Data source wrapper for the inference results.
        """
        ds = self.as_dataset()
        return InferenceOutputSource(ds)

    def as_model(
        self, iter_coord: Literal["time", "lead_time"] = "lead_time"
    ) -> "InferenceOutputModel":
        """
        Create model wrapper over inference results for iteration by time step.

        Parameters
        ----------
        iter_coord : {"time", "lead_time"}, optional
            Coordinate to iterate over. Default is "lead_time".

        Returns
        -------
        InferenceOutputModel
            Model that yields (tensor, coordinate_system) per time step.
        """
        data_source = self.as_data_source()
        return InferenceOutputModel(
            data_source=data_source, iter_coord=iter_coord, device=self.workflow.device
        )


def _convert_time_to_lead_time(
    x: torch.Tensor, coords: CoordSystem, start_time: np.datetime64
) -> tuple[torch.Tensor, CoordSystem]:
    """
    Convert time coordinate to lead_time coordinate.

    Transforms absolute time coordinates to lead times relative to start_time,
    adding a time dimension with the start time.

    Parameters
    ----------
    x : torch.Tensor
        Input tensor.
    coords : CoordSystem
        Coordinate system containing time coordinates.
    start_time : np.datetime64
        Reference start time for lead time calculation.

    Returns
    -------
    tuple[torch.Tensor, CoordSystem]
        Transformed tensor and updated coordinate system.
    """
    coords = coords.copy()
    time = coords["time"]
    dims = list(coords.keys())
    time_dim = dims.index("time")
    if "lead_time" not in dims:
        raise ValueError(
            "Cannot convert to lead_time: 'lead_time' coordinate not found in "
            f"fetch_data output. Available coords: {dims}"
        )
    lead_time_dim = dims.index("lead_time")
    lead_time = time - start_time
    coords["time"] = np.array([start_time])
    coords["lead_time"] = lead_time
    x = x.transpose(time_dim, lead_time_dim)
    return (x, coords)


class InferenceOutputModel(AutoModelMixin, PrognosticMixin):
    """
    Prognostic model wrapper for inference output data sources.

    Wraps an InferenceOutputSource to provide a prognostic model interface that
    can be used in Earth2Studio workflows. Pre-computed inference results are
    consumed as if generated by a live model.

    Parameters
    ----------
    data_source : InferenceOutputSource
        Pre-computed inference data source.
    iter_coord : {"time", "lead_time"}, optional
        Coordinate to iterate over. Default is "lead_time".
    variables : list[str], optional
        Variable names to include. If None, uses all from the data source.
    device : torch.device or str, optional
        Device for tensor operations (e.g. "cuda", "cpu"). Default is "cpu".
    """

    def __init__(
        self,
        data_source: InferenceOutputSource,
        iter_coord: Literal["time", "lead_time"] = "lead_time",
        variables: list[str] | None = None,
        device: torch.device | str = "cpu",
    ):
        self.data_source = data_source
        self.iter_coord = iter_coord
        self.variables = np.array(variables)
        self.device = device

    def input_coords(self) -> CoordSystem:
        """
        Return empty input coordinate system.

        This model reads from a pre-computed data source and requires no input
        coordinates.

        Returns
        -------
        CoordSystem
            Empty ordered dictionary of coordinates.
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array([]),
                "lat": np.empty(0),
                "lon": np.empty(0),
            }
        )

    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """
        Generate output coordinate system based on data source coordinates.

        Constructs the coordinate system for output tensors. Time step is inferred
        from the first two time coordinates if available, otherwise defaults to 6 hours.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system (unused; kept for interface compatibility).

        Returns
        -------
        CoordSystem
            Coordinate system with batch, time, lead_time, variable, lat, lon.
        """
        time_coord = self.data_source.da.coords["time"][:2].values
        if len(time_coord) >= 2:
            time_step = time_coord[1] - time_coord[0]
        else:
            # use a placeholder if we only have one time step of data
            time_step = np.timedelta64(6, "h")
        output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([time_step]),
                "variable": self.variables,
                "lat": self.data_source.da.coords["lat"].values,
                "lon": self.data_source.da.coords["lon"].values,
            }
        )
        return output_coords

    def to(self, device: torch.device | str) -> "InferenceOutputModel":
        """
        Move model to specified device.

        Parameters
        ----------
        device : torch.device or str
            Target device ("cuda", "cpu", or torch.device).

        Returns
        -------
        InferenceOutputModel
            self for method chaining.
        """
        self.device = device
        return self

    def __call__(
        self, x: torch.Tensor | None = None, coords: CoordSystem | None = None
    ) -> tuple[torch.Tensor, CoordSystem]:
        """
        Execute single time-step from the data source.

        Returns the first available time step. Input parameters are ignored;
        data is read from the pre-computed data source.

        Parameters
        ----------
        x : torch.Tensor, optional
            Input tensor (unused; kept for interface compatibility).
        coords : CoordSystem, optional
            Input coordinate system (unused; kept for interface compatibility).

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system for the first time step.
        """
        return next(self.create_iterator(x, coords))

    def create_iterator(
        self, x: torch.Tensor | None = None, coords: CoordSystem | None = None
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """
        Create iterator over time steps from the data source.

        Iterates through all available time steps, optionally converting time
        coordinates to lead_time relative to the first time step.

        Parameters
        ----------
        x : torch.Tensor, optional
            Input tensor (unused; kept for interface compatibility).
        coords : CoordSystem, optional
            Input coordinate system (unused; kept for interface compatibility).

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            (tensor, coordinate_system) for each time step in the data source.
        """
        times = self.data_source.da.coords["time"].values
        start_time = times[0]
        for time in times:
            (x, coords) = fetch_data(
                self.data_source,
                time=np.array([time]),
                variable=self.variables,
                device=self.device,
            )
            if self.iter_coord == "lead_time":
                (x, coords) = _convert_time_to_lead_time(x, coords, start_time)

            yield (x, coords)
