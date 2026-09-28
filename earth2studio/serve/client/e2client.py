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
)
from earth2studio.models.auto import AutoModelMixin  # type: ignore[import-untyped]
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.serve.client import fsspec_utils
from earth2studio.serve.client.client import Earth2StudioClient
from earth2studio.serve.client.exceptions import Earth2StudioAPIError
from earth2studio.serve.client.models import (
    InferenceRequest,
    InferenceRequestResults,
    StorageType,
)
from earth2studio.utils.coords import coord_array, coord_array_like, handshake_dataarray
from earth2studio.utils.type import CoordinateSystem


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
            Model that yields a field DataArray per time step.
        """
        data_source = self.as_data_source()
        return InferenceOutputModel(
            data_source=data_source, iter_coord=iter_coord, device=self.workflow.device
        )


def _convert_time_to_lead_time(
    x: xr.DataArray, start_time: np.datetime64
) -> xr.DataArray:
    if x.sizes.get("lead_time") != 1:
        raise ValueError("Time conversion requires a singleton lead_time dimension")
    lead = x.time.values + x.lead_time.values[0] - start_time
    dims = x.dims
    result = x.isel(lead_time=0, drop=True).rename(time="lead_time")
    return (
        result.assign_coords(lead_time=lead)
        .expand_dims(time=[start_time])
        .transpose(*dims)
    )


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
        if variables is None:
            variables = self.data_source.da.coords["variable"]
        self.variables = np.array(variables)
        self.device = device

    def input_coords(self) -> CoordinateSystem:
        """Declare the selected variables and stored spatial coordinates."""
        source = self.data_source.da.sel(variable=self.variables)
        spatial = tuple(d for d in source.dims if d not in ("time", "variable"))
        coordinates = {
            name: value.variable
            for name, value in source.coords.items()
            if "time" not in value.dims and name != "time"
        }
        coordinates["lead_time"] = np.array([0], dtype="timedelta64[h]")
        return coord_array(
            ("batch", "time", "lead_time", "variable", *spatial),
            coordinates,
            dynamic=("batch", "time"),
            attrs=source.attrs,
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """
        Generate output coordinate system based on data source coordinates.

        Constructs an allocation-free signature. Time step is inferred
        from the first two time coordinates if available, otherwise defaults to 6 hours.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input declaration or field coordinates.

        Returns
        -------
        CoordinateSystem
            Coordinate signature for the next stored step.
        """
        time_coord = self.data_source.da.coords["time"][:2].values
        if len(time_coord) >= 2:
            time_step = time_coord[1] - time_coord[0]
        else:
            # use a placeholder if we only have one time step of data
            time_step = np.timedelta64(6, "h")
        lead = input_coords.lead_time.values
        if (
            lead.size != 1
            or not np.issubdtype(lead.dtype, np.timedelta64)
            or np.isnat(lead).any()
        ):
            raise ValueError("lead_time must contain one finite timedelta")
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
        return coord_array_like(input_coords, {"lead_time": lead + time_step})

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

    def __call__(self, x: xr.DataArray | None = None) -> xr.DataArray:
        """
        Execute single time-step from the data source.

        Returns the first available time step. Input parameters are ignored;
        data is read from the pre-computed data source.

        Parameters
        ----------
        x : xr.DataArray, optional
            Input field (unused; stored results are read in their original order).

        Returns
        -------
        xr.DataArray
            Field for the first stored time step. Iterator hooks are not applied.
        """
        return self._read_step(self.data_source.da.time.values[0])

    def _read_step(self, time: np.datetime64) -> xr.DataArray:
        # Stored labels already describe computed quantities; fetch_data would
        # interpret qualified labels as requests for another temporal reduction.
        field = self.data_source(np.array([time]), self.variables).copy(deep=True)
        signature = self.input_coords()
        # Derive only the selected labels' statistics, without reducing stored values
        # or replacing the source's user/grid metadata with signature-only attrs.
        field.attrs.pop("earth2studio_statistics", None)
        if "earth2studio_statistics" in signature.attrs:
            field.attrs["earth2studio_statistics"] = signature.attrs[
                "earth2studio_statistics"
            ]
        field = field.expand_dims(
            lead_time=np.array([0], dtype="timedelta64[h]")
        ).transpose(*signature.dims[1:])
        if self.iter_coord == "lead_time":
            field = _convert_time_to_lead_time(
                field, self.data_source.da.time.values[0]
            )
        device = torch.device(self.device)
        if device.type == "cuda":
            return field.e2s.as_cupy(device.index)
        if device.type != "cpu":
            raise ValueError("Stored output replay supports only CPU and CUDA")
        return field.e2s.as_numpy()

    def create_iterator(self, x: xr.DataArray | None = None) -> Iterator[xr.DataArray]:
        """
        Create iterator over time steps from the data source.

        Iterates through all available time steps, optionally converting time
        coordinates to lead_time relative to the first time step.

        Parameters
        ----------
        x : xr.DataArray, optional
            Input field (unused; stored results are read in their original order).

        Yields
        ------
        xr.DataArray
            Field for each stored time step, on the selected device.
        """
        times = self.data_source.da.coords["time"].values
        for time in times:
            yield self._read_step(time)
