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
from copy import deepcopy
from typing import Any

import cftime
import numpy as np
import pandas as pd
import torch
import xarray as xr
from earth2studio.models._array_utils import _registered_grid
from loguru import logger

from earth2studio.data import ACE2ERA5Data
from earth2studio.data.ace2 import ACE_GRID_LAT, ACE_GRID_LON
from earth2studio.data.base import DataSource
from earth2studio.data.utils import fetch_data
from earth2studio.grids import LatLonGrid
from earth2studio.lexicon.ace import ACELexicon
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import DataArrayPrognosticMixin
from earth2studio.utils.coords import (
    coord_array,
    coord_array_like,
    handshake_coords,
    handshake_dataarray,
    handshake_dim,
    handshake_size,
    handshake_time,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.interp import LatLonInterpolation
from earth2studio.utils.type import CoordinateSystem, CoordSystem

try:
    # Optional dependency: FME
    from fme.ace.data_loading.batch_data import BatchData, PrognosticState
    from fme.ace.stepper.single_module import Stepper, load_stepper
except ImportError:
    OptionalDependencyFailure("ace2")
    BatchData = Any
    PrognosticState = Any
    Stepper = Any


def _npdatetime64_to_cftime(dt64_array: np.ndarray) -> np.ndarray:
    """Convert np.datetime64[...] array to cftime.DatetimeProlepticGregorian array
    (vectorized). Only supports up to seconds precision."""

    if len(dt64_array.shape) > 1:
        # Flatten the array before applying conversion
        return_shape = list(dt64_array.shape)
        dt64_array = dt64_array.reshape(-1)
    else:
        return_shape = None

    dt_index = pd.to_datetime(dt64_array)

    years = dt_index.year
    months = dt_index.month
    days = dt_index.day
    hours = dt_index.hour
    minutes = dt_index.minute
    seconds = dt_index.second

    result = np.fromiter(
        (
            cftime.DatetimeProlepticGregorian(y, m, d, H, M, S)
            for y, m, d, H, M, S in zip(years, months, days, hours, minutes, seconds)
        ),
        dtype=object,
        count=len(dt64_array),
    )

    if return_shape is not None:
        result = result.reshape(return_shape)
    return result


def _cftime_to_npdatetime64(cftime_array: np.ndarray) -> np.ndarray:
    """Convert cftime.DatetimeProlepticGregorian array to np.datetime64[s] array
    (vectorized-safe). Only supports up to seconds precision. Out-of-range years become
    NaT.
    """

    def _convert_single(t: cftime.DatetimeProlepticGregorian) -> np.datetime64:
        if not (1678 <= t.year <= 2261):
            return np.datetime64("NaT")
        return np.datetime64(
            f"{t.year:04d}-{t.month:02d}-{t.day:02d}T"
            f"{t.hour:02d}:{t.minute:02d}:{t.second:02d}"
        )

    vec_convert = np.vectorize(_convert_single, otypes=["datetime64[s]"])
    return vec_convert(cftime_array)


@check_optional_dependencies()
class ACE2ERA5(torch.nn.Module, AutoModelMixin, DataArrayPrognosticMixin):
    """ACE2-ERA5 prognostic model wrapper.

    ACE2 (Ai2 Climate Emulator v2) is a 450M-parameter autoregressive emulator
    with 6-hour time steps, 1-degree horizontal resolution, and eight vertical
    layers that exactly conserves global dry air mass and moisture and can be
    stepped stably for arbitrarily many steps. ACE2-ERA5 was trained on the ERA5
    dataset and requires forcing data during rollout (see `forcing_data_source`
    parameter). This wrapper makes use of the ``fme`` package to run model forward
    passes.

    Parameters
    ----------
    stepper : Stepper
        ACE2-ERA5 fme.ace.stepper.single_module.Stepper instance loaded from a checkpoint.
    forcing_data_source : DataSource, optional
        Data source providing forcing data during rollout. Must provide all forcing
        variables described in the ACE2-ERA5 paper, by default ACE2ERA5(mode="forcing").
    dt : numpy.timedelta64, optional
        Model timestep used to advance lead time coordinates, by default 6 hours.

    References
    ----------

    - ACE2-ERA5 paper: https://arxiv.org/abs/2411.11268v1
    - ACE2 code: https://github.com/ai2cm/ace
    - Huggingface: https://huggingface.co/allenai/ACE2-ERA5

    Notes
    -----
    For throughput-sensitive GPU inference, enabling TensorFloat-32 matmul kernels
    before importing PyTorch can improve performance on supported NVIDIA GPUs:

        export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1

    For in-process control, this can also be enabled with:

        torch.set_float32_matmul_precision("high")

    Both settings trade some float32 matmul precision for faster matrix operations;
    the environment variable is a process-wide cuBLAS override.

    Warning
    -------
    This model may only be used with input data on the GPU device that the model was
    loaded on. Specifically, the data must be on the same device as whatever
    ``torch.cuda.current_device()`` was set to when the model package was loaded.

    Badges
    ------
    region:global class:climate product:wind product:precip product:temp product:atmos
    product:ocean product:land year:2024 gpu:40gb
    provider:ai2 backend:pytorch
    """

    def __init__(
        self,
        stepper: Stepper,
        forcing_data_source: DataSource = ACE2ERA5Data(mode="forcing"),
        dt: np.timedelta64 = np.timedelta64(6, "h"),
    ):
        super().__init__()

        # Load fme stepper and cache useful metadata
        self.stepper = stepper
        self.register_buffer("device_buffer", torch.empty(0))

        # timestep (lead time increment)
        self._dt = dt

        # Variable layouts
        # Inputs expected by stepper (may include prognostic + forcing variables)
        in_vars = list(self.stepper.prognostic_names) + list(
            self.stepper._input_only_names
        )
        # Outputs predicted by stepper
        out_vars = list(self.stepper.out_names)

        # Use shared lexicon
        self.lexicon = ACELexicon

        # Establish internal variable orders
        self._all_in_variables_fme = sorted(set(in_vars))
        self._all_in_variables_e2s = [
            self.lexicon.get_e2s_from_fme(v) for v in self._all_in_variables_fme
        ]

        self._all_out_variables_fme = out_vars
        self._all_out_variables_e2s = [
            self.lexicon.get_e2s_from_fme(v) for v in self._all_out_variables_fme
        ]

        self._forcing_vars_fme = sorted(self.stepper._input_only_names)
        if (
            "surface_temperature" in self._all_out_variables_fme
            and "surface_temperature" not in self._forcing_vars_fme
        ):
            # ACE2 reuses surface_temperature for both skin temperature of land and ocean
            # `self.stepper._input_only_names` is computed by fme as the set difference of input and prognostic variables,
            # which accidentally drops surface_temperature, so we reinject it here
            self._forcing_vars_fme.append("surface_temperature")
        self._forcing_vars_e2s = [
            self.lexicon.get_e2s_from_fme(v) for v in self._forcing_vars_fme
        ]
        self._prog_vars_fme = sorted(self.stepper.prognostic_names)
        self._prog_vars_e2s = [
            self.lexicon.get_e2s_from_fme(v) for v in self._prog_vars_fme
        ]

        # External forcing data source
        self.forcing_data_source = forcing_data_source
        self._forcing_cache: dict[
            tuple[int, tuple[str, ...], str, str, int],
            tuple[torch.Tensor, CoordSystem],
        ] = {}

        # Grid handling
        self.lat = ACE_GRID_LAT
        self.lon = ACE_GRID_LON
        if hasattr(forcing_data_source, "lat") and hasattr(forcing_data_source, "lon"):
            # Attempt to check for grid compatibility / need to regrid
            if not np.allclose(forcing_data_source.lat, self.lat) or not np.allclose(
                forcing_data_source.lon, self.lon
            ):
                self.needs_regrid = True
                # Need to regrid forcing data to ACE2 grid
                lat_in, lon_in = np.meshgrid(
                    forcing_data_source.lat, forcing_data_source.lon, indexing="ij"
                )
                lat_out, lon_out = np.meshgrid(self.lat, self.lon, indexing="ij")
                self.regridder = LatLonInterpolation(
                    lat_in=lat_in,
                    lon_in=lon_in,
                    lat_out=lat_out,
                    lon_out=lon_out,
                )
            else:
                self.needs_regrid = False
        else:
            self.needs_regrid = False

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the prognostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        coords = coord_array(
            ("batch", "time", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array(
                    [np.timedelta64(0, "h")], dtype="timedelta64[ns]"
                ),
                "variable": np.array(self._prog_vars_e2s, dtype=object),
            },
            dynamic=("batch", "time"),
            grid=_registered_grid(LatLonGrid(self.lat, self.lon)),
        )
        return coords

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the prognostic model

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords
            by default None, will use self.input_coords.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        handshake_dataarray(input_coords, self.input_coords(), relative_lead_time=True)
        lead = input_coords.lead_time
        return coord_array_like(
            input_coords,
            {
                "lead_time": lead.values + self._dt,
                "variable": np.array(self._all_out_variables_e2s),
            },
        )

    @classmethod
    def load_default_package(cls) -> Package:
        """Load default ACE2-ERA5 package from HuggingFace."""
        return Package(
            "hf://allenai/ACE2-ERA5",
            cache_options={
                "cache_storage": Package.default_cache("ace2era5"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        forcing_data_source: DataSource = ACE2ERA5Data(mode="forcing", verbose=False),
        dt: np.timedelta64 = np.timedelta64(6, "h"),
    ) -> PrognosticModel:
        """Load ACE2-ERA5 prognostic model from a package.

        Parameters
        ----------
        package : Package
            Package to load the model checkpoint from.
        forcing_data_source : DataSource, optional
            External forcing data source. Must provide all forcing variables
            described in the ACE2-ERA5 paper, by default ACE2ERA5(mode="forcing").
        dt : numpy.timedelta64, optional
            Timestep for advancing lead time coordinates, by default 6 hours.

        Returns
        -------
        PrognosticModel
            ACE2-ERA5 prognostic model
        """
        checkpoint_path = package.resolve("ace2_era5_ckpt.tar")
        stepper = load_stepper(checkpoint_path)
        return cls(
            stepper=stepper,
            forcing_data_source=forcing_data_source,
            dt=dt,
        )

    def _tensor_to_batch_data(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        forcing_x: torch.Tensor,
        forcing_coords: CoordSystem,
    ) -> tuple[BatchData, PrognosticState]:
        """Pack Earth2Studio (x, coords) into fme BatchData/PrognosticState.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system
        forcing_x : torch.Tensor
            Forcing tensor
        forcing_coords : CoordSystem
            Forcing coordinate system

        Returns
        -------
        tuple[BatchData, PrognosticState]
            Packed fme BatchData/PrognosticState
        """

        # Input validation
        handshake_dim(coords, ("batch", "time", "lead_time", "variable", "lat", "lon"))

        for c in ["batch", "time", "lat", "lon"]:
            handshake_coords(coords, forcing_coords, c)

        # Flatten the time and batch dimensions
        b, t, lt, v, lat, lon = x.shape
        x = x.reshape(b * t, lt, v, lat, lon)
        forcing_x = forcing_x.reshape(
            b * t,
            len(forcing_coords["lead_time"]),
            len(forcing_coords["variable"]),
            lat,
            lon,
        )

        # Build data dict with shape [batch, n_times, *domain]
        forcing_data: dict[str, torch.Tensor] = {}
        state_data: dict[str, torch.Tensor] = {}
        for fme_name in self._all_in_variables_fme:
            e2s_name = self.lexicon.get_e2s_from_fme(fme_name)
            if fme_name == "surface_temperature":
                # Skin temperature is used as both forcing and prognostic, depending on if over land or ocean
                state_idx = list(coords["variable"]).index(e2s_name)
                forcing_idx = list(forcing_coords["variable"]).index(e2s_name)
                forcing_data[fme_name] = forcing_x[:, :, forcing_idx, ...]
                state_data[fme_name] = x[:, :, state_idx, ...]
            elif fme_name in self._forcing_vars_fme:
                j = list(forcing_coords["variable"]).index(e2s_name)
                forcing_data[fme_name] = forcing_x[:, :, j, ...]
            else:
                j = list(coords["variable"]).index(e2s_name)
                state_data[fme_name] = x[:, :, j, ...]

        # Pass a time array and hc_dims to initialize BatchData on device
        times_forcing = np.tile(
            coords["time"][:, None] + forcing_coords["lead_time"][None, :], (b, 1)
        )
        times_state = np.tile(
            coords["time"][:, None] + coords["lead_time"][None, :], (b, 1)
        )
        time_da_forcing = xr.DataArray(
            _npdatetime64_to_cftime(times_forcing), dims=["sample", "time"]
        )
        time_da_state = xr.DataArray(
            _npdatetime64_to_cftime(times_state), dims=["sample", "time"]
        )
        hc_dims = ["lat", "lon"]
        forcing_data = BatchData.new_on_device(
            data=forcing_data,
            time=time_da_forcing,
            labels=[set()],
            horizontal_dims=hc_dims,
        )
        state_data = BatchData.new_on_device(
            data=state_data, time=time_da_state, horizontal_dims=hc_dims, labels=[set()]
        )
        return forcing_data, PrognosticState(state_data)

    def _batch_data_to_tensor(self, data: dict[str, torch.Tensor]) -> torch.Tensor:
        """Convert fme BatchData/PrognosticState back to (x, coords) tensor pair.

        Parameters
        ----------
        data : dict[str, torch.Tensor]
            Dictionary of fme BatchData/PrognosticState data

        Returns
        -------
        torch.Tensor
            Predicted data in (x, coords) tensor format
        """
        pred_vars = self._all_out_variables_fme
        y_list = [data[name] for name in pred_vars]
        # Each element shape: [batch, 1, *domain]; stack along variable position
        y = torch.stack(y_list, dim=2)  # -> [batch, 1, variable, *domain]
        # Add explicit lead_time dim of size 1 at index 2 (after time)
        y = y.unsqueeze(2)
        return y

    def _fetch_forcing_year(
        self, year: int, device: torch.device
    ) -> tuple[torch.Tensor, CoordSystem]:
        device = torch.device(device)
        cache_key = (
            id(self.forcing_data_source),
            tuple(self._forcing_vars_e2s),
            str(device),
            "year",
            year,
        )
        if cache_key in self._forcing_cache:
            return self._forcing_cache[cache_key]

        logger.warning(
            "Loading ACE2 forcing data for year {} onto {}. This replaces any "
            "previous cached forcing year.",
            year,
            device,
        )
        start = np.datetime64(f"{year:04d}-01-01T00:00:00", "ns")
        end = np.datetime64(f"{year + 1:04d}-01-01T00:00:00", "ns")
        year_times = np.arange(start, end, self._dt, dtype="datetime64[ns]")
        lead_time = np.array([np.timedelta64(0, "h")], dtype="timedelta64[ns]")

        forcing = fetch_data(
            self.forcing_data_source,
            time=year_times,
            lead_time=lead_time,
            variable=self._forcing_vars_e2s,
            device=device,
        )
        forcing_x, year_coords = forcing.e2s.to_torch()
        self._forcing_cache.clear()
        self._forcing_cache[cache_key] = (forcing_x, year_coords)
        return self._forcing_cache[cache_key]

    def _fetch_forcing_at_time(
        self, valid_time: np.datetime64, device: torch.device
    ) -> tuple[torch.Tensor, CoordSystem]:
        valid_time = valid_time.astype("datetime64[ns]")
        device = torch.device(device)
        if isinstance(self.forcing_data_source, ACE2ERA5Data):
            year = pd.Timestamp(valid_time).year
            forcing_x, forcing_coords = self._fetch_forcing_year(year, device)

            year_start = np.datetime64(f"{year:04d}-01-01T00:00:00", "ns")
            delta_ns = (
                (valid_time - year_start).astype("timedelta64[ns]").astype(np.int64)
            )
            step_ns = np.asarray(self._dt).astype("timedelta64[ns]").astype(np.int64)
            if delta_ns % step_ns != 0:
                raise ValueError(
                    f"ACE2 forcing time {valid_time} is not on model step {self._dt}."
                )
            time_index = int(delta_ns // step_ns)

            out_coords = forcing_coords.copy()
            out_coords["time"] = np.array([valid_time], dtype="datetime64[ns]")
            return forcing_x[time_index : time_index + 1], out_coords

        cache_key = (
            id(self.forcing_data_source),
            tuple(self._forcing_vars_e2s),
            str(device),
            "time",
            int(valid_time.astype("datetime64[ns]").astype(np.int64)),
        )
        if cache_key not in self._forcing_cache:
            forcing = fetch_data(
                self.forcing_data_source,
                time=np.array([valid_time], dtype="datetime64[ns]"),
                lead_time=np.array([np.timedelta64(0, "h")], dtype="timedelta64[ns]"),
                variable=self._forcing_vars_e2s,
                device=device,
            )
            self._forcing_cache[cache_key] = forcing.e2s.to_torch()
        forcing_x, forcing_coords = self._forcing_cache[cache_key]
        return forcing_x, forcing_coords.copy()

    def _fetch_forcing(
        self, x: torch.Tensor, coords: CoordSystem, lead_times: np.ndarray
    ) -> tuple[torch.Tensor, CoordSystem]:
        handshake_time(coords)
        handshake_time({"lead_time": lead_times}, "lead_time")
        forcing_by_lead = []
        forcing_coords: CoordSystem = OrderedDict()
        for lead_time in lead_times:
            forcing_by_time = []
            for time in coords["time"]:
                forcing_x, forcing_coords = self._fetch_forcing_at_time(
                    time + lead_time, x.device
                )
                forcing_by_time.append(forcing_x)
            forcing_by_lead.append(torch.cat(forcing_by_time, dim=0))

        forcing_x = torch.cat(forcing_by_lead, dim=1)
        forcing_coords["time"] = coords["time"]
        forcing_coords["lead_time"] = lead_times.astype("timedelta64[ns]")

        if self.needs_regrid:
            forcing_x = self.regridder(forcing_x.to(x.device))
            forcing_coords["lat"] = coords["lat"]
            forcing_coords["lon"] = coords["lon"]

        return forcing_x, forcing_coords

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Run one prognostic step using fme predict_paired API.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system 6 hours in the future
        """

        # Validate input lead_time
        handshake_size(coords, "lead_time", 1)

        # Pull forcing data (which is required at both input and output lead times)
        lead_times = np.array(
            [coords["lead_time"][0], coords["lead_time"][0] + self._dt]
        )
        forcing_x, forcing_coords = self._fetch_forcing(
            x=x, coords=coords, lead_times=lead_times
        )

        # Stack along batch dimension as required
        forcing_x = torch.stack([forcing_x] * len(coords["batch"]), dim=0).to(
            device=x.device, dtype=x.dtype
        )
        forcing_coords["batch"] = coords["batch"]
        forcing_coords.move_to_end("batch", last=False)

        # Prepare inputs for fme stepper
        forcing_batch, ic = self._tensor_to_batch_data(
            x, coords.copy(), forcing_x, forcing_coords
        )

        # Predict one step forward
        paired, _ = self.stepper.predict_paired(ic, forcing_batch)
        y = self._batch_data_to_tensor(paired.prediction)
        y = y.reshape(
            x.shape[0], x.shape[1], 1, len(self._all_out_variables_e2s), *x.shape[-2:]
        )
        out_coords = coords.copy()
        out_coords["lead_time"] = coords["lead_time"] + self._dt
        out_coords["variable"] = np.array(self._all_out_variables_e2s)
        return y, out_coords

    @batch_func()
    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Runs one prognostic step using fme predict_paired API.

        Parameters
        ----------
        x : xr.DataArray
            Prognostic state on the declared grid, with arbitrary leading dimensions.

        Returns
        -------
        xr.DataArray
            Prognostic and diagnostic fields one timestep in the future.
        """
        signature = self.output_coords(x)
        handshake_time(x)
        tensor, coords = x.e2s.to_torch()
        out, _ = self._forward(tensor.to(self.device_buffer.device).clone(), coords)
        result = from_torch(out, signature)
        result.encoding = x.encoding.copy()
        return result

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Creates an iterator to perform time-integration of ACE2ERA5.

        Yields the initial state, then continues autoregressively by feeding
        previous outputs as the next prognostic state while fetching/using external
        forcings under the hood via _forward.

        Parameters
        ----------
        x : xr.DataArray
            Initial state; hooks receive owned arrays in original leading dimensions.

        Returns
        -------
        Iterator[xr.DataArray]
            Initial state followed by forecasts.
        """
        handshake_dataarray(x, runtime=True)
        handshake_time(x)
        self.output_coords(x)
        yield x.isel(lead_time=slice(-1, None)).copy(deep=True)
        while True:
            history = self.front_hook(x.copy(deep=True))
            out = self.rear_hook(self(history))
            # Preserve prognostics absent from the checkpoint's output list.
            tensor, _ = history.e2s.to_torch()
            predicted, _ = out.e2s.to_torch()
            tensor = tensor.to(predicted.device).clone()
            axis = history.get_axis_num("variable")
            for i, name in enumerate(self._prog_vars_e2s):
                if name in self._all_out_variables_e2s:
                    tensor.select(axis, i).copy_(
                        predicted.select(axis, self._all_out_variables_e2s.index(name))
                    )
            x = from_torch(
                tensor,
                coord_array_like(out, {"variable": history["variable"].values}).copy(
                    deep=True
                ),
            )
            x.encoding = deepcopy(out.encoding)
            yield out
