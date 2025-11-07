# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
from collections.abc import Generator, Iterator
from typing import Any

import cftime
import numpy as np
import pandas as pd
import torch
import xarray as xr

from earth2studio.data import ACE2ERA5Data
from earth2studio.data.ace2 import ACE_GRID_LAT, ACE_GRID_LON
from earth2studio.data.base import DataSource
from earth2studio.data.utils import fetch_data
from earth2studio.lexicon.ace import ACELexicon
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils.coords import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.interp import LatLonInterpolation
from earth2studio.utils.type import CoordSystem

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


@check_optional_dependencies("ace2")
class ACE2ERA5(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """ACE2-ERA5 prognostic model wrapper.

    ACE2 (Ai2 Climate Emulator v2) is a 450M-parameter autoregressive emulator
    with 6-hour time steps, 1-degree horizontal resolution, and eight vertical
    layers that exactly conserves global dry air mass and moisture and can be
    stepped stably for arbitrarily many steps at about 1500 simulated years
    per wall-clock day. ACE2-ERA5 was trained on the ERA5 dataset and requires
    forcing data during rollout (see `forcing_data_source` parameter). This
    wrapper makes use of the ``fme`` package to run model forward passes.

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

    Warning
    ----------
    This model may only be used with input data on the GPU device that the model was
    loaded on. Specifically, the data must be on the same device as whatever
    ``torch.cuda.current_device()`` was set to when the model package was loaded.
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

        self._forcing_vars_fme = list(self.stepper._input_only_names)
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
        self._prog_vars_fme = list(self.stepper.prognostic_names)
        self._prog_vars_e2s = [
            self.lexicon.get_e2s_from_fme(v) for v in self._prog_vars_fme
        ]

        # External forcing data source
        self.forcing_data_source = forcing_data_source

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

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        coords = CoordSystem(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array(
                    [np.timedelta64(0, "h")], dtype="timedelta64[ns]"
                ),
                "variable": np.array(self._prog_vars_e2s, dtype=object),
                "lat": self.lat,
                "lon": self.lon,
            }
        )
        return coords

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
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
        output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([self._dt]),
                "variable": np.array(self._all_out_variables_e2s),
                "lat": self.lat,
                "lon": self.lon,
            }
        )
        if input_coords is None:
            return output_coords

        test_coords = input_coords.copy()
        test_coords["lead_time"] = (
            test_coords["lead_time"] - input_coords["lead_time"][0]
        )
        target_input_coords = self.input_coords()
        for i, key in enumerate(target_input_coords):
            if key not in ["batch", "time"]:
                handshake_dim(test_coords, key, i)
                handshake_coords(test_coords, target_input_coords, key)

        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]

        output_coords["lead_time"] = (
            input_coords["lead_time"][0] + output_coords["lead_time"]
        )
        return output_coords

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
        if x.ndim != 6:
            raise ValueError(
                "ACE2ERA5 requires input tensor with shape [batch, time, lead_time, variable, lat, lon]"
            )

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
        times_forcing = np.stack(
            [coords["time"] + forcing_coords["lead_time"]] * b, axis=0
        )  # includes both time steps
        times_state = np.stack(
            [coords["time"] + coords["lead_time"]] * b, axis=0
        )  # only includes current (init) time
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
        if len(coords["lead_time"]) != 1:
            raise ValueError("ACE2ERA5 forward expects one input lead_time entry [0h].")

        # Pull forcing data (which is required at both input and output lead times)
        lead_times = np.array(
            [coords["lead_time"][0], coords["lead_time"][0] + self._dt]
        )
        forcing_x, forcing_coords = fetch_data(
            self.forcing_data_source,
            time=coords["time"],
            lead_time=lead_times,
            variable=self._forcing_vars_e2s,
        )

        # Interp to proper coords and stack along batch dimension as required
        if self.needs_regrid:
            forcing_x = self.regridder(forcing_x.to(x.device))
            forcing_coords["lat"] = coords["lat"]
            forcing_coords["lon"] = coords["lon"]
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
        out_coords = self.output_coords(coords)
        return y, out_coords

    def _build_initial_output(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Construct an initial-condition output tensor matching model output schema.

        Fills output-only variables with NaN and copies prognostic variables from the
        provided initial condition tensor so that variable set and tensor shape match
        subsequent forecast steps in iterator mode.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Initial condition output tensor and coordinate system
        """

        # Prepare coords: keep time and lead_time (0h) from input, but use output variables
        ic_coords = coords.copy()
        ic_coords["variable"] = np.array(self._all_out_variables_e2s, dtype=object)

        # Allocate output filled with NaNs [batch, time, lead_time=1, variable_out, lat, lon]
        b, t, _, _, lat, lon = x.shape
        v_out = len(self._all_out_variables_e2s)
        y0 = torch.full(
            (b, t, 1, v_out, lat, lon),
            float("nan"),
            device=x.device,
            dtype=x.dtype,
        )

        # Map prognostic variables from input into output variable positions
        var_to_idx_out = {v: i for i, v in enumerate(self._all_out_variables_e2s)}
        var_to_idx_in = {v: i for i, v in enumerate(self._prog_vars_e2s)}
        for v in self._prog_vars_e2s:
            if v in var_to_idx_out:
                y0[:, :, 0, var_to_idx_out[v], ...] = x[:, :, 0, var_to_idx_in[v], ...]

        return y0, ic_coords

    @batch_func()
    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs one prognostic step using fme predict_paired API.

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
        return self._forward(x, coords)

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        """Generator to perform time-integration of ACE2ERA5.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        Generator[tuple[torch.Tensor, CoordSystem]]
            Generator of output tensors and coordinate systems
        """
        coords = coords.copy()

        # Yield the initial condition (t=0 step) in output schema
        # Output-only variables will be NaN-filled for this step
        ic_tensor, ic_coords = self._build_initial_output(x, coords)
        yield ic_tensor, ic_coords

        # Setup rolling state for subsequent steps: x at next init lead_time uses previous out as state
        while True:
            # Front hook
            x, coords = self.front_hook(x, coords)

            # Forward one step from current state
            out, out_coords = self._forward(x, coords)

            # Rear hook
            out, out_coords = self.rear_hook(out, out_coords)

            yield out, out_coords.copy()

            # Build next input by replacing the prognostic slice in x with the previous output
            # x shape: [batch, time, lead_time=1, variable, lat, lon]
            x_next = x.clone()
            var_to_idx_out = {v: i for i, v in enumerate(self._all_out_variables_e2s)}
            var_to_idx_in = {v: i for i, v in enumerate(self._prog_vars_e2s)}
            for v in self._prog_vars_e2s:
                if v in var_to_idx_out:
                    x_next[:, :, 0, var_to_idx_in[v], ...] = out[
                        :, :, 0, var_to_idx_out[v], ...
                    ]

            x = x_next

            # Advance base time for next step; keep lead_time at [0h]
            coords = coords.copy()
            coords["lead_time"] = coords["lead_time"] + self._dt

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates an iterator to perform time-integration of ACE2ERA5.

        Yields the first forecast step, then continues autoregressively by feeding
        previous outputs as the next prognostic state while fetching/using external
        forcings under the hood via _forward.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator of output tensors and coordinate systems
        """
        yield from self._default_generator(x, coords)
