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

import contextlib
from collections import OrderedDict
from collections.abc import Generator, Iterator
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.data.base import DataSource
from earth2studio.lexicon.samudrace import SamudrACELexicon
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils.coords import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

try:
    # Optional dependency: FME
    import cftime
    from fme.ace.data_loading.batch_data import BatchData, PrognosticState
    from fme.core.distributed.distributed import Distributed
    from fme.coupled.data_loading.batch_data import (
        CoupledBatchData,
        CoupledPrognosticState,
    )
    from fme.coupled.stepper import (
        CoupledStepper,
        CoupledStepperConfig,
        load_coupled_stepper,
    )
except ImportError:
    OptionalDependencyFailure("samudrace")
    cftime = Any
    BatchData = Any
    PrognosticState = Any
    CoupledBatchData = Any
    CoupledPrognosticState = Any
    CoupledStepper = Any
    CoupledStepperConfig = Any
    Distributed = Any
    load_coupled_stepper = Any

# fme requires its distributed context to be entered before any model
# computation; when entered here it is held open for the process lifetime
_FME_DISTRIBUTED_STACK = contextlib.ExitStack()


def _to_e2s_name(fme_name: str) -> str:
    """Map an FME variable name to its Earth2Studio name.

    Names outside the SamudrACE lexicon (e.g. from a custom coupled
    checkpoint) are passed through unchanged.

    Parameters
    ----------
    fme_name : str
        FME variable name.

    Returns
    -------
    str
        Earth2Studio variable name.
    """
    return SamudrACELexicon.VOCAB_REVERSE.get(fme_name, fme_name)


def _datetime64_to_cftime(dt64_array: np.ndarray) -> np.ndarray:
    """Convert a np.datetime64 array to cftime proleptic Gregorian datetimes.

    Second precision; supports dates outside the nanosecond-precision
    timestamp range (SamudrACE times are CM4 model years, e.g. year 151).

    Parameters
    ----------
    dt64_array : np.ndarray
        Array of np.datetime64 values.

    Returns
    -------
    np.ndarray
        Object array of cftime.DatetimeProlepticGregorian values with the
        same shape.
    """
    flat = dt64_array.reshape(-1).astype("datetime64[s]").astype(object)
    result = np.fromiter(
        (cftime.DatetimeProlepticGregorian(*dt.timetuple()[:6]) for dt in flat),
        dtype=object,
        count=len(flat),
    )
    return result.reshape(dt64_array.shape)


def _ensure_fme_distributed() -> None:
    """Enter fme's distributed context if it is not already entered.

    fme requires ``Distributed.context()`` to wrap all model computation.
    Applications embedding fme (such as an fme training script) may have
    already entered the context, in which case this is a no-op; otherwise
    the context is entered once and held open for the process lifetime.
    """
    try:
        Distributed.get_instance()
    except RuntimeError:
        _FME_DISTRIBUTED_STACK.enter_context(Distributed.context())


@check_optional_dependencies()
class SamudrACE(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """SamudrACE coupled climate emulator.

    Couples the ACE2 atmosphere emulator with the Samudra ocean emulator,
    driven by FME's ``CoupledStepper``. The atmosphere advances in 6 hour
    steps and the ocean advances once per coupled (5 day) cycle; all coupling
    logic (SST prescription, flux exchange and averaging, ocean-fraction
    prediction, masking) is owned by FME via
    ``CoupledStepper.predict_paired``, which this wrapper calls exactly once
    per coupled cycle.

    The primary interface is :meth:`create_iterator`, which yields the initial
    condition followed by one atmosphere step at a time. Atmosphere output
    fields update on every step; ocean output fields update once per coupled
    cycle and are held constant between cycle boundaries. Ocean diagnostic
    (non-prognostic) fields are NaN until the first cycle boundary, as are
    atmosphere diagnostic fields at the initial condition step.

    Calling the model directly advances one full coupled cycle internally and
    returns the first atmosphere step of that cycle. The returned tensor is
    not a valid coupled restart state (the ocean fields are those of the
    input state); use :meth:`create_iterator` for trajectories.

    Times are CM4 model years (e.g. year 151), which are outside the range of
    nanosecond-precision timestamps; provide time coordinates as
    second-precision ``np.datetime64`` values. Trajectories that cross a
    February 29 of the proleptic Gregorian time coordinate raise an error
    when the forcing is looked up, since that date has no counterpart on the
    no-leap forcing calendar.

    Note
    ----
    For more information see the following references:

    - https://arxiv.org/abs/2509.12490
    - https://huggingface.co/allenai/SamudrACE-CM4-piControl

    Parameters
    ----------
    stepper : CoupledStepper
        FME coupled stepper holding the atmosphere and ocean component
        steppers and the coupling configuration. Both components must be on
        the same latitude-longitude grid.
    forcing_data_source : DataSource
        Data source providing all exogenous forcing variables required by the
        coupled stepper (both atmosphere and ocean), on the model grid and
        with the model's Earth2Studio variable names (see
        ``SamudrACELexicon``).
    """

    def __init__(
        self,
        stepper: CoupledStepper,
        forcing_data_source: DataSource,
    ):
        super().__init__()
        _ensure_fme_distributed()
        self.stepper = stepper
        self.stepper.set_eval()
        self.forcing_data_source = forcing_data_source
        # Register the fme modules so device placement follows the standard
        # torch.nn.Module pattern; also track device via an empty buffer
        self.core_model = stepper.modules
        self.register_buffer("device_buffer", torch.empty(0))

        # The coupled config is reconstructed from the public state dict; it
        # provides the forcing-window data requirements and timesteps
        config = CoupledStepperConfig.from_state(stepper.get_state()["config"])
        requirements = config.get_forcing_window_data_requirements(n_coupled_steps=1)
        self._n_inner_steps = stepper.n_inner_steps
        # Time arithmetic is kept at second precision: SamudrACE times are
        # CM4 model years (e.g. year 151), which overflow nanosecond
        # precision timestamps
        self._dt = np.timedelta64(int(config.atmosphere_timestep.total_seconds()), "s")
        self._dt_ocean = np.timedelta64(int(config.ocean_timestep.total_seconds()), "s")
        self._atmos_forcing_vars = sorted(requirements.atmosphere_requirements.names)
        self._ocean_forcing_vars = sorted(requirements.ocean_requirements.names)
        if not self._ocean_forcing_vars:
            # When every ocean exogenous forcing variable is shared with the
            # atmosphere, the ocean forcing requirements are empty, but the
            # coupled stepper still reads the ocean window's tensors to size
            # the coupled step; the shared variables are supplied so the window
            # is well formed (the stepper takes their values from the
            # atmosphere window)
            self._ocean_forcing_vars = sorted(config.shared_forcing_exogenous_names)
            if not self._ocean_forcing_vars:
                raise ValueError(
                    "The coupled stepper requires no ocean forcing variables, "
                    "so the ocean forcing window cannot be assembled"
                )

        # Variable layouts derived from the component steppers
        self._atmos_prog_vars = sorted(stepper.atmosphere.prognostic_names)
        self._ocean_prog_vars = sorted(stepper.ocean.prognostic_names)
        overlap = set(self._atmos_prog_vars) & set(self._ocean_prog_vars)
        if overlap:
            raise ValueError(
                "Atmosphere and ocean prognostic variable names must be "
                f"disjoint, got overlapping names {sorted(overlap)}"
            )
        self._in_vars = self._atmos_prog_vars + self._ocean_prog_vars

        self._atmos_out_vars = list(stepper.atmosphere.out_names)
        self._ocean_out_vars = list(stepper.ocean.out_names)
        overlap = set(self._atmos_out_vars) & set(self._ocean_out_vars)
        if overlap:
            raise ValueError(
                "Atmosphere and ocean output variable names must be "
                f"disjoint, got overlapping names {sorted(overlap)}"
            )
        self._out_vars = self._atmos_out_vars + self._ocean_out_vars

        # Public coordinates use Earth2Studio variable names, mapped from the
        # checkpoint's FME names through the SamudrACE lexicon (names outside
        # the lexicon pass through unchanged); the FME-name lists above stay
        # positionally aligned for internal tensor packing
        self._in_vars_e2s = [_to_e2s_name(name) for name in self._in_vars]
        self._out_vars_e2s = [_to_e2s_name(name) for name in self._out_vars]
        for names in (self._in_vars_e2s, self._out_vars_e2s):
            if len(set(names)) != len(names):
                raise ValueError(
                    "Earth2Studio variable names mapped from the checkpoint "
                    "must be unique, got duplicates in "
                    f"{sorted(n for n in names if names.count(n) > 1)}"
                )

        # Grid handling: both components must share one lat/lon grid
        dataset_info = stepper.training_dataset_info
        atmos_hc = dataset_info.atmosphere.horizontal_coordinates
        ocean_hc = dataset_info.ocean.horizontal_coordinates
        for hc in (atmos_hc, ocean_hc):
            if not (hasattr(hc, "lat") and hasattr(hc, "lon")):
                raise ValueError(
                    "SamudrACE requires latitude-longitude horizontal "
                    f"coordinates, got {type(hc).__name__}"
                )
        if not (
            torch.equal(atmos_hc.lat, ocean_hc.lat)
            and torch.equal(atmos_hc.lon, ocean_hc.lon)
        ):
            raise ValueError(
                "SamudrACE requires the atmosphere and ocean components to "
                "share the same latitude-longitude grid"
            )
        model_lat = atmos_hc.lat.cpu().numpy()
        # Public Earth2Studio convention is north-to-south latitude; flip
        # internally when the model grid is south-to-north
        if model_lat[0] < model_lat[-1]:
            self._flip_lat = True
            self.lat = model_lat[::-1].copy()
        else:
            self._flip_lat = False
            self.lat = model_lat.copy()
        self.lon = atmos_hc.lon.cpu().numpy().copy()

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model.

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return CoordSystem(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(self._in_vars_e2s, dtype=object),
                "lat": self.lat,
                "lon": self.lon,
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model.

        Parameters
        ----------
        input_coords : CoordSystem
            Input coordinate system to transform into output_coords

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
                "variable": np.array(self._out_vars_e2s, dtype=object),
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
        """Load the default SamudrACE model package from HuggingFace.

        Returns
        -------
        Package
            Package holding the ``allenai/SamudrACE-CM4-piControl``
            checkpoint, pinned to a specific repository revision
        """
        from earth2studio.data.samudrace import HF_REPO_ID, HF_REVISION

        return Package(
            f"hf://{HF_REPO_ID}@{HF_REVISION}",
            cache_options={
                "cache_storage": Package.default_cache("samudrace"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        forcing_data_source: DataSource | None = None,
        scenario: str = "0311",
    ) -> PrognosticMixin:
        """Load SamudrACE prognostic from package.

        Parameters
        ----------
        package : Package
            Package holding the coupled model checkpoint
        forcing_data_source : DataSource | None, optional
            External data source providing all exogenous forcing variables
            required by the coupled stepper, on the model grid and with the
            model's Earth2Studio variable names. If None, a
            ``SamudrACEForcingData`` source for the selected scenario is
            used, by default None
        scenario : str, optional
            Forcing scenario for the default forcing data source, either
            "0151" or "0311"; ignored when ``forcing_data_source`` is
            provided, by default "0311"

        Returns
        -------
        PrognosticMixin
            Prognostic model
        """
        from earth2studio.data.samudrace import SamudrACEForcingData

        _ensure_fme_distributed()
        checkpoint_path = package.resolve("samudrACE_CM4_piControl_ckpt.tar")
        stepper = load_coupled_stepper(checkpoint_path)
        if forcing_data_source is None:
            forcing_data_source = SamudrACEForcingData(scenario=scenario)
        return cls(stepper, forcing_data_source)

    def _flip(self, x: torch.Tensor) -> torch.Tensor:
        """Flip the latitude dimension between public and model orientation.

        Parameters
        ----------
        x : torch.Tensor
            Tensor with latitude as the second-to-last dimension

        Returns
        -------
        torch.Tensor
            Tensor flipped along latitude when the model grid is
            south-to-north, otherwise unchanged
        """
        if self._flip_lat:
            return torch.flip(x, dims=[-2])
        return x

    def _valid_time_array(
        self, coords: CoordSystem, offsets: np.ndarray, n_batch: int
    ) -> xr.DataArray:
        """Build a cftime valid-time array for an fme data window.

        Parameters
        ----------
        coords : CoordSystem
            Input coordinate system providing base times and lead time
        offsets : np.ndarray
            Lead time offsets of the window relative to the input lead time
        n_batch : int
            Size of the Earth2Studio batch dimension

        Returns
        -------
        xr.DataArray
            Valid times with dims [sample, time], where the sample dimension
            flattens (batch, time) in row-major order
        """
        total_offsets = (coords["lead_time"][0] + offsets).astype("timedelta64[s]")
        valid = coords["time"][None, :, None].astype("datetime64[s]") + total_offsets
        valid = np.broadcast_to(valid, (n_batch, len(coords["time"]), len(offsets)))
        valid = valid.reshape(n_batch * len(coords["time"]), len(offsets))
        return xr.DataArray(_datetime64_to_cftime(valid), dims=["sample", "time"])

    def _fetch_forcing_window(
        self,
        coords: CoordSystem,
        variables: list[str],
        offsets: np.ndarray,
        n_batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> BatchData:
        """Fetch an exogenous forcing window and pack it as fme BatchData.

        Parameters
        ----------
        coords : CoordSystem
            Input coordinate system providing base times and lead time
        variables : list[str]
            Forcing variable names to fetch
        offsets : np.ndarray
            Lead time offsets of the window relative to the input lead time
        n_batch : int
            Size of the Earth2Studio batch dimension
        device : torch.device
            Device to place the forcing data on
        dtype : torch.dtype
            Data type of the forcing tensors

        Returns
        -------
        BatchData
            Forcing window with tensors of shape [sample, time, lat, lon]
        """
        time_da = self._valid_time_array(coords, offsets, n_batch)
        if not variables:
            # No exogenous forcing required by this component beyond the
            # window time coordinates
            return BatchData(data={}, time=time_da, horizontal_dims=["lat", "lon"])

        # The forcing data source is called directly (rather than through
        # fetch_data) at second time precision: SamudrACE times are CM4 model
        # years, which overflow nanosecond precision timestamps
        n_time, n_window = len(coords["time"]), len(offsets)
        total_offsets = (coords["lead_time"][0] + offsets).astype("timedelta64[s]")
        valid = coords["time"][:, None].astype("datetime64[s]") + total_offsets
        variables_e2s = [_to_e2s_name(name) for name in variables]
        da = self.forcing_data_source(
            valid.reshape(-1), np.array(variables_e2s, dtype=object)
        )
        if not (
            np.allclose(da.coords["lat"].values, self.lat)
            and np.allclose(da.coords["lon"].values, self.lon)
        ):
            raise ValueError("Forcing data source must provide data on the model grid")
        # [time * window, variable, lat, lon] -> [sample, window, var, lat, lon]
        forcing_x = torch.as_tensor(
            np.ascontiguousarray(da.transpose("time", "variable", "lat", "lon").values),
            device=device,
            dtype=dtype,
        )
        forcing_x = forcing_x.reshape(
            n_time, n_window, len(variables), *forcing_x.shape[-2:]
        )
        forcing_x = (
            forcing_x.unsqueeze(0)
            .expand(n_batch, -1, -1, -1, -1, -1)
            .reshape(n_batch * n_time, n_window, len(variables), *forcing_x.shape[-2:])
        )
        forcing_x = self._flip(forcing_x)
        data = {name: forcing_x[:, :, j] for j, name in enumerate(variables)}
        return BatchData(data=data, time=time_da, horizontal_dims=["lat", "lon"])

    def _state_from_tensor(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> CoupledPrognosticState:
        """Pack an Earth2Studio state tensor into a CoupledPrognosticState.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch, time, 1, variable, lat, lon]
            holding the coupled prognostic variables
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        CoupledPrognosticState
            Initial condition state for the coupled stepper
        """
        b, t, _, _, n_lat, n_lon = x.shape
        x = self._flip(x.reshape(b * t, 1, len(self._in_vars), n_lat, n_lon))
        time = self._valid_time_array(coords, np.array([np.timedelta64(0, "h")]), b)
        var_index = {name: j for j, name in enumerate(self._in_vars)}

        def component_state(names: list[str]) -> PrognosticState:
            """Build a single-component PrognosticState from the input tensor.

            Parameters
            ----------
            names : list[str]
                Prognostic variable names of the component

            Returns
            -------
            PrognosticState
                Component initial condition state
            """
            data = {name: x[:, :, var_index[name]] for name in names}
            return PrognosticState(
                BatchData(data=data, time=time, horizontal_dims=["lat", "lon"])
            )

        return CoupledPrognosticState(
            ocean_data=component_state(self._ocean_prog_vars),
            atmosphere_data=component_state(self._atmos_prog_vars),
        )

    def _run_cycle(
        self,
        state: CoupledPrognosticState,
        coords: CoordSystem,
        n_batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[
        dict[str, torch.Tensor], dict[str, torch.Tensor], CoupledPrognosticState
    ]:
        """Advance the coupled model by one full coupled (ocean) cycle.

        Parameters
        ----------
        state : CoupledPrognosticState
            Coupled initial condition at the cycle start
        coords : CoordSystem
            Input coordinate system at the cycle start
        n_batch : int
            Size of the Earth2Studio batch dimension
        device : torch.device
            Device to run the model on
        dtype : torch.dtype
            Data type of the forcing tensors

        Returns
        -------
        tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], CoupledPrognosticState]
            Atmosphere predictions with tensors of shape
            [sample, n_inner_steps, lat, lon], ocean predictions with tensors
            of shape [sample, 1, lat, lon], and the coupled state at the
            cycle end
        """
        atmos_offsets = np.arange(self._n_inner_steps + 1) * self._dt
        ocean_offsets = np.arange(2) * self._dt_ocean
        forcing = CoupledBatchData(
            ocean_data=self._fetch_forcing_window(
                coords, self._ocean_forcing_vars, ocean_offsets, n_batch, device, dtype
            ),
            atmosphere_data=self._fetch_forcing_window(
                coords, self._atmos_forcing_vars, atmos_offsets, n_batch, device, dtype
            ),
        )
        with torch.inference_mode():
            paired, next_state = self.stepper.predict_paired(state, forcing)
        return (
            dict(paired.atmosphere_data.prediction),
            dict(paired.ocean_data.prediction),
            next_state,
        )

    def _assemble_step(
        self,
        atmos_prediction: dict[str, torch.Tensor],
        inner_step: int,
        ocean_block: torch.Tensor,
        batch_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Assemble one 6 hour output step in the public output layout.

        Parameters
        ----------
        atmos_prediction : dict[str, torch.Tensor]
            Atmosphere predictions with tensors of shape
            [sample, n_inner_steps, lat, lon]
        inner_step : int
            Index of the atmosphere step within the cycle, starting at 0
        ocean_block : torch.Tensor
            Ocean output fields of shape [sample, ocean_variable, lat, lon]
            in model orientation
        batch_shape : tuple[int, int]
            Sizes of the Earth2Studio (batch, time) dimensions

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch, time, 1, variable, lat, lon]
        """
        atmos = torch.stack(
            [atmos_prediction[name][:, inner_step] for name in self._atmos_out_vars],
            dim=1,
        )
        out = torch.cat([atmos, ocean_block], dim=1)
        out = self._flip(out)
        b, t = batch_shape
        return out.reshape(b, t, 1, len(self._out_vars), *out.shape[-2:])

    def _ocean_block_from_prediction(
        self, ocean_prediction: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Stack an ocean prediction into an ocean output block.

        Parameters
        ----------
        ocean_prediction : dict[str, torch.Tensor]
            Ocean predictions with tensors of shape [sample, 1, lat, lon]

        Returns
        -------
        torch.Tensor
            Ocean output fields of shape [sample, ocean_variable, lat, lon]
            in model orientation
        """
        return torch.stack(
            [ocean_prediction[name][:, 0] for name in self._ocean_out_vars], dim=1
        )

    def _initial_ocean_block(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> torch.Tensor:
        """Build the ocean output block held before the first cycle boundary.

        Ocean prognostic fields are copied from the initial condition; ocean
        diagnostic fields are NaN-filled until the first cycle boundary.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch, time, 1, variable, lat, lon]
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        torch.Tensor
            Ocean output fields of shape [sample, ocean_variable, lat, lon]
            in model orientation
        """
        b, t, _, _, n_lat, n_lon = x.shape
        x = self._flip(x.reshape(b * t, 1, len(self._in_vars), n_lat, n_lon))
        var_index = {name: j for j, name in enumerate(self._in_vars)}
        block = torch.full(
            (b * t, len(self._ocean_out_vars), n_lat, n_lon),
            float("nan"),
            device=x.device,
            dtype=x.dtype,
        )
        for j, name in enumerate(self._ocean_out_vars):
            if name in var_index:
                block[:, j] = x[:, 0, var_index[name]]
        return block

    def _build_initial_output(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Construct an initial-condition output matching the output schema.

        Prognostic variables are copied from the initial condition and
        diagnostic (output-only) variables are NaN-filled, so that the
        variable set and tensor shape match subsequent forecast steps.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch, time, 1, variable, lat, lon]
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Initial condition output tensor and coordinate system
        """
        ic_coords = coords.copy()
        ic_coords["variable"] = np.array(self._out_vars_e2s, dtype=object)

        b, t, _, _, n_lat, n_lon = x.shape
        y0 = torch.full(
            (b, t, 1, len(self._out_vars), n_lat, n_lon),
            float("nan"),
            device=x.device,
            dtype=x.dtype,
        )
        var_index_in = {name: j for j, name in enumerate(self._in_vars)}
        for j, name in enumerate(self._out_vars):
            if name in var_index_in:
                y0[:, :, 0, j] = x[:, :, 0, var_index_in[name]]
        return y0, ic_coords

    def _next_input_tensor(
        self,
        atmos_prediction: dict[str, torch.Tensor],
        ocean_prediction: dict[str, torch.Tensor],
        batch_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Build the next cycle-start input tensor from cycle-end predictions.

        Parameters
        ----------
        atmos_prediction : dict[str, torch.Tensor]
            Atmosphere predictions with tensors of shape
            [sample, n_inner_steps, lat, lon]
        ocean_prediction : dict[str, torch.Tensor]
            Ocean predictions with tensors of shape [sample, 1, lat, lon]
        batch_shape : tuple[int, int]
            Sizes of the Earth2Studio (batch, time) dimensions

        Returns
        -------
        torch.Tensor
            Input tensor of shape [batch, time, 1, variable, lat, lon]
        """
        fields = [atmos_prediction[name][:, -1] for name in self._atmos_prog_vars] + [
            ocean_prediction[name][:, -1] for name in self._ocean_prog_vars
        ]
        x = self._flip(torch.stack(fields, dim=1))
        b, t = batch_shape
        return x.reshape(b, t, 1, len(self._in_vars), *x.shape[-2:])

    def _validate_input(self, x: torch.Tensor, coords: CoordSystem) -> None:
        """Validate an input tensor and coordinate system.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Raises
        ------
        ValueError
            If the input tensor or coordinate system is invalid
        """
        if x.ndim != 6:
            raise ValueError(
                "SamudrACE requires input tensor with shape "
                "[batch, time, lead_time, variable, lat, lon], got shape "
                f"{tuple(x.shape)}"
            )
        if len(coords["lead_time"]) != 1:
            raise ValueError(
                "SamudrACE expects a single input lead_time, got "
                f"{len(coords['lead_time'])}"
            )
        # Raises on coordinate handshake failure
        self.output_coords(coords)

    @batch_func()
    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Advance one coupled cycle and return the first atmosphere step.

        A single call advances the coupled model one full coupled (ocean)
        cycle internally and returns the output at the first atmosphere step,
        6 hours past the input state. Ocean output fields in the returned
        tensor are those of the input state (they update only at cycle
        boundaries), so the returned tensor is not a valid coupled restart
        state; use :meth:`create_iterator` for trajectories.

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
        self._validate_input(x, coords)
        b, t = x.shape[0], x.shape[1]
        state = self._state_from_tensor(x, coords)
        atmos_prediction, _, _ = self._run_cycle(state, coords, b, x.device, x.dtype)
        ocean_block = self._initial_ocean_block(x, coords)
        out = self._assemble_step(atmos_prediction, 0, ocean_block, (b, t))
        return out, self.output_coords(coords)

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        """Generator to perform time-integration of the coupled model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Output tensors and coordinate systems, one atmosphere step at a
            time, starting with the initial condition
        """
        coords = coords.copy()
        self._validate_input(x, coords)
        b, t = x.shape[0], x.shape[1]

        # Yield the initial condition (step 0) in the output schema
        y0, y0_coords = self._build_initial_output(x, coords)
        yield y0, y0_coords

        # The coupled prognostic state is held between cycles; a tensor view
        # of the cycle-start state is maintained for the front hook
        state = self._state_from_tensor(x, coords)
        ocean_block = self._initial_ocean_block(x, coords)

        while True:
            hooked_x, coords = self.front_hook(x, coords)
            if hooked_x is not x:
                # A front hook modified the cycle-start state; rebuild the
                # coupled state from the hooked tensor
                x = hooked_x
                state = self._state_from_tensor(x, coords)

            atmos_prediction, ocean_prediction, state = self._run_cycle(
                state, coords, b, x.device, x.dtype
            )
            next_ocean_block = self._ocean_block_from_prediction(ocean_prediction)

            for inner_step in range(self._n_inner_steps):
                is_boundary = inner_step == self._n_inner_steps - 1
                out = self._assemble_step(
                    atmos_prediction,
                    inner_step,
                    next_ocean_block if is_boundary else ocean_block,
                    (b, t),
                )
                out_coords = coords.copy()
                out_coords["variable"] = np.array(self._out_vars_e2s, dtype=object)
                out_coords["lead_time"] = (
                    coords["lead_time"] + (inner_step + 1) * self._dt
                )
                out, out_coords = self.rear_hook(out, out_coords)
                yield out, out_coords

            ocean_block = next_ocean_block
            x = self._next_input_tensor(atmos_prediction, ocean_prediction, (b, t))
            coords = coords.copy()
            coords["lead_time"] = coords["lead_time"] + self._n_inner_steps * self._dt

    def create_iterator(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates a iterator which can be used to perform time-integration of
        the prognostic model. Will return the initial condition first (0th
        step).

        The iterator yields one atmosphere (6 hour) step at a time. The
        coupled stepper runs lazily at each coupled cycle boundary; ocean
        output fields update once per cycle and are held constant between
        boundaries.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator of output tensors and coordinate systems
        """
        yield from self._default_generator(x, coords)
