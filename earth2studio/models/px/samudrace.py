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

import os
from collections import OrderedDict
from collections.abc import Generator, Iterator
from typing import Any

import cftime
import numpy as np
import pandas as pd
import torch
import xarray as xr
from loguru import logger

from earth2studio.data.ace2 import ACE_GRID_LAT, ACE_GRID_LON
from earth2studio.lexicon.samudrace import SamudrACELexicon
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils.coords import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

try:
    from fme.ace.data_loading.batch_data import BatchData, PrognosticState
    from fme.coupled.stepper import CoupledStepper, load_coupled_stepper
    from huggingface_hub import HfFileSystem
except ImportError:
    OptionalDependencyFailure("samudrace")
    BatchData = Any
    PrognosticState = Any
    CoupledStepper = Any
    HfFileSystem = None


# Number of atmosphere 6h steps in one coupled (ocean) 5-day cycle
N_INNER_STEPS = 20

# Valid forcing scenario identifiers
_VALID_FORCING_SCENARIOS = {"0151", "0311"}


def _npdatetime64_to_cftime(dt64_array: np.ndarray) -> np.ndarray:
    """Convert np.datetime64 array to cftime.DatetimeProlepticGregorian array.

    Parameters
    ----------
    dt64_array : np.ndarray
        Input datetime64 array.

    Returns
    -------
    np.ndarray
        Array of cftime.DatetimeProlepticGregorian objects.
    """
    if len(dt64_array.shape) > 1:
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


@check_optional_dependencies()
class SamudrACE(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """SamudrACE coupled atmosphere-ocean prognostic model wrapper.

    SamudrACE is a coupled climate emulator combining ACE2 (atmosphere, 6-hour
    time steps) with Samudra (ocean, 5-day time steps) through a coupler that
    exchanges surface fluxes and SST/sea-ice between the two components. The
    model operates on a 1-degree horizontal grid (180x360) with 8 atmospheric
    model levels and 19 ocean depth levels.

    Use :class:`earth2studio.data.SamudrACEData` to fetch initial-condition
    data for this model.  Forcing data (``DSWRFtoa`` and static fields) are
    downloaded from HuggingFace automatically based on the ``forcing_scenario``
    parameter.

    Note
    ----
    The atmosphere component uses 8 GFDL CM4 **hybrid sigma-pressure** levels
    (indices 0-7).  Pressure at each level is computed as
    ``P(k) = ak(k) + bk(k) * P_surface``.  The coefficients for the 9 level
    interfaces (half-levels) are:

    ====  ========  =======
    k     ak (Pa)   bk
    ====  ========  =======
    0     100       0.000
    1     5579      0.000
    2     8198      0.043
    3     8095      0.164
    4     6465      0.363
    5     4940      0.519
    6     2367      0.768
    7     1156      0.883
    8     0         1.000
    ====  ========  =======

    The ocean component uses 19 layers defined by 20 interface (half-level)
    depths from the GFDL CM4 ocean grid.  The interface depths in metres are:

    ``[0, 5, 15, 30, 50, 80, 130, 200, 300, 450, 650, 900, 1200, 1600,
    2100, 2700, 3500, 4500, 5500, 6500]``

    Parameters
    ----------
    coupled_stepper : CoupledStepper
        FME CoupledStepper instance loaded from a checkpoint.
    forcing_scenario : str, optional
        Forcing scenario identifier, must be one of ``"0151"`` or ``"0311"``.  The
        corresponding forcing NetCDF file is downloaded from HuggingFace and cached
        automatically, by default ``"0311"``.
    dt : numpy.timedelta64, optional
        Atmosphere model timestep, by default 6 hours.

    References
    ----------
    - SamudrACE paper: https://arxiv.org/abs/2509.12490
    - ACE2 code: https://github.com/ai2cm/ace

    Badges
    ------
    region:global class:cm product:temp product:atmos product:ocean year:2025
    gpu:40gb
    """

    _HF_REPO_ID = "allenai/SamudrACE-CM4-piControl"

    def __init__(
        self,
        coupled_stepper: CoupledStepper,
        forcing_scenario: str = "0311",
        dt: np.timedelta64 = np.timedelta64(6, "h"),
    ):
        super().__init__()

        if forcing_scenario not in _VALID_FORCING_SCENARIOS:
            raise ValueError(
                f"forcing_scenario must be one of {sorted(_VALID_FORCING_SCENARIOS)}, "
                f"got {forcing_scenario!r}"
            )

        self.coupled_stepper = coupled_stepper
        self._dt = dt
        self._forcing_scenario = forcing_scenario
        self._forcing_ds: xr.Dataset | None = None
        self._hf_fs: HfFileSystem | None = None
        self.lexicon = SamudrACELexicon

        # Register a buffer for device tracking
        self.register_buffer("device_buffer", torch.empty(0))

        # Atmosphere variable lists
        atm_stepper = self.coupled_stepper.atmosphere
        atm_prog_fme = list(atm_stepper.prognostic_names)
        atm_forcing_fme = list(atm_stepper._input_only_names)
        atm_out_fme = list(atm_stepper.out_names)

        # Ocean variable lists
        ocean_stepper = self.coupled_stepper.ocean
        ocean_prog_fme = list(ocean_stepper.prognostic_names)
        ocean_forcing_fme = list(ocean_stepper._input_only_names)
        ocean_out_fme = list(ocean_stepper.out_names)

        # Store FME variable lists (prognostic = state vars carried forward)
        self._atm_prog_fme = sorted(atm_prog_fme)
        self._atm_forcing_fme = sorted(atm_forcing_fme)
        self._atm_out_fme = atm_out_fme
        self._ocean_prog_fme = sorted(ocean_prog_fme)
        self._ocean_forcing_fme = sorted(ocean_forcing_fme)
        self._ocean_out_fme = ocean_out_fme

        # Build E2S equivalents
        self._atm_prog_e2s = [
            self.lexicon.get_e2s_from_fme(v) for v in self._atm_prog_fme
        ]
        self._ocean_prog_e2s = [
            self.lexicon.get_e2s_from_fme(v) for v in self._ocean_prog_fme
        ]
        self._atm_out_e2s = [
            self.lexicon.get_e2s_from_fme(v) for v in self._atm_out_fme
        ]
        self._ocean_out_e2s = [
            self.lexicon.get_e2s_from_fme(v) for v in self._ocean_out_fme
        ]

        # Combined prognostic and output variable lists (atmosphere first)
        self._prog_e2s = self._atm_prog_e2s + self._ocean_prog_e2s
        self._all_out_e2s = self._atm_out_e2s + self._ocean_out_e2s

        # Forcing vars not produced by ocean (excludes ocean→atm coupled fields)
        ocean_out_set = set(ocean_out_fme)
        self._exogenous_forcing_fme = sorted(
            v for v in atm_forcing_fme if v not in ocean_out_set
        )
        # External forcing: exogenous vars minus coupled fields (ocean_fraction, etc.)
        _coupled_fields = {"ocean_fraction", "sea_ice_fraction"}
        self._external_forcing_fme: list[str] = []
        for v in self._exogenous_forcing_fme:
            if v not in _coupled_fields and v in self.lexicon.VOCAB_REVERSE:
                self._external_forcing_fme.append(v)

        # Grid (same 1° Gaussian grid as ACE2ERA5)
        self.lat = ACE_GRID_LAT
        self.lon = ACE_GRID_LON

        logger.debug(
            "SamudrACE initialized: {} atm prog, {} ocean prog, "
            "{} atm out, {} ocean out, {} external forcing vars",
            len(self._atm_prog_fme),
            len(self._ocean_prog_fme),
            len(self._atm_out_fme),
            len(self._ocean_out_fme),
            len(self._external_forcing_fme),
        )

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
                "lead_time": np.array(
                    [np.timedelta64(0, "h")], dtype="timedelta64[ns]"
                ),
                "variable": np.array(self._prog_e2s, dtype=object),
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
            Input coordinate system to transform into output_coords.

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
                "variable": np.array(self._all_out_e2s),
                "lat": self.lat,
                "lon": self.lon,
            }
        )

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
        """Load default SamudrACE package from HuggingFace.

        Returns
        -------
        Package
            Model package
        """
        return Package(
            "hf://allenai/SamudrACE-CM4-piControl",
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
        forcing_scenario: str = "0311",
        dt: np.timedelta64 = np.timedelta64(6, "h"),
    ) -> PrognosticModel:
        """Load SamudrACE prognostic model from a package.

        Parameters
        ----------
        package : Package
            Package to load the model checkpoint from.
        forcing_scenario : str, optional
            Forcing scenario identifier.  Must be one of ``"0151"`` or
            ``"0311"``, by default ``"0311"``.  The corresponding forcing
            NetCDF file is downloaded from HuggingFace on first use and
            cached locally.
        dt : numpy.timedelta64, optional
            Timestep for advancing lead time coordinates, by default 6 hours.

        Returns
        -------
        PrognosticModel
            SamudrACE prognostic model
        """
        checkpoint_path = package.resolve("samudrACE_CM4_piControl_ckpt.tar")
        coupled_stepper = load_coupled_stepper(checkpoint_path)
        return cls(
            coupled_stepper=coupled_stepper,
            forcing_scenario=forcing_scenario,
            dt=dt,
        )

    def _build_batch_data(
        self,
        data_dict: dict[str, torch.Tensor],
        time_array: np.ndarray,
        n_batch: int,
    ) -> BatchData:
        """Construct an FME BatchData object from a variable dict and time info.

        Parameters
        ----------
        data_dict : dict[str, torch.Tensor]
            Mapping from FME variable names to tensors of shape
            ``[n_batch, n_times, lat, lon]``.
        time_array : np.ndarray
            1-D array of np.datetime64 timestamps of length ``n_times``.
        n_batch : int
            Batch size.

        Returns
        -------
        BatchData
            FME BatchData instance on device.
        """
        times = np.stack([time_array] * n_batch, axis=0)
        time_da = xr.DataArray(_npdatetime64_to_cftime(times), dims=["sample", "time"])
        return BatchData.new_on_device(
            data=data_dict,
            time=time_da,
            horizontal_dims=["lat", "lon"],
        )

    def _prescribe_ic_sst(
        self,
        atmos_ic_state: PrognosticState,
        atmos_forcing: BatchData,
    ) -> PrognosticState:
        """Blend ocean SST into atmosphere IC using ocean_fraction mask."""
        forcing_ic = atmos_forcing.select_time_slice(
            slice(None, self.coupled_stepper.atmosphere.n_ic_timesteps)
        )
        ic_bd = atmos_ic_state.as_batch_data()
        prescribed_data = self.coupled_stepper.atmosphere.prescribe_sst(
            mask_data=forcing_ic.data,
            gen_data=ic_bd.data,
            target_data=forcing_ic.data,
        )
        return PrognosticState(
            BatchData(
                data=prescribed_data,
                time=ic_bd.time,
                labels=ic_bd.labels,
            )
        )

    def _tensor_to_component_states(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[PrognosticState, PrognosticState]:
        """Convert Earth2Studio input tensor to atmosphere and ocean PrognosticStates.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``[batch, time, lead_time, variable, lat, lon]``.
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[PrognosticState, PrognosticState]
            Atmosphere and ocean PrognosticState objects.
        """
        b, t, lt, v, lat, lon = x.shape
        x_flat = x.reshape(b * t, lt, v, lat, lon)

        var_list = list(coords["variable"])
        time_array = coords["time"] + coords["lead_time"]

        # Split atmosphere state
        atm_data: dict[str, torch.Tensor] = {}
        for fme_name in self._atm_prog_fme:
            e2s_name = self.lexicon.get_e2s_from_fme(fme_name)
            idx = var_list.index(e2s_name)
            atm_data[fme_name] = x_flat[:, :, idx, ...]

        # Split ocean state
        ocean_data: dict[str, torch.Tensor] = {}
        for fme_name in self._ocean_prog_fme:
            e2s_name = self.lexicon.get_e2s_from_fme(fme_name)
            idx = var_list.index(e2s_name)
            ocean_data[fme_name] = x_flat[:, :, idx, ...]

        atm_bd = self._build_batch_data(atm_data, time_array, b * t)
        ocean_bd = self._build_batch_data(ocean_data, time_array, b * t)

        return PrognosticState(atm_bd), PrognosticState(ocean_bd)

    def _ensure_forcing_file(self) -> str:
        """Download the forcing NetCDF from HuggingFace if not already cached.

        Returns
        -------
        str
            Local filesystem path to the cached forcing file.
        """
        cache_dir = Package.default_cache("samudrace")
        local_path = os.path.join(
            cache_dir,
            "forcing_data",
            f"forcing_{self._forcing_scenario}.nc",
        )
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            hf_path = (
                f"{self._HF_REPO_ID}/forcing_data/forcing_{self._forcing_scenario}.nc"
            )
            if self._hf_fs is None:
                self._hf_fs = HfFileSystem()
            self._hf_fs.get_file(hf_path, local_path)
        return local_path

    def _open_forcing_ds(self) -> xr.Dataset:
        """Lazily open the forcing NetCDF dataset (cached across calls)."""
        if self._forcing_ds is None:
            forcing_path = self._ensure_forcing_file()
            self._forcing_ds = xr.open_dataset(forcing_path, engine="netcdf4")
        return self._forcing_ds

    def _get_forcing_slice(
        self,
        fme_name: str,
        month: int,
        day: int,
        hour: int,
        n_flat: int,
    ) -> torch.Tensor:
        """Extract a single forcing field from the forcing NetCDF for a given time.

        Parameters
        ----------
        fme_name : str
            FME variable name.
        month : int
            Target month (1-12).
        day : int
            Target day (1-31).
        hour : int
            Target hour (0, 6, 12, 18).
        n_flat : int
            Flattened batch size.

        Returns
        -------
        torch.Tensor
            Forcing field of shape ``[n_flat, lat, lon]``.
        """
        ds = self._open_forcing_ds()
        da = ds[fme_name]
        device = self.device_buffer.device

        if "time" in da.dims:
            # Time-varying field (e.g. DSWRFtoa): match by month/day/hour
            if not hasattr(self, "_forcing_time_index"):
                cf_times = ds["time"].values
                self._forcing_time_index: dict[tuple[int, int, int], int] = {}
                for i, ct in enumerate(cf_times):
                    self._forcing_time_index[(ct.month, ct.day, ct.hour)] = i
            key = (month, day, hour)
            if key not in self._forcing_time_index:
                raise ValueError(
                    f"No forcing data for month={month}, day={day}, hour={hour}. "
                    f"SamudrACE forcing uses a no-leap calendar at 6h resolution."
                )
            idx = self._forcing_time_index[key]
            vals = torch.as_tensor(da.values[idx], dtype=torch.float32, device=device)
        else:
            # Static field (e.g. HGTsfc, land_fraction)
            vals = torch.as_tensor(da.values, dtype=torch.float32, device=device)

        # Broadcast to [n_flat, lat, lon]
        if vals.ndim == 2:
            vals = vals.unsqueeze(0).expand(n_flat, -1, -1)
        return vals

    def _batch_data_to_tensor(
        self,
        data: dict[str, torch.Tensor],
        var_names_fme: list[str],
    ) -> torch.Tensor:
        """Convert FME output dict to a stacked tensor.

        Parameters
        ----------
        data : dict[str, torch.Tensor]
            Dictionary mapping FME variable names to tensors.
        var_names_fme : list[str]
            Ordered list of variable names to stack.

        Returns
        -------
        torch.Tensor
            Stacked tensor of shape ``[batch, 1, 1, variable, lat, lon]``
            (the second singleton is the ``lead_time`` dimension).
        """
        y_list = [data[name] for name in var_names_fme]
        y = torch.stack(y_list, dim=2)
        y = y.unsqueeze(2)
        return y

    @torch.inference_mode()
    def _forward(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        atm_state: PrognosticState,
        ocean_state: PrognosticState,
        atm_flux_accum: list[dict[str, torch.Tensor]],
        step_in_cycle: int,
    ) -> tuple[
        torch.Tensor,
        CoordSystem,
        PrognosticState,
        PrognosticState,
        list[dict[str, torch.Tensor]],
        int,
    ]:
        """Run one 6h atmosphere step and optionally trigger ocean step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.
        atm_state : PrognosticState
            Current atmosphere prognostic state.
        ocean_state : PrognosticState
            Current ocean prognostic state.
        atm_flux_accum : list[dict[str, torch.Tensor]]
            Accumulated atmosphere flux outputs for ocean forcing.
        step_in_cycle : int
            Current step index within the coupled cycle (0..19).

        Returns
        -------
        tuple
            (output_tensor, output_coords, new_atm_state, new_ocean_state,
             new_flux_accum, new_step_in_cycle)
        """
        if len(coords["lead_time"]) != 1:
            raise ValueError("SamudrACE forward expects exactly one lead_time entry.")

        b, t, _, _, lat_len, lon_len = x.shape
        n_flat = b * t

        # Build atmosphere forcing: external data + ocean-sourced fields
        lead_times = np.array(
            [coords["lead_time"][0], coords["lead_time"][0] + self._dt]
        )
        abs_times = [
            (coords["time"][0] + lt if len(coords["time"]) > 0 else lt)
            for lt in lead_times
        ]
        time_array = np.array(abs_times)

        atm_forcing_data: dict[str, torch.Tensor] = {}

        if self._external_forcing_fme:
            for fme_name in self._external_forcing_fme:
                slices = []
                for abs_t in abs_times:
                    ts = pd.Timestamp(abs_t)
                    val = self._get_forcing_slice(
                        fme_name, ts.month, ts.day, ts.hour, n_flat
                    )
                    slices.append(val)
                atm_forcing_data[fme_name] = torch.stack(slices, dim=1)

        # Ocean-sourced forcing: SST, ocean_fraction, sea_ice_fraction
        ocean_bd = ocean_state.as_batch_data()
        coupled_config = self.coupled_stepper._config

        # SST → surface_temperature
        sst_name = coupled_config.sst_name
        if sst_name in ocean_bd.data:
            sst_val = ocean_bd.data[sst_name][:, -1:, ...]
            atm_forcing_data["surface_temperature"] = sst_val.expand(-1, 2, -1, -1)

        # Derive sea_ice_fraction and ocean_fraction from ocean state
        ocean_sourced_names: list[str] = ["surface_temperature"]
        ofp = coupled_config.ocean_fraction_prediction
        if ofp is not None:
            sic_name = ofp.sea_ice_fraction_name
            land_name = ofp.land_fraction_name
            atm_sic_name = ofp.sea_ice_fraction_name_in_atmosphere

            if sic_name in ocean_bd.data:
                osic = torch.nan_to_num(ocean_bd.data[sic_name][:, -1:, ...])
                land_frac = atm_forcing_data[land_name][:, :1, ...]
                sic_atm = osic * (1.0 - land_frac)
                atm_forcing_data[atm_sic_name] = sic_atm.expand(-1, 2, -1, -1)
                ocean_frac = torch.clip(1.0 - land_frac - sic_atm, min=0)
                atm_forcing_data["ocean_fraction"] = ocean_frac.expand(-1, 2, -1, -1)
            ocean_sourced_names += [atm_sic_name, "ocean_fraction"]
        else:
            # Fallback: use direct sea_ice_fraction/ocean_fraction if available
            if "sea_ice_fraction" in ocean_bd.data:
                sic_val = ocean_bd.data["sea_ice_fraction"][:, -1:, ...]
                atm_forcing_data["sea_ice_fraction"] = sic_val.expand(-1, 2, -1, -1)
                ocean_sourced_names.append("sea_ice_fraction")

            if "ocean_fraction" in ocean_bd.data:
                of_val = ocean_bd.data["ocean_fraction"][:, -1:, ...]
                atm_forcing_data["ocean_fraction"] = of_val.expand(-1, 2, -1, -1)
                ocean_sourced_names.append("ocean_fraction")

        # Zero ocean-invalid grid points using mask provider
        ocean_mask_provider = self.coupled_stepper._ocean_mask_provider
        for oname in ocean_sourced_names:
            if oname in atm_forcing_data:
                mask = ocean_mask_provider.get_mask_tensor_for(oname)
                if mask is not None:
                    mask = mask.to(atm_forcing_data[oname].device)
                    mask = mask.expand(atm_forcing_data[oname].shape)
                    atm_forcing_data[oname] = atm_forcing_data[oname].where(
                        mask != 0, 0
                    )

        atm_forcing = self._build_batch_data(atm_forcing_data, time_array, n_flat)

        # Prescribe ocean SST onto atmosphere IC at start of each coupled cycle
        if step_in_cycle == 0:
            atm_state = self._prescribe_ic_sst(atm_state, atm_forcing)

        # Run atmosphere step
        atm_paired, new_atm_state = self.coupled_stepper.atmosphere.predict_paired(
            atm_state, atm_forcing
        )

        # Accumulate flux outputs for ocean forcing
        new_flux_accum = list(atm_flux_accum)
        flux_step: dict[str, torch.Tensor] = {}
        for fme_name in self._ocean_forcing_fme:
            if fme_name in atm_paired.prediction:
                flux_step[fme_name] = atm_paired.prediction[fme_name][:, -1, ...]
        if flux_step:
            new_flux_accum.append(flux_step)

        new_step = step_in_cycle + 1
        new_ocean_state = ocean_state

        # At the end of a coupled cycle, run the ocean step
        if new_step >= N_INNER_STEPS:
            new_ocean_state = self._run_ocean_step(
                new_ocean_state, new_atm_state, new_flux_accum, coords
            )
            new_flux_accum = []
            new_step = 0

        # Build output tensor: concatenate atmosphere + ocean outputs
        atm_out = self._batch_data_to_tensor(atm_paired.prediction, self._atm_out_fme)
        # For ocean: use latest ocean state (may have just been updated)
        ocean_data = new_ocean_state.as_batch_data().data
        ocean_out_dict: dict[str, torch.Tensor] = {}
        for fme_name in self._ocean_out_fme:
            if fme_name in ocean_data:
                ocean_out_dict[fme_name] = ocean_data[fme_name][:, -1:, ...]
            else:
                ocean_out_dict[fme_name] = torch.full(
                    (n_flat, 1, lat_len, lon_len),
                    float("nan"),
                    device=self.device_buffer.device,
                )
        ocean_out = self._batch_data_to_tensor(ocean_out_dict, self._ocean_out_fme)

        # Concatenate along variable dimension
        y = torch.cat([atm_out, ocean_out], dim=3)
        # Reshape back: [b*t, 1, 1, var, lat, lon] -> [b, t, 1, var, lat, lon]
        y = y.reshape(b, t, 1, len(self._all_out_e2s), lat_len, lon_len)

        out_coords = self.output_coords(coords)
        return y, out_coords, new_atm_state, new_ocean_state, new_flux_accum, new_step

    def _run_ocean_step(
        self,
        ocean_state: PrognosticState,
        atm_state: PrognosticState,
        flux_accum: list[dict[str, torch.Tensor]],
        coords: CoordSystem,
    ) -> PrognosticState:
        """Run a single ocean step using time-averaged atmosphere fluxes.

        Parameters
        ----------
        ocean_state : PrognosticState
            Current ocean state.
        atm_state : PrognosticState
            Current atmosphere state (used for static fields like land_fraction).
        flux_accum : list[dict[str, torch.Tensor]]
            List of atmosphere flux dicts, one per inner atmosphere step.
        coords : CoordSystem
            Current coordinate system.

        Returns
        -------
        PrognosticState
            Updated ocean state after 5-day step.
        """
        if not flux_accum:
            return ocean_state

        # Time-average the accumulated atmosphere fluxes
        avg_forcing: dict[str, torch.Tensor] = {}
        for key in flux_accum[0]:
            stacked = torch.stack([f[key] for f in flux_accum], dim=1)
            avg_forcing[key] = stacked.mean(dim=1, keepdim=True)

        # Build 2-timestep forcing: IC (NaN) + forward step (averaged flux)
        ocean_forcing_data: dict[str, torch.Tensor] = {}
        n_flat = list(avg_forcing.values())[0].shape[0]
        for key, val in avg_forcing.items():
            nan_pad = torch.full_like(val, float("nan"))
            ocean_forcing_data[key] = torch.cat([nan_pad, val], dim=1)

        # Add land_fraction (static field needed by ocean stepper)
        lf_val = self._get_forcing_slice("land_fraction", 1, 1, 0, n_flat)
        ocean_forcing_data["land_fraction"] = lf_val.unsqueeze(1).expand(-1, 2, -1, -1)

        # Build time array for ocean forcing (2 timesteps over 5-day window)
        ocean_dt = np.timedelta64(5, "D")
        time_array = np.array(
            (
                [
                    coords["time"][0] + coords["lead_time"][0],
                    coords["time"][0] + coords["lead_time"][0] + ocean_dt,
                ]
                if len(coords["time"]) > 0
                else [coords["lead_time"][0], coords["lead_time"][0] + ocean_dt]
            ),
        )

        ocean_forcing = self._build_batch_data(ocean_forcing_data, time_array, n_flat)

        _, new_ocean_state = self.coupled_stepper.ocean.predict_paired(
            ocean_state, ocean_forcing
        )
        return new_ocean_state

    def _build_initial_output(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Construct initial-condition output tensor matching model output schema.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Initial condition output tensor and coordinate system.
        """
        ic_coords = coords.copy()
        ic_coords["variable"] = np.array(self._all_out_e2s, dtype=object)

        b, t, _, _, lat_len, lon_len = x.shape
        v_out = len(self._all_out_e2s)
        y0 = torch.full(
            (b, t, 1, v_out, lat_len, lon_len),
            float("nan"),
            device=x.device,
            dtype=x.dtype,
        )

        # Copy prognostic variables from input to output
        var_to_idx_out = {v: i for i, v in enumerate(self._all_out_e2s)}
        var_to_idx_in = {v: i for i, v in enumerate(self._prog_e2s)}
        for v in self._prog_e2s:
            if v in var_to_idx_out:
                y0[:, :, 0, var_to_idx_out[v], ...] = x[:, :, 0, var_to_idx_in[v], ...]

        return y0, ic_coords

    @batch_func()
    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs one prognostic step (6h atmosphere) using FME CoupledStepper.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system 6 hours in the future.
        """
        x = x.to(self.device_buffer.device)

        # Initialize component states from input tensor
        atm_state, ocean_state = self._tensor_to_component_states(x, coords)

        y, out_coords, _, _, _, _ = self._forward(
            x, coords, atm_state, ocean_state, [], 0
        )
        return y, out_coords

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        """Generator to perform coupled time-integration of SamudrACE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system at each 6h time step.
        """
        coords = coords.copy()
        x = x.to(self.device_buffer.device)

        # Yield initial condition
        ic_tensor, ic_coords = self._build_initial_output(x, coords)
        yield ic_tensor, ic_coords

        # Initialize component states
        atm_state, ocean_state = self._tensor_to_component_states(x, coords)
        atm_flux_accum: list[dict[str, torch.Tensor]] = []
        step_in_cycle = 0

        while True:
            x, coords = self.front_hook(x, coords)

            (
                out,
                out_coords,
                atm_state,
                ocean_state,
                atm_flux_accum,
                step_in_cycle,
            ) = self._forward(
                x, coords, atm_state, ocean_state, atm_flux_accum, step_in_cycle
            )

            out, out_coords = self.rear_hook(out, out_coords)
            yield out, out_coords.copy()

            # Update x with latest prognostic variables from output
            x_next = x.clone()
            var_to_idx_out = {v: i for i, v in enumerate(self._all_out_e2s)}
            var_to_idx_in = {v: i for i, v in enumerate(self._prog_e2s)}
            for v in self._prog_e2s:
                if v in var_to_idx_out:
                    x_next[:, :, 0, var_to_idx_in[v], ...] = out[
                        :, :, 0, var_to_idx_out[v], ...
                    ]
            x = x_next

            coords = coords.copy()
            coords["lead_time"] = coords["lead_time"] + self._dt

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates an iterator for time-integration of SamudrACE.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        coords : CoordSystem
            Input coordinate system.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            Predicted state and coordinates at each 6h time step.
        """
        yield from self._default_generator(x, coords)
