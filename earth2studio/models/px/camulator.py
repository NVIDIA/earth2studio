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
from collections.abc import Generator, Iterator

import numpy as np
import torch
import xarray as xr

from earth2studio.data.base import DataSource
from earth2studio.data.camulator import (
    CAMULATOR_GRID_LAT,
    CAMULATOR_GRID_LON,
    CAMULATOR_HF_REVISION,
    CAMulatorForcing,
)
from earth2studio.lexicon.camulator import (
    CAMULATOR_LEVELS,
    CAMULATOR_STEP_SECONDS,
    CAMulatorLexicon,
)
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.nn.camulator import CamulatorNet
from earth2studio.models.nn.camulator_physics import (
    global_energy_fix,
    global_mass_fix,
    global_water_fix,
    grid_area,
    tracer_clip,
    wind_artifact_filter,
)
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import check_optional_dependencies
from earth2studio.utils.type import CoordSystem

# Channel layout of the CAMulator state (prognostic) tensor
_LEVEL_VARS = ["u", "v", "t", "qtot"]
PROGNOSTIC_VARIABLES = [
    f"{var}{k}k" for var in _LEVEL_VARS for k in range(CAMULATOR_LEVELS)
] + ["sp", "t2m"]
DIAGNOSTIC_VARIABLES = [
    "tp06",
    "skt",
    "hcc",
    "lcc",
    "mcc",
    "iews",
    "inss",
    "ws10m",
    "e06",
    "msdwswrf",
    "msdwlwrf",
    "msuwswrf",
    "msuwlwrf",
    "msshf",
    "mslhf",
    "mtuwswrf",
    "mtuwlwrf",
]
OUTPUT_VARIABLES = PROGNOSTIC_VARIABLES + DIAGNOSTIC_VARIABLES
FORCING_VARIABLES = ["mtdwswrf", "sst", "sic", "global_mean_co2"]
STATIC_VARIABLES = ["z_norm", "LANDM_COSLAT"]

_N_STATE = len(PROGNOSTIC_VARIABLES)  # 130
_N_OUT = len(OUTPUT_VARIABLES)  # 147
_U = slice(0, 32)
_V = slice(32, 64)
_T = slice(64, 96)
_Q = slice(96, 128)
_PS = 128
_OUT = {v: i for i, v in enumerate(OUTPUT_VARIABLES)}
_SOLIN_IN = _N_STATE + len(STATIC_VARIABLES)  # 132 in the 136-channel input

# Tracer fixer channels: (output channel, CESM name, min, max). CREDIT resolves
# the per-channel statistics with ``mean_ds[name].values.flatten()[0]`` which,
# for the per-level Qtot statistics, applies the level-0 mean/std to all 32 Qtot
# channels. The shipped checkpoints were evaluated with that behaviour, so it is
# reproduced here.
_TRACERS: list[tuple[int, str, float, float]] = [
    (c, "Qtot", 0.0, float("inf")) for c in range(_Q.start, _Q.stop)
] + [
    (_OUT["tp06"], "PRECT", 0.0, float("inf")),
    (_OUT["hcc"], "CLDHGH", 0.0, 1.0),
    (_OUT["lcc"], "CLDLOW", 0.0, 1.0),
    (_OUT["mcc"], "CLDMED", 0.0, 1.0),
    (_OUT["msdwswrf"], "FSDS_J", 0.0, float("inf")),
    (_OUT["msuwswrf"], "FSUS", 0.0, float("inf")),
    (_OUT["mtuwswrf"], "FSUTOA", 0.0, float("inf")),
    (_OUT["mtuwlwrf"], "FLUT", 0.0, float("inf")),
]

_MEAN_FILE = "normalization/mean_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc"
_STD_FILE = "normalization/std_6h_Coupled_1980_2014_32lev_1.0deg_ERA5scaled_F32_Qtot_Mixed_Modal.nc"
_STATICS_FILE = "normalization/statics_b_credit_runs_f32_02.nc"
_PHYSICS_FILE = (
    "normalization/b.e21.CREDIT_climate.statics_1.0deg_32levs_latlon_F32_hyai_fixed.nc"
)


def _stats_field(ds: xr.Dataset, e2s_name: str, shape: tuple[int, int]) -> np.ndarray:
    """Broadcast the normalization statistic of one output/forcing variable to the
    model grid (statistics are scalars, per-level values or lat/lon maps)."""
    cesm, _ = CAMulatorLexicon[e2s_name]
    if "::" in cesm:
        name, level = cesm.split("::")
        value = np.asarray(ds[name].values)[int(level)]
    else:
        value = np.asarray(ds[cesm].values)
    return np.broadcast_to(np.asarray(value, dtype=np.float64), shape).copy()


class CAMulator(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """CAMulator: NSF NCAR's autoregressive emulator of the CAM6 atmosphere,
    trained in the CREDIT framework. A CrossFormer encoder-decoder advances a 1
    degree (192x288), 32 hybrid-sigma-level atmospheric state by 6 hours given
    prescribed sea-surface temperature, sea-ice fraction, TOA insolation and CO2
    forcing, for climate-length rollouts. The inference-time post-processing of the
    CREDIT toolbox is reproduced: tracer clipping, the jet wind-artifact filter and
    the global dry-air mass, water and total-energy conservation fixers.

    The 130 prognostic variables are the model input; the output additionally
    holds 17 output-only diagnostics (precipitation, surface temperature, cloud
    fractions, surface stresses, 10 m wind speed, evaporation and radiative/heat
    fluxes). In :meth:`create_iterator` the initial condition is yielded in the
    output schema with the diagnostics NaN-filled, and only the prognostic slice is
    fed back at each step. Vertical levels use the ``{var}{k}k`` naming with ``k``
    the CAMulator hybrid level index (0 = top of model, 31 = lowest layer).
    Fluxes CAMulator accumulates over the 6 h step (J m-2) are returned as mean
    rates (W m-2); precipitation and evaporation are 6 h accumulations (m).

    Note
    ----
    Forcing is read at the input valid time from ``forcing_data_source`` on the
    CAMulator grid. The default :class:`~earth2studio.data.CAMulatorForcing`
    serves the shipped climatological (cyclic) year; the forcing files use a
    365-day calendar, so leap days reuse the 28 February forcing.

    Note
    ----
    For more information see:

    - https://arxiv.org/abs/2504.06007
    - https://huggingface.co/willychap/camulator
    - https://github.com/NCAR/miles-credit

    Parameters
    ----------
    core_model : torch.nn.Module
        CAMulator CrossFormer network mapping ``(batch, 136, 1, lat, lon)`` to
        ``(batch, 147, 1, lat, lon)`` in normalized units.
    center : torch.Tensor
        Normalization mean of the 147 output channels, shape (147, lat, lon).
    scale : torch.Tensor
        Normalization std of the 147 output channels, shape (147, lat, lon).
    forcing_center : torch.Tensor
        Normalization mean of the 4 forcing channels, shape (4,).
    forcing_scale : torch.Tensor
        Normalization std of the 4 forcing channels, shape (4,).
    tracer_center : torch.Tensor
        Normalization mean used by the tracer clipping, one per tracer channel.
    tracer_scale : torch.Tensor
        Normalization std used by the tracer clipping, one per tracer channel.
    statics : torch.Tensor
        Static input fields ``z_norm`` and ``LANDM_COSLAT``, shape (2, lat, lon).
    hyai : torch.Tensor
        Hybrid ``a`` interface coefficients (Pa), shape (33,).
    hybi : torch.Tensor
        Hybrid ``b`` interface coefficients, shape (33,).
    area : torch.Tensor
        Grid-cell area (m^2), shape (lat, lon).
    phis : torch.Tensor
        Surface geopotential (m^2 s-2), shape (lat, lon).
    forcing_data_source : DataSource, optional
        Data source providing ``mtdwswrf``, ``sst``, ``sic`` and
        ``global_mean_co2`` on the CAMulator grid, by default
        ``CAMulatorForcing()``.
    conservation_fixers : bool, optional
        Apply the global mass, water and energy fixers, by default True.
    wind_filter : bool, optional
        Apply the wind-artifact filter, by default True.

    Badges
    ------
    region:global class:climate product:wind product:precip product:temp
    product:atmos year:2025 gpu:24gb provider:ncar backend:pytorch
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        center: torch.Tensor,
        scale: torch.Tensor,
        forcing_center: torch.Tensor,
        forcing_scale: torch.Tensor,
        tracer_center: torch.Tensor,
        tracer_scale: torch.Tensor,
        statics: torch.Tensor,
        hyai: torch.Tensor,
        hybi: torch.Tensor,
        area: torch.Tensor,
        phis: torch.Tensor,
        forcing_data_source: DataSource | None = None,
        conservation_fixers: bool = True,
        wind_filter: bool = True,
    ):
        super().__init__()
        self.model = core_model
        # CREDIT applies the statistics in float32, except the gridded surface
        # pressure statistics and the forcing statistics, which stay in float64;
        # the same precision is kept here so results match bit for bit.
        self.register_buffer("center", center.float())
        self.register_buffer("scale", scale.float())
        self.register_buffer("ps_center", center[_PS].double())
        self.register_buffer("ps_scale", scale[_PS].double())
        self.register_buffer("forcing_center", forcing_center.double())
        self.register_buffer("forcing_scale", forcing_scale.double())
        self.register_buffer("tracer_center", tracer_center.float())
        self.register_buffer("tracer_scale", tracer_scale.float())
        self.register_buffer("statics", statics.float())
        self.register_buffer("hyai", hyai.float())
        self.register_buffer("hybi", hybi.float())
        self.register_buffer("area", area.float())
        self.register_buffer("phis", phis.float())
        self.register_buffer("device_buffer", torch.empty(0))

        if forcing_data_source is None:
            forcing_data_source = CAMulatorForcing(verbose=False)
        self.forcing_data_source = forcing_data_source
        self.conservation_fixers = conservation_fixers
        self.wind_filter = wind_filter
        self._dt = np.timedelta64(6, "h")
        self._n_seconds = CAMULATOR_STEP_SECONDS

        self._output_mods = [CAMulatorLexicon[v][1] for v in OUTPUT_VARIABLES]

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(PROGNOSTIC_VARIABLES),
                "lat": CAMULATOR_GRID_LAT.copy(),
                "lon": CAMULATOR_GRID_LON.copy(),
            }
        )

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        """Output coordinate system of the prognostic model

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
                "variable": np.array(OUTPUT_VARIABLES),
                "lat": CAMULATOR_GRID_LAT.copy(),
                "lon": CAMULATOR_GRID_LON.copy(),
            }
        )
        if input_coords is None:
            return output_coords

        target_input_coords = self.input_coords()
        handshake_dim(input_coords, "lead_time", 2)
        handshake_dim(input_coords, "variable", 3)
        handshake_dim(input_coords, "lat", 4)
        handshake_dim(input_coords, "lon", 5)
        handshake_coords(input_coords, target_input_coords, "variable")
        handshake_coords(input_coords, target_input_coords, "lat")
        handshake_coords(input_coords, target_input_coords, "lon")
        if len(input_coords["lead_time"]) != 1:
            raise ValueError("CAMulator expects a single input lead_time")

        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]
        output_coords["lead_time"] = input_coords["lead_time"] + self._dt
        return output_coords

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the default CAMulator package from HuggingFace

        Returns
        -------
        Package
            Package
        """
        return Package(
            f"hf://willychap/camulator@{CAMULATOR_HF_REVISION}",
            cache_options={
                "cache_storage": Package.default_cache("camulator"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        checkpoint: str = "checkpoint.pt00069.pt",
        forcing_data_source: DataSource | None = None,
        conservation_fixers: bool = True,
        wind_filter: bool = True,
    ) -> PrognosticModel:
        """Load prognostic model from package

        Parameters
        ----------
        package : Package
            Package to load model from
        checkpoint : str, optional
            Checkpoint file in the package. The HuggingFace repository hosts
            several training epochs (``checkpoint.pt000NN.pt``); epoch 69 is the
            model card default, by default "checkpoint.pt00069.pt"
        forcing_data_source : DataSource, optional
            Forcing data source, by default ``CAMulatorForcing()``
        conservation_fixers : bool, optional
            Apply the global mass, water and energy fixers, by default True
        wind_filter : bool, optional
            Apply the wind-artifact filter, by default True

        Returns
        -------
        PrognosticModel
            Prognostic model
        """
        mean_ds = xr.open_dataset(package.resolve(_MEAN_FILE)).load()
        std_ds = xr.open_dataset(package.resolve(_STD_FILE)).load()
        statics_ds = xr.open_dataset(package.resolve(_STATICS_FILE))
        physics_ds = xr.open_dataset(package.resolve(_PHYSICS_FILE))

        shape = (len(CAMULATOR_GRID_LAT), len(CAMULATOR_GRID_LON))
        center = torch.from_numpy(
            np.stack([_stats_field(mean_ds, v, shape) for v in OUTPUT_VARIABLES])
        )
        scale = torch.from_numpy(
            np.stack([_stats_field(std_ds, v, shape) for v in OUTPUT_VARIABLES])
        )
        forcing_center = torch.tensor(
            [float(mean_ds[CAMulatorLexicon[v][0]]) for v in FORCING_VARIABLES],
            dtype=torch.float64,
        )
        forcing_scale = torch.tensor(
            [float(std_ds[CAMulatorLexicon[v][0]]) for v in FORCING_VARIABLES],
            dtype=torch.float64,
        )
        tracer_center = torch.tensor(
            [
                float(np.asarray(mean_ds[name].values).flatten()[0])
                for _, name, _, _ in _TRACERS
            ]
        )
        tracer_scale = torch.tensor(
            [
                float(np.asarray(std_ds[name].values).flatten()[0])
                for _, name, _, _ in _TRACERS
            ]
        )
        statics = torch.from_numpy(
            np.stack(
                [statics_ds[v].values.astype(np.float32) for v in STATIC_VARIABLES]
            )
        )
        hyai = torch.from_numpy(physics_ds["hyai"].values.astype(np.float32))
        hybi = torch.from_numpy(physics_ds["hybi"].values.astype(np.float32))
        lat2d = torch.from_numpy(physics_ds["lat2d"].values.astype(np.float32))
        lon2d = torch.from_numpy(physics_ds["lon2d"].values.astype(np.float32))
        phis = torch.from_numpy(physics_ds["PHIS"].values.astype(np.float32))

        core_model = CamulatorNet()
        # CREDIT checkpoints bundle optimizer/scheduler state alongside the weights
        state = torch.load(
            package.resolve(checkpoint),
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )
        state_dict = {
            k: v
            for k, v in state["model_state_dict"].items()
            if not k.startswith("postblock.")
        }
        core_model.load_state_dict(state_dict, strict=True)
        core_model.eval()

        return cls(
            core_model,
            center=center,
            scale=scale,
            forcing_center=forcing_center,
            forcing_scale=forcing_scale,
            tracer_center=tracer_center,
            tracer_scale=tracer_scale,
            statics=statics,
            hyai=hyai,
            hybi=hybi,
            area=grid_area(lat2d, lon2d),
            phis=phis,
            forcing_data_source=forcing_data_source,
            conservation_fixers=conservation_fixers,
            wind_filter=wind_filter,
        )

    def _fetch_forcing(self, coords: CoordSystem, device: torch.device) -> torch.Tensor:
        """Normalized forcing at the input valid times in model (south-to-north)
        orientation, shape (time, 4, lat, lon)."""
        valid_times = np.asarray(coords["time"]) + coords["lead_time"][0]
        da = self.forcing_data_source(valid_times, FORCING_VARIABLES)
        da = da.transpose("time", "variable", "lat", "lon")
        if not np.allclose(da["lat"].values, CAMULATOR_GRID_LAT) or not np.allclose(
            da["lon"].values, CAMULATOR_GRID_LON
        ):
            raise ValueError("CAMulator forcing data must be on the CAMulator grid")
        forcing = torch.from_numpy(np.ascontiguousarray(da.values)).to(device).float()
        forcing = torch.flip(forcing, dims=(-2,))
        forcing[:, 3] = forcing[:, 3] * 1.0e-6  # ppm -> mol mol-1
        # Normalized in float64 like CREDIT, then cast at the network input
        forcing = (
            forcing.double() - self.forcing_center.view(1, -1, 1, 1)
        ) / self.forcing_scale.view(1, -1, 1, 1)
        return forcing.float()

    def _denorm(self, y: torch.Tensor, ch: slice | int) -> torch.Tensor:
        return y[:, ch] * self.scale[ch] + self.center[ch]

    def _renorm(self, y: torch.Tensor, ch: slice | int) -> torch.Tensor:
        return (y - self.center[ch]) / self.scale[ch]

    def _postprocess(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply the CREDIT inference post-processing chain to a normalized
        prediction ``y`` (batch, 147, lat, lon) given the normalized input ``x``
        (batch, 136, lat, lon)."""
        y = tracer_clip(
            y,
            [c for c, _, _, _ in _TRACERS],
            self.tracer_center,
            self.tracer_scale,
            [lo for _, _, lo, _ in _TRACERS],
            [hi for _, _, _, hi in _TRACERS],
        )
        if self.wind_filter:
            y = wind_artifact_filter(
                y,
                levels=CAMULATOR_LEVELS,
                var_offsets=(_U.start, _V.start, _T.start, _Q.start),
                mask_var_offsets=(_U.start, _V.start),
            )
        if not self.conservation_fixers:
            return y

        n = self._n_seconds
        sp_pred = global_mass_fix(
            self._denorm(x, _PS),
            self._denorm(x, _Q),
            self._denorm(y, _PS),
            self._denorm(y, _Q),
            self.hyai,
            self.hybi,
            self.area,
        )
        y[:, _PS] = self._renorm(sp_pred, _PS)

        precip = global_water_fix(
            self._denorm(x, _PS),
            self._denorm(x, _Q),
            self._denorm(y, _PS),
            self._denorm(y, _Q),
            self._denorm(y, _OUT["tp06"]),
            self._denorm(y, _OUT["e06"]),
            self.hyai,
            self.hybi,
            self.area,
            n,
        )
        y[:, _OUT["tp06"]] = self._renorm(precip, _OUT["tp06"])

        solin = x[:, _SOLIN_IN] * self.forcing_scale[0] + self.forcing_center[0]
        t_pred = global_energy_fix(
            self._denorm(x, _PS),
            self._denorm(x, _T),
            self._denorm(x, _Q),
            self._denorm(x, _U),
            self._denorm(x, _V),
            self._denorm(y, _PS),
            self._denorm(y, _T),
            self._denorm(y, _Q),
            self._denorm(y, _U),
            self._denorm(y, _V),
            toa_down_sw=solin * n,
            toa_up_sw=self._denorm(y, _OUT["mtuwswrf"]) * n,
            toa_up_lw=self._denorm(y, _OUT["mtuwlwrf"]) * n,
            surf_down_sw=self._denorm(y, _OUT["msdwswrf"]),
            surf_up_sw=self._denorm(y, _OUT["msuwswrf"]),
            surf_down_lw=self._denorm(y, _OUT["msdwlwrf"]),
            surf_up_lw=self._denorm(y, _OUT["msuwlwrf"]),
            surf_sh=self._denorm(y, _OUT["msshf"]),
            surf_lh=self._denorm(y, _OUT["mslhf"]),
            phis=self.phis,
            hyai=self.hyai,
            hybi=self.hybi,
            area=self.area,
            n_seconds=n,
        )
        y[:, _T] = self._renorm(t_pred, _T)
        return y

    @torch.inference_mode()
    def _normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize a physical state ``(batch, time, 1, 130, lat, lon)`` in
        Earth2Studio orientation into the network's ``(batch, time, 130, lat, lon)``
        south-to-north float32 state. The arithmetic runs in the input dtype, so a
        float64 input reproduces a normalized CAMulator state exactly."""
        state = torch.flip(x[:, :, 0], dims=(-2,))
        center = self.center[:_N_STATE].to(state.dtype)
        scale = self.scale[:_N_STATE].to(state.dtype)
        state_n = (state - center) / scale
        state_n[:, :, _PS] = (
            (state[:, :, _PS].double() - self.ps_center) / self.ps_scale
        ).to(state.dtype)
        return state_n.float()

    @torch.inference_mode()
    def _step(
        self, state_n: torch.Tensor, coords: CoordSystem, device: torch.device
    ) -> torch.Tensor:
        """Advance a normalized state ``(batch, time, 130, lat, lon)`` by one step,
        returning the post-processed normalized prediction
        ``(batch, time, 147, lat, lon)``."""
        b, t, _, h, w = state_n.shape
        forcing = self._fetch_forcing(coords, device)  # (t, 4, h, w)
        statics = self.statics.expand(b, t, -1, h, w)
        inp = torch.cat(
            [state_n, statics, forcing.unsqueeze(0).expand(b, -1, -1, -1, -1)], dim=2
        )
        inp = inp.reshape(b * t, -1, h, w)
        y = self.model(inp.unsqueeze(2)).squeeze(2)
        y = self._postprocess(inp, y)
        return y.reshape(b, t, _N_OUT, h, w)

    @torch.inference_mode()
    def _denormalize_output(self, y_n: torch.Tensor) -> torch.Tensor:
        """Convert a normalized prediction ``(batch, time, 147, lat, lon)`` to
        physical Earth2Studio units and orientation ``(batch, time, 1, 147, lat, lon)``.
        """
        y = y_n * self.scale + self.center
        y[:, :, _PS] = (
            y_n[:, :, _PS].double() * self.ps_scale + self.ps_center
        ).float()
        # Unit conversions in float64, rounded once: the correctly rounded float32
        # value, independent of the device's scalar-division kernel
        for i, mod in enumerate(self._output_mods):
            y[:, :, i] = mod(y[:, :, i].double()).float()
        return torch.flip(y, dims=(-2,)).unsqueeze(2)

    def _hooks_are_default(self) -> bool:
        """True when neither iterator hook has been replaced by the user."""
        default = PrognosticMixin._default_hook
        return (
            getattr(self.front_hook, "__func__", None) is default
            and getattr(self.rear_hook, "__func__", None) is default
        )

    def _forward(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        out_coords = self.output_coords(coords)
        device = self.device_buffer.device
        state_n = self._normalize_state(x.to(device))
        y_n = self._step(state_n, coords, device)
        return self._denormalize_output(y_n), out_coords

    @batch_func()
    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs prognostic model 1 step (6 hours)

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
        coords = coords.copy()
        self.output_coords(coords)
        device = self.device_buffer.device
        x = x.to(device)

        ic_coords = coords.copy()
        ic_coords["variable"] = np.array(OUTPUT_VARIABLES)
        ic = torch.full(
            (*x.shape[:3], _N_OUT, *x.shape[4:]),
            float("nan"),
            device=device,
            dtype=torch.float32,
        )
        ic[:, :, :, :_N_STATE] = x
        yield ic, ic_coords

        # With the default (identity) hooks the normalized state is carried
        # between steps, as in CREDIT, so a rollout does not accumulate physical
        # <-> normalized round-off. With user hooks installed the state is always
        # rebuilt from the (possibly in-place modified) physical tensors.
        default_hooks = self._hooks_are_default()
        state_n = self._normalize_state(x)
        while True:
            x, coords = self.front_hook(x, coords)
            if not default_hooks:
                state_n = self._normalize_state(x.to(device))
            y_n = self._step(state_n, coords, device)
            out = self._denormalize_output(y_n)
            out_coords = self.output_coords(coords)
            out, out_coords = self.rear_hook(out, out_coords)
            yield out, out_coords.copy()

            x = out[:, :, :, :_N_STATE]
            if default_hooks:
                state_n = y_n[:, :, :_N_STATE]
            else:
                state_n = self._normalize_state(x.to(device))
            coords = out_coords.copy()
            coords["variable"] = np.array(PROGNOSTIC_VARIABLES)

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step) in the
        output variable schema with the diagnostic variables set to NaN.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates time-steps of the prognostic model container the
            output data tensor and coordinate system dictionary.
        """
        yield from self._default_generator(x, coords)
