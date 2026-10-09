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

import pickle
from collections.abc import Generator
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import torch
import xarray as xr

from earth2studio.data.base import DataSource
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.px.aurora import _aurora_history
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.models.utils import fork_rng
from earth2studio.utils.coords import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_nonempty,
    handshake_time,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordinateSystem, CoordSystem

try:
    from aurora import AuroraV1p5 as Aurora1p5_model
    from aurora import AuroraV1p5Ensemble as Aurora1p5Ensemble_model
    from aurora import Batch, Metadata
    from aurora.insolation import insolation as aurora_insolation
    from aurora.normalisation import log_untransform as aurora_log_untransform
except ImportError:
    OptionalDependencyFailure("aurora")
    Aurora1p5_model = None
    Aurora1p5Ensemble_model = None
    Batch = None
    Metadata = None
    aurora_insolation = None
    aurora_log_untransform = None

ATMOS_LEVELS = [1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50]

INPUT_VARIABLES = (
    [f"z{lv}" for lv in ATMOS_LEVELS]
    + [f"q{lv}" for lv in ATMOS_LEVELS]
    + [f"t{lv}" for lv in ATMOS_LEVELS]
    + [f"u{lv}" for lv in ATMOS_LEVELS]
    + [f"v{lv}" for lv in ATMOS_LEVELS]
    + [
        "msl",
        "u10m",
        "v10m",
        "t2m",
        "d2m",
        "tcwv",
        "tcc",
        "u100m",
        "v100m",
        "sp",
        "lcc",
        "mcc",
        "hcc",
        "skt",
        "stl1",
        "swvl1",
        "sic",
        "sd",
    ]
)

# Output includes the 7 extra diagnostic vars that the decoder produces but
# that are NOT fed back as AR inputs on the next cycle.
OUTPUT_VARIABLES = INPUT_VARIABLES + [
    "i10fg",
    "blh",
    "uvb1h",
    "ssrd1h",
    "ttr1h",
    "tp1h",
    "sf1h",
]

# Backwards-compatible alias
VARIABLES = INPUT_VARIABLES

# Mapping from Earth2Studio variable names to Aurora internal surf_var names
_SURF_VAR_MAP = {
    "msl": "msl",
    "u10m": "10u",
    "v10m": "10v",
    "t2m": "2t",
    "d2m": "2d",
    "tcwv": "tcwv",
    "tcc": "tcc",
    "u100m": "100u",
    "v100m": "100v",
    "sp": "sp",
    "lcc": "lcc",
    "mcc": "mcc",
    "hcc": "hcc",
    "skt": "skt",
    "stl1": "stl1",
    "swvl1": "swvl1",
    "sic": "ci",
    "sd": "scaled_sd",
}

_SURF_VARS_E2S = list(_SURF_VAR_MAP.keys())

# Output-only surface variables: produced by the decoder but not fed back as AR
# inputs. Mapping is (E2S name, Aurora name, needs_log_untransform).
_OUTPUT_ONLY_SURF_VARS = [
    ("i10fg", "i10fg", False),
    ("blh", "blh", False),
    ("uvb1h", "uvb_1h", False),
    ("ssrd1h", "ssrd_1h", False),
    ("ttr1h", "ttr_1h", False),
    ("tp1h", "scaled_tp_1h", True),
    ("sf1h", "scaled_sf_1h", True),
]

_N_ATMOS_LEVELS = len(ATMOS_LEVELS)
_N_ATMOS_VARS = 5  # z, q, t, u, v
_N_ATMOS = _N_ATMOS_VARS * _N_ATMOS_LEVELS  # 65
_N_SURF = len(_SURF_VARS_E2S)  # 18
_N_OUTPUT_ONLY = len(_OUTPUT_ONLY_SURF_VARS)  # 7
_AR_STEP_HOURS = 6.0

# Tensor-level clipping bounds applied to the AR feedback state at the 6h
# boundary. Mirrors Aurora1p5's default rollout_input_clipping dict.
# Entries are (variable_tensor_index, min_or_None, max_or_None).
_AR_CLIP_BOUNDS: list[tuple[int, float | None, float | None]] = [
    (_N_ATMOS + _SURF_VARS_E2S.index("tcwv"), 0.0, None),
    (_N_ATMOS + _SURF_VARS_E2S.index("tcc"), 0.0, 1.0),
    (_N_ATMOS + _SURF_VARS_E2S.index("lcc"), 0.0, 1.0),
    (_N_ATMOS + _SURF_VARS_E2S.index("mcc"), 0.0, 1.0),
    (_N_ATMOS + _SURF_VARS_E2S.index("swvl1"), 0.0, 70.0),
    (_N_ATMOS + _SURF_VARS_E2S.index("sic"), 0.0, 1.0),
    (_N_ATMOS + _SURF_VARS_E2S.index("sd"), 0.0, 10.0),
    (_N_ATMOS + _SURF_VARS_E2S.index("hcc"), 0.0, 1.0),
]


def _load_aurora1p5_from_package(
    package: Package,
    aurora_cls: type,
    checkpoint_name: str,
) -> tuple[torch.nn.Module, dict[str, torch.Tensor]]:
    """Shared loader for Aurora1p5 and Aurora1p5Ensemble."""
    static_path = package.resolve("aurora-0.25-v1.5-static.pickle")
    with open(static_path, "rb") as f:
        static_raw = pickle.load(f)  # noqa: S301
    # The pickle stores 721-row (pole-inclusive) ERA5 grids; the model operates on
    # 720 rows (endpoint=False), so we drop the south-pole row here.
    static_vars = {
        k: torch.from_numpy(np.asarray(v))[:720, :] for k, v in static_raw.items()
    }

    checkpoint_path = package.resolve(checkpoint_name)
    model = aurora_cls()
    model.load_checkpoint_local(checkpoint_path)
    model.eval()

    return model, static_vars


# Adapted from https://microsoft.github.io/aurora/example_v1p5.html
@dataclass
class _AuroraState:
    history: xr.DataArray
    index: int
    seed: int | None
    rng: dict[str, torch.Tensor] | None
    noise: list[torch.Tensor]


@check_optional_dependencies()
class _Aurora(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Shared xarray execution for the hourly and six-hour Aurora variants."""

    _STEP_HOURS: int = 1
    _ENSEMBLE: bool = False

    def __init__(
        self,
        core_model: torch.nn.Module,
        static_vars: dict[str, torch.Tensor],
    ) -> None:
        super().__init__()

        self.model = core_model
        self._static_var_keys = list(static_vars.keys())
        for key, val in static_vars.items():
            self.register_buffer(f"static_var_{key}", val)

        self.register_buffer("device_buffer", torch.empty(0))
        self.preds_idx = 0

    def _get_static_vars(self) -> dict[str, torch.Tensor]:
        return {k: getattr(self, f"static_var_{k}") for k in self._static_var_keys}

    front_hook_interval = 1

    def default_sources(self) -> DataSource:
        """Recommend ARCO ERA5 initial conditions.

        Returns
        -------
        DataSource
            Raw ARCO ERA5 source for the input slot.
        """
        from earth2studio.data import ARCO_ERA5

        return ARCO_ERA5()

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the prognostic model.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray input signature for the six-hour
            autoregressive history on the south-pole-excluded grid.
        """
        return coord_array(
            ("batch", "time", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": np.array([-6, 0], dtype="timedelta64[h]"),
                "variable": INPUT_VARIABLES,
            },
            dynamic=("batch", "time"),
            grid="latlon-0.25deg-south-pole-excluded",
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the prognostic model.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate signature or DataArray to validate and transform.

        Returns
        -------
        CoordinateSystem
            Allocation-free signature for the complete six-hour forecast chunk
            at the variant's output cadence, including one-hour diagnostics.
        """
        handshake_time(input_coords, allow_dynamic=True)
        handshake_time(input_coords, "lead_time")
        lead = input_coords.lead_time.values
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
        return coord_array_like(
            input_coords,
            {
                "lead_time": input_coords.lead_time.values[-1:]
                + np.arange(self._STEP_HOURS, 7, self._STEP_HOURS).astype(
                    "timedelta64[h]"
                ),
                "variable": OUTPUT_VARIABLES,
            },
        )

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        return Package(
            "hf://microsoft/aurora@c171214768997594e1a3fc6b8d9bbb489e9d21ab",
            cache_options={
                "cache_storage": Package.default_cache("aurora1p5"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
    ) -> PrognosticModel:
        """Load prognostic from package

        Parameters
        ----------
        package : Package
            Package to load model from

        Returns
        -------
        PrognosticModel
            Prognostic model
        """
        if cls._ENSEMBLE:
            aurora_cls = Aurora1p5Ensemble_model
            checkpoint = "aurora-0.25-v1.5-ensemble.ckpt"
        else:
            aurora_cls = Aurora1p5_model
            checkpoint = "aurora-0.25-v1.5.ckpt"
        model, static_vars = _load_aurora1p5_from_package(
            package, aurora_cls, checkpoint
        )
        return cls(model, static_vars)

    def _compute_insolation(
        self,
        dt0: datetime,
        dt1: datetime,
        lat: np.ndarray,
        lon: np.ndarray,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Compute solar insolation for both input time steps.

        Returns
        -------
        torch.Tensor
            Shape (batch_size, 2, H, W)
        """
        # enforce_2d meshgrids the 1-D lat/lon arrays → output (len(dates), H, W)
        insol = aurora_insolation((dt0, dt1), lat, lon, enforce_2d=True)
        insol_t = torch.from_numpy(np.asarray(insol)).float().to(device)
        return insol_t.unsqueeze(0).expand(batch_size, -1, -1, -1)

    def _ts_to_datetime(self, ts: np.datetime64) -> datetime:
        epoch = np.datetime64("1970-01-01T00:00:00")
        seconds = float((ts.astype("datetime64[s]") - epoch) / np.timedelta64(1, "s"))  # type: ignore[operator]
        return datetime.fromtimestamp(seconds, tz=timezone.utc)

    def _prepare_input(self, x: torch.Tensor, coords: CoordSystem) -> Batch:
        """Build an Aurora Batch from a (B, 1, 2, 83, H, W) tensor."""
        B = x.shape[0]

        # x: (B, 1, 2, 83, H, W) — select the single time axis
        inp = x[:, 0]  # (B, 2, 83, H, W)

        # Compute the two input datetimes from coords
        dt0 = self._ts_to_datetime(coords["time"][0] + coords["lead_time"][0])
        dt1 = self._ts_to_datetime(coords["time"][0] + coords["lead_time"][-1])

        # Atmosphere: each (B, 2, 13, H, W)
        atmos_vars = {
            "z": inp[:, :, 0 * _N_ATMOS_LEVELS : 1 * _N_ATMOS_LEVELS],
            "q": inp[:, :, 1 * _N_ATMOS_LEVELS : 2 * _N_ATMOS_LEVELS],
            "t": inp[:, :, 2 * _N_ATMOS_LEVELS : 3 * _N_ATMOS_LEVELS],
            "u": inp[:, :, 3 * _N_ATMOS_LEVELS : 4 * _N_ATMOS_LEVELS],
            "v": inp[:, :, 4 * _N_ATMOS_LEVELS : 5 * _N_ATMOS_LEVELS],
        }

        # Surface: each (B, 2, H, W)
        # ERA5 land-only variables (e.g. swvl1, stl1) are NaN over ocean.
        # Aurora's transformer propagates NaN to all outputs, so fill before building the Batch.
        surf_start = _N_ATMOS
        surf_vars: dict[str, torch.Tensor] = {}
        for i, e2s_name in enumerate(_SURF_VARS_E2S):
            aurora_name = _SURF_VAR_MAP[e2s_name]
            surf_vars[aurora_name] = torch.nan_to_num(
                inp[:, :, surf_start + i], nan=0.0
            )

        # Insolation (computed, not a user input)
        surf_vars["insolation"] = self._compute_insolation(
            dt0, dt1, coords["lat"], coords["lon"], B, x.device
        )

        return Batch(
            surf_vars=surf_vars,
            static_vars=self._get_static_vars(),
            atmos_vars=atmos_vars,
            metadata=Metadata(
                lat=torch.from_numpy(coords["lat"]).to(x.device),
                lon=torch.from_numpy(coords["lon"]).to(x.device),
                time=(dt1,),
                atmos_levels=tuple(int(lv) for lv in ATMOS_LEVELS),
                rollout_step=self.preds_idx,
            ),
        )

    def _prepare_output(self, output: Batch) -> torch.Tensor:
        """Convert Aurora output Batch to (B, 1, 1, 90, H, W) tensor."""
        # Atmosphere: each (B, 1, 13, H, W)
        atmos = torch.cat(
            [
                output.atmos_vars["z"],
                output.atmos_vars["q"],
                output.atmos_vars["t"],
                output.atmos_vars["u"],
                output.atmos_vars["v"],
            ],
            dim=2,
        )  # (B, 1, 65, H, W)

        # Bidirectional surface vars: each (B, 1, H, W) → (B, 1, 1, H, W)
        surf = torch.cat(
            [output.surf_vars[_SURF_VAR_MAP[e]].unsqueeze(2) for e in _SURF_VARS_E2S],
            dim=2,
        )  # (B, 1, 18, H, W)

        # Output-only diagnostic vars. scaled_tp_1h / scaled_sf_1h are stored
        # in log-space by Aurora's post-norm hook; invert to physical units.
        diag_tensors = []
        for _, aurora_name, log_scaled in _OUTPUT_ONLY_SURF_VARS:
            v = output.surf_vars[aurora_name]
            if log_scaled:
                v = aurora_log_untransform(v)
            diag_tensors.append(v.unsqueeze(2))
        diag = torch.cat(diag_tensors, dim=2)  # (B, 1, 7, H, W)

        x = torch.cat([atmos, surf, diag], dim=2)  # (B, 1, 90, H, W)
        return x.view(-1, 1, *x.shape[1:])  # (B, 1, 1, 90, H, W)

    @torch.inference_mode()
    def _forward_sub_steps(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        lead_time_hours: list[int],
    ) -> list[torch.Tensor]:
        """Run the model at each requested lead time from one AR input pair.

        Returns one tensor of shape (B, T, 1, 90, H, W) per requested lead time.
        Raw (unclipped) outputs are returned; clipping for AR feedback is the
        caller's responsibility (see ``_default_generator``).
        """
        B = x.shape[0]
        T = coords["time"].shape[0]
        n_out = _N_ATMOS + _N_SURF + _N_OUTPUT_ONLY

        sub_preds: list[torch.Tensor] = [
            torch.empty(B, T, 1, n_out, *x.shape[-2:], device=x.device, dtype=x.dtype)
            for _ in lead_time_hours
        ]

        for t in range(T):
            t_coords = coords.copy()
            t_coords["time"] = t_coords["time"][t : t + 1]
            # Build the Batch once; reuse for all lead-time queries
            input_batch = self._prepare_input(x[:, t : t + 1], t_coords)

            for i, h in enumerate(lead_time_hours):
                lead_times = torch.full((B,), h, device=x.device, dtype=torch.float32)
                output_batch = self.model.forward(input_batch, lead_times=lead_times)
                sub_preds[i][:, t : t + 1] = self._prepare_output(output_batch)

        return sub_preds

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Predict the complete six-hour forecast chunk without hooks.

        Parameters
        ----------
        x : xr.DataArray
            Initial two-frame history matching ``input_coords()``.

        Returns
        -------
        xr.DataArray
            Forecasts through six hours at this variant's output cadence,
            including output-only diagnostic variables.
        """
        return self.initialize(x)[0]

    def initialize(self, x: xr.DataArray) -> tuple[xr.DataArray, _AuroraState]:
        """Predict the first chunk and retain history and ensemble noise.

        Parameters
        ----------
        x : xr.DataArray
            Two-frame initial history matching ``input_coords()``.

        Returns
        -------
        tuple[xr.DataArray, _AuroraState]
            Complete six-hour forecast chunk at the variant's output cadence and
            continuation history, rollout index, RNG state and ensemble noise
            cache. Iterator hooks are not applied.
        """
        handshake_nonempty(x)
        if self.stochastic and self._rng_seed is None:
            self.set_rng(int(torch.randint(0, 2**31, ()).item()))
        y, state = self._forward(
            x, _AuroraState(x, 0, self._rng_seed, deepcopy(self._rng_states), [])
        )
        self._rng_states = deepcopy(state.rng)
        return y, state

    def step(
        self, y: xr.DataArray, state: _AuroraState
    ) -> tuple[xr.DataArray, _AuroraState]:
        """Advance from the last forecast frame and explicit history/noise state.

        Parameters
        ----------
        y : xr.DataArray
            Previous forecast chunk. Its final frame supplies the next input;
            autoregressive channels are clipped to their physical bounds.
        state : _AuroraState
            History, rollout index, RNG state and noise cache returned by
            ``initialize`` or ``step``.

        Returns
        -------
        tuple[xr.DataArray, _AuroraState]
            Next complete six-hour chunk and updated state, without iterator hooks.
        """
        feedback = y.sel(variable=INPUT_VARIABLES).isel(lead_time=slice(-1, None))
        tensor, _ = feedback.e2s.to_torch()
        clipped = from_torch(
            self._clip_ar_input(tensor), coord_array_like(feedback), name=y.name
        )
        clipped.encoding = y.encoding.copy()
        return self._forward(_aurora_history(state.history, clipped), state)

    def _forward(
        self, x: xr.DataArray, state: _AuroraState
    ) -> tuple[xr.DataArray, _AuroraState]:
        previous = self.preds_idx, self._rng_seed, self._rng_states
        self.preds_idx, self._rng_seed, self._rng_states = (
            state.index,
            state.seed,
            deepcopy(state.rng),
        )
        backbone = getattr(self.model, "backbone", None)
        if self._ENSEMBLE:
            self.model.set_noise_accumulation(n=6 // self._STEP_HOURS)
            if backbone is not None:
                backbone._noise_cache = deepcopy(state.noise)
        try:
            predictions = self._sub_steps(
                x, list(range(self._STEP_HOURS, 7, self._STEP_HOURS))
            )
            y = xr.concat(
                predictions,
                dim="lead_time",
                coords="minimal",
                compat="override",
                join="exact",
            )
            y.encoding = x.encoding.copy()
            noise = (
                deepcopy(backbone._noise_cache)
                if self._ENSEMBLE and backbone is not None
                else []
            )
            return y, _AuroraState(
                x.isel(lead_time=slice(-1, None)).copy(deep=True),
                state.index + 1,
                state.seed,
                deepcopy(self._rng_states),
                noise,
            )
        finally:
            self.preds_idx, self._rng_seed, self._rng_states = previous
            if self._ENSEMBLE:
                self.model.set_noise_accumulation(n=0)

    def _sub_steps(self, x: xr.DataArray, hours: list[int]) -> list[xr.DataArray]:
        self.output_coords(x)
        handshake_time(x)
        packed, restore = batch_func()._compress_array(self, x)
        signature = self.output_coords(packed)
        tensor, coords = packed.e2s.to_torch()
        with fork_rng(
            self._rng_seed, self.device_buffer.device, states=self._rng_states
        ):
            predictions = self._forward_sub_steps(
                tensor.to(self.device_buffer.device).clone(), coords, hours
            )
        results = []
        for h, prediction in zip(hours, predictions):
            out_signature = coord_array_like(
                signature,
                {"lead_time": packed.lead_time.values[-1:] + np.timedelta64(h, "h")},
            )
            out = from_torch(prediction, out_signature, name=x.name)
            out.encoding = x.encoding.copy()
            results.append(restore(out))
        return results

    @staticmethod
    def _clip_ar_input(x: torch.Tensor) -> torch.Tensor:
        """Clamp AR feedback channels to physical bounds (mirrors Aurora1p5 rollout_input_clipping)."""
        x = x.clone()
        for idx, lo, hi in _AR_CLIP_BOUNDS:
            x[..., idx, :, :] = x[..., idx, :, :].clamp(min=lo, max=hi)
        return x

    def create_iterator(self, x: xr.DataArray) -> Generator[xr.DataArray, None, None]:
        """Yield complete forecast chunks, beginning with initialization's prediction.

        Parameters
        ----------
        x : xr.DataArray
            Initial history matching ``input_coords()``.

        Yields
        ------
        xr.DataArray
            Six-hour chunks at the variant's output cadence. The rear hook runs
            once per chunk and the front hook before subsequent steps. The final
            frame supplies autoregressive feedback.
        """
        yield from self._default_create_iterator(x)

    _rng_seed: int | None = None
    _rng_states: dict[str, torch.Tensor] | None = None

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Set the isolated random stream and reset cached ensemble noise.

        Parameters
        ----------
        seed : int
            Seed for reproducible sampling.
        reset : bool, optional
            Reset an existing stream, by default True.
        """
        if reset or self._rng_seed is None:
            self._rng_seed = seed
            self._rng_states = {}
            if self._ENSEMBLE:
                self.model.reset_noise()


@check_optional_dependencies()
class Aurora1p5(_Aurora):
    """Aurora v1.5 0.25 degree global forecast model with hourly output.

    The underlying 6-hour auto-regressive model is queried at t+1h through
    t+6h from the same input pair before advancing the AR state. Inputs are
    two states six hours apart on a (720, 1440) grid, with 5 atmospheric
    variables across 13 pressure levels and 18 surface variables. Outputs
    include 7 additional diagnostic variables.

    Note
    ----
    This model uses the checkpoints from the microsoft/aurora HuggingFace
    repository. For additional information see:

    - https://arxiv.org/abs/2405.13063
    - https://github.com/microsoft/aurora
    - https://huggingface.co/microsoft/aurora
    - https://microsoft.github.io/aurora/example_v1p5.html

    Aurora v1.5 was pretrained on ERA5 and fine-tuned on IFS operational
    analyses and is recommended to be initialized with IFS analyses.
    The open-data IFS does not publish sea ice concentration (``sic``).
    :class:`earth2studio.data.NCAR_ERA5` or :class:`earth2studio.data.ARCO_ERA5`
    may be used instead. GFS is not supported due to missing surface variables.

    The iterator yields six-hour forecast chunks with hourly frames, including
    output-only diagnostic variables from the first yield. Hourly accumulations use
    qualified labels such as ``tp:sum:1h`` and retain their physical units.
    Use :class:`Aurora1p5_6h` for six-hourly output.

    Warning
    -------
    We encourage users to familiarize themselves with the license restrictions
    of this model's checkpoints.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core Aurora1p5 model
    static_vars : dict[str, torch.Tensor]
        Static field tensors, each with shape (720, 1440).

    Badges
    ------
    region:global class:medium-range product:wind product:temp product:atmos product:precip product:land product:ocean product:solar year:2026 gpu:48gb
    provider:microsoft backend:pytorch
    """

    _STEP_HOURS = 1
    _ENSEMBLE = False


@check_optional_dependencies()
class Aurora1p5Ensemble(_Aurora):
    """Aurora v1.5 ensemble 0.25 degree global forecast model. Identical to
    :class:`Aurora1p5` except it uses the stochastic ensemble checkpoint, where
    each forward pass injects fresh Gaussian noise into the backbone conditioning
    context. Calling the model N times (or with a batch of N copies of the same
    initial condition) therefore produces N statistically independent members.

    Like :class:`Aurora1p5`, this wrapper produces hourly output using six
    lead-time queries per 6-hour AR cycle. Use :class:`Aurora1p5Ensemble_6h`
    for six-hourly output. The two cadences consume the RNG stream differently,
    so the same seed does not produce matching trajectories between variants.

    Note
    ----
    This model uses the ensemble checkpoint from the microsoft/aurora
    HuggingFace repository. For additional information see the following resources:

    - https://arxiv.org/abs/2405.13063
    - https://github.com/microsoft/aurora
    - https://huggingface.co/microsoft/aurora
    - https://microsoft.github.io/aurora/example_v1p5.html

    Aurora v1.5 was pretrained on ERA5 and fine-tuned on IFS operational
    analyses. See :class:`Aurora1p5` for data source recommendations.

    Warning
    -------
    We encourage users to familiarize themselves with the license restrictions of
    this model's checkpoints.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core Aurora1p5Ensemble model (stochastic=True)
    static_vars : dict[str, torch.Tensor]
        Dictionary of static field tensors (e.g., lsm, z, slt_*, tvh_*, tvl_*, ...).
        Each tensor should have shape (720, 1440).

    Badges
    ------
    region:global class:medium-range product:wind product:temp product:atmos product:precip product:land product:ocean product:solar year:2026 gpu:48gb
    provider:microsoft backend:pytorch
    """

    _STEP_HOURS = 1
    _ENSEMBLE = True

    def __init__(
        self,
        core_model: torch.nn.Module,
        static_vars: dict[str, torch.Tensor],
    ) -> None:
        super().__init__(core_model, static_vars)

    stochastic = True


@check_optional_dependencies()
class Aurora1p5_6h(_Aurora):
    """Aurora v1.5 0.25 degree global forecast model with six-hourly output.

    Uses the same checkpoint, input history and variables as :class:`Aurora1p5`,
    but queries only t+6h per AR cycle. Both single-step calls and the iterator
    advance six hours. Diagnostic variables suffixed ``1h`` retain their
    one-hour accumulation windows; they are not six-hour totals.

    See :class:`Aurora1p5` for checkpoint references, data source recommendations,
    license information and initial-condition diagnostic handling.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core Aurora1p5 model
    static_vars : dict[str, torch.Tensor]
        Static field tensors, each with shape (720, 1440).

    Badges
    ------
    region:global class:medium-range product:wind product:temp product:atmos product:precip product:land product:ocean product:solar year:2026 gpu:48gb
    provider:microsoft backend:pytorch
    """

    _STEP_HOURS = 6
    _ENSEMBLE = False
    front_hook_interval = 1


@check_optional_dependencies()
class Aurora1p5Ensemble_6h(_Aurora):
    """Aurora v1.5 ensemble global forecast model with six-hourly output.

    Uses the stochastic checkpoint of :class:`Aurora1p5Ensemble`, querying
    only t+6h per AR cycle. Inputs and variables match :class:`Aurora1p5`.
    Diagnostic variables suffixed ``1h`` retain their one-hour accumulation
    windows; they are not six-hour totals. The iterator uses a single-entry
    noise cache, so seeds do not give matching trajectories with the hourly
    ensemble variant.

    See :class:`Aurora1p5Ensemble` for checkpoint references, data source
    recommendations and license information. Initial-condition diagnostic
    handling is described in :class:`Aurora1p5`.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core Aurora1p5Ensemble model (stochastic=True)
    static_vars : dict[str, torch.Tensor]
        Static field tensors, each with shape (720, 1440).

    Badges
    ------
    region:global class:medium-range product:wind product:temp product:atmos product:precip product:land product:ocean product:solar year:2026 gpu:48gb
    provider:microsoft backend:pytorch
    """

    _STEP_HOURS = 6
    _ENSEMBLE = True
    front_hook_interval = 1
    stochastic = True

    def __init__(
        self,
        core_model: torch.nn.Module,
        static_vars: dict[str, torch.Tensor],
    ) -> None:
        super().__init__(core_model, static_vars)
