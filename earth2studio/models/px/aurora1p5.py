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
from collections import OrderedDict
from collections.abc import Generator, Iterator
from datetime import datetime, timezone

import numpy as np
import torch

from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordSystem

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
@check_optional_dependencies()
class Aurora1p5(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Aurora v1.5 0.25 degree global forecast model. This model is the improved
    version of Aurora, featuring an expanded set of surface variables (18 vs 4)
    and a richer set of static fields. It consists of a single auto-regressive
    model with a base time-step of 6 hours, operating on a 0.25 degree lat-lon
    grid (720, 1440) with 5 atmospheric variables across 13 pressure levels and
    18 surface variables plus 7 output-only surface variables.

    This wrapper uses an hourly rollout by default: the underlying 6-hour
    auto-regressive step is queried at each integer lead time from t+1h to t+6h
    before advancing the AR state.

    Note
    ----
    This model uses the checkpoints from the microsoft/aurora HuggingFace
    repository. For additional information see the following resources:

    - https://arxiv.org/abs/2405.13063
    - https://github.com/microsoft/aurora
    - https://huggingface.co/microsoft/aurora
    - https://microsoft.github.io/aurora/example_v1p5.html

    Aurora v1.5 was pretrained on ERA5 and fine-tuned on IFS operational
    analyses and as such recommended to be initialized with IFS analyses.
    The open-data IFS does not publish sea ice concentration (``sic``).
    :class:`earth2studio.data.NCAR_ERA5` or :class:`earth2studio.data.ARCO`
    (which provide all required variables) may be used instead. GFS is not
    supported due to missing surface variables.

    Note
    ----
    The iterator yields the initial condition (h=0) first, as required by the
    Earth2Studio convention. For the 7 output-only diagnostic variables
    (``i10fg``, ``blh``, ``uvb1h``, ``ssrd1h``, ``ttr1h``, ``tp1h``,
    ``sf1h``), the h=0 output contains ``NaN`` because the decoder has not run
    at that step. All subsequent outputs (h≥1) contain real model predictions.

    Warning
    -------
    We encourage users to familiarize themselves with the license restrictions of this
    model's checkpoints.

    Parameters
    ----------
    core_model : torch.nn.Module
        Core Aurora1p5 model
    static_vars : dict[str, torch.Tensor]
        Dictionary of static field tensors (e.g., lsm, z, slt_*, tvh_*, tvl_*, ...).
        Each tensor should have shape (720, 1440).

    Badges
    ------
    region:global class:mrf product:wind product:temp product:atmos product:precip product:land product:ocean product:solar year:2026 gpu:48gb
    """

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

        self._input_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array(
                    [np.timedelta64(-6, "h"), np.timedelta64(0, "h")]
                ),
                "variable": np.array(INPUT_VARIABLES),
                "lat": np.linspace(90, -90, 720, endpoint=False),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )

        self._output_coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.array([np.timedelta64(1, "h")]),
                "variable": np.array(OUTPUT_VARIABLES),
                "lat": np.linspace(90, -90, 720, endpoint=False),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )
        self.device = torch.ones(1).device  # Hack to get default device
        self.preds_idx = 0
        self.seed: int | None = None

    def _get_static_vars(self) -> dict[str, torch.Tensor]:
        return {k: getattr(self, f"static_var_{k}") for k in self._static_var_keys}

    def input_coords(self) -> CoordSystem:
        """Input coordinate system of the prognostic model

        Returns
        -------
        CoordSystem
            Coordinate system dictionary
        """
        return self._input_coords.copy()

    @batch_coords()
    def output_coords(
        self,
        input_coords: CoordSystem,
    ) -> CoordSystem:
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
        output_coords = self._output_coords.copy()

        test_coords = input_coords.copy()
        test_coords["lead_time"] = (
            test_coords["lead_time"] - input_coords["lead_time"][-1]
        )
        target_input_coords = self.input_coords()
        for i, key in enumerate(target_input_coords):
            if key not in ["batch", "time"]:
                handshake_dim(test_coords, key, i)
                handshake_coords(test_coords, target_input_coords, key)

        output_coords["batch"] = input_coords["batch"]
        output_coords["time"] = input_coords["time"]
        output_coords["lead_time"] = (
            input_coords["lead_time"][-1] + output_coords["lead_time"]
        )

        return output_coords

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
        """Load prognostic from package"""
        model, static_vars = _load_aurora1p5_from_package(
            package, Aurora1p5_model, "aurora-0.25-v1.5.ckpt"
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
                lead_times = torch.full(
                    (B,), h, device=x.device, dtype=torch.float32
                )
                output_batch = self.model.forward(input_batch, lead_times=lead_times)
                sub_preds[i][:, t : t + 1] = self._prepare_output(output_batch)

        return sub_preds

    @batch_func()
    def __call__(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Runs prognostic model 1 step.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Output tensor and coordinate system 1 hour in the future
        """
        output_coords = self.output_coords(coords)
        x = self._forward_sub_steps(x, coords, lead_time_hours=[1])[0]
        return x, output_coords

    @staticmethod
    def _clip_ar_input(x: torch.Tensor) -> torch.Tensor:
        """Clamp AR feedback channels to physical bounds (mirrors Aurora1p5 rollout_input_clipping)."""
        x = x.clone()
        for idx, lo, hi in _AR_CLIP_BOUNDS:
            x[..., idx, :, :] = x[..., idx, :, :].clamp(min=lo, max=hi)
        return x

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        self.preds_idx = 0
        coords = coords.copy()

        self.output_coords(coords)

        ic_coords = coords.copy()
        ic_coords["lead_time"] = np.array([coords["lead_time"][-1]])
        ic_coords["variable"] = np.array(OUTPUT_VARIABLES)
        # Pad IC with NaN for the output-only channels so every yield is
        # consistent at 90 variables. NaN signals "not available" — the decoder
        # has not run at t=0, so these slots carry no physical meaning.
        n_in = _N_ATMOS + _N_SURF
        ic_tensor = x[:, :, 1:]
        padding = torch.full(
            (*ic_tensor.shape[:3], _N_OUTPUT_ONLY, *ic_tensor.shape[4:]),
            float("nan"),
            device=ic_tensor.device,
            dtype=ic_tensor.dtype,
        )
        yield torch.cat([ic_tensor, padding], dim=3), ic_coords

        while True:
            # Front hook runs once per 6h AR cycle
            x, coords = self.front_hook(x, coords)
            ar_prev_x = x[:, :, 1:].clone()  # state at current t (83 channels)
            current_t_lead = coords["lead_time"][-1]

            # Compute t+1h … t+6h from the same AR input pair [t-6h, t]
            sub_preds = self._forward_sub_steps(
                x, coords, lead_time_hours=list(range(1, int(_AR_STEP_HOURS) + 1))
            )

            for sub_idx, sub_pred in enumerate(sub_preds):
                h = sub_idx + 1
                sub_lead = current_t_lead + np.timedelta64(h, "h")
                coords_out = coords.copy()
                coords_out["lead_time"] = np.array([sub_lead])
                coords_out["variable"] = np.array(OUTPUT_VARIABLES)

                if h < int(_AR_STEP_HOURS):
                    # Intermediate hourly step: yield with rear hook
                    sub_pred, coords_out = self.rear_hook(sub_pred, coords_out)
                    yield sub_pred, coords_out
                else:
                    # 6h AR boundary: apply rear hook to the full unclipped prediction,
                    # then clip only the input-var slice for AR feedback.
                    # Users receive raw (post-hook, pre-clip) predictions.
                    sub_pred, coords_out = self.rear_hook(sub_pred, coords_out)
                    ar_next = self._clip_ar_input(sub_pred[:, :, :, :n_in])
                    x = torch.cat([ar_prev_x, ar_next], dim=2).clone()

                    self.preds_idx = self.preds_idx + 1
                    coords["lead_time"] = np.array([current_t_lead, sub_lead])

                    yield sub_pred, coords_out

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates time-steps of the prognostic model containing
            the output data tensor and coordinate system dictionary.
        """
        yield from self._default_generator(x, coords)


@check_optional_dependencies()
class Aurora1p5Ensemble(Aurora1p5):
    """Aurora v1.5 ensemble 0.25 degree global forecast model. Identical to
    :class:`Aurora1p5` except it uses the stochastic ensemble checkpoint, where
    each forward pass injects fresh Gaussian noise into the backbone conditioning
    context. Calling the model N times (or with a batch of N copies of the same
    initial condition) therefore produces N statistically independent members.

    Like :class:`Aurora1p5`, this wrapper uses an hourly rollout by default,
    leveraging the 6-hour base time-step to produce hourly lead times without
    additional model evaluations per AR cycle.

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
    seed : int | None, optional
        If specified, sets the random seed via :meth:`set_rng` at the start of
        each :meth:`create_iterator` call for reproducible stochastic noise.
        By default None (non-reproducible).

    Badges
    ------
    region:global class:mrf product:wind product:temp product:atmos product:precip product:land product:ocean product:solar year:2026 gpu:48gb
    """

    def __init__(
        self,
        core_model: torch.nn.Module,
        static_vars: dict[str, torch.Tensor],
        seed: int | None = None,
    ) -> None:
        super().__init__(core_model, static_vars)
        self.seed = seed

    def set_rng(self, seed: int | None) -> None:
        """Seed the global RNG and reset the model's internal noise cache.

        Parameters
        ----------
        seed : int | None
            Seed for :func:`torch.manual_seed`. If None, only resets the noise cache.
        """
        if seed is not None:
            torch.manual_seed(seed)
        self.model.reset_noise()

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
        """Load prognostic from package"""
        model, static_vars = _load_aurora1p5_from_package(
            package, Aurora1p5Ensemble_model, "aurora-0.25-v1.5-ensemble.ckpt"
        )
        return cls(model, static_vars)

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Creates a iterator which can be used to perform time-integration of the
        prognostic model. Will return the initial condition first (0th step).

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        coords : CoordSystem
            Input coordinate system

        Yields
        ------
        Iterator[tuple[torch.Tensor, CoordSystem]]
            Iterator that generates time-steps of the prognostic model containing
            the output data tensor and coordinate system dictionary.
        """
        self.set_rng(self.seed)
        yield from self._default_generator(x, coords)
