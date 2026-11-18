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
from collections.abc import Callable, Iterator
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, TypeVar, cast

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.data.base import DataSource
from earth2studio.grids import HEALPixGrid
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils.coords import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_dim,
    handshake_nonempty,
    handshake_size,
    handshake_time,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.time_statistics import format_duration
from earth2studio.utils.type import (
    CoordinateSystem,
    CoordSystem,
    TimeArray,
    VariableArray,
)

try:
    import earth2grid
    from omegaconf import OmegaConf
    from physicsnemo import Module
    from physicsnemo.utils.insolation import insolation
except ImportError:
    OptionalDependencyFailure("dlesym")
    Module = None
    insolation = None
    OmegaConf = None
    earth2grid = None

_ATMOS_VARIABLES = [
    "z500",
    "tau300-700",
    "z1000",
    "t2m",
    "tcwv",
    "t850",
    "z250",
    "ws10m",
]
_OCEAN_VARIABLES = ["sst"]
_ATMOS_COUPLING_VARIABLES = ["sst"]
_OCEAN_COUPLING_VARIABLES = ["z1000", "ws10m"]

_ATMOS_INPUT_TIMES = np.array([-18, -12, -6, 0], dtype="timedelta64[h]")
_OCEAN_INPUT_TIMES = np.array([-48, 0], dtype="timedelta64[h]")

_ATMOS_OUTPUT_TIMES = np.array(
    [6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96],
    dtype="timedelta64[h]",
)
_OCEAN_OUTPUT_TIMES = np.array([48, 96], dtype="timedelta64[h]")

# Normalize variable names in the model package to match e2studio's lexicon
_ATMOS_VARIABLE_RENAMES = {"ttr-3h": "ttr03", "tp6": "tp06"}
# The package's `tp6` is log(1 + tp / _TP06_EPS), tp the 6-hour total in meters
_TP06_EPS = 1e-8

T = TypeVar("T")


def _healpix_grid(nside: int) -> str | HEALPixGrid:
    """The native grid: HEALPix faces in XY order, north origin, clockwise."""
    if nside == 64:
        return "healpix-l6-xy-north-clockwise-face"
    return HEALPixGrid(
        int(np.log2(nside)),
        ordering="xy",
        layout="face",
        xy_origin="north",
        xy_clockwise=True,
    )


def _insolation_tensor(
    anchor_times: np.ndarray,
    timedeltas: np.ndarray,
    hpx_lat: np.ndarray,
    hpx_lon: np.ndarray,
) -> torch.Tensor:
    """Insolation at each anchor time plus each timedelta.

    Returns a tensor of shape (anchor_times, face, timedeltas, 1, height, width).
    """
    times_flattened = np.array(
        [[a + t for t in timedeltas] for a in anchor_times]
    ).flatten()
    sol = insolation(times_flattened, hpx_lat, hpx_lon)
    _, f, h, w = sol.shape
    sol = torch.from_numpy(sol).view(len(anchor_times), len(timedeltas), 1, f, h, w)
    return sol.permute(0, 3, 1, 2, 4, 5)


def _tp06_to_meters(x: torch.Tensor, dim: int, index: int) -> None:
    """Invert tp06_log = log(1 + tp06 / eps) in place on channel ``index`` of ``dim``.

    Negative values of the log-transformed field mean no rain.
    """
    channel = x.select(dim, index)
    channel.copy_(torch.expm1(channel.clamp(min=0)) * _TP06_EPS)


@dataclass
class _PackageConfig:
    """Config, constants and normalization parsed from a DLESyM package.

    Shared by the fused model and its components, so each loads only the cores
    and buffers it runs. Variable names are in the lexicon.
    """

    package: Package
    nside: int
    hpx_lat: np.ndarray
    hpx_lon: np.ndarray
    scaling: dict[str, tuple[float, float]]
    checkpoints: dict[str, list[str]]
    constants: dict[str, list[str]]
    input_times: dict[str, np.ndarray]
    output_times: dict[str, np.ndarray]
    variables: dict[str, list[str]]
    coupling_variables: dict[str, list[str]]
    diagnostic_variables: dict[str, list[str]]
    use_cln: bool
    condition_shape: int | None

    def core(self, component: str, index: int) -> torch.nn.Module:
        """Load the network core of ``component`` ("atmos" or "ocean")."""
        checkpoints = self.checkpoints[component]
        if index != 0 and len(checkpoints) == 1:
            raise ValueError(
                f"The DLESyM package ships a single {component} checkpoint; "
                f"'{component}_model_idx' must be 0."
            )
        core = Module.from_checkpoint(self.package.resolve(checkpoints[index]))
        core.output_time_dim = len(self.output_times[component])
        return core

    def stats(self, variables: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Means and standard deviations of ``variables``, in order."""
        center = np.array([self.scaling[v][0] for v in variables])
        scale = np.array([self.scaling[v][1] for v in variables])
        return center, scale

    def constant_fields(self, component: str) -> np.ndarray:
        """Constant fields of ``component``, shape (12, constants, nside, nside)."""
        return np.stack(
            [
                np.load(self.package.resolve(f"{name}.npy"))
                for name in self.constants[component]
            ],
            axis=1,
        )

    def fused_kwargs(self, atmos_model_idx: int, ocean_model_idx: int) -> dict:
        """Constructor arguments of the fused :class:`DLESyM`."""
        # Normalization constants cover the full output variable set (prognostics
        # + diagnostics) in the same order as `output_coords`'s `variable` axis.
        center, scale = self.stats(
            self.variables["atmos"]
            + self.diagnostic_variables["atmos"]
            + self.variables["ocean"]
            + self.diagnostic_variables["ocean"]
        )
        return dict(
            atmos_model=self.core("atmos", atmos_model_idx),
            ocean_model=self.core("ocean", ocean_model_idx),
            hpx_lat=self.hpx_lat,
            hpx_lon=self.hpx_lon,
            nside=self.nside,
            center=center[None, None, None, :, None, None, None],
            scale=scale[None, None, None, :, None, None, None],
            atmos_constants=self.constant_fields("atmos"),
            ocean_constants=self.constant_fields("ocean"),
            atmos_input_times=self.input_times["atmos"],
            ocean_input_times=self.input_times["ocean"],
            atmos_output_times=self.output_times["atmos"],
            ocean_output_times=self.output_times["ocean"],
            atmos_variables=self.variables["atmos"],
            ocean_variables=self.variables["ocean"],
            atmos_coupling_variables=self.coupling_variables["atmos"],
            ocean_coupling_variables=self.coupling_variables["ocean"],
            atmos_diagnostic_variables=self.diagnostic_variables["atmos"],
            ocean_diagnostic_variables=self.diagnostic_variables["ocean"],
            use_cln=self.use_cln,
            condition_shape=self.condition_shape,
        )

    def _component_kwargs(self, component: str, index: int) -> dict:
        center, scale = self.stats(
            self.variables[component] + self.diagnostic_variables[component]
        )
        # A component normalizes the forcing it receives with the constants of
        # the component that produced it: the atmosphere's SST with the ocean's,
        # and the ocean's window means with the atmosphere's output constants.
        coupling_center, coupling_scale = self.stats(self.coupling_variables[component])
        return dict(
            core=self.core(component, index),
            hpx_lat=self.hpx_lat,
            hpx_lon=self.hpx_lon,
            nside=self.nside,
            center=center,
            scale=scale,
            coupling_center=coupling_center,
            coupling_scale=coupling_scale,
            constants=self.constant_fields(component),
            input_times=self.input_times[component],
            output_times=self.output_times[component],
            variables=self.variables[component],
            coupling_variables=self.coupling_variables[component],
            diagnostic_variables=self.diagnostic_variables[component],
            use_cln=self.use_cln,
            condition_shape=self.condition_shape,
        )

    def atmos_kwargs(self, atmos_model_idx: int) -> dict:
        """Constructor arguments of :class:`DLESyMAtmos`."""
        return self._component_kwargs("atmos", atmos_model_idx)

    def ocean_kwargs(self, ocean_model_idx: int) -> dict:
        """Constructor arguments of :class:`DLESyMOcean`."""
        return dict(
            self._component_kwargs("ocean", ocean_model_idx),
            atmos_output_times=self.output_times["atmos"],
        )


def _load_package_config(
    package: Package, version: str | None = None
) -> _PackageConfig:
    """Parse a DLESyM package's config, constants and normalization.

    Parameters
    ----------
    package : Package
        Package to read
    version : str, optional
        Checkpoint version listed in the package's top-level ``config.yaml``, or
        None for a package with one unversioned config at its root

    Returns
    -------
    _PackageConfig
        Everything but the network cores, which are loaded on request.
    """
    cfg = OmegaConf.load(Path(package.resolve("config.yaml")))
    directory = {"atmos": "", "ocean": ""}
    if version is not None:
        if version not in cfg.versions:
            raise ValueError(
                f"Unknown DLESyM checkpoint version '{version}', available "
                f"versions: {list(cfg.versions.keys())}"
            )
        version_cfg = cfg.versions[version]
        if version_cfg.get("deprecated", False):
            logger.warning(
                version_cfg.get(
                    "deprecation_message",
                    f"DLESyM checkpoint version '{version}' is deprecated.",
                )
            )
        cfg = OmegaConf.load(Path(package.resolve(f"{version_cfg.path}/config.yaml")))
        directory = {c: f"{version_cfg.path}/{c}/" for c in directory}

    def names(key: str) -> list[str]:
        return [_ATMOS_VARIABLE_RENAMES.get(v, v) for v in cfg.io.get(key, [])]

    cln = cfg.get("cln", {})
    use_cln = bool(cln.get("enabled", False))
    components = ("atmos", "ocean")
    return _PackageConfig(
        package=package,
        nside=cfg.data.nside,
        # Constant fields are shared across checkpoint versions, at the root
        hpx_lat=np.load(package.resolve("hpx_lat.npy")),
        hpx_lon=np.load(package.resolve("hpx_lon.npy")),
        scaling={
            _ATMOS_VARIABLE_RENAMES.get(name, name): (stats["mean"], stats["std"])
            for name, stats in cfg.data.scaling.items()
        },
        checkpoints={
            c: [directory[c] + ckpt for ckpt in cfg.models[f"{c}_model_checkpoints"]]
            for c in components
        },
        constants={c: list(cfg.io[f"{c}_constants"]) for c in components},
        input_times={
            c: np.array(cfg.io[f"{c}_input_times"], dtype="timedelta64[h]")
            for c in components
        },
        output_times={
            c: np.array(cfg.io[f"{c}_output_times"], dtype="timedelta64[h]")
            for c in components
        },
        variables={c: names(f"{c}_variables") for c in components},
        coupling_variables={c: names(f"{c}_coupling_variables") for c in components},
        diagnostic_variables={
            c: names(f"{c}_diagnostic_variables") for c in components
        },
        use_cln=use_cln,
        condition_shape=cln.get("condition_shape", None) if use_cln else None,
    )


def _wrap_output(
    out: torch.Tensor,
    leading: tuple[int, ...],
    signature: CoordinateSystem,
    like: xr.DataArray,
) -> xr.DataArray:
    """Restore the ``leading`` dimensions of a flattened core output and wrap it.

    Field data is fresh; metadata is deep-copied so nested values are not shared
    with ``like``.
    """
    out = out.reshape(*leading, *out.shape[1:])
    result = from_torch(out, signature, name=like.name)
    result.attrs = deepcopy(result.attrs)
    result.encoding = deepcopy(like.encoding)
    return result


class _CLNNoise:
    """Conditional layer norm noise generator of DLESyM and its components.

    The generator is the only random state of a rollout: ``set_rng`` seeds it,
    ``initialize`` and ``step`` return its position as explicit state, and
    ``_with_rng`` replays a step from that position without disturbing the
    generator the caller seeded.
    """

    stochastic: bool
    _cln_generator: torch.Generator | None

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Seed the generator used to sample conditional layer norm noise.

        Matches the `set_rng(seed, reset)` convention used by other
        stochastic prognostic models in earth2studio (e.g. FCN3), so
        callers (ensemble drivers, eval recipes) can reseed this model the
        same way as any other stochastic model via
        `hasattr(model, "set_rng")`. Has no effect if the model is not
        stochastic.

        Parameters
        ----------
        seed : int
            Seed value
        reset : bool, optional
            Reinitialize the generator from `seed`. If False and the
            generator is already initialized, this is a no-op, by default
            True
        """
        if reset or self._cln_generator is None:
            self._cln_generator = torch.Generator().manual_seed(seed)

    def _ensure_rng(self) -> None:
        """Seed from the global RNG when a stochastic model has no generator yet."""
        if self.stochastic and self._cln_generator is None:
            self.set_rng(int(torch.randint(2**31, ()).item()))

    def _rng_state(self) -> torch.Tensor | None:
        """The generator's position, or ``None`` for deterministic execution."""
        if self.stochastic and self._cln_generator is not None:
            return self._cln_generator.get_state().clone()
        return None

    def _with_rng(self, forward: Callable[[], T], state: torch.Tensor | None) -> T:
        """Run ``forward`` from the generator ``state`` an earlier call returned."""
        previous = self._cln_generator
        if state is not None:
            self._cln_generator = torch.Generator().set_state(state.clone())
        try:
            return forward()
        finally:
            self._cln_generator = previous


@check_optional_dependencies()
class DLESyM(torch.nn.Module, AutoModelMixin, PrognosticMixin, _CLNNoise):
    """DLESyM-V1-ERA5 prognostic model. This is an ensemble forecast model for
    global earth system modeling. This model includes an atmosphere and ocean
    component, using atmospheric variables as well as the sea-surface temperature
    on a HEALPix nside=64 (approximately 1 degree) resolution grid. The model
    architecture is a U-Net with padding operations modified to support using
    the HEALPix grid. Because the atmosphere and ocean models are predicted at
    different times, not all entries in the output tensor are valid. As a result,
    we provide convenience methods for retrieving the valid atmospheric and
    oceanic outputs.

    The default package provided for this model contains the checkpoints used
    in the ECMWF AI Weather Quest S2S competition. These checkpoints are trained
    with CRPS loss and use sampled random noise to produce ensemble variability
    each forward pass, seeded with the ``set_rng`` method.

    The model runs on the HEALPix grid with derived input variables. Its
    recommended source, :class:`DLESyMInputSource`, prepares them from a lat/lon
    source; regrid the outputs with a :class:`~earth2studio.grids.Regridder` if
    lat/lon fields are needed. The atmosphere and ocean are also available as
    the standalone :class:`DLESyMAtmos` and :class:`DLESyMOcean` models, coupled
    under :class:`~earth2studio.coupling.CoupledRunner`.

    Parameters
    ----------
    atmos_model : torch.nn.Module
        Atmosphere model
    ocean_model : torch.nn.Module
        Ocean model
    hpx_lat : np.ndarray
        HEALPix latitude coordinates, shape (12, nside, nside)
    hpx_lon : np.ndarray
        HEALPix longitude coordinates, shape (12, nside, nside)
    nside : int
        HEALPix nside
    center : np.ndarray
        Means of the full output variable set (prognostics + diagnostics,
        in the same order as `output_coords`'s `variable` axis), shape
        (1, 1, 1, num_output_variables, 1, 1, 1)
    scale : np.ndarray
        Standard deviations of the full output variable set, same shape
        and ordering as `center`
    atmos_constants : np.ndarray
        Constants for the atmosphere model, shape (12, num_atmos_constants, nside, nside)
    ocean_constants : np.ndarray
        Constants for the ocean model, shape (12, num_ocean_constants, nside, nside)
    atmos_input_times : np.ndarray
        Atmospheric input times, shape (num_atmos_input_times,)
    ocean_input_times : np.ndarray
        Ocean input times, shape (num_ocean_input_times,)
    atmos_output_times : np.ndarray
        Atmospheric output times, shape (num_atmos_output_times,)
    ocean_output_times : np.ndarray
        Ocean output times, shape (num_ocean_output_times,)
    atmos_variables : list[str]
        Atmospheric variables
    ocean_variables : list[str]
        Ocean variables
    atmos_coupling_variables : list[str]
        Atmospheric coupling variables
    ocean_coupling_variables : list[str]
        Ocean coupling variables
    atmos_diagnostic_variables : list[str], optional
        Atmospheric diagnostic output variables. These are produced by the
        atmos model but, unlike `atmos_variables`, are not fed back in as
        input to the next autoregressive step, by default []
    ocean_diagnostic_variables : list[str], optional
        Ocean diagnostic output variables, analogous to
        `atmos_diagnostic_variables`, by default []
    use_cln : bool, optional
        Whether the atmos/ocean models use conditional layer norm, which
        requires sampling and passing noise to the model to produce
        ensemble variability from a single set of weights, by default False
    condition_shape : int, optional
        Dimension of the conditional layer norm noise vector. Required if
        `use_cln` is True, by default None

    Note
    ----
    For more information about this model see:

    - https://arxiv.org/abs/2409.16247
    - https://arxiv.org/abs/2311.06253v2
    - https://huggingface.co/nvidia/dlesym-v1-era5

    For more information about the HEALPix grid see:

    - https://github.com/NVlabs/earth2grid

    Example
    -------
    ```python
    pkg = DLESyM.load_default_package()
    model = DLESyM.load_model(pkg)

    # Create iterator
    iterator = model.create_iterator(x)

    for step, x in enumerate(iterator):
        if step > 0:
            # Valid atmos and ocean predictions with their respective coordinates extracted below
            atmos_outputs = model.retrieve_valid_atmos_outputs(x)
            ocean_outputs = model.retrieve_valid_ocean_outputs(x)
            ...

    ```
    Badges
    ------
    region:global class:subseasonal-seasonal product:wind product:temp product:atmos product:ocean year:2025
    gpu:40gb
    provider:nvidia backend:pytorch
    """

    def __init__(
        self,
        atmos_model: torch.nn.Module,
        ocean_model: torch.nn.Module,
        hpx_lat: np.ndarray,
        hpx_lon: np.ndarray,
        nside: int,
        center: np.ndarray,
        scale: np.ndarray,
        atmos_constants: np.ndarray,
        ocean_constants: np.ndarray,
        atmos_input_times: np.ndarray,
        ocean_input_times: np.ndarray,
        atmos_output_times: np.ndarray,
        ocean_output_times: np.ndarray,
        atmos_variables: list[str],
        ocean_variables: list[str],
        atmos_coupling_variables: list[str],
        ocean_coupling_variables: list[str],
        atmos_diagnostic_variables: list[str] | None = None,
        ocean_diagnostic_variables: list[str] | None = None,
        use_cln: bool = False,
        condition_shape: int | None = None,
    ):

        super().__init__()
        self.atmos_model = atmos_model.eval()
        self.ocean_model = ocean_model.eval()

        if use_cln and condition_shape is None:
            raise ValueError("'condition_shape' is required when 'use_cln' is True")
        self.use_cln = use_cln
        self.condition_shape = condition_shape
        self._cln_generator: torch.Generator | None = None

        self.register_buffer("center", torch.from_numpy(center).to(dtype=torch.float32))
        self.register_buffer("scale", torch.from_numpy(scale).to(dtype=torch.float32))
        self.register_buffer(
            "atmos_constants", torch.from_numpy(atmos_constants).to(dtype=torch.float32)
        )
        self.register_buffer(
            "ocean_constants", torch.from_numpy(ocean_constants).to(dtype=torch.float32)
        )

        self.hpx_lat = hpx_lat
        self.hpx_lon = hpx_lon
        self.nside = nside
        self.atmos_variables = atmos_variables
        self.ocean_variables = ocean_variables
        self.atmos_coupling_variables = atmos_coupling_variables
        self.ocean_coupling_variables = ocean_coupling_variables
        self.atmos_diagnostic_variables = atmos_diagnostic_variables or []
        self.ocean_diagnostic_variables = ocean_diagnostic_variables or []

        # Validate the input and output times
        for name, times in zip(
            [
                "atmos_input_times",
                "ocean_input_times",
                "atmos_output_times",
                "ocean_output_times",
            ],
            [
                atmos_input_times,
                ocean_input_times,
                atmos_output_times,
                ocean_output_times,
            ],
        ):
            if not np.issubdtype(times.dtype, np.timedelta64):
                raise ValueError(
                    f"Input and output times must be of type 'timedelta64', got {times.dtype} for {name}"
                )
        if ocean_input_times[0] > atmos_input_times[0]:
            raise ValueError(
                f"Ocean input time must be equal to or before atmos input time, got {ocean_input_times[0]} and {atmos_input_times[0]}"
            )
        if len(atmos_input_times) < len(ocean_input_times):
            raise ValueError(
                f"Atmos input times must be equal to or longer than ocean input times, got {len(atmos_input_times)} and {len(ocean_input_times)}"
            )
        if len(atmos_output_times) < len(ocean_output_times):
            raise ValueError(
                f"Atmos output times must be equal to or longer than ocean output times, got {len(atmos_output_times)} and {len(ocean_output_times)}"
            )

        self.atmos_input_times = atmos_input_times
        self.ocean_input_times = ocean_input_times
        # Full input times are a merge of the atmos and ocean input times:
        #  - Need to go as far back as -48h for the first ocean input time
        #  - Need to use the same temporal resolution (6h) as the atmos input times
        self.full_input_times = np.arange(
            self.ocean_input_times[0],
            self.atmos_input_times[-1] + 1,
            atmos_input_times[1] - atmos_input_times[0],
        )
        self.atmos_output_times = atmos_output_times
        self.ocean_output_times = ocean_output_times

        # Setup the insolation and coupling times from the input and output times
        self.atmos_sol_times = np.concatenate(
            [self.atmos_input_times, self.atmos_output_times]
        )
        self.ocean_sol_times = np.concatenate(
            [self.ocean_input_times, self.ocean_output_times]
        )
        n_atmos_steps = 1 + max(
            self.atmos_model.output_time_dim // self.atmos_model.input_time_dim, 1
        )
        self.atmos_coupling_times = np.array(
            [self.atmos_input_times[-1]] * n_atmos_steps
        )
        self.ocean_coupling_times = self.atmos_output_times

        # Setup the lead time indices for [atmos, ocean] [input, coupled input, output]
        in_signature = self.input_coords()
        out_signature = self.output_coords(in_signature)
        in_coords = {k: v.values for k, v in in_signature.coords.items()}
        out_coords = {k: v.values for k, v in out_signature.coords.items()}
        out_coords["variable"] = np.array(
            self.atmos_variables
            + self.atmos_diagnostic_variables
            + self.ocean_variables
            + self.ocean_diagnostic_variables
        )
        self.atmos_input_lt_idx = [
            list(in_coords["lead_time"]).index(t) for t in self.atmos_input_times
        ]
        self.ocean_input_lt_idx = [
            list(in_coords["lead_time"]).index(t) for t in self.ocean_input_times
        ]
        self.atmos_coupled_input_lt_idx = [
            list(in_coords["lead_time"]).index(t) for t in self.atmos_coupling_times
        ]
        self.ocean_coupled_input_lt_idx = [
            list(out_coords["lead_time"]).index(t) for t in self.ocean_coupling_times
        ]
        self.atmos_output_lt_idx = [
            list(out_coords["lead_time"]).index(t) for t in self.atmos_output_times
        ]
        self.register_buffer(
            "ocean_output_lt_idx",
            torch.tensor(
                [
                    list(out_coords["lead_time"]).index(t)
                    for t in self.ocean_output_times
                ],
                dtype=torch.long,
            ),
        )

        # Setup the variable indices for [atmos, ocean]
        # Input-side indices select prognostic variables out of `x`, which
        # always carries the canonical `atmos_variables + ocean_variables`
        # layout (diagnostics are dropped before being fed back in -- see
        # `_next_step_inputs`). This must NOT be sourced from
        # `self.input_coords()`: a subclass may report a different public
        # variable set there (e.g. `ttr` instead of `rlut`) that only exists
        # before `_initial_state` converts it to the canonical set --
        # `prepare_input_data` itself only ever sees the canonical layout.
        # `prognostic_variables` is `atmos_variables + ocean_variables` by
        # construction, so each component's own indices within it are just
        # the contiguous range it occupies.
        prognostic_variables = self.atmos_variables + self.ocean_variables
        self.atmos_var_idx = list(range(len(self.atmos_variables)))
        self.ocean_var_idx = list(
            range(len(self.atmos_variables), len(prognostic_variables))
        )
        self.atmos_coupling_var_idx = [
            prognostic_variables.index(var) for var in self.atmos_coupling_variables
        ]
        # Ocean coupling is built from the atmos model's own raw output
        # tensor, whose channel layout is
        # [atmos_variables, atmos_diagnostic_variables] regardless of
        # where those variables land in the merged `out_coords`.
        self.ocean_coupling_var_idx = [
            list(self.atmos_variables + self.atmos_diagnostic_variables).index(var)
            for var in self.ocean_coupling_variables
        ]

        # Output-side indices scatter each component's raw output tensor
        # (channel layout [prognostics, diagnostics]) into the merged
        # output tensor, whose variable axis follows `out_coords["variable"]`.
        # Registered as buffers (rather than plain lists) so they follow
        # the module to whatever device it's moved to.
        self.register_buffer(
            "atmos_output_var_idx",
            torch.tensor(
                [
                    list(out_coords["variable"]).index(var)
                    for var in self.atmos_variables + self.atmos_diagnostic_variables
                ],
                dtype=torch.long,
            ),
        )
        self.register_buffer(
            "ocean_output_var_idx",
            torch.tensor(
                [
                    list(out_coords["variable"]).index(var)
                    for var in self.ocean_variables + self.ocean_diagnostic_variables
                ],
                dtype=torch.long,
            ),
        )

        # Diagnostic-only output variables are dropped before being fed
        # back in as the next step's input.
        self._has_diagnostic_variables = bool(self.atmos_diagnostic_variables) or bool(
            self.ocean_diagnostic_variables
        )
        self.register_buffer(
            "_prognostic_out_idx",
            torch.tensor(
                [
                    list(out_coords["variable"]).index(var)
                    for var in self.atmos_variables + self.ocean_variables
                ],
                dtype=torch.long,
            ),
        )

        # `self.center`/`self.scale` cover the full output variable set
        # (prognostics + diagnostics, in `out_coords["variable"]` order) --
        # correct for `_denormalize_output`, but `_normalize_input` needs
        # the prognostic-only subset in `input_coords`'s order to match
        # the input tensor's (smaller) variable axis.
        self.register_buffer(
            "input_center", self.center.index_select(3, self._prognostic_out_idx)
        )
        self.register_buffer(
            "input_scale", self.scale.index_select(3, self._prognostic_out_idx)
        )

        # Output channel of the log-transformed precipitation diagnostic
        # (see `_denormalize_output`), None if the model has no `tp06` output
        out_variables = list(out_coords["variable"])
        self._tp06_out_idx = (
            out_variables.index("tp06") if "tp06" in out_variables else None
        )

    def default_sources(self) -> DataSource:
        """Recommend ARCO ERA5, prepared as the model's native inputs.

        Returns
        -------
        DataSource
            :class:`DLESyMInputSource` over ARCO ERA5: it computes the derived
            variables, fills sea surface temperature over land and regrids onto
            HEALPix. Wrap another lat/lon source in :class:`DLESyMInputSource`
            to serve the model from it.
        """
        return DLESyMInputSource(nside=self.nside)

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the prognostic model.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray input signature.
        """
        return coord_array(
            ("batch", "time", "lead_time", "variable", "face", "height", "width"),
            {
                "lead_time": self.full_input_times,
                "variable": np.array(self.atmos_variables + self.ocean_variables),
            },
            dynamic=("batch", "time"),
            grid=_healpix_grid(self.nside),
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
            Allocation-free DataArray output signature.
        """

        handshake_time(input_coords, allow_dynamic=True)
        handshake_time(input_coords, "lead_time")
        lead = input_coords.lead_time
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead.values - lead.values[-1]),
            self.input_coords(),
        )
        return coord_array_like(
            input_coords,
            {
                "lead_time": lead.values[-1] + self.atmos_output_times,
                "variable": np.array(
                    self.atmos_variables
                    + self.atmos_diagnostic_variables
                    + self.ocean_variables
                    + self.ocean_diagnostic_variables
                ),
            },
        )

    @classmethod
    def load_default_package(cls) -> Package:
        """Default DLESyM model package on NGC

        The package's top-level ``config.yaml`` lists the available
        checkpoint versions; see :func:`load_model` for how to select
        between them.
        """
        package = Package(
            "hf://nvidia/dlesym-v1-era5@b88155cfc2c988a2a9058d5e35a41220e2b01941",
            cache_options={
                "cache_storage": Package.default_cache("dlesym"),
            },
        )
        return package

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        atmos_model_idx: int = 0,
        ocean_model_idx: int = 0,
        version: Literal["v1.0", "v1.1"] = "v1.1",
    ) -> PrognosticModel:
        """Load prognostic from package

        Parameters
        ----------
        package : Package
            Package to load model from
        version : {"v1.0", "v1.1"}, optional
            Checkpoint version to load; see each version's entry in the
            package's `config.yaml` for details. `v1.1` is the checkpoint
            submitted to the ECMWF AI Weather Quest competition
            (https://aiweatherquest.ecmwf.int/). `v1.0` is the previous
            checkpoint, kept for reproducibility; loading it logs a
            deprecation warning, by default "v1.1"
        atmos_model_idx : int, optional
            Index of atmos model weights to load. Only meaningful for
            checkpoint versions that ship multiple atmos checkpoints
            (used to build ensembles without conditional layer norm), by
            default 0
        ocean_model_idx : int, optional
            Index of ocean model weights to load. Only meaningful for
            checkpoint versions that ship multiple ocean checkpoints, by
            default 0

        Returns
        -------
        PrognosticModel
            Prognostic model
        """

        cfg = _load_package_config(package, version)
        return cls(**cfg.fused_kwargs(atmos_model_idx, ocean_model_idx))

    def prepare_input_data(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Prepare input data for the atmos and ocean models.
        From the data in `x`, we will build a list of tensors for each model.
        Assumes `x` is a tensor of shape (batch, time, lead_time, variable, face, height, width).
        The list order is [state, insolation, constants, coupled_inputs].
        Models expect input data in the following shapes:
            state: (batch, face, lead_time, variable, height, width)
            insolation: (batch, face, lead_time, variable(=1), height, width)
            constants: (face, variable, height, width)
            coupled_inputs: (lead_time, batch, variable, face, height, width)

        Parameters
        ----------
        x : torch.Tensor
            Input data
        coords : CoordSystem
            Input coordinates

        Returns
        -------
        tuple[List[torch.Tensor], List[torch.Tensor]]
            Prepared input data for atmos and ocean models, respectively
        """

        handshake_dim(
            coords,
            ("batch", "time", "lead_time", "variable", "face", "height", "width"),
        )

        # Flatten the batch and time dimensions
        # Before flattening, check if we have multiple batch elements
        if x.shape[0] > 1:
            # Roll the times for each batch element into one array for computing insolation
            stacked_times = np.concatenate([coords["time"]] * x.shape[0], axis=0)
        else:
            stacked_times = coords["time"]

        x = x.reshape(-1, *x.shape[2:])

        # Atmos inputs: state, insolation, constants, coupled inputs
        atmos_state = x[:, self.atmos_input_lt_idx][
            ..., self.atmos_var_idx, :, :, :
        ].permute(0, 3, 1, 2, 4, 5)
        atmos_insolation = self._make_insolation_tensor(
            anchor_times=stacked_times,
            timedeltas=self.atmos_sol_times + coords["lead_time"][-1],
        )
        atmos_coupling = self._make_atmos_coupling(x, coords)
        atmos_inputs = list(
            map(
                lambda y: y.to(x.device, x.dtype),
                [atmos_state, atmos_insolation, self.atmos_constants, atmos_coupling],
            )
        )

        # Ocean inputs: state, insolation, constants, coupled inputs
        # Coupling is not set as the ocean coupling comes after the atmos model forward pass
        ocean_state = x[:, self.ocean_input_lt_idx][
            ..., self.ocean_var_idx, :, :, :
        ].permute(0, 3, 1, 2, 4, 5)
        ocean_insolation = self._make_insolation_tensor(
            anchor_times=stacked_times,
            timedeltas=self.ocean_sol_times + coords["lead_time"][-1],
        )
        ocean_inputs = list(
            map(
                lambda y: y.to(x.device, x.dtype),
                [ocean_state, ocean_insolation, self.ocean_constants],
            )
        )

        return atmos_inputs, ocean_inputs

    def prepare_output_data(
        self,
        atmos_outputs: torch.Tensor,
        ocean_outputs: torch.Tensor,
        coords: CoordSystem,
    ) -> torch.Tensor:
        """Prepare output data for the atmos and ocean models.
        From the data in `atmos_outputs` and `ocean_outputs`, we will build a tensor of shape (batch, time, lead_time, variable, face, height, width).
        Assumes the atmosphere has the finest temporal resolution over output lead times.
        Expects input data to be of shape (batch * time, face, lead_time, variable, height, width).

        Parameters
        ----------
        atmos_outputs : torch.Tensor
            Atmos outputs
        ocean_outputs : torch.Tensor
            Ocean outputs
        coords : CoordSystem
            Input coordinates

        Returns
        -------
        output_data : torch.Tensor
            Output data
        """

        # Variable dim covers the full output variable set (prognostics +
        # diagnostics for each component), which is larger than
        # `coords["variable"]` (the input, prognostic-only variable set)
        # whenever diagnostic output variables are configured.
        n_output_vars = len(self.atmos_output_var_idx) + len(self.ocean_output_var_idx)
        output_data = torch.full(
            (
                len(coords["batch"]),
                len(coords["time"]),
                len(self.atmos_output_times),
                n_output_vars,
                len(coords["face"]),
                len(coords["height"]),
                len(coords["width"]),
            ),
            float("nan"),
            device=atmos_outputs.device,
        )

        # Reorder the face dim to after variable dim, restore the batch and time dimensions
        atmos_outputs = atmos_outputs.permute(0, 2, 3, 1, 4, 5)
        ocean_outputs = ocean_outputs.permute(0, 2, 3, 1, 4, 5)
        atmos_outputs = atmos_outputs.reshape(
            len(coords["batch"]), len(coords["time"]), *atmos_outputs.shape[1:]
        )
        ocean_outputs = ocean_outputs.reshape(
            len(coords["batch"]), len(coords["time"]), *ocean_outputs.shape[1:]
        )

        output_data[:, :, :, self.atmos_output_var_idx, :, :, :] = atmos_outputs
        for src_idx, dst_idx in enumerate(self.ocean_output_var_idx):
            output_data[:, :, self.ocean_output_lt_idx, dst_idx, :, :, :] = (
                ocean_outputs[:, :, :, src_idx, :, :, :]
            )

        return output_data

    @check_optional_dependencies()
    def _make_insolation_tensor(
        self, anchor_times: np.ndarray, timedeltas: np.ndarray
    ) -> torch.Tensor:
        """Make insolation tensor from anchor times and timedeltas

        Parameters
        ----------
        anchor_times : np.ndarray
            Anchor times
        timedeltas : np.ndarray
            Timedeltas

        Returns
        -------
        torch.Tensor
            Insolation tensor of shape ("batch" (anchor_times), face, lead_time, variable, height, width)
        """

        return _insolation_tensor(anchor_times, timedeltas, self.hpx_lat, self.hpx_lon)

    def _make_atmos_coupling(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> torch.Tensor:
        """Make atmos coupling tensor from input data
        Assumes `x` is a tensor of shape (batch, lead_time, variable, face, height, width).
        Also assumes `x` has been properly normalized.

        Parameters
        ----------
        x : torch.Tensor
            Input data
        coords : CoordSystem
            Input coordinates

        Returns
        -------
        torch.Tensor
            Atmos coupling tensor: shape (lead_time, batch, variable, face, height, width)
        """

        atmos_coupling = x[:, self.atmos_coupled_input_lt_idx][
            ..., self.atmos_coupling_var_idx, :, :, :
        ].permute(1, 0, 2, 3, 4, 5)

        return atmos_coupling

    def _make_ocean_coupling(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> torch.Tensor:
        """Make ocean coupling tensor from atmos outputs
        Assumes `x` is a tensor of shape (batch, face, lead_time, variable, height, width).
        Also assumes `x` has been properly normalized.

        Parameters
        ----------
        x : torch.Tensor
            Input data
        coords : CoordSystem
            Input coordinates

        Returns
        -------
        torch.Tensor
            Ocean coupling tensor: shape (lead_time, batch, variable, face, height, width)
        """

        # Subselect the ocean coupling variables -- we use all atmos lead times here so we can average over them
        ocean_coupling = x[:, :, :, self.ocean_coupling_var_idx, :, :]

        # Slice along the lead_time dim, average, and stack along the variable dim
        # (different time-averaged quantities are defined as different variables)
        slices = ocean_coupling.chunk(len(self.ocean_output_times), dim=2)
        ocean_coupling = torch.concat(
            [s.mean(dim=2, keepdim=True) for s in slices], dim=3
        )
        ocean_coupling = ocean_coupling.permute(2, 0, 3, 1, 4, 5)
        return ocean_coupling

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input data"""

        return (x - self.input_center) / self.input_scale

    def _denormalize_output(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize output data and convert `tp06` to meters"""

        x = x * self.scale + self.center
        if self._tp06_out_idx is not None:
            _tp06_to_meters(x, 3, self._tp06_out_idx)
        return x

    def retrieve_valid_ocean_outputs(self, x: xr.DataArray) -> xr.DataArray:
        """Retrieve the valid ocean model outputs from an output data tensor.
        Because we use a dense grid of output times for the coupled model, some of the output times
        for the ocean model may not be valid because it takes a coarser time-step.
        This function will retrieve the valid outputs and return them in a tensor of shape (batch, time, lead_time, variable, face, height, width).

        Parameters
        ----------
        x : torch.Tensor
            Output data tensor
        coords : CoordSystem
            Input coordinates

        Returns
        -------
        torch.Tensor
            Ocean outputs
        CoordSystem
            Output coordinates
        """

        self._validate_output_coords(x)
        result = x.isel(
            variable=self.ocean_output_var_idx.cpu().numpy(),
            lead_time=self.ocean_output_lt_idx.cpu().numpy(),
        )
        result.attrs = {
            k: v
            for k, v in coord_array_like(result).attrs.items()
            if k
            not in (
                "earth2studio_kind",
                "earth2studio_schema_version",
                "earth2studio_dynamic_dims",
            )
        }
        return result

    def retrieve_valid_atmos_outputs(self, x: xr.DataArray) -> xr.DataArray:
        """Retrieve the valid atmospheric model outputs from an output data tensor.
        This function will retrieve the valid outputs and return them in a tensor of shape (batch, time, lead_time, variable, face, height, width).

        Parameters
        ----------
        x : torch.Tensor
            Output data tensor
        coords : CoordSystem
            Input coordinates

        Returns
        -------
        torch.Tensor
            Atmos outputs
        CoordSystem
            Output coordinates
        """

        self._validate_output_coords(x)
        result = x.isel(variable=self.atmos_output_var_idx.cpu().numpy())
        result.attrs = {
            k: v
            for k, v in coord_array_like(result).attrs.items()
            if k
            not in (
                "earth2studio_kind",
                "earth2studio_schema_version",
                "earth2studio_dynamic_dims",
            )
        }
        return result

    def _validate_output_coords(self, coords: CoordinateSystem) -> None:
        """Validate the coordinates passed to the output subselection methods

        Parameters
        ----------
        coords : CoordSystem
            Output coordinates to be validated

        Raises
        ------
        ValueError
            If the coordinates are invalid (missing or incorrect length lead_time dim)
        """
        handshake_time(coords, "lead_time")
        handshake_size(coords, "lead_time", len(self.atmos_output_times))

    @property
    def stochastic(self) -> bool:  # type: ignore[override]
        """Whether conditional layer norm noise is sampled during a rollout."""
        return bool(self.use_cln)

    def _sample_conditions_cln(
        self,
        batch_size: int,
        n_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        """Draw a fresh conditional layer norm noise tensor for each
        internal integration step of a component model.

        Draws are pulled from `self._cln_generator` when set via
        `set_rng`, otherwise fall back to the global RNG. Nothing is
        cached: each call advances the generator's state, so a full
        autoregressive rollout for a given seed produces a unique,
        reproducible noise trajectory across all steps rather than a
        single fixed draw held constant for the whole forecast.

        Parameters
        ----------
        batch_size : int
            Batch size, matching the target model's flattened
            (batch * time) input dimension
        n_steps : int
            Number of noise tensors to draw, matching the target model's
            `integration_steps`
        device : torch.device
            Target device
        dtype : torch.dtype
            Target dtype

        Returns
        -------
        list[torch.Tensor]
            List of length `n_steps`, each of shape
            (batch_size, condition_shape)
        """
        return [
            torch.randn(
                batch_size, self.condition_shape, generator=self._cln_generator
            ).to(device=device, dtype=dtype)
            for _ in range(n_steps)
        ]

    @torch.inference_mode()
    def _forward_tensor(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
    ) -> torch.Tensor:

        x = self._normalize_input(x)

        # Forward pass of atmos model first
        atmos_inputs, ocean_inputs = self.prepare_input_data(x, coords)
        if self.use_cln:
            atmos_conditions = self._sample_conditions_cln(
                batch_size=atmos_inputs[0].shape[0],
                n_steps=self.atmos_model.integration_steps,
                device=x.device,
                dtype=x.dtype,
            )
            atmos_outputs = self.atmos_model(
                atmos_inputs, conditions_cln=atmos_conditions
            )
        else:
            atmos_outputs = self.atmos_model(atmos_inputs)

        # Use atmos outputs as coupling for ocean, run forward pass of ocean model
        ocean_inputs.append(self._make_ocean_coupling(atmos_outputs, coords))
        if self.use_cln:
            ocean_conditions = self._sample_conditions_cln(
                batch_size=ocean_inputs[0].shape[0],
                n_steps=self.ocean_model.integration_steps,
                device=x.device,
                dtype=x.dtype,
            )
            ocean_outputs = self.ocean_model(
                ocean_inputs, conditions_cln=ocean_conditions
            )
        else:
            ocean_outputs = self.ocean_model(ocean_inputs)

        output_data = self.prepare_output_data(atmos_outputs, ocean_outputs, coords)
        output_data = self._denormalize_output(output_data)

        return output_data

    def __call__(
        self,
        x: xr.DataArray,
    ) -> xr.DataArray:
        """Run one complete coupled cycle without iterator hooks.

        Parameters
        ----------
        x : xr.DataArray
            Input history on the declared grid, with arbitrary leading dimensions.

        Returns
        -------
        xr.DataArray
            Complete coupled-cycle prediction. Invalid ocean times contain NaNs.
        """

        return self.initialize(x)[0]

    def initialize(self, x: xr.DataArray) -> tuple[xr.DataArray, torch.Tensor | None]:
        """Compute a coupled forecast and save the continuation random stream.

        Parameters
        ----------
        x : xr.DataArray
            Initial coupled history matching ``input_coords()`` on the native grid.

        Returns
        -------
        tuple[xr.DataArray, torch.Tensor | None]
            Complete coupled forecast chunk and conditional-layer-normalization
            RNG state, or ``None`` for deterministic execution. No hooks run.
        """
        handshake_nonempty(x)
        self.output_coords(x)
        return self._forward(self._initial_state(x))

    def step(
        self, y: xr.DataArray, state: torch.Tensor | None
    ) -> tuple[xr.DataArray, torch.Tensor | None]:
        """Advance the coupled components using the preceding output history.

        Parameters
        ----------
        y : xr.DataArray
            Previous complete forecast chunk in native variables, including
            caller-applied edits. Initial-input conversions are not repeated.
        state : torch.Tensor | None
            RNG state returned by ``initialize`` or ``step``, or ``None`` for
            deterministic execution.

        Returns
        -------
        tuple[xr.DataArray, torch.Tensor | None]
            Next coupled forecast chunk and updated RNG state, without hooks.
        """
        x = y.isel(lead_time=slice(-len(self.full_input_times), None)).sel(
            variable=self.atmos_variables + self.ocean_variables
        )
        return self._with_rng(lambda: self._forward(x), state)

    def _forward(self, x: xr.DataArray) -> tuple[xr.DataArray, torch.Tensor | None]:
        handshake_nonempty(x)
        handshake_dataarray(
            x.assign_coords(lead_time=x.lead_time.values - x.lead_time.values[-1]),
            DLESyM.input_coords(self),
        )
        self._ensure_rng()
        return self._forward_array(x), self._rng_state()

    def _initial_state(self, x: xr.DataArray) -> xr.DataArray:
        return x

    def _forward_array(self, x: xr.DataArray) -> xr.DataArray:
        handshake_time(x)
        tensor, _ = x.e2s.to_torch()
        # Preserve arbitrary leading axes, including axes without labels.
        leading = x.dims[:-6]
        tensor = tensor.to(self.center.device).reshape(-1, *tensor.shape[-6:])
        coords = OrderedDict(
            batch=np.arange(tensor.shape[0]),
            **{d: x.coords[d].values for d in x.dims[-6:]},
        )
        out = self._forward_tensor(tensor, coords)
        signature = coord_array_like(
            x,
            {
                "lead_time": x.lead_time.values[-1] + self.atmos_output_times,
                "variable": np.array(
                    self.atmos_variables
                    + self.atmos_diagnostic_variables
                    + self.ocean_variables
                    + self.ocean_diagnostic_variables
                ),
            },
        )
        return _wrap_output(out, tuple(x.sizes[d] for d in leading), signature, x)

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Create a forecasts-only iterator over complete coupled cycles.

        Parameters
        ----------
        x : xr.DataArray
            Initial history matching ``input_coords()`` on the native grid.


        Yields
        ------
        xr.DataArray
            Complete coupled cycles starting with initialization's prediction.
            The rear hook runs before every yield and the front hook before each
            subsequent step; both hooks feed recurrence directly.
        """
        return self._default_create_iterator(x)


class DLESyMInputSource:
    """Serve DLESyM's native HEALPix input variables from a lat/lon source.

    DLESyM and its components declare their inputs on the HEALPix grid, with
    derived variables that no raw data source provides. This source fetches the
    base variables from a lat/lon source and prepares them:

    - ``ws10m`` is computed from ``u10m`` and ``v10m`` and ``tau300-700`` from
      ``z300`` and ``z700``;
    - ``sst`` is filled over land by interpolating along longitude and latitude;
    - every field is regridded bilinearly onto the HEALPix grid.

    Any other variable, such as ``ttr03`` or ``ttr``, is requested from the
    wrapped source under the same name and regridded. Only the base variables a
    request needs are fetched. It is what the native models recommend from
    ``default_sources()``, wrapping ARCO ERA5; wrap another source to serve the
    models from it.

    Parameters
    ----------
    source : DataSource, optional
        Source of the base variables on a regular lat/lon grid, with ``lat`` and
        ``lon`` its trailing dimensions, by default
        :class:`~earth2studio.data.ARCO_ERA5`
    nside : int, optional
        HEALPix nside of the model the source serves, by default 64

    Example
    -------
    ```python
    model = DLESyM.load_model(DLESyM.load_default_package())
    source = DLESyMInputSource(WB2ERA5(), nside=model.nside)
    Pipeline.from_model(model, source, output={"forecast": io}).run(items)
    ```
    """

    # Native variable -> (base variables, function of the base fields)
    _DERIVED: dict[str, tuple[tuple[str, ...], Callable[..., xr.DataArray]]] = {
        "ws10m": (("u10m", "v10m"), lambda u, v: np.sqrt(u**2 + v**2)),
        "tau300-700": (("z300", "z700"), lambda upper, lower: upper - lower),
    }

    @check_optional_dependencies()
    def __init__(self, source: DataSource | None = None, nside: int = 64) -> None:
        if source is None:
            from earth2studio.data import ARCO_ERA5

            source = ARCO_ERA5()
        self.source = source
        self.nside = nside
        self._grid = earth2grid.healpix.Grid(
            level=int(np.log2(nside)),
            pixel_order=earth2grid.healpix.HEALPIX_PAD_XY,
        )
        # One regridder per source grid, built on the first fetch.
        self._regridder: tuple[tuple[bytes, bytes], torch.nn.Module] | None = None

    def _base_variables(self, variables: list[str]) -> list[str]:
        """The source variables needed to prepare ``variables``, without repeats."""
        base: list[str] = []
        for name in variables:
            for v in self._DERIVED.get(name, ((name,), None))[0]:
                if v not in base:
                    base.append(v)
        return base

    @staticmethod
    def _fill_sst(sst: xr.DataArray) -> xr.DataArray:
        """Fill NaNs over land by interpolating along longitude, then latitude."""
        sst = xr.DataArray(np.asarray(sst.values), dims=sst.dims)
        filled = sst.interpolate_na(dim="lon", method="linear", use_coordinate=False)
        # Second pass across the longitude seam, third pass along latitude.
        for dim in ("lon", "lat"):
            shift = filled.sizes[dim] // 2
            filled = (
                filled.roll({dim: shift}, roll_coords=False)
                .interpolate_na(dim=dim, method="linear", use_coordinate=False)
                .roll({dim: filled.sizes[dim] - shift}, roll_coords=False)
            )
        return filled

    def _to_hpx(
        self, values: np.ndarray, lat: np.ndarray, lon: np.ndarray
    ) -> torch.Tensor:
        """Regrid ``(..., lat, lon)`` values onto ``(..., face, height, width)``."""
        key = (lat.tobytes(), lon.tobytes())
        if self._regridder is None or self._regridder[0] != key:
            grid = earth2grid.latlon.LatLonGrid(lat.tolist(), lon.tolist())
            regridder = earth2grid.get_regridder(grid, self._grid).to(torch.float32)
            self._regridder = (key, regridder)
        x = torch.from_numpy(values).reshape(-1, len(lat), len(lon))
        x = self._regridder[1](x)
        return x.reshape(*values.shape[:-2], 12, self.nside, self.nside)

    def _prepare(self, x: xr.DataArray, variables: list[str]) -> xr.DataArray:
        """Derive, fill and regrid the fetched base fields into ``variables``."""
        missing = [d for d in ("variable", "lat", "lon") if d not in x.dims]
        if missing or x["lat"].ndim != 1 or x["lon"].ndim != 1:
            raise ValueError(
                "DLESyMInputSource needs a source on a regular lat/lon grid with "
                f"'variable', 'lat' and 'lon' dimensions; got {tuple(x.dims)}"
            )
        x = x.e2s.as_numpy()
        leading = [d for d in x.dims if d not in ("variable", "lat", "lon")]
        x = x.transpose(*leading, "variable", "lat", "lon")
        fields = []
        for name in variables:
            if name in self._DERIVED:
                base, derive = self._DERIVED[name]
                field = derive(*(x.sel(variable=v) for v in base))
            else:
                field = x.sel(variable=name)
            if name == "sst":
                field = self._fill_sst(field)
            fields.append(np.asarray(field.values, dtype=np.float32))
        values = self._to_hpx(
            np.stack(fields, axis=len(leading)), x["lat"].values, x["lon"].values
        )
        coords = coord_array(
            (*leading, "variable", "face", "height", "width"),
            {**{d: x[d].values for d in leading}, "variable": np.array(variables)},
            grid=_healpix_grid(self.nside),
        )
        return from_torch(values, coords)

    def __call__(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Return the native variables requested at each time.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Times to return data for
        variable : str | list[str] | VariableArray
            Native variables, such as those of :meth:`DLESyM.input_coords`

        Returns
        -------
        xr.DataArray
            Fields with dimensions ``(time, variable, face, height, width)``
        """
        variables = [str(v) for v in np.atleast_1d(np.asarray(variable))]
        return self._prepare(
            self.source(time, self._base_variables(variables)), variables
        )

    async def fetch(
        self,
        time: datetime | list[datetime] | TimeArray,
        variable: str | list[str] | VariableArray,
    ) -> xr.DataArray:
        """Async form of :meth:`__call__`.

        The lat/lon source is fetched asynchronously when it supports it and
        called synchronously otherwise.

        Parameters
        ----------
        time : datetime | list[datetime] | TimeArray
            Times to return data for
        variable : str | list[str] | VariableArray
            Native variables, such as those of :meth:`DLESyM.input_coords`

        Returns
        -------
        xr.DataArray
            Fields with dimensions ``(time, variable, face, height, width)``
        """
        variables = [str(v) for v in np.atleast_1d(np.asarray(variable))]
        base = self._base_variables(variables)
        if hasattr(self.source, "fetch"):
            x = await self.source.fetch(time, base)
        else:
            x = self.source(time, base)
        return self._prepare(x, variables)


class _DLESyMComponent(torch.nn.Module, AutoModelMixin, PrognosticMixin, _CLNNoise):
    """One component of DLESyM as a standalone prognostic model.

    The fused :class:`DLESyM` advances its atmosphere and ocean inside one
    ``step``. Each component is an ordinary prognostic model on the native
    HEALPix grid with its own input, forcing and output slots, so the exchange
    between them runs through :class:`~earth2studio.coupling.Connector` edges
    under :class:`~earth2studio.coupling.CoupledRunner`. A component holds only
    its own network core, constants and normalization, and samples its
    conditional layer norm noise from its own generator, seeded through
    ``set_rng``.

    Parameters
    ----------
    core : torch.nn.Module
        Network core of the component
    hpx_lat : np.ndarray
        HEALPix latitude coordinates, shape (12, nside, nside)
    hpx_lon : np.ndarray
        HEALPix longitude coordinates, shape (12, nside, nside)
    nside : int
        HEALPix nside
    center : np.ndarray
        Means of the output variables (prognostics then diagnostics), shape
        (num_output_variables,)
    scale : np.ndarray
        Standard deviations of the output variables, same shape as ``center``
    coupling_center : np.ndarray
        Means of the coupling variables received as forcing, shape
        (num_coupling_variables,)
    coupling_scale : np.ndarray
        Standard deviations of the coupling variables, same shape as
        ``coupling_center``
    constants : np.ndarray
        Constant fields, shape (12, num_constants, nside, nside)
    input_times : np.ndarray
        Input lead times, shape (num_input_times,)
    output_times : np.ndarray
        Output lead times, shape (num_output_times,)
    variables : list[str]
        Prognostic variables
    coupling_variables : list[str]
        Variables of the other component received as forcing
    diagnostic_variables : list[str], optional
        Diagnostic output variables, which are not fed back as input, by
        default []
    use_cln : bool, optional
        Whether the core uses conditional layer norm, by default False
    condition_shape : int, optional
        Dimension of the conditional layer norm noise vector. Required if
        ``use_cln`` is True, by default None
    """

    def __init__(
        self,
        core: torch.nn.Module,
        hpx_lat: np.ndarray,
        hpx_lon: np.ndarray,
        nside: int,
        center: np.ndarray,
        scale: np.ndarray,
        coupling_center: np.ndarray,
        coupling_scale: np.ndarray,
        constants: np.ndarray,
        input_times: np.ndarray,
        output_times: np.ndarray,
        variables: list[str],
        coupling_variables: list[str],
        diagnostic_variables: list[str] | None = None,
        use_cln: bool = False,
        condition_shape: int | None = None,
    ) -> None:
        super().__init__()
        self.core = core.eval()
        if use_cln and condition_shape is None:
            raise ValueError("'condition_shape' is required when 'use_cln' is True")
        self.use_cln = use_cln
        self.condition_shape = condition_shape
        self._cln_generator: torch.Generator | None = None

        self.hpx_lat = hpx_lat
        self.hpx_lon = hpx_lon
        self.nside = nside
        self.variables = list(variables)
        self.coupling_variables = list(coupling_variables)
        self.diagnostic_variables = list(diagnostic_variables or [])
        for name, times in (
            ("input_times", input_times),
            ("output_times", output_times),
        ):
            if not np.issubdtype(times.dtype, np.timedelta64):
                raise ValueError(
                    f"Input and output times must be of type 'timedelta64', got "
                    f"{times.dtype} for {name}"
                )
        self.input_times = input_times
        self.output_times = output_times
        self.sol_times = np.concatenate([input_times, output_times])

        outputs = self.variables + self.diagnostic_variables
        for name, value, labels in (
            ("center", center, outputs),
            ("scale", scale, outputs),
            ("coupling_center", coupling_center, self.coupling_variables),
            ("coupling_scale", coupling_scale, self.coupling_variables),
        ):
            value = np.asarray(value).reshape(-1)
            if len(value) != len(labels):
                raise ValueError(
                    f"'{name}' has {len(value)} entries for the {len(labels)} "
                    f"variables {labels}"
                )
            self.register_buffer(name, torch.from_numpy(value).to(torch.float32))
        self.register_buffer(
            "constants", torch.from_numpy(constants).to(dtype=torch.float32)
        )
        # Output channel of the log-transformed precipitation diagnostic.
        self._tp06_out_idx = outputs.index("tp06") if "tp06" in outputs else None

    @property
    def stochastic(self) -> bool:  # type: ignore[override]
        """Whether conditional layer norm noise is sampled during a rollout."""
        return bool(self.use_cln)

    def default_sources(self) -> tuple[DataSource | None, ...]:
        """Recommend ARCO ERA5, prepared as native inputs, for the history slot.

        Returns
        -------
        tuple[DataSource | None, ...]
            :class:`DLESyMInputSource` over ARCO ERA5 for the input slot and no
            recommendation for the forcing slot, which a connector feeds.
        """
        return (DLESyMInputSource(nside=self.nside), None)

    @classmethod
    def load_default_package(cls) -> Package:
        """Default DLESyM model package, shared with the fused :class:`DLESyM`."""
        return DLESyM.load_default_package()

    # -- declarations ----------------------------------------------------------

    def _signature(
        self, lead_times: np.ndarray, variables: list[str]
    ) -> CoordinateSystem:
        """A slot signature on the native HEALPix grid."""
        return coord_array(
            ("batch", "time", "lead_time", "variable", "face", "height", "width"),
            {"lead_time": lead_times, "variable": np.array(variables)},
            dynamic=("batch", "time"),
            grid=_healpix_grid(self.nside),
        )

    def _native_input_coords(self) -> CoordinateSystem:
        """Input signature in the variables the core is trained on."""
        return self._signature(self.input_times, self.variables)

    def _initial_state(self, x: xr.DataArray) -> xr.DataArray:
        """Convert a public initial history to the core's variables."""
        return x

    def _advance(
        self, input_coords: CoordinateSystem, signature: CoordinateSystem
    ) -> CoordinateSystem:
        """Validate ``input_coords`` against ``signature`` and shift to the outputs."""
        handshake_time(input_coords, allow_dynamic=True)
        handshake_time(input_coords, "lead_time")
        lead = input_coords.lead_time.values
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), signature
        )
        return coord_array_like(
            input_coords,
            {
                "lead_time": lead[-1] + self.output_times,
                "variable": np.array(self.variables + self.diagnostic_variables),
            },
        )

    # -- tensors ---------------------------------------------------------------

    def _flatten(
        self, x: xr.DataArray, with_anchors: bool = True
    ) -> tuple[torch.Tensor, np.ndarray, tuple[int, ...]]:
        """Flatten the leading dimensions of a validated field.

        Returns the tensor as ``(N, lead_time, variable, face, height, width)`` on
        the model's device, the anchor time of each of the ``N`` rows for
        insolation (empty unless ``with_anchors``), and the leading shape to
        restore.
        """
        tensor, _ = x.e2s.to_torch()
        leading_dims = x.dims[:-5]
        leading = tuple(x.sizes[d] for d in leading_dims)
        tensor = tensor.to(self.center.device).reshape(-1, *tensor.shape[-5:])
        if not with_anchors:
            return tensor, np.empty(0, dtype="datetime64[ns]"), leading
        times = x.coords["time"].values
        if "time" in leading_dims:
            shape = tuple(x.sizes[d] if d == "time" else 1 for d in leading_dims)
            anchors = np.broadcast_to(times.reshape(shape), leading)
        else:
            anchors = np.full(leading, times)
        return tensor, np.asarray(anchors).reshape(-1), leading

    def _select(
        self,
        forcing: xr.DataArray,
        variables: list[str],
        lead_times: np.ndarray,
        anchor: np.timedelta64,
    ) -> xr.DataArray:
        """Select the forcing variables and frames a call needs, validated.

        The frames are checked against ``forcing_coords()`` with their lead times
        taken relative to ``anchor``, the state's final lead time, so grid,
        dimension order and metadata mismatches are reported here rather than
        inside the network core.
        """
        leads = np.asarray(lead_times).astype("timedelta64[ns]")
        absent = [d for d in ("lead_time", "variable") if d not in forcing.dims]
        if absent:
            raise ValueError(
                f"Forcing must have dimensions {absent}; got {tuple(forcing.dims)}"
            )
        try:
            frame = forcing.sel(variable=list(variables), lead_time=leads)
        except KeyError as error:
            raise ValueError(
                f"Forcing lacks variables {variables} at lead times {leads}: {error}"
            ) from error
        signature = self.forcing_coords()
        handshake_dataarray(
            frame.assign_coords(lead_time=frame.lead_time.values - anchor),
            cast(CoordinateSystem, signature),
        )
        return frame

    @staticmethod
    def _align(forcing: xr.DataArray, like: xr.DataArray) -> xr.DataArray:
        """Lay ``forcing`` out on ``like``'s leading dimensions."""
        extra = [d for d in forcing.dims if d not in like.dims]
        if extra:
            raise ValueError(f"Forcing has dimensions {extra} that the state lacks")
        if "time" in forcing.coords and "time" in like.coords:
            theirs = np.atleast_1d(forcing.coords["time"].values)
            ours = np.atleast_1d(like.coords["time"].values)
            if theirs.shape != ours.shape or not np.array_equal(theirs, ours):
                raise ValueError(
                    f"Forcing is valid for times {theirs}, the state for {ours}"
                )
        missing = {d: like.sizes[d] for d in like.dims[:-5] if d not in forcing.dims}
        if missing:
            # expand_dims broadcasts to a read-only view; the core needs writable
            # memory to wrap.
            forcing = forcing.expand_dims(missing).copy()
        forcing = forcing.transpose(*[d for d in like.dims if d in forcing.dims])
        leading = tuple(like.sizes[d] for d in like.dims[:-5])
        if forcing.shape[: len(leading)] != leading:
            raise ValueError(
                f"Forcing leading shape {forcing.shape[: len(leading)]} does not "
                f"match the state's {leading}"
            )
        return forcing

    @staticmethod
    def _normalize(
        tensor: torch.Tensor, center: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Normalize the variable axis (third) of a ``(N, L, V, ...)`` tensor."""
        center, scale = (c.view(1, 1, -1, 1, 1, 1).to(tensor) for c in (center, scale))
        return (tensor - center) / scale

    def _run(
        self,
        x: xr.DataArray,
        state: torch.Tensor,
        anchors: np.ndarray,
        coupling: torch.Tensor,
    ) -> torch.Tensor:
        """Advance the core from a flattened state and its prepared coupling.

        Normalizes ``state``, calls the core with insolation, constants and
        ``coupling`` (and conditional layer norm noise if configured), and
        returns the denormalized ``(N, lead_time, variable, face, height, width)``
        output.
        """
        n = len(self.variables)
        state = self._normalize(state, self.center[:n], self.scale[:n])
        insolation = _insolation_tensor(
            anchors, self.sol_times + x.lead_time.values[-1], self.hpx_lat, self.hpx_lon
        )
        inputs = [
            t.to(state.device, state.dtype)
            for t in (
                state.permute(0, 3, 1, 2, 4, 5),
                insolation,
                self.constants,
                coupling,
            )
        ]
        self._ensure_rng()
        if self.use_cln:
            conditions = [
                torch.randn(
                    inputs[0].shape[0],
                    self.condition_shape,
                    generator=self._cln_generator,
                ).to(device=state.device, dtype=state.dtype)
                for _ in range(self.core.integration_steps)
            ]
            out = self.core(inputs, conditions_cln=conditions)
        else:
            out = self.core(inputs)
        out = out.permute(0, 2, 3, 1, 4, 5)
        out = out * self.scale.view(1, 1, -1, 1, 1, 1) + self.center.view(
            1, 1, -1, 1, 1, 1
        )
        if self._tp06_out_idx is not None:
            _tp06_to_meters(out, 2, self._tp06_out_idx)
        return out

    # -- execution -------------------------------------------------------------

    def _forward_array(self, x: xr.DataArray, forcing: xr.DataArray) -> xr.DataArray:
        """Advance one validated cycle; each component prepares its coupling."""
        raise NotImplementedError

    def _forward(
        self, x: xr.DataArray, forcing: xr.DataArray
    ) -> tuple[xr.DataArray, torch.Tensor | None]:
        """Advance one cycle and return it with the generator's position."""
        handshake_nonempty(x)
        return self._forward_array(x, forcing), self._rng_state()


@check_optional_dependencies()
class DLESyMAtmos(_DLESyMComponent):
    """Atmosphere component of DLESyM, as a standalone prognostic model.

    Takes the atmosphere's history on the native HEALPix grid and the sea surface
    temperature at the history's final lead time as forcing, and returns the
    atmosphere at every output lead time of one coupled cycle. The fused
    :class:`DLESyM` drives each cycle with the SST at the cycle's start, which is
    the ocean's previous output; under
    :class:`~earth2studio.coupling.CoupledRunner`, connect
    :class:`DLESyMOcean`'s SST into forcing slot 0 with ``time="hold"``.

    The model loads only the atmosphere core from the package shared with
    :class:`DLESyM`.

    Parameters
    ----------
    core : torch.nn.Module
        Atmosphere network core
    hpx_lat : np.ndarray
        HEALPix latitude coordinates, shape (12, nside, nside)
    hpx_lon : np.ndarray
        HEALPix longitude coordinates, shape (12, nside, nside)
    nside : int
        HEALPix nside
    center : np.ndarray
        Means of the output variables (prognostics then diagnostics), shape
        (num_output_variables,)
    scale : np.ndarray
        Standard deviations of the output variables, same shape as ``center``
    coupling_center : np.ndarray
        Means of the ocean variables received as forcing, shape
        (num_coupling_variables,)
    coupling_scale : np.ndarray
        Standard deviations of the ocean variables received as forcing, same
        shape as ``coupling_center``
    constants : np.ndarray
        Constant fields, shape (12, num_constants, nside, nside)
    input_times : np.ndarray
        Input lead times, shape (num_input_times,)
    output_times : np.ndarray
        Output lead times, shape (num_output_times,)
    variables : list[str]
        Prognostic variables
    coupling_variables : list[str]
        Ocean variables received as forcing
    diagnostic_variables : list[str], optional
        Diagnostic output variables, which are not fed back as input, by
        default []
    use_cln : bool, optional
        Whether the core uses conditional layer norm, by default False
    condition_shape : int, optional
        Dimension of the conditional layer norm noise vector. Required if
        ``use_cln`` is True, by default None

    Note
    ----
    See :class:`DLESyM` for more information about the model.

    Example
    -------
    ```python
    package = DLESyMAtmos.load_default_package()
    atmos = DLESyMAtmos.load_model(package)
    ocean = DLESyMOcean.load_model(package)
    runner = CoupledRunner(
        {"atmos": atmos, "ocean": ocean},
        [Connector("ocean", "atmos", time="hold"), Connector("atmos", "ocean")],
    )
    ```

    Badges
    ------
    region:global class:subseasonal-seasonal product:wind product:temp product:atmos year:2025
    gpu:40gb
    provider:nvidia backend:pytorch
    """

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        atmos_model_idx: int = 0,
        version: Literal["v1.0", "v1.1"] = "v1.1",
    ) -> PrognosticModel:
        """Load the atmosphere component from package

        Parameters
        ----------
        package : Package
            Package to load model from, the one :class:`DLESyM` loads
        atmos_model_idx : int, optional
            Index of atmos model weights to load, see :meth:`DLESyM.load_model`,
            by default 0
        version : {"v1.0", "v1.1"}, optional
            Checkpoint version to load, see :meth:`DLESyM.load_model`, by
            default "v1.1"

        Returns
        -------
        PrognosticModel
            Atmosphere prognostic model
        """
        return cls(
            **_load_package_config(package, version).atmos_kwargs(atmos_model_idx)
        )

    @property
    def coupling_times(self) -> np.ndarray:
        """Lead time of the SST frame, once per internal step of the core."""
        steps = 1 + max(self.core.output_time_dim // self.core.input_time_dim, 1)
        return np.array([self.input_times[-1]] * steps)

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the atmosphere.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray input signature.
        """
        return self._native_input_coords()

    def forcing_coords(self) -> CoordinateSystem:  # type: ignore[override]
        """Sea surface temperature at the history's final lead time.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray forcing signature.
        """
        return self._signature(self.input_times[-1:], self.coupling_variables)

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the atmosphere.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate signature or DataArray to validate and transform.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray output signature.
        """
        return self._advance(input_coords, self.input_coords())

    @torch.inference_mode()
    def _forward_array(self, x: xr.DataArray, sst: xr.DataArray) -> xr.DataArray:
        signature = self._advance(x, self._native_input_coords())
        lead = x.lead_time.values
        frame = self._align(
            self._select(sst, self.coupling_variables, lead[-1:], lead[-1]), x
        )
        state, anchors, leading = self._flatten(x)
        coupling, _, _ = self._flatten(frame, with_anchors=False)
        # The SST forcing is normalized with the ocean's constants.
        coupling = self._normalize(coupling, self.coupling_center, self.coupling_scale)
        coupling = coupling[:, 0]
        coupling = coupling.unsqueeze(0).expand(
            len(self.coupling_times), *coupling.shape
        )
        out = self._run(x, state, anchors, coupling)
        return _wrap_output(out, leading, signature, x)

    def __call__(self, x: xr.DataArray, sst: xr.DataArray) -> xr.DataArray:
        """Advance one coupled cycle without hooks.

        Parameters
        ----------
        x : xr.DataArray
            Atmosphere history matching ``input_coords()``.
        sst : xr.DataArray
            Sea surface temperature matching ``forcing_coords()``.

        Returns
        -------
        xr.DataArray
            Atmosphere at the cycle's output lead times.
        """
        return self.initialize(x, sst)[0]

    def initialize(
        self, x: xr.DataArray, sst: xr.DataArray
    ) -> tuple[xr.DataArray, torch.Tensor | None]:
        """Compute the first cycle and the continuation random stream.

        Parameters
        ----------
        x : xr.DataArray
            Atmosphere history matching ``input_coords()``.
        sst : xr.DataArray
            Sea surface temperature matching ``forcing_coords()``.

        Returns
        -------
        tuple[xr.DataArray, torch.Tensor | None]
            First cycle and conditional layer norm RNG state, or ``None`` for
            deterministic execution.
        """
        handshake_nonempty(x)
        self.output_coords(x)
        return self._forward(self._initial_state(x), sst)

    def step(
        self,
        y: xr.DataArray,
        sst: xr.DataArray,
        state: torch.Tensor | None,
    ) -> tuple[xr.DataArray, torch.Tensor | None]:
        """Advance one cycle from the previous output.

        Parameters
        ----------
        y : xr.DataArray
            Previous cycle, including caller-applied edits.
        sst : xr.DataArray
            Sea surface temperature holding the frame at ``y``'s final lead time.
        state : torch.Tensor | None
            RNG state returned by ``initialize`` or ``step``.

        Returns
        -------
        tuple[xr.DataArray, torch.Tensor | None]
            Next cycle and updated RNG state.
        """
        x = y.isel(lead_time=slice(-len(self.input_times), None)).sel(
            variable=self.variables
        )
        return self._with_rng(lambda: self._forward(x, sst), state)

    def create_iterator(  # type: ignore[override]
        self, x: xr.DataArray, sst: xr.DataArray
    ) -> Iterator[xr.DataArray]:
        """Yield cycles with hooks, taking each step's SST through ``send``.

        Parameters
        ----------
        x : xr.DataArray
            Atmosphere history matching ``input_coords()``.
        sst : xr.DataArray
            Sea surface temperature matching ``forcing_coords()``.

        Yields
        ------
        xr.DataArray
            Cycles, starting with initialization's prediction.
        """
        return self._default_create_iterator(x, sst)  # type: ignore[return-value]


@check_optional_dependencies()
class DLESyMOcean(_DLESyMComponent):
    """Ocean component of DLESyM, as a standalone prognostic model.

    Takes the ocean's history on the native HEALPix grid and, as forcing, the
    means of the atmosphere's coupling variables over the atmosphere frames of
    each ocean window, declared as qualified labels such as
    ``z1000:mean:48h`` valid at the ocean's own output lead times. The fused
    :class:`DLESyM` averages the atmosphere's output frames of the same cycle, so
    under :class:`~earth2studio.coupling.CoupledRunner` an ``exact`` connector
    from :class:`DLESyMAtmos` reduces its frames onto this slot.

    The model loads only the ocean core from the package shared with
    :class:`DLESyM`, so running the ocean alone does not load the atmosphere.

    Parameters
    ----------
    core : torch.nn.Module
        Ocean network core
    hpx_lat : np.ndarray
        HEALPix latitude coordinates, shape (12, nside, nside)
    hpx_lon : np.ndarray
        HEALPix longitude coordinates, shape (12, nside, nside)
    nside : int
        HEALPix nside
    center : np.ndarray
        Means of the output variables (prognostics then diagnostics), shape
        (num_output_variables,)
    scale : np.ndarray
        Standard deviations of the output variables, same shape as ``center``
    coupling_center : np.ndarray
        Means of the atmosphere variables received as forcing: the atmosphere's
        own output constants, shape (num_coupling_variables,)
    coupling_scale : np.ndarray
        Standard deviations of the atmosphere variables received as forcing,
        same shape as ``coupling_center``
    constants : np.ndarray
        Constant fields, shape (12, num_constants, nside, nside)
    input_times : np.ndarray
        Input lead times, shape (num_input_times,)
    output_times : np.ndarray
        Output lead times, shape (num_output_times,)
    atmos_output_times : np.ndarray
        Output lead times of the atmosphere, whose frames each ocean window
        averages, shape (num_atmos_output_times,)
    variables : list[str]
        Prognostic variables
    coupling_variables : list[str]
        Atmosphere variables whose window means are received as forcing
    diagnostic_variables : list[str], optional
        Diagnostic output variables, which are not fed back as input, by
        default []
    use_cln : bool, optional
        Whether the core uses conditional layer norm, by default False
    condition_shape : int, optional
        Dimension of the conditional layer norm noise vector. Required if
        ``use_cln`` is True, by default None

    Note
    ----
    See :class:`DLESyM` for more information about the model and
    :class:`DLESyMAtmos` for a coupled example.

    Badges
    ------
    region:global class:subseasonal-seasonal product:ocean year:2025
    gpu:40gb
    provider:nvidia backend:pytorch
    """

    def __init__(
        self,
        core: torch.nn.Module,
        hpx_lat: np.ndarray,
        hpx_lon: np.ndarray,
        nside: int,
        center: np.ndarray,
        scale: np.ndarray,
        coupling_center: np.ndarray,
        coupling_scale: np.ndarray,
        constants: np.ndarray,
        input_times: np.ndarray,
        output_times: np.ndarray,
        atmos_output_times: np.ndarray,
        variables: list[str],
        coupling_variables: list[str],
        diagnostic_variables: list[str] | None = None,
        use_cln: bool = False,
        condition_shape: int | None = None,
    ) -> None:
        super().__init__(
            core,
            hpx_lat=hpx_lat,
            hpx_lon=hpx_lon,
            nside=nside,
            center=center,
            scale=scale,
            coupling_center=coupling_center,
            coupling_scale=coupling_scale,
            constants=constants,
            input_times=input_times,
            output_times=output_times,
            variables=variables,
            coupling_variables=coupling_variables,
            diagnostic_variables=diagnostic_variables,
            use_cln=use_cln,
            condition_shape=condition_shape,
        )
        atmos_times, ocean_times = atmos_output_times, output_times
        n_atmos, n_ocean = len(atmos_times), len(ocean_times)
        if n_atmos < 2 or n_atmos % n_ocean:
            raise ValueError(
                f"{n_atmos} atmosphere output times cannot be averaged into "
                f"{n_ocean} equal ocean windows"
            )
        per_window = n_atmos // n_ocean
        spacing = atmos_times[1] - atmos_times[0]
        if not np.all(np.diff(atmos_times) == spacing):
            raise ValueError(
                "Atmosphere output times must be evenly spaced to declare the "
                f"ocean windows as means, got {atmos_times}"
            )
        # Window i averages the atmosphere frames of chunk i, as the fused model
        # does; declare it relative to ocean output time i as a right-closed
        # window (start, end], which holds the frames after start up to end.
        starts = atmos_times[::per_window] - spacing - ocean_times
        ends = atmos_times[per_window - 1 :: per_window] - ocean_times
        if not (np.all(starts == starts[0]) and np.all(ends == ends[0])):
            raise ValueError(
                "Ocean output times must sit at one offset from their atmosphere "
                f"windows to declare them as means, got {ocean_times} against "
                f"{atmos_times}"
            )
        self.window = per_window * spacing
        self.window_offsets = (starts[0], ends[0])
        if starts[0] == -self.window and ends[0] == np.timedelta64(0, "h"):
            # A window ending at the output time is written by its length alone.
            modifier = f"mean:{format_duration(self.window)}"
        else:
            modifier = (
                f"mean:{format_duration(starts[0], signed=True)}:"
                f"{format_duration(ends[0], signed=True)}"
            )
        self.forcing_variables = [f"{v}:{modifier}" for v in self.coupling_variables]

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        ocean_model_idx: int = 0,
        version: Literal["v1.0", "v1.1"] = "v1.1",
    ) -> PrognosticModel:
        """Load the ocean component from package

        Besides the ocean's own constants, this loads the normalization of the
        atmosphere variables the ocean is forced with: the forcing is means of
        atmosphere outputs, which the ocean normalizes with the atmosphere's
        output constants.

        Parameters
        ----------
        package : Package
            Package to load model from, the one :class:`DLESyM` loads
        ocean_model_idx : int, optional
            Index of ocean model weights to load, see :meth:`DLESyM.load_model`,
            by default 0
        version : {"v1.0", "v1.1"}, optional
            Checkpoint version to load, see :meth:`DLESyM.load_model`, by
            default "v1.1"

        Returns
        -------
        PrognosticModel
            Ocean prognostic model
        """
        return cls(
            **_load_package_config(package, version).ocean_kwargs(ocean_model_idx)
        )

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the ocean.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray input signature.
        """
        return self._native_input_coords()

    def forcing_coords(self) -> CoordinateSystem:  # type: ignore[override]
        """Window means of the atmosphere's coupling variables.

        One frame per ocean output time, valid at that time, averaging the
        atmosphere frames of the same cycle that lead up to it.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray forcing signature.
        """
        return self._signature(self.output_times, self.forcing_variables)

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the ocean.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate signature or DataArray to validate and transform.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray output signature.
        """
        return self._advance(input_coords, self.input_coords())

    @torch.inference_mode()
    def _forward_array(self, x: xr.DataArray, forcing: xr.DataArray) -> xr.DataArray:
        signature = self._advance(x, self._native_input_coords())
        lead = x.lead_time.values
        windows = self._align(
            self._select(
                forcing, self.forcing_variables, lead[-1] + self.output_times, lead[-1]
            ),
            x,
        )
        state, anchors, leading = self._flatten(x)
        coupling, _, _ = self._flatten(windows, with_anchors=False)
        # The window means are atmosphere outputs, normalized with the
        # atmosphere's output constants.
        coupling = self._normalize(coupling, self.coupling_center, self.coupling_scale)
        # Window-major variables, as the fused model stacks its chunk means.
        coupling = coupling.reshape(
            coupling.shape[0], -1, *coupling.shape[3:]
        ).unsqueeze(0)
        out = self._run(x, state, anchors, coupling)
        return _wrap_output(out, leading, signature, x)

    def __call__(self, x: xr.DataArray, forcing: xr.DataArray) -> xr.DataArray:
        """Advance one coupled cycle without hooks.

        Parameters
        ----------
        x : xr.DataArray
            Ocean history matching ``input_coords()``.
        forcing : xr.DataArray
            Atmosphere window means matching ``forcing_coords()``.

        Returns
        -------
        xr.DataArray
            Ocean at the cycle's output lead times.
        """
        return self.initialize(x, forcing)[0]

    def initialize(
        self, x: xr.DataArray, forcing: xr.DataArray
    ) -> tuple[xr.DataArray, torch.Tensor | None]:
        """Compute the first cycle and the continuation random stream.

        Parameters
        ----------
        x : xr.DataArray
            Ocean history matching ``input_coords()``.
        forcing : xr.DataArray
            Atmosphere window means matching ``forcing_coords()``.

        Returns
        -------
        tuple[xr.DataArray, torch.Tensor | None]
            First cycle and conditional layer norm RNG state, or ``None`` for
            deterministic execution.
        """
        handshake_nonempty(x)
        self.output_coords(x)
        return self._forward(self._initial_state(x), forcing)

    def step(
        self,
        y: xr.DataArray,
        forcing: xr.DataArray,
        state: torch.Tensor | None,
    ) -> tuple[xr.DataArray, torch.Tensor | None]:
        """Advance one cycle from the previous output.

        Parameters
        ----------
        y : xr.DataArray
            Previous cycle, including caller-applied edits.
        forcing : xr.DataArray
            Atmosphere window means holding the frames at ``y``'s final lead time
            plus each ocean output lead time.
        state : torch.Tensor | None
            RNG state returned by ``initialize`` or ``step``.

        Returns
        -------
        tuple[xr.DataArray, torch.Tensor | None]
            Next cycle and updated RNG state.
        """
        x = y.isel(lead_time=slice(-len(self.input_times), None)).sel(
            variable=self.variables
        )
        return self._with_rng(lambda: self._forward(x, forcing), state)

    def create_iterator(  # type: ignore[override]
        self, x: xr.DataArray, forcing: xr.DataArray
    ) -> Iterator[xr.DataArray]:
        """Yield cycles with hooks, taking each step's forcing through ``send``.

        Parameters
        ----------
        x : xr.DataArray
            Ocean history matching ``input_coords()``.
        forcing : xr.DataArray
            Atmosphere window means matching ``forcing_coords()``.

        Yields
        ------
        xr.DataArray
            Cycles, starting with initialization's prediction.
        """
        return self._default_create_iterator(x, forcing)  # type: ignore[return-value]
