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

import json
from collections import OrderedDict
from collections.abc import Callable, Generator, Mapping
from copy import copy, deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar, cast

import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from loguru import logger

from earth2studio.data import HRRR
from earth2studio.data.base import DataSource, ForecastSource
from earth2studio.grids import (
    BilinearRegridder,
    CurvilinearGrid,
    GridDefinition,
    NearestRegridder,
    Region,
)
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import (
    coord_array,
    coord_array_like,
    handshake_coords,
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
from earth2studio.utils.type import CoordinateSystem, CoordSystem

if TYPE_CHECKING:
    from earth2studio.models.nn.stormscope_flash import RegionInfo, _FlashStep

try:
    from physicsnemo import Module
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.samplers import Solver, sample
    from physicsnemo.utils.zenith_angle import cos_zenith_angle

    from earth2studio.models._diffusion import SOLVERS, seeded_rng
    from earth2studio.models._diffusion import solver_options as _solver_options
except ImportError:
    OptionalDependencyFailure("stormscope")
    Module = None  # type: ignore[assignment]
    Solver = object
    cos_zenith_angle = None  # type: ignore[assignment]

# Default regridders, shared by coupled models and by the masks computed at load.
# Tables for the observation grids take seconds to build and hold one entry per
# model gridpoint, so only the most recent few are kept.
_REGRIDDERS: OrderedDict[tuple[str, str, str, str], Any] = OrderedDict()
_REGRIDDER_CACHE_SIZE = 8

_BuiltinRegridder = TypeVar("_BuiltinRegridder", NearestRegridder, BilinearRegridder)


def _default_regridder(
    cls: type[_BuiltinRegridder],
    source_grid: str,
    target_grid: GridDefinition,
    **options: Any,
) -> _BuiltinRegridder:
    """Build, or reuse, the regridder of a default source onto a model grid.

    The defaults are the built-in classes the models were trained with, rather
    than registry resolutions, so their ``coverage`` defines the valid masks.
    """
    key = (
        cls.__name__,
        source_grid,
        target_grid.fingerprint(),
        repr(sorted(options.items())),
    )
    if key in _REGRIDDERS:
        _REGRIDDERS.move_to_end(key)
        return _REGRIDDERS[key]
    regridder = cls(source_grid, target_grid, **options)
    _REGRIDDERS[key] = regridder
    if len(_REGRIDDERS) > _REGRIDDER_CACHE_SIZE:
        _REGRIDDERS.popitem(last=False)
    return regridder


def crop_to_region(
    y: xr.DataArray, info: "RegionInfo", mask: bool = True
) -> xr.DataArray:
    """Crop a field on a regional view's grid to its requested region.

    Parameters
    ----------
    y : xr.DataArray
        Field on the view's grid, with ``y`` and ``x`` trailing.
    info : RegionInfo
        Region resolved by the view.
    mask : bool, optional
        Set gridpoints outside the requested region to NaN, by default True.

    Returns
    -------
    xr.DataArray
        Field on the smallest rectangle of gridpoints containing the region.
    """
    if y.dims[-2:] != ("y", "x") or y.shape[-2:] != info.input_shape:
        raise ValueError("Forecast spatial dimensions do not match this region")
    ys, xs = info.output_slices
    out = y.isel(y=ys, x=xs)
    if mask:
        data = out.data
        try:
            import cupy

            xp = cupy.get_array_module(data)
        except ImportError:
            xp = np
        requested = xp.asarray(info.requested_mask)
        out = out.copy(data=xp.where(requested, data, xp.nan))
    if "shape" in out.attrs:
        out.attrs["shape"] = list(out.shape[-2:])
    return out


@check_optional_dependencies()
class StormScopeBase(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """StormScope diffusion prognostic base model with staged denoising.

    Variants should subclass to define dataset/resolution specifics (e.g., grids,
    variables, and data loading).

    This base class supports an "ensemble of experts" approach where the diffusion
    sampler is invoked in multiple stages, each using a different set of model
    weights applicable to a specific sigma range. Stages are defined by `model_spec`.
    Single-stage (standard diffusion sampling with one denoiser) is also supported.

    Parameters
    ----------
    model_spec : list[dict[str, Any]]
        Sequence of stage specifications. Each entry must contain:

        - 'model': physicsnemo.Module
        - 'sigma_min': float
        - 'sigma_max': float

        The sigma interval determines which expert is applied during denoising.
    means : torch.Tensor
        Per-variable mean used to normalize inputs. Expected shape [1, C, 1, 1].
    stds : torch.Tensor
        Per-variable std used to normalize inputs. Expected shape [1, C, 1, 1].
    variables : np.ndarray
        Names of prognostic variables corresponding to the high-resolution state.
    latitudes : torch.Tensor
        Latitudes of the grid, expected shape [H, W].
    longitudes : torch.Tensor
        Longitudes of the grid, expected shape [H, W].
    conditioning_means : torch.Tensor | None, optional
        Means to normalize any external conditioning data. Default is None.
    conditioning_stds : torch.Tensor | None, optional
        Stds to normalize any external conditioning data. Default is None.
    conditioning_variables : np.ndarray | None, optional
        Names of external conditioning variables. Default is None.
    sampler_args : dict[str, Any] | None, optional
        Default sampler arguments passed to the diffusion sampler.
        Default is {"num_steps": 100, "S_churn": 10}.
    input_times : np.ndarray, optional
        Input timesteps, of type timedelta64. Default is [0 m] (i.e., the current time).
    output_times : np.ndarray, optional
        Output timesteps, of type timedelta64. Default is [60 m] (i.e., 1 hour from the current time).
    y_coords : np.ndarray | None, optional
        Y coordinates of the grid, expected shape [H, W]. Default is None, in which
        case the model uses the enumerated indices inferred from the latitude and
        longitude grid shapes.
    x_coords : np.ndarray | None, optional
        X coordinates of the grid, expected shape [H, W]. Default is None, in which
        case the model uses the enumerated indices inferred from the latitude and
        longitude grid shapes.
    glm_mask : torch.Tensor | None, optional
        Boolean mask of shape [C] over the state channels, True where a channel is
        normalized with log1p/expm1 (GLM — Geostationary Lightning Mapper — style)
        rather than mean/std. Default is None, in which case all channels use mean/std.
    conditioning_glm_mask : torch.Tensor | None, optional
        Boolean mask of shape [C_cond] over the conditioning channels, True where a
        channel is log1p-normalized. Default is None (all conditioning channels use
        mean/std).
    max_dist_km : float, optional
        Distance in kilometers beyond which the default nearest-neighbor
        regridders leave a model gridpoint without data. ``load_model`` uses six
        kilometers times the variant's spatial downsampling. Default is 6.0.
    amp : bool, optional
        Enable automatic mixed precision (autocast) for the diffusion sampler's
        network forward passes. The sampler's latent/state math is kept in
        ``_SAMPLER_DTYPE`` (fp64); only the DiT forward passes run under
        autocast. Can also be toggled after construction via the ``amp``
        attribute. Default is True.
    compile : bool, optional
        Compile each staged denoising expert with ``torch.compile`` (using the
        ``"reduce-overhead"`` mode) for faster repeated sampling. Can also be
        invoked after construction via :meth:`compile_experts`. Default is False.

    Note
    ----
    Inputs and conditioning are declared on the model grid. Each subclass's
    ``default_sources`` regrids its observations onto that grid, and
    ``load_model`` sets ``valid_mask`` and ``conditioning_valid_mask`` to the
    gridpoints those regridders cover. Uncovered gridpoints are filled before
    sampling and are NaN in the forecast. Building those regridders takes several
    seconds at load; ``default_sources`` and coupled models on the same grid
    reuse them.
    """

    # Registered grids of the default input and conditioning sources, whose
    # coverage of the model grid defines the valid masks.
    _INPUT_GRID: str | None = None
    _CONDITIONING_GRID: str | None = None
    # Nearest-neighbor distance for the conditioning source, if not max_dist_km.
    _CONDITIONING_MAX_DIST_KM: float | None = None

    # Constants used to normalize lat/lon input features
    _CENTRAL_LAT_CONSTANT = 38.5
    _LAT_SCALE = 15
    _CENTRAL_LON_CONSTANT = 262.5
    _LON_SCALE = 36

    # Whether to place the state first in the conditioning input
    _STATE_FIRST = True

    # Dtype to use in the EDM sampler
    _SAMPLER_DTYPE = torch.float64

    # Dtype expected by the diffusion network (the PhysicsNeMo preconditioner
    # does not down-cast its model input, so the sampler feeds it this dtype)
    _MODEL_DTYPE = torch.float32

    # Constant to fill invalid gridpoints in the input after normalization
    _INPUT_INVALID_FILL_CONSTANT = 0.0

    def __init__(
        self,
        model_spec: list[dict[str, Any]],
        means: torch.Tensor,
        stds: torch.Tensor,
        variables: np.ndarray,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
        conditioning_means: torch.Tensor | None = None,
        conditioning_stds: torch.Tensor | None = None,
        conditioning_variables: np.ndarray | None = None,
        glm_mask: torch.Tensor | None = None,
        conditioning_glm_mask: torch.Tensor | None = None,
        topo: torch.Tensor | None = None,
        nexrad_proximity: torch.Tensor | None = None,
        sampler_args: dict[str, Any] | None = {"num_steps": 100, "S_churn": 10},
        input_times: np.ndarray = np.array([np.timedelta64(0, "h")]),
        output_times: np.ndarray = np.array([np.timedelta64(1, "h")]),
        y_coords: np.ndarray | None = None,
        x_coords: np.ndarray | None = None,
        max_dist_km: float = 6.0,
        amp: bool = True,
        compile: bool = False,
        *,
        solver: str = "edm_stochastic_heun",
        solver_options: dict[str, Any] | None = None,
    ):
        super().__init__()
        if solver not in SOLVERS:
            raise ValueError(
                f"Unknown solver {solver!r}; choose from {tuple(SOLVERS)}."
            )
        self.solver = solver
        if solver_options is not None and "num_steps" in solver_options:
            raise ValueError(
                "Do not pass 'num_steps' in solver_options; use sampler_args['num_steps'] instead."
            )
        self.solver_options = dict(solver_options or {})
        # Validate and store staged models
        if not isinstance(model_spec, list) or len(model_spec) == 0:
            raise ValueError("model_spec must be a non-empty list of stage dicts.")

        self.input_times = input_times
        self.output_times = output_times
        self.sliding_window = len(input_times) > len(output_times)

        self.register_buffer("latitudes", latitudes)
        self.register_buffer("longitudes", longitudes)
        self._lat_cpu_copy = self.latitudes.cpu().numpy()
        self._lon_cpu_copy = self.longitudes.cpu().numpy()
        self.register_buffer("valid_mask", torch.ones_like(latitudes, dtype=torch.bool))
        self.register_buffer(
            "conditioning_valid_mask", torch.ones_like(latitudes, dtype=torch.bool)
        )
        self.max_dist_km = max_dist_km

        if y_coords is not None and x_coords is not None:
            self.y = y_coords
            self.x = x_coords
        else:
            self.y = np.arange(latitudes.shape[0])
            self.x = np.arange(longitudes.shape[1])

        for i, spec in enumerate(model_spec):
            if not isinstance(spec, dict):
                raise TypeError(f"model_spec[{i}] must be a dict.")
            for key in ("model", "sigma_min", "sigma_max"):
                if key not in spec:
                    raise KeyError(f"model_spec[{i}] missing required key '{key}'.")

        # Sort stages by descending sigma_max to ensure large->small sampling schedule
        self.model_spec = sorted(
            model_spec, key=lambda s: float(s["sigma_max"]), reverse=True
        )
        # Store in ModuleList so `.to(device)` works
        self.stage_models = nn.ModuleList([spec["model"] for spec in self.model_spec])
        self.start_sigma = self.model_spec[-1]["sigma_min"]
        self.end_sigma = self.model_spec[0]["sigma_max"]

        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.variables = variables

        # Per-channel mask marking log1p/expm1 (GLM-style) channels. Defaults to
        # all-False (every channel uses mean/std), so callers that do not use GLM
        # need not supply it.
        if glm_mask is None:
            glm_mask = torch.zeros(means.shape[1], dtype=torch.bool)
        self.register_buffer("glm_mask", glm_mask)

        if conditioning_means is not None:
            self.register_buffer("conditioning_means", conditioning_means)
        if conditioning_stds is not None:
            self.register_buffer("conditioning_stds", conditioning_stds)
        if conditioning_glm_mask is None and conditioning_variables is not None:
            conditioning_glm_mask = torch.zeros(
                len(conditioning_variables), dtype=torch.bool
            )
        if conditioning_glm_mask is not None:
            self.register_buffer("conditioning_glm_mask", conditioning_glm_mask)

        # Static invariant channels (appended to the conditioning input). Present
        # only for variants whose registry entry sets the corresponding flag.
        if topo is not None:
            self.register_buffer("topo", topo)
        if nexrad_proximity is not None:
            self.register_buffer("nexrad_proximity", nexrad_proximity)

        self.conditioning_variables = conditioning_variables
        self.sampler_args = sampler_args or {}
        self.variables = variables

        # Mixed-precision toggle for the diffusion sampler. When True the DiT
        # forward passes run under torch.autocast; the sampler's latent/state
        # math stays in _SAMPLER_DTYPE (fp64). Mutable so it can be toggled
        # after construction.
        self.amp = amp
        self._flash_plan: list[_FlashStep] | None = None
        self._flash_plan_sources: tuple[torch.Tensor, ...] = ()

        # Regional views (see with_region): context gridpoints kept around a Flash
        # region, the full-footprint geometry shared with views, and the region
        # this model covers, which is None for the full footprint.
        self.region_padding = 25
        self._footprint: dict[str, Any] | None = None
        self.region_info: RegionInfo | None = None

        # Optionally torch.compile each staged expert ("reduce-overhead" mode).
        self._experts_compiled = False
        if compile:
            self.compile_experts()

    # Top-level key in registry.json that holds this model's variants. Set by
    # subclasses (e.g. "goes", "mrms") so a single shared registry can describe
    # every StormScope model without name collisions.
    _REGISTRY_KEY: str | None = None
    _is_flash: bool = False

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the default StormScope package from Hugging Face."""
        return Package(
            "hf://nvidia/stormscope-goes-mrms@dcbe14a506e849b1afec9962dbf774f07d7aece8",
            cache_options={
                "cache_storage": Package.default_cache("stormscope"),
                "same_names": False,  # Distinguish matching filenames across directories.
            },
        )

    @staticmethod
    def _load_registry(package: Package) -> dict[str, Any]:
        """Load and parse the package ``registry.json``."""
        with open(package.resolve("registry.json")) as f:
            return json.load(f)

    @classmethod
    def _resolve_model_entry(
        cls, package: Package, model_name: str
    ) -> tuple[str, dict[str, Any]]:
        """Resolve a (possibly aliased) ``model_name`` to its registry entry.

        The package ``registry.json`` is structured by source::

            {
              "normalization": {<group>: {"file_prefix": ..., "order": [...]}, ...},
              "<source>": {
                "models": {"<canonical name>": {<variant spec>}, ...},
                "aliases": {"<alias>": "<canonical name>", ...}
              }
            }

        where ``<source>`` is the subclass's ``_REGISTRY_KEY``. Aliases (including
        legacy training names) are resolved to their canonical entry.

        Parameters
        ----------
        package : Package
            Package to load the registry from.
        model_name : str
            Canonical variant name or an alias.

        Returns
        -------
        tuple[str, dict[str, Any]]
            The resolved canonical name and its variant specification.
        """
        if cls._REGISTRY_KEY is None:
            raise NotImplementedError(
                "StormScope subclasses must set _REGISTRY_KEY to select a "
                "section of registry.json."
            )

        registry = cls._load_registry(package)

        if cls._REGISTRY_KEY not in registry:
            raise KeyError(
                f"registry.json has no '{cls._REGISTRY_KEY}' section for {cls.__name__}."
            )
        section = registry[cls._REGISTRY_KEY]
        models = section["models"]
        aliases = section.get("aliases", {})

        resolved = aliases.get(model_name, model_name)

        if resolved not in models:
            available = ", ".join(sorted(models))
            raise KeyError(
                f"Unknown StormScope '{cls._REGISTRY_KEY}' model '{model_name}'. "
                f"Available variants: {available}. "
                f"Use {cls.__name__}.list_available_models() to inspect them."
            )

        entry = models[resolved]
        if entry.get("deprecated", False):
            logger.warning(
                f"StormScope '{resolved}' is a legacy (nearcasting) checkpoint kept "
                "for backwards compatibility; the supported defaults are the CONUS "
                "nowcasting variants. See list_available_models() for alternatives."
            )
        return resolved, entry

    @classmethod
    def list_available_models(
        cls, package: Package | None = None
    ) -> dict[str, dict[str, Any]]:
        """List the model variants available in a package for this model.

        Parameters
        ----------
        package : Package | None, optional
            Package to inspect. If None, the default package is loaded.

        Returns
        -------
        dict[str, dict[str, Any]]
            Mapping of canonical variant name to a metadata dict with
            ``description`` and ``deprecated`` keys.
        """
        if package is None:
            package = cls.load_default_package()
        if cls._REGISTRY_KEY is None:
            raise NotImplementedError(
                "StormScope subclasses must set _REGISTRY_KEY to select a "
                "section of registry.json."
            )
        registry = cls._load_registry(package)
        if cls._REGISTRY_KEY not in registry:
            raise KeyError(
                f"registry.json has no '{cls._REGISTRY_KEY}' section for {cls.__name__}."
            )
        section = registry[cls._REGISTRY_KEY]
        return {
            name: {
                "description": spec.get("description", ""),
                "deprecated": spec.get("deprecated", False),
            }
            for name, spec in section["models"].items()
        }

    @classmethod
    def _build_normalization(
        cls, package: Package, registry: dict[str, Any], names: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build per-channel normalization stats for a list of variable ``names``.

        Channels are selected **by name** from the package's canonical-order
        normalization arrays (the top-level ``normalization`` block of
        ``registry.json``), not by position — a variant's ``variables`` /
        ``conditioning_vars`` may be any subset, in any order, and may span
        multiple normalization groups (e.g. MRMS ``refc``/``refc_base`` plus a GLM
        channel).

        GLM channels (a group with ``transform: "log1p"`` and no ``file_prefix``)
        carry no mean/std; they are flagged in the returned mask and handled by the
        ``log1p`` / ``expm1`` transform in the (de)normalization path. Their
        placeholder mean/std are 0/1 so the affine path is a no-op if ever applied.

        Parameters
        ----------
        package : Package
            Package to resolve the ``*_means.npy`` / ``*_stds.npy`` arrays from.
        registry : dict[str, Any]
            Parsed ``registry.json`` (must contain the ``normalization`` block).
        names : np.ndarray
            Variable names in the desired channel order.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            ``means`` and ``stds`` of shape ``[1, C, 1, 1]`` and a boolean
            ``glm_mask`` of shape ``[C]`` (True where the channel is log1p GLM).
        """
        if "normalization" not in registry:
            raise KeyError(
                "registry.json is missing the top-level 'normalization' block "
                "required to select normalization channels by name."
            )
        norm = registry["normalization"]

        # Map every known channel name to its (group, index) in canonical order.
        name_to_loc: dict[str, tuple[str, int]] = {}
        for gkey, group in norm.items():
            for idx, nm in enumerate(group["order"]):
                name_to_loc[nm] = (gkey, idx)

        n = len(names)
        means = np.zeros(n, dtype=np.float32)
        stds = np.ones(n, dtype=np.float32)
        glm_mask = np.zeros(n, dtype=bool)

        array_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for j, name in enumerate(names):
            if name not in name_to_loc:
                raise KeyError(
                    f"Variable '{name}' is not present in any registry "
                    f"'normalization' group; cannot determine its normalization."
                )
            gkey, idx = name_to_loc[name]
            group = norm[gkey]
            # GLM-style groups use a parameter-free transform (log1p/expm1) and
            # have no stats array; leave placeholder mean/std and flag the channel.
            if group.get("file_prefix") is None or group.get("transform") == "log1p":
                glm_mask[j] = True
                continue
            if gkey not in array_cache:
                prefix = group["file_prefix"]
                m = np.atleast_1d(np.load(package.resolve(f"{prefix}_means.npy")))
                s = np.atleast_1d(np.load(package.resolve(f"{prefix}_stds.npy")))
                array_cache[gkey] = (m, s)
            m, s = array_cache[gkey]
            means[j] = m[idx]
            stds[j] = s[idx]

        means_t = torch.from_numpy(means)[None, :, None, None]
        stds_t = torch.from_numpy(stds)[None, :, None, None]
        glm_mask_t = torch.from_numpy(glm_mask)
        return means_t, stds_t, glm_mask_t

    @staticmethod
    def _crop_invariant(
        arr: torch.Tensor, image_size: list[int], spatial_downsample: int
    ) -> torch.Tensor:
        """Center-crop a full-HRRR-grid 2D array to ``image_size`` then stride by
        ``spatial_downsample`` — the same transform applied to ``lat``/``lon`` and
        the static ``topo`` / ``nexrad_proximity`` invariants."""
        full_y, full_x = arr.shape[0], arr.shape[1]
        anchor_y = int((full_y - image_size[0]) / 2)
        anchor_x = int((full_x - image_size[1]) / 2)
        arr = arr[
            anchor_y : anchor_y + image_size[0], anchor_x : anchor_x + image_size[1]
        ]
        return arr[::spatial_downsample, ::spatial_downsample]

    @classmethod
    def _load_invariant(
        cls, package: Package, filename: str, pkg: dict[str, Any]
    ) -> torch.Tensor:
        """Load a static 2D invariant (e.g. ``topo.npy``) from the package and
        crop/stride it onto the variant's model grid."""
        arr = torch.from_numpy(np.load(package.resolve(filename))).to(
            dtype=torch.float32
        )
        return cls._crop_invariant(arr, pkg["image_size"], pkg["spatial_downsample"])

    @staticmethod
    def _build_grid_and_times(
        package: Package, pkg: dict[str, Any]
    ) -> tuple[
        torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int
    ]:
        """Build the model grid and input/output timesteps from a registry entry.

        Crops (and optionally downsamples) a subregion of the HRRR grid according
        to the variant's ``image_size`` and ``spatial_downsample``, and derives the
        input/output lead times from its ``step_interval`` / sliding-window config.

        Parameters
        ----------
        package : Package
            Package to resolve grid files (``lat.npy``, ``lon.npy``) from.
        pkg : dict[str, Any]
            Resolved registry entry for the variant.

        Returns
        -------
        tuple
            ``(latitudes, longitudes, y, x, input_times, output_times, spatial_downsample)``.
        """
        # Grid coordinates: crop a subregion from the HRRR grid
        image_size = pkg["image_size"]
        spatial_downsample = pkg["spatial_downsample"]
        latitudes = torch.from_numpy(np.load(package.resolve("lat.npy")))
        longitudes = (
            torch.from_numpy(np.load(package.resolve("lon.npy"))) + 360.0
        ) % 360.0
        hrrr_y, hrrr_x = HRRR.HRRR_Y, HRRR.HRRR_X
        full_y, full_x = latitudes.shape[0], longitudes.shape[1]
        anchor_y = int((full_y - image_size[0]) / 2)
        anchor_x = int((full_x - image_size[1]) / 2)
        latitudes = StormScopeBase._crop_invariant(
            latitudes, image_size, spatial_downsample
        )
        longitudes = StormScopeBase._crop_invariant(
            longitudes, image_size, spatial_downsample
        )
        y = hrrr_y[anchor_y : anchor_y + image_size[0]][::spatial_downsample]
        x = hrrr_x[anchor_x : anchor_x + image_size[1]][::spatial_downsample]

        # Input/output timesteps configuration
        if pkg["sliding_window"]:
            # N input timesteps, 1 output timestep, with resolution step_interval
            n_steps, step_interval = pkg["n_steps"], pkg["step_interval"]
            input_times = np.arange(-n_steps + 1, 1) * np.timedelta64(
                step_interval, "m"
            )
            output_times = np.array([np.timedelta64(step_interval, "m")])
        else:
            # 1 input, 1 output, with resolution step_interval
            input_times = np.array([np.timedelta64(0, "m")])
            output_times = np.array([np.timedelta64(pkg["step_interval"], "m")])

        return (
            latitudes,
            longitudes,
            y,
            x,
            input_times,
            output_times,
            spatial_downsample,
        )

    @classmethod
    def _load_checkpoints(
        cls, package: Package, pkg: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Load the staged-denoising experts for a variant from its registry entry.

        Baseline files contain ``EDMPreconditioner(ConcatConditionWrapper(DiT))``;
        Flash files contain ``FlashModel`` with packed interval heads. Both use
        PhysicsNeMo checkpoint loading; Flash additionally validates its metadata.
        """
        if pkg.get("flash", False):
            from earth2studio.models.nn.stormscope_flash import load_flash_experts

            return load_flash_experts(package, pkg, cast(str, cls._REGISTRY_KEY))
        model_spec = []
        for m in pkg["checkpoints"]:
            model = Module.from_checkpoint(package.resolve(m["path"]))
            model_spec.append(
                {
                    "model": model,
                    "sigma_min": float(m["sigma_min"]),
                    "sigma_max": float(m["sigma_max"]),
                }
            )
        return model_spec

    @classmethod
    def load_model(
        cls,
        package: Package,
        *args: Any,
        **kwargs: Any,
    ) -> PrognosticModel:
        """Load model from package. Must be implemented by subclasses."""
        raise NotImplementedError(
            "StormScopeBase.load_model must be implemented by a subclass."
        )

    def _grid(self) -> CurvilinearGrid:
        """Grid of every input, conditioning and output slot."""
        return CurvilinearGrid(self._lat_cpu_copy, self._lon_cpu_copy, self.y, self.x)

    def _set_default_coverage(self) -> None:
        """Set the valid masks to the model gridpoints the default sources cover."""
        grid = self._grid()
        for name, source_grid, max_dist_km in (
            ("valid_mask", self._INPUT_GRID, self.max_dist_km),
            (
                "conditioning_valid_mask",
                self._CONDITIONING_GRID,
                self._conditioning_max_dist_km(),
            ),
        ):
            if source_grid is None:
                continue
            regridder = _default_regridder(
                NearestRegridder, source_grid, grid, max_dist_km=max_dist_km
            )
            coverage = torch.from_numpy(np.array(regridder.coverage))
            self.register_buffer(name, coverage.to(self.latitudes.device))

    def _conditioning_max_dist_km(self) -> float:
        """Nearest-neighbor distance of the default conditioning regridder."""
        if self._CONDITIONING_MAX_DIST_KM is None:
            return self.max_dist_km
        return self._CONDITIONING_MAX_DIST_KM

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the prognostic model.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray input signature on the model's
            curvilinear grid.
        """
        return coord_array(
            ("batch", "time", "lead_time", "variable", "y", "x"),
            {"lead_time": self.input_times, "variable": np.array(self.variables)},
            dynamic=("batch", "time"),
            grid=self._grid(),
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the prognostic model.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate signature or DataArray to validate and transform,
            with absolute forecast lead times.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray output signature for the configured
            output lead-time window.
        """
        handshake_time(input_coords, allow_dynamic=True)
        handshake_time(input_coords, "lead_time")
        lead = input_coords.lead_time.values
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
        return coord_array_like(
            input_coords, {"lead_time": self.output_times + lead[-1]}
        )

    def normalize_conditioning(
        self, conditioning: torch.Tensor | None
    ) -> torch.Tensor | None:
        """Normalize external conditioning with stored stats if available.

        Channels flagged in ``conditioning_glm_mask`` are normalized with
        ``log1p`` instead of mean/std; all others use the affine ``(x-mean)/std``.
        """
        if conditioning is None:
            return None
        affine = conditioning
        if "conditioning_means" in self._buffers:
            affine = affine - self.conditioning_means
        if "conditioning_stds" in self._buffers:
            affine = affine / self.conditioning_stds

        mask = self._buffers.get("conditioning_glm_mask", None)
        if mask is None or not bool(mask.any()):
            if self._is_flash and self._REGISTRY_KEY == "mrms":
                return torch.where(torch.isnan(affine), 0.0, affine)
            return affine
        glm_view = mask.view(1, -1, 1, 1)
        return torch.where(glm_view, torch.log1p(conditioning), affine)

    def _stack_lead_times(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Stack lead times along the channel dimension. Reshapes tensors from (..., n_lt, n_vars, y, x)
        to (..., 1, n_lt * n_vars, y, x) and updates the coordinate system to reflect the new shape.
        The last lead time from the original coordinate system is used as the lead time for the stacked tensor.
        """
        lt_dim = list(coords.keys()).index("lead_time")
        handshake_dim(coords, "variable", lt_dim + 1)
        n_lt = len(coords["lead_time"])
        n_vars = len(coords["variable"])
        stacked = x.reshape(
            *x.shape[:lt_dim], n_lt * n_vars, *x.shape[lt_dim + 2 :]
        ).unsqueeze(lt_dim)
        stacked_coords = coords.copy()
        stacked_coords["lead_time"] = stacked_coords["lead_time"][-1:]
        stacked_coords["variable"] = np.array(
            [
                f"{var}(t+{str(lt)})"
                for lt in stacked_coords["lead_time"]
                for var in stacked_coords["variable"]
            ]
        )
        return stacked, stacked_coords

    def build_condition(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        conditioning: torch.Tensor | None = None,
        conditioning_coords: CoordSystem | None = None,
    ) -> torch.Tensor:
        """Construct the full conditioning input to the diffusion model.
        Subclasses can override to customize the channel layout.

        Parameters
        ----------
        x_norm : torch.Tensor
            Normalized input high-resolution state.
        coords : CoordSystem
            Input coordinate system to provide time and lead time for cos zenith angle calculation
        conditioning : torch.Tensor | None, optional
            Optional normalized external conditioning, by default None.
        conditioning_coords : CoordSystem | None, optional
            Optional coordinate system for the conditioning, by default None.

        Returns
        -------
        torch.Tensor
            Conditioning tensor to pass to the diffusion sampler.
        """

        # Split the state into its main (mean/std) channels and any GLM (log1p)
        # channels, so each becomes its own contiguous, separately-stacked obs
        # block (mrms_obs | glm_obs) rather than interleaving GLM into the state.
        if bool(self.glm_mask.any()):
            main_sel = (~self.glm_mask).nonzero(as_tuple=True)[0]
            glm_sel = self.glm_mask.nonzero(as_tuple=True)[0]
            var = np.asarray(coords["variable"])
            x_main = x[:, :, :, main_sel, :, :]
            main_coords = coords.copy()
            main_coords["variable"] = var[main_sel.cpu().numpy()]
            x_glm = x[:, :, :, glm_sel, :, :]
            glm_coords = coords.copy()
            glm_coords["variable"] = var[glm_sel.cpu().numpy()]
        else:
            x_main, main_coords = x, coords
            x_glm, glm_coords = None, None

        if self.sliding_window:
            # Reshape each block to (..., 1, n_lt * n_vars, y, x)
            x_main, main_coords = self._stack_lead_times(x_main, main_coords)
            if x_glm is not None and glm_coords is not None:
                x_glm, glm_coords = self._stack_lead_times(x_glm, glm_coords)
            if conditioning is not None:
                if conditioning_coords is None:
                    raise ValueError(
                        "Expected conditioning_coords when conditioning is provided"
                    )
                conditioning, conditioning_coords = self._stack_lead_times(
                    conditioning, conditioning_coords
                )

        # Fold batch/time/lead_time dimensions
        b, t, lt, _, _, _ = x_main.shape
        handshake_size(main_coords, "lead_time", 1)
        coords = main_coords  # used below for cos-zenith times
        x_main = x_main.reshape(b * t * lt, *x_main.shape[3:])
        if x_glm is not None:
            x_glm = x_glm.reshape(b * t * lt, *x_glm.shape[3:])
        if conditioning is not None:
            conditioning = conditioning.reshape(b * t * lt, *conditioning.shape[3:])

        # Assemble the obs blocks in canonical order:
        #   [ goes_obs(conditioning) | mrms_obs(x_main) | glm_obs(x_glm) ]
        # _STATE_FIRST places the state ahead of the external conditioning (GOES
        # model); for the MRMS model conditioning (GOES) leads.
        parts = []
        if conditioning is not None and not self._STATE_FIRST:
            parts.append(conditioning)
        parts.append(x_main)
        if x_glm is not None:
            parts.append(x_glm)
        if conditioning is not None and self._STATE_FIRST:
            parts.append(conditioning)

        if self.latitudes is not None and self.longitudes is not None:
            normed_lat = (self.latitudes - self._CENTRAL_LAT_CONSTANT) / self._LAT_SCALE
            normed_lon = (
                self.longitudes + 180.0
            ) % 360.0 - 180.0  # trained models expect [-180, 180] longitudes
            normed_lon = (normed_lon - self._CENTRAL_LON_CONSTANT) / self._LON_SCALE
            latlon_input = torch.cat(
                (normed_lat[None, None, :, :], normed_lon[None, None, :, :]), dim=1
            )
            parts.append(latlon_input.repeat(b * t, 1, 1, 1))

        # Build cos zenith features from input and target times
        times = np.array(coords["time"]).astype(np.datetime64)
        lead_times = np.array(coords["lead_time"]).astype(np.timedelta64)
        input_cz_times = np.concatenate(
            [times + lead_times[-1]] * b, axis=0
        )  # batch and time dims are folded together
        target_cz_times = input_cz_times + self.output_times[-1]
        input_cz_times = np.array(
            [
                datetime.fromtimestamp(
                    t.astype("datetime64[s]").astype(int), tz=timezone.utc
                )
                for t in input_cz_times
            ]
        )
        target_cz_times = np.array(
            [
                datetime.fromtimestamp(
                    t.astype("datetime64[s]").astype(int), tz=timezone.utc
                )
                for t in target_cz_times
            ]
        )
        cz_0 = np.stack(
            [
                cos_zenith_angle(t, self._lon_cpu_copy, self._lat_cpu_copy)
                for t in input_cz_times
            ],
            axis=0,
        )  # shape [B*T, H, W]
        cz_1 = np.stack(
            [
                cos_zenith_angle(t, self._lon_cpu_copy, self._lat_cpu_copy)
                for t in target_cz_times
            ],
            axis=0,
        )  # shape [B*T, H, W]

        parts.extend(
            [
                torch.from_numpy(cz_0).to(device=x.device, dtype=x.dtype)[
                    :, None, :, :
                ],
                torch.from_numpy(cz_1).to(device=x.device, dtype=x.dtype)[
                    :, None, :, :
                ],
            ]
        )

        # Static invariants, in the documented trailing order: nexrad_proximity
        # then topo. Each is [H, W]; broadcast to [B*T, 1, H, W] like lat/lon.
        for invariant in (
            self._buffers.get("nexrad_proximity", None),
            self._buffers.get("topo", None),
        ):
            if invariant is not None:
                parts.append(  # noqa: PERF401
                    invariant.to(device=x.device, dtype=x.dtype)[None, None].repeat(
                        b * t, 1, 1, 1
                    )
                )

        return torch.cat(parts, dim=1)

    def _randn_like(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn(
            x.shape,
            dtype=x.dtype,
            device=x.device if self._noise_generator is None else "cpu",
            generator=self._noise_generator,
        ).to(x.device)

    stochastic = True
    _noise_generator: torch.Generator | None = None

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Seed initial latents and per-step EDM churn noise.

        Parameters
        ----------
        seed : int
            Seed for diffusion noise.
        reset : bool, optional
            Reset existing noise state, by default True.
        """
        if reset or self._noise_generator is None:
            self._noise_generator = torch.Generator().manual_seed(seed)

    @torch.inference_mode()
    def _forward_tensor(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        conditioning: torch.Tensor | None = None,
        conditioning_coords: CoordSystem | None = None,
    ) -> torch.Tensor:
        """Run a single prognostic step using staged diffusion denoising.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor for a single step with shape [B, T, L, C, H, W].
        coords : CoordSystem
            Coordinates describing `x`.
        conditioning : torch.Tensor | None, optional
            External conditioning aligned to `x` if used, by default None.
        conditioning_coords : CoordSystem | None, optional
            Coordinate system for the conditioning, by default None.

        Returns
        -------
        torch.Tensor
            Output tensor for the same step with denoised prognostic variables.
        """

        handshake_dim(coords, ("batch", "time", "lead_time", "variable", "y", "x"))
        if conditioning_coords is not None:
            for index, dim in enumerate(("batch", "time", "lead_time", "variable")):
                handshake_dim(conditioning_coords, dim, index)

        if self._is_flash:
            from earth2studio.models.nn.stormscope_flash import bind_flash_domain

            bind_flash_domain(self)
        b, t, lt, _, _, _ = x.shape

        # Scale input and fill invalid gridpoints. GLM-style channels use log1p;
        # all others use the affine (x-mean)/std. Invalid mean/std channels are
        # filled with _INPUT_INVALID_FILL_CONSTANT (0.0 by default); GLM channels
        # use log1p(0)=0.
        x_norm = self._normalize_state(x)
        if self._is_flash and self._REGISTRY_KEY == "goes":
            from earth2studio.models.nn.stormscope_flash import prepare_flash_goes_state

            x_norm = prepare_flash_goes_state(self, x, x_norm)
        fill = torch.where(
            self.glm_mask.view(1, -1, 1, 1),
            torch.zeros((), dtype=x_norm.dtype, device=x_norm.device),
            torch.full(
                (),
                self._INPUT_INVALID_FILL_CONSTANT,
                dtype=x_norm.dtype,
                device=x_norm.device,
            ),
        )
        x_norm = torch.where(self.valid_mask, x_norm, fill)
        output_dtype = x_norm.dtype

        # Scale conditioning and zero-fill invalid gridpoints
        if conditioning is not None:
            conditioning_norm = self.normalize_conditioning(conditioning)
            conditioning_norm = torch.where(
                self.conditioning_valid_mask, conditioning_norm, 0.0
            )
        else:
            conditioning_norm = None

        # valid_mask/conditioning_valid_mask only zero-fill gridpoints they mark
        # invalid; a non-finite value at a gridpoint marked *valid* (e.g. a GOES
        # fill value that lands inside the static valid_mask) passes straight
        # through. Fail loudly rather than feeding it into the diffusion sampler.
        # Done here rather than on the assembled condition so the guard reduces
        # the smaller pre-assembly tensors, costs a single device sync for both,
        # and fails before build_condition does its work.
        finite = torch.isfinite(x_norm).all()
        if conditioning_norm is not None:
            finite = finite & torch.isfinite(conditioning_norm).all()
        if not bool(finite):
            self._raise_nonfinite_input(
                x_norm, coords, conditioning_norm, conditioning_coords
            )

        # Combined conditioning to diffusion model: state, conditioning variables, extras (lat/lon, cos zenith angle).
        # Cast to the network dtype: autocast leaves float64 inputs (e.g. GOES) alone.
        condition = self.build_condition(
            x=x_norm,
            coords=coords,
            conditioning=conditioning_norm,
            conditioning_coords=conditioning_coords,
        ).to(self._MODEL_DTYPE)

        latents = self._randn_like(x[:, :, -1].reshape(b * t, *x.shape[3:]))

        # Run diffusion sampler. When AMP is enabled, autocast accelerates the
        # DiT forward passes inside the sampler; the latent/state math stays in
        # _SAMPLER_DTYPE (fp64) since autocast only affects autocast-eligible
        # network ops, not the explicit fp64 pointwise updates. autocast is a
        # no-op when self.amp is False, so the default path is unchanged.
        generator = None
        if (
            self._noise_generator is not None
            and not self._is_flash
            and self.solver.startswith("edm_stochastic_")
        ):
            seed = int(
                torch.randint(2**63 - 1, (), generator=self._noise_generator).item()
            )
            generator = torch.Generator(device=x.device).manual_seed(seed)
        with (
            seeded_rng(generator),
            torch.autocast(device_type=x.device.type, enabled=self.amp),
        ):
            out = self._edm_sampler(
                latents=latents,
                condition=condition,
                sigma_min=self.start_sigma,
                sigma_max=self.end_sigma,
                **self.sampler_args,
            ).to(output_dtype)

        out = out.reshape(b, t, len(self.output_times), *out.shape[1:])

        out = torch.where(self.valid_mask, out, torch.nan)
        out = self._denormalize_state(out)
        return out

    def _raise_nonfinite_input(
        self,
        x_norm: torch.Tensor,
        coords: CoordSystem,
        conditioning_norm: torch.Tensor | None,
        conditioning_coords: CoordSystem | None,
    ) -> None:
        """Raise a ValueError naming the variables carrying non-finite values.

        Only called once the cheap guard in :meth:`_forward` has found a
        non-finite value, so the extra reductions here are off the happy path.
        """
        details = []
        for name, tensor, tensor_coords in (
            ("input state", x_norm, coords),
            ("conditioning", conditioning_norm, conditioning_coords),
        ):
            if tensor is None:
                continue
            bad = ~torch.isfinite(tensor)
            count = int(bad.sum())
            if count == 0:
                continue
            # Tensors are [B, T, L, C, H, W]; reduce everything but the channel
            channels = bad.any(dim=(0, 1, 2, 4, 5)).nonzero(as_tuple=True)[0]
            labels = channels.tolist()
            if tensor_coords is not None and "variable" in tensor_coords:
                variables = np.asarray(tensor_coords["variable"])
                if len(variables) == tensor.shape[3]:
                    labels = variables[channels.cpu().numpy()].tolist()
            details.append(f"{count} value(s) in {name} variable(s) {labels}")

        raise ValueError(
            f"Non-finite values found before diffusion sampling "
            f"({'; '.join(details)}) not sanitized by valid_mask/"
            "conditioning_valid_mask. This likely indicates missing or "
            "fill-valued data (e.g. from the GOES source) at gridpoints marked "
            "valid."
        )

    def _normalize_state(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize state channels: log1p for GLM channels, affine otherwise."""
        affine = (x - self.means) / self.stds
        if not bool(self.glm_mask.any()):
            return affine
        glm_view = self.glm_mask.view(1, -1, 1, 1)
        return torch.where(glm_view, torch.log1p(x), affine)

    def _denormalize_state(self, out: torch.Tensor) -> torch.Tensor:
        """Invert :meth:`_normalize_state`: expm1 (clamped >=0) for GLM channels,
        affine otherwise. NaNs (invalid gridpoints) propagate unchanged."""
        affine = out * self.stds + self.means
        if not bool(self.glm_mask.any()):
            return affine
        glm_view = self.glm_mask.view(1, -1, 1, 1)
        glm = torch.clamp(torch.expm1(out), min=0.0)
        return torch.where(glm_view, glm, affine)

    def _edm_sampler(
        self,
        latents: torch.Tensor,
        condition: torch.Tensor | None = None,
        num_steps: int = 18,
        sigma_max: float = 500,
        sigma_min: float = 0.004,
        rho: float = 7,
        S_churn: float = 0,
        S_min: float = 0,
        S_max: float = float("inf"),
        S_noise: float = 1,
        progress_bar: Any | None = None,
    ) -> torch.Tensor:
        if self._is_flash:
            from earth2studio.models.nn.stormscope_flash import sample_flash

            return sample_flash(self, latents.to(self._SAMPLER_DTYPE), condition)
        scheduler = EDMNoiseScheduler(sigma_min=sigma_min, sigma_max=sigma_max, rho=rho)
        options = _solver_options(
            scheduler,
            self.solver,
            num_steps=num_steps,
            defaults=dict(S_churn=S_churn, S_min=S_min, S_max=S_max, S_noise=S_noise),
            options=self.solver_options,
        )
        staged_solver = _StagedSolver(
            self, condition, scheduler, self.solver, options, progress_bar
        )
        steps = scheduler.timesteps(
            num_steps, device=latents.device, dtype=self._SAMPLER_DTYPE
        )
        return sample(
            staged_solver.denoiser,
            latents.to(self._SAMPLER_DTYPE) * steps[0],
            scheduler,
            num_steps,
            solver=staged_solver,
            time_steps=steps,
        )

    def _select_expert(self, t_cur: torch.Tensor) -> nn.Module:
        """Select the active denoising expert based on the current time step."""

        eps = 1e-7  # small epsilon to avoid floating point issues
        for i, stage in enumerate(self.model_spec):
            if t_cur >= stage["sigma_min"] - eps and t_cur < stage["sigma_max"] + eps:
                return self.stage_models[i]
        raise ValueError(
            f"No denoising expert found for time step {t_cur.cpu().item()}, {stage['sigma_min']}"
        )

    def _configure_variant(self, entry: Mapping[str, Any], compile: bool) -> None:
        self._is_flash = bool(entry.get("flash", False))
        if self._is_flash:
            self.sampler_args = {}
        if compile:
            self.compile_experts()

    # Gridded state that a view crops from the full footprint.
    _REGIONAL_BUFFERS = (
        "latitudes",
        "longitudes",
        "valid_mask",
        "conditioning_valid_mask",
        "topo",
        "nexrad_proximity",
        "mrms_coverage_mask",
    )

    def with_region(self, region: Region, /) -> Self:
        """Return a view of this model covering a geographic box.

        The view shares the loaded experts and crops the grid, invariants and
        valid masks of the full footprint, so neither this model nor other views
        change, and a view can select any region of the full footprint. Flash
        variants run on the region plus up to ``region_padding`` gridpoints of
        context per side (25 by default), expanded to at least 200 gridpoints per
        axis and aligned to four-gridpoint patches. Other variants run on the full
        footprint, which covers every supported region. Use :meth:`crop_output` to
        present forecasts on the requested box.

        Views share expert networks, so run them in sequence, not concurrently.
        Couple GOES and MRMS views made with the same region and
        ``region_padding``, so that their grids match.

        Parameters
        ----------
        region : Region
            Box (:meth:`~earth2studio.grids.Region.box`) inside the full footprint.

        Returns
        -------
        Self
            View whose coordinate signatures describe the selected grid.

        Raises
        ------
        ValueError
            If the region is not a box or extends outside the full footprint.
        """
        from earth2studio.models.nn.stormscope_flash import resolve_region

        if region.kind != "box":
            raise ValueError(
                f"{type(self).__name__} supports only box regions, not {region.kind!r}"
            )
        west, south, east, north = region.params
        footprint = self._full_footprint()
        latitude = footprint["lat"]
        info = resolve_region(
            latitude,
            footprint["lon"],
            {"lat": (south, north), "lon": (west, east)},
            self.region_padding if self._is_flash else 0,
        )
        if not self._is_flash:
            height, width = latitude.shape
            y0, y1, x0, x1 = info.output_bounds
            info = replace(
                info,
                input_bounds=(0, height, 0, width),
                padding=(y0, height - y1, x0, width - x1),
            )
        return self._view(info)

    def _full_footprint(self) -> dict[str, Any]:
        """Full-footprint geometry, captured from the loaded model and shared with views."""
        if self._footprint is None:
            self._footprint = {
                "lat": self._lat_cpu_copy,
                "lon": self._lon_cpu_copy,
                "y": self.y,
                "x": self.x,
                "buffers": {
                    name: self._buffers[name]
                    for name in self._REGIONAL_BUFFERS
                    if self._buffers.get(name) is not None
                },
            }
        return self._footprint

    def _view(self, info: "RegionInfo") -> Self:
        """Copy of this model on a crop of the full footprint, sharing the experts."""
        footprint = self._full_footprint()
        view = copy(self)
        view._buffers = self._buffers.copy()
        view._modules = self._modules.copy()
        view._parameters = self._parameters.copy()
        view._non_persistent_buffers_set = self._non_persistent_buffers_set.copy()
        view.stage_models = nn.ModuleList(self.stage_models)
        view.model_spec = [stage.copy() for stage in self.model_spec]
        view.sampler_args = self.sampler_args.copy()
        for name in (
            "input_times",
            "output_times",
            "variables",
            "conditioning_variables",
            "glm_variables",
        ):
            value = getattr(self, name, None)
            if value is not None:
                setattr(view, name, value.copy())
        view._flash_plan = None
        view._flash_plan_sources = ()
        # Each view seeds its own noise stream.
        view._noise_generator = None
        view.region_info = info

        y0, y1, x0, x1 = info.input_bounds
        for name, value in footprint["buffers"].items():
            view.register_buffer(
                name, value[y0:y1, x0:x1].clone().to(self.latitudes.device)
            )
        view.y = footprint["y"][y0:y1].copy()
        view.x = footprint["x"][x0:x1].copy()
        view._lat_cpu_copy = footprint["lat"][y0:y1, x0:x1].copy()
        view._lon_cpu_copy = footprint["lon"][y0:y1, x0:x1].copy()
        return view

    def crop_output(self, y: xr.DataArray, mask: bool = True) -> xr.DataArray:
        """Crop a forecast of a regional view to its requested box, for presentation.

        Keep stepping with the uncropped forecast, whose context feeds the rollout.
        Models on the full footprint return forecasts unchanged.

        Parameters
        ----------
        y : xr.DataArray
            Forecast on this model's grid, with ``y`` and ``x`` trailing.
        mask : bool, optional
            Set gridpoints outside the requested box to NaN, by default True.

        Returns
        -------
        xr.DataArray
            Forecast on the smallest rectangle of gridpoints containing the box.
        """
        if self.region_info is None:
            return y
        return crop_to_region(y, self.region_info, mask)

    def _apply(
        self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True
    ) -> Any:
        if self._is_flash:
            self._flash_plan = None
            self._flash_plan_sources = ()
        return super()._apply(fn, recurse=recurse)  # type: ignore[no-untyped-call]

    def compile_experts(self, mode: str = "reduce-overhead") -> None:
        """Compile each staged denoising expert in place with ``torch.compile``.

        Compilation is lazy (each expert compiles on its first forward pass), so
        this is safe to call before or after moving the model to a device.
        Repeated calls are a no-op.

        Parameters
        ----------
        mode : str, optional
            ``torch.compile`` mode, by default ``"reduce-overhead"`` (matching
            the StormScope reference inference scripts).
        """
        if self._experts_compiled:
            return
        if self._is_flash:
            self._flash_plan = None
            self._flash_plan_sources = ()
        for i in range(len(self.stage_models)):
            self.stage_models[i] = torch.compile(
                self.stage_models[i],
                mode=mode,
                dynamic=False if self._is_flash else None,
            )
        self._experts_compiled = True

    def prep_input(
        self, x: torch.Tensor, coords: CoordSystem, conditioning: bool = False
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Zero-fill gridpoints outside the valid mask of the input or conditioning.

        Data must be on the model grid. Filled points are filled again after
        normalization in :meth:`_forward_tensor`.
        """
        handshake_coords(coords, {"y": self.y, "x": self.x}, ["y", "x"])
        valid_mask = self.conditioning_valid_mask if conditioning else self.valid_mask
        return torch.where(~valid_mask, 0.0, x), coords.copy()

    def next_input(
        self,
        pred: xr.DataArray,
        x: xr.DataArray,
    ) -> xr.DataArray:
        """Gets the inputs for the next step of the prognostic model given a pair
        of inputs and predictions that were just run. If the model uses a sliding
        window, the oldest lead time in the input is removed and the latest
        prediction is added for the next step. If the model does not
        use a sliding window, the prediction is returned as is.

        Parameters
        ----------
        pred : xr.DataArray
            Prediction field.
        x : xr.DataArray
            Previous history field.
        """
        if self.sliding_window:
            self.output_coords(x)
            n_out = len(self.output_times)
            previous, _ = x.isel(lead_time=slice(n_out, None)).e2s.to_torch()
            prediction, _ = pred.e2s.to_torch()
            next_input = torch.cat(
                [previous.to(prediction.device), prediction],
                dim=x.get_axis_num("lead_time"),
            )
            next_input_coords = coord_array_like(
                pred,
                {"lead_time": self.input_times + np.asarray(pred["lead_time"])[-1]},
            )
        else:
            return pred.copy(deep=True)
        result = from_torch(next_input, next_input_coords)
        result.encoding = pred.encoding.copy()
        return result

    def forcing_coords(self) -> CoordinateSystem | None:
        """Describe the conditioning history on the checkpoint grid.

        Returns
        -------
        CoordinateSystem | None
            Allocation-free initial conditioning signature. Subsequent steps
            consume only the new frames at the current forecast's lead times.
            None for unforced GOES forecasts.
        """
        if self.conditioning_variables is None or not len(self.conditioning_variables):
            return None
        return coord_array_like(
            self.input_coords(), {"variable": self.conditioning_variables}
        )

    def _initialize(
        self, x: xr.DataArray, conditioning: xr.DataArray | None
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Compute the first forecast and retain missing history and sampling state.

        Parameters
        ----------
        x : xr.DataArray
            Initial history matching ``input_coords()``.
        conditioning : xr.DataArray
            Full conditioning history matching ``forcing_coords()`` on the
            checkpoint grid. The caller supplies all required channels.

        Returns
        -------
        tuple[xr.DataArray, dict[str, Any]]
            First forecast chunk and missing input/conditioning history and RNG
            state. Iterator hooks are not applied.
        """
        handshake_nonempty(x)
        if conditioning is not None:
            handshake_nonempty(conditioning)
        # Views share experts; follow them if another view moved them.
        experts = self._modules.get("stage_models")
        expert = None if experts is None else next(experts.parameters(), None)
        if expert is not None and expert.device != self.latitudes.device:
            self.to(expert.device)
        if self._noise_generator is None:
            self.set_rng(int(torch.randint(2**31, ())))
        y = self._predict(x, conditioning)
        count = len(self.output_times)
        return y, {
            "history": x.isel(lead_time=slice(count, None)).copy(deep=True),
            "conditioning": (
                conditioning.isel(lead_time=slice(count, None)).copy(deep=True)
                if conditioning is not None
                else None
            ),
            "rng": cast(torch.Generator, self._noise_generator).get_state().clone(),
        }

    def _forward(
        self, y: xr.DataArray, conditioning: xr.DataArray | None, state: dict[str, Any]
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Advance with newly supplied conditioning and explicit continuation state.

        Parameters
        ----------
        y : xr.DataArray
            Previous forecast chunk, including any caller-applied edits.
        conditioning : xr.DataArray
            New conditioning frames at the lead times of ``y``; earlier context
            is retained in ``state``.
        state : dict[str, Any]
            Missing history and RNG state returned by ``initialize`` or ``step``.

        Returns
        -------
        tuple[xr.DataArray, dict[str, Any]]
            Next forecast chunk and updated state, without iterator hooks.
        """
        if conditioning is not None:
            declaration = self.forcing_coords()
            if declaration is None:
                raise ValueError("This model does not accept forcing")
            handshake_dataarray(
                conditioning,
                coord_array_like(
                    declaration,
                    {
                        "lead_time": y.lead_time.values,
                    },
                ),
            )
            for dim in y.dims[:-3]:
                handshake_size(conditioning, dim, y.sizes[dim])
                if dim in y.coords:
                    handshake_coords(conditioning, y, dim)
        tensor, _ = y.e2s.to_torch()
        history, _ = state["history"].e2s.to_torch()
        signature = coord_array_like(
            y,
            {
                "lead_time": np.concatenate(
                    [state["history"].lead_time.values, y.lead_time.values]
                )
            },
        )
        x = from_torch(
            torch.cat(
                [history.to(tensor.device), tensor], dim=y.get_axis_num("lead_time")
            ),
            signature,
            name=y.name,
        )
        x.attrs, x.encoding = deepcopy(y.attrs), deepcopy(y.encoding)
        if conditioning is not None:
            current, _ = conditioning.e2s.to_torch()
            history, _ = state["conditioning"].e2s.to_torch()
            conditioning = from_torch(
                torch.cat(
                    [history.to(current.device), current],
                    dim=conditioning.get_axis_num("lead_time"),
                ),
                coord_array_like(x, {"variable": self.conditioning_variables}),
            )
        previous = self._noise_generator
        self._noise_generator = torch.Generator().set_state(state["rng"].clone())
        try:
            return self._initialize(x, conditioning)
        finally:
            self._noise_generator = previous

    @torch.inference_mode()
    def _predict(
        self,
        x: xr.DataArray,
        conditioning: xr.DataArray | None,
    ) -> xr.DataArray:
        """Calls the prognostic model with explicitly provided conditioning. Useful when
        combining multiple cross-conditioned models during rollout (does not require
        model to define a conditioning data source).

        Parameters
        ----------
        x : xr.DataArray
            Input field.
        conditioning : xr.DataArray
            Explicit conditioning field with matching leading and temporal axes.

        Returns
        -------
        xr.DataArray
            Forecast field.
        """

        self.output_coords(x)
        if conditioning is None:
            packed, restore = batch_func()._compress_array(self, x)
            tensor, coords = packed.e2s.to_torch()
            tensor, coords = self.prep_input(
                tensor.to(self.means.device).clone(), coords
            )
            out = from_torch(
                self._forward_tensor(tensor, coords),
                self.output_coords(packed),
                name=x.name,
            )
            out.attrs, out.encoding = deepcopy(out.attrs), deepcopy(x.encoding)
            return restore(out)
        handshake_nonempty(conditioning)
        handshake_time(conditioning)
        handshake_time(conditioning, "lead_time")
        handshake_dataarray(
            conditioning,
            coord_array_like(
                self.input_coords(),
                {
                    "lead_time": np.asarray(x.lead_time),
                    "variable": self.conditioning_variables,
                },
            ),
        )
        handshake_dim(conditioning, x.dims)
        for dim in x.dims[:-3]:
            handshake_size(conditioning, dim, x.sizes[dim])
            if dim in x.coords or dim in conditioning.coords:
                handshake_coords(conditioning, x, dim)
        packed, restore = batch_func()._compress_array(self, x)
        # Pack the conditioning's leading axes in the same order as the state members.
        if "batch" in conditioning.coords and "batch" not in conditioning.dims:
            # The conditioning kernel needs only dimension labels. The state
            # restore callback retains its public scalar batch auxiliary.
            conditioning = conditioning.drop_vars("batch")
        leading = conditioning.dims[:-5]
        if leading:
            temporary = "_conditioning_batch"
            while temporary in conditioning.dims or temporary in conditioning.coords:
                temporary += "_"
            condition = conditioning.e2s.batch(leading, batch_dim=temporary)
            condition = condition.rename({temporary: "batch"})
        else:
            condition = conditioning.expand_dims(batch=[0])
        output_coords = self.output_coords(packed)
        x, tensor_coords = packed.e2s.to_torch()
        conditioning, conditioning_coords = condition.e2s.to_torch()
        x = x.to(self.means.device).clone()
        conditioning = conditioning.to(self.means.device)
        x, x_coords = self.prep_input(x, tensor_coords)
        conditioning, conditioning_coords = self.prep_input(
            conditioning, conditioning_coords, conditioning=True
        )

        x = self._forward_tensor(
            x,
            x_coords,
            conditioning=conditioning,
            conditioning_coords=conditioning_coords,
        )
        out = from_torch(x, output_coords, name=packed.name)
        out.attrs = deepcopy(out.attrs)
        out.encoding = deepcopy(packed.encoding)
        return restore(out)


class StormScopeGOES(StormScopeBase):
    """StormScope model forecasting GOES data on the HRRR grid.

    This model supports multiple variants at different spatiotemporal resolutions,
    selected by passing ``model_name`` to ``load_model`` (default: ``"3km_10min"``).
    The primary focus is CONUS nowcasting at 3km resolution; coarser 6km
    nearcasting variants are retained as
    legacy checkpoints. Variant names are semantic (``<resolution>_<cadence>``):

      - ``3km_10min``  : 3km resolution, 10 minute timestep (CONUS nowcasting)
      - ``3km_10min_flash``: accelerated 3km / 10 minute CONUS nowcasting
      - ``6km_1hr``    : 6km resolution, 60 minute timestep (legacy nearcasting),
        conditioned on NWP fields; loads as :py:class:`StormScopeGOESConditioned`

    Use :py:meth:`list_available_models` to inspect the variants in a given package
    (including any added after this release). Legacy training-style names are still
    accepted as aliases.

    Flash uses five backbone calls per step. Select a region within the native
    CONUS domain with :meth:`~StormScopeBase.with_region`.

    Variants whose input cadence is finer than their output cadence use a sliding
    window of input timesteps and predict one output timestep; others use a single
    input timestep and predict one output timestep.

    Note
    ----
    For more information see the following references:

    - https://huggingface.co/nvidia/stormscope-goes-mrms
    - MRMS forecasting skill: evaluation procedure, metrics, and score table:
      https://huggingface.co/nvidia/stormscope-goes-mrms/blob/dcbe14a506e849b1afec9962dbf774f07d7aece8/mrms_forecast_skill.md

    Parameters
    ----------
    model_spec : list[dict[str, Any]]
        Sequence of stage specifications; see `StormScopeBase`.
    means : torch.Tensor
        Per-variable mean for normalization, shape [1, C, 1, 1].
    stds : torch.Tensor
        Per-variable std for normalization, shape [1, C, 1, 1].
    latitudes : torch.Tensor
        Latitudes of the grid, expected shape [H, W].
    longitudes : torch.Tensor
        Longitudes of the grid, expected shape [H, W].
    variables : np.ndarray, optional
        GOES input variables. Default is
        ["abi01c", "abi02c", "abi03c", "abi07c", "abi08c", "abi09c", "abi10c", "abi13c"].
    conditioning_variables : np.ndarray | None, optional
        Must be empty or None; conditioned variants use
        :py:class:`StormScopeGOESConditioned`. By default None.
    conditioning_means : torch.Tensor | None, optional
        Means to normalize any external conditioning data. Default is None.
    conditioning_stds : torch.Tensor | None, optional
        Stds to normalize any external conditioning data. Default is None.
    sampler_args : dict[str, Any] | None, optional
        Default sampler arguments passed to the diffusion sampler.
        Default is {"num_steps": 100, "S_churn": 10}.
    input_times : np.ndarray, optional
        Input timesteps, of type timedelta64. Default is [0 m] (i.e., the current time).
    output_times : np.ndarray, optional
        Output timesteps, of type timedelta64. Default is [60 m] (i.e., 1 hour from the current time).
    y_coords : np.ndarray | None, optional
        Y coordinates of the grid, expected shape [H, W]. Default is None, in which
        case the model uses the enumerated indices inferred from the latitude and
        longitude grid shapes.
    x_coords : np.ndarray | None, optional
        X coordinates of the grid, expected shape [H, W]. Default is None, in which
        case the model uses the enumerated indices inferred from the latitude and
        longitude grid shapes.
    max_dist_km : float, optional
        Distance in kilometers beyond which the default GOES regridder leaves a
        model gridpoint without data. Default is 6.0.

    Note
    ----
    To have a unified coordinate system over CONUS for convenience, the model uses the HRRR grid.
    As a result, there are portions of the domain which go beyond the extent of the GOES-East data,
    so these portions are masked as invalid (set to NaN).

    solver : str, optional
        Select the numerical method for integrating the sampling ODE in reverse
        time. Stochastic variants also inject noise along the trajectory, by
        default "edm_stochastic_heun"
    solver_options : dict[str, Any] | None, optional
        Options passed to the selected solver to customize its behavior or
        override defaults. Do not pass ``num_steps`` here; set the sampling step
        count with ``sampler_args["num_steps"]`` instead. By default None

    Badges
    ------
    region:na class:nowcasting product:sat year:2026 gpu:80gb
    provider:nvidia backend:pytorch
    """

    _REGISTRY_KEY = "goes"
    _INPUT_GRID = "goes-east-conus"
    _CONDITIONED = False

    def __init__(
        self,
        model_spec: list[dict[str, Any]],
        means: torch.Tensor,
        stds: torch.Tensor,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
        variables: np.ndarray = np.array(
            [
                "abi01c",
                "abi02c",
                "abi03c",
                "abi07c",
                "abi08c",
                "abi09c",
                "abi10c",
                "abi13c",
            ]
        ),
        conditioning_variables: np.ndarray | None = None,
        conditioning_means: torch.Tensor | None = None,
        conditioning_stds: torch.Tensor | None = None,
        glm_mask: torch.Tensor | None = None,
        conditioning_glm_mask: torch.Tensor | None = None,
        topo: torch.Tensor | None = None,
        nexrad_proximity: torch.Tensor | None = None,
        sampler_args: dict[str, float | int] | None = {"num_steps": 100, "S_churn": 10},
        input_times: np.ndarray = np.array([np.timedelta64(0, "h")]),
        output_times: np.ndarray = np.array([np.timedelta64(1, "h")]),
        y_coords: np.ndarray | None = None,
        x_coords: np.ndarray | None = None,
        max_dist_km: float = 6.0,
        amp: bool = True,
        compile: bool = False,
        *,
        solver: str = "edm_stochastic_heun",
        solver_options: dict[str, Any] | None = None,
    ):

        conditioned = conditioning_variables is not None and bool(
            len(conditioning_variables)
        )
        if conditioned and not self._CONDITIONED:
            raise ValueError(
                "StormScopeGOES is unforced; use StormScopeGOESConditioned for "
                "conditioned checkpoints"
            )
        if self._CONDITIONED and not conditioned:
            raise ValueError(f"{type(self).__name__} requires conditioning variables")
        super().__init__(
            model_spec=model_spec,
            solver=solver,
            solver_options=solver_options,
            means=means,
            stds=stds,
            variables=variables,
            latitudes=latitudes,
            longitudes=longitudes,
            conditioning_means=conditioning_means,
            conditioning_variables=conditioning_variables,
            conditioning_stds=conditioning_stds,
            glm_mask=glm_mask,
            conditioning_glm_mask=conditioning_glm_mask,
            topo=topo,
            nexrad_proximity=nexrad_proximity,
            sampler_args=sampler_args,
            input_times=input_times,
            output_times=output_times,
            y_coords=y_coords,
            x_coords=x_coords,
            max_dist_km=max_dist_km,
            amp=amp,
            compile=compile,
        )

    def default_sources(self) -> DataSource:
        """Recommend GOES-East CONUS imagery on the model grid.

        Returns
        -------
        DataSource
            GOES input source, regridded to the model grid by nearest neighbor
            within ``max_dist_km``; uncovered gridpoints are NaN. This model has
            no forcing slots.
        """
        from earth2studio.data import GOES, RegriddedSource

        return RegriddedSource(
            GOES(scan_mode="C"),
            _default_regridder(
                NearestRegridder,
                self._INPUT_GRID,
                self._grid(),
                max_dist_km=self.max_dist_km,
            ),
        )

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Compute the first GOES forecast without hooks.

        Parameters
        ----------
        x : xr.DataArray
            Initial GOES history matching ``input_coords()``.

        Returns
        -------
        xr.DataArray
            First forecast chunk.
        """
        return self.initialize(x)[0]

    def initialize(self, x: xr.DataArray) -> tuple[xr.DataArray, dict[str, Any]]:
        """Initialize an unforced GOES forecast without hooks.

        Parameters
        ----------
        x : xr.DataArray
            Complete initial GOES history.

        Returns
        -------
        tuple[xr.DataArray, dict[str, Any]]
            First forecast and missing history plus RNG state.
        """
        return self._initialize(x, None)

    def step(
        self, y: xr.DataArray, state: dict[str, Any]
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Advance GOES using the previous forecast and explicit state.

        Parameters
        ----------
        y : xr.DataArray
            Previous GOES forecast, including caller edits.
        state : dict[str, Any]
            Missing history and RNG state from the previous advance.

        Returns
        -------
        tuple[xr.DataArray, dict[str, Any]]
            Next forecast and updated state. No hooks are applied.
        """
        return self._forward(y, None, state)

    def create_iterator(self, x: xr.DataArray) -> Generator[xr.DataArray, None, None]:
        """Create an iterator for unforced GOES time-integration.

        Parameters
        ----------
        x : xr.DataArray
            Initial GOES history.

        Yields
        ------
        xr.DataArray
            Forecast fields, beginning with the first prediction.
        """
        yield from self._default_create_iterator(x)

    @classmethod
    def load_model(
        cls,
        package: Package,
        model_name: Literal["3km_10min", "6km_1hr", "3km_10min_flash"] = "3km_10min",
        amp: bool = True,
        compile: bool = False,
        *,
        solver: str | None = None,
        solver_options: dict[str, Any] | None = None,
    ) -> PrognosticModel:
        """Load model from package.

        Parameters
        ----------
        package : Package
            Package to load model from
        model_name : Literal["3km_10min", "6km_1hr", "3km_10min_flash"], optional
            Variant to load, by default ``"3km_10min"`` (the recommended CONUS
            nowcasting variant). Available variants (see
            :py:meth:`list_available_models`):

            - ``"3km_10min"``: 3km resolution, 10 minute timestep (CONUS nowcasting)
            - ``"3km_10min_flash"``: distilled three-kilometer, ten-minute variant
            - ``"6km_1hr": 6km resolution, 60 minute timestep (legacy nearcasting),
              conditioned on NWP fields

            Legacy training-style names are accepted as aliases.
        amp : bool, optional
            Enable automatic mixed precision (autocast) for the sampler's network
            forward passes. Default is True.
        compile : bool, optional
            Compile each staged expert with ``torch.compile`` ("reduce-overhead").
            Default is False.

        solver : str | None, optional
            Select the numerical method for integrating the sampling ODE in
            reverse time. Stochastic variants also inject noise along the
            trajectory. None selects stochastic Heun for the teacher; Flash
            variants reject solver overrides, by default None
        solver_options : dict[str, Any] | None, optional
            Options passed to the selected solver to customize its behavior or
            override defaults. Do not pass ``num_steps`` here; set the sampling step
            count with ``sampler_args["num_steps"]`` instead. By default None

        Returns
        -------
        PrognosticModel
            Instantiated model: :py:class:`StormScopeGOESConditioned` for variants
            with conditioning variables, otherwise StormScopeGOES.
        """
        if model_name == "3km_10min_flash" and (solver is not None or solver_options):
            raise ValueError(
                "StormScope Flash uses a fixed rollout and does not accept solver overrides."
            )
        try:
            package.resolve("config.json")  # HF tracking download statistics
        except FileNotFoundError:
            pass

        try:
            import natten
        except (ImportError, OSError) as exc:
            raise ImportError(
                "StormScope requires NATTEN for neighborhood attention. Install "
                "with `pip install earth2studio[stormscope]` or "
                "`uv add earth2studio --extra stormscope`."
            ) from exc
        if getattr(natten, "HAS_LIBNATTEN", True) is False:
            raise ImportError(
                "StormScope requires NATTEN with libnatten enabled. Reinstall "
                "NATTEN with a wheel or source build matching the active PyTorch "
                "and CUDA environment."
            )

        registry = cls._load_registry(package)
        _, pkg = cls._resolve_model_entry(package, model_name)
        if pkg.get("flash", False) and (solver is not None or solver_options):
            raise ValueError(
                "StormScope Flash uses a fixed rollout and does not accept solver overrides."
            )
        model_spec = cls._load_checkpoints(package, pkg)
        (
            latitudes,
            longitudes,
            y,
            x,
            input_times,
            output_times,
            spatial_downsample,
        ) = cls._build_grid_and_times(package, pkg)

        # State variables and conditioning variables (from the registry entry)
        variables = np.array(pkg["variables"])
        conditioning_variables = np.array(pkg["conditioning_vars"])

        # Normalization constants, selected by name from the canonical-order arrays
        means, stds, glm_mask = cls._build_normalization(package, registry, variables)
        if len(conditioning_variables) > 0:
            conditioning_means, conditioning_stds, conditioning_glm_mask = (
                cls._build_normalization(package, registry, conditioning_variables)
            )
        else:
            conditioning_means = torch.empty(0)
            conditioning_stds = torch.empty(0)
            conditioning_glm_mask = None

        # Static invariant channels (loaded only when the variant requests them)
        topo = (
            cls._load_invariant(package, "topo.npy", pkg) if pkg.get("topo") else None
        )
        nexrad_proximity = (
            cls._load_invariant(package, "nexrad_proximity.npy", pkg)
            if pkg.get("nexrad_proximity")
            else None
        )

        model_cls = cls
        if cls is StormScopeGOES and len(conditioning_variables):
            model_cls = StormScopeGOESConditioned
        model = model_cls(
            model_spec=model_spec,
            solver=solver or "edm_stochastic_heun",
            solver_options=solver_options,
            means=means.to(dtype=torch.float32),
            stds=stds.to(dtype=torch.float32),
            variables=variables,
            latitudes=latitudes.to(dtype=torch.float32),
            longitudes=longitudes.to(dtype=torch.float32),
            conditioning_means=conditioning_means.to(dtype=torch.float32),
            conditioning_stds=conditioning_stds.to(dtype=torch.float32),
            conditioning_variables=conditioning_variables,
            glm_mask=glm_mask,
            conditioning_glm_mask=conditioning_glm_mask,
            topo=topo,
            nexrad_proximity=nexrad_proximity,
            input_times=input_times,
            output_times=output_times,
            y_coords=y,
            x_coords=x,
            max_dist_km=6.0 * spatial_downsample,
            amp=amp,
            compile=False,
        )
        model._configure_variant(pkg, compile)
        model._set_default_coverage()
        return model


class StormScopeGOESConditioned(StormScopeGOES):
    """StormScope GOES forecasts conditioned on numerical weather prediction fields.

    Used by the legacy ``6km_1hr`` nearcasting variant, which conditions each step
    on 500 hPa geopotential (``z500``) at the input times.
    :py:meth:`StormScopeGOES.load_model` returns this class for variants with
    conditioning variables. Conditioning follows the same protocol as
    :py:class:`StormScopeMRMS`: pass the initial conditioning history, then the
    new frames at the lead times of each forecast.

    Parameters are as for :py:class:`StormScopeGOES`, except that
    ``conditioning_variables``, ``conditioning_means`` and ``conditioning_stds``
    are required.

    Badges
    ------
    region:na class:nowcasting product:sat year:2026 gpu:80gb
    provider:nvidia backend:pytorch
    """

    _CONDITIONED = True
    _CONDITIONING_GRID = "latlon-0.25deg"
    # Nearest-neighbor distance used for NWP conditioning before 1.0; the 0.25
    # degree grid needs more than the GOES distance to cover every gridpoint.
    _CONDITIONING_MAX_DIST_KM = 26.0

    def default_sources(  # type: ignore[override]
        self,
    ) -> tuple[DataSource, ForecastSource]:
        """Recommend GOES-East CONUS imagery and GFS conditioning on the model grid.

        Returns
        -------
        tuple[DataSource, ForecastSource]
            GOES input source, followed by GFS forecasts of the conditioning
            variables. Both are regridded to the model grid by nearest neighbor;
            uncovered gridpoints are NaN.
        """
        from earth2studio.data import GFS_FX, RegriddedSource

        gfs = RegriddedSource(
            GFS_FX(),
            _default_regridder(
                NearestRegridder,
                self._CONDITIONING_GRID,
                self._grid(),
                max_dist_km=self._conditioning_max_dist_km(),
            ),
        )
        return super().default_sources(), gfs

    def __call__(  # type: ignore[override]
        self, x: xr.DataArray, conditioning: xr.DataArray
    ) -> xr.DataArray:
        """Compute the first GOES forecast using caller-supplied conditioning.

        Parameters
        ----------
        x : xr.DataArray
            Initial GOES history matching ``input_coords()``.
        conditioning : xr.DataArray
            Initial conditioning history matching ``forcing_coords()``.

        Returns
        -------
        xr.DataArray
            First forecast chunk, without hooks.
        """
        return self.initialize(x, conditioning)[0]

    def initialize(  # type: ignore[override]
        self, x: xr.DataArray, conditioning: xr.DataArray
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Initialize GOES forecasts from GOES and conditioning histories.

        Parameters
        ----------
        x : xr.DataArray
            Complete initial GOES history.
        conditioning : xr.DataArray
            Complete conditioning history matching ``forcing_coords()``.

        Returns
        -------
        tuple[xr.DataArray, dict[str, Any]]
            First forecast and missing GOES/conditioning history plus RNG state.
            No hooks are applied and no data is fetched.
        """
        return self._initialize(x, conditioning)

    def step(  # type: ignore[override]
        self, y: xr.DataArray, conditioning: xr.DataArray, state: dict[str, Any]
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Advance GOES using the previous forecast and new conditioning.

        Parameters
        ----------
        y : xr.DataArray
            Previous GOES forecast, including caller edits.
        conditioning : xr.DataArray
            New conditioning frames at the lead times of ``y``. Earlier context is
            in state.
        state : dict[str, Any]
            Missing history and RNG state returned by the previous advance.

        Returns
        -------
        tuple[xr.DataArray, dict[str, Any]]
            Next forecast and updated state, without hooks or data fetching.
        """
        return self._forward(y, conditioning, state)

    def create_iterator(  # type: ignore[override]
        self, x: xr.DataArray, conditioning: xr.DataArray
    ) -> Generator[xr.DataArray, xr.DataArray | tuple[xr.DataArray, ...] | None, None]:
        """Create an iterator for conditioned GOES time-integration.

        Parameters
        ----------
        x : xr.DataArray
            Initial GOES history.
        conditioning : xr.DataArray
            Initial conditioning history. Send the new conditioning frames at the
            lead times of the latest yielded forecast to advance.

        Yields
        ------
        xr.DataArray
            Forecast fields, beginning with the first prediction.
        """
        yield from self._default_create_iterator(x, conditioning)

    @classmethod
    def load_model(  # type: ignore[override]
        cls,
        package: Package,
        model_name: Literal["6km_1hr"] = "6km_1hr",
        amp: bool = True,
        compile: bool = False,
        *,
        solver: str | None = None,
        solver_options: dict[str, Any] | None = None,
    ) -> PrognosticModel:
        """Load a conditioned GOES variant from a package.

        Parameters
        ----------
        package : Package
            Package to load model from.
        model_name : Literal["6km_1hr"], optional
            Conditioned variant to load, by default ``"6km_1hr"``.
        amp : bool, optional
            Enable automatic mixed precision (autocast) for the sampler's network
            forward passes. Default is True.
        compile : bool, optional
            Compile each staged expert with ``torch.compile`` ("reduce-overhead").
            Default is False.

        solver : str | None, optional
            Numerical method for integrating the sampling ODE in reverse time.
            None preserves stochastic Heun, by default None.
        solver_options : dict[str, Any] | None, optional
            Options passed to the selected solver to customize its behavior or
            override defaults. Set the sampling step count through
            ``sampler_args["num_steps"]``, by default None.

        Returns
        -------
        PrognosticModel
            Instantiated StormScopeGOESConditioned model.
        """
        return super().load_model(
            package,
            model_name=model_name,
            amp=amp,
            compile=compile,
            solver=solver,
            solver_options=solver_options,
        )


class StormScopeMRMS(StormScopeBase):
    """StormScope model forecasting MRMS data on the HRRR grid.

    This model supports multiple variants at different temporal resolutions,
    selected by passing ``model_name`` to ``load_model`` (default: ``"3km_10min"``).
    Variant names are semantic (``<resolution>_<cadence>``):

      - ``3km_10min``: 3km resolution, 10 minute timestep, MRMS+GLM nowcasting (default)
      - ``3km_10min_flash``: accelerated 3km / 10 minute CONUS MRMS+GLM nowcasting
      - ``6km_1hr``: 6km resolution, 60 minute timestep (legacy nearcasting)

    Use :py:meth:`list_available_models` to inspect the variants in a given package.
    Legacy training-style names are still accepted as aliases.

    Flash uses seven backbone calls per step. For regional inference within the
    native CONUS domain, call :meth:`~StormScopeBase.with_region` with the same
    region and ``region_padding`` as the coupled GOES model.

    Variants whose input cadence is finer than their output cadence use a sliding
    window of input timesteps and predict one output timestep; others use a single
    input timestep and predict one output timestep.

    All StormScopeMRMS models expect GOES-East data as conditioning. In a forecasting
    run this is typically supplied by a StormScopeGOES model: pass the GOES history to
    ``initialize`` (or ``create_iterator``), then pass the GOES forecast frames at the
    lead times of the previous radar forecast as ``conditioning`` to ``step`` (or send
    them to the iterator). When coupling with a StormScopeGOES model, initialize both
    models from the same GOES history and advance MRMS before GOES.

    Note
    ----
    For more information see the following references:

    - https://huggingface.co/nvidia/stormscope-goes-mrms
    - MRMS forecasting skill: evaluation procedure, metrics, and score table:
      https://huggingface.co/nvidia/stormscope-goes-mrms/blob/dcbe14a506e849b1afec9962dbf774f07d7aece8/mrms_forecast_skill.md

    Parameters
    ----------
    model_spec : list[dict[str, Any]]
        Sequence of stage specifications; see `StormScopeBase`.
    means : torch.Tensor
        Per-variable mean for normalization, shape [1, C, 1, 1].
    stds : torch.Tensor
        Per-variable std for normalization, shape [1, C, 1, 1].
    latitudes : torch.Tensor
        Latitudes of the grid, expected shape [H, W].
    longitudes : torch.Tensor
        Longitudes of the grid, expected shape [H, W].
    variables : np.ndarray, optional
        MRMS input variables. Default is ["refc"].
    conditioning_variables : np.ndarray, optional
        Auxiliary conditioning variables (typically GOES channels). Default is
        ["abi01c", "abi02c", "abi03c", "abi07c", "abi08c", "abi09c", "abi10c", "abi13c"].
    conditioning_means : torch.Tensor | None, optional
        Means to normalize any external conditioning data. Default is None.
    conditioning_stds : torch.Tensor | None, optional
        Stds to normalize any external conditioning data. Default is None.
    sampler_args : dict[str, float | int] | None, optional
        Default sampler arguments passed to the diffusion sampler.
        Default is {"num_steps": 100, "S_churn": 10}.
    y_coords : np.ndarray | None, optional
        Y coordinates of the grid, expected shape [H, W]. Default is None, in which
        case the model uses the enumerated indices inferred from the latitude and
        longitude grid shapes.
    x_coords : np.ndarray | None, optional
        X coordinates of the grid, expected shape [H, W]. Default is None, in which
        case the model uses the enumerated indices inferred from the latitude and
        longitude grid shapes.
    input_times : np.ndarray, optional
        Input timesteps, of type timedelta64. Default is [0 h] (i.e., the current time).
    output_times : np.ndarray, optional
        Output timesteps, of type timedelta64. Default is [1 h] (i.e., 1 hour from the current time).
    max_dist_km : float, optional
        Distance in kilometers beyond which the default MRMS and GOES regridders
        leave a model gridpoint without data. Default is 6.0.
    mrms_coverage_mask : torch.Tensor | None, optional
        Boolean NEXRAD-coverage mask of shape ``[H, W]`` on the model grid, True
        where MRMS data is considered valid (inside NEXRAD circular coverage).
        When provided, it is used as the initial ``valid_mask`` and is ANDed with
        the coverage of the default MRMS regridder. Loaded automatically from the
        package for non-deprecated variants. Default is None.

    Note
    ----
    To have a unified coordinate system over CONUS for convenience, the model uses the HRRR grid.
    As a result, there are portions of the domain which go beyond the extent of the MRMS data,
    so these portions are masked as invalid (set to NaN).

    Note
    ----
    **GLM state channel.**  The ``3km_10min`` and ``3km_10min_flash`` variants
    include a ``glm_density`` channel (gridded GLM lightning counts, normalized
    with ``log1p``) as part of its *state* — both an input observation and a predicted output (the
    ``6km_1hr`` variant has no GLM channel).  Because the GLM source lives on a
    different native grid from MRMS, the default input source composes MRMS with
    :py:class:`earth2studio.data.GOESGLMGrid`, regridded bilinearly (as in
    training) with zero fill outside its grid.

    The GLM channels of ``x`` are observed before ``initialize``. During ``step``,
    GLM flows autoregressively from the previous radar forecast. GOES forcing is
    passed separately by the caller; neither method fetches observations or
    applies iterator hooks.

    solver : str, optional
        Select the numerical method for integrating the sampling ODE in reverse
        time. Stochastic variants also inject noise along the trajectory, by
        default "edm_stochastic_heun"
    solver_options : dict[str, Any] | None, optional
        Options passed to the selected solver to customize its behavior or
        override defaults. Do not pass ``num_steps`` here; set the sampling step
        count with ``sampler_args["num_steps"]`` instead. By default None

    Badges
    ------
    region:na class:nowcasting product:radar year:2026 gpu:80gb
    provider:nvidia backend:pytorch
    """

    _REGISTRY_KEY = "mrms"
    _INPUT_GRID = "mrms-conus-0.01deg"
    _CONDITIONING_GRID = "goes-east-conus"
    _GLM_GRID = "glm-conus-0.1deg"
    _STATE_FIRST = False
    # Legacy 6 km / 1 hr nearcast checkpoints were trained with -10 dBZ infill
    # (physical space) before normalization; that maps to this normalized value.
    _LEGACY_INPUT_INVALID_FILL_CONSTANT = -0.25285158

    def __init__(
        self,
        model_spec: list[dict[str, Any]],
        means: torch.Tensor,
        stds: torch.Tensor,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
        variables: np.ndarray = np.array(["refc"]),
        conditioning_variables: np.ndarray = np.array(
            [
                "abi01c",
                "abi02c",
                "abi03c",
                "abi07c",
                "abi08c",
                "abi09c",
                "abi10c",
                "abi13c",
            ]
        ),
        conditioning_means: torch.Tensor | None = None,
        conditioning_stds: torch.Tensor | None = None,
        glm_mask: torch.Tensor | None = None,
        conditioning_glm_mask: torch.Tensor | None = None,
        topo: torch.Tensor | None = None,
        nexrad_proximity: torch.Tensor | None = None,
        mrms_coverage_mask: torch.Tensor | None = None,
        sampler_args: dict[str, float | int] | None = {"num_steps": 100, "S_churn": 10},
        y_coords: np.ndarray | None = None,
        x_coords: np.ndarray | None = None,
        input_times: np.ndarray = np.array([np.timedelta64(0, "h")]),
        output_times: np.ndarray = np.array([np.timedelta64(1, "h")]),
        max_dist_km: float = 6.0,
        amp: bool = True,
        compile: bool = False,
        *,
        solver: str = "edm_stochastic_heun",
        solver_options: dict[str, Any] | None = None,
    ):

        super().__init__(
            model_spec=model_spec,
            solver=solver,
            solver_options=solver_options,
            means=means,
            stds=stds,
            variables=variables,
            latitudes=latitudes,
            longitudes=longitudes,
            conditioning_means=conditioning_means,
            conditioning_variables=conditioning_variables,
            conditioning_stds=conditioning_stds,
            glm_mask=glm_mask,
            conditioning_glm_mask=conditioning_glm_mask,
            topo=topo,
            nexrad_proximity=nexrad_proximity,
            sampler_args=sampler_args,
            y_coords=y_coords,
            x_coords=x_coords,
            input_times=input_times,
            output_times=output_times,
            max_dist_km=max_dist_km,
            amp=amp,
            compile=compile,
        )
        # NEXRAD circular coverage mask. When set, it defines which pixels are
        # valid MRMS observations (matching training-time infilling boundaries).
        # Used as the initial valid_mask and ANDed into the regridder coverage.
        if mrms_coverage_mask is not None:
            self.register_buffer("mrms_coverage_mask", mrms_coverage_mask)
            self.register_buffer("valid_mask", mrms_coverage_mask.clone())

        # GLM is a state channel (`glm_density`) normalized with log1p/expm1, regridded
        # from its own 0.1-degree gridded source onto the model grid by the default
        # input source. ``n_glm_channels`` counts the log1p (GLM) channels among
        # `variables`.
        self.n_glm_channels = int(self.glm_mask.sum().item())
        # Names of the GLM (log1p) state channels, in `variables` order.
        self.glm_variables = np.asarray(self.variables)[self.glm_mask.cpu().numpy()]

    def _set_default_coverage(self) -> None:
        super()._set_default_coverage()
        coverage = self._buffers.get("mrms_coverage_mask", None)
        if coverage is not None:
            self.register_buffer(
                "valid_mask",
                self.valid_mask & coverage.to(device=self.valid_mask.device),
            )

    def default_sources(self) -> tuple[DataSource, DataSource]:
        """Recommend MRMS radar and GLM inputs and GOES-East CONUS conditioning.

        Every source is regridded to the model grid. MRMS and GOES use nearest
        neighbor within ``max_dist_km`` and leave uncovered gridpoints NaN. GLM, for
        variants with a GLM channel, uses bilinear interpolation with zero fill.

        Returns
        -------
        tuple[DataSource, DataSource]
            Input source, composing MRMS with gridded GLM where the variant has a
            GLM channel, followed by the GOES conditioning source.
        """
        from earth2studio.data import (
            GOES,
            MRMS,
            CompositeSource,
            GOESGLMGrid,
            RegriddedSource,
        )

        grid = self._grid()
        source: DataSource = RegriddedSource(
            MRMS(),
            _default_regridder(
                NearestRegridder, self._INPUT_GRID, grid, max_dist_km=self.max_dist_km
            ),
        )
        if len(self.glm_variables):
            glm = RegriddedSource(
                GOESGLMGrid(),
                _default_regridder(
                    BilinearRegridder, self._GLM_GRID, grid, fill_value=0.0
                ),
            )
            radar = [v for v in self.variables if v not in set(self.glm_variables)]
            source = CompositeSource(
                {"radar": source, "glm": glm},
                {"radar": radar, "glm": list(self.glm_variables)},
            )
        goes = RegriddedSource(
            GOES(scan_mode="C"),
            _default_regridder(
                NearestRegridder,
                self._CONDITIONING_GRID,
                grid,
                max_dist_km=self.max_dist_km,
            ),
        )
        return source, goes

    def __call__(self, x: xr.DataArray, conditioning: xr.DataArray) -> xr.DataArray:
        """Compute the first radar forecast using caller-supplied GOES history.

        Parameters
        ----------
        x : xr.DataArray
            Initial radar history, including checkpoint GLM channels if required.
        conditioning : xr.DataArray
            Initial GOES history matching ``forcing_coords()``.

        Returns
        -------
        xr.DataArray
            First radar forecast, without hooks.
        """
        return self.initialize(x, conditioning)[0]

    def initialize(
        self, x: xr.DataArray, conditioning: xr.DataArray
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Initialize radar forecasts from radar and GOES histories.

        Parameters
        ----------
        x : xr.DataArray
            Complete initial radar history.
        conditioning : xr.DataArray
            Complete GOES history matching ``forcing_coords()``.

        Returns
        -------
        tuple[xr.DataArray, dict[str, Any]]
            First forecast and missing radar/GOES history plus RNG state.
            No hooks are applied and no data is fetched.
        """
        return self._initialize(x, conditioning)

    def step(
        self, y: xr.DataArray, conditioning: xr.DataArray, state: dict[str, Any]
    ) -> tuple[xr.DataArray, dict[str, Any]]:
        """Advance radar using the previous radar and GOES forecasts.

        Parameters
        ----------
        y : xr.DataArray
            Previous radar forecast, including caller edits.
        conditioning : xr.DataArray
            New GOES frames at the lead times of ``y``. Earlier context is in state.
        state : dict[str, Any]
            Missing history and RNG state returned by the previous advance.

        Returns
        -------
        tuple[xr.DataArray, dict[str, Any]]
            Next radar forecast and updated state, without hooks or data fetching.
        """
        return self._forward(y, conditioning, state)

    def create_iterator(
        self, x: xr.DataArray, conditioning: xr.DataArray
    ) -> Generator[xr.DataArray, xr.DataArray | tuple[xr.DataArray, ...] | None, None]:
        """Create an iterator for radar time-integration with GOES conditioning.

        Parameters
        ----------
        x : xr.DataArray
            Initial radar history.
        conditioning : xr.DataArray
            Initial GOES history. Send the new GOES frames at the lead times of the
            latest yielded forecast to advance.

        Yields
        ------
        xr.DataArray
            Forecast fields, beginning with the first prediction.
        """
        yield from self._default_create_iterator(x, conditioning)

    def prep_input(
        self, x: torch.Tensor, coords: CoordSystem, conditioning: bool = False
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Prepares the input tensor for the MRMS prognostic model. Same behavior as the base
        class, but with additional value imputation/standardization for low reflectivity values.
        """

        x, x_coords = super().prep_input(x, coords, conditioning=conditioning)

        if not conditioning:
            x = torch.where(
                x <= -20.0, -10, x
            )  # Impute -10 for low reflectivity values

        return x, x_coords

    @classmethod
    def load_model(
        cls,
        package: Package,
        model_name: Literal["3km_10min", "6km_1hr", "3km_10min_flash"] = "3km_10min",
        amp: bool = True,
        compile: bool = False,
        *,
        solver: str | None = None,
        solver_options: dict[str, Any] | None = None,
    ) -> PrognosticModel:
        """Load model from package.

        Parameters
        ----------
        package : Package
            Package to load model from
        model_name : Literal["3km_10min", "6km_1hr", "3km_10min_flash"], optional
            Variant to load. Available variants (see
            :py:meth:`list_available_models`):

            - ``"3km_10min"``: 3km resolution, 10 minute timestep, MRMS+GLM nowcasting
            - ``"3km_10min_flash"``: distilled three-kilometer, ten-minute variant
            - ``"6km_1hr": 6km resolution, 60 minute timestep, MRMS+GLM nearcasting

            Legacy training-style names are accepted as aliases.
            Default is ``"3km_10min"``.
        amp : bool, optional
            Enable automatic mixed precision (autocast) for the sampler's network
            forward passes. Default is True.
        compile : bool, optional
            Compile each staged expert with ``torch.compile`` ("reduce-overhead").
            Default is False.

        solver : str | None, optional
            Select the numerical method for integrating the sampling ODE in
            reverse time. Stochastic variants also inject noise along the
            trajectory. None selects stochastic Heun for the teacher; Flash
            variants reject solver overrides, by default None
        solver_options : dict[str, Any] | None, optional
            Options passed to the selected solver to customize its behavior or
            override defaults. Do not pass ``num_steps`` here; set the sampling step
            count with ``sampler_args["num_steps"]`` instead. By default None

        Returns
        -------
        PrognosticModel
            Instantiated StormScopeMRMS model
        """
        if model_name == "3km_10min_flash" and (solver is not None or solver_options):
            raise ValueError(
                "StormScope Flash uses a fixed rollout and does not accept solver overrides."
            )
        try:
            package.resolve("config.json")  # HF tracking download statistics
        except FileNotFoundError:
            pass

        try:
            import natten
        except (ImportError, OSError) as exc:
            raise ImportError(
                "StormScope requires NATTEN for neighborhood attention. Install "
                "with `pip install earth2studio[stormscope]` or "
                "`uv add earth2studio --extra stormscope`."
            ) from exc
        if getattr(natten, "HAS_LIBNATTEN", True) is False:
            raise ImportError(
                "StormScope requires NATTEN with libnatten enabled. Reinstall "
                "NATTEN with a wheel or source build matching the active PyTorch "
                "and CUDA environment."
            )

        registry = cls._load_registry(package)
        _, pkg = cls._resolve_model_entry(package, model_name)
        if pkg.get("flash", False) and (solver is not None or solver_options):
            raise ValueError(
                "StormScope Flash uses a fixed rollout and does not accept solver overrides."
            )
        model_spec = cls._load_checkpoints(package, pkg)
        (
            latitudes,
            longitudes,
            y,
            x,
            input_times,
            output_times,
            spatial_downsample,
        ) = cls._build_grid_and_times(package, pkg)

        # State variables and conditioning variables (from the registry entry)
        variables = np.array(pkg["variables"])
        conditioning_variables = np.array(pkg["conditioning_vars"])

        # Normalization constants, selected by name from the canonical-order arrays
        means, stds, glm_mask = cls._build_normalization(package, registry, variables)
        if len(conditioning_variables) > 0:
            conditioning_means, conditioning_stds, conditioning_glm_mask = (
                cls._build_normalization(package, registry, conditioning_variables)
            )
        else:
            conditioning_means = torch.empty(0)
            conditioning_stds = torch.empty(0)
            conditioning_glm_mask = None

        # Static invariant channels (loaded only when the variant requests them)
        topo = (
            cls._load_invariant(package, "topo.npy", pkg) if pkg.get("topo") else None
        )
        nexrad_proximity = (
            cls._load_invariant(package, "nexrad_proximity.npy", pkg)
            if pkg.get("nexrad_proximity")
            else None
        )

        # NEXRAD circular coverage mask: defines which pixels were valid MRMS
        # observations during training (inside NEXRAD radar coverage). Loaded as
        # bool so that it ANDs cleanly with the regridder-derived valid_mask.
        # File convention: True = outside coverage / void pixel. Inverted here
        # so the buffer and constructor use True = valid (data present).
        if pkg.get("mrms_coverage_mask"):
            arr = ~torch.from_numpy(
                np.load(package.resolve("mrms_coverage_mask.npy"))
            ).bool()
            mrms_coverage_mask = cls._crop_invariant(
                arr, pkg["image_size"], pkg["spatial_downsample"]
            )
        else:
            mrms_coverage_mask = None

        model = cls(
            model_spec=model_spec,
            solver=solver or "edm_stochastic_heun",
            solver_options=solver_options,
            means=means.to(dtype=torch.float32),
            stds=stds.to(dtype=torch.float32),
            variables=variables,
            latitudes=latitudes.to(dtype=torch.float32),
            longitudes=longitudes.to(dtype=torch.float32),
            conditioning_means=conditioning_means.to(dtype=torch.float32),
            conditioning_stds=conditioning_stds.to(dtype=torch.float32),
            conditioning_variables=conditioning_variables,
            glm_mask=glm_mask,
            conditioning_glm_mask=conditioning_glm_mask,
            topo=topo,
            nexrad_proximity=nexrad_proximity,
            mrms_coverage_mask=mrms_coverage_mask,
            y_coords=y,
            x_coords=x,
            input_times=input_times,
            output_times=output_times,
            max_dist_km=6.0 * spatial_downsample,
            amp=amp,
            compile=False,
        )
        model._configure_variant(pkg, compile)
        model._set_default_coverage()
        if pkg.get("deprecated", False):
            model._INPUT_INVALID_FILL_CONSTANT = cls._LEGACY_INPUT_INVALID_FILL_CONSTANT
        return model


class _StagedSolver(Solver):
    def __init__(
        self,
        model: StormScopeBase,
        condition: torch.Tensor | None,
        scheduler: "EDMNoiseScheduler",
        solver: str,
        options: dict[str, Any],
        progress_bar: Any | None,
    ) -> None:
        self.model = model
        self.condition = condition
        self.progress_bar = progress_bar
        self.current_time = torch.as_tensor(
            scheduler.sigma_max, device=model.means.device, dtype=model._SAMPLER_DTYPE
        )
        self.denoiser = scheduler.get_denoiser(x0_predictor=self._predict_x0)
        self.solver = SOLVERS[solver](self.denoiser, **options)

    def _predict_x0(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        # Churn changes the network input noise but keeps the departure-time expert.
        expert = self.model._select_expert(torch.minimum(t[0], self.current_time))
        return expert(
            x.to(self.model._MODEL_DTYPE),
            t.to(self.model._MODEL_DTYPE),
            condition=self.condition,
        ).to(self.model._SAMPLER_DTYPE)

    def step(
        self,
        x: torch.Tensor,
        t_cur: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        self.current_time = t_cur[0]
        result = self.solver.step(x, t_cur, t_next)
        if self.progress_bar is not None:
            self.progress_bar.update()
        return result
