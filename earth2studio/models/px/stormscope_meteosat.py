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

import types
from collections.abc import Callable, Generator
from contextlib import AbstractContextManager, nullcontext
from copy import copy, deepcopy
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal, Self

import numpy as np
import torch
import xarray as xr
from numpy.typing import NDArray
from pyproj import CRS, Transformer

from earth2studio.data.base import DataSource
from earth2studio.grids import GridDefinition, ProjectedGrid, Region, resolve_grid
from earth2studio.grids._regrid import _check_axes, _grid_attrs
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.models.batch import batch_func
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.stormscope import crop_to_region
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.models.utils import fork_rng
from earth2studio.utils import (
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
from earth2studio.utils.type import CoordinateSystem

if TYPE_CHECKING:
    from earth2studio.models.nn.stormscope_flash import RegionInfo

try:
    import cupy as cp
except ImportError:
    cp = None

try:
    import physicsnemo.nn.module.dit_layers as _dit_layers
    from omegaconf import OmegaConf
    from physicsnemo.diffusion.noise_schedulers import EDMNoiseScheduler
    from physicsnemo.diffusion.preconditioners import EDMPreconditioner
    from physicsnemo.diffusion.samplers import sample
    from physicsnemo.utils.zenith_angle import zenith_azimuth_angles
except ImportError:
    OptionalDependencyFailure("stormscope")
    OmegaConf = None


# Variables used in StormScopeMeteosatEU
VARIABLES = (
    "fci04vis",
    "fci05vis",
    "fci06vis",
    "fci08vis",
    "fci09vis",
    "fci13nir",
    "fci16nir",
    "fci22nir",
    "fci38ir",
    "fci63wv",
    "fci73wv",
    "fci87ir",
    "fci97ir",
    "fci105ir",
    "fci123ir",
    "fci133ir",
)


def _same_state(left: xr.DataArray, right: xr.DataArray) -> bool:
    return left.variable.equals(right.variable) and all(
        left.coords[d].variable.equals(right.coords[d].variable)
        for d in left.dims
        if d in left.coords
    )


# FCI grid whose crop the model runs on, and the finer grids its channels come on.
_MODEL_GRID = "mtg-fci-fulldisk-2km"
_BLOCK = {"2km": 1, "1km": 2}


def _crop_bounds(axis: NDArray[Any], labels: NDArray[Any]) -> tuple[int, int]:
    """Positions ``[start, stop)`` of ``labels`` as a contiguous run of ``axis``."""
    start = int(np.abs(axis - labels[0]).argmin())
    stop = start + labels.size
    tolerance = 1e-3 * np.abs(np.diff(axis[:2])).item()
    if stop > axis.size or not np.allclose(
        axis[start:stop], labels, rtol=0, atol=tolerance
    ):
        raise ValueError(f"Model grid is not a crop of the {_MODEL_GRID} grid")
    return start, stop


class _BlockMeanRegridder:
    """Average blocks of FCI pixels onto a crop of the 2 km FCI grid.

    Each 2 km pixel is centred on a 2x2 block of 1 km pixels, so 1 km channels are
    block means and 2 km channels a selection (a 1x1 block). Blocks are summed in
    row-major order in the field's floating dtype and divided by their size, as
    ``torch.nn.functional.avg_pool2d`` does on CUDA. Calls run on the field's NumPy
    or CuPy backing.

    Parameters
    ----------
    resolution : {"2km", "1km"}
        FCI resolution of the fields to regrid.
    target_grid : ProjectedGrid
        Crop of the ``mtg-fci-fulldisk-2km`` grid to produce.
    """

    def __init__(
        self, resolution: Literal["2km", "1km"], target_grid: ProjectedGrid
    ) -> None:
        coarse = resolve_grid(_MODEL_GRID)
        fine = resolve_grid(f"mtg-fci-fulldisk-{resolution}")
        if not isinstance(coarse, ProjectedGrid) or not isinstance(fine, ProjectedGrid):
            raise TypeError("FCI grids must be projected grids")
        self.factor = _BLOCK[resolution]
        rows = _crop_bounds(coarse.y, target_grid.y)
        cols = _crop_bounds(coarse.x, target_grid.x)
        (i0, i1), (j0, j1) = (
            (self.factor * rows[0], self.factor * rows[1]),
            (self.factor * cols[0], self.factor * cols[1]),
        )
        self.pixel_bbox = ((i0, i1), (j0, j1))
        self._source_grid = ProjectedGrid(fine.y[i0:i1], fine.x[j0:j1], fine.crs)
        self._target_grid: GridDefinition = target_grid
        self._target_coords = target_grid.coords()

    @property
    def source_grid(self) -> GridDefinition:
        """FCI crop the regridder accepts, at its native resolution."""
        return self._source_grid

    @property
    def target_grid(self) -> GridDefinition:
        """Crop of the 2 km FCI grid the regridder produces."""
        return self._target_grid

    def to(self, device: Any) -> "_BlockMeanRegridder":
        """Return this regridder; it holds no device tables."""
        return self

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Average the spatial blocks of an FCI field onto the target grid.

        Parameters
        ----------
        x : xr.DataArray
            NumPy- or CuPy-backed field on ``source_grid``.

        Returns
        -------
        xr.DataArray
            Field on ``target_grid`` with the backing of ``x``, its non-spatial
            dimensions first.
        """
        dims = self._source_grid.dims
        if tuple(x.sizes.get(dim) for dim in dims) != self._source_grid.shape:
            raise ValueError(
                f"Field dimensions {dict(x.sizes)} do not match the source grid "
                f"{dict(zip(dims, self._source_grid.shape))}"
            )
        _check_axes(x, self._source_grid)
        leading = tuple(dim for dim in x.dims if dim not in dims)
        x = x.transpose(*leading, *dims)
        data = x.data
        context: AbstractContextManager[Any] = nullcontext()
        if cp is not None and isinstance(data, cp.ndarray):
            xp, context = cp, cp.cuda.Device(data.device.id)
        else:
            xp, data = np, np.asarray(data)
        dtype = data.dtype if np.issubdtype(data.dtype, np.floating) else np.float32
        k = self.factor
        with context:
            data = data.astype(dtype, copy=False)
            if k > 1:
                height, width = self._target_grid.shape
                blocks = data.reshape(*data.shape[:-2], height, k, width, k)
                total = xp.zeros(blocks.shape[:-3] + (width,), dtype=dtype)
                for di in range(k):
                    for dj in range(k):
                        total = total + blocks[..., di, :, dj]
                data = total / dtype.type(k * k)
        coords = {
            name: coordinate.variable
            for name, coordinate in x.coords.items()
            if not set(coordinate.dims).intersection(dims)
        }
        coords.update(
            {name: value.variable for name, value in self._target_coords.items()}
        )
        return xr.DataArray(
            data,
            dims=(*leading, *dims),
            coords=coords,
            name=x.name,
            attrs=_grid_attrs(x.attrs, self._source_grid, self._target_grid, None),
        )


@check_optional_dependencies()
class StormScopeMeteosatEU(torch.nn.Module, AutoModelMixin, PrognosticMixin):
    """Generative diffusion nowcasting model for MTG-I1 FCI satellite imagery.

    Predicts MTG Full Combined Imager (FCI) frames from ``len(input_times)``
    consecutive input frames at 10-min resolution. Uses an "ensemble of experts"
    denoising strategy: the diffusion sampler is invoked in stages, each using a
    different set of model weights applicable to a specific sigma range, as defined
    by ``model_spec``.

    The model runs on a crop of the 2 km FCI full-disk grid
    (``mtg-fci-fulldisk-2km``) over Europe, labelled by FCI scan angles.
    :meth:`default_sources` fetches every channel on that crop, block-averaging the
    1 km visible and near-infrared channels. Regional views from
    :meth:`with_region` run on the full domain, because the networks' positional
    embeddings fix their input size, and :meth:`crop_output` crops forecasts to the
    requested region.

    Parameters
    ----------
    model_spec : list[dict[str, Any]]
        Sequence of stage specifications. Each entry must contain:

        - ``model``: torch.nn.Module
        - ``sigma_min``: float
        - ``sigma_max``: float

        The sigma interval determines which expert is applied during denoising.
        The ranges must combine to a single continuous interval with no gaps or overlaps.
    means : torch.Tensor
        Per-channel mean of raw pixel counts used for normalisation, shape ``(C, 1, 1)``.
    stds : torch.Tensor
        Per-channel standard deviation of raw pixel counts used for normalisation,
        shape ``(C, 1, 1)``.
    scale_factor : torch.Tensor
        Per-channel multiplicative factor for the raw-to-physical radiance conversion,
        shape ``(C, 1, 1)``.
    add_offset : torch.Tensor
        Per-channel additive offset for the raw-to-physical radiance conversion,
        shape ``(C, 1, 1)``.
    invariants : torch.Tensor
        Static invariant fields (e.g. orography, land-sea mask) on the model grid,
        shape ``(1, C_inv, H, W)``.
    latitudes : torch.Tensor
        Latitudes of the model grid used for solar angles, shape ``(H, W)``.
    longitudes : torch.Tensor
        Longitudes of the model grid used for solar angles, shape ``(H, W)``.
    earth_mask : torch.Tensor
        Boolean mask where ``True`` indicates pixels over Earth (land or ocean),
        shape ``(H, W)``.
    y_coords : np.ndarray
        FCI y scan angles (radians) of the model grid rows, a contiguous run of the
        2 km full-disk grid, shape ``(H,)``.
    x_coords : np.ndarray
        FCI x scan angles (radians) of the model grid columns, a contiguous run of
        the 2 km full-disk grid, shape ``(W,)``.
    variables : np.ndarray, optional
        Channel names corresponding to the 16 FCI bands,
        by default ``np.array(VARIABLES)``
    sampler_args : dict[str, float | int] | None, optional
        Overrides for the EDM sampler/scheduler. Recognised keys:
        ``sigma_min``, ``sigma_max``, ``rho`` (scheduler), ``num_steps``
        (sampling steps, 48 by default) and ``S_churn``, ``S_min``, ``S_max``,
        ``S_noise`` (solver).
        ``sigma_min``/``sigma_max`` must fall within the combined sigma range
        spanned by ``model_spec`` and default to that combined range if not
        given. Other unspecified keys use sensible defaults, by default None
    input_times : np.ndarray, optional
        Context-window time offsets relative to the analysis time; each element is a
        ``np.timedelta64``. The number of elements ``L`` determines the sliding-window
        length, by default a 6-frame window ``[-50 min, …, 0 min]``
    output_times : np.ndarray, optional
        Output time offsets relative to the analysis time; single-step prediction
        only, by default ``[10 min]``
    ir_38_warm_scale_factor : float, optional
        Multiplier applied to the above-threshold branch of the ir_38 channel after
        raw-to-physical conversion, by default 0.024222141
    ir_38_warm_threshold : float, optional
        Raw digital-count boundary (the 12-bit maximum) above which the warm/HDR
        ir_38 scaling is applied, by default 4095.0
    batch_size : int, optional
        Maximum number of samples processed per forward pass, by default 1
    amp : bool, optional
        Whether to use automatic mixed precision (bfloat16) during the diffusion
        forward pass, by default True
    compile : bool, optional
        Compile each denoising expert with ``torch.compile``; see
        :meth:`compile_experts`. Default is False.

    Badges
    ------
    region:eu class:nowcasting product:sat year:2026 gpu:80gb
    provider:nvidia backend:pytorch
    """

    def __init__(
        self,
        model_spec: list[dict[str, Any]],
        means: torch.Tensor,
        stds: torch.Tensor,
        scale_factor: torch.Tensor,
        add_offset: torch.Tensor,
        invariants: torch.Tensor,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
        earth_mask: torch.Tensor,
        y_coords: np.ndarray,
        x_coords: np.ndarray,
        variables: np.ndarray = np.array(VARIABLES),
        sampler_args: dict[str, float | int] | None = None,
        input_times: np.ndarray = np.arange(-5, 1) * np.timedelta64(10, "m"),
        output_times: np.ndarray = np.array([np.timedelta64(10, "m")]),
        ir_38_warm_scale_factor: float = 0.024222141,
        ir_38_warm_threshold: float = 4095.0,
        batch_size: int = 1,
        amp: bool = True,
        compile: bool = False,
    ):
        super().__init__()

        # Sort experts by ascending sigma
        model_spec = sorted(model_spec, key=lambda s: float(s["sigma_min"]))
        # Store in tensor / ModuleList so `.to(device)` works
        self.register_buffer(
            "expert_sigma_min",
            torch.as_tensor([spec["sigma_min"] for spec in model_spec]),
        )
        self.register_buffer(
            "expert_sigma_max",
            torch.as_tensor([spec["sigma_max"] for spec in model_spec]),
        )
        if not (self.expert_sigma_min[1:] == self.expert_sigma_max[:-1]).all():
            raise ValueError(
                "Expert sigma ranges must combine to a single continuous range."
            )
        self.experts = torch.nn.ModuleList([spec["model"] for spec in model_spec])
        combined_sigma_min = model_spec[0]["sigma_min"]
        combined_sigma_max = model_spec[-1]["sigma_max"]

        self.register_buffer("means", means)
        self.register_buffer("stds", stds)
        self.register_buffer("scale_factor", scale_factor)
        self.register_buffer("add_offset", add_offset)
        self.register_buffer("invariants", invariants)
        self.sampler_args = {
            "sigma_min": combined_sigma_min,
            "sigma_max": combined_sigma_max,
            "sigma_data": 1.0,
            "rho": 7,
            "S_churn": 0.0,
            "S_min": 0.0,
            "S_max": float("inf"),
            "S_noise": 1,
            "num_steps": 48,
        }
        if sampler_args is not None:
            self.sampler_args.update(sampler_args)
        if (
            self.sampler_args["sigma_min"] < combined_sigma_min
            or self.sampler_args["sigma_max"] > combined_sigma_max
        ):
            raise ValueError(
                f"The sampled sigma range [{self.sampler_args['sigma_min']}, "
                f"{self.sampler_args['sigma_max']}] must be contained within the "
                f"model sigma range [{combined_sigma_min}, {combined_sigma_max}]."
            )

        if latitudes.shape != (len(y_coords), len(x_coords)):
            raise ValueError("Grid coordinates must match the latitude shape")
        self.grid = ProjectedGrid(y_coords, x_coords, resolve_grid(_MODEL_GRID).crs)
        self._signature: xr.DataArray | None = None
        # Region of a view from with_region; None on the full domain.
        self.region_info: RegionInfo | None = None

        self.variables = variables

        # Grid fields needed for GPU ops
        self.register_buffer("latitudes", torch.as_tensor(latitudes))
        self.register_buffer("longitudes", torch.as_tensor(longitudes))
        self.register_buffer("off_earth_mask", ~torch.as_tensor(earth_mask))

        # precalculate some normalization factors
        self.register_buffer("mean_inv_std", -self.means / self.stds)
        self.register_buffer("inv_std", 1.0 / self.stds)
        self.register_buffer("offset_inv_scale", -self.add_offset / self.scale_factor)
        self.register_buffer("inv_scale", 1.0 / self.scale_factor)

        # min and max values for clamping
        num_variables = len(variables)
        min_values = self.raw_to_normalized(
            torch.as_tensor([0] * num_variables).reshape(1, num_variables, 1, 1)
        )
        max_values = torch.as_tensor([4095] * num_variables)
        self.ir_38_channel: int | None = None
        if "fci38ir" in variables:
            # unlike other MTG-FCI channels, ir_38 can have uint16 values up to 2**13-1 == 8191
            # values > 2**12-1 == 4095 use a different scaling
            self.ir_38_channel = list(variables).index("fci38ir")
            self.register_buffer(
                "ir_38_warm_multiplier",
                torch.as_tensor(
                    ir_38_warm_scale_factor / scale_factor[self.ir_38_channel]
                ).to(dtype=torch.float32),
            )
            self.register_buffer(
                "ir_38_warm_threshold",
                torch.as_tensor(ir_38_warm_threshold).to(dtype=torch.float32),
            )
            max_values[self.ir_38_channel] = 8191

        max_values = self.raw_to_normalized(max_values.reshape(1, num_variables, 1, 1))
        self.register_buffer("min_values", min_values)
        self.register_buffer("max_values", max_values)

        _scheduler_keys = {"sigma_min", "sigma_max", "sigma_data", "rho"}
        self.scheduler = EDMNoiseScheduler(
            **{k: v for k, v in self.sampler_args.items() if k in _scheduler_keys}
        )

        self.batch_size = batch_size
        self.amp = amp

        self.input_times = input_times
        self.output_times = output_times

        self._experts_compiled = False
        if compile:
            self.compile_experts()

    def default_sources(self) -> DataSource:
        """Recommend MTG FCI imagery on the model grid.

        Combines two :py:class:`~earth2studio.data.MeteosatFCI` sources cropped to
        the model domain: 2 km channels selected onto the model grid, and the
        remaining channels, available only at 1 km, averaged over the 2x2 blocks
        of 1 km pixels that form each 2 km pixel.

        Returns
        -------
        DataSource
            FCI source returning every model variable on the model grid.
        """
        from earth2studio.data import CompositeSource, MeteosatFCI, RegriddedSource

        sources: dict[str, DataSource] = {}
        variables: dict[str, list[str]] = {}
        claimed: set[str] = set()
        resolutions: tuple[Literal["2km", "1km"], ...] = ("2km", "1km")
        for resolution in resolutions:
            regridder = _BlockMeanRegridder(resolution, self.grid)
            source = MeteosatFCI(resolution=resolution, pixel_bbox=regridder.pixel_bbox)
            available = source.available_variables() - claimed
            names = [str(v) for v in self.variables if v in available]
            if names:
                sources[resolution] = RegriddedSource(source, regridder)
                variables[resolution] = names
                claimed.update(names)
        missing = set(map(str, self.variables)) - claimed
        if missing:
            raise ValueError(f"No FCI resolution provides {sorted(missing)}")
        return CompositeSource(sources, variables)

    def input_coords(self) -> CoordinateSystem:
        """Input coordinate system of the prognostic model.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray input signature on the 2 km FCI grid crop.
        """
        # Projecting the grid's coordinates is slow and the grid is fixed.
        if self._signature is None:
            self._signature = coord_array(
                ("batch", "time", "lead_time", "variable", "y", "x"),
                {
                    "lead_time": self.input_times,
                    "variable": np.array(self.variables),
                },
                dynamic=("batch", "time"),
                grid=self.grid,
            )
        return self._signature.copy(deep=False)

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Output coordinate system of the prognostic model.

        Parameters
        ----------
        input_coords : CoordinateSystem
            Input coordinate signature or DataArray to validate and transform.

        Returns
        -------
        CoordinateSystem
            Allocation-free DataArray output signature on the checkpoint pixel
            grid.
        """
        handshake_time(input_coords, allow_dynamic=True)
        handshake_time(input_coords, "lead_time")
        lead = np.asarray(input_coords.lead_time)
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
        return coord_array_like(
            input_coords, {"lead_time": self.output_times + lead[-1]}
        )

    def with_region(self, region: Region, /) -> Self:
        """Return a view of this model covering a geographic box.

        The view runs on the full model domain, because the experts' positional
        embeddings fix their input size, and shares the loaded experts. It records
        the requested region so that :meth:`crop_output` can present forecasts on
        it. Neither this model nor other views change.

        Parameters
        ----------
        region : Region
            Box (:meth:`~earth2studio.grids.Region.box`) inside the model domain
            and on the Earth's disk.

        Returns
        -------
        Self
            View with the same coordinate signatures as this model.

        Raises
        ------
        ValueError
            If the region is not a box, or extends outside the model domain.
        """
        from earth2studio.models.nn.stormscope_flash import RegionInfo

        if region.kind != "box":
            raise ValueError(
                f"{type(self).__name__} supports only box regions, not {region.kind!r}"
            )
        west, south, east, north = region.params
        if not west < east or not south < north:
            raise ValueError("Region box must have positive extent within the domain")
        self._check_covers(west, south, east, north)

        latitude = self.latitudes.cpu().numpy()
        longitude = (self.longitudes.cpu().numpy() + 180) % 360 - 180
        with np.errstate(invalid="ignore"):
            mask = (
                ~self.off_earth_mask.cpu().numpy()
                & (latitude >= south)
                & (latitude <= north)
                & (longitude >= west)
                & (longitude <= east)
            )
        rows, cols = np.nonzero(mask)
        if not rows.size:
            raise ValueError("Requested region contains no model grid points")
        y0, y1 = int(rows.min()), int(rows.max()) + 1
        x0, x1 = int(cols.min()), int(cols.max()) + 1
        height, width = mask.shape
        requested = mask[y0:y1, x0:x1].copy()
        requested.flags.writeable = False

        view = copy(self)
        view._buffers = self._buffers.copy()
        view._modules = self._modules.copy()
        view._parameters = self._parameters.copy()
        view._non_persistent_buffers_set = self._non_persistent_buffers_set.copy()
        view.experts = torch.nn.ModuleList(self.experts)
        view.sampler_args = self.sampler_args.copy()
        # Each view seeds its own noise stream.
        view._rng_seed = None
        view._rng_states = None
        view.region_info = RegionInfo(
            requested_bounds=(south, north, west, east),
            input_bounds=(0, height, 0, width),
            output_bounds=(y0, y1, x0, x1),
            requested_mask=requested,
            padding=(y0, height - y1, x0, width - x1),
        )
        return view

    def _check_covers(
        self, west: float, south: float, east: float, north: float
    ) -> None:
        """Raise unless the box boundary lies on the disk inside the model grid."""
        # Box edges are meridians and parallels, curved in the projection; sample them.
        t = np.linspace(0.0, 1.0, 65)
        longitude = np.concatenate(
            [
                west + (east - west) * t,
                np.full_like(t, east),
                east - (east - west) * t,
                np.full_like(t, west),
            ]
        )
        latitude = np.concatenate(
            [
                np.full_like(t, south),
                south + (north - south) * t,
                np.full_like(t, north),
                north - (north - south) * t,
            ]
        )
        x, y = Transformer.from_crs(
            CRS.from_epsg(4326), self.grid.crs, always_xy=True
        ).transform(longitude, latitude)
        inside = np.isfinite(x) & np.isfinite(y)
        for values, axis in ((x, self.grid.x), (y, self.grid.y)):
            half = 0.5 * abs(axis[1] - axis[0])
            with np.errstate(invalid="ignore"):
                inside &= (values >= axis.min() - half) & (values <= axis.max() + half)
        if not inside.all():
            raise ValueError(
                "Requested region extends outside the model domain or the Earth's disk"
            )

    def crop_output(self, y: xr.DataArray, mask: bool = True) -> xr.DataArray:
        """Crop a forecast of a regional view to its requested box, for presentation.

        Keep stepping with the uncropped forecast. Models without a region return
        forecasts unchanged.

        Parameters
        ----------
        y : xr.DataArray
            Forecast on the model grid, with ``y`` and ``x`` trailing.
        mask : bool, optional
            Set gridpoints outside the requested box, or off the Earth's disk, to
            NaN, by default True.

        Returns
        -------
        xr.DataArray
            Forecast on the smallest rectangle of gridpoints containing the box.
        """
        if self.region_info is None:
            return y
        return crop_to_region(y, self.region_info, mask)

    def compile_experts(self) -> None:
        """Compile each denoising expert in place with ``torch.compile``.

        Compilation is lazy, so this is safe before or after moving the model to a
        device. Repeated calls are a no-op.
        """
        if self._experts_compiled:
            return
        for i in range(len(self.experts)):
            self.experts[i] = torch.compile(self.experts[i], dynamic=False)
        self._experts_compiled = True

    def raw_to_physical(self, x: torch.Tensor, in_place: bool = False) -> torch.Tensor:
        """Convert raw MTG data file values to physical values."""
        if self.ir_38_channel is not None:
            if not in_place:
                x = x.clone()
            x_ir_38 = x[..., self.ir_38_channel, :, :]
            ir_38_mask = x_ir_38 > self.ir_38_warm_threshold
            x_ir_38[ir_38_mask] = (
                x_ir_38[ir_38_mask] - self.ir_38_warm_threshold
            ) * self.ir_38_warm_multiplier + self.ir_38_warm_threshold

        return torch.addcmul(
            self.add_offset, x, self.scale_factor, out=x if in_place else None
        )

    def physical_to_raw(self, x: torch.Tensor, in_place: bool = False) -> torch.Tensor:
        """Convert physical values to raw MTG data file values."""
        x = torch.addcmul(
            self.offset_inv_scale, x, self.inv_scale, out=x if in_place else None
        )

        if self.ir_38_channel is not None:
            x_ir_38 = x[..., self.ir_38_channel, :, :]
            ir_38_mask = x_ir_38 > self.ir_38_warm_threshold
            x_ir_38[ir_38_mask] = (x_ir_38[ir_38_mask] - self.ir_38_warm_threshold) * (
                1.0 / self.ir_38_warm_multiplier
            ) + self.ir_38_warm_threshold

        return x

    def raw_to_normalized(
        self, x: torch.Tensor, in_place: bool = False
    ) -> torch.Tensor:
        """Convert ``x`` from raw values to zero mean, unit variance."""
        return torch.addcmul(
            self.mean_inv_std, x, self.inv_std, out=x if in_place else None
        )

    def normalized_to_raw(
        self, x: torch.Tensor, in_place: bool = False
    ) -> torch.Tensor:
        """Convert ``x`` from zero mean, unit variance to raw values."""
        return torch.addcmul(self.means, x, self.stds, out=x if in_place else None)

    def normalize(self, x: torch.Tensor, in_place: bool = False) -> torch.Tensor:
        """Normalise physical radiances ``x`` to zero mean, unit variance.

        Opposite of ``denormalize``.
        """
        x = self.physical_to_raw(x, in_place=in_place)
        x = self.raw_to_normalized(x, in_place=True)
        x.masked_fill_(self.off_earth_mask, 0)
        return x

    def denormalize(self, x: torch.Tensor, in_place: bool = False) -> torch.Tensor:
        """Denormalise model output ``x`` to physical units.

        The model is trained on raw uint16 values from FCI normalized to zero mean
        and unit variance. To get from model outputs to physical values, we first
        convert the outputs to the scale of the raw data, then convert the raw values
        to physical radiances.
        """
        x = self.normalized_to_raw(x, in_place=in_place)
        x = self.raw_to_physical(x, in_place=True)
        x.masked_fill_(self.off_earth_mask, torch.nan)
        return x

    def _azimuth_zenith(
        self, times: list[np.datetime64], zen_azi: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute solar angles for a list of times.

        Parameters
        ----------
        times : list of np.datetime64
            Sequence of ``T`` times at which to evaluate the angles.
        zen_azi : torch.Tensor, optional
            Pre-allocated output buffer of shape ``(3, T, H, W)``.  If
            ``None`` (default) a new tensor is allocated on the same device
            and dtype as ``self``.

        Returns
        -------
        torch.Tensor
            Shape ``(3, T, H, W)``.  Dimension 0 contains, in order:
            cos-zenith, sin-azimuth, cos-azimuth.  The time-outer memory
            layout means ``view(3*T, H, W)`` gives the angle-type-outer
            ordering expected by the diffusion model.
        """

        if zen_azi is None:
            H, W = self.longitudes.shape
            T = len(times)
            zen_azi = torch.empty(
                3, T, H, W, dtype=self.longitudes.dtype, device=self.longitudes.device
            )

        # Compute solar angles for every time step in the list
        for i, t in enumerate(times):
            t_dt = datetime.fromisoformat(str(t))
            _, zen_azi[0, i], zen_azi[1, i], zen_azi[2, i] = zenith_azimuth_angles(
                t_dt, self.longitudes, self.latitudes
            )

        zen_azi.masked_fill_(self.off_earth_mask, 0)

        return zen_azi

    @torch.no_grad()
    def _forward(
        self,
        x: torch.Tensor,
        zen_azi: torch.Tensor,
    ) -> torch.Tensor:
        """Run one diffusion forward step on a normalised context window.

        Builds the conditioning tensor from solar angles, the flattened
        context history, and invariant fields; samples the diffusion model;
        then advances the context window in-place by dropping the oldest
        frame and appending the new prediction.

        Parameters
        ----------
        x : torch.Tensor
            Normalised context window, shape ``(batch, L, C, H, W)``, where
            ``L = len(self.input_times)`` and frames are ordered oldest first.
            Modified in-place: on return the window is shifted by one step
            (oldest frame dropped, new prediction appended at index ``-1``).
        zen_azi : torch.Tensor
            Solar-angle tensor of shape ``(3, L+1, H, W)`` covering the
            ``L`` context times plus the one target time, as returned by
            :meth:`_azimuth_zenith`.

        Returns
        -------
        torch.Tensor
            Updated context window, shape ``(batch, L, C, H, W)``, normalised.
            The last slice ``[:, -1]`` holds the predicted next frame.
        """
        # reshape x to the order needed by the model
        B, L, C, H, W = x.shape
        x = x.reshape(B, C * L, H, W)

        zen_azi = zen_azi.view(1, 3 * (L + 1), H, W).expand(B, -1, -1, -1)

        invariant_tensor = self.invariants.repeat(B, 1, 1, 1)

        # Channel order matches training background_channels in mtg.py:
        #   [cos_zen × (N+1), sin_azi × (N+1), cos_azi × (N+1),
        #    MTG_history × N (oldest first), invariants]
        condition = torch.cat(
            (zen_azi, x, invariant_tensor),
            dim=1,
        )
        denoiser = self.scheduler.get_denoiser(
            x0_predictor=self._conditioned_x0_predictor(condition)
        )
        return self._sample(
            denoiser, shape=(B, C, H, W), dtype=x.dtype, device=x.device
        )

    def _conditioned_x0_predictor(
        self, condition: torch.Tensor
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Return an x0-predictor closure conditioned on ``condition``."""

        def x0_predictor(x_noisy: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            if t[0] < self.expert_sigma_max[0]:
                expert_idx = 0
            elif t[0] >= self.expert_sigma_min[-1]:
                expert_idx = len(self.experts) - 1
            else:
                expert_idx = (
                    ((t[0] >= self.expert_sigma_min) & (t[0] < self.expert_sigma_max))
                    .to(dtype=torch.int64)
                    .argmax()
                )
            diffusion_model = self.experts[expert_idx]

            with torch.autocast(
                x_noisy.device.type, dtype=torch.bfloat16, enabled=self.amp
            ):
                x0 = diffusion_model(x_noisy, t, condition=condition)
                x0 = torch.clamp(x0, min=self.min_values, max=self.max_values, out=x0)
                return x0

        return x0_predictor

    def _sample(
        self,
        denoiser: Any,
        shape: tuple,
        dtype: torch.dtype,
        device: torch.device,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        """Draw a diffusion sample.

        Parameters
        ----------
        state : torch.Tensor
            Reference tensor; its shape, device, and dtype are used for the initial latents.
        denoiser :
            Configured denoiser (EDM or SDA) returned by ``scheduler.get_denoiser``.
        num_steps : int, optional
            Number of diffusion steps; defaults to ``sampler_args["num_steps"]``.

        Returns
        -------
        torch.Tensor
            Normalised diffusion output, same shape as ``state``.
        """
        if num_steps is None:
            num_steps = int(self.sampler_args["num_steps"])
        # PhysicsNeMo's stochastic Heun solver has no generator argument.
        with fork_rng(self._rng_seed, device, states=self._rng_states):
            latents = self.sampler_args["sigma_max"] * torch.randn(
                shape, dtype=dtype, device=device
            )
            return sample(
                denoiser,
                latents,
                noise_scheduler=self.scheduler,
                num_steps=num_steps,
                solver="edm_stochastic_heun",
                solver_options={
                    "S_churn": self.sampler_args["S_churn"],
                    "S_min": self.sampler_args["S_min"],
                    "S_max": self.sampler_args["S_max"],
                    "S_noise": self.sampler_args["S_noise"],
                },
            )

    stochastic = True
    _rng_seed: int | None = None
    _rng_states: dict[str, torch.Tensor] | None = None

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Seed Meteosat latent sampling and EDM churn noise.

        Parameters
        ----------
        seed : int
            Seed for diffusion sampling.
        reset : bool, optional
            Reset existing sampling state, by default True.
        """
        if reset or self._rng_seed is None:
            self._rng_seed = seed
            self._rng_states = {}

    @torch.no_grad()
    def _predict(
        self,
        x: xr.DataArray,
        native: torch.Tensor | None = None,
    ) -> tuple[xr.DataArray, torch.Tensor]:
        """Run the prognostic model one step forward.

        Parameters
        ----------
        x : xr.DataArray
            Input tensor of shape ``(batch, time, lead_time, variable, y, x)``
            containing ``L = len(self.input_times)`` consecutive raw MTG frames per
            ``(batch, time)`` entry, ordered oldest first along the ``lead_time``
            dimension.

        Returns
        -------
        xr.DataArray
            Predicted next frame as a denormalised tensor of shape
            ``(batch, time, 1, variable, y, x)`` and the corresponding output
            coordinate system.
        """

        self.output_coords(x)
        name = x.name
        encoding = deepcopy(x.encoding)
        x, restore = batch_func()._compress_array(self, x)
        output_coords = self.output_coords(x)
        x, coords = x.e2s.to_torch()
        x = x.to(self.means.device)

        # x: (batch, time, n_input_times, C, H, W)
        B, T, L, C, H, W = x.shape
        x = self.normalize(x) if native is None else native.clone()

        for j, time in enumerate(coords["time"]):
            # all_times: N input times + 1 output time (N+1 entries for solar angles)
            all_times = list(time + coords["lead_time"]) + [
                time + coords["lead_time"][-1] + self.output_times[0]
            ]
            zen_azi = self._azimuth_zenith(all_times)

            for i0 in range(0, B, self.batch_size):
                i1 = i0 + self.batch_size
                prediction = self._forward(x[i0:i1, j], zen_azi)
                prediction.masked_fill_(self.off_earth_mask, 0)
                x[i0:i1, j] = torch.cat((x[i0:i1, j, 1:], prediction[:, None]), dim=1)

        out = from_torch(self.denormalize(x[:, :, -1:]), output_coords, name=name)
        out.attrs = deepcopy(out.attrs)
        out.encoding = encoding
        return restore(out), x

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Compute the first Meteosat imagery forecast without iterator hooks.

        Parameters
        ----------
        x : xr.DataArray
            Imagery history matching ``input_coords()``.

        Returns
        -------
        xr.DataArray
            Forecast imagery matching ``output_coords(x)`` with off-Earth
            pixels masked to zero.
        """
        return self.initialize(x)[0]

    def initialize(self, x: xr.DataArray) -> tuple[xr.DataArray, dict]:
        """Predict the first frame and retain missing context and sampling state.

        Parameters
        ----------
        x : xr.DataArray
            Initial radiance history matching ``input_coords()``.

        Returns
        -------
        tuple[xr.DataArray, dict]
            First forecast and native normalized context, missing history,
            published reference and RNG state. Iterator hooks are not applied.
        """
        handshake_nonempty(x)
        if self._rng_seed is None:
            self.set_rng(int(torch.randint(2**31, ())))
        y, native = self._predict(x)
        return y, {
            "history": x.isel(lead_time=slice(1, None)).copy(deep=True),
            "native": native,
            "published": y.copy(deep=True),
            "seed": self._rng_seed,
            "rng": deepcopy(self._rng_states),
        }

    def step(self, y: xr.DataArray, state: dict) -> tuple[xr.DataArray, dict]:
        """Advance the sliding context using explicit history and RNG state.

        Parameters
        ----------
        y : xr.DataArray
            Previous radiance forecast, including any caller-applied edits.
        state : dict
            Native context, history, published reference and RNG state returned
            by ``initialize`` or ``step``.

        Returns
        -------
        tuple[xr.DataArray, dict]
            Next forecast and updated state, without iterator hooks.
        """
        history = state["history"].e2s.to_torch()[0]
        current = y.e2s.to_torch()[0]
        x = from_torch(
            torch.cat(
                (history.to(current.device), current), dim=y.get_axis_num("lead_time")
            ),
            coord_array_like(
                y, {"lead_time": y.lead_time.values[-1] + self.input_times}
            ),
            name=y.name,
        )
        x.attrs, x.encoding = deepcopy(y.attrs), deepcopy(y.encoding)
        native = state["native"].clone()
        current, _ = batch_func()._compress_array(self, y)
        published, _ = batch_func()._compress_array(self, state["published"])
        current = current.e2s.to_torch()[0].to(native.device)
        published = published.e2s.to_torch()[0].to(native.device)
        unchanged = (current == published) | (
            torch.isnan(current) & torch.isnan(published)
        )
        native[:, :, -1:] = torch.where(
            unchanged, native[:, :, -1:], self.normalize(current)
        )
        previous = self._rng_seed, self._rng_states
        self._rng_seed, self._rng_states = state["seed"], deepcopy(state["rng"])
        try:
            prediction, native = self._predict(x, native)
            return prediction, {
                "history": x.isel(lead_time=slice(1, None)).copy(deep=True),
                "native": native,
                "published": prediction.copy(deep=True),
                "seed": self._rng_seed,
                "rng": deepcopy(self._rng_states),
            }
        finally:
            self._rng_seed, self._rng_states = previous

    def create_iterator(
        self,
        x: xr.DataArray,
    ) -> Generator[xr.DataArray, None, None]:
        """Yield imagery forecasts beginning with the first prediction.

        Parameters
        ----------
        x : xr.DataArray
            Initial imagery history matching ``input_coords()``.

        Yields
        ------
        xr.DataArray
            Forecast DataArray. The rear hook runs before every yield and the
            front hook before each subsequent step.
        """
        yield from self._default_create_iterator(x)

    @classmethod
    def load_default_package(cls) -> Package:
        """Load prognostic package"""
        package = Package(
            "hf://nvidia/stormscope-meteosat@17148d712f7ef2c0b48355032dcad85173dd230e",
            cache_options={
                "cache_storage": Package.default_cache("stormscope-meteosat"),
                "same_names": True,
            },
        )
        return package

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        amp: bool = True,
        compile: bool = False,
    ) -> PrognosticModel:
        """Load the prognostic model from a package.

        Parameters
        ----------
        package : Package
            Model package containing checkpoints and metadata.
        amp : bool, optional
            Use automatic mixed precision (bfloat16) in the experts, by default True
        compile : bool, optional
            Compile each expert with ``torch.compile``, by default False

        Returns
        -------
        PrognosticModel
            Loaded and initialised :class:`StormScopeMeteosatEU` model.
        """
        # load model registry:
        config = OmegaConf.load(package.resolve("config.yaml"))

        # Checkpoints embed layernorm_backend=apex in the model constructors, which will
        # fail if apex is not installed. Here we patch APEX_AVAILABLE in physicsnemo
        # dit_layers to avoid the import error. FusedLayerNorm uses elementwise_affine=False
        # (no learned parameters), so torch.nn.LayerNorm is state-dict-compatible.
        # TODO remove when upstream physicsnemo falls back to torch with a warning.
        _apex_restore: tuple | None = None
        if not _dit_layers.APEX_AVAILABLE:
            _apex_restore = (_dit_layers.APEX_AVAILABLE, _dit_layers.apex_normalization)
            _dit_layers.APEX_AVAILABLE = True
            _dit_layers.apex_normalization = types.SimpleNamespace(
                FusedLayerNorm=torch.nn.LayerNorm
            )

        try:
            model_spec = [
                {
                    "sigma_min": float(spec["sigma_min"]),
                    "sigma_max": float(spec["sigma_max"]),
                    "model": EDMPreconditioner.from_checkpoint(
                        package.resolve(spec["path"])
                    ).eval(),
                }
                for spec in config["model_spec"]
            ]
        finally:
            if _apex_restore is not None:
                _dit_layers.APEX_AVAILABLE, _dit_layers.apex_normalization = (
                    _apex_restore
                )

        # Load metadata: means, stds, grid
        with xr.open_dataset(package.resolve("metadata.nc")) as metadata:
            variables = np.array(metadata["variable_state"], copy=True)

            # Expand dims and tensorify normalization buffers
            means = torch.from_numpy(metadata["means"].values[:, None, None])
            stds = torch.from_numpy(metadata["stds"].values[:, None, None])
            scale_factor = torch.from_numpy(
                metadata["scale_factor"].values[:, None, None]
            )
            add_offset = torch.from_numpy(metadata["add_offset"].values[:, None, None])
            ir_38_warm_scale_factor = metadata.attrs["ir_38_warm_scale_factor"]
            ir_38_warm_threshold = metadata.attrs["ir_38_warm_threshold"]

            # Load invariants
            invariants = metadata["invariants"].values
            invariants = torch.from_numpy(invariants).repeat(1, 1, 1, 1)
            # The package grid is a crop of the 2 km FCI grid labelled by scan angle
            latitudes = torch.from_numpy(metadata["lat"].values)
            longitudes = torch.from_numpy(metadata["lon"].values)
            earth_mask = torch.from_numpy(metadata["earth_mask"].values)
            y_coords = metadata["mtg_y"].values
            x_coords = metadata["mtg_x"].values

        # EDM sampler arguments
        sampler_args = OmegaConf.to_container(config.sampler_args)

        num_input_time_steps = config.get(
            "num_input_time_steps", config.get("num_input_steps", 6)
        )
        step_interval = config.get("step_interval_minutes", 10)
        input_times = np.arange(-num_input_time_steps + 1, 1) * np.timedelta64(
            step_interval, "m"
        )
        output_times = np.array([np.timedelta64(step_interval, "m")])

        return cls(
            model_spec,
            means=means,
            stds=stds,
            invariants=invariants,
            scale_factor=scale_factor,
            add_offset=add_offset,
            ir_38_warm_scale_factor=ir_38_warm_scale_factor,
            ir_38_warm_threshold=ir_38_warm_threshold,
            latitudes=latitudes,
            longitudes=longitudes,
            earth_mask=earth_mask,
            variables=variables,
            sampler_args=sampler_args,
            y_coords=y_coords,
            x_coords=x_coords,
            input_times=input_times,
            output_times=output_times,
            amp=amp,
            compile=compile,
        )
