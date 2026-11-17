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

"""HiROv2 diagnostic downscaler.

HiROv2 is the diffusion downscaler in HiRO-ACEv2. It maps coarse ACE2S fields
on a 1 degree grid to a 1/32 degree grid (about 3 km). The packaged checkpoint
is a mixture of two diffusion experts and is loaded with the ``ace2`` extra.
"""

from __future__ import annotations

import contextlib
import random
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch
import xarray as xr
from loguru import logger

from earth2studio.grids import GridDefinition, LatLonGrid, resolve_grid
from earth2studio.lexicon.ace import ACELexicon
from earth2studio.models.auto import AutoModelMixin, Package
from earth2studio.utils.coords import (
    coord_array,
    handshake_dataarray,
    handshake_nonempty,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import (
    OptionalDependencyFailure,
    check_optional_dependencies,
)
from earth2studio.utils.type import CoordinateSystem

try:
    from fme.core.coordinates import LatLonCoordinates
    from fme.core.distributed import Distributed
    from fme.downscaling.data.static import StaticInput, StaticInputs
    from fme.downscaling.predictors.serial_denoising import DenoisingMoEPredictor
except ImportError:
    OptionalDependencyFailure("ace2")
    LatLonCoordinates = Any
    Distributed = Any
    StaticInput = Any
    StaticInputs = Any
    DenoisingMoEPredictor = Any

_FME_DISTRIBUTED_STACK = contextlib.ExitStack()
_FME_DISTRIBUTED_OPEN = False
_LAT_LIMIT = 88.0
_COORD_TOLERANCE = 1e-4


def _e2s_name(fme_name: str) -> str:
    try:
        return ACELexicon.get_e2s_from_fme(fme_name)
    except KeyError:
        return fme_name


def _tile_starts(length: int, window: int, overlap: int) -> list[int]:
    if window < 1:
        raise ValueError(f"patch size must be positive, got {window}")
    if length < window:
        raise ValueError(
            f"Region length {length} is smaller than the HiROv2 patch of {window}"
        )
    if overlap < 0 or overlap >= window:
        raise ValueError(f"overlap must be in [0, {window}), got {overlap}")
    stride = window - overlap
    starts = list(range(0, length - window + 1, stride))
    last = length - window
    if starts[-1] != last:
        starts.append(last)
    return starts


def _grid_indices(values: np.ndarray, reference: np.ndarray, name: str) -> np.ndarray:
    """Return the reference index of each value, raising if one is missing."""
    matches = np.isclose(
        values[:, None], reference[None, :], atol=_COORD_TOLERANCE, rtol=0.0
    )
    if not np.all(matches.any(axis=1)):
        missing = values[~matches.any(axis=1)]
        raise ValueError(
            f"{name} coordinates {missing.tolist()} are not points of the "
            "reference grid used by the checkpoint"
        )
    return np.argmax(matches, axis=1)


def _fork_global_rng(seed: int) -> contextlib.AbstractContextManager[None]:
    """Seed the global generators fme reads, then restore them."""

    @contextlib.contextmanager
    def _fork() -> Any:
        devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(int(seed))
                if devices:
                    torch.cuda.manual_seed_all(int(seed))
                np.random.seed(int(seed) % (2**32))
                random.seed(int(seed))
                yield
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)

    return _fork()


def _ensure_fme_distributed() -> None:
    """Enter fme's distributed context once per process."""
    global _FME_DISTRIBUTED_OPEN
    if _FME_DISTRIBUTED_OPEN:
        return
    _FME_DISTRIBUTED_STACK.enter_context(Distributed.context())
    _FME_DISTRIBUTED_OPEN = True


def _output_signature(
    x: xr.DataArray,
    grid: GridDefinition,
    variables: Sequence[str],
    number_of_samples: int,
) -> CoordinateSystem:
    """Replace the spatial grid and insert a sample axis before ``variable``."""
    spatial = tuple(x.attrs["dims"])
    removed = set(spatial)
    if not np.array_equal(x.coords["variable"].values, np.asarray(variables)):
        removed.add("variable")
    dims = list(x.dims[: -len(spatial)])
    dims.insert(dims.index("variable"), "sample")
    dims.extend(grid.dims)
    coords = {
        key: deepcopy(value.variable)
        for key, value in x.coords.items()
        if not removed.intersection(value.dims)
    }
    if "variable" not in coords:
        coords["variable"] = np.asarray(variables, dtype=object)
    coords["sample"] = np.arange(number_of_samples)
    dynamic = tuple(
        dim for dim in x.attrs.get("earth2studio_dynamic_dims", ()) if dim in dims
    )
    return coord_array(
        dims,
        coords,
        grid=grid,
        dynamic=dynamic,
        sizes={
            dim: x.sizes[dim]
            for dim in dims
            if dim in x.sizes and dim not in removed and dim not in spatial
        },
        dtype=x.dtype,
        name=x.name,
        attrs=deepcopy(x.attrs),
    )


def _select_extent(
    reference_grid: str,
    lat_extent: tuple[float, float],
    lon_extent: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Select reference-grid points inside a lat/lon box.

    Parameters
    ----------
    reference_grid : str
        Registered grid the coarse model lives on.
    lat_extent : tuple[float, float]
        ``(south, north)`` in degrees.
    lon_extent : tuple[float, float]
        ``(west, east)`` in degrees on ``[0, 360)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Latitudes north-to-south and longitudes increasing.
    """
    south, north = lat_extent
    west, east = lon_extent
    if south > north:
        raise ValueError(f"lat_extent south ({south}) must not exceed north ({north})")
    if west > east:
        raise ValueError(f"lon_extent west ({west}) must not exceed east ({east})")
    grid = resolve_grid(reference_grid)
    lat = np.asarray(grid.coords()["lat"].values)
    lon = np.asarray(grid.coords()["lon"].values)
    # Registered grids may store latitude south-to-north; HiROv2 expects north-to-south.
    selected_lat = np.sort(lat[(lat <= north) & (lat >= south)])[::-1]
    selected_lon = np.sort(lon[(lon >= west) & (lon <= east)])
    if selected_lat.size == 0 or selected_lon.size == 0:
        raise ValueError(
            f"No {reference_grid} points inside lat {lat_extent} and lon {lon_extent}"
        )
    return selected_lat, selected_lon


def _checkpoint_variables(state: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Return runtime input and output names of a bundled HiROv2 checkpoint."""
    expert_config = state["experts"][0]["config"]
    renames = state.get("expert_renames") or [None]
    rename = renames[0] or {}
    in_names = [rename.get(name, name) for name in expert_config["in_names"]]
    out_names = [rename.get(name, name) for name in expert_config["out_names"]]
    return in_names, out_names


def _disable_apex_group_norm(state: dict[str, Any]) -> None:
    """Use torch GroupNorm instead of NVIDIA apex, which is not required."""
    for expert in state["experts"]:
        module_config = expert["config"]["module"]["config"]
        if module_config.get("use_apex_gn", False):
            logger.info("Disabling apex GroupNorm in HiROv2 experts (torch fallback)")
            module_config["use_apex_gn"] = False


@check_optional_dependencies()
class HiROv2(torch.nn.Module, AutoModelMixin):
    """HiROv2 regional downscaler from ~100 km ACE fields to ~3 km.

    The checkpoint is a mixture of two diffusion experts, and each generates one
    training patch at a time. A larger region is split into those patches with
    ``overlap`` coarse cells shared between neighbors, and shared fine cells
    are averaged. Fine coordinates come from the checkpoint's own fine grid
    (1/32 degree). Latitude is limited to 88°S–88°N. The model was trained
    between 66°S and 70°N.

    Parameters
    ----------
    model : DenoisingMoEPredictor
        Loaded fme predictor. It must expose ``coarse_shape``,
        ``downscale_factor``, ``full_fine_coords``, ``static_inputs`` and
        ``generate(coarse, static_inputs, n_samples)``, which returns a dict of
        tensors shaped ``[batch, sample, fine_lat, fine_lon]``, possibly inside
        a tuple.
    lat : numpy.ndarray
        Coarse latitudes, strictly north to south, inside [-88, 88]. Each value
        must be a point of ``reference_grid``.
    lon : numpy.ndarray
        Coarse longitudes, strictly increasing, on [0, 360]. Each value must be
        a point of ``reference_grid``.
    in_names : Sequence[str]
        Runtime input names of the checkpoint, such as ``PRATEsfc``.
    out_names : Sequence[str]
        Runtime output names of the checkpoint.
    reference_grid : str, optional
        Registered coarse grid of the checkpoint, by default ``gaussian-f90``.
    number_of_samples : int, optional
        Ensemble members drawn on every call, by default 1.
    overlap : int, optional
        Coarse cells shared by neighboring patches, by default 1. The published
        HiRO-ACEv2 config uses 1.

    References
    ----------
    - HiRO-ACEv2: https://arxiv.org/abs/2610.17693
    - Hugging Face: https://huggingface.co/allenai/HiRO-ACEv2

    Badges
    ------
    region:global class:downscaling class:climate product:wind product:precip
    product:atmos year:2026 provider:ai2 backend:pytorch
    """

    stochastic: bool = True

    def __init__(
        self,
        model: Any,
        lat: np.ndarray,
        lon: np.ndarray,
        *,
        in_names: Sequence[str],
        out_names: Sequence[str],
        reference_grid: str = "gaussian-f90",
        number_of_samples: int = 1,
        overlap: int = 1,
    ) -> None:
        super().__init__()
        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        if lat.ndim != 1 or lon.ndim != 1:
            raise ValueError("lat and lon must be one-dimensional")
        if np.any(np.diff(lat) >= 0):
            raise ValueError("latitude must be strictly north-to-south")
        if np.any(np.diff(lon) <= 0):
            raise ValueError("longitude must be strictly increasing")
        if np.any(lat < -_LAT_LIMIT) or np.any(lat > _LAT_LIMIT):
            raise ValueError("HiROv2 cannot run outside 88S-88N")
        if np.any(lon < 0) or np.any(lon > 360):
            raise ValueError("longitude must be inside [0, 360]")
        if not isinstance(number_of_samples, int) or number_of_samples < 1:
            raise ValueError("number_of_samples must be a positive integer")

        coarse_shape = tuple(int(size) for size in model.coarse_shape)
        if len(coarse_shape) != 2:
            raise ValueError(f"coarse_shape must be (lat, lon), got {coarse_shape}")
        factor = int(model.downscale_factor)
        if factor < 1:
            raise ValueError(f"downscale factor must be positive, got {factor}")
        self._patch = coarse_shape
        self._overlap = int(overlap)
        _tile_starts(lat.size, coarse_shape[0], self._overlap)
        _tile_starts(lon.size, coarse_shape[1], self._overlap)

        # Map the region onto the reference grid. Work in ascending index order,
        # which is the order fme uses internally.
        reference = resolve_grid(reference_grid).coords()
        ref_lat = np.asarray(reference["lat"].values, dtype=np.float64)
        ref_lon = np.asarray(reference["lon"].values, dtype=np.float64)
        if ref_lat[0] > ref_lat[-1]:
            raise ValueError(f"{reference_grid} must store latitude south-to-north")
        full_fine_lat = model.full_fine_coords.lat
        full_fine_lon = model.full_fine_coords.lon
        if ref_lat.size * factor != int(
            full_fine_lat.shape[0]
        ) or ref_lon.size * factor != int(full_fine_lon.shape[0]):
            raise ValueError(
                f"{reference_grid} does not match the checkpoint's fine grid "
                f"with downscale factor {factor}"
            )
        lat_index = np.sort(_grid_indices(lat, ref_lat, "latitude"))
        lon_index = np.sort(_grid_indices(lon, ref_lon, "longitude"))
        self._lat_index = lat_index
        self._lon_index = lon_index
        block = np.arange(factor)
        fine_lat_index = (lat_index[:, None] * factor + block).reshape(-1)
        fine_lon_index = (lon_index[:, None] * factor + block).reshape(-1)
        self._fine_lat_index = torch.as_tensor(
            fine_lat_index, dtype=torch.long, device=full_fine_lat.device
        )
        self._fine_lon_index = torch.as_tensor(
            fine_lon_index, dtype=torch.long, device=full_fine_lon.device
        )
        self._fine_lat_coords = full_fine_lat[self._fine_lat_index]
        self._fine_lon_coords = full_fine_lon[self._fine_lon_index]

        self.model = model
        self.number_of_samples = number_of_samples
        self._factor = factor
        self._in_fme = list(in_names)
        self._out_fme = list(out_names)
        self._in_e2s = [_e2s_name(name) for name in self._in_fme]
        self._out_e2s = [_e2s_name(name) for name in self._out_fme]
        self.grid = LatLonGrid(lat, lon)
        fine_lat_public = self._fine_lat_coords.detach().double().cpu().numpy()[::-1]
        fine_lon_public = self._fine_lon_coords.detach().double().cpu().numpy()
        self.fine_grid = LatLonGrid(fine_lat_public, fine_lon_public)
        self._rng: torch.Generator | None = None

        static = getattr(model, "static_inputs", None)
        self._static_fields: list[torch.Tensor] | None = None
        if static is not None and len(static.fields) > 0:
            self._static_fields = [field.data for field in static.fields]

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Seed the stream used by later calls.

        Parameters
        ----------
        seed : int
            Seed for subsequent :meth:`__call__` draws.
        reset : bool, optional
            When True, replace any existing stream. When False, seed only if
            no stream exists yet, by default True.
        """
        if reset or self._rng is None:
            self._rng = torch.Generator()
            self._rng.manual_seed(int(seed))

    def input_coords(self) -> CoordinateSystem:
        """Declare the coarse regional input.

        Returns
        -------
        CoordinateSystem
            Allocation-free signature for the checkpoint's coarse variables on
            the configured regional grid. ``batch`` and ``time`` are dynamic.
        """
        return coord_array(
            ("batch", "time", "variable", "lat", "lon"),
            {"variable": np.asarray(self._in_e2s, dtype=object)},
            dynamic=("batch", "time"),
            grid=self.grid,
        )

    def output_coords(self, x: CoordinateSystem) -> CoordinateSystem:
        """Plan the fine ensemble coordinates.

        Parameters
        ----------
        x : CoordinateSystem
            Coarse input signature or DataArray to validate.

        Returns
        -------
        CoordinateSystem
            Allocation-free signature on the refined grid, with a ``sample``
            axis before ``variable``.
        """
        handshake_dataarray(x, self.input_coords())
        return _output_signature(
            x, self.fine_grid, self._out_e2s, self.number_of_samples
        )

    def default_sources(self) -> None:
        """No recommended source.

        Returns
        -------
        None
            Coarse fields come from ACE2S-X-SHiELD, or from a caller-supplied
            array on this region's grid.
        """
        return None

    def _call_seed(self) -> int:
        if self._rng is None:
            return int(torch.randint(0, 2**31 - 1, (1,)).item())
        return int(torch.randint(0, 2**31 - 1, (1,), generator=self._rng).item())

    def _static_window(self, fine_lat: slice, fine_lon: slice) -> StaticInputs | None:
        """Topography for one patch, sliced from the checkpoint's global fields."""
        if self._static_fields is None:
            return None
        rows = self._fine_lat_index[fine_lat]
        cols = self._fine_lon_index[fine_lon]
        fields = [
            StaticInput(data=field[rows][:, cols]) for field in self._static_fields
        ]
        return StaticInputs(
            fields=fields,
            coords=LatLonCoordinates(
                lat=self._fine_lat_coords[fine_lat],
                lon=self._fine_lon_coords[fine_lon],
            ),
        )

    def _generate_window(
        self,
        coarse: dict[str, torch.Tensor],
        static: StaticInputs | None,
    ) -> dict[str, torch.Tensor]:
        generated = self.model.generate(
            coarse, static, n_samples=self.number_of_samples
        )
        if isinstance(generated, tuple):
            generated = generated[0]
        fields: dict[str, torch.Tensor] = {}
        for name, value in generated.items():
            if value.ndim == 3:
                value = value.unsqueeze(1)
            fields[name] = value
        return fields

    def _downscale(self, flat: torch.Tensor) -> torch.Tensor:
        """Downscale a batch of coarse frames.

        Parameters
        ----------
        flat : torch.Tensor
            Coarse tensor shaped ``[batch, variable, lat, lon]`` in
            ``self._in_e2s`` order and north-to-south latitude.

        Returns
        -------
        torch.Tensor
            Fine tensor shaped ``[batch, sample, variable, lat, lon]`` in
            north-to-south latitude.
        """
        # The model runs on fme's device and works in ascending latitude.
        device = self._fine_lat_coords.device
        flat = flat.to(device=device).flip(-2)
        n_batch, _, n_lat, n_lon = flat.shape
        factor = self._factor
        fine_h = n_lat * factor
        fine_w = n_lon * factor
        accumulator = torch.zeros(
            (n_batch, self.number_of_samples, len(self._out_fme), fine_h, fine_w),
            dtype=torch.float32,
            device=device,
        )
        weight = torch.zeros((fine_h, fine_w), dtype=torch.float32, device=device)
        patch_h, patch_w = self._patch
        variable_index = {name: index for index, name in enumerate(self._in_e2s)}
        for i0 in _tile_starts(n_lat, patch_h, self._overlap):
            i1 = i0 + patch_h
            for j0 in _tile_starts(n_lon, patch_w, self._overlap):
                j1 = j0 + patch_w
                coarse = {
                    fme_name: flat[
                        :, variable_index[e2s_name], i0:i1, j0:j1
                    ].contiguous()
                    for fme_name, e2s_name in zip(self._in_fme, self._in_e2s)
                }
                static = self._static_window(
                    slice(i0 * factor, i1 * factor),
                    slice(j0 * factor, j1 * factor),
                )
                fields = self._generate_window(coarse, static)
                fi0, fi1 = i0 * factor, i1 * factor
                fj0, fj1 = j0 * factor, j1 * factor
                for channel, fme_name in enumerate(self._out_fme):
                    tile = fields[fme_name].to(device=device, dtype=torch.float32)
                    accumulator[:, :, channel, fi0:fi1, fj0:fj1] += tile
                weight[fi0:fi1, fj0:fj1] += 1
        return (accumulator / weight.clamp_min(1)).flip(-2)

    @torch.inference_mode()
    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Downscale coarse ACE fields to a fine ensemble.

        Parameters
        ----------
        x : xr.DataArray
            Coarse fields matching :meth:`input_coords`. ``tp`` is the 6-hour
            mean precipitation rate, ``u10m`` and ``v10m`` are 10 m winds, and
            ``msl`` is sea-level pressure, when those are the checkpoint inputs.

        Returns
        -------
        xr.DataArray
            Fine fields on the refined grid. ``sample`` indexes the ensemble.
            Precipitation is the 6-hour mean rate. Winds and sea-level pressure
            are instantaneous. The input is not modified.
        """
        handshake_nonempty(x)
        signature = self.output_coords(x)
        name, encoding = x.name, deepcopy(x.encoding)
        tensor, _ = x.e2s.to_torch()
        order = list(x.dims)
        leading = [dim for dim in order if dim not in ("variable", "lat", "lon")]
        permutation = [order.index(dim) for dim in (*leading, "variable", "lat", "lon")]
        tensor = tensor.permute(*permutation).contiguous()
        lead_shape = tensor.shape[: len(leading)]
        flat = tensor.reshape(-1, *tensor.shape[len(leading) :])
        seed = self._call_seed()
        with _fork_global_rng(seed):
            fine = self._downscale(flat)
        fine = fine.reshape(
            *lead_shape,
            self.number_of_samples,
            len(self._out_fme),
            self.fine_grid.latitude.size,
            self.fine_grid.longitude.size,
        )
        result = from_torch(fine, signature, name=name)
        result.attrs = deepcopy(result.attrs)
        result.encoding = encoding
        return result

    @classmethod
    def load_default_package(cls) -> Package:
        """Load the HiRO-ACEv2 package.

        Returns
        -------
        Package
            Hugging Face package ``allenai/HiRO-ACEv2``.
        """
        return Package(
            "hf://allenai/HiRO-ACEv2",
            cache_options={
                "cache_storage": Package.default_cache("hiro_acev2"),
                "same_names": True,
            },
        )

    @classmethod
    @check_optional_dependencies()
    def load_model(
        cls,
        package: Package,
        lat: np.ndarray | None = None,
        lon: np.ndarray | None = None,
        lat_extent: tuple[float, float] | None = None,
        lon_extent: tuple[float, float] | None = None,
        reference_grid: str = "gaussian-f90",
        number_of_samples: int = 1,
        overlap: int = 1,
    ) -> HiROv2:
        """Load HiROv2 and configure the coarse region.

        Parameters
        ----------
        package : Package
            Package that contains ``HiROv2.ckpt``.
        lat : numpy.ndarray, optional
            Coarse latitudes. Required together with ``lon`` when extents are
            omitted.
        lon : numpy.ndarray, optional
            Coarse longitudes.
        lat_extent : tuple[float, float], optional
            ``(south, north)`` used to slice ``reference_grid`` when ``lat`` is
            omitted.
        lon_extent : tuple[float, float], optional
            ``(west, east)`` on ``[0, 360)``.
        reference_grid : str, optional
            Registered grid sliced by the extents, by default ``gaussian-f90``.
        number_of_samples : int, optional
            Ensemble members per call, by default 1.
        overlap : int, optional
            Coarse-cell overlap between patches, by default 1.

        Returns
        -------
        HiROv2
            Loaded downscaler on fme's current device.
        """
        if (lat is None) != (lon is None):
            raise ValueError("lat and lon must be passed together")
        if lat is None:
            if lat_extent is None or lon_extent is None:
                raise ValueError("Pass lat and lon, or lat_extent and lon_extent")
            lat, lon = _select_extent(reference_grid, lat_extent, lon_extent)
        _ensure_fme_distributed()
        checkpoint_path = package.resolve("HiROv2.ckpt")
        # The bundle is a trusted allenai checkpoint, read the same way fme does.
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        in_names, out_names = _checkpoint_variables(state)
        _disable_apex_group_norm(state)
        predictor = DenoisingMoEPredictor.from_state(state)
        for module in predictor.modules:
            module.eval()
        return cls(
            predictor,
            np.asarray(lat),
            np.asarray(lon),
            in_names=in_names,
            out_names=out_names,
            reference_grid=reference_grid,
            number_of_samples=number_of_samples,
            overlap=overlap,
        )
