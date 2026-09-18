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

from typing import Any

import numpy as np
import xarray as xr
from pyproj import CRS, Transformer

from earth2studio.grids import (
    E2S_CRS,
    E2S_GRID_ID,
    GridDefinition,
    infer_grid,
    resolve_grid,
)
from earth2studio.utils.coords import coord_array


def _standard_grid(array: xr.DataArray) -> tuple[xr.DataArray, dict[str, str]]:
    spatial_dims = array.attrs.get("dims", ())
    topology = array.attrs.get("topology", "")
    native = {
        "projected": ("y", "x"),
        "curvilinear": ("y", "x"),
        "rectilinear": ("lat", "lon"),
        "points": ("x",),
    }.get(topology)
    renamed = (
        dict(zip(spatial_dims, native, strict=True))
        if native and len(spatial_dims) == len(native)
        else {}
    )
    renamed = {
        key: value
        for key, value in renamed.items()
        if key != value and key in array.dims
    }
    return array.rename(renamed), renamed


def _grid_metadata(
    array: xr.DataArray, definition: GridDefinition, grid_id: str | None = None
) -> None:
    for key in (
        E2S_GRID_ID,
        E2S_CRS,
        "type",
        "dims",
        "shape",
        "topology",
        "crs",
        "level",
        "ordering",
        "layout",
        "origin",
    ):
        array.attrs.pop(key, None)
    array.attrs.update(definition.attrs)
    array.attrs["shape"] = [array.sizes[dim] for dim in definition.dims]
    if definition.crs is not None:
        array.attrs[E2S_CRS] = definition.crs.to_string()
    if grid_id is not None:
        array.attrs[E2S_GRID_ID] = grid_id


def _validated_grid(array: xr.DataArray) -> GridDefinition:
    if E2S_GRID_ID in array.attrs:
        registered = resolve_grid(array.attrs[E2S_GRID_ID])
        coords = registered.coords()
        for dim in registered.dims:
            if (
                dim not in array.coords
                or not np.isin(array.coords[dim], coords[dim]).all()
            ):
                raise ValueError(
                    f"Coordinates disagree with registered grid on '{dim}'"
                )
        expected = xr.Dataset(coords=coords).sel(
            {dim: array.coords[dim] for dim in registered.dims}
        )
        for name in ("lat", "lon"):
            if (
                name in array.coords
                and name in expected.coords
                and not np.array_equal(array.coords[name], expected.coords[name])
            ):
                raise ValueError(
                    f"Coordinates disagree with registered grid on '{name}'"
                )
    definition = infer_grid(array)
    declared_crs = array.attrs.get(E2S_CRS)
    if (
        declared_crs is not None
        and definition.crs is not None
        and CRS.from_user_input(declared_crs) != definition.crs
    ):
        raise ValueError("CRS metadata disagrees with grid geometry")
    return definition


def _map_grid(array: xr.DataArray, target: xr.DataArray, method: str) -> xr.DataArray:
    array, _ = _standard_grid(array)
    target, renamed = _standard_grid(target)
    source_grid = _validated_grid(array)
    target_grid = _validated_grid(target)
    target_coords = target_grid.coords(
        {dim: np.asarray(target.coords[dim]) for dim in target_grid.dims}
    )
    # Exact label selection avoids interpolation and preserves values and dtype.
    selected = None
    if (
        source_grid.dims == target_grid.dims
        and source_grid.topology == target_grid.topology
        and source_grid.crs == target_grid.crs
    ):
        indexers = {
            dim: array.get_index(dim).get_indexer(target_coords[dim].values)
            for dim in source_grid.dims
        }
        if all(np.all(index >= 0) for index in indexers.values()):
            candidate = array.isel(indexers)
            geometry_matches = all(
                name not in target_coords
                or (
                    name in candidate.coords
                    and np.array_equal(candidate.coords[name], target_coords[name])
                )
                for name in ("lat", "lon")
            )
            if geometry_matches:
                selected = candidate
    if selected is None:
        if source_grid.topology not in {"rectilinear", "projected"}:
            raise NotImplementedError(
                f"Interpolation from {source_grid.topology} grids is not supported; use exact native-grid selection"
            )
        if source_grid.topology == "rectilinear":
            indexers = {"lat": target_coords["lat"], "lon": target_coords["lon"]}
        else:
            lat, lon = xr.broadcast(target_coords["lat"], target_coords["lon"])
            x, y = Transformer.from_crs(
                "EPSG:4326", source_grid.crs, always_xy=True
            ).transform(lon.values, lat.values)
            indexers = {
                "y": xr.DataArray(y, dims=lat.dims),
                "x": xr.DataArray(x, dims=lat.dims),
            }
        # Remove source spatial auxiliaries; target geometry supplies replacements.
        auxiliary = [
            name
            for name, coord in array.coords.items()
            if name not in array.dims and set(coord.dims).intersection(source_grid.dims)
        ]
        selected = array.drop_vars(auxiliary).interp(indexers, method=method)  # type: ignore[arg-type]
        obsolete = [
            name
            for name in selected.coords
            if name in source_grid.dims and name not in target_grid.dims
        ]
        selected = selected.drop_vars(obsolete)
    selected = selected.assign_coords(target_coords)
    _grid_metadata(selected, target_grid, target.attrs.get(E2S_GRID_ID))
    if renamed:
        selected = selected.rename({value: key for key, value in renamed.items()})
        selected.attrs["dims"] = [
            next((key for key, value in renamed.items() if value == dim), dim)
            for dim in target_grid.dims
        ]
    return selected


def map_fetch_grid(
    array: xr.DataArray,
    target: xr.DataArray | GridDefinition | str | None,
    method: str,
    bounds: tuple[float, float, float, float] | None,
    bounds_crs: Any | None,
) -> xr.DataArray:
    """Map fetched fields to target grid coordinates, then subset by bounds.

    Parameters
    ----------
    array : xr.DataArray
        Source fields.
    target : xr.DataArray | GridDefinition | str | None
        Target signature or grid; None retains the native grid.
    method : str
        Xarray interpolation method.
    bounds : tuple[float, float, float, float] | None
        Spatial subset bounds.
    bounds_crs : Any | None
        CRS for the bounds, defaulting to geographic longitude/latitude.

    Returns
    -------
    xr.DataArray
        Spatially mapped field array.
    """
    if isinstance(target, str) or isinstance(target, GridDefinition):
        definition = resolve_grid(target) if isinstance(target, str) else target
        target = coord_array(definition.dims, grid=target)
    if target is not None:
        if not isinstance(target, xr.DataArray):
            raise TypeError(
                "interp_to must be a coordinate DataArray, grid definition or registered grid name"
            )
        standard, _ = _standard_grid(target)
        # Nonspatial signatures are still checked by the caller's handshake.
        if (
            E2S_GRID_ID in target.attrs
            or "lat" in standard.coords
            or {"y", "x"}.issubset(standard.coords)
        ):
            array = _map_grid(array, target, method)
    if bounds is not None or bounds_crs is not None:
        standard, renamed = _standard_grid(array)
        definition = infer_grid(standard)
        standard = standard.isel(
            definition.subset_indexers(
                standard.coords, bounds=bounds, bounds_crs=bounds_crs
            )
        )
        _grid_metadata(standard, definition, array.attrs.get(E2S_GRID_ID))
        array = standard.rename({value: key for key, value in renamed.items()})
        array.attrs["dims"] = [
            next((key for key, value in renamed.items() if value == dim), dim)
            for dim in definition.dims
        ]
    return array
