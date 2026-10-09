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

"""Field mapping and output-layout helpers shared by workflows and runners."""

from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
import xarray as xr

from earth2studio.io.utils import output_template
from earth2studio.models.px import PrognosticModel
from earth2studio.utils.coords import CoordSystem, coord_array_like


def _dimension_coords(x: xr.DataArray) -> CoordSystem:
    """Read dimension labels without materializing a coordinate signature."""
    return OrderedDict((dim, x.coords[dim].values) for dim in x.dims)


def _map_field(x: xr.DataArray, target: xr.DataArray | CoordSystem) -> xr.DataArray:
    """Select model variables/domain, retaining runtime times and auxiliary geometry.

    Exact contiguous selections are views; nearest numeric selections preserve the
    legacy runner's mapping behavior. Curvilinear auxiliaries are never indexed as
    dimensions. Model validation checks their geometry at the inference boundary.
    """
    coordinates = (
        _dimension_coords(target) if isinstance(target, xr.DataArray) else target
    )
    for dim, values in coordinates.items():
        if dim in ("batch", "time", "lead_time") or len(values) == 0:
            continue
        source = x.coords[dim].values
        if np.array_equal(source, values):
            continue
        if source.ndim != 1 or np.asarray(values).ndim != 1:
            raise ValueError(f"Cannot map multidimensional coordinate {dim}")
        index = x.get_index(dim).get_indexer(values)
        if (index < 0).any():
            if not np.issubdtype(source.dtype, np.number):
                raise ValueError(f"Missing labels for coordinate {dim}: {values}")
            index = np.abs(source[:, None] - values[None, :]).argmin(axis=0)
        selection = (
            slice(int(index[0]), int(index[-1]) + 1)
            if np.all(np.diff(index) == 1)
            else index
        )
        x = x.isel({dim: selection}).assign_coords({dim: values})
        if dim == "variable":
            statistics = coord_array_like(x).attrs.get("earth2studio_statistics")
            x.attrs = dict(x.attrs)
            x.attrs.pop("earth2studio_statistics", None)
            if statistics:
                x.attrs["earth2studio_statistics"] = statistics
    if isinstance(target, xr.DataArray) and "dims" in target.attrs:
        spatial_dims = set(target.attrs["dims"])
        for name, coordinate in target.coords.items():
            if not spatial_dims.intersection(coordinate.dims):
                continue
            if (
                name not in x.coords
                or x.coords[name].dims != coordinate.dims
                or not np.array_equal(x.coords[name], coordinate)
            ):
                raise ValueError(
                    f"Source geometry does not match target coordinate {name}"
                )
        actual_crs = x.attrs.get("earth2studio_crs")
        if actual_crs is not None and actual_crs != target.attrs.get(
            "earth2studio_crs"
        ):
            raise ValueError("Source CRS does not match target CRS")
        x = x.copy(deep=False)
        for key in (
            "type",
            "dims",
            "shape",
            "topology",
            "crs",
            "earth2studio_crs",
            "earth2studio_grid_id",
            "level",
            "nside",
            "ordering",
            "layout",
            "origin",
            "clockwise",
        ):
            x.attrs.pop(key, None)
            if key in target.attrs:
                x.attrs[key] = target.attrs[key]
    return x


def _output_dimensions(
    prognostic: PrognosticModel, time: np.ndarray, nsteps: int
) -> CoordSystem:
    """Plan the legacy IO dimensions from a native model declaration."""
    signature = prognostic.output_coords(prognostic.input_coords())
    coords = OrderedDict(
        (dim, values)
        for dim, values in _dimension_coords(signature).items()
        if signature.sizes[dim]
    )
    coords["time"] = time
    coords["lead_time"] = _lead_times(signature, nsteps)
    coords.move_to_end("lead_time", last=False)
    coords.move_to_end("time", last=False)
    return coords


def _lead_times(signature: xr.DataArray, nsteps: int) -> np.ndarray:
    """Return a rollout's full lead-time extent from one step's declaration."""
    leads = signature.coords["lead_time"].values
    return np.concatenate(
        [
            np.zeros(1, dtype=leads.dtype),
            *(leads + leads[-1] * i for i in range(nsteps)),
        ]
    )


def _io_template(
    signature: xr.DataArray,
    nsteps: int,
    leading: Mapping[str, np.ndarray],
    output_coords: CoordSystem,
    lead_time: np.ndarray | None = None,
) -> xr.DataArray:
    """Plan an IO template for a rollout from one step's output declaration.

    ``output_coords`` restricts the stored domain and variables the way the
    workflows have always allowed: spatial and variable selections are applied to
    the signature, since :func:`output_template` only replaces non-spatial labels.
    The full rollout lead-time extent comes from ``_lead_times`` unless ``lead_time``
    is supplied (e.g. the diagnostic workflow stores the prognostic rollout extent).
    """
    signature = _map_field(signature, output_coords)
    leads = np.asarray(
        output_coords.get(
            "lead_time",
            lead_time if lead_time is not None else _lead_times(signature, nsteps),
        )
    )
    leading = {
        dim: np.asarray(output_coords.get(dim, values))
        for dim, values in leading.items()
    }
    # output_template rejects spatial replacements; _map_field already applied them.
    # Only lead_time is a fixed-dim replacement here; time/ensemble are leading dims.
    return output_template(signature, leading=leading, coords={"lead_time": leads})
