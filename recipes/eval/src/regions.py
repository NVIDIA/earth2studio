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

"""Named regional splits, shared by both scoring pathways.

The ``scoring.regions`` config block maps region names to rectangular
boxes on the scored grid.  A box maps spatial dimension names to
``[min, max]`` ranges in that dimension's coordinate values — ``lat``/
``lon`` degrees on a global grid, or projection/index coordinates on a
limited-area grid.  Dimensions left out of a box cover their full
extent.  A region may also be ``null`` (the whole grid) or a LIST of
boxes scored as their union (e.g. the extra-tropics as both
``|lat| >= 20`` bands).

Two special cases apply to geographic dimension names:

* ``lon`` compares on the [0, 360) circle — negative bounds mean degrees
  west, a box whose normalized min exceeds its max wraps across the
  dateline, and a span of 360 degrees or more means every longitude.
* ``lat`` bounds must lie in [-90, 90].
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from earth2studio.statistics.weights import lat_weight
from earth2studio.utils.type import CoordSystem

# Dimensions that never count as spatial when scanning a grid's
# coordinate system for its spatial axes.
NON_SPATIAL = frozenset({"batch", "time", "lead_time", "variable", "ensemble"})


def spatial_dims(spatial_coords: CoordSystem) -> list[str]:
    """Return the spatial dimension names of a coordinate system."""
    return [d for d in spatial_coords if d not in NON_SPATIAL]


def parse_regions(value: Any) -> dict[str, list[dict] | None] | None:
    """Check and normalize the ``scoring.regions`` block.

    Each region is ``null`` (whole grid), one box, or a LIST of boxes
    whose union defines the region.  A box maps spatial dimension names
    to ``[min, max]`` coordinate ranges; dimensions left out cover their
    full extent.  The parser turns single boxes into one-element lists so
    the mask builder handles one shape.  :func:`region_masks` checks box
    dimensions against the scored grid later, once the grid exists.
    """
    if value is None:
        return None
    if isinstance(value, DictConfig):
        value = OmegaConf.to_container(value, resolve=True)
    if not isinstance(value, dict) or not value:
        raise ValueError(
            "scoring.regions must be a non-empty mapping of "
            "name -> null | {<dim>: [min, max], ...} | list of such boxes "
            "(their union)."
        )

    def _one_box(name: str, spec: Any) -> dict:
        if not isinstance(spec, dict) or not spec:
            raise ValueError(
                f"Region '{name}' boxes must map spatial dimension names to "
                f"[min, max] ranges; got {spec!r}."
            )
        box: dict[str, list[float]] = {}
        for key, bounds_in in spec.items():
            try:
                bounds = [float(b) for b in bounds_in]
            except (TypeError, ValueError) as err:
                raise ValueError(
                    f"Region '{name}': '{key}' must be [min, max]; "
                    f"got {bounds_in!r}."
                ) from err
            if len(bounds) != 2:
                raise ValueError(
                    f"Region '{name}': '{key}' must be [min, max]; "
                    f"got {bounds_in!r}."
                )
            box[str(key)] = bounds
        if "lat" in box:
            lat_lo, lat_hi = box["lat"]
            if not (-90.0 <= lat_lo < lat_hi <= 90.0):
                raise ValueError(
                    f"Region '{name}': lat bounds must satisfy "
                    f"-90 <= min < max <= 90; got {box['lat']}."
                )
        for key, (lo, hi) in box.items():
            # lon may wrap (min > max means crossing the dateline); every
            # other dimension is an ordinary interval.
            if key != "lon" and lo >= hi:
                raise ValueError(
                    f"Region '{name}': '{key}' must satisfy min < max; "
                    f"got [{lo}, {hi}]."
                )
        return box

    out: dict[str, list[dict] | None] = {}
    for name, spec in value.items():
        if spec is None:
            out[str(name)] = None
        elif isinstance(spec, list):
            if not spec:
                raise ValueError(f"Region '{name}': box list must be non-empty.")
            out[str(name)] = [_one_box(str(name), b) for b in spec]
        else:
            out[str(name)] = [_one_box(str(name), spec)]
    return out


def region_masks(
    spatial_coords: CoordSystem,
    regions: dict[str, list[dict] | None],
) -> "OrderedDict[str, torch.Tensor]":
    """Compute each region's {0, 1} mask on the scored grid.

    Parameters
    ----------
    spatial_coords : CoordSystem
        Spatial coordinate arrays of the scored grid (1D per dimension).
    regions : dict[str, list[dict] | None]
        Parsed ``scoring.regions`` (see :func:`parse_regions`).

    Returns
    -------
    OrderedDict[str, torch.Tensor]
        Float64 mask of the full spatial shape per region, in config
        order.

    Raises
    ------
    ValueError
        If a box names a dimension the grid does not have, or a region
        selects no gridpoints.
    """
    dims = spatial_dims(spatial_coords)
    full_shape = [len(np.asarray(spatial_coords[d])) for d in dims]
    axes = {
        d: torch.tensor(np.asarray(spatial_coords[d]), dtype=torch.float64)
        for d in dims
    }

    def _box_mask(name: str, spec: dict) -> torch.Tensor:
        unknown = sorted(set(spec) - set(dims))
        if unknown:
            raise ValueError(
                f"Region '{name}' uses dimensions {unknown} that are not "
                f"spatial dimensions of the scored grid; got {dims}."
            )
        mask = torch.ones(full_shape, dtype=torch.float64)
        for key, (lo, hi) in spec.items():
            vals = axes[key]
            if key == "lon":
                # Longitudes compare on [0, 360); a box whose normalized
                # min exceeds its max wraps across the dateline/meridian,
                # and a span of >= 360 degrees means every longitude (a
                # [0, 360] bound must not normalize into an empty span).
                vals_n = vals % 360.0
                if hi - lo >= 360.0:
                    axis_mask = torch.ones_like(vals_n, dtype=torch.bool)
                else:
                    lon_lo, lon_hi = lo % 360.0, hi % 360.0
                    if lon_lo <= lon_hi:
                        axis_mask = (vals_n >= lon_lo) & (vals_n <= lon_hi)
                    else:
                        axis_mask = (vals_n >= lon_lo) | (vals_n <= lon_hi)
            else:
                axis_mask = (vals >= lo) & (vals <= hi)
            view = [1] * len(dims)
            view[dims.index(key)] = -1
            mask = mask * axis_mask.double().reshape(view)
        return mask

    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for name, spec in regions.items():
        if spec is None:
            mask = torch.ones(full_shape, dtype=torch.float64)
        else:
            # A region is the union of its boxes.  parse_regions
            # normalizes single boxes to one-element lists; accept a bare
            # box here too so direct callers keep working.
            boxes = spec if isinstance(spec, (list, tuple)) else [spec]
            mask = torch.zeros(full_shape, dtype=torch.float64)
            for box in boxes:
                mask = torch.maximum(mask, _box_mask(name, box))
        if spec is not None and not mask.any():
            raise ValueError(
                f"Region '{name}' selects no gridpoints on the scored "
                "grid — check its boxes."
            )
        out[name] = mask
    return out


def build_spatial_weights(
    spatial_coords: CoordSystem,
    lat_weights: bool,
    regions: dict[str, list[dict] | None] | None = None,
) -> torch.Tensor:
    """Build the spatial weight tensor for online reductions.

    Mirrors the offline scorer's weighting: cosine-latitude weights when
    ``scoring.lat_weights`` is true and the grid has a ``lat`` dimension,
    uniform weights otherwise.  Region-free, the returned tensor has one
    axis per spatial dimension so it broadcasts against
    ``[..., <spatial...>]`` tensors.  With ``regions`` configured the
    tensor becomes ``[region, <spatial...>]``: the same weights multiplied
    by each region's {0, 1} mask, evaluated on the actual scored grid so
    box edges land exactly on gridpoints.

    Parameters
    ----------
    spatial_coords : CoordSystem
        Spatial coordinate arrays of the scored grid.
    lat_weights : bool
        Whether to apply cosine-latitude weighting.
    regions : dict[str, list[dict] | None] | None
        Parsed ``scoring.regions`` (see :func:`parse_regions`).

    Returns
    -------
    torch.Tensor
        Float64 weights (broadcastable, or full-shaped per region).
    """
    dims = spatial_dims(spatial_coords)
    shape = [1] * len(dims)

    if lat_weights and "lat" in dims:
        lat_vals = np.asarray(spatial_coords["lat"])
        w = lat_weight(torch.tensor(lat_vals, dtype=torch.float64))
        shape[dims.index("lat")] = len(lat_vals)
        base = w.reshape(shape)
    else:
        base = torch.ones(shape, dtype=torch.float64)

    if regions is None:
        return base

    full_shape = [len(np.asarray(spatial_coords[d])) for d in dims]
    masks = region_masks(spatial_coords, regions)
    return torch.stack(
        [base.expand(full_shape) * mask for mask in masks.values()], dim=0
    )
