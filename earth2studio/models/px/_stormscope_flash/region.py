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

"""Resolve geographic requests to real, patch-aligned native-grid context."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from earth2studio.utils.type import CoordSystem


@dataclass(frozen=True)
class RegionInfo:
    """Resolved native input, requested rectangle, and exact geographic mask.

    Index bounds are half-open and ordered ``(y0, y1, x0, x1)``. Padding is
    ``(top, bottom, left, right)`` in array order and includes size/alignment
    expansion. ``requested_mask`` is relative to the requested output rectangle.
    """

    requested_bounds: tuple[float, float, float, float] | None
    input_bounds: tuple[int, int, int, int]
    output_bounds: tuple[int, int, int, int]
    requested_mask: np.ndarray
    padding: tuple[int, int, int, int]

    @property
    def input_shape(self) -> tuple[int, int]:
        """Actual model input dimensions, including available context."""
        y0, y1, x0, x1 = self.input_bounds
        return y1 - y0, x1 - x0

    @property
    def output_slices(self) -> tuple[slice, slice]:
        """Output rectangle relative to the retained model input."""
        y0, _, x0, _ = self.input_bounds
        a, b, c, d = self.output_bounds
        return slice(a - y0, b - y0), slice(c - x0, d - x0)

    def crop_output(
        self, forecast: torch.Tensor, coords: CoordSystem, *, mask: bool = True
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Crop presentation/output fields without modifying rollout histories."""
        if forecast.shape[-2:] != self.input_shape:
            raise ValueError("Forecast spatial dimensions do not match this region")
        ys, xs = self.output_slices
        result = forecast[..., ys, xs].clone()
        output_coords = coords.copy()
        for name, index in (("y", ys), ("x", xs)):
            if name in output_coords:
                output_coords[name] = output_coords[name][index].copy()
        for name in ("lat", "lon"):
            if name in output_coords:
                value = output_coords[name]
                output_coords[name] = (
                    value[ys, xs].copy() if value.ndim == 2 else value.copy()
                )
        if mask:
            valid = torch.as_tensor(self.requested_mask.copy(), device=result.device)
            result = torch.where(valid, result, torch.nan)
        return result, output_coords


def _axis_context(start: int, stop: int, size: int, padding: int) -> tuple[int, int]:
    start, stop = max(0, start - padding), min(size, stop + padding)
    missing = max(0, 200 - (stop - start))
    start = max(0, start - missing // 2)
    stop = min(size, max(stop, start + 200))
    start = max(0, min(start, stop - 200))
    return (start // 4) * 4, min(size, ((stop + 3) // 4) * 4)


def _inside_grid(
    corners: np.ndarray, latitude: np.ndarray, longitude: np.ndarray
) -> bool:
    # Ray casting on the native curvilinear perimeter, including its boundary.
    lat = np.concatenate(
        (latitude[0], latitude[1:, -1], latitude[-1, -2::-1], latitude[-2:0:-1, 0])
    )
    lon = np.concatenate(
        (longitude[0], longitude[1:, -1], longitude[-1, -2::-1], longitude[-2:0:-1, 0])
    )
    x1, y1, x2, y2 = lon, lat, np.roll(lon, -1), np.roll(lat, -1)
    for x, y in corners:
        dx, dy = x2 - x1, y2 - y1
        cross = (x - x1) * dy - (y - y1) * dx
        on = (
            (np.abs(cross) <= 1e-6 * np.maximum(np.hypot(dx, dy), 1e-6))
            & (x >= np.minimum(x1, x2) - 1e-6)
            & (x <= np.maximum(x1, x2) + 1e-6)
            & (y >= np.minimum(y1, y2) - 1e-6)
            & (y <= np.maximum(y1, y2) + 1e-6)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            crossings = ((y1 > y) != (y2 > y)) & (x < dx * (y - y1) / dy + x1)
        if not bool(on.any()) and not bool(np.count_nonzero(crossings) % 2):
            return False
    return True


def resolve_region(
    latitude: np.ndarray,
    longitude: np.ndarray,
    region: Mapping[str, Sequence[float]] | None,
    padding: int = 25,
) -> RegionInfo:
    """Resolve an in-domain lat/lon box with clipped real context and a 200px floor."""
    if isinstance(padding, bool) or not isinstance(padding, int) or padding < 0:
        raise ValueError("padding must be a non-negative integer")
    if latitude.ndim != 2 or latitude.shape != longitude.shape:
        raise ValueError(
            "Latitude and longitude must be matching two-dimensional grids"
        )
    h, w = latitude.shape
    if min(h, w) < 200 or h % 4 or w % 4:
        raise ValueError("Native grid dimensions must be >=200 and divisible by four")
    bounds = None
    if region is None:
        mask = np.ones((h, w), dtype=bool)
        y0, y1, x0, x1 = 0, h, 0, w
    else:
        if set(region) != {"lat", "lon"}:
            raise ValueError("region must contain exactly lat and lon bounds")
        if len(region["lat"]) != 2 or len(region["lon"]) != 2:
            raise ValueError("Each region bound must contain two values")
        south, north = map(float, region["lat"])
        raw_west, raw_east = map(float, region["lon"])
        west, east = (raw_west + 180) % 360 - 180, (raw_east + 180) % 360 - 180
        if (
            not np.isfinite([south, north, raw_west, raw_east]).all()
            or not -90 <= south < north <= 90
            or not west < east
            or abs(raw_east - raw_west) >= 360
        ):
            raise ValueError(
                "Use finite increasing latitude and longitude bounds within the native CONUS domain"
            )
        signed: np.ndarray = (longitude.astype(np.float64) + 180) % 360 - 180
        if not _inside_grid(
            np.array([(west, south), (west, north), (east, south), (east, north)]),
            latitude,
            signed,
        ):
            raise ValueError(
                "Requested region extends outside the native model footprint"
            )
        mask = (
            (latitude >= south)
            & (latitude <= north)
            & (signed >= west)
            & (signed <= east)
        )
        rows, cols = np.nonzero(mask)
        if not rows.size:
            raise ValueError("Requested region contains no native grid points")
        y0, y1, x0, x1 = (
            int(rows.min()),
            int(rows.max()) + 1,
            int(cols.min()),
            int(cols.max()) + 1,
        )
        bounds = south, north, west, east
    iy0, iy1 = _axis_context(y0, y1, h, padding)
    ix0, ix1 = _axis_context(x0, x1, w, padding)
    roi_mask = mask[y0:y1, x0:x1].copy()
    roi_mask.flags.writeable = False
    return RegionInfo(
        bounds,
        (iy0, iy1, ix0, ix1),
        (y0, y1, x0, x1),
        roi_mask,
        (y0 - iy0, iy1 - y1, x0 - ix0, ix1 - x1),
    )
