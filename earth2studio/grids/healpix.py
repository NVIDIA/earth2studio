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

"""HEALPix grids."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from earth2studio.grids._utils import geographic_subset_indexers, metadata
from earth2studio.grids.base import GridTopology

HEALPixOrdering = Literal["nested", "ring", "xy"]
HEALPixLayout = Literal["flat", "face"]
HEALPixOrigin = Literal["south", "east", "north", "west"]


def _rotate(
    nside: int, rotations: int, x: NDArray[Any], y: NDArray[Any]
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Rotate local face coordinates counter-clockwise."""
    match rotations % 4:
        case 1:
            return nside - y - 1, x
        case 2:
            return nside - x - 1, nside - y - 1
        case 3:
            return y, nside - x - 1
        case _:
            return x, y


def _xy_to_nested(
    nside: int,
    pixels: NDArray[Any],
    origin: HEALPixOrigin,
    clockwise: bool,
) -> NDArray[np.int64]:
    """Convert an Earth2Grid-style XY index to NESTED ordering."""
    pixels = np.asarray(pixels, dtype=np.int64)
    face = pixels // nside**2
    local = pixels % nside**2
    y, x = np.divmod(local, nside)
    x, y = _rotate(nside, ("south", "east", "north", "west").index(origin), x, y)
    if clockwise:
        x, y = y, x

    nested = np.zeros_like(pixels)
    for bit in range(nside.bit_length() - 1):
        nested |= ((x >> bit) & 1) << (2 * bit)
        nested |= ((y >> bit) & 1) << (2 * bit + 1)
    return nested + face * nside**2


def _healpix_coordinates(
    nside: int,
    ordering: HEALPixOrdering,
    pixels: NDArray[Any],
    origin: HEALPixOrigin,
    clockwise: bool,
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Return pixel-center latitude and longitude."""
    if ordering == "xy":
        pixels = _xy_to_nested(nside, pixels, origin, clockwise)
        ordering = "nested"
    if ordering == "ring":
        npix = 12 * nside**2
        ncap = 2 * nside * (nside - 1)
        pixels = np.asarray(pixels, dtype=np.int64)
        z = np.empty(pixels.shape, dtype=float)
        longitude = np.empty(pixels.shape, dtype=float)

        north = pixels < ncap
        equatorial = (pixels >= ncap) & (pixels < npix - ncap)
        south = pixels >= npix - ncap
        if np.any(north):
            ipix = pixels[north]
            ring = np.floor(0.5 * (1 + np.sqrt(1 + 2 * ipix))).astype(int)
            azimuth = ipix + 1 - 2 * ring * (ring - 1)
            z[north] = 1 - ring**2 / (3 * nside**2)
            longitude[north] = (azimuth - 0.5) * np.pi / (2 * ring)
        if np.any(equatorial):
            ipix = pixels[equatorial] - ncap
            ring = ipix // (4 * nside) + nside
            azimuth = ipix % (4 * nside) + 1
            shift = 0.5 * (1 + (ring + nside) % 2)
            z[equatorial] = (2 * nside - ring) * 2 / (3 * nside)
            longitude[equatorial] = (azimuth - shift) * np.pi / (2 * nside)
        if np.any(south):
            ipix = npix - pixels[south]
            ring = np.floor(0.5 * (1 + np.sqrt(2 * ipix - 1))).astype(int)
            azimuth = 4 * ring + 1 - (ipix - 2 * ring * (ring - 1))
            z[south] = -1 + ring**2 / (3 * nside**2)
            longitude[south] = (azimuth - 0.5) * np.pi / (2 * ring)
        latitude = 90 - np.degrees(np.arccos(np.clip(z, -1, 1)))
        return latitude, np.mod(np.degrees(longitude), 360)

    level = nside.bit_length() - 1
    pixels = np.asarray(pixels, dtype=np.int64)
    local = pixels % nside**2
    x = np.zeros_like(local)
    y = np.zeros_like(local)
    for bit in range(level):
        x |= ((local >> (2 * bit)) & 1) << bit
        y |= ((local >> (2 * bit + 1)) & 1) << bit

    face = pixels // nside**2
    x = (x + 0.5) / nside
    y = (y + 0.5) / nside
    x_origin = np.array([1, 2, 3, 4, 0, 1, 2, 3, 0, 1, 2, 3])
    y_origin = np.array([1, 2, 3, 4, 1, 2, 3, 4, 2, 3, 4, 5])
    x_rot = x_origin[face] + x
    y_rot = -y_origin[face] + y
    xs = (x_rot - y_rot - 1) * np.pi / 4
    ys = (x_rot + y_rot) * np.pi / 4

    polar = np.abs(ys) > np.pi / 4
    longitude = xs.copy()
    longitude[polar] -= (
        (np.abs(ys[polar]) - np.pi / 4)
        / (np.abs(ys[polar]) - np.pi / 2)
        * (np.mod(xs[polar], np.pi / 2) - np.pi / 4)
    )
    z = 8 * ys / (3 * np.pi)
    term = 2 - 4 * np.abs(ys[polar]) / np.pi
    z[polar] = (1 - term**2 / 3) * np.sign(ys[polar])
    latitude = 90 - np.degrees(np.arccos(np.clip(z, -1, 1)))
    return latitude, np.mod(np.degrees(longitude), 360)


@dataclass(frozen=True)
class HEALPixGrid:
    """Define a flat or face-oriented HEALPix grid."""

    level: int
    ordering: HEALPixOrdering = "nested"
    layout: HEALPixLayout = "flat"
    xy_origin: HEALPixOrigin = "south"
    xy_clockwise: bool = False

    def __post_init__(self) -> None:
        if self.level < 0:
            raise ValueError("HEALPix level must be nonnegative")
        if self.ordering not in {"nested", "ring", "xy"}:
            raise ValueError("HEALPix ordering must be 'nested', 'ring', or 'xy'")
        if self.layout not in {"flat", "face"}:
            raise ValueError("HEALPix layout must be 'flat' or 'face'")
        if self.layout == "face" and self.ordering != "xy":
            raise ValueError("HEALPix face layout requires XY ordering")
        if self.xy_origin not in {"south", "east", "north", "west"}:
            raise ValueError("HEALPix XY origin must be a cardinal direction")
        if self.ordering != "xy" and (self.xy_origin != "south" or self.xy_clockwise):
            raise ValueError("HEALPix XY orientation requires XY ordering")

    @property
    def nside(self) -> int:
        return 2**self.level

    @property
    def dims(self) -> tuple[str, ...]:
        return ("face", "height", "width") if self.layout == "face" else ("hpx",)

    @property
    def shape(self) -> tuple[int, ...]:
        return (
            (12, self.nside, self.nside)
            if self.layout == "face"
            else (12 * self.nside**2,)
        )

    @property
    def topology(self) -> GridTopology:
        return "healpix"

    @property
    def crs(self) -> None:
        return None

    def coords(
        self,
        indexes: Mapping[str, NDArray[Any]] | None = None,
        *,
        only_index: bool = False,
    ) -> xr.Coordinates:
        """Return HEALPix indexes and geographic coordinates."""
        indexes = indexes or {
            dimension: np.arange(size)
            for dimension, size in zip(self.dims, self.shape, strict=True)
        }
        coordinates = xr.Coordinates(
            {dimension: indexes[dimension] for dimension in self.dims}
        )
        if only_index:
            return coordinates
        if self.layout == "flat":
            pixels = np.asarray(indexes["hpx"])
            latitude, longitude = _healpix_coordinates(
                self.nside,
                self.ordering,
                pixels,
                self.xy_origin,
                self.xy_clockwise,
            )
            return coordinates.assign(lat=("hpx", latitude), lon=("hpx", longitude))

        face, height, width = np.meshgrid(
            indexes["face"], indexes["height"], indexes["width"], indexing="ij"
        )
        pixels = face * self.nside**2 + height * self.nside + width
        latitude, longitude = _healpix_coordinates(
            self.nside, "xy", pixels, self.xy_origin, self.xy_clockwise
        )
        return coordinates.assign(lat=(self.dims, latitude), lon=(self.dims, longitude))

    def subset_indexers(
        self,
        coordinates: xr.Coordinates,
        *,
        bounds: tuple[float, float, float, float] | None = None,
        bounds_crs: Any | None = None,
        faces: int | Sequence[int] | NDArray[Any] | None = None,
    ) -> dict[str, Any]:
        """Translate bounds and faces into indexers."""
        if faces is None:
            return geographic_subset_indexers(
                self, coordinates, bounds=bounds, bounds_crs=bounds_crs
            )
        face_values: NDArray[np.int64] = np.asarray(
            np.atleast_1d(faces), dtype=np.int64
        )
        if face_values.size == 0 or np.any((face_values < 0) | (face_values > 11)):
            raise ValueError("HEALPix faces must be integers from 0 through 11")
        if self.ordering == "ring":
            raise NotImplementedError(
                "HEALPix face selection does not support RING ordering"
            )

        if self.layout == "face":
            face_positions = np.flatnonzero(
                np.isin(np.asarray(coordinates["face"]), np.unique(face_values))
            )
            bounded = geographic_subset_indexers(
                self,
                xr.Dataset(coords=coordinates).isel(face=face_positions).coords,
                bounds=bounds,
                bounds_crs=bounds_crs,
            )
            if not bounded:
                return {"face": face_positions}
            local_faces = np.arange(face_positions.size)[bounded.pop("face")]
            return {"face": face_positions[local_faces], **bounded}

        pixels = np.asarray(coordinates["hpx"])
        mask = np.isin(pixels // self.nside**2, np.unique(face_values))
        positions = np.flatnonzero(mask)
        bounded = geographic_subset_indexers(
            self,
            xr.Coordinates({"hpx": ("hpx", pixels[positions])}),
            bounds=bounds,
            bounds_crs=bounds_crs,
        )
        if bounded:
            positions = positions[bounded["hpx"]]
        if positions.size == 0:
            raise ValueError("Grid subset must contain at least one spatial cell")
        return {"hpx": positions}

    def cell_bounds(self, indexes: Mapping[str, NDArray[Any]]) -> None:
        """Return no cell bounds."""
        return None

    @property
    def attrs(self) -> dict[str, Any]:
        """Return serializable grid attributes."""
        details: dict[str, Any] = {
            "level": self.level,
            "nside": self.nside,
            "ordering": self.ordering,
            "layout": self.layout,
        }
        if self.ordering == "xy":
            details.update(origin=self.xy_origin, clockwise=self.xy_clockwise)
        return metadata(self, **details)

    def fingerprint(self) -> str:
        """Return stable geometry identity."""
        return (
            f"healpix:{self.level}:{self.ordering}:{self.layout}:"
            f"{self.xy_origin}:{self.xy_clockwise}"
        )
