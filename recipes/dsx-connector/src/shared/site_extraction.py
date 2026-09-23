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

"""Extract model values from the grid cell nearest each configured site.

The module supports two grid layouts:

- :class:`CurvilinearSiteExtractor` accepts two-dimensional latitude and longitude arrays.
- :class:`RegularLatLonSiteExtractor` accepts separate one-dimensional latitude and longitude
  axes.

Both implement the :class:`SiteExtractor` interface, allowing collectors to extract site values
without depending on a particular grid layout. Both use nearest-neighbor sampling, which is simple
and efficient when extracting values for a small number of sites. Additional sampling methods,
such as bilinear interpolation, can implement the same interface without changing the collectors.
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree  # type: ignore

# Base distance used to detect sites outside the model grid. It is intentionally larger than
# HRRR's roughly 3 km spacing. Regular grids may increase the limit for larger grid cells.
# This prevents invalid coordinates from silently being mapped to an edge cell.
_MAX_SITE_DISTANCE_KM = 25.0
_EARTH_RADIUS_KM = 6371.0


def validate_and_normalize_sites(
    sites: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate site configurations and return copies with coordinates converted to floats.

    At least one site is required. Every site must have a unique, non-empty identifier and valid
    latitude and longitude values. Numeric coordinate strings are converted to floats in the
    returned copies. The supplied dictionaries are not changed.

    Parameters
    ----------
    sites : list[dict[str, Any]]
        Site configurations containing ``id``, ``lat``, and ``lon``.

    Returns
    -------
    list[dict[str, Any]]
        Copies of the site configurations with ``lat`` and ``lon`` stored as floats.

    Raises
    ------
    ValueError
        If the list is empty or a site identifier or coordinate is invalid.
    """
    if not sites:
        raise ValueError("sites: at least one site is required")
    seen_ids: set[str] = set()
    normalized_sites: list[dict[str, Any]] = []
    for site_index, site in enumerate(sites):
        site_id = site.get("id")
        if not isinstance(site_id, str) or not site_id.strip():
            raise ValueError(f"sites[{site_index}]: 'id' must be a non-empty string")
        if site_id in seen_ids:
            raise ValueError(f"sites: duplicate site id {site_id!r}")
        seen_ids.add(site_id)

        normalized_site = dict(site)
        # Accept both common longitude conventions: -180 to 180 and 0 to 360 degrees.
        for key, minimum, maximum in (
            ("lat", -90.0, 90.0),
            ("lon", -180.0, 360.0),
        ):
            try:
                raw_value = site[key]
                if isinstance(raw_value, (bool, np.bool_)):
                    raise TypeError
                value = float(raw_value)
            except (KeyError, TypeError, ValueError):
                raise ValueError(
                    f"site {site_id}: '{key}' must be a finite number"
                ) from None
            if not math.isfinite(value) or not (minimum <= value <= maximum):
                raise ValueError(
                    f"site {site_id}: '{key}'={site.get(key)!r} out of range "
                    f"[{minimum}, {maximum}]"
                )
            normalized_site[key] = value
        normalized_sites.append(normalized_site)
    return normalized_sites


class SiteExtractor(Protocol):
    """Define the attributes and method that every site extractor must provide.

    A protocol describes an interface without providing an implementation. Collectors can therefore
    use any extractor that follows this interface, regardless of the model's grid layout.
    ``site_lon`` stores each sampled location's longitude in degrees from 0 inclusive to 360
    exclusive. It is available for workflows that need to rotate grid-relative winds.
    """

    site_lon: dict[str, float]

    def extract(self, field_yx: Any) -> dict[str, float]:
        """Return one sampled value per site from a 2-D NumPy array or Torch tensor."""
        ...


class CurvilinearSiteExtractor:
    """Map each site to its nearest cell on a grid with 2-D coordinates.

    Parameters
    ----------
    lat : np.ndarray
        Two-dimensional latitude values for the model grid.
    lon : np.ndarray
        Two-dimensional longitude values for the model grid. Either common longitude convention is
        accepted.
    sites : list[dict[str, Any]]
        Site configurations, each containing ``id``, ``lat``, and ``lon``.

    Raises
    ------
    ValueError
        If the coordinate arrays are invalid or have different shapes, or if a site is too far from
        the nearest grid cell.
    """

    def __init__(
        self, lat: np.ndarray, lon: np.ndarray, sites: list[dict[str, Any]]
    ) -> None:
        sites = validate_and_normalize_sites(sites)
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        if lat.ndim != 2 or lon.ndim != 2:
            raise ValueError(
                f"lat and lon must be 2-D, got {lat.ndim}-D and {lon.ndim}-D"
            )
        if lat.shape != lon.shape:
            raise ValueError(
                f"lat and lon shapes must match, got {lat.shape} and {lon.shape}"
            )
        if lat.size == 0:
            raise ValueError("lat and lon grids must not be empty")
        if not (np.all(np.isfinite(lat)) and np.all(np.isfinite(lon))):
            raise ValueError("lat and lon grids must contain only finite values")

        self._shape = lat.shape
        # The tree is needed only while finding each site's grid indices.
        tree = cKDTree(self._to_unit_vectors(lat.ravel(), lon.ravel()))
        self._site_indices: dict[str, tuple[int, int]] = {}
        # Store sampled longitudes in one convention for workflows that rotate winds.
        self.site_lon: dict[str, float] = {}
        for site in sites:
            query_point = self._to_unit_vectors(
                np.array([site["lat"]]), np.array([site["lon"]])
            )
            chord_distance, flat_index = tree.query(query_point, k=1)
            # cKDTree returns the Euclidean chord on the unit sphere; convert to a
            # great-circle distance to threshold against the grid.
            distance_km = (
                2.0
                * _EARTH_RADIUS_KM
                * float(np.arcsin(np.clip(chord_distance[0] / 2.0, 0.0, 1.0)))
            )
            if distance_km > _MAX_SITE_DISTANCE_KM:
                raise ValueError(
                    f"site {site['id']} ({site['lat']}, {site['lon']}) is "
                    f"{distance_km:.0f} km from the "
                    f"nearest model grid cell (> {_MAX_SITE_DISTANCE_KM:.0f} km); it is outside "
                    "the model grid (check the coordinates, or the subregion crop)"
                )
            y_index, x_index = np.unravel_index(int(flat_index[0]), self._shape)
            site_id = site["id"]
            self._site_indices[site_id] = (int(y_index), int(x_index))
            self.site_lon[site_id] = float(lon[y_index, x_index] % 360.0)

    @staticmethod
    def _to_unit_vectors(lat: ArrayLike, lon: ArrayLike) -> np.ndarray:
        """Convert latitude and longitude to Cartesian points on a unit sphere.

        Representing locations this way lets the KD-tree measure across the longitude wraparound
        correctly, regardless of whether longitudes use -180 to 180 or 0 to 360 degrees.
        """
        latitude_radians = np.deg2rad(lat)
        longitude_radians = np.deg2rad(lon)
        return np.stack(
            [
                np.cos(latitude_radians) * np.cos(longitude_radians),
                np.cos(latitude_radians) * np.sin(longitude_radians),
                np.sin(latitude_radians),
            ],
            axis=-1,
        )

    def extract(self, field_yx: Any) -> dict[str, float]:
        """Sample a 2-D model field at each site's grid cell.

        Parameters
        ----------
        field_yx : Any
            A 2-D NumPy array or Torch tensor with the same shape as the latitude and longitude
            grids.

        Returns
        -------
        dict[str, float]
            Mapping of site identifier to the sampled value.

        Raises
        ------
        ValueError
            If the field does not have the same shape as the coordinate grids.
        """
        # Require the coordinate-grid shape so the stored indices select the intended cells.
        if tuple(field_yx.shape) != self._shape:
            raise ValueError(
                f"field shape {tuple(field_yx.shape)} does not match the 2-D grid {self._shape}"
            )
        return {
            site_id: float(field_yx[y_index, x_index])
            for site_id, (y_index, x_index) in self._site_indices.items()
        }


def _great_circle_km(
    latitude1: float,
    longitude1: float,
    latitude2: float,
    longitude2: float,
) -> float:
    """Return the shortest distance along Earth's surface between two locations."""
    latitude1_radians = math.radians(latitude1)
    latitude2_radians = math.radians(latitude2)
    latitude_difference = math.radians(latitude2 - latitude1)
    longitude_difference = math.radians(longitude2 - longitude1)
    haversine_term = (
        math.sin(latitude_difference / 2.0) ** 2
        + math.cos(latitude1_radians)
        * math.cos(latitude2_radians)
        * math.sin(longitude_difference / 2.0) ** 2
    )
    return 2.0 * _EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(haversine_term)))


class RegularLatLonSiteExtractor:
    """Map each site to its nearest cell on a grid with separate latitude and longitude axes.

    The axes may be ascending or descending and may use any regular spacing. Longitude distance
    wraps around the globe, so a site just below 0 degrees can map to a cell just below 360 degrees.
    Global and regional grids are supported. An axis that crosses the longitude boundary must remain
    monotonic, for example by using 350 to 370 degrees instead of resetting from 359 to 0.

    Parameters
    ----------
    lat : np.ndarray
        One-dimensional, strictly monotonic latitude axis in degrees. It corresponds to axis 0 of
        the model field.
    lon : np.ndarray
        One-dimensional, strictly monotonic longitude axis in degrees. It corresponds to axis 1 of
        the model field. Either common longitude convention is accepted.
    sites : list[dict[str, Any]]
        Site configurations, each containing ``id``, ``lat``, and ``lon``.

    Raises
    ------
    ValueError
        If either axis is invalid or a site is too far from the nearest grid cell.
    """

    def __init__(
        self, lat: np.ndarray, lon: np.ndarray, sites: list[dict[str, Any]]
    ) -> None:
        sites = validate_and_normalize_sites(sites)
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        self._validate_axis("lat", lat)
        self._validate_axis("lon", lon)
        # The axes are needed only while finding each site's grid indices and longitude.
        longitude_360 = lon % 360.0
        self._shape = (lat.size, lon.size)
        distance_limit_km = self._site_distance_limit_km(lat, lon)
        self._site_indices: dict[str, tuple[int, int]] = {}
        # Store sampled longitudes in one convention for workflows that rotate winds.
        self.site_lon: dict[str, float] = {}
        for site in sites:
            y_index = int(np.argmin(np.abs(lat - site["lat"])))
            # Cyclic longitude distance in [0, 180].
            longitude_distance = np.abs(
                ((longitude_360 - (site["lon"] % 360.0) + 180.0) % 360.0) - 180.0
            )
            x_index = int(np.argmin(longitude_distance))
            distance_km = _great_circle_km(
                site["lat"],
                site["lon"],
                float(lat[y_index]),
                float(longitude_360[x_index]),
            )
            if distance_km > distance_limit_km:
                raise ValueError(
                    f"site {site['id']} ({site['lat']}, {site['lon']}) is "
                    f"{distance_km:.0f} km from the nearest grid cell "
                    f"(> {distance_limit_km:.0f} km); check the coordinates"
                )
            site_id = site["id"]
            self._site_indices[site_id] = (y_index, x_index)
            self.site_lon[site_id] = float(longitude_360[x_index])

    @staticmethod
    def _validate_axis(name: str, axis: np.ndarray) -> None:
        """Check that an axis is usable for nearest-cell lookup."""
        if axis.ndim != 1:
            raise ValueError(f"{name} axis must be 1-D, got {axis.ndim}-D")
        if axis.size == 0:
            raise ValueError(f"{name} axis is empty")
        if not np.all(np.isfinite(axis)):
            raise ValueError(f"{name} axis has non-finite values")
        if axis.size > 1:
            differences = np.diff(axis)
            if not (np.all(differences > 0) or np.all(differences < 0)):
                raise ValueError(f"{name} axis must be strictly monotonic")

    @staticmethod
    def _site_distance_limit_km(lat: np.ndarray, lon: np.ndarray) -> float:
        """Return a site-distance limit based on the grid's largest cell.

        Coarser grids need a larger limit because valid sites can be farther from a cell center. The
        calculation uses the original longitude axis so a regional grid does not appear to contain
        an artificial large gap at the longitude wraparound.
        """
        latitude_step = float(np.max(np.abs(np.diff(lat)))) if lat.size > 1 else 180.0
        longitude_step = float(np.max(np.abs(np.diff(lon)))) if lon.size > 1 else 360.0
        half_diagonal_km = (
            0.5
            * math.hypot(latitude_step, longitude_step)
            * (math.pi / 180.0)
            * _EARTH_RADIUS_KM
        )
        return max(_MAX_SITE_DISTANCE_KM, 1.5 * half_diagonal_km)

    def extract(self, field_yx: Any) -> dict[str, float]:
        """Sample a 2-D ``(lat, lon)`` model field at each site's grid cell.

        Parameters
        ----------
        field_yx : Any
            A 2-D NumPy array or Torch tensor shaped like the model grid ``(lat, lon)``.

        Returns
        -------
        dict[str, float]
            Mapping of site identifier to the sampled value.

        Raises
        ------
        ValueError
            If the field does not have the shape defined by the latitude and longitude axes.
        """
        # Require the coordinate-grid shape so the stored indices select the intended cells.
        if tuple(field_yx.shape) != self._shape:
            raise ValueError(
                f"field shape {tuple(field_yx.shape)} does not match the 2-D grid "
                f"(lat, lon) = {self._shape}"
            )
        return {
            site_id: float(field_yx[y_index, x_index])
            for site_id, (y_index, x_index) in self._site_indices.items()
        }
