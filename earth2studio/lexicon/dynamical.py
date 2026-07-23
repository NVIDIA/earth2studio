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

from collections.abc import Callable

import numpy as np

from .base import LexiconType

# Standard gravity used to convert geopotential height [m] to geopotential
# [m2 s-2], matching the Earth2Studio ``z<level>`` convention.
GRAVITY = 9.80665

# Pressure levels (hPa) currently served by dynamical.org collections as named
# variables (e.g. ``geopotential_height_500hpa``). Requesting a level a given
# collection does not provide raises a clear error in the data source after it
# checks the collection's STAC ``cube:variables``.
LEVELS = [500, 850, 925]


class DynamicalLexicon(metaclass=LexiconType):
    """dynamical.org Lexicon.

    Maps Earth2Studio variable ids to the descriptive variable names used by
    dynamical.org datasets (e.g. ``t2m`` -> ``temperature_2m``,
    ``z500`` -> ``geopotential_height_500hpa``). dynamical.org bakes the
    vertical level into the variable name rather than using a level dimension,
    so pressure-level fields map to dedicated ``*_<level>hpa`` names.

    Unit conversions follow the same pattern as other Earth2Studio lexicons:
    the modifier returned by :meth:`get_item` converts from the source's native
    unit to the Earth2Studio convention (e.g. geopotential height in metres to
    geopotential in m2 s-2, Celsius to Kelvin). Variable names absent from this
    lexicon may still be requested by their native dynamical.org name as a
    pass-through in the data source.

    Note
    ----
    Variable inventory varies per collection. See the dynamical.org STAC catalog
    for the authoritative per-dataset variable list:

    - https://stac.dynamical.org/catalog.json
    """

    VOCAB = {
        # Single-level / surface
        "u10m": "wind_u_10m",
        "v10m": "wind_v_10m",
        "u100m": "wind_u_100m",
        "v100m": "wind_v_100m",
        "t2m": "temperature_2m",
        "d2m": "dew_point_temperature_2m",
        "r2m": "relative_humidity_2m",
        "sp": "pressure_surface",
        "msl": "pressure_reduced_to_mean_sea_level",
        "tcc": "total_cloud_cover_atmosphere",
        "tcwv": "precipitable_water_atmosphere",
    }
    # Pressure-level fields (level baked into the dynamical.org variable name).
    VOCAB.update({f"z{level}": f"geopotential_height_{level}hpa" for level in LEVELS})
    VOCAB.update({f"t{level}": f"temperature_{level}hpa" for level in LEVELS})

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return the dynamical.org variable name and a unit-conversion modifier.

        Conversions to the Earth2Studio convention:

        - ``temperature_2m``, ``dew_point_temperature_2m``, ``temperature_*hpa``:
          Celsius -> Kelvin
        - ``geopotential_height_*``: metres -> geopotential (m2 s-2)
        - ``total_cloud_cover_atmosphere``: percent -> fraction

        Parameters
        ----------
        val : str
            Earth2Studio variable id.

        Returns
        -------
        tuple[str, Callable]
            The dynamical.org variable name and a unit-conversion modifier.
        """
        dynamical_name = cls.VOCAB[val]

        if val in ("t2m", "d2m") or (val.startswith("t") and val[1:].isdigit()):
            # Celsius -> Kelvin (2 m, dew point, and pressure-level temperatures)
            def mod(x: np.ndarray) -> np.ndarray:
                """Convert Celsius to Kelvin."""
                return np.asarray(x) + 273.15

        elif val.startswith("z") and val[1:].isdigit():
            # Geopotential height (m) -> geopotential (m2 s-2)
            def mod(x: np.ndarray) -> np.ndarray:
                """Convert geopotential height to geopotential."""
                return np.asarray(x) * GRAVITY

        elif val == "tcc":
            # Percent -> fraction
            def mod(x: np.ndarray) -> np.ndarray:
                """Convert percent to fraction."""
                return np.asarray(x) / 100.0

        else:

            def mod(x: np.ndarray) -> np.ndarray:
                """Identity modifier."""
                return x

        return dynamical_name, mod
