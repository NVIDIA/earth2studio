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

"""Metadata-driven lexicon for Arraylake / Earthmover Marketplace data.

Unlike most Earth2Studio lexicons, the Arraylake connector does **not** hard-code
the variable names of any single dataset. Earthmover Marketplace repositories use
a variety of native naming schemes (ECMWF cfVarName like ``t2m``, GRIB short names
like ``2t``, or descriptive CF names like ``temperature_2m``). What they share is
adherence to *established metadata conventions* -- the ECMWF parameter database
(``GRIB_paramId`` / ``GRIB_shortName``) and CF (``standard_name``, ``units``, and
vertical-level coordinates).

This module therefore stores a single, dataset-agnostic crosswalk from each
Earth2Studio variable id to its ECMWF/CF descriptor (:class:`VariableSpec`). The
data source (:mod:`earth2studio.data.arraylake`) resolves these descriptors against
whatever metadata a given repository actually exposes, in priority order
``GRIB_paramId`` -> ``GRIB_shortName`` / variable name -> ``standard_name``.

Earth2Studio variable ids follow the ECMWF parameter database. See:

- https://codes.ecmwf.int/grib/param-db/
- https://cfconventions.org/
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .base import LexiconType

#: Standard gravitational acceleration (m s-2), used to convert geopotential
#: height (m) to geopotential (m2 s-2).
GRAVITY = 9.80665


@dataclass(frozen=True)
class VariableSpec:
    """Dataset-agnostic descriptor for an Earth2Studio variable.

    The descriptor captures the identifiers that Marketplace repositories expose
    through established metadata conventions, so a variable can be located in a
    repository without knowing its native variable name in advance.

    Parameters
    ----------
    e2s : str
        Earth2Studio variable id (e.g. ``t2m``, ``z500``).
    short_name : str
        ECMWF GRIB short name / cfVarName (e.g. ``2t`` for ``t2m``, ``t`` for
        ``t850``). Note these are *not* unique across vertical levels.
    param_id : int
        ECMWF parameter database id. Authoritative and unambiguous when present.
    standard_name : str
        CF ``standard_name`` for the field, or ``""`` when CF does not define one.
    level_type : str
        One of ``"surface"`` (single-level / height-above-ground) or
        ``"isobaric"`` (pressure level).
    level : float | None
        Pressure level in hPa for ``isobaric`` variables, else ``None``.
    canonical_units : str
        The physical units Earth2Studio expects for this variable. Used together
        with the repository variable's ``units`` attribute to choose a unit
        conversion at resolution time.
    """

    e2s: str
    short_name: str
    param_id: int
    standard_name: str
    level_type: str
    level: float | None
    canonical_units: str


# ---------------------------------------------------------------------------
# Surface / single-level variables: (e2s, short_name, param_id, standard_name,
# canonical_units)
# ---------------------------------------------------------------------------
_SURFACE: list[tuple[str, str, int, str, str]] = [
    ("u10m", "10u", 165, "eastward_wind", "m s-1"),
    ("v10m", "10v", 166, "northward_wind", "m s-1"),
    ("u100m", "100u", 228246, "eastward_wind", "m s-1"),
    ("v100m", "100v", 228247, "northward_wind", "m s-1"),
    ("t2m", "2t", 167, "air_temperature", "K"),
    ("d2m", "2d", 168, "dew_point_temperature", "K"),
    ("fg10m", "10fg", 49, "wind_speed_of_gust", "m s-1"),
    ("sst", "sst", 34, "sea_surface_temperature", "K"),
    ("skt", "skt", 235, "surface_temperature", "K"),
    ("sp", "sp", 134, "surface_air_pressure", "Pa"),
    ("msl", "msl", 151, "air_pressure_at_mean_sea_level", "Pa"),
    ("tcwv", "tcwv", 137, "atmosphere_mass_content_of_water_vapor", "kg m-2"),
    ("tcw", "tcw", 136, "", "kg m-2"),
    ("tcc", "tcc", 164, "cloud_area_fraction", "1"),
    ("hcc", "hcc", 188, "", "1"),
    ("mcc", "mcc", 187, "", "1"),
    ("lcc", "lcc", 186, "", "1"),
    ("tp", "tp", 228, "", "m"),
    ("cp", "cp", 143, "lwe_thickness_of_convective_precipitation_amount", "m"),
    ("lsp", "lsp", 142, "lwe_thickness_of_stratiform_precipitation_amount", "m"),
    ("sd", "sd", 141, "lwe_thickness_of_surface_snow_amount", "m"),
    ("sf", "sf", 144, "lwe_thickness_of_snowfall_amount", "kg m-2"),
    ("ssrd", "ssrd", 169, "surface_downwelling_shortwave_flux_in_air", "J m-2"),
    ("strd", "strd", 175, "surface_downwelling_longwave_flux_in_air", "J m-2"),
    ("z", "z", 129, "geopotential", "m2 s-2"),
    ("lsm", "lsm", 172, "land_binary_mask", "1"),
    ("stl1", "stl1", 139, "surface_temperature", "K"),
    ("stl2", "stl2", 170, "surface_temperature", "K"),
    ("swvl1", "swvl1", 39, "volume_fraction_of_condensed_water_in_soil", "m3 m-3"),
    ("swvl2", "swvl2", 40, "volume_fraction_of_condensed_water_in_soil", "m3 m-3"),
]

# ---------------------------------------------------------------------------
# Pressure-level variables: (base, short_name, param_id, standard_name,
# canonical_units). The Earth2Studio id is f"{base}{level}".
# ---------------------------------------------------------------------------
_PRESSURE: list[tuple[str, str, int, str, str]] = [
    ("u", "u", 131, "eastward_wind", "m s-1"),
    ("v", "v", 132, "northward_wind", "m s-1"),
    ("t", "t", 130, "air_temperature", "K"),
    ("z", "z", 129, "geopotential", "m2 s-2"),
    ("q", "q", 133, "specific_humidity", "kg kg-1"),
    ("r", "r", 157, "relative_humidity", "%"),
    ("w", "w", 135, "lagrangian_tendency_of_air_pressure", "Pa s-1"),
    ("d", "d", 155, "divergence_of_wind", "s-1"),
    ("vo", "vo", 138, "atmosphere_relative_vorticity", "s-1"),
]

#: Pressure levels (hPa) for which pressure-level Earth2Studio ids are defined.
PRESSURE_LEVELS: list[int] = [
    10,
    50,
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    850,
    925,
    1000,
]


def _build_specs() -> dict[str, VariableSpec]:
    """Build the full Earth2Studio id -> :class:`VariableSpec` crosswalk."""
    specs: dict[str, VariableSpec] = {}
    for e2s, short, pid, sn, units in _SURFACE:
        specs[e2s] = VariableSpec(e2s, short, pid, sn, "surface", None, units)
    for base, short, pid, sn, units in _PRESSURE:
        for level in PRESSURE_LEVELS:
            e2s = f"{base}{level}"
            specs[e2s] = VariableSpec(
                e2s, short, pid, sn, "isobaric", float(level), units
            )
    return specs


def normalize_units(units: str | None) -> str:
    """Normalize a CF/GRIB units string to a canonical comparison token.

    Handles the common spelling variations seen across Marketplace datasets, e.g.
    ``"m s**-1"`` and ``"m s-1"``, ``"(0 - 1)"`` and ``"1"``,
    ``"degree_Celsius"`` and ``"degC"``.

    Parameters
    ----------
    units : str | None
        Raw ``units`` attribute value.

    Returns
    -------
    str
        Normalized units token (may be ``""`` if unknown / missing).
    """
    if not units:
        return ""
    u = units.strip().lower()
    u = u.replace("**", "").replace("^", "")
    u = u.replace(" ", "")
    # Celsius spellings
    if u in {"degree_celsius", "degreec", "degc", "celsius", "°c", "degreescelsius"}:
        return "degc"
    # Kelvin
    if u in {"k", "kelvin"}:
        return "k"
    # Fraction (0-1)
    if u in {"1", "(0-1)", "0-1", "fraction", "dimensionless", ""}:
        return "1" if u != "" else ""
    # Percent
    if u in {"%", "percent"}:
        return "percent"
    # Water-equivalent depth
    if u in {"mofwaterequivalent", "mwe"}:
        return "m"
    return u


def make_modifier(spec: VariableSpec, src_units: str | None) -> Callable:
    """Return a post-processing function aligning source values to E2S units.

    The conversion is chosen purely from metadata: the repository variable's
    ``units`` attribute compared against the variable's
    :attr:`VariableSpec.canonical_units`. Supported conversions cover the cases
    observed across Marketplace repositories; unknown mismatches fall through to
    the identity (the data source logs a warning).

    Parameters
    ----------
    spec : VariableSpec
        The resolved variable descriptor.
    src_units : str | None
        The ``units`` attribute of the matched repository variable.

    Returns
    -------
    Callable
        ``Callable[[np.ndarray], np.ndarray]`` transforming raw values in place.
    """
    src = normalize_units(src_units)
    dst = normalize_units(spec.canonical_units)

    if src == dst or src == "":
        return lambda x: x

    # Temperature: Celsius -> Kelvin
    if src == "degc" and dst == "k":
        return lambda x: x + 273.15
    if src == "k" and dst == "degc":
        return lambda x: x - 273.15

    # Geopotential height (m) -> geopotential (m2 s-2)
    if spec.short_name == "z" and src == "m" and dst == "m2s-2":
        return lambda x: x * GRAVITY
    if spec.short_name == "z" and src == "m2s-2" and dst == "m":
        return lambda x: x / GRAVITY

    # Cloud cover / fractions: percent <-> 0-1
    if src == "percent" and dst == "1":
        return lambda x: x / 100.0
    if src == "1" and dst == "percent":
        return lambda x: x * 100.0

    # Relative humidity stored as fraction -> percent
    if dst == "percent" and src == "1":
        return lambda x: x * 100.0

    # Precip / snow water equivalent: kg m-2 (== mm) -> m
    if src == "kgm-2" and dst == "m":
        return lambda x: x / 1000.0
    if src == "m" and dst == "kgm-2":
        return lambda x: x * 1000.0

    # Unknown mismatch: identity (data source emits a warning).
    return lambda x: x


class ArraylakeLexicon(metaclass=LexiconType):
    """Lexicon for Earthmover Arraylake / Marketplace repositories.

    This lexicon is metadata-driven: ``VOCAB`` encodes the Earth2Studio <-> ECMWF
    standard crosswalk (it does **not** name the variables of any specific
    repository). The :class:`~earth2studio.data.arraylake.Arraylake` data source
    uses :attr:`SPECS` to locate the matching variable inside an opened repository
    via its ``GRIB_paramId`` / ``GRIB_shortName`` / ``standard_name`` metadata.

    Note
    ----
    Earth2Studio variable ids follow the ECMWF parameter database:

    - https://codes.ecmwf.int/grib/param-db/
    - https://cfconventions.org/
    """

    #: Earth2Studio id -> :class:`VariableSpec`.
    SPECS: dict[str, VariableSpec] = _build_specs()

    #: Earth2Studio id -> ``"{short_name}::{param_id}::{level_type}::{level}"``,
    #: mirroring the ``::`` convention used by other Earth2Studio lexicons.
    VOCAB: dict[str, str] = {
        e2s: f"{s.short_name}::{s.param_id}::{s.level_type}::"
        f"{'' if s.level is None else int(s.level)}"
        for e2s, s in SPECS.items()
    }

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Return the standard crosswalk key and an identity modifier.

        Parameters
        ----------
        val : str
            Earth2Studio variable id.

        Returns
        -------
        tuple[str, Callable]
            The ``::``-encoded ECMWF descriptor string and an identity modifier.

        Note
        ----
        Unit normalization for Arraylake is metadata-driven and depends on the
        *repository* variable's ``units`` attribute, which is not known here. The
        actual modifier is built by :func:`make_modifier` inside the data source
        once a repository variable has been matched; ``get_item`` therefore
        returns the identity to satisfy the lexicon protocol.
        """
        return cls.VOCAB[val], (lambda x: x)

    @classmethod
    def spec(cls, val: str) -> VariableSpec:
        """Return the :class:`VariableSpec` for an Earth2Studio variable id.

        Parameters
        ----------
        val : str
            Earth2Studio variable id.

        Returns
        -------
        VariableSpec
            The dataset-agnostic descriptor.

        Raises
        ------
        KeyError
            If ``val`` is not a known Earth2Studio variable id.
        """
        return cls.SPECS[val]
