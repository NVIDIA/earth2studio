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

"""Earthmover IFS lexicon.

The public vocabulary matches the IFS variables currently available in the
Brightband Earthmover Marketplace forecast dataset. The extra metadata
descriptors are intentionally broader so additional Earthmover datasets can
reuse the same resolution utilities.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .base import LexiconType
from .ecmwf import IFSLexicon

GRAVITY = 9.80665


@dataclass(frozen=True)
class VariableSpec:
    """Earthmover descriptor for an IFS variable."""

    e2s: str
    short_name: str
    param_id: int | None
    standard_name: str
    level_type: str
    level: float | None
    canonical_units: str
    aliases: tuple[str, ...] = ()


_PARAM_IDS: dict[str, int] = {
    "100u": 228246,
    "100v": 228247,
    "10fg": 49,
    "10u": 165,
    "10v": 166,
    "2d": 168,
    "2t": 167,
    "asn": 32,
    "cp": 143,
    "d": 155,
    "fdir": 228021,
    "fg10": 49,
    "gh": 129,
    "hcc": 188,
    "lcc": 186,
    "lsm": 172,
    "mcc": 187,
    "msl": 151,
    "q": 133,
    "r": 157,
    "sd": 141,
    "sf": 144,
    "skt": 235,
    "snowc": 260038,
    "sot": 228139,
    "sp": 134,
    "ssr": 176,
    "ssrd": 169,
    "str": 177,
    "strd": 175,
    "t": 130,
    "tcc": 164,
    "tcw": 136,
    "tcwv": 137,
    "tp": 228,
    "tprate": 260048,
    "u": 131,
    "v": 132,
    "vo": 138,
    "vsw": 39,
    "w": 135,
    "z": 129,
}

_STANDARD_NAMES: dict[str, str] = {
    "100u": "eastward_wind",
    "100v": "northward_wind",
    "10fg": "wind_speed_of_gust",
    "10u": "eastward_wind",
    "10v": "northward_wind",
    "2d": "dew_point_temperature",
    "2t": "air_temperature",
    "cp": "lwe_thickness_of_convective_precipitation_amount",
    "d": "divergence_of_wind",
    "gh": "geopotential_height",
    "lsm": "land_binary_mask",
    "msl": "air_pressure_at_mean_sea_level",
    "q": "specific_humidity",
    "r": "relative_humidity",
    "sd": "lwe_thickness_of_surface_snow_amount",
    "sf": "lwe_thickness_of_snowfall_amount",
    "skt": "surface_temperature",
    "sp": "surface_air_pressure",
    "ssrd": "surface_downwelling_shortwave_flux_in_air",
    "strd": "surface_downwelling_longwave_flux_in_air",
    "t": "air_temperature",
    "tcc": "cloud_area_fraction",
    "tcwv": "atmosphere_mass_content_of_water_vapor",
    "u": "eastward_wind",
    "v": "northward_wind",
    "vo": "atmosphere_relative_vorticity",
    "w": "lagrangian_tendency_of_air_pressure",
    "z": "geopotential",
}

_CANONICAL_UNITS_BY_SHORT_NAME: dict[str, str] = {
    "100u": "m s-1",
    "100v": "m s-1",
    "10fg": "m s-1",
    "10u": "m s-1",
    "10v": "m s-1",
    "2d": "K",
    "2t": "K",
    "cp": "m",
    "d": "s-1",
    "fdir": "J m-2",
    "gh": "m2 s-2",
    "hcc": "1",
    "lcc": "1",
    "lsm": "1",
    "mcc": "1",
    "msl": "Pa",
    "q": "kg kg-1",
    "r": "%",
    "sd": "m",
    "sf": "kg m-2",
    "skt": "K",
    "snowc": "%",
    "sot": "K",
    "sp": "Pa",
    "ssr": "J m-2",
    "ssrd": "J m-2",
    "str": "J m-2",
    "strd": "J m-2",
    "t": "K",
    "tcc": "1",
    "tcw": "kg m-2",
    "tcwv": "kg m-2",
    "tp": "m",
    "tprate": "kg m-2 s-1",
    "u": "m s-1",
    "v": "m s-1",
    "vo": "s-1",
    "vsw": "m3 m-3",
    "w": "Pa s-1",
    "z": "m2 s-2",
}

_ALIASES_BY_E2S: dict[str, tuple[str, ...]] = {
    "u10m": ("u10",),
    "v10m": ("v10",),
    "u100m": ("u100",),
    "v100m": ("v100",),
}

_ALIASES_BY_SHORT_NAME: dict[str, tuple[str, ...]] = {
    "gh": ("z",),
}


def _level_type(value: str) -> str:
    if value == "pl":
        return "isobaric"
    if value == "sl":
        return "soil"
    if value == "sfc":
        return "surface"
    return value


def _level(value: str) -> float | None:
    return float(value) if value else None


def _build_specs(vocab: dict[str, str]) -> dict[str, VariableSpec]:
    specs: dict[str, VariableSpec] = {}
    for e2s, ifs_key in vocab.items():
        short_name, raw_level_type, raw_level = ifs_key.split("::")
        aliases = _ALIASES_BY_E2S.get(e2s, ()) + _ALIASES_BY_SHORT_NAME.get(
            short_name, ()
        )
        specs[e2s] = VariableSpec(
            e2s=e2s,
            short_name=short_name,
            param_id=_PARAM_IDS.get(short_name),
            standard_name=_STANDARD_NAMES.get(short_name, ""),
            level_type=_level_type(raw_level_type),
            level=_level(raw_level),
            canonical_units=_CANONICAL_UNITS_BY_SHORT_NAME.get(short_name, ""),
            aliases=aliases,
        )
    return specs


def normalize_units(units: str | None) -> str:
    """Normalize common CF/GRIB units spellings."""
    if not units:
        return ""
    u = units.strip().lower()
    u = u.replace("**", "").replace("^", "")
    u = u.replace(" ", "")
    if u in {"degree_celsius", "degreec", "degc", "celsius", "degreescelsius"}:
        return "degc"
    if u in {"k", "kelvin"}:
        return "k"
    if u in {"1", "(0-1)", "0-1", "fraction", "dimensionless"}:
        return "1"
    if u in {"%", "percent"}:
        return "percent"
    if u in {"mofwaterequivalent", "mwe"}:
        return "m"
    return u


def make_modifier(spec: VariableSpec, src_units: str | None) -> Callable:
    """Return a modifier that aligns source values to Earth2Studio units."""
    if spec.e2s == "cos_mwd":
        return lambda x: np.cos(np.deg2rad(x))
    if spec.e2s == "sin_mwd":
        return lambda x: np.sin(np.deg2rad(x))

    src = normalize_units(src_units)
    dst = normalize_units(spec.canonical_units)
    if src == dst or src == "" or dst == "":
        return lambda x: x

    if src == "degc" and dst == "k":
        return lambda x: x + 273.15
    if src == "k" and dst == "degc":
        return lambda x: x - 273.15

    if spec.short_name in {"gh", "z"} and src == "m" and dst == "m2s-2":
        return lambda x: x * GRAVITY
    if spec.short_name in {"gh", "z"} and src == "m2s-2" and dst == "m":
        return lambda x: x / GRAVITY

    if src == "percent" and dst == "1":
        return lambda x: x / 100.0
    if src == "1" and dst == "percent":
        return lambda x: x * 100.0

    if src == "kgm-2" and dst == "m":
        return lambda x: x / 1000.0
    if src == "m" and dst == "kgm-2":
        return lambda x: x * 1000.0

    return lambda x: x


class EarthMoverIFSLexicon(metaclass=LexiconType):
    """Earthmover Brightband IFS Marketplace lexicon."""

    VOCAB: dict[str, str] = {
        "u100m": "100u::sfc::",
        "v100m": "100v::sfc::",
        "u10m": "10u::sfc::",
        "v10m": "10v::sfc::",
        "d2m": "2d::sfc::",
        "t2m": "2t::sfc::",
        "cp": "cp::sfc::",
        "fdir": "fdir::sfc::",
        "hcc": "hcc::sfc::",
        "lcc": "lcc::sfc::",
        "mcc": "mcc::sfc::",
        "msl": "msl::sfc::",
        "sd": "sd::sfc::",
        "ssrd": "ssrd::sfc::",
        "tp": "tp::sfc::",
    }
    SPECS: dict[str, VariableSpec] = _build_specs(VOCAB)

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Retrieve name from vocabulary."""
        if val not in cls.VOCAB:
            raise KeyError(val)
        if val in IFSLexicon.VOCAB:
            return IFSLexicon.get_item(val)
        return cls.VOCAB[val], lambda x: x

    @classmethod
    def spec(cls, val: str) -> VariableSpec:
        """Return the Earthmover metadata descriptor for a variable."""
        return cls.SPECS[val]
