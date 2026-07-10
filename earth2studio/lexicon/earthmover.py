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

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .base import LexiconType

GRAVITY = 9.80665


@dataclass(frozen=True)
class VariableSpec:
    """Earthmover descriptor for a dataset variable."""

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
    "blh": 159,
    "cape": 59,
    "cp": 143,
    "d": 155,
    "fdir": 228021,
    "fsr": 244,
    "fg10": 49,
    "gh": 129,
    "hcc": 188,
    "lcc": 186,
    "lsp": 142,
    "lsm": 172,
    "mcc": 187,
    "msl": 151,
    "pv": 60,
    "q": 133,
    "r": 157,
    "sd": 141,
    "sf": 144,
    "skt": 235,
    "slhf": 147,
    "snowc": 260038,
    "sot": 228139,
    "sp": 134,
    "ssr": 176,
    "ssrd": 169,
    "sst": 34,
    "stl1": 139,
    "stl2": 170,
    "stl3": 183,
    "stl4": 236,
    "str": 177,
    "strd": 175,
    "swvl1": 39,
    "swvl2": 40,
    "t": 130,
    "tcc": 164,
    "tcw": 136,
    "tcwv": 137,
    "tisr": 212,
    "tp": 228,
    "tprate": 260048,
    "tsr": 178,
    "u": 131,
    "v": 132,
    "vo": 138,
    "vsw": 39,
    "w": 135,
    "z": 129,
    "zust": 228003,
}

_STANDARD_NAMES: dict[str, str] = {
    "100u": "eastward_wind",
    "100v": "northward_wind",
    "10fg": "wind_speed_of_gust",
    "10u": "eastward_wind",
    "10v": "northward_wind",
    "2d": "dew_point_temperature",
    "2t": "air_temperature",
    "blh": "atmosphere_boundary_layer_thickness",
    "cape": "atmosphere_convective_available_potential_energy",
    "cp": "lwe_thickness_of_convective_precipitation_amount",
    "d": "divergence_of_wind",
    "gh": "geopotential_height",
    "lsp": "lwe_thickness_of_stratiform_precipitation_amount",
    "lsm": "land_binary_mask",
    "msl": "air_pressure_at_mean_sea_level",
    "pv": "ertel_potential_vorticity",
    "q": "specific_humidity",
    "r": "relative_humidity",
    "sd": "lwe_thickness_of_surface_snow_amount",
    "sf": "lwe_thickness_of_snowfall_amount",
    "skt": "surface_temperature",
    "slhf": "surface_upward_latent_heat_flux",
    "sp": "surface_air_pressure",
    "sst": "sea_surface_temperature",
    "stl1": "surface_temperature",
    "stl2": "surface_temperature",
    "stl3": "surface_temperature",
    "stl4": "surface_temperature",
    "ssrd": "surface_downwelling_shortwave_flux_in_air",
    "strd": "surface_downwelling_longwave_flux_in_air",
    "t": "air_temperature",
    "tcc": "cloud_area_fraction",
    "tcw": "atmosphere_mass_content_of_water",
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
    "blh": "m",
    "cape": "J kg-1",
    "cp": "m",
    "d": "s-1",
    "fdir": "J m-2",
    "fsr": "m",
    "gh": "m2 s-2",
    "hcc": "1",
    "lcc": "1",
    "lsp": "m",
    "lsm": "1",
    "mcc": "1",
    "msl": "Pa",
    "pv": "K m2 kg-1 s-1",
    "q": "kg kg-1",
    "r": "%",
    "sd": "m",
    "sf": "kg m-2",
    "skt": "K",
    "slhf": "J m-2",
    "snowc": "%",
    "sot": "K",
    "sp": "Pa",
    "ssr": "J m-2",
    "ssrd": "J m-2",
    "sst": "K",
    "stl1": "K",
    "stl2": "K",
    "stl3": "K",
    "stl4": "K",
    "str": "J m-2",
    "strd": "J m-2",
    "swvl1": "m3 m-3",
    "swvl2": "m3 m-3",
    "t": "K",
    "tcc": "1",
    "tcw": "kg m-2",
    "tcwv": "kg m-2",
    "tisr": "J m-2",
    "tp": "m",
    "tprate": "kg m-2 s-1",
    "tsr": "J m-2",
    "u": "m s-1",
    "v": "m s-1",
    "vo": "s-1",
    "vsw": "m3 m-3",
    "w": "Pa s-1",
    "z": "m2 s-2",
    "zust": "m s-1",
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

_METADATA_SHORT_NAME_ALIASES: dict[str, str] = {
    "d2m": "2d",
    "fg10": "10fg",
    "t2m": "2t",
    "u10": "10u",
    "u100": "100u",
    "v10": "10v",
    "v100": "100v",
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
        metadata_short_name = _METADATA_SHORT_NAME_ALIASES.get(short_name, short_name)
        aliases = _ALIASES_BY_E2S.get(e2s, ()) + _ALIASES_BY_SHORT_NAME.get(
            short_name, ()
        )
        if metadata_short_name != short_name:
            aliases += (metadata_short_name,)
        specs[e2s] = VariableSpec(
            e2s=e2s,
            short_name=short_name,
            param_id=_PARAM_IDS.get(metadata_short_name),
            standard_name=_STANDARD_NAMES.get(metadata_short_name, ""),
            level_type=_level_type(raw_level_type),
            level=_level(raw_level),
            canonical_units=_CANONICAL_UNITS_BY_SHORT_NAME.get(metadata_short_name, ""),
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


_IFS_IC_PRESSURE_LEVELS = (
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
)


class _EarthMoverLexiconBase(metaclass=LexiconType):
    """Common Earthmover Arraylake lexicon helpers."""

    VOCAB: dict[str, str]
    SPECS: dict[str, VariableSpec]

    @classmethod
    def get_item(cls, val: str) -> tuple[str, Callable]:
        """Retrieve name from vocabulary."""
        if val not in cls.VOCAB:
            raise KeyError(val)
        return cls.VOCAB[val], lambda x: x

    @classmethod
    def spec(cls, val: str) -> VariableSpec:
        """Return the Earthmover metadata descriptor for a variable."""
        return cls.SPECS[val]


class EarthMoverIFSInitialConditionLexicon(_EarthMoverLexiconBase):
    """Earthmover Brightband IFS initial-condition Marketplace lexicon."""

    PRESSURE_LEVELS = _IFS_IC_PRESSURE_LEVELS
    SURFACE_VARIABLES: dict[str, str] = {
        "u100m": "100u::sfc::",
        "v100m": "100v::sfc::",
        "u10m": "10u::sfc::",
        "v10m": "10v::sfc::",
        "d2m": "2d::sfc::",
        "t2m": "2t::sfc::",
        "hcc": "hcc::sfc::",
        "lcc": "lcc::sfc::",
        "mcc": "mcc::sfc::",
        "msl": "msl::sfc::",
        "skt": "skt::sfc::",
        "sp": "sp::sfc::",
        "sst": "sst::sfc::",
        "stl1": "stl1::sfc::",
        "stl2": "stl2::sfc::",
        "swvl1": "swvl1::sfc::",
        "swvl2": "swvl2::sfc::",
        "tcc": "tcc::sfc::",
        "tcw": "tcw::sfc::",
        "tcwv": "tcwv::sfc::",
    }
    PRESSURE_VARIABLES: dict[str, str] = {
        f"{name}{level}": f"{name}::pl::{level}"
        for name in ("q", "t", "u", "v", "w", "z")
        for level in _IFS_IC_PRESSURE_LEVELS
    }
    VOCAB: dict[str, str] = {
        **SURFACE_VARIABLES,
        **PRESSURE_VARIABLES,
    }
    SPECS: dict[str, VariableSpec] = _build_specs(VOCAB)


class EarthMoverIFSLexicon(_EarthMoverLexiconBase):
    """Earthmover Brightband IFS forecast Marketplace lexicon."""

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


_ERA5_LEVELS = (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
_ERA5_PRESSURE_VARIABLES = ("pv", "q", "r", "t", "u", "v", "w", "z")


class EarthMoverERA5Lexicon(_EarthMoverLexiconBase):
    """Earthmover ERA5 Marketplace lexicon."""

    PRESSURE_LEVELS = _ERA5_LEVELS
    SURFACE_VARIABLES: dict[str, str] = {
        "blh": "blh::sfc::",
        "cape": "cape::sfc::",
        "cp": "cp::sfc::",
        "d2m": "d2m::sfc::",
        "fdir": "fdir::sfc::",
        "fg10m": "fg10::sfc::",
        "fsr": "fsr::sfc::",
        "hcc": "hcc::sfc::",
        "ie": "ie::sfc::",
        "lcc": "lcc::sfc::",
        "lsp": "lsp::sfc::",
        "mcc": "mcc::sfc::",
        "msl": "msl::sfc::",
        "sd": "sd::sfc::",
        "sf": "sf::sfc::",
        "skt": "skt::sfc::",
        "slhf": "slhf::sfc::",
        "sp": "sp::sfc::",
        "ssr": "ssr::sfc::",
        "ssrd": "ssrd::sfc::",
        "sst": "sst::sfc::",
        "stl1": "stl1::sfc::",
        "stl2": "stl2::sfc::",
        "stl3": "stl3::sfc::",
        "stl4": "stl4::sfc::",
        "swvl1": "swvl1::sfc::",
        "t2m": "t2m::sfc::",
        "tcc": "tcc::sfc::",
        "tcw": "tcw::sfc::",
        "tcwv": "tcwv::sfc::",
        "tisr": "tisr::sfc::",
        "tp": "tp::sfc::",
        "tsr": "tsr::sfc::",
        "u10m": "u10::sfc::",
        "u100m": "u100::sfc::",
        "v10m": "v10::sfc::",
        "v100m": "v100::sfc::",
        "zust": "zust::sfc::",
    }
    PRESSURE_VARIABLES: dict[str, str] = {
        f"{name}{level}": f"{name}::pl::{level}"
        for name in _ERA5_PRESSURE_VARIABLES
        for level in _ERA5_LEVELS
    }
    VOCAB: dict[str, str] = {
        **SURFACE_VARIABLES,
        **PRESSURE_VARIABLES,
    }
    SPECS: dict[str, VariableSpec] = _build_specs(VOCAB)
