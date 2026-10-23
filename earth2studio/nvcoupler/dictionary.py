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

"""CF-style field dictionary: canonical standard names, units, and aliases.

The NUOPC field-dictionary analog. Connectors match exported to imported
fields by *standard name*, never by raw model variable strings; aliases map
model vocabularies (``z1000``, ``ws10m``) onto standard names. Derived
fields (e.g. a 48-hour mean) are first-class entries carrying a
:class:`CellMethod`, which lets ``couple()`` synthesize the right mediator
instead of string-parsing suffixes.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .errors import UnitsMismatchError, UnknownFieldError

# Small normalization table so cosmetically-different unit strings compare
# equal. v1 checks equality only; it never converts values.
#
# Kept behaviorally aligned with the lexicon normalizer in
# earth2studio/lexicon/earthmover.py (normalize_units), which lowercases and
# strips '**', '^', and spaces before matching — any two spellings that
# normalizer treats as equal (e.g. 'm s**-1', 'm s^-1', 'M/S'; '(0-1)',
# 'fraction', 'dimensionless'; 'degC', 'celsius'; '%', 'percent') compare
# equal here too. Not imported from the lexicon so nvcoupler stays importable
# standalone. Keys are in collapsed form (lowercase, no '**'/'^'/spaces).
_UNIT_SYNONYMS = {
    "m2/s2": "m2 s-2",
    "m2s-2": "m2 s-2",
    "m/s": "m s-1",
    "ms-1": "m s-1",
    "kelvin": "K",
    "k": "K",
    "pa": "Pa",
    "hpa": "hPa",
    "kg/m2": "kg m-2",
    "kgm-2": "kg m-2",
    "mm": "kg m-2",  # precipitation depth-equivalence
    "1": "",
    "(0-1)": "",
    "0-1": "",
    "fraction": "",
    "dimensionless": "",
    "degree_celsius": "degC",
    "degreec": "degC",
    "degc": "degC",
    "celsius": "degC",
    "degreescelsius": "degC",
    "%": "percent",
    "percent": "percent",
    "mofwaterequivalent": "m",
    "mwe": "m",
}


def normalize_units(units: str) -> str:
    """Normalize a unit string for comparison (no value conversion).

    Collapses case, '**'/'^' exponent markers, and whitespace exactly like
    ``earth2studio.lexicon.earthmover.normalize_units`` before applying the
    synonym table, so the two normalizers agree on overlapping inputs.
    """
    u = units.strip().lower().replace("**", "").replace("^", "").replace(" ", "")
    return _UNIT_SYNONYMS.get(u, u)


@dataclass(frozen=True)
class CellMethod:
    """CF-style cell method describing a derived (time-reduced) field.

    A field entry with a cell method declares "I am `method` of `base` over
    `window`" — e.g. the 48 h mean of geopotential_at_1000hpa. This is the
    machine-readable convention that lets auto-wiring insert the right
    AccumulationMediator between components of different cadence.
    """

    base: str
    method: Literal["mean", "sum", "max", "min"]
    window: np.timedelta64

    def __post_init__(self) -> None:
        if self.method not in ("mean", "sum", "max", "min"):
            raise ValueError(f"Unsupported cell method {self.method!r}")


@dataclass(frozen=True)
class FieldEntry:
    """One entry in the field dictionary."""

    standard_name: str
    canonical_units: str
    description: str = ""
    aliases: frozenset[str] = field(default_factory=frozenset)
    cell_method: CellMethod | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.aliases, frozenset):
            object.__setattr__(self, "aliases", frozenset(self.aliases))


class FieldDictionary:
    """Registry resolving standard names and aliases to :class:`FieldEntry`.

    Lookup is case-sensitive on standard names and aliases. An alias may map
    to exactly one standard name; a standard name may have many aliases.
    """

    def __init__(self, entries: "FieldDictionary | list[FieldEntry] | None" = None):
        self._entries: dict[str, FieldEntry] = {}
        self._aliases: dict[str, str] = {}
        if isinstance(entries, FieldDictionary):
            self._entries = dict(entries._entries)
            self._aliases = dict(entries._aliases)
        elif entries:
            for entry in entries:
                self.register(entry)

    def register(self, entry: FieldEntry) -> None:
        """Register an entry; re-registering a standard name replaces it."""
        if entry.standard_name in self._aliases:
            raise ValueError(
                f"{entry.standard_name!r} is already an alias for "
                f"{self._aliases[entry.standard_name]!r}"
            )
        self._entries[entry.standard_name] = entry
        for alias in entry.aliases:
            self.add_alias(entry.standard_name, alias)

    def add_alias(self, standard_name: str, alias: str) -> None:
        if standard_name not in self._entries:
            raise UnknownFieldError(standard_name, self._entries.keys())
        existing = self._aliases.get(alias)
        if existing is not None and existing != standard_name:
            raise ValueError(
                f"Alias {alias!r} already maps to {existing!r}; cannot remap "
                f"to {standard_name!r}"
            )
        if alias in self._entries and alias != standard_name:
            raise ValueError(f"Alias {alias!r} collides with a standard name")
        self._aliases[alias] = standard_name

    def resolve(self, name: str) -> FieldEntry:
        """Resolve a standard name or alias to its entry."""
        if name in self._entries:
            return self._entries[name]
        if name in self._aliases:
            return self._entries[self._aliases[name]]
        raise UnknownFieldError(
            name, list(self._entries.keys()) + list(self._aliases.keys())
        )

    def standard_name(self, name: str) -> str:
        return self.resolve(name).standard_name

    def __contains__(self, name: str) -> bool:
        return name in self._entries or name in self._aliases

    def standard_names(self) -> list[str]:
        return list(self._entries.keys())

    def check_units(
        self, standard_name: str, units: str, *, src: str, dst: str
    ) -> None:
        """Raise :class:`UnitsMismatchError` if units disagree with canonical."""
        canonical = self.resolve(standard_name).canonical_units
        if normalize_units(units) != normalize_units(canonical):
            raise UnitsMismatchError(standard_name, src, units, dst, canonical)

    def derived_from(self, standard_name: str) -> CellMethod | None:
        """Return the cell method if `standard_name` is a derived field."""
        return self.resolve(standard_name).cell_method


def _default_entries() -> list[FieldEntry]:
    """Curated v1 vocabulary covering the earth2studio surface variables and
    pressure-level fields used by the coupled models in this repo, plus the
    accumulation/impact fields the mediators produce."""
    e = FieldEntry
    hours = lambda h: np.timedelta64(h, "h")  # noqa: E731
    return [
        # surface / single-level
        e("sea_surface_temperature", "K", "SST", frozenset({"sst"})),
        e("air_temperature_2m", "K", "2 m air temperature", frozenset({"t2m"})),
        e("dewpoint_temperature_2m", "K", "2 m dewpoint", frozenset({"d2m"})),
        e("wind_speed_10m", "m s-1", "10 m wind speed", frozenset({"ws10m", "ws10"})),
        e("eastward_wind_10m", "m s-1", "10 m u-wind", frozenset({"u10m", "u10"})),
        e("northward_wind_10m", "m s-1", "10 m v-wind", frozenset({"v10m", "v10"})),
        e("surface_pressure", "Pa", "surface pressure", frozenset({"sp"})),
        e("mean_sea_level_pressure", "Pa", "MSLP", frozenset({"msl", "mslp"})),
        e("total_column_water_vapour", "kg m-2", "TCWV", frozenset({"tcwv"})),
        e(
            "total_precipitation_6h",
            "kg m-2",
            "6 h accumulated precip",
            frozenset({"tp06"}),
        ),
        # pressure-level (levels encoded in the name, earth2studio convention)
        e("geopotential_at_1000hpa", "m2 s-2", "z at 1000 hPa", frozenset({"z1000"})),
        e("geopotential_at_500hpa", "m2 s-2", "z at 500 hPa", frozenset({"z500"})),
        e("geopotential_at_250hpa", "m2 s-2", "z at 250 hPa", frozenset({"z250"})),
        e("air_temperature_at_850hpa", "K", "t at 850 hPa", frozenset({"t850"})),
        e(
            "geopotential_thickness_300_700hpa",
            "m2 s-2",
            "z300 - z700 thickness",
            frozenset({"tau300-700"}),
        ),
        # derived / windowed fields (CellMethod-carrying entries)
        e(
            "geopotential_at_1000hpa_48h_mean",
            "m2 s-2",
            "trailing 48 h mean of z1000",
            frozenset({"z1000-48H"}),
            CellMethod("geopotential_at_1000hpa", "mean", hours(48)),
        ),
        e(
            "wind_speed_10m_48h_mean",
            "m s-1",
            "trailing 48 h mean of 10 m wind speed",
            frozenset({"ws10-48H"}),
            CellMethod("wind_speed_10m", "mean", hours(48)),
        ),
        e(
            "total_precipitation_48h_sum",
            "kg m-2",
            "48 h accumulated precipitation",
            frozenset(),
            CellMethod("total_precipitation_6h", "sum", hours(48)),
        ),
        e(
            "air_temperature_2m_24h_max",
            "K",
            "24 h maximum 2 m temperature",
            frozenset(),
            CellMethod("air_temperature_2m", "max", hours(24)),
        ),
    ]


DEFAULT_DICTIONARY = FieldDictionary(_default_entries())
