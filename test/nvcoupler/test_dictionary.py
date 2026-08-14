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

import numpy as np
import pytest

from earth2studio.nvcoupler.dictionary import (
    DEFAULT_DICTIONARY,
    CellMethod,
    FieldDictionary,
    FieldEntry,
    normalize_units,
)
from earth2studio.nvcoupler.errors import UnitsMismatchError, UnknownFieldError


def test_alias_resolution():
    entry = DEFAULT_DICTIONARY.resolve("z1000")
    assert entry.standard_name == "geopotential_at_1000hpa"
    assert entry.canonical_units == "m2 s-2"
    # standard name resolves to itself
    assert DEFAULT_DICTIONARY.standard_name("geopotential_at_1000hpa") == (
        "geopotential_at_1000hpa"
    )


def test_unknown_name_suggestions():
    with pytest.raises(UnknownFieldError) as err:
        DEFAULT_DICTIONARY.resolve("z1000h")
    assert "z1000" in str(err.value)
    assert "register" in str(err.value).lower()


def test_register_and_alias():
    d = FieldDictionary(DEFAULT_DICTIONARY)
    d.register(FieldEntry("sea_ice_fraction", "", aliases=frozenset({"sic"})))
    assert d.standard_name("sic") == "sea_ice_fraction"
    # remapping an alias to a different standard name is an error
    with pytest.raises(ValueError):
        d.add_alias("sea_surface_temperature", "sic")
    # alias colliding with a standard name is an error
    with pytest.raises(ValueError):
        d.add_alias("sea_surface_temperature", "sea_ice_fraction")
    # copy did not pollute the default dictionary
    assert "sic" not in DEFAULT_DICTIONARY


def test_units_check():
    DEFAULT_DICTIONARY.check_units(
        "geopotential_at_1000hpa", "m**2 s**-2", src="a", dst="b"
    )  # synonym normalizes, no raise
    with pytest.raises(UnitsMismatchError) as err:
        DEFAULT_DICTIONARY.check_units(
            "geopotential_at_1000hpa", "K", src="ocean", dst="atmos"
        )
    assert "ocean" in str(err.value) and "K" in str(err.value)


def test_normalize_units():
    assert normalize_units("m/s") == "m s-1"
    assert normalize_units("Kelvin") == "K"
    # unknown units pass through in collapsed comparison form
    assert normalize_units("W m-2") == normalize_units("w m**-2")


def test_normalize_units_agrees_with_earthmover_lexicon():
    """nvcoupler's normalizer must not disagree with the earthmover lexicon
    normalizer (earth2studio/lexicon/earthmover.py): every pair of spellings
    the lexicon treats as equivalent compares equal here too."""
    equal_pairs = [
        ("m s**-1", "m s-1"),
        ("m s^-1", "m s-1"),
        ("M/S", "m s**-1"),
        ("m2 s-2", "m**2 s**-2"),
        ("m^2/s^2", "m2s-2"),
        ("(0-1)", "dimensionless"),
        ("0-1", "fraction"),
        ("1", "Dimensionless"),
        ("%", "percent"),
        ("degC", "celsius"),
        ("degree_celsius", "Degrees Celsius"),
        ("m of water equivalent", "mwe"),
        ("KELVIN", "k"),
    ]
    for a, b in equal_pairs:
        assert normalize_units(a) == normalize_units(b), (a, b)

    # cross-check directly against the lexicon normalizer: anything it
    # equates, we equate (nvcoupler may equate more, e.g. 'm/s' vs 'm s-1')
    from earth2studio.lexicon.earthmover import (
        normalize_units as lexicon_normalize_units,
    )

    for a, b in equal_pairs:
        if lexicon_normalize_units(a) == lexicon_normalize_units(b):
            assert normalize_units(a) == normalize_units(b), (a, b)

    # distinct units stay distinct
    assert normalize_units("K") != normalize_units("degC")
    assert normalize_units("Pa") != normalize_units("hPa")


def test_cell_method_derived_field():
    entry = DEFAULT_DICTIONARY.resolve("geopotential_at_1000hpa_48h_mean")
    cm = entry.cell_method
    assert cm is not None
    assert cm.base == "geopotential_at_1000hpa"
    assert cm.method == "mean"
    assert cm.window == np.timedelta64(48, "h")
    # non-derived fields have none
    assert DEFAULT_DICTIONARY.derived_from("sea_surface_temperature") is None
    with pytest.raises(ValueError):
        CellMethod("x", "median", np.timedelta64(1, "h"))
