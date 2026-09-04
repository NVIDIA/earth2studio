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

"""YAML config round-trip tests on the synthetic coupled system."""

import numpy as np
import pytest
import yaml

from earth2studio.nvcoupler.clock import Clock, as_timedelta
from earth2studio.nvcoupler.config import from_yaml, to_yaml
from earth2studio.nvcoupler.dictionary import (
    DEFAULT_DICTIONARY,
    CellMethod,
    FieldDictionary,
    FieldEntry,
)
from earth2studio.nvcoupler.driver import Driver
from earth2studio.nvcoupler.errors import CouplingError
from earth2studio.nvcoupler.mediator import (
    AccumulationMediator,
    TrailingAverageMediator,
)
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

T0 = "2024-01-01"
T96 = "2024-01-05"

DSL = """
@6h
  atmos -> med
  ocean -> atmos
  atmos
@48h
  med.compute
  med -> ocean
  ocean
@
"""


def make_tagged_driver():
    """The test_driver.py toy system, with yaml_spec-tagged toy components."""
    atmos = fake_atmos(gain=1.0)
    atmos.yaml_spec = {
        "class": "earth2studio.nvcoupler.testing.fake_atmos",
        "kwargs": {"gain": 1.0, "timestep": "6h"},
    }
    ocean = fake_ocean(gain=1.0)
    ocean.yaml_spec = {
        "class": "earth2studio.nvcoupler.testing.fake_ocean",
        "kwargs": {"gain": 1.0, "timestep": "48h"},
    }
    med = TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"])
    return Driver(
        {"atmos": atmos, "ocean": ocean, "med": med}, DSL, Clock(T0, T96, "6h")
    )


def run_all(driver):
    driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
    return driver.run()


def test_round_trip_identical_outputs(tmp_path):
    driver = make_tagged_driver()
    path = tmp_path / "system.yaml"
    text = to_yaml(driver, path=path)
    assert path.read_text() == text

    rebuilt = from_yaml(path)
    assert sorted(rebuilt.components) == sorted(driver.components)
    assert str(rebuilt.sequence) == str(driver.sequence)
    assert rebuilt.clock.start == driver.clock.start
    assert rebuilt.clock.stop == driver.clock.stop
    assert rebuilt.clock.dt == driver.clock.dt

    ds_a = run_all(driver)
    ds_b = run_all(rebuilt)
    for comp in ("atmos", "ocean", "med"):
        for var in ds_a[comp].data_vars:
            assert np.array_equal(ds_a[comp][var].values, ds_b[comp][var].values)
    # sanity anchor from the hand-computed driver tests
    assert np.allclose(
        ds_b["atmos"]["geopotential_at_1000hpa"].values[-1], 19.2336, atol=1e-4
    )


def test_from_yaml_accepts_text_and_path(tmp_path):
    driver = make_tagged_driver()
    text = to_yaml(driver)
    rebuilt = from_yaml(text)  # text, not a path
    assert sorted(rebuilt.components) == ["atmos", "med", "ocean"]


def test_closure_component_without_yaml_spec_raises():
    driver = make_tagged_driver()
    del driver.components["atmos"].yaml_spec
    with pytest.raises(CouplingError, match="yaml_spec"):
        to_yaml(driver)


def test_mediator_window_serialized_as_string():
    driver = make_tagged_driver()
    doc = yaml.safe_load(to_yaml(driver))
    window = doc["components"]["med"]["kwargs"]["window"]
    assert isinstance(window, str)
    assert as_timedelta(window) == np.timedelta64(48, "h")


def test_custom_dictionary_entry_round_trip():
    """A non-default derived entry (12 h mean, np.timedelta64 window) survives
    the YAML round-trip as a '12h'-style string."""
    dictionary = FieldDictionary(DEFAULT_DICTIONARY)
    dictionary.register(
        FieldEntry(
            "geopotential_at_1000hpa_12h_mean",
            "m2 s-2",
            "trailing 12 h mean of z1000",
            frozenset({"z12m"}),
            CellMethod("geopotential_at_1000hpa", "mean", np.timedelta64(12, "h")),
        )
    )
    atmos = fake_atmos()
    atmos.yaml_spec = {
        "class": "earth2studio.nvcoupler.testing.fake_atmos",
        "kwargs": {"gain": 1.0, "timestep": "6h"},
    }
    med = AccumulationMediator(
        "med", ["geopotential_at_1000hpa_12h_mean"], dictionary=dictionary
    )
    dsl = "@6h\n  atmos -> med\n  atmos\n@12h\n  med.compute\n@"
    driver = Driver({"atmos": atmos, "med": med}, dsl, Clock(T0, "2024-01-02", "6h"))
    # the toy atmos intentionally runs on constant IC forcing (its SST import
    # is fed by nothing in this two-component system)
    driver.allow_unfed_imports = True
    text = to_yaml(driver)
    doc = yaml.safe_load(text)
    entries = {e["standard_name"]: e for e in doc["dictionary"]}
    entry = entries["geopotential_at_1000hpa_12h_mean"]
    assert isinstance(entry["cell_method"]["window"], str)
    assert as_timedelta(entry["cell_method"]["window"]) == np.timedelta64(12, "h")
    # default entries stay out of the file
    assert "geopotential_at_1000hpa" not in entries

    rebuilt = from_yaml(text)
    rebuilt.allow_unfed_imports = True
    med2 = rebuilt.components["med"]
    assert med2.timestep == np.timedelta64(12, "h").astype("timedelta64[ns]")
    driver.initialize({"atmos": atmos_ic()})
    rebuilt.initialize({"atmos": atmos_ic()})
    ds_a, ds_b = driver.run(), rebuilt.run()
    assert np.array_equal(
        ds_a["med"]["geopotential_at_1000hpa_12h_mean"].values,
        ds_b["med"]["geopotential_at_1000hpa_12h_mean"].values,
    )


def test_add_alias_round_trip():
    """Aliases added via FieldDictionary.add_alias() after registration must
    survive the YAML round-trip even though the FieldEntry itself still
    equals the default (regression: only non-default entries were dumped)."""
    dictionary = FieldDictionary(DEFAULT_DICTIONARY)
    dictionary.add_alias("geopotential_at_1000hpa", "phi1000")
    atmos = fake_atmos()
    atmos.yaml_spec = {
        "class": "earth2studio.nvcoupler.testing.fake_atmos",
        "kwargs": {"gain": 1.0, "timestep": "6h"},
    }
    med = TrailingAverageMediator(
        "med", ["geopotential_at_1000hpa_48h_mean"], dictionary=dictionary
    )
    dsl = "@6h\n  atmos -> med\n  atmos\n@48h\n  med.compute\n@"
    driver = Driver({"atmos": atmos, "med": med}, dsl, Clock(T0, "2024-01-03", "6h"))

    text = to_yaml(driver)
    doc = yaml.safe_load(text)
    assert doc["aliases"] == {"phi1000": "geopotential_at_1000hpa"}
    # the default entry itself stays out of the dictionary section
    assert not any(
        e["standard_name"] == "geopotential_at_1000hpa"
        for e in doc.get("dictionary", [])
    )

    rebuilt = from_yaml(text)
    med2 = rebuilt.components["med"]
    assert med2.dictionary.standard_name("phi1000") == "geopotential_at_1000hpa"
    # and the round-trip is stable: dumping again re-emits the alias
    # (re-tag atmos: closures never carry yaml_spec across from_yaml)
    rebuilt.components["atmos"].yaml_spec = atmos.yaml_spec
    doc2 = yaml.safe_load(to_yaml(rebuilt))
    assert doc2["aliases"] == {"phi1000": "geopotential_at_1000hpa"}


def test_connectors_round_trip():
    from earth2studio.nvcoupler.connector import Connector

    driver = make_tagged_driver()
    atmos, med = driver.components["atmos"], driver.components["med"]
    driver._connectors[("atmos", "med")] = Connector(
        atmos, med, fields=["geopotential_at_1000hpa"], time_policy="constant"
    )
    doc = yaml.safe_load(to_yaml(driver))
    assert doc["connectors"] == [
        {
            "src": "atmos",
            "dst": "med",
            "time_policy": "constant",
            "fill": "none",
            "fields": ["geopotential_at_1000hpa"],
        }
    ]
    rebuilt = from_yaml(to_yaml(driver))
    conn = rebuilt._connectors[("atmos", "med")]
    assert conn._fields == ["geopotential_at_1000hpa"]


def make_tagged_declarative_driver():
    """The toy system declared as a graph (derived sequence, windowed
    connector for the 48 h mean) with yaml_spec-tagged components."""
    from earth2studio.nvcoupler.api import couple

    atmos = fake_atmos(gain=1.0)
    atmos.yaml_spec = {
        "class": "earth2studio.nvcoupler.testing.fake_atmos",
        "kwargs": {"gain": 1.0, "timestep": "6h"},
    }
    ocean = fake_ocean(gain=1.0)
    ocean.yaml_spec = {
        "class": "earth2studio.nvcoupler.testing.fake_ocean",
        "kwargs": {"gain": 1.0, "timestep": "48h"},
    }
    return couple(atmos, ocean, start=T0, stop=T96)


def test_derived_sequence_round_trip():
    driver = make_tagged_declarative_driver()
    assert driver.sequence_derived
    text = to_yaml(driver)
    doc = yaml.safe_load(text)
    # derived sequences serialize with the flag plus the (informational) text
    assert doc["sequence"]["derived"] is True
    assert doc["sequence"]["text"] == str(driver.sequence)
    # the windowed connector carries its reduction options
    conn_doc = next(
        c for c in doc["connectors"] if (c["src"], c["dst"]) == ("atmos", "ocean")
    )
    assert as_timedelta(conn_doc["window"]) == np.timedelta64(48, "h")
    assert conn_doc["reduce"] == "mean"

    rebuilt = from_yaml(text)
    assert rebuilt.sequence_derived  # re-derived, not replayed
    assert str(rebuilt.sequence) == str(driver.sequence)
    conn = rebuilt._connectors[("atmos", "ocean")]
    assert conn.window == np.timedelta64(48, "h").astype("timedelta64[ns]")
    assert conn.reduce == "mean"

    ds_a = run_all(driver)
    ds_b = run_all(rebuilt)
    for comp in ("atmos", "ocean"):
        for var in ds_a[comp].data_vars:
            assert np.array_equal(ds_a[comp][var].values, ds_b[comp][var].values)
    assert np.allclose(
        ds_b["atmos"]["geopotential_at_1000hpa"].values[-1], 19.2336, atol=1e-4
    )
    assert np.allclose(
        ds_b["ocean"]["sea_surface_temperature"].values[-1], 2.180147, atol=1e-4
    )


def test_explicit_sequence_still_serializes_as_text():
    doc = yaml.safe_load(to_yaml(make_tagged_driver()))
    assert isinstance(doc["sequence"], str)
    assert "@6h" in doc["sequence"]


def test_bad_sequence_mapping_raises():
    text = to_yaml(make_tagged_declarative_driver())
    bad = text.replace("derived: true", "derived: false")
    with pytest.raises(CouplingError, match="derived"):
        from_yaml(bad)


def test_helpful_error_on_bad_import_path():
    text = to_yaml(make_tagged_driver()).replace(
        "earth2studio.nvcoupler.testing.fake_atmos", "no.such.module.fake"
    )
    with pytest.raises(CouplingError, match="no.such.module"):
        from_yaml(text)
