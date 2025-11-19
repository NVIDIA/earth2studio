# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pathlib
import shutil
from datetime import datetime

import numpy as np
import pytest

from earth2studio.data import CMIP6
from earth2studio.data.cmip6 import CMIP6MultiRealm


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "table_id, variable",
    [
        pytest.param("Amon", ["u10m", "v10m"], id="atmos_multi"),
        pytest.param("Omon", ["sst"], id="ocean_sst"),
    ],
)
@pytest.mark.parametrize(
    "time",
    [
        pytest.param(datetime(2015, 1, 16), id="single_stamp"),
        pytest.param(
            [datetime(2015, 1, 15), datetime(2015, 1, 17)],
            id="two_stamps",
        ),
        pytest.param(
            datetime(2014, 1, 15),
            id="non_matching_timespan",
        ),
    ],
)
def test_cmip6_fetch(table_id, variable, time):
    """Generic CMIP6 fetch test covering both atmospheric and ocean tables."""
    ds = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id=table_id,
        variant_label="r1i1p2f1",
    )

    data = ds(time, variable)
    shape = data.shape

    if isinstance(time, datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "table_id, variable",
    [
        pytest.param("Omon", ["sst"]),
    ],
)
@pytest.mark.parametrize(
    "time",
    [
        pytest.param([datetime(2015, 1, 16)]),
    ],
)
@pytest.mark.parametrize("cache", [True, False])
def test_cmip6_cache(table_id, variable, time, cache):

    ds = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id=table_id,
        variant_label="r1i1p2f1",
        cache=cache,
    )
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 291
    assert shape[3] == 360
    # Cache should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 291
    assert shape[3] == 360

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


def test_cmip6_init_valid():
    """CMIP6 initialisation with typical parameters should succeed."""

    ds = CMIP6(
        experiment_id="historical",
        source_id="MPI-ESM1-2-LR",
        table_id="Amon",
        variant_label="r1i1p1f1",
    )

    # Attributes should be stored as-is
    assert ds.experiment_id == "historical"
    assert ds.source_id == "MPI-ESM1-2-LR"
    assert ds.table_id == "Amon"
    assert ds.variant_label == "r1i1p1f1"


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "variable, expected_exc",
    [
        pytest.param("not_a_variable", KeyError, id="invalid_name"),
        pytest.param("sst", ValueError, id="ocean_var_on_day_table"),
        pytest.param("t550", KeyError, id="pressure_level_missing"),
    ],
)
def test_cmip6_input(variable, expected_exc):
    """Trigger specific validation errors inside the CMIP6 datasource."""

    ds = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="day",  # daily atmospheric table
        variant_label="r1i1p2f1",
    )

    with pytest.raises(expected_exc):
        _ = ds(datetime(2015, 1, 16), variable)


@pytest.mark.parametrize(
    "time, source_id, expected",
    [
        pytest.param(datetime(2015, 1, 16), "CanESM5", True, id="exists"),
        pytest.param(datetime(1800, 1, 1), "CanESM5", False, id="too_early"),
        pytest.param(datetime(2015, 1, 16), "NonExistentModel", False, id="bad_model"),
    ],
)
def test_cmip6_available(time, source_id, expected):
    """Test the CMIP6.available class-method."""

    result = CMIP6.available(
        time,
        experiment_id="ssp585",
        source_id=source_id,
        table_id="day",
        variant_label="r1i1p2f1",
    )

    assert result is expected


def test_cmip6_multi_realm_empty_list():
    """Test that empty source list raises ValueError."""
    with pytest.raises(ValueError, match="cannot be empty"):
        CMIP6MultiRealm([])


def test_cmip6_multi_realm_invalid_type():
    """Test that non-CMIP6 instances in list raise TypeError."""
    with pytest.raises(TypeError, match="not a CMIP6 instance"):
        CMIP6MultiRealm(["not_a_cmip6_source"])


def test_cmip6_multi_realm_available_variables():
    """Test that available_variables returns union of all sources."""
    atmos = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Amon",
        variant_label="r1i1p2f1",
    )

    ocean = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Omon",
        variant_label="r1i1p2f1",
    )

    multi = CMIP6MultiRealm([atmos, ocean])

    # available_variables should be union of both sources
    assert isinstance(multi.available_variables, set)
    assert len(multi.available_variables) > 0
    # Should contain variables from both atmosphere and ocean
    assert multi.available_variables == (
        atmos.available_variables | ocean.available_variables
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    "time, expected",
    [
        pytest.param(datetime(2015, 1, 16), True, id="available"),
        pytest.param(datetime(1800, 1, 1), False, id="too_early"),
    ],
)
@pytest.mark.xfail(
    reason="available() downloads large amounts of data to check timestamp availability"
)
def test_cmip6_multi_realm_available(time, expected):
    """Test the CMIP6MultiRealm.available class-method.

    Note: This test uses daily datasets (day and SIday) to ensure timestamp alignment.
    Marked as xfail because available() downloads significant data (multiple GB).
    """
    atmos = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="day",
        variant_label="r1i1p2f1",
    )

    # Use daily sea ice data (SIday) to ensure timestamp alignment with day
    sea_ice = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="SIday",
        variant_label="r1i1p2f1",
    )

    sources = [atmos, sea_ice]
    result = CMIP6MultiRealm.available(time, sources)

    assert result is expected


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(90)
def test_cmip6_multi_realm_basic():
    """Test combining atmospheric and ocean sources."""
    # Atmospheric source
    atmos = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Amon",
        variant_label="r1i1p2f1",
    )

    # Ocean source
    ocean = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Omon",
        variant_label="r1i1p2f1",
    )

    multi = CMIP6MultiRealm([atmos, ocean])

    time = datetime(2015, 1, 16)
    variables = ["u10m", "sst"]  # one from each realm

    data = multi(time, variables)

    assert data.shape[0] == 1  # time
    assert data.shape[1] == 2  # variables
    assert np.array_equal(data.coords["variable"].values, np.array(variables))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(90)
def test_cmip6_multi_realm_regridding():
    """Test that curvilinear ocean grid is regridded to atmospheric grid."""
    atmos = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Amon",
        variant_label="r1i1p2f1",
    )

    ocean = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Omon",
        variant_label="r1i1p2f1",
    )

    multi = CMIP6MultiRealm([atmos, ocean])

    time = datetime(2015, 1, 16)
    variables = ["u10m", "sst"]

    data = multi(time, variables)

    # Both variables should be on same grid (atmospheric grid)
    assert "lat" in data.coords
    assert "lon" in data.coords
    assert "_lat" not in data.coords  # curvilinear coords should be gone
    assert "_lon" not in data.coords

    # Both variables should have same spatial dimensions
    assert data.isel(variable=0).shape == data.isel(variable=1).shape


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(90)
def test_cmip6_multi_realm_variable_priority():
    """Test that variables are fetched from first available source."""
    atmos1 = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Amon",
        variant_label="r1i1p2f1",
    )

    atmos2 = CMIP6(
        experiment_id="historical",
        source_id="CanESM5",
        table_id="Amon",
        variant_label="r1i1p2f1",
    )

    multi = CMIP6MultiRealm([atmos1, atmos2])

    time = datetime(2015, 1, 16)
    variables = ["u10m"]  # available in both

    data = multi(time, variables)

    # Should succeed and only fetch from first source
    assert data.shape[0] == 1
    assert data.shape[1] == 1


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(90)
def test_cmip6_multi_realm_with_sea_ice():
    """Test combining atmospheric and sea ice sources."""
    atmos = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Amon",
        variant_label="r1i1p2f1",
    )

    sea_ice = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="SImon",
        variant_label="r1i1p2f1",
    )

    multi = CMIP6MultiRealm([atmos, sea_ice])

    time = datetime(2015, 1, 16)
    variables = ["t2m", "siconc"]  # temperature and sea ice concentration

    data = multi(time, variables)

    assert data.shape[0] == 1
    assert data.shape[1] == 2
    # Should be on regular grid
    assert "lat" in data.coords
    assert "lon" in data.coords


# Tests for exact_time_match feature


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_cmip6_exact_time_match_fails_on_mismatch():
    """Test that exact_time_match=True raises ValueError when time doesn't match exactly."""
    ds = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Amon",
        variant_label="r1i1p2f1",
        exact_time_match=True,
    )

    # Request a time that's definitely outside the dataset range of ssp585
    time = datetime(1800, 1, 1)

    with pytest.raises(ValueError, match="Exact time match required"):
        _ = ds(time, ["u10m"])


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_cmip6_exact_time_match_succeeds_on_match():
    """Test that exact_time_match=True succeeds when time matches exactly."""
    # First fetch with default to discover exact timestamp
    ds_default = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Amon",
        variant_label="r1i1p2f1",
        exact_time_match=False,
        verbose=False,
    )

    # Get approximate time
    approx_time = datetime(2015, 1, 16)
    data = ds_default(approx_time, ["u10m"])
    actual_time = data.coords["time"].values[0]

    # Now use exact matching with discovered time
    ds_exact = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Amon",
        variant_label="r1i1p2f1",
        exact_time_match=True,
        verbose=False,
    )

    data_exact = ds_exact(actual_time, ["u10m"])
    assert data_exact.shape[0] == 1
    assert data_exact.shape[1] == 1


# Tests for CMIP6MultiRealm exact_time_match validation


def test_cmip6_multi_realm_mismatched_exact_time_match():
    """Test that CMIP6MultiRealm raises error when sources have different exact_time_match settings."""
    atmos = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Amon",
        variant_label="r1i1p2f1",
        exact_time_match=False,
    )

    ocean = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Omon",
        variant_label="r1i1p2f1",
        exact_time_match=True,  # Different!
    )

    with pytest.raises(
        ValueError,
        match="All CMIP6 sources must have the same exact_time_match setting",
    ):
        CMIP6MultiRealm([atmos, ocean])


@pytest.mark.parametrize(
    "exact_time_match",
    [
        pytest.param(False, id="exact_false"),
        pytest.param(True, id="exact_true"),
    ],
)
def test_cmip6_multi_realm_consistent_exact_time_match(exact_time_match):
    """Test that CMIP6MultiRealm succeeds when all sources have same exact_time_match setting."""
    atmos = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Amon",
        variant_label="r1i1p2f1",
        exact_time_match=exact_time_match,
    )

    ocean = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Omon",
        variant_label="r1i1p2f1",
        exact_time_match=exact_time_match,  # Same!
    )

    # Should not raise
    multi = CMIP6MultiRealm([atmos, ocean])
    assert multi is not None


# Tests for CMIP6MultiRealm missing variable validation


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_cmip6_multi_realm_partial_missing_variables():
    """Test that CMIP6MultiRealm raises error when some requested variables are not found.

    Tests the case where some variables are successfully found but others are missing.
    The error should list which variables were not found.
    """
    atmos = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Amon",
        variant_label="r1i1p2f1",
    )

    ocean = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Omon",
        variant_label="r1i1p2f1",
    )

    multi = CMIP6MultiRealm([atmos, ocean])

    time = datetime(2015, 1, 16)
    # Mix of valid variables (u10m from atmos, sst from ocean) and invalid (nonexistent_var)
    variables = ["u10m", "sst", "nonexistent_var"]

    with pytest.raises(
        ValueError, match="not found in any of the provided CMIP6 sources"
    ):
        _ = multi(time, variables)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_cmip6_multi_realm_all_missing_variables():
    """Test that CMIP6MultiRealm raises error when none of the requested variables exist."""
    atmos = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Amon",
        variant_label="r1i1p2f1",
    )

    ocean = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Omon",
        variant_label="r1i1p2f1",
    )

    multi = CMIP6MultiRealm([atmos, ocean])

    time = datetime(2015, 1, 16)
    # Request only variables that don't exist
    variables = ["fake_var1", "fake_var2"]

    with pytest.raises(ValueError, match="None of the requested variables"):
        _ = multi(time, variables)
