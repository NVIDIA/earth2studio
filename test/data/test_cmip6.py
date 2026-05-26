# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
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
import xarray as xr

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
        pytest.param("sst", IndexError, id="ocean_var_on_day_table"),
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


def test_cmip6_pressure_level_tolerance():

    class DummyCatalog:
        def __init__(self, dataset: xr.Dataset):
            self._dataset = dataset

        def to_dataset_dict(self, *args, **kwargs):
            return {"ta": self._dataset}

    time = np.array([np.datetime64("2000-01-01")])
    lat = np.array([-10.0, 10.0])
    lon = np.array([0.0, 90.0])
    plev = np.array([50000.5])  # Pa, slightly offset from 50000
    data = np.ones((1, 1, len(lat), len(lon)), dtype=np.float32)

    fake_ds = xr.Dataset(
        {"ta": (("time", "plev", "lat", "lon"), data)},
        coords={"time": time, "plev": plev, "lat": lat, "lon": lon},
    )

    ds = CMIP6.__new__(CMIP6)
    ds.experiment_id = "test"
    ds.source_id = "test"
    ds.table_id = "Amon"
    ds.variant_label = "r1i1p1f1"
    ds.file_start = None
    ds.file_end = None
    ds._cache = True
    ds._verbose = False
    ds._exact_time_match = False
    ds.available_variables = {"ta"}
    ds.catalog = DummyCatalog(fake_ds)
    ds._search_catalog = lambda *args, **kwargs: None  # type: ignore[assignment]

    ds._pressure_level_tolerance = 0
    with pytest.raises(KeyError):
        _ = ds(datetime(2000, 1, 1), ["t500"])

    ds._pressure_level_tolerance = 1
    data_ok = ds(datetime(2000, 1, 1), ["t500"])
    assert data_ok.shape == (1, 1, len(lat), len(lon))


@pytest.mark.xfail
@pytest.mark.timeout(90)
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
        table_id="Amon",
        variant_label="r1i1p2f1",
    )

    assert result is expected


def test_cmip6_multi_realm_validation_errors():
    """Test CMIP6MultiRealm validation errors in a single test."""
    with pytest.raises(ValueError, match="cannot be empty"):
        CMIP6MultiRealm([])

    with pytest.raises(TypeError, match="not a CMIP6 instance"):
        CMIP6MultiRealm(["not_a_cmip6_source"])

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


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
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
@pytest.mark.xfail
@pytest.mark.parametrize(
    "time, expected",
    [
        pytest.param(datetime(2015, 1, 16), True, id="available"),
        pytest.param(datetime(1800, 1, 1), False, id="too_early"),
    ],
)
def test_cmip6_multi_realm_available(time, expected):
    """Test the CMIP6MultiRealm.available class-method.

    Note: This test uses daily datasets (day and SIday) to ensure timestamp alignment.
    Marked as xfail because available() downloads significant data (multiple GB).
    """
    atmos = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Amon",
        variant_label="r1i1p2f1",
    )

    # Use daily sea ice data (SIday) to ensure timestamp alignment with day
    sea_ice = CMIP6(
        experiment_id="ssp585",
        source_id="CanESM5",
        table_id="Omon",
        variant_label="r1i1p2f1",
    )

    sources = [atmos, sea_ice]
    result = CMIP6MultiRealm.available(time, sources)

    assert result is expected


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_cmip6_multi_realm():
    """Combine multi-realm tests to reduce repeated downloads."""
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

    time = datetime(2015, 1, 16)

    # Basic combine + regridding checks
    multi = CMIP6MultiRealm([atmos, ocean])
    variables = ["u10m", "sst"]
    data = multi(time, variables)
    assert data.shape[0] == 1  # time
    assert data.shape[1] == 2  # variables
    assert np.array_equal(data.coords["variable"].values, np.array(variables))
    assert "lat" in data.coords
    assert "lon" in data.coords
    assert "_lat" not in data.coords
    assert "_lon" not in data.coords
    assert data.isel(variable=0).shape == data.isel(variable=1).shape

    # Missing variable
    variables_missing_partial = ["u10m", "sst", "nonexistent_var"]
    with pytest.raises(KeyError):
        _ = multi(time, variables_missing_partial)

    # Variable priority (separate sources, same test to avoid extra startup)
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
    multi_priority = CMIP6MultiRealm([atmos1, atmos2])
    data_priority = multi_priority(time, ["u10m"])
    assert data_priority.shape[0] == 1
    assert data_priority.shape[1] == 1


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
