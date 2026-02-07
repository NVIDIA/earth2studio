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

import pathlib
import shutil
from datetime import datetime

import numpy as np
import pytest

from earth2studio.data import JPSS


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "satellite,product_type,time,variable",
    [
        (
            "noaa-20",
            "I",
            datetime(year=2024, month=6, day=25, hour=12, minute=0, second=0),
            "viirs01i",
        ),
        (
            "noaa-20",
            "M",
            datetime(year=2024, month=6, day=25, hour=12, minute=0, second=0),
            ["viirs02m", "viirs03m"],
        ),
        (
            "noaa-21",
            "I",
            [
                datetime(year=2024, month=6, day=25, hour=12, minute=0, second=0),
                datetime(year=2024, month=6, day=25, hour=12, minute=1, second=0),
            ],
            "viirs05i",
        ),
        (
            "snpp",
            "L2",
            datetime(year=2025, month=6, day=25, hour=12, minute=0, second=0),
            "lst",
        ),
    ],
)
def test_jpss_fetch(satellite, product_type, time, variable):
    """Test JPSS data fetching across satellites, product types, and variable formats."""

    ds = JPSS(satellite=satellite, product_type=product_type, cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable_list = [variable]
    else:
        variable_list = list(variable)

    if isinstance(time, datetime):
        time_list = [time]
    else:
        time_list = list(time)

    expected_dims = JPSS.PRODUCT_DIMENSIONS[product_type]

    assert shape[0] == len(time_list)
    assert shape[1] == len(variable_list) + 2  # include _lat and _lon
    assert shape[2] == expected_dims[0]
    assert shape[3] == expected_dims[1]

    expected_variables = variable_list + ["_lat", "_lon"]
    assert np.array_equal(data.coords["variable"].values, np.array(expected_variables))

    assert JPSS.available(
        time_list[0],
        variable=variable_list[0],
        satellite=satellite,
        product_type=product_type,
    )


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize("cache", [True, False])
def test_jpss_cache(cache):
    """Test JPSS caching behavior for both enabled and disabled cache options."""

    ds = JPSS(satellite="noaa-20", product_type="M", cache=cache)
    time = np.array([np.datetime64("2024-06-25T12:00:00")])
    variable = ["viirs01m", "viirs02m", "viirs03m"]

    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == len(variable) + 2
    expected_dims = JPSS.PRODUCT_DIMENSIONS["M"]
    assert shape[2] == expected_dims[0]
    assert shape[3] == expected_dims[1]
    assert JPSS.available(
        time[0], variable=variable[0], satellite="noaa-20", product_type="M"
    )

    # Cache directory should exist only when caching is enabled
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Re-fetch single variable (should include geolocation variables)
    single_variable = variable[0]
    data = ds(time, single_variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == len([single_variable]) + 2
    assert shape[2] == expected_dims[0]
    assert shape[3] == expected_dims[1]

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "satellite,product_type,time,variable,valid",
    [
        (
            "noaa-20",
            "I",
            datetime(year=2024, month=6, day=25, hour=12, minute=0, second=0),
            "viirs01i",
            True,
        ),
        (
            "noaa-21",
            "M",
            datetime(year=2024, month=6, day=25, hour=12, minute=0, second=0),
            "viirs02m",
            True,
        ),
        (
            "snpp",
            "L2",
            datetime(year=2025, month=6, day=25, hour=12, minute=0, second=0),
            "lst",
            True,
        ),
        (
            "foo",
            "I",
            datetime(year=2024, month=6, day=25, hour=12, minute=0, second=0),
            "viirs01i",
            False,
        ),
        (
            "noaa-20",
            "X",
            datetime(year=2024, month=6, day=25, hour=12, minute=0, second=0),
            "viirs01i",
            False,
        ),
    ],
)
def test_jpss_sources(satellite, product_type, time, variable, valid):
    """Test JPSS data source initialization and basic fetching with different parameters."""

    if not valid:
        with pytest.raises(ValueError):
            JPSS(satellite=satellite, product_type=product_type, cache=False)
        return

    ds = JPSS(satellite=satellite, product_type=product_type, cache=False)
    data = ds(time, variable)
    shape = data.shape

    expected_dims = JPSS.PRODUCT_DIMENSIONS[product_type]

    assert shape[0] == 1
    assert shape[1] == len([variable]) + 2
    assert shape[2] == expected_dims[0]
    assert shape[3] == expected_dims[1]


@pytest.mark.timeout(15)
def test_jpss_invalid_variable():
    """Ensure requesting unknown JPSS variables raises a ValueError before fetching."""

    ds = JPSS(satellite="noaa-20", product_type="I", cache=False)

    with pytest.raises(ValueError):
        ds(
            [datetime(year=2024, month=6, day=25, hour=12, minute=0, second=0)],
            "invalid_variable",
        )


@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        datetime(2010, 1, 1, 0, 0, 0),
        datetime(2025, 6, 25, 12, 0, 0),
    ],
)
def test_jpss_available(time):
    """Test JPSS availability checks across satellites and product types."""

    if time < datetime(2012, 1, 1, 0, 0, 0):
        assert not JPSS.available(
            time, variable="viirs01i", satellite="noaa-20", product_type="I"
        )
        assert not JPSS.available(
            time, variable="viirs01i", satellite="noaa-21", product_type="I"
        )
        assert not JPSS.available(
            time, variable="viirs01i", satellite="snpp", product_type="I"
        )
    else:
        assert JPSS.available(
            time, variable="viirs01i", satellite="noaa-20", product_type="I"
        )
        assert JPSS.available(
            time, variable="viirs02m", satellite="noaa-21", product_type="M"
        )
        assert JPSS.available(time, variable="lst", satellite="snpp", product_type="L2")


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "satellite,product_type,variable,expected_error",
    [
        ("invalid-satellite", "I", "viirs01i", "Invalid satellite"),
        ("noaa-20", "invalid-product", "viirs01i", "Invalid product_type"),
        ("foo", "M", "viirs02m", "Invalid satellite"),
        ("noaa-21", "X", "viirs01i", "Invalid product_type"),
        ("noaa-20", "I", "invalid_variable", "Unknown VIIRS variables"),
    ],
)
def test_jpss_available_invalid_parameters(
    satellite, product_type, variable, expected_error
):
    """Test that JPSS.available raises appropriate errors for invalid parameters."""

    time = datetime(2024, 6, 25, 12, 0, 0)

    with pytest.raises(ValueError, match=expected_error):
        JPSS.available(
            time, variable=variable, satellite=satellite, product_type=product_type
        )
