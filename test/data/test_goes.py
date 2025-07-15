# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

from earth2studio.data import GOES


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "satellite,time,variable,scan_mode",
    [
        # GOES-16: Operational from Dec 18, 2017 to Apr 7, 2025
        (
            "goes16",
            datetime(year=2022, month=6, day=25, hour=12, minute=0, second=0),
            "vis047",
            "F",
        ),
        (
            "goes16",
            datetime(year=2022, month=6, day=25, hour=12, minute=0, second=0),
            ["vis064", "nir086"],
            "C",
        ),
        (
            "goes16",
            [
                datetime(year=2022, month=6, day=25, hour=12, minute=0, second=0),
                datetime(year=2022, month=6, day=25, hour=12, minute=10, second=0),
            ],
            "vis047",
            "C",
        ),
        # GOES-17: Operational from Feb 12, 2019 to Jan 4, 2023
        (
            "goes17",
            datetime(year=2022, month=6, day=25, hour=12, minute=0, second=0),
            "vis047",
            "F",
        ),
        (
            "goes17",
            datetime(year=2022, month=6, day=25, hour=12, minute=0, second=0),
            ["vis064", "nir086"],
            "C",
        ),
        (
            "goes17",
            [
                datetime(year=2022, month=6, day=25, hour=12, minute=0, second=0),
                datetime(year=2022, month=6, day=25, hour=12, minute=5, second=0),
            ],
            "vis047",
            "C",
        ),
        # GOES-18: Operational from Jan 4, 2023 onwards
        (
            "goes18",
            datetime(year=2023, month=6, day=25, hour=12, minute=0, second=0),
            "vis047",
            "F",
        ),
        (
            "goes18",
            datetime(year=2023, month=6, day=25, hour=12, minute=0, second=0),
            ["vis064", "nir086"],
            "C",
        ),
        (
            "goes18",
            [
                datetime(year=2023, month=6, day=25, hour=12, minute=0, second=0),
                datetime(year=2023, month=6, day=25, hour=12, minute=10, second=0),
            ],
            "vis047",
            "C",
        ),
        # GOES-19: Operational from Apr 7, 2025 onwards (future date for testing)
        (
            "goes19",
            datetime(year=2025, month=6, day=25, hour=12, minute=0, second=0),
            "vis047",
            "F",
        ),
        (
            "goes19",
            datetime(year=2025, month=6, day=25, hour=12, minute=0, second=0),
            ["vis064", "nir086"],
            "C",
        ),
        (
            "goes19",
            [
                datetime(year=2025, month=6, day=25, hour=12, minute=0, second=0),
                datetime(year=2025, month=6, day=25, hour=12, minute=10, second=0),
            ],
            "vis047",
            "C",
        ),
    ],
)
def test_goes_fetch(satellite, time, variable, scan_mode):
    """Test GOES data fetching for all satellites and scan modes with valid dates."""

    ds = GOES(satellite=satellite, scan_mode=scan_mode, cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime):
        time = [time]

    # Expected dimensions based on scan mode
    expected_dims = GOES.SCAN_DIMENSIONS[scan_mode]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == expected_dims[0]  # x dimension
    assert shape[3] == expected_dims[1]  # y dimension
    assert GOES.available(time[0], satellite=satellite, scan_mode=scan_mode)
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2022-06-25T12:00:00")]),
    ],
)
@pytest.mark.parametrize("variable", [["vis047", "vis064", "nir086"]])
@pytest.mark.parametrize("cache", [True, False])
def test_goes_cache(time, variable, cache):
    """Test GOES caching functionality."""

    ds = GOES(satellite="goes16", scan_mode="C", cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 3
    assert shape[2] == 1500  # C scan mode x dimension
    assert shape[3] == 2500  # C scan mode y dimension
    assert GOES.available(time[0], satellite="goes16", scan_mode="C")
    # Cache should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cache or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 1500
    assert shape[3] == 2500
    assert GOES.available(time[0], satellite="goes16", scan_mode="C")

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "satellite,scan_mode,time,variable,valid",
    [
        ("goes16", "F", datetime(2022, 6, 25, 12, 0, 0), "vis047", True),
        ("goes17", "C", datetime(2022, 6, 25, 12, 0, 0), "vis064", True),
        ("foo", "F", datetime(2022, 6, 25, 12, 0, 0), "vis047", False),
        ("goes16", "X", datetime(2022, 6, 25, 12, 0, 0), "vis047", False),
        ("goes16", "X", datetime(2022, 6, 25, 12, 0, 0), "foo", False),
    ],
)
def test_goes_sources(satellite, scan_mode, time, variable, valid):
    """Test GOES data source initialization with different parameters."""

    if not valid:
        with pytest.raises(ValueError):
            ds = GOES(satellite=satellite, scan_mode=scan_mode, cache=False)
        return

    ds = GOES(satellite=satellite, scan_mode=scan_mode, cache=False)
    data = ds(time, variable)
    shape = data.shape

    expected_dims = GOES.SCAN_DIMENSIONS[scan_mode]

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == expected_dims[0]
    assert shape[3] == expected_dims[1]


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        datetime(2015, 12, 31, 0, 0, 0),
        datetime(2020, 6, 25, 12, 5, 30),
        datetime(2025, 6, 25, 12, 0, 0),
    ],
)
@pytest.mark.parametrize("variable", ["vis047"])
def test_goes_available(time, variable):
    """Test GOES availability checks."""

    # Test availability check
    if time < datetime(2017, 1, 1, 0, 0, 0):
        assert not GOES.available(time, satellite="goes16", scan_mode="F")
        assert not GOES.available(time, satellite="goes17", scan_mode="F")
        assert not GOES.available(time, satellite="goes18", scan_mode="F")
        assert not GOES.available(time, satellite="goes19", scan_mode="F")
    elif time > datetime(2025, 6, 25, 12, 0, 0):
        assert not GOES.available(time, satellite="goes16", scan_mode="F")
        assert not GOES.available(
            time, satellite="goes17", scan_mode="F"
        )  # GOES-17 is not available after 2023-01-04
        assert GOES.available(time, satellite="goes18", scan_mode="F")
        assert GOES.available(time, satellite="goes19", scan_mode="F")

    # Test that invalid times raise ValueError
    with pytest.raises(ValueError):
        ds = GOES(satellite="goes16", scan_mode="F")
        ds([time], variable)


@pytest.mark.parametrize(
    "satellite,scan_mode,expected_shape",
    [
        ("goes16", "F", (5424, 5424)),
        ("goes16", "C", (1500, 2500)),
        ("goes17", "F", (5424, 5424)),
        ("goes17", "C", (1500, 2500)),
        ("goes18", "F", (5424, 5424)),
        ("goes18", "C", (1500, 2500)),
        ("goes19", "F", (5424, 5424)),
        ("goes19", "C", (1500, 2500)),
    ],
)
def test_goes_grid(satellite, scan_mode, expected_shape):
    """Test GOES grid method returns correct lat/lon coordinates."""

    lat, lon = GOES.grid(satellite=satellite, scan_mode=scan_mode)

    # Check shapes match expected dimensions
    assert lat.shape == expected_shape
    assert lon.shape == expected_shape
