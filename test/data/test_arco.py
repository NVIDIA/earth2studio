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

import datetime
import pathlib
import shutil

import numpy as np
import pytest

from earth2studio.data import ARCO, ARCO_ERA5


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=1959, month=1, day=31),
        [
            datetime.datetime(year=1971, month=6, day=1, hour=6),
            datetime.datetime(year=2021, month=11, day=23, hour=12),
        ],
        np.array([np.datetime64("1993-04-05T00:00")]),
    ],
)
@pytest.mark.parametrize(
    "variable,disable_msc", [("tcwv", True), ("t136k", True), (["u500", "u200"], False)]
)
def test_arco_fetch(time, variable, disable_msc, monkeypatch):

    monkeypatch.setenv("EARTH2STUDIO_DISABLE_MSC", disable_msc)
    ds = ARCO_ERA5(cache=False)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime.datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == len(variable)
    assert shape[2] == 721
    assert shape[3] == 1440
    assert np.array_equal(data.coords["variable"].values, np.array(variable))
    assert not np.isnan(data.values).any()


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [np.array([np.datetime64("1993-04-05T00:00")])],
)
@pytest.mark.parametrize("variable", [["z500", "q200"]])
@pytest.mark.parametrize("cache", [True, False])
def test_arco_cache(time, variable, cache):

    ds = ARCO_ERA5(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 2
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    # Cache should be present
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cach or refetch
    data = ds(time, variable[0])
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 1
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.xfail
@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=1939, month=2, day=25),
        datetime.datetime(year=1, month=1, day=1, hour=13, minute=1),
        datetime.datetime.now(),
    ],
)
@pytest.mark.parametrize("variable", ["mpl"])
def test_arco_available(time, variable):
    assert not ARCO_ERA5.available(time)
    with pytest.raises(ValueError):
        ds = ARCO_ERA5()
        ds(time, variable)


def test_arco_deprecation_warning():
    """The legacy ARCO alias instantiates the ERA5 data source with a warning."""
    with pytest.warns(DeprecationWarning, match="ARCO.*will be removed"):
        data_source = ARCO()

    assert isinstance(data_source, ARCO_ERA5)
    assert isinstance(data_source, ARCO)
    assert ARCO.available(datetime.datetime(1939, 2, 25)) is False


@pytest.mark.timeout(30)
def test_arco_time_statistics_mock(mock_statistic_zarr):
    ds = ARCO_ERA5(cache=True, verbose=True)
    reads = mock_statistic_zarr(ds)
    time = datetime.datetime(2024, 1, 2)
    index = ds._get_time_index(time)
    variables = ["t2m", "t2m:mean:6h", "t2m:sum:6h", "t2m:max:6h", "t2m:min:+1h:+4h"]
    data = ds(time, variables)

    assert data.shape == (1, 5, 2, 2)
    values = [float(data.sel(variable=v).values[0, 0, 0]) for v in variables]
    # Window [-6h, 0h) covers hourly fields index-6 .. index-1
    assert values == [
        index,
        index - 3.5,
        6 * index - 21,
        index - 1,
        index + 1,
    ]
    assert set(data.attrs["earth2studio_statistics"]) == set(variables[1:])
    # Each distinct hourly field is read exactly once: -6h..+3h plus the instant
    assert sorted(reads) == list(range(index - 6, index + 4))

    # Plain variables share the request pool with statistics: the instantaneous
    # field overlapping a forward window is read once, not once per path
    reads.clear()
    data = ds(time, ["t2m", "t2m:mean:+0h:+2h"])
    np.testing.assert_allclose(data.values[0, :, 0, 0], [index, index + 0.5])
    assert sorted(reads) == [index, index + 1]

    # Two pressure levels of one hour (and a window over a third) are one chunk
    reads.clear()
    data = ds(time, ["t500", "t850", "t500:max:+0h:+2h"])
    # Level index 0 (500 hPa) is +0 and index 1 (850 hPa) is +1 in the fake chunk
    assert data.values[0, 0, 0, 0] == index
    assert data.values[0, 1, 0, 0] == index + 1
    assert data.values[0, 2, 0, 0] == index + 1
    assert sorted(reads) == [index, index + 1]

    # A trailing-sum variable (read directly) next to its base field still works
    reads.clear()
    data = ds(time, ["tp", "tp06"])
    assert data.values[0, 0, 0, 0] == index
    assert data.values[0, 1, 0, 0] == 6 * index - 15

    # Statistics of already-aggregated variables or bad windows are rejected
    with pytest.raises(ValueError):
        ds(time, ["t2m:mean:90m"])
