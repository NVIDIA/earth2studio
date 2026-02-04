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
from datetime import datetime, timedelta

import numpy as np
import pytest

from earth2studio.data import MRMS


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        datetime(year=2024, month=7, day=1, hour=0, minute=0, second=0),
        [
            datetime(year=2020, month=10, day=31, hour=0, minute=0, second=0),
            datetime(year=2024, month=7, day=1, hour=0, minute=30, second=0),
        ],
    ],
)
@pytest.mark.parametrize("variable", ["refc", ["refc"]])
def test_mrms_fetch(time, variable):
    ds = MRMS(cache=False, max_offset_minutes=15)
    data = ds(time, variable)
    shape = data.shape

    if isinstance(variable, str):
        variable = [variable]

    if isinstance(time, datetime):
        time = [time]

    # Check basic dims/time/variable
    assert shape[0] == len(time)
    assert shape[1] == len(variable)

    # Must include lat/lon coords and dims
    assert "lat" in data.dims
    assert "lon" in data.dims
    assert "lat" in data.coords
    assert "lon" in data.coords

    # Variables coord matches request
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize("max_offset_minutes", [5, 10, 15])
def test_mrms_time_tolerance(max_offset_minutes):
    """Requested time may not match an exact file; ensure resolution within tolerance."""
    request_time = datetime(2024, 7, 1, 0, 0, 17)
    ds = MRMS(cache=False, max_offset_minutes=max_offset_minutes)
    data = ds(request_time, "refc")
    resolved = data["time"].isel(time=0).values.astype("datetime64[s]")
    req = np.datetime64(request_time, "s")
    diff = np.abs(resolved - req).astype("timedelta64[s]")
    assert diff <= np.timedelta64(max_offset_minutes, "m")


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2024-07-01T00:00:00")]),
    ],
)
@pytest.mark.parametrize("variable", [["refc"]])
@pytest.mark.parametrize("cache", [True, False])
def test_mrms_cache(time, variable, cache):
    ds = MRMS(cache=cache, max_offset_minutes=10)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == len(variable)
    assert "lat" in data.dims and "lon" in data.dims

    # Cache presence matches flag
    assert pathlib.Path(ds.cache).is_dir() == cache

    # Load from cache or refetch with single variable
    data = ds(time, variable[0])
    shape = data.shape
    assert shape[0] == 1
    assert shape[1] == 1
    assert "lat" in data.dims and "lon" in data.dims

    try:
        shutil.rmtree(ds.cache)
    except FileNotFoundError:
        pass


@pytest.mark.timeout(15)
def test_mrms_available():
    # Out-of-bounds times should not be available
    assert not MRMS.available(np.datetime64("2019-01-01T00:00:00"), variable="refc")
    assert not MRMS.available(datetime.now() + timedelta(days=1), variable="refc")
    assert MRMS.available(datetime(2023, 8, 23, 12, 0, 0), variable="refc")

    # And attempting to fetch should raise ValueError
    with pytest.raises(ValueError):
        ds = MRMS(cache=False)
        ds([datetime(2019, 1, 1, 0, 0, 0)], "refc")


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(60)
def test_mrms_partial_missing_time_nan_fill():
    # https://noaa-mrms-pds.s3.amazonaws.com/index.html#CONUS/MergedReflectivityComposite_00.50/20251107/
    # MRMS_MergedReflectivityComposite_00.50_20251107-000040.grib2.gz
    t_valid = datetime(2025, 11, 7, 0, 0, 35)
    t_missing = datetime(2025, 11, 7, 0, 1, 10)  # This will fail with tolerance 0.1

    ds = MRMS(cache=False, max_offset_minutes=0.1)
    data = ds([t_valid, t_missing], "refc")

    # Expect two times and one variable
    assert data.shape[0] == 2
    assert data.shape[1] == 1
    assert "lat" in data.dims and "lon" in data.dims

    # First time should contain some real values (not all NaN)
    first_values = data.isel(time=0).values
    assert not np.isnan(first_values).all()

    # Second time should be entirely NaN-filled
    second_values = data.isel(time=1).values
    assert np.isnan(second_values).all()
