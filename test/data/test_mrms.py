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
    for v in variable:
        actual_time_coord = f"actual_time_{v}"
        assert actual_time_coord in data.coords
        assert data.coords[actual_time_coord].shape[0] == len(time)


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
def test_mrms_fetch_multivar(time):
    variable = ["refc", "refc_base"]
    ds = MRMS(cache=False, max_offset_minutes=15)
    data = ds(time, variable)
    shape = data.shape

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
    for v in variable:
        actual_time_coord = f"actual_time_{v}"
        assert actual_time_coord in data.coords
        assert data.coords[actual_time_coord].shape[0] == len(time)


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
@pytest.mark.timeout(120)
def test_mrms_corrupted_nearest_file_fallback():
    """Corrupted nearest file fallback behavior.

    The nearest MRMS file for refc_base around 2021-01-14T23:59:57 is known to
    be corrupted/truncated on the NOAA S3 bucket:

        s3://noaa-mrms-pds/CONUS/MergedBaseReflectivityQC_00.50/20210114/
        MRMS_MergedBaseReflectivityQC_00.50_20210114-235957.grib2.gz

    This file decompresses successfully but pygrib/eccodes cannot read the GRIB
    message (``RuntimeError: End of resource reached when reading message``).
    The GRIB2 payload is truncated: actual size 229376 bytes vs declared message
    length 912765, and it is missing the ``7777`` end-of-message marker.

    Step 1: Very tight tolerance — only the corrupted file matches, so the
    pipeline raises RuntimeError (no valid grid can be inferred).

    Step 2: Wider tolerance — _resolve_s3_time_candidates lists all nearby
    files. The corrupted one is tried first (nearest), skipped by _fetch_task,
    and the next-nearest valid candidate is used as a fallback. The returned
    data should contain real (non-NaN) values.
    """
    # Request the exact timestamp of the corrupted file so it is resolved first.
    t_corrupted = datetime(2021, 1, 14, 23, 59, 57)

    # --- Step 1: Very tight tolerance (0.1 min = 6 s). Only the corrupted file
    # matches; no valid fallback candidate exists.  fetch() cannot infer a
    # lat/lon grid from zero successful results, so it raises RuntimeError
    # ("All MRMS fetches failed; no data available.").
    ds_tight = MRMS(cache=False, max_offset_minutes=0.1)
    with pytest.raises(RuntimeError, match="All MRMS fetches failed"):
        ds_tight([t_corrupted], "refc_base")

    # --- Step 2: Wider tolerance (10 min). The corrupted file is resolved first
    # but _fetch_task should skip it and fall back to the next-nearest valid
    # candidate within the window.
    ds = MRMS(cache=False, max_offset_minutes=10)
    data = ds([t_corrupted], "refc_base")

    # Should still return valid shape
    assert data.shape[0] == 1
    assert data.shape[1] == 1
    assert "lat" in data.dims and "lon" in data.dims

    # The fallback candidate should have produced real data, not all NaN.
    values = data.isel(time=0).values
    assert not np.isnan(values).all(), (
        "Expected fallback to next-nearest valid file after skipping corrupted "
        "MRMS_MergedBaseReflectivityQC_00.50_20210114-235957.grib2.gz, "
        "but got all-NaN output."
    )

    # actual_time_refc_base should exist and differ from the corrupted timestamp
    assert "actual_time_refc_base" in data.coords
    assert data.coords["actual_time_refc_base"].values != np.datetime64(
        "1970-01-01T00:00:00"
    )


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
