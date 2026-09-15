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

import asyncio
import hashlib
import pathlib
import shutil
from datetime import datetime
from unittest.mock import MagicMock, patch

import h5py  # type: ignore
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


class _FakeListOnlyStore:
    """Minimal obspec List store for available() probes."""

    def __init__(self, entries):
        self._entries = entries

    def list(self, prefix=None, **kwargs):
        return iter([self._entries] if self._entries else [])


@pytest.mark.timeout(15)
def test_jpss_available_mock():
    """Test JPSS.available with a fake obspec store (no network)."""

    with patch(
        "earth2studio.data.jpss.obstore_store_from_url",
        return_value=_FakeListOnlyStore([{"path": "some_file.h5"}]),
    ):
        assert JPSS.available(
            datetime(2024, 6, 25, 12, 0, 0),
            variable="viirs01i",
            satellite="noaa-20",
            product_type="I",
        )


@pytest.mark.timeout(15)
def test_jpss_available_mock_data_gap():
    """Test JPSS.available returns False when the S3 listing is empty
    (obstore lists a missing prefix as empty rather than raising)."""

    with patch(
        "earth2studio.data.jpss.obstore_store_from_url",
        return_value=_FakeListOnlyStore([]),
    ):
        assert not JPSS.available(
            datetime(2024, 6, 25, 12, 0, 0),
            variable="viirs01i",
            satellite="noaa-20",
            product_type="I",
        )


class _FakeAsyncListStore:
    """Minimal obspec ListAsync store yielding bucket-relative keys."""

    def __init__(self, keys):
        self._keys = keys
        self.list_calls = 0

    def list_async(self, prefix=None, **kwargs):
        self.list_calls += 1
        matching = [k for k in self._keys if k.startswith(prefix or "")]

        async def _gen():
            if matching:
                yield [{"path": k} for k in matching]

        return _gen()


@pytest.mark.timeout(15)
def test_jpss_get_s3_path_mock():
    """Test _get_s3_path closest-file selection with a fake obspec store."""

    day_prefix = "VIIRS-I1-SDR/2024/06/25"
    near = (
        f"{day_prefix}/SVI01_j01_d20240625_t1200000_e1201242_b12345_"
        "c20240625121530000000_oebc_ops.h5"
    )
    far = (
        f"{day_prefix}/SVI01_j01_d20240625_t0600000_e0601242_b12340_"
        "c20240625061530000000_oebc_ops.h5"
    )
    fake = _FakeAsyncListStore([near, far])

    ds = JPSS(satellite="noaa-20", product_type="I", cache=False)
    ds.store = fake

    path, timestamp = asyncio.run(
        ds._get_s3_path(datetime(2024, 6, 25, 12, 0, 30), "viirs01i")
    )
    # Bucket-prefixed path of the closest-timestamp file
    assert path == f"noaa-nesdis-n20-pds/{near}"
    assert timestamp == datetime(2024, 6, 25, 12, 0, 0)

    # Second request over the same (past) day reuses the memoized listing
    path, timestamp = asyncio.run(
        ds._get_s3_path(datetime(2024, 6, 25, 7, 0, 0), "viirs01i")
    )
    assert path == f"noaa-nesdis-n20-pds/{far}"
    assert timestamp == datetime(2024, 6, 25, 6, 0, 0)
    assert fake.list_calls == 1

    # Missing day directory lists empty -> FileNotFoundError
    with pytest.raises(FileNotFoundError):
        asyncio.run(ds._get_s3_path(datetime(2023, 1, 1, 0, 0, 0), "viirs01i"))


@pytest.mark.timeout(30)
def test_jpss_call_mock(tmp_path: pathlib.Path):
    """Test full __call__ path with a fake obspec store (no network)."""

    bucket = "noaa-nesdis-n20-pds"
    small_dims = {"I": (2, 3), "M": (2, 3), "L2": (2, 3)}
    time_part = "d20240625_t1200000_e1201242_b12345"

    data_key = (
        f"VIIRS-I1-SDR/2024/06/25/SVI01_j01_{time_part}_"
        "c20240625121530000000_oebc_ops.h5"
    )
    geo_key = (
        f"VIIRS-IMG-GEO-TC/2024/06/25/GITCO_j01_{time_part}_"
        "c20240625121530000000_oebc_ops.h5"
    )

    # Write tiny fake VIIRS HDF5 granules
    radiance = np.array([[100, 200, 300], [400, 500, 600]], dtype=np.uint16)
    lat = np.linspace(10, 11, 6, dtype=np.float32).reshape(2, 3)
    lon = np.linspace(120, 121, 6, dtype=np.float32).reshape(2, 3)

    data_path = tmp_path / "data.h5"
    with h5py.File(data_path, "w") as f:
        f.create_dataset("All_Data/VIIRS-I1-SDR_All/Radiance", data=radiance)
    geo_path = tmp_path / "geo.h5"
    with h5py.File(geo_path, "w") as f:
        f.create_dataset("All_Data/VIIRS-IMG-GEO-TC_All/Latitude", data=lat)
        f.create_dataset("All_Data/VIIRS-IMG-GEO-TC_All/Longitude", data=lon)

    local_files = {data_key: data_path, geo_key: geo_path}

    class _FakeStore(_FakeAsyncListStore):
        async def get_async(self, path, *, options=None):
            data = local_files[path].read_bytes()

            class _Result:
                async def buffer_async(self):
                    return data

            return _Result()

    cache_dir = str(tmp_path / "cache")
    with (
        patch.object(JPSS, "PRODUCT_DIMENSIONS", small_dims),
        patch.object(JPSS, "cache", property(lambda self: cache_dir)),
    ):
        ds = JPSS(satellite="noaa-20", product_type="I", verbose=False)
        # Inject the fake store — no obstore internals need patching
        ds.store = _FakeStore(list(local_files))

        data = ds(datetime(2024, 6, 25, 12, 0, 0), "viirs01i")

    assert data.shape == (1, 3, 2, 3)
    assert list(data.coords["variable"].values) == ["viirs01i", "_lat", "_lon"]
    assert np.array_equal(data.values[0, 0], radiance.astype(np.float32))
    assert np.array_equal(data.values[0, 1], lat)
    assert np.array_equal(data.values[0, 2], lon)
    # Cache keys hash the historical bucket-prefixed path so pre-migration
    # warm caches remain valid
    for key in (data_key, geo_key):
        sha = hashlib.sha256(f"{bucket}/{key}".encode()).hexdigest()
        assert (pathlib.Path(cache_dir) / sha).is_file()


@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "scale_factor,add_offset,raw,expected",
    [
        (0.005, 200.0, 19292.0, 296.46),  # typical LST pixel
        (0.005, 200.0, np.nan, np.nan),  # fill value already NaN-masked
        (1.0, 0.0, 42.0, 42.0),  # identity scaling
        (None, None, 42.0, 42.0),  # no attrs — passthrough
        (2.0, None, 3.0, 6.0),  # scale only
        (None, 10.0, 3.0, 13.0),  # offset only
    ],
)
def test_jpss_cf_scaling(scale_factor, add_offset, raw, expected):
    """Verify _apply_cf_scaling preserves float32 dtype and applies the linear transform."""

    data = np.array([[raw]], dtype=np.float32)

    dataset = MagicMock()
    dataset.attrs.get = lambda key, default=None: (
        np.array([scale_factor])
        if key == "scale_factor" and scale_factor is not None
        else (
            np.array([add_offset])
            if key == "add_offset" and add_offset is not None
            else default
        )
    )

    result = JPSS._apply_cf_scaling(data, dataset)

    assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"
    if np.isnan(expected):
        assert np.isnan(result[0, 0])
    else:
        assert result[0, 0] == pytest.approx(expected, rel=1e-5)
