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
import pathlib
import shutil
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr

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
            "abi01c",
            "F",
        ),
        (
            "goes16",
            datetime(year=2022, month=6, day=25, hour=12, minute=0, second=0),
            ["abi02c", "abi03c"],
            "C",
        ),
        (
            "goes16",
            [
                datetime(year=2022, month=6, day=25, hour=12, minute=0, second=0),
                datetime(year=2022, month=6, day=25, hour=12, minute=10, second=0),
            ],
            "abi01c",
            "C",
        ),
        # GOES-17: Operational from Feb 12, 2019 to Jan 4, 2023
        (
            "goes17",
            datetime(year=2022, month=6, day=25, hour=12, minute=0, second=0),
            "abi01c",
            "F",
        ),
        (
            "goes17",
            datetime(year=2022, month=6, day=25, hour=12, minute=0, second=0),
            ["abi02c", "abi03c"],
            "C",
        ),
        (
            "goes17",
            [
                datetime(year=2022, month=6, day=25, hour=12, minute=0, second=0),
                datetime(year=2022, month=6, day=25, hour=12, minute=5, second=0),
            ],
            "abi01c",
            "C",
        ),
        # GOES-18: Operational from Jan 4, 2023 onwards
        (
            "goes18",
            datetime(year=2023, month=6, day=25, hour=12, minute=0, second=0),
            "abi01c",
            "F",
        ),
        (
            "goes18",
            datetime(year=2023, month=6, day=25, hour=12, minute=0, second=0),
            ["abi02c", "abi03c"],
            "C",
        ),
        (
            "goes18",
            [
                datetime(year=2023, month=6, day=25, hour=12, minute=0, second=0),
                datetime(year=2023, month=6, day=25, hour=12, minute=10, second=0),
            ],
            "abi01c",
            "C",
        ),
        # GOES-19: Operational from Apr 7, 2025 onwards (future date for testing)
        (
            "goes19",
            datetime(year=2025, month=6, day=25, hour=12, minute=0, second=0),
            "abi01c",
            "F",
        ),
        (
            "goes19",
            datetime(year=2025, month=6, day=25, hour=12, minute=0, second=0),
            ["abi02c", "abi03c"],
            "C",
        ),
        (
            "goes19",
            [
                datetime(year=2025, month=6, day=25, hour=12, minute=0, second=0),
                datetime(year=2025, month=6, day=25, hour=12, minute=10, second=0),
            ],
            "abi01c",
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
@pytest.mark.parametrize("variable", [["abi01c", "abi02c", "abi03c"]])
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
        ("goes16", "F", datetime(2022, 6, 25, 12, 0, 0), "abi01c", True),
        ("goes17", "C", datetime(2022, 6, 25, 12, 0, 0), "abi02c", True),
        ("foo", "F", datetime(2022, 6, 25, 12, 0, 0), "abi01c", False),
        ("goes16", "X", datetime(2022, 6, 25, 12, 0, 0), "abi01c", False),
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


@pytest.mark.xfail
@pytest.mark.timeout(15)
@pytest.mark.parametrize(
    "time",
    [
        datetime(2015, 12, 31, 0, 0, 0),
        datetime(2020, 6, 25, 12, 5, 30),
        datetime(2025, 6, 25, 12, 0, 0),
    ],
)
@pytest.mark.parametrize("variable", ["abi01c"])
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


# A plain fake satisfying the obspec ListAsync protocol — no obstore
# internals need patching, the store is simply injected.
class _FakeListStore:
    def __init__(self):
        self.calls = 0

    def list_async(self, prefix=None, **kwargs):
        self.calls += 1

        async def _gen():
            yield [{"path": f"{prefix}OR_ABI-L2-MCMIPC_fake.nc"}]

        return _gen()


def test_goes_list_hour_files_memoization():
    # Complete (past) hours are listed once and memoized; an incomplete hour
    # is re-listed on every call so new scans are discovered
    ds = GOES(satellite="goes16", scan_mode="C", cache=False)
    fake = _FakeListStore()
    ds.store = fake

    # Complete (past) hour: one LIST request, then served from the memo
    past = datetime(2024, 6, 1, 18, 0, 0)
    files = asyncio.run(ds._list_hour_files(past))
    assert files == [f"noaa-goes16/{ds._hour_prefix(past)}OR_ABI-L2-MCMIPC_fake.nc"]
    assert asyncio.run(ds._list_hour_files(past)) == files
    assert fake.calls == 1
    assert ds._hour_prefix(past) in ds._hour_listing_cache

    # An incomplete hour: re-listed on every call, never memoized. The next
    # hour is used rather than the current one so the assertion cannot flake
    # when the wall-clock hour rolls over mid-test.
    incomplete = datetime.now(timezone.utc).replace(
        minute=0, second=0, microsecond=0, tzinfo=None
    ) + timedelta(hours=1)
    asyncio.run(ds._list_hour_files(incomplete))
    asyncio.run(ds._list_hour_files(incomplete))
    assert fake.calls == 3
    assert ds._hour_prefix(incomplete) not in ds._hour_listing_cache


def test_goes_fetch_no_gather_timeout(monkeypatch):
    # fetch() must not pass a gather-level task_timeout: it would wrap the
    # whole fetch_wrapper (including async_retry's retry loop) in a wait_for
    # of the same magnitude as the per-attempt timeout, cancelling retries on
    # the first slow attempt
    from earth2studio.data.utils import gather_with_concurrency

    ds = GOES(satellite="goes16", scan_mode="C", cache=False, verbose=False)
    ds.store = _FakeListStore()

    seen_kwargs = {}

    async def spy_gather(coros, **kwargs):
        seen_kwargs.update(kwargs)
        return await gather_with_concurrency(coros, **kwargs)

    monkeypatch.setattr("earth2studio.data.goes.gather_with_concurrency", spy_gather)

    async def fake_fetch_array(time, variable):
        return np.zeros((len(variable), *GOES.SCAN_DIMENSIONS["C"]))

    monkeypatch.setattr(ds, "fetch_array", fake_fetch_array)

    out = asyncio.run(ds.fetch([datetime(2024, 6, 1, 18, 0, 0)], ["abi01c"]))
    assert out.shape == (1, 1, *GOES.SCAN_DIMENSIONS["C"])
    assert seen_kwargs.get("task_timeout") is None


def test_goes_fetch_array_on_disk_mask(monkeypatch, tmp_path):
    """fetch_array should warn about NaNs at on-disk pixels and ignore NaNs
    off the Earth disk, which are always fill-valued.

    The missing pixel here also has a fill-valued ``DQF``, which is how NOAA
    writes pixels inside a missing scan wedge: keying the mask off "finite
    DQF == navigated" would drop exactly this pixel and report nothing.
    """
    monkeypatch.setitem(GOES.SCAN_DIMENSIONS, "C", (2, 3))

    # (0,0) good; (0,1) on-disk but missing radiance -> should be warned about;
    # (0,2) good; (1,0)/(1,2) off-disk -> excluded even though NaN; (1,1) good.
    cmi = np.array([[1.0, np.nan, 3.0], [np.nan, 5.0, np.nan]], dtype=np.float32)
    dqf = np.array([[0.0, np.nan, 0.0], [np.nan, 0.0, np.nan]], dtype=np.float32)
    on_disk = np.array([[True, True, True], [False, True, False]])
    fake_ds = xr.Dataset({"CMI_C01": (("y", "x"), cmi), "DQF_C01": (("y", "x"), dqf)})
    nc_path = tmp_path / "fake_goes.nc"
    fake_ds.to_netcdf(nc_path)

    ds = GOES(satellite="goes16", scan_mode="C", cache=False, verbose=False)
    ds._on_disk_mask = on_disk
    ds._on_disk_count = int(np.count_nonzero(on_disk))
    # No limb band here: this test is about the interior/off-disk distinction,
    # not the limb-band split (see test_goes_fetch_array_limb_band_is_debug).
    ds._limb_band_mask = np.zeros_like(on_disk)

    async def fake_get_s3_path(time):
        return "fake/path.nc"

    async def fake_fetch_remote_file(path):
        return str(nc_path)

    monkeypatch.setattr(ds, "_get_s3_path", fake_get_s3_path)
    monkeypatch.setattr(ds, "_fetch_remote_file", fake_fetch_remote_file)

    caught = []
    monkeypatch.setattr(
        "earth2studio.data.goes.logger.warning", lambda msg: caught.append(msg)
    )

    x = asyncio.run(ds.fetch_array(datetime(2024, 6, 1, 18, 0, 0), ["abi01c"]))

    assert x.shape == (1, 2, 3)
    assert np.array_equal(np.isnan(x[0]), np.isnan(cmi))
    assert len(caught) == 1
    assert "1 missing pixel" in caught[0]
    assert "25.0000%" in caught[0]  # 1 missing out of 4 on-disk pixels


def test_goes_fetch_array_no_warning_when_clean(monkeypatch, tmp_path):
    """No warning should be emitted when every on-disk pixel is finite, even
    though the off-disk pixels are NaN."""
    monkeypatch.setitem(GOES.SCAN_DIMENSIONS, "C", (2, 3))

    cmi = np.array([[1.0, 2.0, 3.0], [np.nan, 5.0, np.nan]], dtype=np.float32)
    on_disk = np.array([[True, True, True], [False, True, False]])
    fake_ds = xr.Dataset({"CMI_C01": (("y", "x"), cmi)})
    nc_path = tmp_path / "fake_goes_clean.nc"
    fake_ds.to_netcdf(nc_path)

    ds = GOES(satellite="goes16", scan_mode="C", cache=False, verbose=False)
    ds._on_disk_mask = on_disk
    ds._on_disk_count = int(np.count_nonzero(on_disk))
    ds._limb_band_mask = np.zeros_like(on_disk)

    async def fake_get_s3_path(time):
        return "fake/path.nc"

    async def fake_fetch_remote_file(path):
        return str(nc_path)

    monkeypatch.setattr(ds, "_get_s3_path", fake_get_s3_path)
    monkeypatch.setattr(ds, "_fetch_remote_file", fake_fetch_remote_file)

    caught = []
    monkeypatch.setattr(
        "earth2studio.data.goes.logger.warning", lambda msg: caught.append(msg)
    )

    asyncio.run(ds.fetch_array(datetime(2024, 6, 1, 18, 0, 0), ["abi01c"]))

    assert caught == []


def test_goes_fetch_array_limb_band_is_debug_not_warning(monkeypatch, tmp_path):
    """NaNs within the limb band should be logged at debug level, not warning;
    NaNs elsewhere on-disk should still warn. Covers the split described in the
    class docstring: at the disk edge our center-ray geometry and NOAA's
    retrieval can disagree about visibility, confined to a measured 3px rind.
    """
    monkeypatch.setitem(GOES.SCAN_DIMENSIONS, "C", (3, 3))

    # (0,0) limb-band NaN -> debug only; (2,2) interior NaN -> warning; rest finite.
    cmi = np.array(
        [[np.nan, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, np.nan]], dtype=np.float32
    )
    on_disk = np.ones((3, 3), dtype=bool)
    limb_band = np.zeros((3, 3), dtype=bool)
    limb_band[0, 0] = True
    fake_ds = xr.Dataset({"CMI_C01": (("y", "x"), cmi)})
    nc_path = tmp_path / "fake_goes_limb.nc"
    fake_ds.to_netcdf(nc_path)

    ds = GOES(satellite="goes16", scan_mode="C", cache=False, verbose=False)
    ds._on_disk_mask = on_disk
    ds._on_disk_count = int(np.count_nonzero(on_disk))
    ds._limb_band_mask = limb_band

    async def fake_get_s3_path(time):
        return "fake/path.nc"

    async def fake_fetch_remote_file(path):
        return str(nc_path)

    monkeypatch.setattr(ds, "_get_s3_path", fake_get_s3_path)
    monkeypatch.setattr(ds, "_fetch_remote_file", fake_fetch_remote_file)

    warnings, debugs = [], []
    monkeypatch.setattr(
        "earth2studio.data.goes.logger.warning", lambda msg: warnings.append(msg)
    )
    monkeypatch.setattr(
        "earth2studio.data.goes.logger.debug", lambda msg: debugs.append(msg)
    )

    asyncio.run(ds.fetch_array(datetime(2024, 6, 1, 18, 0, 0), ["abi01c"]))

    assert len(warnings) == 1
    assert "1 missing pixel" in warnings[0]
    assert len(debugs) >= 1
    limb_debug = [m for m in debugs if "missing pixel" in m]
    assert len(limb_debug) == 1
    assert "3px" in limb_debug[0]
    assert "limb" in limb_debug[0]
