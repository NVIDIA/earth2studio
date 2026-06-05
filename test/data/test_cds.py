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
import hashlib
import pathlib
import shutil
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from earth2studio.data import CDS

CDS_API_URL = "https://cds.climate.copernicus.eu/api"


@pytest.fixture(autouse=True)
def _set_cdsapi_url(monkeypatch):
    """Point cdsapi at the CDS endpoint for all tests in this module."""
    monkeypatch.setenv("CDSAPI_URL", CDS_API_URL)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        [
            datetime.datetime(year=1971, month=6, day=1, hour=6),
            datetime.datetime(year=2021, month=11, day=23, hour=12),
        ],
        np.array([np.datetime64("2024-01-01T00:00")]),
    ],
)
@pytest.mark.parametrize("variable", ["tcwv", ["sp", "w500"]])
def test_cds_fetch(time, variable):

    ds = CDS(cache=False)
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
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(variable))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2024-01-01T00:00")]),
    ],
)
def test_cds_tp06_fetch(time):

    ds = CDS(cache=False)
    data = ds(time, "tp06")
    shape = data.shape

    if isinstance(time, datetime.datetime):
        time = [time]

    assert shape[0] == len(time)
    assert shape[1] == 1
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    assert np.array_equal(data.coords["variable"].values, np.array(["tp06"]))


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(90)
@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2024-01-01T00:00")]),
    ],
)
@pytest.mark.parametrize("variable", [["z500", "r200"]])
@pytest.mark.parametrize("cache", [True, False])
def test_cds_cache(time, variable, cache):

    ds = CDS(cache=cache)
    data = ds(time, variable)
    shape = data.shape

    assert shape[0] == 1
    assert shape[1] == 2
    assert shape[2] == 721
    assert shape[3] == 1440
    assert not np.isnan(data.values).any()
    # Cahce should be present
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
@pytest.mark.timeout(30)
@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=1939, month=2, day=25),
        datetime.datetime(year=1, month=1, day=1, hour=13, minute=1),
        datetime.datetime.now(),
    ],
)
@pytest.mark.parametrize("variable", ["mpl"])
def test_cds_available(time, variable):
    with pytest.raises(ValueError):
        ds = CDS()
        ds(time, variable)


# ======================== Lazy client init tests ========================


def _compute_cache_filename(dataset_name: str, variable: str, level: list[str], time):
    """Compute the SHA-256 cache filename matching CDS._download_cds_grib_cached."""
    sha = hashlib.sha256(f"{dataset_name}_{variable}_{'_'.join(level)}_{time}".encode())
    return sha.hexdigest()


@patch("earth2studio.data.cds.cdsapi")
def test_cds_lazy_client_no_init(mock_cdsapi, tmp_path, monkeypatch):
    """Test that CDS() does not create a cdsapi.Client during __init__."""
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))
    ds = CDS()
    # Client should not be created yet
    assert ds._cds_client is None
    mock_cdsapi.Client.assert_not_called()


@patch("earth2studio.data.cds.cdsapi")
def test_cds_lazy_client_cache_hit_no_client(mock_cdsapi, tmp_path, monkeypatch):
    """Test that cache hits do not trigger cdsapi.Client creation."""
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    # Create the cache directory structure
    cache_dir = tmp_path / "cds"
    cache_dir.mkdir(parents=True, exist_ok=True)

    time = datetime.datetime(year=2024, month=1, day=1, hour=0)

    # t2m maps to reanalysis-era5-single-levels::2m_temperature::
    dataset_name = "reanalysis-era5-single-levels"
    cds_variable = "2m_temperature"
    level = [""]

    # Create a fake cached file with valid data
    lat = np.linspace(90, -90, 721)
    lon = np.linspace(0, 359.75, 1440)
    data = np.random.randn(721, 1440).astype(np.float32)
    da = xr.DataArray(
        data=data,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
    )

    # Write to cache path with the correct hash filename
    filename = _compute_cache_filename(dataset_name, cds_variable, level, time)
    cache_path = cache_dir / filename
    da.to_netcdf(str(cache_path))

    ds = CDS()

    # Mock xr.open_dataarray since we saved as netcdf, not grib
    with patch("xarray.open_dataarray") as mock_open:
        mock_open.return_value = da
        result = ds(time, "t2m")

    # Client should never have been created
    assert ds._cds_client is None
    mock_cdsapi.Client.assert_not_called()

    assert result.shape == (1, 1, 721, 1440)
    assert result.coords["variable"].values[0] == "t2m"


@patch("earth2studio.data.cds.cdsapi")
def test_cds_lazy_client_cache_miss_creates_client(mock_cdsapi, tmp_path, monkeypatch):
    """Test that cache miss triggers lazy cdsapi.Client creation."""
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))

    # Set up mock client and its retrieve method
    mock_reply = {"request_id": "test-123", "state": "completed"}
    mock_result = mock_cdsapi.Client.return_value.retrieve.return_value
    mock_result.update.return_value = None
    mock_result.reply = mock_reply
    mock_result.download.return_value = None

    ds = CDS()

    # Client not yet created
    assert ds._cds_client is None

    time = datetime.datetime(year=2024, month=1, day=1, hour=0)

    # Create cache dir so the code can run
    cache_dir = tmp_path / "cds"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # t2m is not cached, so accessing it should trigger client creation
    # The download will "succeed" via mock, but open_dataarray will be called
    # on the (empty) downloaded file. We mock that too.
    lat = np.linspace(90, -90, 721)
    lon = np.linspace(0, 359.75, 1440)
    data = np.random.randn(721, 1440).astype(np.float32)
    da = xr.DataArray(
        data=data,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon},
    )

    def fake_download(path):
        """Create a file at path to simulate download."""
        da.to_netcdf(path)

    mock_result.download.side_effect = fake_download

    with patch("xarray.open_dataarray") as mock_open:
        mock_open.return_value = da
        ds(time, "t2m")

    # Client should now be created
    mock_cdsapi.Client.assert_called_once()
    assert ds._cds_client is not None
