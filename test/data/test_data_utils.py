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

import asyncio
import datetime
import os
from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr
from fsspec.implementations.http import HTTPFileSystem

from earth2studio.data import (
    DataArrayFile,
    Random,
    datasource_to_file,
    fetch_data,
    prep_data_array,
)
from earth2studio.data.utils import AsyncCachingFileSystem, datasource_cache_root


@pytest.fixture
def foo_data_array():
    time0 = datetime.datetime.now()
    return xr.DataArray(
        data=np.random.rand(8, 16, 32),
        dims=["one", "two", "three"],
        coords={
            "one": [time0 + i * datetime.timedelta(hours=6) for i in range(8)],
            "two": [f"{i}" for i in range(16)],
            "three": np.linspace(0, 1, 32),
        },
    )


@pytest.fixture
def equilinear_data_array():
    lat = np.linspace(-90, 90, 13)
    lon = np.linspace(0, 180, 24, endpoint=False)
    data = np.random.rand(2, 1, 13, 24)
    return xr.DataArray(
        data=data,
        dims=["time", "variable", "lat", "lon"],
        coords={
            "time": [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")],
            "variable": ["temp"],
            "lat": lat,
            "lon": lon,
        },
    )


@pytest.fixture
def curvilinear_data_array():
    y, x = np.mgrid[0:10, 0:12]
    lat = 30 + y * 2 + np.sin(x * 0.5) * 0.5
    lon = x * 2 + np.cos(y * 0.5) * 0.5
    data = np.random.rand(2, 1, 10, 12)

    return xr.DataArray(
        data=data,
        dims=["time", "variable", "hrrr_y", "hrrr_x"],
        coords={
            "time": [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")],
            "variable": ["temp"],
            "lat": (["hrrr_y", "hrrr_x"], lat),
            "lon": (["hrrr_y", "hrrr_x"], lon),
            "hrrr_y": np.arange(10),
            "hrrr_x": np.arange(12),
        },
    )


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
@pytest.mark.parametrize("dims", [["one", "two", "three"], ["three", "one", "two"]])
def test_prep_dataarray(foo_data_array, dims, device):

    data_array = foo_data_array.transpose(*dims)
    out, outc = prep_data_array(data_array, device)

    assert str(out.device) == device
    assert list(outc.keys()) == list(data_array.dims)
    for key in outc.keys():
        assert (outc[key] == np.array(data_array.coords[key])).all()
    assert out.shape == data_array.data.shape


def test_prep_data_array_curvilinear(equilinear_data_array, curvilinear_data_array):
    # Create another curvilinear grid
    y, x = np.mgrid[0:20, 0:24]
    target_lat = 30 + y * 1 + np.cos(x * 0.3) * 0.3
    target_lon = x * 1 + np.sin(y * 0.3) * 0.3

    target_coords = OrderedDict({"lat": target_lat, "lon": target_lon})

    out, coords = prep_data_array(
        curvilinear_data_array, interp_to=target_coords, interp_method="linear"
    )

    # Check output shape matches target grid
    assert out.shape == (2, 1, 20, 24)

    # Check coordinates are transformed correctly
    assert np.array_equal(coords["lat"], target_lat)
    assert np.array_equal(coords["lon"], target_lon)

    # Check HRRR-specific coordinates are removed
    assert "hrrr_y" not in coords
    assert "hrrr_x" not in coords

    out, coords = prep_data_array(
        equilinear_data_array, interp_to=target_coords, interp_method="linear"
    )

    # Check output shape matches target grid
    assert out.shape == (2, 1, 20, 24)

    # Check coordinates are transformed correctly
    assert np.array_equal(coords["lat"], target_lat)
    assert np.array_equal(coords["lon"], target_lon)

    # Check HRRR-specific coordinates are removed
    assert "hrrr_y" not in coords
    assert "hrrr_x" not in coords


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("1993-04-05T00:00")]),
        np.array(
            [
                np.datetime64("1999-10-11T12:00"),
                np.datetime64("2001-06-04T00:00"),
            ]
        ),
    ],
)
@pytest.mark.parametrize(
    "lead_time",
    [
        np.array([np.timedelta64(0, "h")]),
        np.array([np.timedelta64(-6, "h"), np.timedelta64(0, "h")]),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fetch_data(time, lead_time, device):
    variable = np.array(["a", "b", "c"])
    domain = OrderedDict({"lat": np.random.randn(720), "lon": np.random.randn(1440)})
    r = Random(domain)

    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    assert x.device == torch.device(device)
    assert np.all(coords["time"] == time)
    assert np.all(coords["lead_time"] == lead_time)
    assert np.all(coords["variable"] == variable)
    assert not torch.isnan(x).any()


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("1993-04-05T00:00")]),
        np.array(
            [
                np.datetime64("1999-10-11T12:00"),
                np.datetime64("2001-06-04T00:00"),
            ]
        ),
    ],
)
@pytest.mark.parametrize(
    "lead_time",
    [
        np.array([np.timedelta64(0, "h")]),
        np.array([np.timedelta64(-6, "h"), np.timedelta64(0, "h")]),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fetch_data_interp(time, lead_time, device):
    # Original (source) domain
    variable = np.array(["a", "b", "c"])
    domain = OrderedDict(
        {
            "lat": np.linspace(90, -90, 721, endpoint=True),
            "lon": np.linspace(0, 360, 1440),
        }
    )
    r = Random(domain)

    # Target domain, 1d lat/lon coords
    lat = np.linspace(60, 20, num=256)
    lon = np.linspace(130, 60, num=512)
    target_coords = OrderedDict(
        {
            "lat": lat,
            "lon": lon,
        }
    )

    # nearest neighbor interp
    x, coords = fetch_data(
        r,
        time,
        variable,
        lead_time,
        device=device,
        interp_to=target_coords,
        interp_method="nearest",
    )

    assert x.device == torch.device(device)
    assert np.all(coords["time"] == time)
    assert np.all(coords["lead_time"] == lead_time)
    assert np.all(coords["variable"] == variable)
    assert coords["lat"].shape == (256,)
    assert coords["lon"].shape == (512,)
    assert not torch.isnan(x).any()

    # bilinear interp
    x, coords = fetch_data(
        r,
        time,
        variable,
        lead_time,
        device=device,
        interp_to=target_coords,
        interp_method="linear",
    )

    assert x.device == torch.device(device)
    assert np.all(coords["time"] == time)
    assert np.all(coords["lead_time"] == lead_time)
    assert np.all(coords["variable"] == variable)
    assert coords["lat"].shape == (256,)
    assert coords["lon"].shape == (512,)
    assert not torch.isnan(x).any()

    # Target domain, 2d lat/lon coords
    lat = np.linspace(60, 20, num=256)
    lon = np.linspace(130, 60, num=512)
    lat2d, lon2d = np.meshgrid(lat, lon, indexing="ij")
    target_coords = OrderedDict(
        {
            "lat": lat2d,
            "lon": lon2d,
        }
    )

    # nearest neighbor interp
    x, coords = fetch_data(
        r,
        time,
        variable,
        lead_time,
        device=device,
        interp_to=target_coords,
        interp_method="nearest",
    )

    assert x.device == torch.device(device)
    assert np.all(coords["time"] == time)
    assert np.all(coords["lead_time"] == lead_time)
    assert np.all(coords["variable"] == variable)
    assert coords["lat"].shape == (256, 512)
    assert coords["lon"].shape == (256, 512)
    assert not torch.isnan(x).any()

    # bilinear interp
    x, coords = fetch_data(
        r,
        time,
        variable,
        lead_time,
        device=device,
        interp_to=target_coords,
        interp_method="linear",
    )

    assert x.device == torch.device(device)
    assert np.all(coords["time"] == time)
    assert np.all(coords["lead_time"] == lead_time)
    assert np.all(coords["variable"] == variable)
    assert coords["lat"].shape == (256, 512)
    assert coords["lon"].shape == (256, 512)
    assert not torch.isnan(x).any()


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("1993-04-05T00:00")]),
        np.array(
            [
                np.datetime64("1999-10-11T12:00"),
                np.datetime64("2001-06-04T00:00"),
            ]
        ),
    ],
)
@pytest.mark.parametrize(
    "lead_time",
    [
        np.array([np.timedelta64(0, "h")]),
        np.array([np.timedelta64(-6, "h"), np.timedelta64(0, "h")]),
    ],
)
@pytest.mark.parametrize(
    "backend",
    ["netcdf", "zarr"],
)
def test_datasource_to_file(time, lead_time, backend, tmp_path):

    variable = np.array(["a", "b", "c"])
    domain = OrderedDict({"lat": np.random.randn(720), "lon": np.random.randn(1440)})
    ds = Random(domain)

    if backend == "netcdf":
        file_name = str(tmp_path) + "/temp.nc"
    else:
        file_name = str(tmp_path) + "/temp.zarr"
    datasource_to_file(
        file_name,
        ds,
        time=time,
        variable=variable,
        lead_time=lead_time,
        backend=backend,
    )

    # To check attempt to get input data from saved file
    ds = DataArrayFile(file_name)
    x, coords = fetch_data(ds, time, variable, lead_time)

    assert np.all(coords["time"] == time)
    assert np.all(coords["lead_time"] == lead_time)
    assert np.all(coords["variable"] == variable)
    assert np.all(coords["lat"] == domain["lat"])
    assert np.all(coords["lon"] == domain["lon"])
    assert not torch.isnan(x).any()


def test_datasource_cache(tmp_path, monkeypatch):

    # Test with custom path via environment variable
    custom_path = str(tmp_path / "custom_cache")
    monkeypatch.setenv("EARTH2STUDIO_CACHE", custom_path)
    assert datasource_cache_root() == custom_path
    assert os.path.exists(custom_path)

    nonexistent_parent = str(tmp_path / "nonexistent")
    invalid_path = os.path.join(nonexistent_parent, "test")
    monkeypatch.setenv("EARTH2STUDIO_CACHE", invalid_path)

    def mock_makedirs(*args, **kwargs):
        raise OSError("Permission denied")

    with monkeypatch.context() as m:
        # Spoof this to make the directory creation fail
        m.setattr(os, "makedirs", mock_makedirs)
        with pytest.raises(OSError):
            datasource_cache_root()


# Async fsspec file system
@pytest.mark.asyncio
async def test_init_and_cache_dir(tmp_path):
    fs = HTTPFileSystem()
    cache_dir = tmp_path / "cache"
    acfs = AsyncCachingFileSystem(fs=fs, cache_storage=str(cache_dir))
    assert os.path.exists(cache_dir)
    assert acfs.fs is fs
    assert acfs.storage[-1] == str(cache_dir)


def test_cache_size(tmp_path):
    fs = HTTPFileSystem()
    cache_dir = tmp_path / "cache"
    acfs = AsyncCachingFileSystem(fs=fs, cache_storage=str(cache_dir))

    # List files in tmp_path
    files = os.listdir(cache_dir)
    assert len(files) == 0  # Should only contain cache directory

    # For some reason empty cache has some populated data in it
    assert acfs.cache_size() == 4096


def test_clear_cache(tmp_path):
    fs = HTTPFileSystem()
    cache_dir = tmp_path / "cache"
    acfs = AsyncCachingFileSystem(fs=fs, cache_storage=str(cache_dir))
    # Create a dummy file in cache
    dummy_file = os.path.join(cache_dir, "dummy.txt")
    with open(dummy_file, "w") as f:
        f.write("test")
    assert os.path.exists(dummy_file)
    acfs.clear_cache()
    # Cache directory should still exist, but file should be gone
    assert os.path.exists(cache_dir)
    assert not os.path.exists(dummy_file)


@pytest.mark.asyncio
async def test_async_cache_fs_storage_handling(tmp_path):
    fs = HTTPFileSystem()

    # Test TMP storage
    cache_fs = AsyncCachingFileSystem(fs=fs, cache_storage="TMP")
    assert len(cache_fs.storage) == 1
    assert cache_fs.storage[0] != "TMP"  # Should be converted to actual temp path

    # Test multiple storage locations
    multi_storage = [str(tmp_path / "cache1"), str(tmp_path / "cache2")]
    cache_fs = AsyncCachingFileSystem(fs=fs, cache_storage=multi_storage)
    assert list(cache_fs.storage) == multi_storage
    assert os.path.exists(multi_storage[-1])


@pytest.mark.asyncio
@pytest.mark.parametrize("expiry_time,wait_time", [(60, 1.0)])
async def test_async_cache_fs_cache_operations(tmp_path, expiry_time, wait_time):
    fs = HTTPFileSystem(asynchronous=True)
    cache_fs = AsyncCachingFileSystem(
        fs=fs,
        cache_storage=str(tmp_path),
        cache_check=0.1,
        expiry_time=expiry_time,
        asynchronous=True,
    )

    # Test cache size calculation
    initial_size = cache_fs.cache_size()
    remote_file = "https://raw.githubusercontent.com/NVIDIA/earth2studio/refs/heads/main/README.md"
    await cache_fs._cat_file(remote_file)
    await asyncio.sleep(wait_time)

    cache_fs._check_cache()

    assert initial_size < cache_fs.cache_size()
    assert cache_fs._check_file(remote_file) is not False
    # Test clear cache
    cache_fs.clear_cache()
    assert cache_fs._check_file(remote_file) is False

    remote_file = "https://raw.githubusercontent.com/NVIDIA/earth2studio/refs/heads/main/README.md"
    await cache_fs._cat_file(remote_file)
    await asyncio.sleep(wait_time)

    cache_fs.clear_expired_cache(expiry_time=0.1)
