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

import time
import os
import tempfile
from collections import OrderedDict
from importlib.metadata import version
from functools import partial

import fsspec
import numpy as np
import pytest
import torch
import zarr
import s3fs

try:
    zarr_version = version("zarr")
    zarr_major_version = int(zarr_version.split(".")[0])
except Exception:
    zarr_major_version = 2
if zarr_major_version < 3:
    pytest.skip("Zarr version 2 not supported")

from earth2studio.io import AsyncZarrBackend
from earth2studio.utils.coords import convert_multidim_to_singledim, split_coords


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("1958-01-31")],
        [
            np.datetime64("1971-06-01T06:00:00"),
            np.datetime64("2021-11-23T18:00:00"),
            np.datetime64("2021-11-24T00:00:00"),
        ],
    ],
)
@pytest.mark.parametrize(
    "variable",
    [["t2m"], ["t2m", "tcwv"]],
)
@pytest.mark.parametrize(
    "fs",
    [fsspec.filesystem("memory"), fsspec.filesystem("file")],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_async_zarr_write(
    time: list[np.datetime64],
    variable: list[str],
    fs: fsspec.spec.AbstractFileSystem,
    device: str,
    tmp_path: str,
) -> None:

    index_coords = {
        "time": np.asarray(time),
    }
    z = AsyncZarrBackend(f"{tmp_path}/output.zarr", index_coords=index_coords, fs=fs)

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )
    shape = [v.shape[0] for v in total_coords.values()]
    x = torch.randn(shape, device=device, dtype=torch.float32)
    for i, time0 in enumerate(time):
        total_coords["time"] = np.array([time0])
        z.write(x[i : i + 1], total_coords, "fields_1")
        assert "fields_1" in z.zs
        assert z.zs["fields_1"].shape == x.shape
        assert np.allclose(z.zs["fields_1"][i], x[i].to("cpu").numpy())
    z.close()
    assert np.allclose(z.zs["fields_1"], x.to("cpu").numpy())

    total_coords = OrderedDict(
        {
            "variable": np.asarray(variable),
            "time": np.asarray(time),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )
    shape = [v.shape[0] for v in total_coords.values()]
    x = torch.randn(shape, device=device, dtype=torch.float32)
    for i, time0 in enumerate(time):
        total_coords["time"] = np.array([time0])
        z.write(x[:, i : i + 1], total_coords, "fields_2")
        assert "fields_2" in z.zs
        assert z.zs["fields_2"].shape == x.shape
        assert np.allclose(z.zs["fields_2"][:, i], x[:, i].to("cpu").numpy())
    z.close()
    assert np.allclose(z.zs["fields_2"], x.to("cpu").numpy())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fs",
    [fsspec.filesystem("memory"), fsspec.filesystem("file")],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
async def test_async_zarr_async_write(
    fs: fsspec.spec.AbstractFileSystem, device: str, tmp_path: str
) -> None:
    time = [
        np.datetime64("1971-06-01T06:00:00"),
        np.datetime64("2021-11-23T18:00:00"),
        np.datetime64("2021-11-24T00:00:00"),
    ]
    index_coords = {
        "time": np.asarray(time),
    }
    z = AsyncZarrBackend(f"{tmp_path}/output.zarr", index_coords=index_coords, fs=fs)

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(["t2m", "tcwv"]),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )
    shape = [v.shape[0] for v in total_coords.values()]
    x = torch.randn(shape, device=device, dtype=torch.float32)
    for i, time0 in enumerate(time):
        total_coords["time"] = np.array([time0])
        await z.async_write(x[i : i + 1], total_coords, "fields_1")
        assert "fields_1" in z.zs
        assert z.zs["fields_1"].shape == x.shape
        assert np.allclose(z.zs["fields_1"][i], x[i].to("cpu").numpy())
    z.close()
    assert np.allclose(z.zs["fields_1"], x.to("cpu").numpy())

@pytest.mark.asyncio
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
async def test_async_zarr_non_blocking(
    device: str, tmp_path: str
) -> None:
    fs = fsspec.filesystem("file")
    times = [
        np.datetime64("1971-06-01T06:00:00"),
        np.datetime64("2021-11-23T18:00:00"),
        np.datetime64("2021-11-24T00:00:00"),
        np.datetime64("2021-11-25T00:00:00"),
        np.datetime64("2021-11-26T00:00:00"),
        np.datetime64("2021-11-27T00:00:00"),
        np.datetime64("2021-11-28T00:00:00"),
        np.datetime64("2021-11-29T00:00:00"),
    ]
    index_coords = {
        "time": np.asarray(times),
    }

    total_coords = OrderedDict(
        {
            "time": np.asarray(times),
            "variable": np.asarray(["t2m", "tcwv", "msl", "u10m"]),
            "lat": np.linspace(-90, 90, 720),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        }
    )
    shape = [v.shape[0] for v in total_coords.values()]
    x = torch.randn(shape, device=device, dtype=torch.float32)
    
    z_blocking = AsyncZarrBackend(f"{tmp_path}/output_blocking.zarr", index_coords=index_coords, fs=fs, blocking=True)
    start_time = time.perf_counter()
    for i, time0 in enumerate(times):
        total_coords["time"] = np.array([time0])
        z_blocking.write(x[i : i + 1], total_coords, "fields_1")
    blocking_time = time.perf_counter() - start_time

    z_nonblocking = AsyncZarrBackend(f"{tmp_path}/output_nonblocking.zarr", index_coords=index_coords, fs=fs, blocking=False)
    start_time = time.perf_counter()
    for i, time0 in enumerate(times):
        total_coords["time"] = np.array([time0])
        z_nonblocking.write(x[i : i + 1], total_coords, "fields_1")
    nonblocking_time = time.perf_counter() - start_time
    z_nonblocking.close()

    assert blocking_time > nonblocking_time, f"Blocking ({blocking_time:.3f}s) should be slower than non-blocking ({nonblocking_time:.3f}s)"
    assert np.allclose(z_blocking.zs["fields_1"], z_nonblocking.zs["fields_1"])

@pytest.mark.parametrize(
    "time,lead_time",
    [
        ([np.datetime64("1958-01-31")], [np.timedelta64(0, 'h'), np.timedelta64(12, 'h')]),
        ([
            np.datetime64("1971-06-01T06:00:00"),
            np.datetime64("2021-11-23T18:00:00"),
            np.datetime64("2021-11-24T00:00:00"),
        ], [np.timedelta64(0, 'h'), np.timedelta64(12, 'h'), np.timedelta64(24, 'h')]),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_async_zarr_2d_index(
    time: list[np.datetime64],
    lead_time: list[np.timedelta64],
    device: str,
    tmp_path: str,
) -> None:

    fs = fsspec.filesystem("file")
    variable=["v10m", "tcwv"]
    index_coords = {
        "time": np.asarray(time),
        "lead_time": np.asarray(lead_time),
    }
    z = AsyncZarrBackend(f"{tmp_path}/output.zarr", index_coords=index_coords, fs=fs, blocking=True)

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lead_time": np.asarray(lead_time),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )
    shape = [v.shape[0] for v in total_coords.values()]
    x = torch.randn(shape, device=device, dtype=torch.float32)
    for i, time0 in enumerate(time):
         for j, lead0 in enumerate(lead_time):
            total_coords["time"] = np.array([time0])
            total_coords["lead_time"] = np.array([lead0])
            z.write(x[i : i + 1, :, j : j + 1], total_coords, "fields_1")
            assert "fields_1" in z.zs
            assert z.zs["fields_1"].shape == x.shape
            assert np.allclose(z.zs["fields_1"][i,:,j], x[i,:,j].to("cpu").numpy())
    z.close()
    assert np.allclose(z.zs["fields_1"], x.to("cpu").numpy())



@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_async_zarr_split_variables(
    device: str,
    tmp_path: str,
) -> None:

    fs = fsspec.filesystem("file")
    times = [
        np.datetime64("1971-06-01T06:00:00"),
        np.datetime64("2021-11-23T18:00:00"),
        np.datetime64("2021-11-24T00:00:00"),
        np.datetime64("2021-11-25T00:00:00"),
        np.datetime64("2021-11-26T00:00:00"),
        np.datetime64("2021-11-27T00:00:00"),
        np.datetime64("2021-11-28T00:00:00"),
        np.datetime64("2021-11-29T00:00:00"),
    ]
    index_coords = {
        "time": np.asarray(times),
    }
    variable = np.asarray(["t2m", "tcwv"])

    total_coords = OrderedDict(
        {
            "time": np.asarray(times),
            "variable": variable,
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )
    shape = [v.shape[0] for v in total_coords.values()]
    x = torch.randn(shape, device=device, dtype=torch.float32)

    z = AsyncZarrBackend(f"{tmp_path}/output_nonblocking.zarr", index_coords=index_coords, fs=fs, blocking=False)
    for i, time0 in enumerate(times):
        total_coords["time"] = np.array([time0])
        split_x, coords, array_names = split_coords(x[i : i + 1], total_coords, dim="variable")
        z.write(split_x, coords, array_names)
    z.close()

    for i, v in enumerate(variable):
        assert np.allclose(z.zs[v], x[:,i].to("cpu").numpy())


@pytest.mark.parametrize("blocking", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.skipif(
    "S3FS_CI_KEY" not in os.environ or "S3FS_CI_SECRET" not in os.environ,
    reason="S3FS CI credentials not found in environment"
)
def test_async_zarr_remote(
    blocking: bool,
    device: str,
    tmp_path: str,
) -> None:

    fs = s3fs.S3FileSystem(
        key=os.environ["S3FS_CI_KEY"],
        secret=os.environ["S3FS_CI_SECRET"],
        client_kwargs={"endpoint_url": os.environ.get("S3FS_CI_ENDPOINT", None)},
        asynchronous=True,
    )

    import functools
    fs_factory = functools.partial(
        s3fs.S3FileSystem,
        key=os.environ["S3FS_CI_KEY"],
        secret=os.environ["S3FS_CI_SECRET"],
        client_kwargs={"endpoint_url": os.environ.get("S3FS_CI_ENDPOINT", None)},
    )

    times = [
        np.datetime64("1971-06-01T06:00:00"),
        np.datetime64("2021-11-23T18:00:00"),
        np.datetime64("2021-11-24T00:00:00"),
        np.datetime64("2021-11-25T00:00:00"),
        np.datetime64("2021-11-26T00:00:00"),
        np.datetime64("2021-11-27T00:00:00"),
        np.datetime64("2021-11-28T00:00:00"),
        np.datetime64("2021-11-29T00:00:00"),
    ]
    index_coords = {
        "time": np.asarray(times),
    }
    variable = np.asarray(["t2m", "tcwv"])

    total_coords = OrderedDict(
        {
            "time": np.asarray(times),
            "variable": variable,
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )
    shape = [v.shape[0] for v in total_coords.values()]
    x = torch.randn(shape, device=device, dtype=torch.float32)

    root = f"earth2studio/ci/{tmp_path}/.zarr"
    z = AsyncZarrBackend(root, index_coords=index_coords, fs=fs, blocking=blocking)

    for i, time0 in enumerate(times):
        total_coords["time"] = np.array([time0])
        split_x, coords, array_names = split_coords(x[i : i + 1], total_coords, dim="variable")
        z.write(split_x, coords, array_names)
    z.close()

    # # Open the zarr store with xarray and verify contents
    # import xarray as xr
    # mapper = fs.get_mapper(root)
    # ds = xr.open_zarr(mapper, storage_options={
    #     "key": os.environ["S3FS_CI_KEY"],
    #     "secret": os.environ["S3FS_CI_SECRET"],
    #     "client_kwargs": {"endpoint_url": os.environ.get("S3FS_CI_ENDPOINT", None)}
    # })
    # for i, v in enumerate(variable):
    #     assert v in ds
    #     assert np.allclose(ds[v].values, x[:,i].to("cpu").numpy())

    # Delete the zarr store
    # fs.rm(root, recursive=True)