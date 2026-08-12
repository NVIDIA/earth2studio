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
import concurrent.futures
import datetime
import functools
import os
import threading
import time
from collections import OrderedDict
from collections.abc import Callable

import fsspec
import numpy as np
import pytest
import s3fs
import torch
import xarray as xr
import zarr
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem

from earth2studio.io import AsyncZarrBackend
from earth2studio.utils.coords import split_coords


@pytest.mark.asyncio
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
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
async def test_async_zarr_write(
    time: list[np.datetime64],
    variable: list[str],
    device: str,
    tmp_path: str,
) -> None:

    parallel_coords = {
        "time": np.asarray(time),
    }
    z = AsyncZarrBackend(
        f"{tmp_path}/output.zarr",
        parallel_coords=parallel_coords,
        fs_factory=LocalFileSystem,
    )
    zsync = zarr.open(f"{tmp_path}/output.zarr")

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
        assert "fields_1" in zsync
        assert zsync["fields_1"].shape == x.shape
        assert np.allclose(zsync["fields_1"][i], x[i].to("cpu").numpy())
    z.close()
    assert np.allclose(zsync["fields_1"], x.to("cpu").numpy())

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
        assert "fields_2" in zsync
        assert zsync["fields_2"].shape == x.shape
        assert np.allclose(zsync["fields_2"][:, i], x[:, i].to("cpu").numpy())
    z.close()
    assert np.allclose(zsync["fields_2"], x.to("cpu").numpy())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "variable",
    [["t2m"], ["t2m", "tcwv"]],
)
@pytest.mark.parametrize(
    "fs_factory",
    [MemoryFileSystem, LocalFileSystem],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
async def test_async_zarr_async_write(
    variable: list[str],
    fs_factory: Callable[..., fsspec.spec.AbstractFileSystem],
    device: str,
    tmp_path: str,
) -> None:
    time = [
        np.datetime64("1971-06-01T06:00:00"),
        np.datetime64("2021-11-23T18:00:00"),
        np.datetime64("2021-11-24T00:00:00"),
    ]
    parallel_coords = {
        "time": np.asarray(time),
    }
    z = AsyncZarrBackend(
        f"{tmp_path}/output.zarr",
        parallel_coords=parallel_coords,
        fs_factory=fs_factory,
    )

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
        await z.async_write(x[i : i + 1], total_coords, "fields_1")
        assert "fields_1" in [key async for key in z.root.array_keys()]
        data = await (await z.root.get("fields_1")).getitem(slice(None))
        assert data.shape == x.shape
        assert np.allclose(data[i], x[i].to("cpu").numpy())
    z.close()
    data = await (await z.root.get("fields_1")).getitem(slice(None))
    assert np.allclose(data, x.to("cpu").numpy())


@pytest.mark.asyncio
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
async def test_async_zarr_non_blocking(device: str, tmp_path: str) -> None:
    fs_factory = functools.partial(fsspec.filesystem, "file")
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
    parallel_coords = {
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

    z_blocking = AsyncZarrBackend(
        f"{tmp_path}/output_blocking.zarr",
        parallel_coords=parallel_coords,
        fs_factory=fs_factory,
        blocking=True,
    )
    start_time = time.perf_counter()
    for i, time0 in enumerate(times):
        total_coords["time"] = np.array([time0])
        z_blocking.write(x[i : i + 1], total_coords, "fields_1")
    blocking_time = time.perf_counter() - start_time

    z_nonblocking = AsyncZarrBackend(
        f"{tmp_path}/output_nonblocking.zarr",
        parallel_coords=parallel_coords,
        fs_factory=fs_factory,
        blocking=False,
    )
    start_time = time.perf_counter()
    for i, time0 in enumerate(times):
        total_coords["time"] = np.array([time0])
        z_nonblocking.write(x[i : i + 1], total_coords, "fields_1")
    nonblocking_time = time.perf_counter() - start_time
    z_nonblocking.close()

    assert (
        blocking_time > nonblocking_time
    ), f"Blocking ({blocking_time:.3f}s) should be slower than non-blocking ({nonblocking_time:.3f}s)"

    data1 = await (await z_blocking.root.get("fields_1")).getitem(slice(None))
    data2 = await (await z_nonblocking.root.get("fields_1")).getitem(slice(None))
    assert np.allclose(data1, data2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "time,lead_time",
    [
        (
            [np.datetime64("1958-01-31")],
            [np.timedelta64(0, "h"), np.timedelta64(12, "h")],
        ),
        (
            [
                np.datetime64("1971-06-01T06:00:00"),
                np.datetime64("2021-11-23T18:00:00"),
                np.datetime64("2021-11-24T00:00:00"),
            ],
            [np.timedelta64(0, "h"), np.timedelta64(12, "h"), np.timedelta64(24, "h")],
        ),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
async def test_async_zarr_2d_index(
    time: list[np.datetime64],
    lead_time: list[np.timedelta64],
    device: str,
    tmp_path: str,
) -> None:

    fs_factory = functools.partial(fsspec.filesystem, "file")
    variable = ["v10m", "tcwv"]
    parallel_coords = {
        "time": np.asarray(time),
        "lead_time": np.asarray(lead_time),
    }
    z = AsyncZarrBackend(
        f"{tmp_path}/output.zarr",
        parallel_coords=parallel_coords,
        fs_factory=fs_factory,
        blocking=True,
    )
    z.chunked_coords = {"lat": 60}  # Also check custom chunking

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
            assert "fields_1" in [key async for key in z.root.array_keys()]
            data = await (await z.root.get("fields_1")).getitem(slice(None))
            assert data.shape == x.shape
            assert np.allclose(data[i, :, j], x[i, :, j].to("cpu").numpy())
    z.close()
    array = await z.root.get("fields_1")
    data = await array.getitem(slice(None))
    assert np.allclose(data, x.to("cpu").numpy())
    # Check chunk size is expected
    codec = await array.info_complete()
    assert codec._chunk_shape == (1, 2, 1, 60, 360)


@pytest.mark.asyncio
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
async def test_async_zarr_split_variables(
    device: str,
    tmp_path: str,
) -> None:

    fs_factory = functools.partial(fsspec.filesystem, "file")
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
    parallel_coords = {
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

    z = AsyncZarrBackend(
        f"{tmp_path}/output_nonblocking.zarr",
        parallel_coords=parallel_coords,
        fs_factory=fs_factory,
        blocking=False,
    )
    for i, time0 in enumerate(times):
        total_coords["time"] = np.array([time0])
        split_x, coords, array_names = split_coords(
            x[i : i + 1], total_coords, dim="variable"
        )
        z.write(split_x, coords, array_names)
    z.close()

    for i, v in enumerate(variable):
        data = await (await z.root.get(v)).getitem(slice(None))
        assert np.allclose(data, x[:, i].to("cpu").numpy())


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize("blocking", [True, False])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.skipif(
    "S3FS_KEY" not in os.environ or "S3FS_SECRET" not in os.environ,
    reason="S3FS credentials not found in environment",
)
def test_async_zarr_remote(
    blocking: bool,
    device: str,
) -> None:
    import uuid

    random_uuid = uuid.uuid4()
    root = os.path.join("earth2studio", "ci", "pytest", f"{random_uuid}.zarr")

    fs_factory = functools.partial(
        s3fs.S3FileSystem,
        key=os.environ["S3FS_KEY"],
        secret=os.environ["S3FS_SECRET"],
        client_kwargs={"endpoint_url": os.environ.get("S3FS_ENDPOINT", None)},
        asynchronous=True,
    )

    times = [
        np.datetime64("1971-06-01T06:00:00"),
        np.datetime64("2021-11-23T18:00:00"),
        np.datetime64("2021-11-24T00:00:00"),
        np.datetime64("2021-11-25T00:00:00"),
        np.datetime64("2021-11-26T00:00:00"),
    ]
    parallel_coords = {
        "time": np.asarray(times),
    }
    variable = np.asarray(["t2m", "tcwv"])

    total_coords = OrderedDict(
        {
            "time": np.asarray(times),
            "variable": variable,
            "lat": np.linspace(-90, 90, 8),
            "lon": np.linspace(0, 360, 16, endpoint=False),
        }
    )
    shape = [v.shape[0] for v in total_coords.values()]
    x = torch.randn(shape, device=device, dtype=torch.float32)
    z = AsyncZarrBackend(
        root, parallel_coords=parallel_coords, fs_factory=fs_factory, blocking=blocking
    )

    for i, time0 in enumerate(times):
        total_coords["time"] = np.array([time0])
        split_x, coords, array_names = split_coords(
            x[i : i + 1], total_coords, dim="variable"
        )
        z.write(split_x, coords, array_names)
    z.close()

    # Open the zarr store with xarray and verify contents
    ds = xr.open_zarr(
        f"s3://{root}",
        storage_options={
            "key": os.environ["S3FS_KEY"],
            "secret": os.environ["S3FS_SECRET"],
            "client_kwargs": {"endpoint_url": os.environ.get("S3FS_ENDPOINT", None)},
        },
    )
    for i, v in enumerate(variable):
        assert v in ds
        assert np.allclose(ds[v].values, x[:, i].to("cpu").numpy())

    # Delete the zarr store
    fs = s3fs.S3FileSystem(
        key=os.environ["S3FS_KEY"],
        secret=os.environ["S3FS_SECRET"],
        client_kwargs={"endpoint_url": os.environ.get("S3FS_ENDPOINT", None)},
    )
    try:
        fs.rm(root, recursive=True)
    except FileNotFoundError:
        pass


@pytest.mark.asyncio
async def test_async_zarr_errors(tmp_path: str) -> None:
    # Non-callable fsspec factory
    with pytest.raises(TypeError):
        AsyncZarrBackend(
            f"{tmp_path}/test.zarr", parallel_coords={}, fs_factory="not_callable"
        )

    # Invalid index coords
    parallel_coords = {
        "time": np.array([np.datetime64("2021-01-01"), np.datetime64("2021-01-01")])
    }
    with pytest.raises(ValueError):
        AsyncZarrBackend(f"{tmp_path}/test.zarr", parallel_coords=parallel_coords)

    # Create a mock filesystem that's not asynchronous
    class NonAsyncFileSystem(fsspec.AbstractFileSystem):
        def __init__(self):
            super().__init__()
            self.asynchronous = False
            self.protocol = "s3"

    def fs_factory():
        return NonAsyncFileSystem()

    with pytest.raises(TypeError):
        AsyncZarrBackend(
            f"{tmp_path}/test.zarr", parallel_coords={}, fs_factory=fs_factory
        )

    # Miss match between input data and array names
    z = AsyncZarrBackend(f"{tmp_path}/test.zarr", parallel_coords={})
    coords = OrderedDict(
        {
            "time": np.array([np.datetime64("2021-01-01")]),
            "lat": np.linspace(-90, 90, 10),
            "lon": np.linspace(0, 360, 10, endpoint=False),
        }
    )
    x = torch.randn(1, 10, 10)
    array_names = ["array1", "array2"]

    with pytest.raises(ValueError):
        await z.prepare_inputs(x, coords, array_names)

    # If input coordinate value belonging to an index coord is not present
    parallel_coords = {
        "time": np.array([np.datetime64("2021-01-01"), np.datetime64("2021-01-02")])
    }
    z = AsyncZarrBackend(f"{tmp_path}/test.zarr", parallel_coords=parallel_coords)
    coords = OrderedDict(
        {
            "time": np.array([np.datetime64("2021-01-03")]),  # Not in parallel_coords
            "lat": np.linspace(-90, 90, 10),
            "lon": np.linspace(0, 360, 10, endpoint=False),
        }
    )
    x = torch.randn(1, 10, 10)

    with pytest.raises(ValueError):
        await z.prepare_inputs(x, coords, "test_array")

    # Test shapeless coordiante
    z = AsyncZarrBackend(f"{tmp_path}/test.zarr", parallel_coords=parallel_coords)
    coords = OrderedDict(
        {
            "time": np.array([np.datetime64("2021-01-01")]),  # Not in parallel_coords
            "lat": np.array(0),
            "lon": np.linspace(0, 360, 10, endpoint=False),
        }
    )
    x = torch.randn(1, 10, 10)

    with pytest.raises(ValueError):
        await z.prepare_inputs(x, coords, "test_array")


@pytest.mark.asyncio
async def test_async_zarr_close(tmp_path: str) -> None:
    z = AsyncZarrBackend(
        f"{tmp_path}/test.zarr", parallel_coords={}, blocking=False, pool_size=2
    )
    coords = OrderedDict(
        {
            "time": np.array([np.datetime64("2021-01-01")]),
            "lat": np.linspace(-90, 90, 10),
            "lon": np.linspace(0, 360, 10, endpoint=False),
        }
    )

    x = torch.randn(1, 10, 10)

    z.write(x, coords, "test_array")
    z.write(x, coords, "test_array2")
    assert len(z.io_futures) > 0
    z.close()
    assert len(z.io_futures) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "zarr_codecs",
    [
        zarr.codecs.BloscCodec(
            cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.shuffle
        ),
        zarr.codecs.GzipCodec(level=3),
        zarr.codecs.ZstdCodec(level=1),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
async def test_async_zarr_codecs(
    zarr_codecs: dict | None,
    device: str,
    tmp_path: str,
) -> None:
    time = [np.datetime64("2021-11-23T18:00:00")]
    parallel_coords = {"time": np.asarray(time)}

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

    # Create AsyncZarrBackend with specified codecs
    z = AsyncZarrBackend(
        f"{tmp_path}/output_codecs.zarr",
        parallel_coords=parallel_coords,
        zarr_codecs=zarr_codecs,
    )
    for i, time0 in enumerate(time):
        total_coords["time"] = np.array([time0])
        z.write(x[i : i + 1], total_coords, "fields_codecs")
    z.close()

    assert "fields_codecs" in [key async for key in z.root.array_keys()]

    array = await z.root.get("fields_codecs")
    data = await array.getitem(slice(None))
    assert data.shape == x.shape
    assert np.allclose(data, x.to("cpu").numpy())

    # Verify compression was applied if codecs were specified
    if zarr_codecs is not None:
        assert hasattr(array, "metadata")
        assert hasattr(array.metadata, "codecs")
        codec = await array.info_complete()
        # Not the cleanest but good enough hopefully
        assert codec._compressors[0].__class__ == zarr_codecs.__class__


@pytest.mark.asyncio
async def test_async_zarr_existing_store(tmp_path: str) -> None:
    # Create an initial Zarr store with some data
    initial_time = [np.datetime64("2021-01-01"), np.datetime64("2021-01-02")]
    initial_parallel_coords = {"time": np.asarray(initial_time)}
    z_initial = AsyncZarrBackend(
        f"{tmp_path}/existing_store.zarr",
        parallel_coords=initial_parallel_coords,
        fs_factory=LocalFileSystem,
    )

    # Write some data to create the store
    coords = OrderedDict(
        {
            "time": np.asarray(initial_time),
            "variable": np.asarray(["t2m"]),
            "lat": np.linspace(-90, 90, 10),
            "lon": np.linspace(0, 360, 10, endpoint=False),
        }
    )
    x = torch.randn(2, 1, 10, 10)

    for i, time0 in enumerate(initial_time):
        coords["time"] = np.array([time0])
        z_initial.write(x[i : i + 1], coords, "test_array")

    z_initial.close()

    # Try to initialize with invalid parallel_coords that differ from existing store
    invalid_time = [
        np.datetime64("2021-01-01"),
        np.datetime64("2021-01-03"),
    ]  # Different second time
    invalid_parallel_coords = {"time": np.asarray(invalid_time)}

    with pytest.raises(ValueError):
        AsyncZarrBackend(
            f"{tmp_path}/existing_store.zarr",
            parallel_coords=invalid_parallel_coords,
            fs_factory=LocalFileSystem,
        )

    # Try to initialize with subset of parallel_coords that differ from existing store
    invalid_time = [initial_time[0]]
    invalid_parallel_coords = {"time": np.asarray(invalid_time)}

    with pytest.raises(ValueError):
        AsyncZarrBackend(
            f"{tmp_path}/existing_store.zarr",
            parallel_coords=invalid_parallel_coords,
            fs_factory=LocalFileSystem,
        )

    # Initialize with valid parallel_coords that match existing store
    valid_parallel_coords = {"time": np.asarray(initial_time)}

    z_valid = AsyncZarrBackend(
        f"{tmp_path}/existing_store.zarr",
        parallel_coords=valid_parallel_coords,
        fs_factory=LocalFileSystem,
    )
    new_coords = OrderedDict(
        {
            "time": np.asarray([initial_time[0]]),  # Use first time
            "variable": np.asarray(["t2m"]),
            "lat": np.linspace(-90, 90, 10),
            "lon": np.linspace(0, 360, 10, endpoint=False),
        }
    )
    new_x = torch.randn(1, 1, 10, 10)
    z_valid.write(new_x, new_coords, "new_array")

    # Verify the new array was created
    assert "new_array" in [key async for key in z_valid.root.array_keys()]

    # Verify we can read the data back
    data = await (await z_valid.root.get("new_array")).getitem(slice(None))
    assert data.shape == (2, 1, 10, 10)  # Should have shape of full array
    assert np.allclose(
        data[0], new_x.to("cpu").numpy()
    )  # First time slice should match

    z_valid.close()


def _shard_test_coords(
    lead_time: np.ndarray, variable: list[str]
) -> "OrderedDict[str, np.ndarray]":
    """Small helper to build a lead time major coordinate system"""
    return OrderedDict(
        {
            "lead_time": lead_time,
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 16),
            "lon": np.linspace(0, 360, 32, endpoint=False),
        }
    )


def _count_chunk_files(array_path: str) -> int:
    """Counts the number of stored chunk/shard objects of a local Zarr array"""
    total = 0
    for _, _, files in os.walk(os.path.join(array_path, "c")):
        total += len(files)
    return total


@pytest.mark.asyncio
@pytest.mark.parametrize("nsteps,shard_size", [(8, 4), (8, 8), (10, 4), (7, 8)])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
async def test_async_zarr_shard_write(
    nsteps: int, shard_size: int, device: str, tmp_path: str
) -> None:
    """Sharded writes must round trip exactly and collapse chunks into shard files.

    Covers both shard aligned lead times and trailing partial shards, which are only
    flushed on close().
    """
    lead_time = np.arange(nsteps).astype("timedelta64[h]")
    variable = ["t2m", "tcwv"]
    parallel_coords = {"lead_time": lead_time}

    z = AsyncZarrBackend(
        f"{tmp_path}/output.zarr",
        parallel_coords=parallel_coords,
        fs_factory=LocalFileSystem,
        blocking=False,
        pool_size=8,
        shard_coords={"lead_time": shard_size},
    )

    total_coords = _shard_test_coords(lead_time, variable)
    shape = [v.shape[0] for v in total_coords.values()]
    x = torch.randn(shape, device=device, dtype=torch.float32)

    for i in range(nsteps):
        total_coords["lead_time"] = lead_time[i : i + 1]
        z.write(x[i : i + 1], total_coords, "fields")
    z.close()

    array = await z.root.get("fields")
    assert array.shards is not None
    assert array.chunks[0] == 1
    assert array.shards[0] == shard_size

    data = await array.getitem(slice(None))
    assert np.allclose(data, x.to("cpu").numpy())

    # The point of the feature, one file per shard rather than one per lead time
    expected_shards = -(-nsteps // shard_size)
    assert _count_chunk_files(f"{tmp_path}/output.zarr/fields") == expected_shards


@pytest.mark.asyncio
async def test_async_zarr_shard_single_write_per_shard(
    tmp_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each shard must be flushed exactly once, on the non-merging fast path.

    A shard is a single file, so a second write to one means Zarr is doing a
    read-modify-write of it, which is what silently loses data when it happens
    concurrently. Under a pool of 8, consecutive lead times are staged by different
    threads into the same shard, so this also exercises the cross thread bookkeeping.
    """
    nsteps, shard_size = 8, 4
    lead_time = np.arange(nsteps).astype("timedelta64[h]")
    parallel_coords = {"lead_time": lead_time}

    z = AsyncZarrBackend(
        f"{tmp_path}/output.zarr",
        parallel_coords=parallel_coords,
        fs_factory=LocalFileSystem,
        blocking=False,
        pool_size=8,
        shard_coords={"lead_time": shard_size},
    )

    flushes: list[tuple[tuple, bool]] = []
    flush_lock = threading.Lock()
    original_flush = AsyncZarrBackend._flush_buffer

    async def recording_flush(
        self: AsyncZarrBackend,
        name: str,
        zarray: object,
        key: tuple,
        buffer: object,
    ) -> None:
        with flush_lock:
            flushes.append((key, buffer.preexisting))
        await original_flush(self, name, zarray, key, buffer)

    monkeypatch.setattr(AsyncZarrBackend, "_flush_buffer", recording_flush)

    total_coords = _shard_test_coords(lead_time, ["t2m"])
    shape = [v.shape[0] for v in total_coords.values()]
    x = torch.randn(shape, dtype=torch.float32)

    for i in range(nsteps):
        total_coords["lead_time"] = lead_time[i : i + 1]
        z.write(x[i : i + 1], total_coords, "fields")
    z.close()

    keys = [key for key, _ in flushes]
    assert len(keys) == len(set(keys)), f"a shard was flushed twice: {keys}"
    assert set(keys) == {("fields", (0, 0, 0, 0)), ("fields", (1, 0, 0, 0))}
    assert not any(
        preexisting for _, preexisting in flushes
    ), "shards took the read-modify-write merge path on a clean store"

    data = await (await z.root.get("fields")).getitem(slice(None))
    assert np.allclose(data, x.numpy())


@pytest.mark.asyncio
async def test_async_zarr_shard_restart(tmp_path: str) -> None:
    """Restarting into a store with an incomplete shard must not destroy its data.

    Run one writes part of a shard and closes, flushing it partially. Run two is a
    fresh backend with no memory of that, so it has to detect the existing shard in
    the store and merge into it rather than overwrite it.
    """
    nsteps, shard_size = 8, 8
    lead_time = np.arange(nsteps).astype("timedelta64[h]")
    parallel_coords = {"lead_time": lead_time}
    store = f"{tmp_path}/restart.zarr"

    total_coords = _shard_test_coords(lead_time, ["t2m"])
    shape = [v.shape[0] for v in total_coords.values()]
    x = torch.randn(shape, dtype=torch.float32)

    z1 = AsyncZarrBackend(
        store,
        parallel_coords=parallel_coords,
        fs_factory=LocalFileSystem,
        shard_coords={"lead_time": shard_size},
    )
    for i in range(6):
        total_coords["lead_time"] = lead_time[i : i + 1]
        z1.write(x[i : i + 1], total_coords, "fields")
    z1.close()

    # Partial shard is on disk, the remaining lead times read back as fill
    data = await (await z1.root.get("fields")).getitem(slice(None))
    assert np.allclose(data[:6], x[:6].numpy())

    z2 = AsyncZarrBackend(
        store,
        parallel_coords=parallel_coords,
        fs_factory=LocalFileSystem,
        shard_coords={"lead_time": shard_size},
    )
    for i in range(6, nsteps):
        total_coords["lead_time"] = lead_time[i : i + 1]
        z2.write(x[i : i + 1], total_coords, "fields")
    z2.close()

    data = await (await z2.root.get("fields")).getitem(slice(None))
    assert np.allclose(data, x.numpy()), "restart clobbered the pre-existing shard"


@pytest.mark.asyncio
async def test_async_zarr_shard_multi_dim(tmp_path: str) -> None:
    """Sharding across two parallel coordinates at once"""
    time = np.asarray(
        [np.datetime64("2021-01-01"), np.datetime64("2021-01-02")],
    )
    lead_time = np.arange(4).astype("timedelta64[h]")
    parallel_coords = {"time": time, "lead_time": lead_time}

    z = AsyncZarrBackend(
        f"{tmp_path}/output.zarr",
        parallel_coords=parallel_coords,
        fs_factory=LocalFileSystem,
        blocking=False,
        pool_size=4,
        shard_coords={"time": 2, "lead_time": 2},
    )

    total_coords = OrderedDict(
        {
            "time": time,
            "lead_time": lead_time,
            "variable": np.asarray(["t2m"]),
            "lat": np.linspace(-90, 90, 8),
            "lon": np.linspace(0, 360, 16, endpoint=False),
        }
    )
    shape = [v.shape[0] for v in total_coords.values()]
    x = torch.randn(shape, dtype=torch.float32)

    for i in range(time.shape[0]):
        for j in range(lead_time.shape[0]):
            total_coords["time"] = time[i : i + 1]
            total_coords["lead_time"] = lead_time[j : j + 1]
            z.write(x[i : i + 1, j : j + 1], total_coords, "fields")
    z.close()

    array = await z.root.get("fields")
    assert array.shards[:2] == (2, 2)
    data = await array.getitem(slice(None))
    assert np.allclose(data, x.numpy())
    # 2 time x 4 lead_time chunks collapse into 1 x 2 shards
    assert _count_chunk_files(f"{tmp_path}/output.zarr/fields") == 2


@pytest.mark.asyncio
async def test_async_zarr_shard_with_codecs_and_chunked_coords(tmp_path: str) -> None:
    """Sharding composes with compression and with explicit chunking of other dims"""
    nsteps = 4
    lead_time = np.arange(nsteps).astype("timedelta64[h]")
    parallel_coords = {"lead_time": lead_time}

    z = AsyncZarrBackend(
        f"{tmp_path}/output.zarr",
        parallel_coords=parallel_coords,
        fs_factory=LocalFileSystem,
        zarr_codecs=zarr.codecs.BloscCodec(cname="zstd"),
        chunked_coords={"lat": 8},
        shard_coords={"lead_time": 2, "lat": 16},
    )

    total_coords = _shard_test_coords(lead_time, ["t2m"])
    shape = [v.shape[0] for v in total_coords.values()]
    x = torch.randn(shape, dtype=torch.float32)

    for i in range(nsteps):
        total_coords["lead_time"] = lead_time[i : i + 1]
        z.write(x[i : i + 1], total_coords, "fields")
    z.close()

    array = await z.root.get("fields")
    assert array.chunks == (1, 1, 8, 32)
    assert array.shards == (2, 1, 16, 32)
    data = await array.getitem(slice(None))
    assert np.allclose(data, x.numpy())


@pytest.mark.asyncio
async def test_async_zarr_shard_async_write(tmp_path: str) -> None:
    """The async API takes the same buffering path, flushed via async_flush"""
    nsteps, shard_size = 6, 4
    lead_time = np.arange(nsteps).astype("timedelta64[h]")
    parallel_coords = {"lead_time": lead_time}

    z = AsyncZarrBackend(
        f"{tmp_path}/output.zarr",
        parallel_coords=parallel_coords,
        fs_factory=MemoryFileSystem,
        shard_coords={"lead_time": shard_size},
    )

    total_coords = _shard_test_coords(lead_time, ["t2m"])
    shape = [v.shape[0] for v in total_coords.values()]
    x = torch.randn(shape, dtype=torch.float32)

    for i in range(nsteps):
        total_coords["lead_time"] = lead_time[i : i + 1]
        await z.async_write(x[i : i + 1], total_coords, "fields")
    await z.async_flush()

    data = await (await z.root.get("fields")).getitem(slice(None))
    assert np.allclose(data, x.numpy())


def test_async_zarr_pool_throttle_counts_pending(tmp_path: str) -> None:
    """The throttle must count running writes, not submitted ones.

    Under sharding most writes only copy into a shard buffer and finish immediately,
    while one in every shard's worth of writes does the actual IO. Counting
    submissions instead lets a shard larger than the pool serialize every flush
    against the one before it.
    """
    z = AsyncZarrBackend(
        f"{tmp_path}/throttle.zarr",
        parallel_coords={"lead_time": np.arange(2).astype("timedelta64[h]")},
        fs_factory=LocalFileSystem,
        blocking=False,
        pool_size=4,
    )

    done: list[concurrent.futures.Future] = []
    for _ in range(10):
        f: concurrent.futures.Future = concurrent.futures.Future()
        f.set_result(None)
        done.append(f)
    running: concurrent.futures.Future = concurrent.futures.Future()

    z.io_futures = done[:5] + [running] + done[5:]
    # Cap of 3 pending, but only one future is actually pending so nothing blocks
    z._limit_pool_size(3)
    assert z.io_futures == [running], "completed futures were not pruned"

    running.set_result(None)
    z.close()


def test_async_zarr_pool_throttle_no_head_of_line_block(tmp_path: str) -> None:
    """The throttle must wait for any write to finish, not the oldest one.

    When sharding, a write that only fills a shard buffer finishes quickly while the
    write that flushes a shard takes far longer. Waiting on the head of the queue
    stalls the caller on the slowest operation, which serializes every flush against
    the one before it.
    """
    z = AsyncZarrBackend(
        f"{tmp_path}/headofline.zarr",
        parallel_coords={"lead_time": np.arange(2).astype("timedelta64[h]")},
        fs_factory=LocalFileSystem,
        blocking=False,
        pool_size=4,
    )

    slow: concurrent.futures.Future = concurrent.futures.Future()
    quick: concurrent.futures.Future = concurrent.futures.Future()
    z.io_futures = [slow, quick]

    timer = threading.Timer(0.1, lambda: quick.set_result(None))
    timer.start()
    try:
        # Over the cap of one, so it must block, but only until `quick` lands
        z._limit_pool_size(1)
    finally:
        timer.cancel()

    assert z.io_futures == [slow]
    assert not slow.done(), "throttle waited on the oldest future instead of any"

    slow.set_result(None)
    z.io_futures = []
    z.close()


def test_async_zarr_pool_throttle_surfaces_errors(tmp_path: str) -> None:
    """A write that failed must not have its exception silently discarded"""
    z = AsyncZarrBackend(
        f"{tmp_path}/throttle_err.zarr",
        parallel_coords={"lead_time": np.arange(2).astype("timedelta64[h]")},
        fs_factory=LocalFileSystem,
        blocking=False,
        pool_size=4,
    )

    failed: concurrent.futures.Future = concurrent.futures.Future()
    failed.set_exception(RuntimeError("write blew up"))
    z.io_futures = [failed]

    with pytest.raises(RuntimeError, match="write blew up"):
        z._limit_pool_size(8)

    z.io_futures = []
    z.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("max_inflight", [1, 3])
async def test_async_zarr_shard_inflight_limit(
    max_inflight: int, tmp_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent shard flushes must stay within max_inflight_shards.

    Each concurrent flush holds its buffer plus the codec's copies of it, so this is
    the bound that keeps sharding's host memory predictable.
    """
    nsteps, shard_size = 12, 2
    lead_time = np.arange(nsteps).astype("timedelta64[h]")

    z = AsyncZarrBackend(
        f"{tmp_path}/inflight.zarr",
        parallel_coords={"lead_time": lead_time},
        fs_factory=LocalFileSystem,
        blocking=False,
        pool_size=8,
        shard_coords={"lead_time": shard_size},
        max_inflight_shards=max_inflight,
    )

    # A slot is held from the moment _acquire_flush_slot admits a flush until that
    # flush's future completes, which is what frees the slot again. Counting the
    # release off a done callback rather than off _flush_buffer returning keeps the
    # counter's window identical to the one the gate enforces, otherwise the next
    # flush can be admitted before this one is decremented and the count reads high
    live = 0
    peak = 0
    counter_lock = threading.Lock()
    original_acquire = AsyncZarrBackend._acquire_flush_slot

    def release(_: object) -> None:
        nonlocal live
        with counter_lock:
            live -= 1

    async def counting_acquire(
        self: AsyncZarrBackend, current: concurrent.futures.Future
    ) -> None:
        nonlocal live, peak
        await original_acquire(self, current)
        with counter_lock:
            live += 1
            peak = max(peak, live)
        current.add_done_callback(release)
        # Hold the slot so overlapping flushes are actually observable, the real
        # writes here are far too small to overlap on their own
        await asyncio.sleep(0.05)

    monkeypatch.setattr(AsyncZarrBackend, "_acquire_flush_slot", counting_acquire)

    total_coords = _shard_test_coords(lead_time, ["t2m"])
    shape = [v.shape[0] for v in total_coords.values()]
    x = torch.randn(shape, dtype=torch.float32)

    for i in range(nsteps):
        total_coords["lead_time"] = lead_time[i : i + 1]
        z.write(x[i : i + 1], total_coords, "fields")
    z.close()

    assert (
        peak <= max_inflight
    ), f"{peak} shard flushes ran at once with max_inflight_shards={max_inflight}"
    assert peak >= 1

    data = await (await z.root.get("fields")).getitem(slice(None))
    assert np.allclose(data, x.numpy())


@pytest.mark.asyncio
async def test_async_zarr_shard_validation(tmp_path: str) -> None:
    """Invalid shard configurations must fail with a clear error"""
    lead_time = np.arange(4).astype("timedelta64[h]")
    parallel_coords = {"lead_time": lead_time}
    total_coords = _shard_test_coords(lead_time, ["t2m"])
    x = torch.randn([v.shape[0] for v in total_coords.values()], dtype=torch.float32)

    # Non positive shard size is rejected up front
    with pytest.raises(ValueError):
        AsyncZarrBackend(
            f"{tmp_path}/bad0.zarr",
            parallel_coords=parallel_coords,
            fs_factory=LocalFileSystem,
            shard_coords={"lead_time": 0},
        )

    # Shard size must be a multiple of the chunk size of that coordinate
    z = AsyncZarrBackend(
        f"{tmp_path}/bad1.zarr",
        parallel_coords=parallel_coords,
        fs_factory=LocalFileSystem,
        chunked_coords={"lat": 8},
        shard_coords={"lat": 12},
    )
    total_coords["lead_time"] = lead_time[0:1]
    with pytest.raises(ValueError):
        z.write(x[0:1], total_coords, "fields")

    # A shard size equal to the chunk size is a no-op, array stays unsharded
    z = AsyncZarrBackend(
        f"{tmp_path}/noop.zarr",
        parallel_coords=parallel_coords,
        fs_factory=LocalFileSystem,
        shard_coords={"lat": 16},
    )
    z.write(x[0:1], total_coords, "fields")
    z.close()
    assert (await z.root.get("fields")).shards is None


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_async_zarr_nonblocking_no_aliasing(tmp_path: str, device: str) -> None:
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("cuda not available")

    lead_time = np.array([np.timedelta64(0, "h"), np.timedelta64(6, "h")])
    total_coords = OrderedDict(
        {
            "lead_time": lead_time[0:1],
            "lat": np.linspace(-90, 90, 8),
            "lon": np.linspace(0, 360, 16, endpoint=False),
        }
    )

    z = AsyncZarrBackend(
        f"{tmp_path}/aliasing.zarr",
        parallel_coords=OrderedDict({"lead_time": lead_time}),
        fs_factory=LocalFileSystem,
        blocking=False,
        pool_size=4,
    )

    # One buffer reused across steps, exactly as an in-place model would
    buffer = torch.ones(1, 8, 16, device=device)
    for i in range(len(lead_time)):
        buffer.fill_(float(i + 1))
        total_coords["lead_time"] = lead_time[i : i + 1]
        z.write(buffer, total_coords, "fields")
        # The model would now compute the next step straight into `buffer`
        buffer.fill_(-999.0)
    z.close()

    stored = zarr.open(f"{tmp_path}/aliasing.zarr")["fields"][:]
    assert np.all(stored[0] == 1.0)
    assert np.all(stored[1] == 2.0)


def test_async_zarr_datetime_coords_converted(tmp_path: str) -> None:
    z = AsyncZarrBackend(
        f"{tmp_path}/datetime.zarr",
        parallel_coords={},
        fs_factory=LocalFileSystem,
        blocking=True,
    )
    total_coords = OrderedDict(
        {
            "time": np.array([datetime.datetime(2024, 1, 1)], dtype=object),
            "lat": np.linspace(-90, 90, 4),
        }
    )
    z.write(torch.ones(1, 4), total_coords, "fields")
    z.close()

    assert zarr.open(f"{tmp_path}/datetime.zarr")["time"].dtype.kind == "M"


def test_async_zarr_write_after_consolidate(tmp_path: str) -> None:
    lead_time = np.array([np.timedelta64(0, "h"), np.timedelta64(6, "h")])
    total_coords = OrderedDict(
        {
            "lead_time": lead_time[0:1],
            "lat": np.linspace(-90, 90, 8),
            "lon": np.linspace(0, 360, 16, endpoint=False),
        }
    )

    def backend() -> AsyncZarrBackend:
        return AsyncZarrBackend(
            f"{tmp_path}/consolidated.zarr",
            parallel_coords=OrderedDict({"lead_time": lead_time}),
            fs_factory=LocalFileSystem,
            blocking=True,
        )

    run1 = backend()
    run1.write(torch.ones(1, 8, 16), total_coords, "a")
    run1.close()
    zarr.consolidate_metadata(run1.root.store)

    # Second run creates an array the consolidated snapshot does not know about
    run2 = backend()
    run2.write(torch.full((1, 8, 16), 2.0), total_coords, "b")
    total_coords["lead_time"] = lead_time[1:2]
    run2.write(torch.full((1, 8, 16), 3.0), total_coords, "b")
    run2.close()
    zarr.consolidate_metadata(run2.root.store)

    stored = zarr.open(f"{tmp_path}/consolidated.zarr")["b"][:]
    assert np.all(stored[0] == 2.0) and np.all(stored[1] == 3.0)
    assert np.all(zarr.open(f"{tmp_path}/consolidated.zarr")["a"][0] == 1.0)


def test_async_zarr_shard_with_chunked_spatial_coord(tmp_path: str) -> None:
    lead_time = np.array([np.timedelta64(6 * i, "h") for i in range(8)])
    total_coords = OrderedDict(
        {
            "lead_time": lead_time[0:1],
            "lat": np.linspace(-90, 90, 32),
            "lon": np.linspace(0, 360, 16, endpoint=False),
        }
    )

    z = AsyncZarrBackend(
        f"{tmp_path}/shard_chunk.zarr",
        parallel_coords=OrderedDict({"lead_time": lead_time}),
        chunked_coords={"lat": 4},
        shard_coords={"lead_time": 4},
        fs_factory=LocalFileSystem,
        blocking=True,
    )
    for i in range(len(lead_time)):
        total_coords["lead_time"] = lead_time[i : i + 1]
        z.write(torch.full((1, 32, 16), float(i)), total_coords, "fields")
    z.close()

    stored = zarr.open(f"{tmp_path}/shard_chunk.zarr")["fields"][:]
    for i in range(len(lead_time)):
        assert np.all(stored[i] == float(i))
