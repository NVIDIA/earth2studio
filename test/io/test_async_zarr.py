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

import functools
import os
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
