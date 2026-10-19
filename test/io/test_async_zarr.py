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

"""Backend specific tests for AsyncZarrBackend.

The shared IO protocol rules are covered by test/io/test_io_contract.py; what is
exercised here is everything that is specific to this backend: the parallel
coordinate contract, non-blocking writes, sharding, the loop pool and the
object-store write paths.
"""

import asyncio
import concurrent.futures
import datetime
import functools
import os
import threading
from collections import OrderedDict
from collections.abc import Callable, Mapping

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
from earth2studio.utils.coords import coord_array
from earth2studio.utils.cupy import from_torch

TIMES = np.array(
    [
        np.datetime64("1971-06-01T06:00:00"),
        np.datetime64("2021-11-23T18:00:00"),
        np.datetime64("2021-11-24T00:00:00"),
    ]
)
LAT = np.linspace(-90, 90, 16)
LON = np.linspace(0, 360, 32, endpoint=False)


def template(coords: Mapping[str, np.ndarray], name: str | None = None) -> xr.DataArray:
    """Allocation-free signature over `coords`, in the order given"""
    return coord_array(tuple(coords), coords, name=name)


def field(
    coords: Mapping[str, np.ndarray], data: torch.Tensor, name: str = "fields"
) -> xr.DataArray:
    """Concrete field holding `data` over `coords`, in the order given.

    `name` selects the stored array, except when the coordinates hold a
    `variable` dimension, whose labels name the arrays instead.
    """
    return from_torch(data, OrderedDict(coords), name=name)


def lead_times(nsteps: int) -> np.ndarray:
    return np.arange(nsteps).astype("timedelta64[h]")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "times",
    [TIMES[:1], TIMES],
)
async def test_async_zarr_write(times: np.ndarray, tmp_path: str) -> None:
    """One slice per parallel step, for arrays holding the parallel dimension in
    different positions"""
    z = AsyncZarrBackend(
        f"{tmp_path}/output.zarr",
        parallel_coords={"time": times},
    )
    zsync = zarr.open(f"{tmp_path}/output.zarr")

    # Parallel dimension leading
    coords = OrderedDict({"time": times, "lat": LAT, "lon": LON})
    z.add_array(template(coords, name="fields_1"))
    x = torch.randn(len(times), 16, 32)
    for i, time0 in enumerate(times):
        step = OrderedDict(coords, time=times[i : i + 1])
        z.write(field(step, x[i : i + 1], name="fields_1"))
        assert zsync["fields_1"].shape == x.shape
        assert np.allclose(zsync["fields_1"][i], x[i].numpy())
    assert np.allclose(zsync["fields_1"], x.numpy())

    # Parallel dimension in the middle, the write is located by label either way
    coords = OrderedDict({"lat": LAT, "time": times, "lon": LON})
    z.add_array(template(coords, name="fields_2"))
    y = torch.randn(16, len(times), 32)
    for i, time0 in enumerate(times):
        step = OrderedDict(coords, time=times[i : i + 1])
        z.write(field(step, y[:, i : i + 1], name="fields_2"))
        assert np.allclose(zsync["fields_2"][:, i], y[:, i].numpy())
    z.close()
    assert np.allclose(zsync["fields_2"], y.numpy())


@pytest.mark.asyncio
@pytest.mark.parametrize("fs_factory", [MemoryFileSystem, LocalFileSystem])
async def test_async_zarr_async_write(
    fs_factory: Callable[..., fsspec.spec.AbstractFileSystem], tmp_path: str
) -> None:
    """The coroutine API writes through the backend's own loop"""
    z = AsyncZarrBackend(
        f"{tmp_path}/output.zarr",
        parallel_coords={"time": TIMES},
        fs_factory=fs_factory,
    )
    coords = OrderedDict({"time": TIMES, "lat": LAT, "lon": LON})
    z.add_array(template(coords, name="fields_1"))

    x = torch.randn(len(TIMES), 16, 32)
    for i in range(len(TIMES)):
        step = OrderedDict(coords, time=TIMES[i : i + 1])
        await z.async_write(field(step, x[i : i + 1], name="fields_1"))
        data = await (await z.root.get("fields_1")).getitem(slice(None))
        assert np.allclose(data[i], x[i].numpy())
    z.close()
    data = await (await z.root.get("fields_1")).getitem(slice(None))
    assert np.allclose(data, x.numpy())


@pytest.mark.asyncio
async def test_async_zarr_non_blocking(tmp_path: str) -> None:
    """Non-blocking writes return before the data lands, copy their input, and store the same bytes"""
    fs_factory = functools.partial(fsspec.filesystem, "file")
    times = np.array(
        [
            np.datetime64("2021-11-23T18:00:00") + np.timedelta64(i, "D")
            for i in range(8)
        ]
    )
    coords = OrderedDict(
        {
            "time": times,
            "lat": np.linspace(-90, 90, 720),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        }
    )
    x = torch.randn(8, 720, 1440)

    def make(path: str, blocking: bool) -> AsyncZarrBackend:
        z = AsyncZarrBackend(
            path,
            parallel_coords={"time": times},
            fs_factory=fs_factory,
            blocking=blocking,
        )
        z.add_array(template(coords, name="fields_1"))
        return z

    def write_steps(z: AsyncZarrBackend, data: torch.Tensor) -> None:
        for i in range(len(times)):
            step = OrderedDict(coords, time=times[i : i + 1])
            z.write(field(step, data[i : i + 1], name="fields_1"))

    z_blocking = make(f"{tmp_path}/blocking.zarr", True)
    write_steps(z_blocking, x)

    # Hold every background write at a gate: a non-blocking write must return
    # while its data has not landed, and must not depend on the caller's buffer
    z_nonblocking = make(f"{tmp_path}/nonblocking.zarr", False)
    gate = threading.Event()
    write_impl = z_nonblocking._write

    async def gated_write(*args, **kwargs):
        await asyncio.get_running_loop().run_in_executor(None, gate.wait)
        return await write_impl(*args, **kwargs)

    z_nonblocking._write = gated_write
    reused = x.clone()
    try:
        write_steps(z_nonblocking, reused)
        assert z_nonblocking.io_futures
        assert not any(future.done() for future in z_nonblocking.io_futures)
        reused.zero_()
    finally:
        gate.set()
    z_nonblocking.close()

    data1 = await (await z_blocking.root.get("fields_1")).getitem(slice(None))
    data2 = await (await z_nonblocking.root.get("fields_1")).getitem(slice(None))
    assert np.allclose(data1, data2)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "times,leads",
    [(TIMES[:1], lead_times(2)), (TIMES, lead_times(3))],
)
async def test_async_zarr_2d_index(
    times: np.ndarray, leads: np.ndarray, tmp_path: str
) -> None:
    """Two parallel dimensions at once, with a custom chunk size on a third"""
    z = AsyncZarrBackend(
        f"{tmp_path}/output.zarr",
        parallel_coords={"time": times, "lead_time": leads},
        chunked_coords={"lat": 8},
        blocking=True,
    )
    coords = OrderedDict({"time": times, "lat": LAT, "lead_time": leads, "lon": LON})
    z.add_array(template(coords, name="fields_1"))

    x = torch.randn(len(times), 16, len(leads), 32)
    for i in range(len(times)):
        for j in range(len(leads)):
            step = OrderedDict(
                coords, time=times[i : i + 1], lead_time=leads[j : j + 1]
            )
            z.write(field(step, x[i : i + 1, :, j : j + 1], name="fields_1"))
            data = await (await z.root.get("fields_1")).getitem(slice(None))
            assert np.allclose(data[i, :, j], x[i, :, j].numpy())
    z.close()

    array = await z.root.get("fields_1")
    assert np.allclose(await array.getitem(slice(None)), x.numpy())
    # Parallel dimensions are chunked with a size of one, `lat` as requested
    codec = await array.info_complete()
    assert codec._chunk_shape == (1, 8, 1, 32)


@pytest.mark.asyncio
async def test_async_zarr_split_variables(tmp_path: str) -> None:
    """A `variable` dimension becomes one array per label, written together"""
    variables = ["t2m", "tcwv"]
    z = AsyncZarrBackend(
        f"{tmp_path}/output.zarr",
        parallel_coords={"time": TIMES},
        blocking=False,
    )
    coords = OrderedDict(
        {"time": TIMES, "variable": np.asarray(variables), "lat": LAT, "lon": LON}
    )
    z.add_array(template(coords))

    x = torch.randn(len(TIMES), 2, 16, 32)
    for i in range(len(TIMES)):
        step = OrderedDict(coords, time=TIMES[i : i + 1])
        z.write(field(step, x[i : i + 1]))
    z.close()

    for i, variable in enumerate(variables):
        data = await (await z.root.get(variable)).getitem(slice(None))
        assert np.allclose(data, x[:, i].numpy())

    # A subset of variables is a valid write, each one is its own array
    z2 = AsyncZarrBackend(
        f"{tmp_path}/output.zarr",
        parallel_coords={"time": TIMES},
    )
    single = OrderedDict(
        {"time": TIMES[:1], "variable": np.asarray(["t2m"]), "lat": LAT, "lon": LON}
    )
    z2.write(field(single, torch.zeros(1, 1, 16, 32)))
    z2.close()
    data = await (await z2.root.get("t2m")).getitem(slice(None))
    assert np.all(data[0] == 0.0)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize("blocking", [True, False])
@pytest.mark.skipif(
    "S3FS_KEY" not in os.environ or "S3FS_SECRET" not in os.environ,
    reason="S3FS credentials not found in environment",
)
def test_async_zarr_remote(blocking: bool) -> None:
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

    variables = ["t2m", "tcwv"]
    coords = OrderedDict(
        {
            "time": TIMES,
            "variable": np.asarray(variables),
            "lat": np.linspace(-90, 90, 8),
            "lon": np.linspace(0, 360, 16, endpoint=False),
        }
    )
    x = torch.randn(len(TIMES), 2, 8, 16)
    z = AsyncZarrBackend(
        root,
        parallel_coords={"time": TIMES},
        fs_factory=fs_factory,
        blocking=blocking,
    )
    z.add_array(template(coords))

    for i in range(len(TIMES)):
        step = OrderedDict(coords, time=TIMES[i : i + 1])
        z.write(field(step, x[i : i + 1]))
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
    for i, variable in enumerate(variables):
        assert variable in ds
        assert np.allclose(ds[variable].values, x[:, i].numpy())

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


def test_async_zarr_construction_errors(tmp_path: str) -> None:
    """Constructor validation of the filesystem factory and parallel coordinates"""
    # Non-callable fsspec factory
    with pytest.raises(TypeError):
        AsyncZarrBackend(
            f"{tmp_path}/test.zarr", parallel_coords={}, fs_factory="not_callable"
        )

    # Repeated values in a parallel coordinate cannot be located
    with pytest.raises(ValueError):
        AsyncZarrBackend(
            f"{tmp_path}/test.zarr",
            parallel_coords={"time": np.array([TIMES[0], TIMES[0]])},
        )

    # A filesystem that is not asynchronous cannot be written through
    class NonAsyncFileSystem(fsspec.AbstractFileSystem):
        def __init__(self) -> None:
            super().__init__()
            self.asynchronous = False
            self.protocol = "s3"

    with pytest.raises(TypeError):
        AsyncZarrBackend(
            f"{tmp_path}/test.zarr",
            parallel_coords={},
            fs_factory=lambda: NonAsyncFileSystem(),
        )


def test_async_zarr_parallel_coord_extent(tmp_path: str) -> None:
    """A template must hold the full parallel coordinate, in order"""
    z = AsyncZarrBackend(
        f"{tmp_path}/extent.zarr",
        parallel_coords={"time": TIMES},
    )
    coords = OrderedDict({"time": TIMES, "lat": LAT, "lon": LON})

    # A subset of the parallel values would size the array short
    with pytest.raises(ValueError, match="parallel_coords"):
        z.add_array(template(OrderedDict(coords, time=TIMES[:2]), name="short"))
    # ... as would a reordering, the stored coordinate must match the template
    with pytest.raises(ValueError, match="parallel_coords"):
        z.add_array(template(OrderedDict(coords, time=TIMES[::-1]), name="flipped"))
    # ... or values that were never declared
    with pytest.raises(ValueError, match="parallel_coords"):
        other = TIMES + np.timedelta64(1, "D")
        z.add_array(template(OrderedDict(coords, time=other), name="other"))

    z.add_array(template(coords, name="fields"))
    z.close()
    assert z["fields"].shape == (3, 16, 32)


def test_async_zarr_write_errors(tmp_path: str) -> None:
    """Writes are validated against the store before anything is written"""
    leads = lead_times(4)
    z = AsyncZarrBackend(
        f"{tmp_path}/write_errors.zarr",
        parallel_coords={"lead_time": leads},
    )
    coords = OrderedDict({"lead_time": leads, "lat": LAT, "lon": LON})

    # Writes never create arrays
    with pytest.raises(ValueError):
        z.write(field(coords, torch.zeros(4, 16, 32)))

    z.add_array(template(coords, name="fields"))

    # A non-parallel dimension is a single chunk, so it must be written in full
    partial = OrderedDict(coords, lead_time=leads[:1], lat=LAT[:8])
    with pytest.raises(ValueError, match="written in full"):
        z.write(field(partial, torch.zeros(1, 8, 32)))

    # Labels that are not in the store cannot be located
    unknown = OrderedDict(coords, lead_time=np.array([np.timedelta64(99, "h")]))
    with pytest.raises(ValueError):
        z.write(field(unknown, torch.zeros(1, 16, 32)))

    # Nothing of the above landed in the store
    z.flush()
    assert np.isnan(np.asarray(z["fields"][:])).all()

    z.close()
    with pytest.raises(RuntimeError):
        z.write(field(coords, torch.zeros(4, 16, 32)))
    with pytest.raises(RuntimeError):
        z.add_array(template(coords, name="after_close"))


@pytest.mark.asyncio
async def test_async_zarr_close(tmp_path: str) -> None:
    """close() drains the in flight writes of a non-blocking backend"""
    leads = lead_times(2)
    z = AsyncZarrBackend(
        f"{tmp_path}/test.zarr",
        parallel_coords={"lead_time": leads},
        blocking=False,
        pool_size=2,
    )
    coords = OrderedDict({"lead_time": leads, "lat": LAT, "lon": LON})
    z.add_array(template(coords, name="test_array"))
    z.add_array(template(coords, name="test_array2"))

    x = torch.randn(2, 16, 32)
    z.write(field(coords, x, name="test_array"))
    z.write(field(coords, x, name="test_array2"))
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
async def test_async_zarr_codecs(zarr_codecs: object, tmp_path: str) -> None:
    """Data arrays are created with the requested compressor"""
    times = TIMES[:1]
    z = AsyncZarrBackend(
        f"{tmp_path}/output_codecs.zarr",
        parallel_coords={"time": times},
        zarr_codecs=zarr_codecs,
    )
    coords = OrderedDict({"time": times, "lat": LAT, "lon": LON})
    z.add_array(template(coords, name="fields_codecs"))

    x = torch.randn(1, 16, 32)
    z.write(field(coords, x, name="fields_codecs"))
    z.close()

    array = await z.root.get("fields_codecs")
    data = await array.getitem(slice(None))
    assert np.allclose(data, x.numpy())

    codec = await array.info_complete()
    # Not the cleanest but good enough hopefully
    assert codec._compressors[0].__class__ == zarr_codecs.__class__


@pytest.mark.asyncio
async def test_async_zarr_existing_store(tmp_path: str) -> None:
    """Reopening a store validates the parallel coordinates against it"""
    initial_time = np.array([np.datetime64("2021-01-01"), np.datetime64("2021-01-02")])
    store = f"{tmp_path}/existing_store.zarr"
    coords = OrderedDict({"time": initial_time, "lat": LAT, "lon": LON})

    z_initial = AsyncZarrBackend(store, parallel_coords={"time": initial_time})
    z_initial.add_array(template(coords, name="test_array"))
    x = torch.randn(2, 16, 32)
    for i in range(len(initial_time)):
        step = OrderedDict(coords, time=initial_time[i : i + 1])
        z_initial.write(field(step, x[i : i + 1], name="test_array"))
    z_initial.close()

    # Parallel coordinates that disagree with the stored ones are rejected up front
    for invalid in (
        np.array([np.datetime64("2021-01-01"), np.datetime64("2021-01-03")]),
        initial_time[:1],
    ):
        with pytest.raises(ValueError):
            AsyncZarrBackend(
                store,
                parallel_coords={"time": invalid},
            )

    # Matching ones reopen the store, which keeps its arrays and coordinates
    z_valid = AsyncZarrBackend(store, parallel_coords={"time": initial_time})
    assert "test_array" in z_valid
    # Re-adding the stored array is a no-op, the data survives
    z_valid.add_array(template(coords, name="test_array"))
    assert np.allclose(np.asarray(z_valid["test_array"][:]), x.numpy())

    # A new array joins the existing coordinates
    z_valid.add_array(template(coords, name="new_array"))
    new_x = torch.randn(1, 16, 32)
    z_valid.write(
        field(OrderedDict(coords, time=initial_time[:1]), new_x, name="new_array")
    )

    data = await (await z_valid.root.get("new_array")).getitem(slice(None))
    assert data.shape == (2, 16, 32)
    assert np.allclose(data[0], new_x[0].numpy())
    z_valid.close()


def _shard_coords(leads: np.ndarray) -> "OrderedDict[str, np.ndarray]":
    """Small helper to build a lead time major coordinate system"""
    return OrderedDict({"lead_time": leads, "lat": LAT, "lon": LON})


def _count_chunk_files(array_path: str) -> int:
    """Counts the number of stored chunk/shard objects of a local Zarr array"""
    total = 0
    for _, _, files in os.walk(os.path.join(array_path, "c")):
        total += len(files)
    return total


@pytest.mark.asyncio
@pytest.mark.parametrize("nsteps,shard_size", [(8, 4), (8, 8), (10, 4), (7, 8)])
async def test_async_zarr_shard_write(
    nsteps: int, shard_size: int, tmp_path: str
) -> None:
    """Sharded writes must round trip exactly and collapse chunks into shard files.

    Covers both shard aligned lead times and trailing partial shards, which are only
    flushed on close().
    """
    leads = lead_times(nsteps)
    z = AsyncZarrBackend(
        f"{tmp_path}/output.zarr",
        parallel_coords={"lead_time": leads},
        fs_factory=LocalFileSystem,
        blocking=False,
        pool_size=8,
        shard_coords={"lead_time": shard_size},
    )

    coords = _shard_coords(leads)
    z.add_array(template(coords, name="fields"))
    x = torch.randn(nsteps, 16, 32)
    for i in range(nsteps):
        z.write(field(OrderedDict(coords, lead_time=leads[i : i + 1]), x[i : i + 1]))
    z.close()

    array = await z.root.get("fields")
    assert array.shards is not None
    assert array.chunks[0] == 1
    assert array.shards[0] == shard_size

    data = await array.getitem(slice(None))
    assert np.allclose(data, x.numpy())

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
    leads = lead_times(nsteps)

    z = AsyncZarrBackend(
        f"{tmp_path}/output.zarr",
        parallel_coords={"lead_time": leads},
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

    coords = _shard_coords(leads)
    z.add_array(template(coords, name="fields"))
    x = torch.randn(nsteps, 16, 32)
    for i in range(nsteps):
        z.write(field(OrderedDict(coords, lead_time=leads[i : i + 1]), x[i : i + 1]))
    z.close()

    keys = [key for key, _ in flushes]
    assert len(keys) == len(set(keys)), f"a shard was flushed twice: {keys}"
    assert set(keys) == {("fields", (0, 0, 0)), ("fields", (1, 0, 0))}
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
    leads = lead_times(nsteps)
    store = f"{tmp_path}/restart.zarr"
    coords = _shard_coords(leads)
    x = torch.randn(nsteps, 16, 32)

    def backend() -> AsyncZarrBackend:
        z = AsyncZarrBackend(
            store,
            parallel_coords={"lead_time": leads},
            shard_coords={"lead_time": shard_size},
        )
        z.add_array(template(coords, name="fields"))
        return z

    z1 = backend()
    for i in range(6):
        z1.write(field(OrderedDict(coords, lead_time=leads[i : i + 1]), x[i : i + 1]))
    z1.close()

    # Partial shard is on disk, the remaining lead times read back as fill
    data = await (await z1.root.get("fields")).getitem(slice(None))
    assert np.allclose(data[:6], x[:6].numpy())

    z2 = backend()
    for i in range(6, nsteps):
        z2.write(field(OrderedDict(coords, lead_time=leads[i : i + 1]), x[i : i + 1]))
    z2.close()

    data = await (await z2.root.get("fields")).getitem(slice(None))
    assert np.allclose(data, x.numpy()), "restart clobbered the pre-existing shard"


@pytest.mark.asyncio
async def test_async_zarr_shard_reopen_without_add_array(tmp_path: str) -> None:
    """A reopened store stages whole shards even when add_array is not called again.

    Unstaged chunk writes to one shard would race on read-modify-write and lose data.
    """
    nsteps = 8
    leads = lead_times(nsteps)
    coords = _shard_coords(leads)
    x = torch.randn(nsteps, 16, 32)

    def backend() -> AsyncZarrBackend:
        return AsyncZarrBackend(
            f"{tmp_path}/reopen.zarr",
            parallel_coords={"lead_time": leads},
            shard_coords={"lead_time": nsteps},
            blocking=False,
            pool_size=8,
        )

    z1 = backend()
    z1.add_array(template(coords, name="fields"))
    for i in range(2):
        z1.write(field(OrderedDict(coords, lead_time=leads[i : i + 1]), x[i : i + 1]))
    z1.close()

    z2 = backend()
    assert z2._shard_specs["fields"] is not None
    for i in range(2, nsteps):
        z2.write(field(OrderedDict(coords, lead_time=leads[i : i + 1]), x[i : i + 1]))
    z2.close()

    data = await (await z2.root.get("fields")).getitem(slice(None))
    assert np.allclose(data, x.numpy())


@pytest.mark.asyncio
async def test_async_zarr_shard_multi_dim(tmp_path: str) -> None:
    """Sharding across two parallel coordinates at once"""
    times = np.asarray([np.datetime64("2021-01-01"), np.datetime64("2021-01-02")])
    leads = lead_times(4)

    z = AsyncZarrBackend(
        f"{tmp_path}/output.zarr",
        parallel_coords={"time": times, "lead_time": leads},
        blocking=False,
        pool_size=4,
        shard_coords={"time": 2, "lead_time": 2},
    )
    coords = OrderedDict(
        {
            "time": times,
            "lead_time": leads,
            "lat": np.linspace(-90, 90, 8),
            "lon": np.linspace(0, 360, 16, endpoint=False),
        }
    )
    z.add_array(template(coords, name="fields"))

    x = torch.randn(2, 4, 8, 16)
    for i in range(len(times)):
        for j in range(len(leads)):
            step = OrderedDict(
                coords, time=times[i : i + 1], lead_time=leads[j : j + 1]
            )
            z.write(field(step, x[i : i + 1, j : j + 1]))
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
    leads = lead_times(nsteps)
    z = AsyncZarrBackend(
        f"{tmp_path}/output.zarr",
        parallel_coords={"lead_time": leads},
        zarr_codecs=zarr.codecs.BloscCodec(cname="zstd"),
        chunked_coords={"lat": 8},
        shard_coords={"lead_time": 2, "lat": 16},
    )

    coords = _shard_coords(leads)
    z.add_array(template(coords, name="fields"))
    x = torch.randn(nsteps, 16, 32)
    for i in range(nsteps):
        z.write(field(OrderedDict(coords, lead_time=leads[i : i + 1]), x[i : i + 1]))
    z.close()

    array = await z.root.get("fields")
    assert array.chunks == (1, 8, 32)
    assert array.shards == (2, 16, 32)
    data = await array.getitem(slice(None))
    assert np.allclose(data, x.numpy())


@pytest.mark.asyncio
async def test_async_zarr_shard_async_write(tmp_path: str) -> None:
    """The async API takes the same buffering path, flushed via async_flush"""
    nsteps, shard_size = 6, 4
    leads = lead_times(nsteps)
    z = AsyncZarrBackend(
        f"{tmp_path}/output.zarr",
        parallel_coords={"lead_time": leads},
        fs_factory=MemoryFileSystem,
        shard_coords={"lead_time": shard_size},
    )

    coords = _shard_coords(leads)
    z.add_array(template(coords, name="fields"))
    x = torch.randn(nsteps, 16, 32)
    for i in range(nsteps):
        await z.async_write(
            field(OrderedDict(coords, lead_time=leads[i : i + 1]), x[i : i + 1])
        )
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
        parallel_coords={"lead_time": lead_times(2)},
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
        parallel_coords={"lead_time": lead_times(2)},
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
        parallel_coords={"lead_time": lead_times(2)},
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
    leads = lead_times(nsteps)

    z = AsyncZarrBackend(
        f"{tmp_path}/inflight.zarr",
        parallel_coords={"lead_time": leads},
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

    coords = _shard_coords(leads)
    z.add_array(template(coords, name="fields"))
    x = torch.randn(nsteps, 16, 32)
    for i in range(nsteps):
        z.write(field(OrderedDict(coords, lead_time=leads[i : i + 1]), x[i : i + 1]))
    z.close()

    assert (
        peak <= max_inflight
    ), f"{peak} shard flushes ran at once with max_inflight_shards={max_inflight}"
    assert peak >= 1

    data = await (await z.root.get("fields")).getitem(slice(None))
    assert np.allclose(data, x.numpy())


def test_async_zarr_chunked_coords_validation(tmp_path: str) -> None:
    """`chunked_coords` must never widen the chunk of a parallel coordinate"""
    leads = lead_times(4)
    coords = _shard_coords(leads)

    # A parallel coordinate is chunked at 1 so concurrent writers never share a
    # chunk; asking for more is a configuration error, not a silent override
    with pytest.raises(ValueError, match="parallel_coords"):
        AsyncZarrBackend(
            f"{tmp_path}/widened.zarr",
            parallel_coords={"lead_time": leads},
            chunked_coords={"lead_time": 2},
        )

    with pytest.raises(ValueError, match="must be positive"):
        AsyncZarrBackend(
            f"{tmp_path}/nonpositive.zarr",
            parallel_coords={"lead_time": leads},
            chunked_coords={"lat": 0},
        )

    # Restating the implied chunk of 1 is consistent, so it is allowed
    z = AsyncZarrBackend(
        f"{tmp_path}/restated.zarr",
        parallel_coords={"lead_time": leads},
        chunked_coords={"lead_time": 1, "lat": 8},
        blocking=True,
    )
    z.add_array(template(coords, name="fields"))
    z.write(field(OrderedDict(coords, lead_time=leads[:1]), torch.zeros(1, 16, 32)))
    z.close()
    assert z._compute_chunks({"lead_time": 4, "lat": 16, "lon": 32}) == {
        "lead_time": 1,
        "lat": 8,
        "lon": 32,
    }


@pytest.mark.asyncio
async def test_async_zarr_shard_validation(tmp_path: str) -> None:
    """Invalid shard configurations must fail with a clear error"""
    leads = lead_times(4)
    coords = _shard_coords(leads)

    # Non positive shard size is rejected up front
    with pytest.raises(ValueError):
        AsyncZarrBackend(
            f"{tmp_path}/bad0.zarr",
            parallel_coords={"lead_time": leads},
            shard_coords={"lead_time": 0},
        )

    # Shard size must be a multiple of the chunk size of that coordinate
    z = AsyncZarrBackend(
        f"{tmp_path}/bad1.zarr",
        parallel_coords={"lead_time": leads},
        chunked_coords={"lat": 8},
        shard_coords={"lat": 12},
    )
    with pytest.raises(ValueError):
        z.add_array(template(coords, name="fields"))

    # A shard size equal to the chunk size is a no-op, array stays unsharded
    z = AsyncZarrBackend(
        f"{tmp_path}/noop.zarr",
        parallel_coords={"lead_time": leads},
        shard_coords={"lat": 16},
    )
    z.add_array(template(coords, name="fields"))
    z.write(field(OrderedDict(coords, lead_time=leads[:1]), torch.zeros(1, 16, 32)))
    z.close()
    assert (await z.root.get("fields")).shards is None


def test_async_zarr_nonblocking_no_aliasing(tmp_path: str) -> None:
    """A non-blocking write must copy, the caller reuses its buffer immediately"""
    leads = lead_times(2)
    coords = _shard_coords(leads)
    z = AsyncZarrBackend(
        f"{tmp_path}/aliasing.zarr",
        parallel_coords={"lead_time": leads},
        blocking=False,
        pool_size=4,
    )
    z.add_array(template(coords, name="fields"))

    # One buffer reused across steps, exactly as an in-place model would
    buffer = torch.ones(1, 16, 32)
    for i in range(len(leads)):
        buffer.fill_(float(i + 1))
        step = OrderedDict(coords, lead_time=leads[i : i + 1])
        z.write(field(step, buffer))
        # The model would now compute the next step straight into `buffer`
        buffer.fill_(-999.0)
    z.close()

    stored = zarr.open(f"{tmp_path}/aliasing.zarr")["fields"][:]
    assert np.all(stored[0] == 1.0)
    assert np.all(stored[1] == 2.0)


def test_async_zarr_datetime_coords_converted(tmp_path: str) -> None:
    """Python datetimes in a template are stored as datetime64 coordinates"""
    stamps = np.array([datetime.datetime(2024, 1, 1)], dtype=object)
    z = AsyncZarrBackend(
        f"{tmp_path}/datetime.zarr",
        parallel_coords={"time": stamps},
        blocking=True,
    )
    coords = OrderedDict({"time": stamps, "lat": LAT})
    z.add_array(template(coords, name="fields"))
    z.write(field(coords, torch.ones(1, 16)))
    z.close()

    assert zarr.open(f"{tmp_path}/datetime.zarr")["time"].dtype.kind == "M"
    assert np.all(zarr.open(f"{tmp_path}/datetime.zarr")["fields"][:] == 1.0)


def test_async_zarr_write_after_consolidate(tmp_path: str) -> None:
    """A run after consolidation still sees arrays the snapshot does not know of"""
    leads = lead_times(2)
    coords = _shard_coords(leads)

    def backend() -> AsyncZarrBackend:
        return AsyncZarrBackend(
            f"{tmp_path}/consolidated.zarr",
            parallel_coords={"lead_time": leads},
            blocking=True,
        )

    run1 = backend()
    run1.add_array(template(coords, name="a"))
    run1.write(
        field(OrderedDict(coords, lead_time=leads[:1]), torch.ones(1, 16, 32), name="a")
    )
    run1.close()
    zarr.consolidate_metadata(run1.root.store)

    # Second run creates an array the consolidated snapshot does not know about
    run2 = backend()
    run2.add_array(template(coords, name="b"))
    for i, value in enumerate((2.0, 3.0)):
        step = OrderedDict(coords, lead_time=leads[i : i + 1])
        run2.write(field(step, torch.full((1, 16, 32), value), name="b"))
    run2.close()
    zarr.consolidate_metadata(run2.root.store)

    stored = zarr.open(f"{tmp_path}/consolidated.zarr")["b"][:]
    assert np.all(stored[0] == 2.0) and np.all(stored[1] == 3.0)
    assert np.all(zarr.open(f"{tmp_path}/consolidated.zarr")["a"][0] == 1.0)


def test_async_zarr_shard_with_chunked_spatial_coord(tmp_path: str) -> None:
    """Shards spanning several chunks of a spatial coordinate round trip"""
    leads = np.array([np.timedelta64(6 * i, "h") for i in range(8)])
    coords = OrderedDict(
        {"lead_time": leads, "lat": np.linspace(-90, 90, 32), "lon": LON}
    )

    z = AsyncZarrBackend(
        f"{tmp_path}/shard_chunk.zarr",
        parallel_coords={"lead_time": leads},
        chunked_coords={"lat": 4},
        shard_coords={"lead_time": 4},
        blocking=True,
    )
    z.add_array(template(coords, name="fields"))
    for i in range(len(leads)):
        step = OrderedDict(coords, lead_time=leads[i : i + 1])
        z.write(field(step, torch.full((1, 32, 32), float(i))))
    z.close()

    stored = zarr.open(f"{tmp_path}/shard_chunk.zarr")["fields"][:]
    for i in range(len(leads)):
        assert np.all(stored[i] == float(i))


# ---------------------------------------------------------------------------
# Obstore-backed store parameter (zarr ObjectStore write path)
# ---------------------------------------------------------------------------


def test_async_zarr_object_store_memory_url() -> None:
    """store as memory:// URL: writes bypass fsspec entirely"""
    times = np.array([np.datetime64("2024-01-01"), np.datetime64("2024-01-02")])
    z = AsyncZarrBackend(
        "ignored.zarr",
        parallel_coords=OrderedDict({"time": times}),
        store="memory:///",
    )
    # No fsspec filesystem is constructed on the object-store path
    assert z.fs is None
    assert all(fs is None for fs in z.fs_pool)

    coords = OrderedDict({"time": times, "lat": LAT, "lon": LON})
    z.add_array(template(coords, name="fields"))
    x = torch.randn(2, 16, 32)
    for i in range(len(times)):
        step = OrderedDict(coords, time=times[i : i + 1])
        z.write(field(step, x[i : i + 1]))
    z.close()

    stored = zarr.open_group(z._object_store, mode="r")["fields"][:]
    assert np.allclose(stored, x.numpy())


def test_async_zarr_default_local_is_fsspec_free(tmp_path: str) -> None:
    """Default path (no store, no fs_factory) writes via a zarr LocalStore
    without any fsspec filesystem; fs_factory warns as deprecated"""
    times = np.array([np.datetime64("2024-01-01")])
    coords = OrderedDict({"time": times, "lat": LAT, "lon": LON})
    x = torch.randn(1, 16, 32)

    z = AsyncZarrBackend(
        f"{tmp_path}/default_local.zarr", parallel_coords={"time": times}
    )
    assert z.fs is None
    assert all(fs is None for fs in z.fs_pool)

    z.add_array(template(coords, name="fields"))
    z.write(field(coords, x))
    z.close()
    assert np.allclose(zarr.open(f"{tmp_path}/default_local.zarr")["fields"][0], x[0])

    # Passing fs_factory still works but is deprecated
    with pytest.warns(FutureWarning, match="fs_factory is deprecated"):
        AsyncZarrBackend(
            f"{tmp_path}/legacy.zarr",
            parallel_coords={"time": times},
            fs_factory=LocalFileSystem,
        )

    # store also accepts a plain local path (no URL scheme required)
    z3 = AsyncZarrBackend(
        None, parallel_coords={"time": times}, store=f"{tmp_path}/plain_path.zarr"
    )
    z3.add_array(template(coords, name="fields"))
    z3.write(field(coords, x))
    z3.close()
    assert np.allclose(zarr.open(f"{tmp_path}/plain_path.zarr")["fields"][0], x[0])

    # ... but local paths take no store_kwargs
    with pytest.raises(ValueError):
        AsyncZarrBackend(
            None,
            parallel_coords={"time": times},
            store=f"{tmp_path}/plain_kwargs.zarr",
            store_kwargs={"region": "us-east-1"},
        )


def test_async_zarr_object_store_ignores_fs_factory() -> None:
    """fs_factory and file_name are ignored (not validated) when a store is
    provided; the store path wins over a callable fs_factory"""
    times = np.array([np.datetime64("2024-01-01")])
    z = AsyncZarrBackend(
        None,  # file_name not required on the store path
        parallel_coords=OrderedDict({"time": times}),
        fs_factory=None,  # non-callable placeholder must not raise
        store="memory:///",
    )
    assert z.fs is None

    coords = OrderedDict({"time": times, "lat": LAT, "lon": LON})
    x = torch.randn(1, 16, 32)
    z.add_array(template(coords, name="fields"))
    z.write(field(coords, x))
    z.close()
    stored = zarr.open_group(z._object_store, mode="r")["fields"][:]
    assert np.allclose(stored, x.numpy())

    # A callable fs_factory alongside a store still takes the store path
    z2 = AsyncZarrBackend(
        None,
        parallel_coords=OrderedDict({"time": times}),
        fs_factory=LocalFileSystem,
        store="memory:///",
    )
    assert z2.fs is None
    assert all(fs is None for fs in z2.fs_pool)

    # Without a store, file_name is still required
    with pytest.raises(ValueError):
        AsyncZarrBackend(None, parallel_coords=OrderedDict({"time": times}))

    # store_kwargs only applies to URL stores
    import obstore.store

    with pytest.raises(ValueError):
        AsyncZarrBackend(
            None,
            parallel_coords=OrderedDict({"time": times}),
            store=obstore.store.MemoryStore(),
            store_kwargs={"region": "us-east-1"},
        )


def test_async_zarr_object_store_local_instance(tmp_path: str) -> None:
    """store as obstore store instance, on-disk layout readable by plain zarr"""
    import obstore.store

    times = np.array([np.datetime64("2024-01-01"), np.datetime64("2024-01-02")])
    root = f"{tmp_path}/obstore_local"
    os.makedirs(root, exist_ok=True)
    coords = OrderedDict({"time": times, "lat": LAT, "lon": LON})
    x = torch.randn(2, 16, 32)

    z = AsyncZarrBackend(
        "ignored.zarr",
        parallel_coords=OrderedDict({"time": times}),
        store=obstore.store.LocalStore(root),
    )
    z.add_array(template(coords, name="fields"))
    z.write(field(OrderedDict(coords, time=times[0:1]), x[0:1]))
    z.close()

    # The written store is a standard zarr v3 store on disk
    assert np.allclose(zarr.open(root)["fields"][0], x[0].numpy())

    # Restart into the same store with a fresh backend and complete the array
    z2 = AsyncZarrBackend(
        "ignored.zarr",
        parallel_coords=OrderedDict({"time": times}),
        store=obstore.store.LocalStore(root),
    )
    z2.write(field(OrderedDict(coords, time=times[1:2]), x[1:2]))
    z2.close()
    assert np.allclose(zarr.open(root)["fields"][:], x.numpy())


def test_async_zarr_object_store_file_url(tmp_path: str) -> None:
    """store as file:// URL resolved through obstore.store.from_url"""
    times = np.array([np.datetime64("2024-01-01")])
    root = f"{tmp_path}/url_store"
    os.makedirs(root, exist_ok=True)

    z = AsyncZarrBackend(
        "ignored.zarr",
        parallel_coords=OrderedDict({"time": times}),
        store=f"file://{root}",
    )
    coords = OrderedDict({"time": times, "lat": LAT, "lon": LON})
    x = torch.randn(1, 16, 32)
    z.add_array(template(coords, name="fields"))
    z.write(field(coords, x))
    z.close()

    assert np.allclose(zarr.open(root)["fields"][0], x[0].numpy())


def test_async_zarr_object_store_zarr_instance() -> None:
    """store as an already constructed zarr store; read-only stores rejected"""
    import obstore.store

    times = np.array([np.datetime64("2024-01-01")])
    zstore = zarr.storage.ObjectStore(obstore.store.MemoryStore(), read_only=False)
    z = AsyncZarrBackend(
        "ignored.zarr",
        parallel_coords=OrderedDict({"time": times}),
        store=zstore,
    )
    coords = OrderedDict({"time": times, "lat": LAT, "lon": LON})
    x = torch.randn(1, 16, 32)
    z.add_array(template(coords, name="fields"))
    z.write(field(coords, x))
    z.close()
    assert np.allclose(zarr.open_group(zstore, mode="r")["fields"][0], x[0].numpy())

    with pytest.raises(ValueError):
        AsyncZarrBackend(
            "ignored.zarr",
            parallel_coords=OrderedDict({"time": times}),
            store=zstore.with_read_only(True),
        )


def test_async_zarr_object_store_non_blocking(tmp_path: str) -> None:
    """non-blocking pool shares one obstore-backed store across loops"""
    import obstore.store

    times = np.array([np.datetime64(f"2024-01-0{i}") for i in range(1, 5)])
    root = f"{tmp_path}/obstore_nb"
    os.makedirs(root, exist_ok=True)

    z = AsyncZarrBackend(
        "ignored.zarr",
        parallel_coords=OrderedDict({"time": times}),
        store=obstore.store.LocalStore(root),
        blocking=False,
        pool_size=2,
    )
    coords = OrderedDict({"time": times, "lat": LAT, "lon": LON})
    z.add_array(template(coords, name="fields"))
    x = torch.randn(4, 16, 32)
    for i in range(len(times)):
        z.write(field(OrderedDict(coords, time=times[i : i + 1]), x[i : i + 1]))
    z.close()

    assert np.allclose(zarr.open(root)["fields"][:], x.numpy())


def test_async_zarr_object_store_sharded(tmp_path: str) -> None:
    """sharded writes flush through the obstore-backed store"""
    import obstore.store

    leads = np.array([np.timedelta64(6 * i, "h") for i in range(4)])
    root = f"{tmp_path}/obstore_shard"
    os.makedirs(root, exist_ok=True)

    z = AsyncZarrBackend(
        "ignored.zarr",
        parallel_coords=OrderedDict({"lead_time": leads}),
        shard_coords={"lead_time": 2},
        store=obstore.store.LocalStore(root),
    )
    coords = _shard_coords(leads)
    z.add_array(template(coords, name="fields"))
    for i in range(len(leads)):
        step = OrderedDict(coords, lead_time=leads[i : i + 1])
        z.write(field(step, torch.full((1, 16, 32), float(i))))
    z.close()

    stored = zarr.open(root)["fields"]
    assert stored.shards == (2, 16, 32)
    for i in range(len(leads)):
        assert np.all(stored[i] == float(i))


@pytest.mark.asyncio
async def test_async_zarr_object_store_async_write() -> None:
    """async_write path with fs=None on the object-store branch"""
    times = np.array([np.datetime64("2024-01-01")])
    z = AsyncZarrBackend(
        "ignored.zarr",
        parallel_coords=OrderedDict({"time": times}),
        store="memory:///",
    )
    coords = OrderedDict({"time": times, "lat": LAT, "lon": LON})
    x = torch.randn(1, 16, 32)
    z.add_array(template(coords, name="fields"))
    await z.async_write(field(coords, x))
    stored = zarr.open_group(z._object_store, mode="r")["fields"][:]
    assert np.allclose(stored[0], x[0].numpy())


def test_async_zarr_object_store_parity(tmp_path: str) -> None:
    """object-store output matches the fsspec LocalFileSystem output exactly"""
    import obstore.store

    times = np.array([np.datetime64("2024-01-01"), np.datetime64("2024-01-02")])
    coords = OrderedDict({"time": times, "lat": LAT, "lon": LON})
    x = torch.randn(2, 16, 32)

    fs_root = f"{tmp_path}/parity_fs.zarr"
    zf = AsyncZarrBackend(
        fs_root,
        parallel_coords=OrderedDict({"time": times}),
        fs_factory=LocalFileSystem,
    )
    ob_root = f"{tmp_path}/parity_ob"
    os.makedirs(ob_root, exist_ok=True)
    zo = AsyncZarrBackend(
        "ignored.zarr",
        parallel_coords=OrderedDict({"time": times}),
        store=obstore.store.LocalStore(ob_root),
    )
    for backend in (zf, zo):
        backend.add_array(template(coords, name="fields"))
        for i in range(len(times)):
            backend.write(
                field(OrderedDict(coords, time=times[i : i + 1]), x[i : i + 1])
            )
        backend.close()

    a = xr.open_zarr(fs_root, consolidated=False)
    b = xr.open_zarr(ob_root, consolidated=False)
    xr.testing.assert_identical(a, b)


@pytest.mark.slow
@pytest.mark.xfail
@pytest.mark.timeout(120)
@pytest.mark.parametrize("blocking", [True, False])
@pytest.mark.skipif(
    "S3FS_KEY" not in os.environ or "S3FS_SECRET" not in os.environ,
    reason="S3FS credentials not found in environment",
)
def test_async_zarr_object_store_remote(blocking: bool) -> None:
    """Cloud write smoke test for the obstore-backed store path, the ObjectStore
    twin of test_async_zarr_remote"""
    import uuid

    import obstore
    import obstore.store

    config: dict[str, str] = {
        "access_key_id": os.environ["S3FS_KEY"],
        "secret_access_key": os.environ["S3FS_SECRET"],
    }
    if endpoint := os.environ.get("S3FS_ENDPOINT"):
        config["endpoint"] = endpoint

    root = f"s3://earth2studio/ci/pytest/{uuid.uuid4()}.zarr"
    store = obstore.store.from_url(root, config=config)

    variables = ["t2m", "tcwv"]
    coords = OrderedDict(
        {
            "time": TIMES,
            "variable": np.asarray(variables),
            "lat": np.linspace(-90, 90, 8),
            "lon": np.linspace(0, 360, 16, endpoint=False),
        }
    )
    x = torch.randn(len(TIMES), 2, 8, 16)
    z = AsyncZarrBackend(
        "unused.zarr",
        parallel_coords={"time": TIMES},
        store=store,
        blocking=blocking,
    )
    z.add_array(template(coords))

    try:
        for i in range(len(TIMES)):
            z.write(field(OrderedDict(coords, time=TIMES[i : i + 1]), x[i : i + 1]))
        z.close()

        # Verify contents through a fresh read-only store
        read_store = zarr.storage.ObjectStore(
            obstore.store.from_url(root, config=config), read_only=True
        )
        ds = xr.open_zarr(read_store, consolidated=False)
        for i, variable in enumerate(variables):
            assert variable in ds
            assert np.allclose(ds[variable].values, x[:, i].numpy())
    finally:
        # Delete the zarr store via obstore
        keys = [entry["path"] for chunk in obstore.list(store) for entry in chunk]
        if keys:
            obstore.delete(store, keys)


def test_async_zarr_add_array_eager_and_idempotent(tmp_path: str) -> None:
    """Arrays exist before any write, and re-adding one keeps its data"""
    leads = lead_times(2)
    coords = _shard_coords(leads)
    z = AsyncZarrBackend(
        f"{tmp_path}/add_array.zarr",
        parallel_coords=OrderedDict({"lead_time": leads}),
        blocking=True,
    )
    z.add_array(template(coords, name="fields_1"))
    z.add_array(template(coords, name="fields_2"))
    assert "fields_1" in z and "fields_2" in z
    assert "never_added" not in z
    assert z["fields_1"].shape == (2, 16, 32)

    # Re-adding an existing array must not raise or destroy data
    step = OrderedDict(coords, lead_time=leads[0:1])
    z.write(field(step, torch.ones(1, 16, 32), name="fields_1"))
    z.add_array(template(coords, name="fields_1"))
    assert np.all(np.asarray(z["fields_1"][0]) == 1.0)
    z.close()


def test_async_zarr_add_array_heterogeneous_dims(tmp_path: str) -> None:
    """Arrays of the store need not share a dimension set"""
    leads = lead_times(2)
    z = AsyncZarrBackend(
        f"{tmp_path}/hetero.zarr",
        parallel_coords=OrderedDict({"lead_time": leads}),
        blocking=True,
    )
    z.add_array(
        template(
            OrderedDict({"lead_time": leads, "lat": np.linspace(-90, 90, 4)}),
            name="with_lat",
        )
    )
    z.add_array(
        template(
            OrderedDict({"lead_time": leads, "member": np.arange(3)}),
            name="with_member",
        )
    )
    z.close()
    assert z["with_lat"].shape == (2, 4)
    assert z["with_member"].shape == (2, 3)


def test_async_zarr_add_array_coord_mismatch_raises(tmp_path: str) -> None:
    """A coordinate cannot hold two different sets of labels in one store"""
    z = AsyncZarrBackend(
        f"{tmp_path}/mismatch.zarr",
        parallel_coords={},
        blocking=True,
    )
    z.add_array(
        template(OrderedDict({"quantile": np.array([0.1, 0.5, 0.9])}), name="a")
    )
    with pytest.raises(ValueError, match="quantile"):
        z.add_array(
            template(OrderedDict({"quantile": np.array([0.25, 0.75])}), name="b")
        )
    # The same check survives a restart, the store is read back on construction
    z2 = AsyncZarrBackend(
        f"{tmp_path}/mismatch.zarr",
        parallel_coords={},
        blocking=True,
    )
    with pytest.raises(ValueError, match="quantile"):
        z2.add_array(
            template(OrderedDict({"quantile": np.array([0.25, 0.75])}), name="c")
        )


def test_async_zarr_mapping_surface(tmp_path: str) -> None:
    """The read-only mapping surface reports what the store holds"""
    leads = lead_times(2)
    coords = _shard_coords(leads)
    z = AsyncZarrBackend(
        f"{tmp_path}/surface.zarr",
        parallel_coords=OrderedDict({"lead_time": leads}),
        blocking=True,
    )
    z.add_array(template(coords, name="fields"))

    assert "fields" in z
    assert "absent" not in z
    assert set(z) == {"fields", "lead_time", "lat", "lon"}
    assert len(z) == 4
    assert z["fields"].shape == (2, 16, 32)

    # Coordinates are read back from the store, in dimension order
    assert list(z.coords) == ["lead_time", "lat", "lon"]
    for dim, values in coords.items():
        np.testing.assert_array_equal(np.asarray(z.coords[dim]), values)

    zarr.consolidate_metadata(z.store)
    z.close()
