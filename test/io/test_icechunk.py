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

import os
import tempfile
from collections import OrderedDict

import numpy as np
import pytest
import torch
import xarray as xr
import zarr

from earth2studio.grids import LatLonGrid
from earth2studio.utils.coords import coord_array
from earth2studio.utils.cupy import from_torch

icechunk = pytest.importorskip("icechunk", reason="icechunk not installed")

from earth2studio.io import AsyncZarrBackend, IceChunkBackend  # noqa: E402

# Protocol behavior is covered by test_io_contract.py

TIMES = np.array(["2021-01-01T00", "2021-01-02T00"], dtype="datetime64[h]")
VARIABLES = ["t2m", "tcwv"]
GRID = LatLonGrid(np.linspace(90, -90, 8), np.linspace(0, 360, 16, endpoint=False))


def make_template(variables: list[str] = VARIABLES) -> xr.DataArray:
    return coord_array(
        ("time", "variable", *GRID.dims),
        {"time": TIMES, "variable": variables},
        grid=GRID,
        attrs={"units": "K"},
    )


def make_field(variables: list[str] = VARIABLES, seed: int = 0) -> xr.DataArray:
    shape = (len(TIMES), len(variables), *GRID.shape)
    data = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    return xr.DataArray(
        data,
        dims=("time", "variable", *GRID.dims),
        coords={
            "time": TIMES,
            "variable": variables,
            "lat": GRID.latitude,
            "lon": GRID.longitude,
        },
    )


def assert_round_trip(io: IceChunkBackend, field: xr.DataArray) -> None:
    """The store holds `field`, by label and through a fresh readonly session."""
    read = io.read(field)
    np.testing.assert_allclose(read.transpose(*field.dims).values, field.values)
    io.flush()
    root = zarr.open_group(io.repo.readonly_session(io.branch).store, mode="r")
    for variable in field.coords["variable"].values:
        np.testing.assert_allclose(
            root[str(variable)][:], field.sel(variable=variable, drop=True).values
        )


@pytest.mark.parametrize("variables", [["t2m"], VARIABLES])
def test_icechunk_memory_repo(variables: list[str]) -> None:

    io = IceChunkBackend()
    assert isinstance(io.repo, icechunk.Repository)
    assert isinstance(io.root, zarr.Group)

    io.add_array(make_template(variables))
    for dim in ("time", "lat", "lon"):
        assert dim in io
        assert dim in io.coords
    for variable in variables:
        assert variable in io
        assert io[variable].shape == (len(TIMES), *GRID.shape)
    assert "variable" not in io

    field = make_field(variables)
    io.write(field)
    snapshot_id = io.commit("write fields")
    assert isinstance(snapshot_id, str)
    assert_round_trip(io, field)


def test_icechunk_local_filesystem_and_branch() -> None:

    with tempfile.TemporaryDirectory() as td:
        repo_path = os.path.join(td, "repo")

        io = IceChunkBackend(repo_path, branch="experiment")
        assert io.branch == "experiment"
        assert "experiment" in io.repo.list_branches()

        io.add_array(make_template())
        field = make_field()
        io.write(field)
        io.commit("write fields")

        # Reopen the repository on the same branch and confirm data persisted
        io2 = IceChunkBackend(repo_path, branch="experiment")
        assert "t2m" in io2
        np.testing.assert_array_equal(io2.coords["time"], TIMES)
        read = io2.read(field)
        np.testing.assert_allclose(read.transpose(*field.dims).values, field.values)

        # A resumed run may re-add the same arrays and keep writing
        io2.add_array(make_template())
        io2.write(make_field(seed=1))
        io2.commit("write again")
        assert_round_trip(io2, make_field(seed=1))


def test_icechunk_reopen_with_scalar_array() -> None:
    storage = icechunk.in_memory_storage()
    io = IceChunkBackend(storage=storage)
    coords = OrderedDict({"lat": np.linspace(-90, 90, 8)})
    io.add_array(coords, "fields")
    # Scalar array, e.g. a CF grid_mapping variable; zarr stores () as None
    io.root.create_array(
        "crs", shape=(), dtype="int32", fill_value=0, dimension_names=()
    )
    io.root.create_group("diagnostics")
    io.commit("add scalar")

    io = IceChunkBackend(storage=storage)
    assert list(io.coords) == ["lat"]
    assert "crs" in io
    assert "diagnostics" in io.root
    assert "diagnostics" not in io.coords

    # Write and commit through the reopened backend, then reopen and read back
    x = torch.randn(8)
    io.write(x, coords, "fields")
    io.commit("write after reopen")

    io = IceChunkBackend(storage=storage)
    xx, _ = io.read(coords, "fields")
    assert torch.allclose(x, xx)
    assert "crs" in io


def test_icechunk_empty_commit() -> None:

    io = IceChunkBackend()
    io.add_array(make_template(["t2m"]))
    io.write(make_field(["t2m"]))
    io.commit("write fields")

    # Committing with no new writes raises unless allow_empty is passed
    with pytest.raises(icechunk.IcechunkError):
        io.commit("nothing changed")
    snapshot_id = io.commit("nothing changed", allow_empty=True)
    assert isinstance(snapshot_id, str)

    # Backend still usable after commits
    field = make_field(["t2m"], seed=2)
    io.write(field)
    io.commit("write again")
    assert_round_trip(io, field)


def test_icechunk_with_async_zarr_backend() -> None:
    """An Icechunk session store is a Zarr store, so it can back the async
    backend directly for non-blocking transactional writes."""

    with tempfile.TemporaryDirectory() as td:
        repo = icechunk.Repository.open_or_create(
            icechunk.local_filesystem_storage(os.path.join(td, "repo"))
        )
        session = repo.writable_session("main")

        times = np.asarray([np.datetime64("2021-01-01")])
        lead_times = np.asarray([np.timedelta64(6 * i, "h") for i in range(4)])
        coords = OrderedDict(
            {
                "time": times,
                "lead_time": lead_times,
                "lat": np.linspace(90, -90, 32),
                "lon": np.linspace(0, 360, 64, endpoint=False),
            }
        )

        io = AsyncZarrBackend(
            "unused",
            parallel_coords=OrderedDict({"time": times, "lead_time": lead_times}),
            store=session.store,
        )
        io.add_array(coord_array(tuple(coords), coords, name="t2m"))
        x = torch.randn(1, 4, 32, 64, dtype=torch.float32)
        for i in range(4):
            step_coords = OrderedDict(coords, lead_time=lead_times[i : i + 1])
            io.write(from_torch(x[:, i : i + 1], step_coords, name="t2m"))
        io.close()
        session.commit("async write")

        readonly = repo.readonly_session("main")
        root = zarr.open_group(readonly.store, mode="r")
        assert np.allclose(root["t2m"][:], x.numpy())


def test_icechunk_snapshot_time_travel() -> None:
    """Old snapshots remain readable after the branch moves on."""

    io = IceChunkBackend()
    io.add_array(make_template(["t2m"]))
    first = make_field(["t2m"], seed=0)
    io.write(first)
    snap1 = io.commit("first")

    second = make_field(["t2m"], seed=1)
    io.write(second)
    io.commit("second")

    # Branch tip has the new data
    assert_round_trip(io, second)

    # The first snapshot still has the original
    old = io.repo.readonly_session(snapshot_id=snap1)
    root = zarr.open_group(old.store, mode="r")
    np.testing.assert_allclose(
        root["t2m"][:], first.sel(variable="t2m", drop=True).values
    )


def test_icechunk_commit_conflict() -> None:
    """Concurrent commits to one branch: the second writer raises."""

    with tempfile.TemporaryDirectory() as td:
        repo_path = os.path.join(td, "repo")
        io1 = IceChunkBackend(repo_path)
        io1.add_array(make_template(["t2m"]))
        io1.write(make_field(["t2m"]))
        io1.commit("init")

        io2 = IceChunkBackend(repo_path)
        io1.write(make_field(["t2m"], seed=1))
        io2.write(make_field(["t2m"], seed=2))
        io1.commit("winner")
        # Note: ConflictError is a sibling of IcechunkError, not a subclass
        io2.flush()
        with pytest.raises(icechunk.ConflictError):
            io2.session.commit("loser")


def test_icechunk_codecs_and_chunks() -> None:
    """zarr_codecs compression and the chunks layout apply through icechunk."""

    io = IceChunkBackend(
        chunks={"time": 1, "lat": 4, "lon": 16},
        zarr_codecs=zarr.codecs.BloscCodec(cname="zstd"),
    )
    io.add_array(make_template(["t2m"]))
    assert io["t2m"].chunks == (1, 4, 16)
    assert any("blosc" in str(codec).lower() for codec in io["t2m"].compressors)

    field = make_field(["t2m"])
    io.write(field)
    io.commit("compressed")
    assert_round_trip(io, field)


def test_icechunk_to_xarray() -> None:
    """to_xarray opens the session store, flushing pending writes first."""

    io = IceChunkBackend()
    io.add_array(make_template())
    field = make_field()
    io.write(field)

    stored = io.to_xarray()
    assert set(stored.data_vars) == set(VARIABLES)
    np.testing.assert_allclose(
        stored["t2m"].transpose("time", "lat", "lon").values,
        field.sel(variable="t2m", drop=True).values,
    )
    np.testing.assert_array_equal(stored["time"].values, TIMES)
    assert stored["t2m"].attrs["units"] == "K"
    io.commit("write fields")


def test_icechunk_close_commits() -> None:
    """close persists pending writes, then rejects further writes."""

    with tempfile.TemporaryDirectory() as td:
        repo_path = os.path.join(td, "repo")
        io = IceChunkBackend(repo_path)
        io.add_array(make_template(["t2m"]))
        field = make_field(["t2m"])
        io.write(field)
        io.close()
        assert not io.session.has_uncommitted_changes

        with pytest.raises(RuntimeError):
            io.write(field)
        io.close()  # repeated close is a no-op

        root = zarr.open_group(
            IceChunkBackend(repo_path).repo.readonly_session("main").store, mode="r"
        )
        np.testing.assert_allclose(
            root["t2m"][:], field.sel(variable="t2m", drop=True).values
        )


def test_icechunk_close_conflict() -> None:
    """A conflicting commit on close raises, leaves the backend closed, and keeps
    the pending writes recoverable with commit(rebase_with=...)."""

    with tempfile.TemporaryDirectory() as td:
        repo_path = os.path.join(td, "repo")
        io1 = IceChunkBackend(repo_path)
        io1.add_array(make_template(["t2m"]))
        io1.write(make_field(["t2m"]))
        io1.commit("init")

        io2 = IceChunkBackend(repo_path)
        io1.write(make_field(["t2m"], seed=1))
        field = make_field(["t2m"], seed=2)
        io2.write(field)
        io1.commit("winner")

        with pytest.raises(icechunk.ConflictError):
            io2.close()
        with pytest.raises(RuntimeError, match="closed"):
            io2.write(field)
        io2.close()  # already closed, does not retry the commit
        assert io2.session.has_uncommitted_changes

        io2.commit(
            "recovered",
            rebase_with=icechunk.BasicConflictSolver(
                on_chunk_conflict=icechunk.VersionSelection.UseOurs
            ),
        )
        root = zarr.open_group(io2.repo.readonly_session("main").store, mode="r")
        np.testing.assert_allclose(
            root["t2m"][:], field.sel(variable="t2m", drop=True).values
        )


def test_icechunk_uncommitted_warning() -> None:
    from loguru import logger

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        # Dropping a backend with uncommitted writes warns
        io = IceChunkBackend()
        io.add_array(make_template(["t2m"]))
        io.write(make_field(["t2m"]))
        io.__del__()
        assert any("uncommitted" in m for m in messages)

        # Dropping a committed backend does not
        messages.clear()
        io2 = IceChunkBackend()
        io2.add_array(make_template(["t2m"]))
        io2.write(make_field(["t2m"]))
        io2.commit("write fields")
        io2.__del__()
        assert not messages

        # Nor does dropping a closed backend
        io3 = IceChunkBackend()
        io3.add_array(make_template(["t2m"]))
        io3.write(make_field(["t2m"]))
        io3.close()
        io3.__del__()
        assert not messages

        # Dropping a fresh, never-written backend does not
        io4 = IceChunkBackend()
        io4.__del__()
        assert not messages
    finally:
        logger.remove(sink_id)


def test_icechunk_nonblocking_write_by_default() -> None:
    field = make_field(["t2m"])

    # Default: write() submits to a background thread and returns immediately,
    # leaving the write pending until a flush point.
    io = IceChunkBackend()
    assert io._blocking is False
    assert io._executor is not None
    io.add_array(make_template(["t2m"]))
    io.write(field)
    assert len(io._pending) == 1

    # read()/__getitem__/commit() all flush pending writes first
    read = io.read(field)
    np.testing.assert_allclose(read.transpose(*field.dims).values, field.values)
    assert len(io._pending) == 0

    io.write(field)
    assert len(io._pending) == 1
    np.testing.assert_allclose(
        io["t2m"][:], field.sel(variable="t2m", drop=True).values
    )
    assert len(io._pending) == 0

    io.write(field)
    io.commit("write fields")
    assert len(io._pending) == 0

    # blocking=True writes synchronously, same as ZarrBackend
    io_blocking = IceChunkBackend(blocking=True)
    assert io_blocking._executor is None
    io_blocking.add_array(make_template(["t2m"]))
    io_blocking.write(field)
    assert io_blocking._pending == []
    np.testing.assert_allclose(
        io_blocking["t2m"][:], field.sel(variable="t2m", drop=True).values
    )
    io_blocking.commit("write fields")


def test_icechunk_nonblocking_write_error_surfaces_on_flush() -> None:
    io = IceChunkBackend()
    io.add_array(make_template(["t2m"]))
    io.write(make_field(["t2m"]))

    def _boom() -> None:
        raise ZeroDivisionError("simulated background write failure")

    io._pending.append(io._executor.submit(_boom))

    with pytest.raises(ZeroDivisionError):
        io.flush()
