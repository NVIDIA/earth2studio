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
import zarr

icechunk = pytest.importorskip("icechunk", reason="icechunk not installed")

from earth2studio.io import AsyncZarrBackend, IceChunkBackend  # noqa: E402


@pytest.mark.parametrize(
    "time",
    [
        [np.datetime64("1958-01-31")],
        [np.datetime64("1971-06-01T06:00:00"), np.datetime64("2021-11-23T12:00:00")],
    ],
)
@pytest.mark.parametrize(
    "variable",
    [["t2m"], ["t2m", "tcwv"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_icechunk_field(
    time: list[np.datetime64], variable: list[str], device: str
) -> None:

    total_coords = OrderedDict(
        {
            "time": np.asarray(time),
            "variable": np.asarray(variable),
            "lat": np.linspace(-90, 90, 180),
            "lon": np.linspace(0, 360, 360, endpoint=False),
        }
    )

    chunks = OrderedDict({"time": 1, "variable": 1, "lat": 180, "lon": 180})

    # Test in-memory repository
    io = IceChunkBackend(chunks=chunks)
    assert isinstance(io.repo, icechunk.Repository)
    assert isinstance(io.root, zarr.Group)

    array_name = "fields"
    io.add_array(total_coords, array_name)

    for dim in total_coords:
        assert dim in io
        assert dim in io.coords
        assert io[dim].shape == total_coords[dim].shape

    assert array_name in io

    shape = tuple(len(dim) for dim in total_coords.values())
    assert io[array_name].shape == shape

    x = torch.randn(shape, device=device, dtype=torch.float32)
    io.write(x, total_coords, array_name)
    assert np.allclose(io[array_name][:], x.to("cpu").numpy())

    xx, _ = io.read(total_coords, array_name, device=device)
    assert torch.allclose(x, xx)

    # Commit and confirm the write survives a fresh readonly session
    snapshot_id = io.commit("write fields")
    assert isinstance(snapshot_id, str)

    readonly = io.repo.readonly_session(io.branch)
    root = zarr.open_group(readonly.store, mode="r")
    assert np.allclose(root[array_name][:], x.to("cpu").numpy())


def test_icechunk_local_filesystem_and_branch() -> None:

    total_coords = OrderedDict(
        {
            "time": np.asarray([np.datetime64("2021-01-01")]),
            "variable": np.asarray(["t2m"]),
            "lat": np.linspace(-90, 90, 8),
            "lon": np.linspace(0, 360, 16, endpoint=False),
        }
    )
    array_name = "fields"
    shape = tuple(len(dim) for dim in total_coords.values())

    with tempfile.TemporaryDirectory() as td:
        repo_path = os.path.join(td, "repo")

        io = IceChunkBackend(repo_path, branch="experiment")
        assert io.branch == "experiment"
        assert "experiment" in io.repo.list_branches()

        io.add_array(total_coords, array_name)
        x = torch.randn(shape, dtype=torch.float32)
        io.write(x, total_coords, array_name)
        io.commit("write fields")

        # Reopen the repository on the same branch and confirm data persisted
        io2 = IceChunkBackend(repo_path, branch="experiment")
        assert array_name in io2
        xx, _ = io2.read(total_coords, array_name)
        assert torch.allclose(x, xx)


def test_icechunk_empty_commit() -> None:

    io = IceChunkBackend()
    total_coords = OrderedDict(
        {
            "time": np.asarray([np.datetime64("2021-01-01")]),
            "lat": np.linspace(-90, 90, 8),
        }
    )
    io.add_array(total_coords, "fields")
    io.write(torch.randn(1, 8), total_coords, "fields")
    io.commit("write fields")

    # Committing with no new writes raises unless allow_empty is passed
    with pytest.raises(icechunk.IcechunkError):
        io.commit("nothing changed")
    snapshot_id = io.commit("nothing changed", allow_empty=True)
    assert isinstance(snapshot_id, str)

    # Backend still usable after commits
    x = torch.randn(1, 8)
    io.write(x, total_coords, "fields")
    io.commit("write again")
    xx, _ = io.read(total_coords, "fields")
    assert torch.allclose(x, xx)


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
        x = torch.randn(1, 4, 32, 64, dtype=torch.float32)
        for i in range(4):
            step_coords = OrderedDict(coords)
            step_coords["lead_time"] = lead_times[i : i + 1]
            io.write(x[:, i : i + 1], step_coords, "t2m")
        io.close()
        session.commit("async write")

        readonly = repo.readonly_session("main")
        root = zarr.open_group(readonly.store, mode="r")
        assert np.allclose(root["t2m"][:], x.numpy())


def test_icechunk_snapshot_time_travel() -> None:
    """Old snapshots remain readable after the branch moves on."""

    total_coords = OrderedDict(
        {
            "time": np.asarray([np.datetime64("2021-01-01")]),
            "lat": np.linspace(-90, 90, 8),
        }
    )

    io = IceChunkBackend()
    io.add_array(total_coords, "fields")
    x1 = torch.randn(1, 8, dtype=torch.float32)
    io.write(x1, total_coords, "fields")
    snap1 = io.commit("first")

    x2 = torch.randn(1, 8, dtype=torch.float32)
    io.write(x2, total_coords, "fields")
    io.commit("second")

    # Branch tip has the new data
    xx, _ = io.read(total_coords, "fields")
    assert torch.allclose(x2, xx)

    # The first snapshot still has the original
    old = io.repo.readonly_session(snapshot_id=snap1)
    root = zarr.open_group(old.store, mode="r")
    assert np.allclose(root["fields"][:], x1.numpy())


def test_icechunk_commit_conflict() -> None:
    """Concurrent commits to one branch: the second writer raises."""

    total_coords = OrderedDict(
        {
            "time": np.asarray([np.datetime64("2021-01-01")]),
            "lat": np.linspace(-90, 90, 8),
        }
    )

    with tempfile.TemporaryDirectory() as td:
        repo_path = os.path.join(td, "repo")
        io1 = IceChunkBackend(repo_path)
        io1.add_array(total_coords, "fields")
        io1.write(torch.randn(1, 8), total_coords, "fields")
        io1.commit("init")

        io2 = IceChunkBackend(repo_path)
        io1.write(torch.randn(1, 8), total_coords, "fields")
        io2.write(torch.randn(1, 8), total_coords, "fields")
        io1.commit("winner")
        # Note: ConflictError is a sibling of IcechunkError, not a subclass
        with pytest.raises(icechunk.ConflictError):
            io2.session.commit("loser")


def test_icechunk_codecs_and_chunks() -> None:
    """zarr_codecs compression and the chunks layout apply through icechunk."""

    total_coords = OrderedDict(
        {
            "time": np.asarray(
                [np.datetime64("2021-01-01"), np.datetime64("2021-01-02")]
            ),
            "lat": np.linspace(-90, 90, 16),
            "lon": np.linspace(0, 360, 32, endpoint=False),
        }
    )

    io = IceChunkBackend(
        chunks={"time": 1, "lat": 8, "lon": 32},
        zarr_codecs=zarr.codecs.BloscCodec(cname="zstd"),
    )
    io.add_array(total_coords, "fields")
    assert io["fields"].chunks == (1, 8, 32)
    assert any("blosc" in str(c).lower() for c in io["fields"].compressors)

    x = torch.randn(2, 16, 32, dtype=torch.float32)
    io.write(x, total_coords, "fields")
    io.commit("compressed")

    readonly = io.repo.readonly_session(io.branch)
    root = zarr.open_group(readonly.store, mode="r")
    assert np.allclose(root["fields"][:], x.numpy())


def test_icechunk_uncommitted_warning() -> None:
    from loguru import logger

    total_coords = OrderedDict(
        {
            "time": np.asarray([np.datetime64("2021-01-01")]),
            "lat": np.linspace(-90, 90, 8),
        }
    )

    messages: list[str] = []
    sink_id = logger.add(messages.append, level="WARNING")
    try:
        # Dropping a backend with uncommitted writes warns
        io = IceChunkBackend()
        io.add_array(total_coords, "fields")
        io.write(torch.randn(1, 8), total_coords, "fields")
        io.__del__()
        assert any("uncommitted" in m for m in messages)

        # Dropping a committed backend does not
        messages.clear()
        io2 = IceChunkBackend()
        io2.add_array(total_coords, "fields")
        io2.write(torch.randn(1, 8), total_coords, "fields")
        io2.commit("write fields")
        io2.__del__()
        assert not messages

        # Dropping a fresh, never-written backend does not
        io3 = IceChunkBackend()
        io3.__del__()
        assert not messages
    finally:
        logger.remove(sink_id)
