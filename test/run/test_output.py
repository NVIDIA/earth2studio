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

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
import xarray as xr
from loguru import logger

from earth2studio.grids import CurvilinearGrid, LatLonGrid
from earth2studio.io import AsyncZarrBackend, KVBackend, XarrayBackend, ZarrBackend
from earth2studio.io.utils import output_template
from earth2studio.run.output import OutputManager
from earth2studio.run.work import WorkItem
from earth2studio.utils.coords import coord_array

GRID = LatLonGrid(np.array([1.0, 0.0]), np.array([0.0, 1.0, 2.0]))
LEADS = np.array([0, 6], dtype="timedelta64[h]").astype("timedelta64[ns]")
TIME_A = np.datetime64("2024-01-01T00:00:00")
TIME_B = np.datetime64("2024-01-02T00:00:00")
HORIZON = np.timedelta64(6, "h")
VARIABLES = ["t2m", "tp:sum:6h"]


def signature(grid: Any = GRID) -> xr.DataArray:
    """A prognostic stream signature: a dynamic prefix, then the fixed dims."""
    return coord_array(
        ("batch", "lead_time", "variable", *grid.dims),
        {"lead_time": LEADS, "variable": VARIABLES},
        dynamic=("batch",),
        grid=grid,
        name="forecast",
    )


def deterministic_items() -> list[WorkItem]:
    return [WorkItem(TIME_A, HORIZON), WorkItem(TIME_B, HORIZON)]


def ensemble_items() -> list[WorkItem]:
    """Two member groups per time: (0, 1) and (2, 3)."""
    return [
        WorkItem(time, HORIZON, members)
        for time in (TIME_A, TIME_B)
        for members in ((0, 1), (2, 3))
    ]


def field(time: np.datetime64, members: tuple[int, ...] = ()) -> xr.DataArray:
    """One item's output for the `signature` stream."""
    dims = ("time", "lead_time", "variable", *GRID.dims)
    coords: dict[str, Any] = {
        "time": np.array([time], dtype="datetime64[ns]"),
        "lead_time": LEADS,
        "variable": VARIABLES,
        **{key: np.asarray(value) for key, value in GRID.coords().items()},
    }
    shape = [1, len(LEADS), len(VARIABLES), *GRID.shape]
    if members:
        dims = ("ensemble", *dims)
        coords["ensemble"] = np.asarray(members)
        shape = [len(members), *shape]
    return xr.DataArray(
        np.ones(shape, dtype=np.float32), dims=dims, coords=coords, name="forecast"
    )


@pytest.fixture
def warnings_log() -> Iterator[list[str]]:
    """Collect the warnings loguru emits during a test."""
    messages: list[str] = []
    handler = logger.add(lambda message: messages.append(message), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(handler)


class FailingCloseBackend(KVBackend):
    """A backend whose commit fails on close, as a transactional store's can."""

    def close(self) -> None:
        super().close()
        raise RuntimeError("commit failed")


def prepared(backend: Any, items: list[WorkItem] | None = None) -> OutputManager:
    """An output manager whose one `forecast` store is prepared for `items`."""
    manager = OutputManager({"forecast": backend})
    manager.prepare({"forecast": signature()}, items or deterministic_items())
    return manager


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def test_prepare_builds_template_from_all_items(tmp_path: Any) -> None:
    backend = ZarrBackend(file_name=str(tmp_path / "forecast.zarr"))
    prepared(backend)

    stored = backend.to_xarray()
    assert stored["t2m"].dims == ("time", "lead_time", "lat", "lon")
    np.testing.assert_array_equal(
        stored["time"].values, np.array([TIME_A, TIME_B], dtype="datetime64[ns]")
    )
    np.testing.assert_array_equal(stored["lead_time"].values, LEADS)
    assert set(stored.data_vars) == set(VARIABLES)


def test_prepare_times_are_sorted_and_unique(tmp_path: Any) -> None:
    backend = ZarrBackend(file_name=str(tmp_path / "forecast.zarr"))
    items = [
        WorkItem(TIME_B, HORIZON, (0,)),
        WorkItem(TIME_A, HORIZON, (0,)),
        WorkItem(TIME_B, HORIZON, (1,)),
    ]
    prepared(backend, items)
    np.testing.assert_array_equal(
        backend.to_xarray()["time"].values,
        np.array([TIME_A, TIME_B], dtype="datetime64[ns]"),
    )


def test_prepare_ensemble_uses_union_of_member_ids(tmp_path: Any) -> None:
    backend = ZarrBackend(file_name=str(tmp_path / "forecast.zarr"))
    # Only one rank's items would hold (0, 1); the store spans every member.
    prepared(backend, ensemble_items())
    stored = backend.to_xarray()
    assert stored["t2m"].dims == ("ensemble", "time", "lead_time", "lat", "lon")
    np.testing.assert_array_equal(stored["ensemble"].values, [0, 1, 2, 3])


def test_prepare_carries_grid_metadata_and_auxiliary_coords(tmp_path: Any) -> None:
    grid = CurvilinearGrid(
        np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([[0.0, 1.0], [2.0, 3.0]])
    )
    backend = ZarrBackend(file_name=str(tmp_path / "forecast.zarr"))
    OutputManager({"forecast": backend}).prepare(
        {"forecast": signature(grid)}, deterministic_items()
    )
    stored = backend.to_xarray()
    assert stored["t2m"].dims == ("time", "lead_time", *grid.dims)
    assert "lat" in stored.coords and stored["lat"].dims == grid.dims
    assert stored["t2m"].attrs["topology"] == "curvilinear"


def test_prepare_reopening_a_store_is_a_no_op(tmp_path: Any) -> None:
    path = str(tmp_path / "forecast.zarr")
    manager = prepared(ZarrBackend(file_name=path))
    manager.write("forecast", field(TIME_A))
    manager.close()

    second = ZarrBackend(file_name=path)
    prepared(second)
    assert not np.isnan(second.to_xarray()["t2m"].sel(time=TIME_A)).any()


@pytest.mark.parametrize("backend_type", ["zarr", "async_zarr"])
def test_prepare_on_two_backends_of_one_store(tmp_path: Any, backend_type: str) -> None:
    # Each rank builds its backend before rank 0 creates the arrays.
    path = str(tmp_path / "forecast.zarr")
    times = np.array([TIME_A, TIME_B], dtype="datetime64[ns]")

    def build() -> Any:
        if backend_type == "zarr":
            return ZarrBackend(file_name=path)
        return AsyncZarrBackend(path, parallel_coords={"time": times}, blocking=True)

    rank0, rank1 = build(), build()
    prepared(rank0)
    prepared(rank1).write("forecast", field(TIME_B))
    rank1.close()
    stored = xr.open_zarr(path, consolidated=False)
    np.testing.assert_allclose(stored["t2m"].sel(time=TIME_B).values, 1.0)


def test_prepare_rejects_a_backend_without_a_schema() -> None:
    manager = OutputManager({"forcast": KVBackend()})
    with pytest.raises(ValueError, match="unknown streams"):
        manager.prepare({"forecast": signature()}, deterministic_items())


def test_prepare_rejects_empty_items() -> None:
    with pytest.raises(ValueError, match="without work items"):
        OutputManager({"forecast": KVBackend()}).prepare({"forecast": signature()}, [])


def test_prepare_rejects_items_sharing_a_time_and_member() -> None:
    items = [WorkItem(TIME_A, HORIZON), WorkItem(TIME_A, 2 * HORIZON)]
    with pytest.raises(ValueError, match="Repeated"):
        prepared(KVBackend(), items)


def test_prepare_without_backends_does_nothing() -> None:
    OutputManager({}).prepare({"forecast": signature()}, [])


# ---------------------------------------------------------------------------
# write / flush / close
# ---------------------------------------------------------------------------


def test_write_forwards_and_skips_unwritten_streams(tmp_path: Any) -> None:
    backend = ZarrBackend(file_name=str(tmp_path / "forecast.zarr"))
    manager = prepared(backend)

    manager.write("forecast", field(TIME_A))
    manager.write("scores", field(TIME_B))  # no backend: silently skipped
    manager.flush()

    stored = backend.to_xarray()
    np.testing.assert_allclose(stored["t2m"].sel(time=TIME_A).values, 1.0)
    assert np.isnan(stored["t2m"].sel(time=TIME_B).values).all()


def test_close_keeps_stores_readable(tmp_path: Any) -> None:
    backend = ZarrBackend(file_name=str(tmp_path / "forecast.zarr"))
    manager = prepared(backend)
    manager.write("forecast", field(TIME_A))
    assert manager.close() is None
    assert manager.close() is None
    np.testing.assert_allclose(backend.to_xarray()["t2m"].sel(time=TIME_A).values, 1.0)


def test_close_returns_the_first_failure_after_closing_the_rest() -> None:
    healthy = KVBackend()
    manager = OutputManager({"a": FailingCloseBackend(), "b": healthy})

    error = manager.close()

    assert isinstance(error, RuntimeError)
    assert "commit failed" in str(error)
    with pytest.raises(RuntimeError):  # closed, so writes raise (I11)
        healthy.write(field(TIME_A))
    assert manager.close() is None


# ---------------------------------------------------------------------------
# Distributed writes (X14, X15)
# ---------------------------------------------------------------------------


def test_default_layouts_pass(tmp_path: Any, warnings_log: list[str]) -> None:
    times = np.array([TIME_A, TIME_B], dtype="datetime64[ns]")
    backends = [
        ZarrBackend(file_name=str(tmp_path / "a.zarr")),
        AsyncZarrBackend(
            str(tmp_path / "b.zarr"),
            parallel_coords={"time": times, "ensemble": np.arange(4)},
            shard_coords={"lead_time": 2},
            blocking=True,
        ),
    ]
    for backend in backends:
        prepared(backend, ensemble_items()).check_distributed()
    assert warnings_log == []


@pytest.mark.parametrize(
    "chunks, items",
    [
        ({"time": 2}, deterministic_items()),
        ({}, deterministic_items()),  # no entry: one chunk along the dimension
        ({"time": 1, "ensemble": 2}, ensemble_items()),
    ],
)
def test_zarr_rejects_blocks_wider_than_one(
    tmp_path: Any, chunks: dict[str, int], items: list[WorkItem]
) -> None:
    backend = ZarrBackend(file_name=str(tmp_path / "forecast.zarr"), chunks=chunks)
    with pytest.raises(ValueError, match="blocks"):
        prepared(backend, items).check_distributed()


@pytest.mark.parametrize(
    "parallel, shards",
    [
        (("lead_time",), {}),  # time not parallel: one chunk along it
        (("time",), {"time": 2}),
        (("time", "ensemble"), {"ensemble": 4}),
    ],
)
def test_async_zarr_rejects_blocks_wider_than_one(
    tmp_path: Any, parallel: tuple[str, ...], shards: dict[str, int]
) -> None:
    labels = {
        "time": np.array([TIME_A, TIME_B], dtype="datetime64[ns]"),
        "ensemble": np.arange(4),
        "lead_time": LEADS,
    }
    backend = AsyncZarrBackend(
        str(tmp_path / "forecast.zarr"),
        parallel_coords={dim: labels[dim] for dim in parallel},
        shard_coords=shards,
        blocking=True,
    )
    with pytest.raises(ValueError, match="blocks"):
        prepared(backend, ensemble_items()).check_distributed()


def test_reopened_stores_are_checked_by_their_stored_layout(tmp_path: Any) -> None:
    times = np.array([TIME_A, TIME_B], dtype="datetime64[ns]")
    template = output_template(signature(), {"time": times})
    zarr_path, async_path = str(tmp_path / "a.zarr"), str(tmp_path / "b.zarr")
    ZarrBackend(file_name=zarr_path, chunks={"time": 2}).add_array(template)
    created = AsyncZarrBackend(
        async_path, parallel_coords={"time": times}, shard_coords={"time": 2}
    )
    created.add_array(template)
    created.close()

    # Reopened with safe settings, each store still writes its stored blocks.
    for backend in (
        ZarrBackend(file_name=zarr_path, chunks={"time": 1}),
        AsyncZarrBackend(async_path, parallel_coords={"time": times}, blocking=True),
    ):
        with pytest.raises(ValueError, match="blocks"):
            prepared(backend).check_distributed()


def test_unchecked_backends_warn_once_per_type(warnings_log: list[str]) -> None:
    backends = {"a": KVBackend(), "b": KVBackend(), "c": XarrayBackend()}
    manager = OutputManager(backends)
    manager.prepare(dict.fromkeys(backends, signature()), deterministic_items())
    manager.check_distributed()
    assert len(warnings_log) == 2
    assert all("cannot be checked" in message for message in warnings_log)
