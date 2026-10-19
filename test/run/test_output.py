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
from earth2studio.run import output as output_module
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


@pytest.fixture
def distributed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend a two-rank process group without initializing one.

    `run.work`'s rank helpers are replaced in place, so ordering and the
    shard-ownership checks are testable single-process.
    """
    monkeypatch.setattr(output_module, "rank_info", lambda: (0, 2))


class RecordingBackend:
    """Minimal `IOBackend` that records its lifecycle calls."""

    def __init__(self, fail_close: bool = False) -> None:
        self.templates: list[xr.DataArray] = []
        self.writes: list[xr.DataArray] = []
        self.flushes = 0
        self.closes = 0
        self.fail_close = fail_close

    def add_array(self, template: xr.DataArray) -> None:
        self.templates.append(template)

    def write(self, x: xr.DataArray) -> None:
        self.writes.append(x)

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.closes += 1
        if self.fail_close:
            raise RuntimeError("commit failed")


# ---------------------------------------------------------------------------
# prepare
# ---------------------------------------------------------------------------


def test_prepare_builds_template_from_all_items(tmp_path: Any) -> None:
    backend = ZarrBackend(file_name=str(tmp_path / "forecast.zarr"))
    manager = OutputManager({"forecast": backend})
    manager.prepare({"forecast": signature()}, deterministic_items())

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
    OutputManager({"forecast": backend}).prepare({"forecast": signature()}, items)
    np.testing.assert_array_equal(
        backend.to_xarray()["time"].values,
        np.array([TIME_A, TIME_B], dtype="datetime64[ns]"),
    )


def test_prepare_ensemble_uses_union_of_member_ids(tmp_path: Any) -> None:
    backend = ZarrBackend(file_name=str(tmp_path / "forecast.zarr"))
    # Only one rank's items would hold (0, 1); the store spans every member.
    OutputManager({"forecast": backend}).prepare(
        {"forecast": signature()}, ensemble_items()
    )
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


def test_prepare_is_rank0_first(monkeypatch: pytest.MonkeyPatch) -> None:
    order: list[str] = []
    backend = RecordingBackend()

    def rank0_first(fn: Any) -> Any:
        order.append("rank0")
        result = fn()
        order.append("others")
        return result

    monkeypatch.setattr(output_module, "rank0_first", rank0_first)
    manager = OutputManager({"forecast": backend})
    manager.prepare({"forecast": signature()}, deterministic_items())
    assert order == ["rank0", "others"]
    assert len(backend.templates) == 1


def test_prepare_reopening_a_store_is_a_no_op(tmp_path: Any) -> None:
    path = str(tmp_path / "forecast.zarr")
    first = ZarrBackend(file_name=path)
    manager = OutputManager({"forecast": first})
    manager.prepare({"forecast": signature()}, deterministic_items())
    manager.write("forecast", field(TIME_A))
    manager.close()

    second = ZarrBackend(file_name=path)
    OutputManager({"forecast": second}).prepare(
        {"forecast": signature()}, deterministic_items()
    )
    stored = second.to_xarray()
    assert not np.isnan(stored["t2m"].sel(time=TIME_A)).any()


def test_prepare_rejects_a_backend_without_a_schema() -> None:
    manager = OutputManager({"forcast": RecordingBackend()})
    with pytest.raises(ValueError, match="unknown streams"):
        manager.prepare({"forecast": signature()}, deterministic_items())


def test_prepare_rejects_empty_items() -> None:
    manager = OutputManager({"forecast": RecordingBackend()})
    with pytest.raises(ValueError, match="without work items"):
        manager.prepare({"forecast": signature()}, [])


def test_prepare_without_backends_does_nothing() -> None:
    OutputManager({}).prepare({"forecast": signature()}, [])


# ---------------------------------------------------------------------------
# write / flush / close
# ---------------------------------------------------------------------------


def test_write_forwards_and_skips_unwritten_streams(tmp_path: Any) -> None:
    backend = ZarrBackend(file_name=str(tmp_path / "forecast.zarr"))
    manager = OutputManager({"forecast": backend})
    manager.prepare({"forecast": signature()}, deterministic_items())

    manager.write("forecast", field(TIME_A))
    manager.write("scores", field(TIME_B))  # no backend: silently skipped
    manager.flush()

    stored = backend.to_xarray()
    np.testing.assert_allclose(stored["t2m"].sel(time=TIME_A).values, 1.0)
    assert np.isnan(stored["t2m"].sel(time=TIME_B).values).all()


def test_flush_flushes_every_backend() -> None:
    backends = {"a": RecordingBackend(), "b": RecordingBackend()}
    manager = OutputManager(backends)
    manager.flush()
    assert [backend.flushes for backend in backends.values()] == [1, 1]


def test_close_closes_every_backend_once_and_barriers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barriers: list[int] = []
    monkeypatch.setattr(output_module, "barrier", lambda: barriers.append(1))
    backends = {"a": RecordingBackend(), "b": RecordingBackend()}
    manager = OutputManager(backends)

    assert manager.close() is None
    assert manager.close() is None

    assert [backend.closes for backend in backends.values()] == [1, 1]
    assert barriers == [1]


def test_close_closes_the_rest_after_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    barriers: list[int] = []
    monkeypatch.setattr(output_module, "barrier", lambda: barriers.append(1))
    failing = RecordingBackend(fail_close=True)
    healthy = RecordingBackend()
    manager = OutputManager({"a": failing, "b": healthy})

    error = manager.close()

    assert isinstance(error, RuntimeError)
    assert "commit failed" in str(error)
    assert healthy.closes == 1
    assert barriers == [1]
    assert manager.close() is None


def test_close_keeps_stores_readable(tmp_path: Any) -> None:
    backend = ZarrBackend(file_name=str(tmp_path / "forecast.zarr"))
    manager = OutputManager({"forecast": backend})
    manager.prepare({"forecast": signature()}, deterministic_items())
    manager.write("forecast", field(TIME_A))
    manager.close()
    np.testing.assert_allclose(backend.to_xarray()["t2m"].sel(time=TIME_A).values, 1.0)


# ---------------------------------------------------------------------------
# Shard ownership (X14, X15)
# ---------------------------------------------------------------------------


def test_single_process_skips_the_ownership_check(tmp_path: Any) -> None:
    backend = ZarrBackend(file_name=str(tmp_path / "forecast.zarr"), chunks={"time": 2})
    OutputManager({"forecast": backend}).prepare(
        {"forecast": signature()}, deterministic_items()
    )


def test_default_zarr_chunking_passes(
    tmp_path: Any, distributed: None, warnings_log: list[str]
) -> None:
    deterministic = ZarrBackend(file_name=str(tmp_path / "a.zarr"))
    OutputManager({"forecast": deterministic}).prepare(
        {"forecast": signature()}, deterministic_items()
    )
    ensemble = ZarrBackend(file_name=str(tmp_path / "b.zarr"))
    OutputManager({"forecast": ensemble}).prepare(
        {"forecast": signature()}, ensemble_items()
    )
    assert warnings_log == []


def test_zarr_rejects_shared_time_chunks(tmp_path: Any, distributed: None) -> None:
    backend = ZarrBackend(file_name=str(tmp_path / "forecast.zarr"), chunks={"time": 2})
    with pytest.raises(ValueError, match="'time'"):
        OutputManager({"forecast": backend}).prepare(
            {"forecast": signature()}, deterministic_items()
        )


def test_zarr_rejects_unchunked_time(tmp_path: Any, distributed: None) -> None:
    # No `time` entry stores the dimension in a single chunk.
    backend = ZarrBackend(file_name=str(tmp_path / "forecast.zarr"), chunks={})
    with pytest.raises(ValueError, match="'time'"):
        OutputManager({"forecast": backend}).prepare(
            {"forecast": signature()}, deterministic_items()
        )


def test_zarr_rejects_ensemble_chunks_spanning_member_groups(
    tmp_path: Any, distributed: None
) -> None:
    backend = ZarrBackend(
        file_name=str(tmp_path / "forecast.zarr"), chunks={"time": 1, "ensemble": 4}
    )
    with pytest.raises(ValueError, match="'ensemble'"):
        OutputManager({"forecast": backend}).prepare(
            {"forecast": signature()}, ensemble_items()
        )


def test_zarr_accepts_ensemble_chunks_inside_member_groups(
    tmp_path: Any, distributed: None
) -> None:
    # Groups are (0, 1) and (2, 3), so a chunk of two stays within one group.
    backend = ZarrBackend(
        file_name=str(tmp_path / "forecast.zarr"), chunks={"time": 1, "ensemble": 2}
    )
    OutputManager({"forecast": backend}).prepare(
        {"forecast": signature()}, ensemble_items()
    )


def test_zarr_rejects_a_reopened_store_with_wide_chunks(
    tmp_path: Any, distributed: None
) -> None:
    path = str(tmp_path / "forecast.zarr")
    created = ZarrBackend(file_name=path, chunks={"time": 2})
    created.add_array(
        output_template(
            signature(),
            {"time": np.array([TIME_A, TIME_B], dtype="datetime64[ns]")},
        )
    )
    created.close()
    # The store carries the chunking it was created with, whatever is requested.
    reopened = ZarrBackend(file_name=path, chunks={"time": 1})
    with pytest.raises(ValueError, match="'time'"):
        OutputManager({"forecast": reopened}).prepare(
            {"forecast": signature()}, deterministic_items()
        )


def test_async_zarr_requires_parallel_coords(tmp_path: Any, distributed: None) -> None:
    backend = AsyncZarrBackend(
        str(tmp_path / "forecast.zarr"),
        parallel_coords={"lead_time": LEADS},
        blocking=True,
    )
    with pytest.raises(ValueError, match="parallel_coords"):
        OutputManager({"forecast": backend}).prepare(
            {"forecast": signature()}, deterministic_items()
        )


def test_async_zarr_rejects_shards_spanning_items(
    tmp_path: Any, distributed: None
) -> None:
    times = np.array([TIME_A, TIME_B], dtype="datetime64[ns]")
    backend = AsyncZarrBackend(
        str(tmp_path / "forecast.zarr"),
        parallel_coords={"time": times},
        shard_coords={"time": 2},
        blocking=True,
    )
    with pytest.raises(ValueError, match="'time'"):
        OutputManager({"forecast": backend}).prepare(
            {"forecast": signature()}, deterministic_items()
        )


def test_async_zarr_rejects_a_reopened_store_with_wide_shards(
    tmp_path: Any, distributed: None
) -> None:
    path = str(tmp_path / "forecast.zarr")
    times = np.array([TIME_A, TIME_B], dtype="datetime64[ns]")
    created = AsyncZarrBackend(
        path, parallel_coords={"time": times}, shard_coords={"time": 2}, blocking=True
    )
    created.add_array(
        output_template(signature(), {"time": times}),
    )
    created.close()

    # Reopened without shard_coords, the store still writes its two-time shards.
    reopened = AsyncZarrBackend(path, parallel_coords={"time": times}, blocking=True)
    with pytest.raises(ValueError, match="shards of 2 on 'time'"):
        OutputManager({"forecast": reopened}).prepare(
            {"forecast": signature()}, deterministic_items()
        )


def test_prepare_rejects_items_sharing_a_time_and_member() -> None:
    items = [WorkItem(TIME_A, HORIZON), WorkItem(TIME_A, 2 * HORIZON)]
    with pytest.raises(ValueError, match="Repeated"):
        OutputManager({"forecast": RecordingBackend()}).prepare(
            {"forecast": signature()}, items
        )


def test_async_zarr_accepts_lead_time_shards(
    tmp_path: Any, distributed: None, warnings_log: list[str]
) -> None:
    times = np.array([TIME_A, TIME_B], dtype="datetime64[ns]")
    backend = AsyncZarrBackend(
        str(tmp_path / "forecast.zarr"),
        parallel_coords={"time": times, "lead_time": LEADS},
        shard_coords={"lead_time": 2},
        blocking=True,
    )
    OutputManager({"forecast": backend}).prepare(
        {"forecast": signature()}, deterministic_items()
    )
    assert warnings_log == []


def test_async_zarr_rejects_ensemble_shards_spanning_groups(
    tmp_path: Any, distributed: None
) -> None:
    times = np.array([TIME_A, TIME_B], dtype="datetime64[ns]")
    backend = AsyncZarrBackend(
        str(tmp_path / "forecast.zarr"),
        parallel_coords={"time": times, "ensemble": np.arange(4)},
        shard_coords={"ensemble": 4},
        blocking=True,
    )
    with pytest.raises(ValueError, match="'ensemble'"):
        OutputManager({"forecast": backend}).prepare(
            {"forecast": signature()}, ensemble_items()
        )


@pytest.mark.parametrize("backend_type", [XarrayBackend, KVBackend])
def test_process_local_backends_warn_once(
    backend_type: Any, distributed: None, warnings_log: list[str]
) -> None:
    manager = OutputManager({"a": backend_type(), "b": backend_type()})
    manager.prepare({"a": signature(), "b": signature()}, deterministic_items())
    assert len(warnings_log) == 1
    assert "process-local" in warnings_log[0]


def test_unknown_backends_warn_once(distributed: None, warnings_log: list[str]) -> None:
    manager = OutputManager({"a": RecordingBackend(), "b": RecordingBackend()})
    manager.prepare({"a": signature(), "b": signature()}, deterministic_items())
    assert len(warnings_log) == 1
    assert "cannot be checked" in warnings_log[0]


def test_warnings_are_one_per_backend_type(
    distributed: None, warnings_log: list[str]
) -> None:
    manager = OutputManager({"a": XarrayBackend(), "b": RecordingBackend()})
    manager.prepare({"a": signature(), "b": signature()}, deterministic_items())
    assert len(warnings_log) == 2
