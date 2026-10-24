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

"""Pipeline supervision over the built-in runners.

The model adds one per step and the source returns zeros, so step ``k`` holds
the value ``k`` everywhere. Distributed checks launch this module under
``torchrun`` with two CPU ranks on gloo; see :func:`_torchrun_main`.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import xarray as xr
from loguru import logger
from test_runner import _model, _Ones, _stochastic_model, _WindSpeed, _Zeros

from earth2studio.grids import BilinearRegridder, LatLonGrid
from earth2studio.io import KVBackend, ZarrBackend
from earth2studio.models.dx import Identity
from earth2studio.run import (
    DiagnosticRunner,
    Pipeline,
    PrognosticRunner,
    ProgressStore,
    WorkItem,
    work_items,
)
from earth2studio.run.work import rank0_first, rank_info
from earth2studio.utils.type import CoordinateSystem

T0 = np.datetime64("2026-01-01T00")
T1 = np.datetime64("2026-01-02T00")
T2 = np.datetime64("2026-01-03T00")
T3 = np.datetime64("2026-01-04T00")
H12 = np.timedelta64(12, "h")


class _Recording:
    """Wrap a runner, recording item times and failing at one of them."""

    def __init__(
        self, inner: PrognosticRunner, fail_at: np.datetime64 | None = None
    ) -> None:
        self.inner = inner
        self.fail_at = fail_at
        self.fail_to = False
        self.ran: list[np.datetime64] = []
        self.supports_member_batching = inner.supports_member_batching
        self.seed = inner.seed

    def to(self, device: torch.device) -> _Recording:
        if self.fail_to:
            raise RuntimeError("injected setup failure")
        self.inner.to(device)
        return self

    def output_coords(
        self, final_lead_time: np.timedelta64
    ) -> Mapping[str, CoordinateSystem]:
        return self.inner.output_coords(final_lead_time)

    def data_requests(self, item: WorkItem) -> None:
        return None

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        self.ran.append(item.time)
        for k, step in enumerate(self.inner.run_item(item)):
            if item.time == self.fail_at and k == 1:
                raise RuntimeError("injected failure")
            yield step


class _Scorer:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def begin_item(self, item: WorkItem) -> None:
        self.calls.append(("begin", item.time))

    def update(self, item: WorkItem, stream: str, x: xr.DataArray) -> None:
        self.calls.append(("update", stream, tuple(x.dims), float(x.values.max())))

    def finish_item(self, item: WorkItem) -> None:
        self.calls.append(("finish", item.time))


@pytest.fixture
def info_log() -> Iterator[list[str]]:
    messages: list[str] = []
    handler = logger.add(lambda message: messages.append(message), level="INFO")
    try:
        yield messages
    finally:
        logger.remove(handler)


def _runner(**kwargs: object) -> PrognosticRunner:
    return PrognosticRunner(_model(), _Zeros(), **kwargs)  # type: ignore[arg-type]


# -- integration check 3: one store per stream, any runner ------------------
# The prognostic case is test_runner_output.test_signatures_plan_one_store_per_stream.


def test_diagnostic_runner_is_driven_identically() -> None:
    store = KVBackend()
    runner = DiagnosticRunner({"wind": _WindSpeed()}, _Ones(), step=np.timedelta64(6, "h"))  # type: ignore[dict-item]
    Pipeline(runner, output={"wind": store}, device="cpu").run(
        work_items([T0, T1], H12)
    )
    assert store["ws10m"].shape == (2, 3, 2, 3)
    np.testing.assert_allclose(store["ws10m"], np.hypot(1, 1), rtol=1e-6)


def test_from_model_wraps_a_prognostic_runner() -> None:
    store = KVBackend()
    pipeline = Pipeline.from_model(
        _model(), _Zeros(), seed=7, output={"forecast": store}, device="cpu"  # type: ignore[arg-type]
    )
    assert isinstance(pipeline.runner, PrognosticRunner)
    assert pipeline.runner.seed == 7
    pipeline.run(work_items([T0], H12))
    np.testing.assert_array_equal(store["v10m"][0, :, 1, 2], np.arange(3))


def test_empty_items_do_nothing() -> None:
    runner = _Recording(_runner())
    Pipeline(runner, output={"forecast": KVBackend()}, device="cpu").run([])
    assert runner.ran == []


# -- per-stream settings ------------------------------------------------------


def test_output_variables_filter_each_stream() -> None:
    forecast, copy = KVBackend(), KVBackend()
    Pipeline(
        _runner(diagnostics={"copy": Identity()}),
        output={"forecast": forecast, "copy": copy},
        output_variables={"forecast": ["v10m"]},
        device="cpu",
    ).run(work_items([T0], H12))
    assert "v10m" in forecast and "u10m" not in forecast
    assert "v10m" in copy and "u10m" in copy  # unfiltered stream keeps all


def test_stream_without_backend_is_logged_once(info_log: list[str]) -> None:
    Pipeline(
        _runner(diagnostics={"copy": Identity()}),
        output={"forecast": KVBackend()},
        device="cpu",
    ).run(work_items([T0, T1], H12))
    assert sum("'copy' has no output backend" in m for m in info_log) == 1


@pytest.mark.parametrize("setting", ["output", "output_variables", "regridder"])
def test_unknown_stream_names_are_rejected(setting: str) -> None:
    grid = LatLonGrid(np.array([10.0, 0.0]), np.array([0.0, 10.0, 20.0]))
    values = {
        "output": KVBackend(),
        "output_variables": ["u10m"],
        "regridder": BilinearRegridder(grid, grid),
    }
    kwargs: dict = {"output": {"forecast": KVBackend()}}
    kwargs[setting] = {"forcast": values[setting]}
    runner = _Recording(_runner())
    with pytest.raises(ValueError, match="unknown streams"):
        Pipeline(runner, device="cpu", **kwargs).run(work_items([T0], H12))
    assert runner.ran == []


def test_unknown_output_variable_is_rejected() -> None:
    pipeline = Pipeline(
        _runner(),
        output={"forecast": KVBackend()},
        output_variables={"forecast": ["t2m"]},
        device="cpu",
    )
    with pytest.raises(ValueError, match="unknown variables"):
        pipeline.run(work_items([T0], H12))


def test_regridder_stores_the_target_grid() -> None:
    source = LatLonGrid(np.array([10.0, 0.0]), np.array([0.0, 10.0, 20.0]))
    target = LatLonGrid(np.array([10.0, 5.0, 0.0]), np.arange(0.0, 21.0, 5.0))
    store = KVBackend()
    scorer = _Scorer()
    Pipeline(
        _runner(),
        output={"forecast": store},
        scorer=scorer,
        output_variables={"forecast": ["u10m"]},
        regridder={"forecast": BilinearRegridder(source, target)},
        device="cpu",
    ).run(work_items([T0], H12))

    assert store["u10m"].shape == (1, 3, 3, 5)
    np.testing.assert_array_equal(store.coords["lat"], [10.0, 5.0, 0.0])
    # Bilinear weights of a constant field reproduce it.
    np.testing.assert_allclose(store["u10m"][0, :, 1, 1], np.arange(3))
    # The scorer sees the filtered, regridded field.
    updates = [call for call in scorer.calls if call[0] == "update"]
    assert updates[0][2] == ("time", "lead_time", "variable", "lat", "lon")


def test_regridder_on_another_grid_is_rejected() -> None:
    other = LatLonGrid(np.array([20.0, 10.0]), np.array([0.0, 10.0, 20.0]))
    pipeline = Pipeline(
        _runner(),
        output={"forecast": KVBackend()},
        regridder={"forecast": BilinearRegridder(other, other)},
        device="cpu",
    )
    with pytest.raises(ValueError, match="source_grid"):
        pipeline.run(work_items([T0], H12))


# -- items, members and scoring -----------------------------------------------


def test_ensemble_dimension_is_added_per_member() -> None:
    store = KVBackend()
    runner = PrognosticRunner(_stochastic_model(), _Zeros())  # type: ignore[arg-type]
    Pipeline(runner, output={"forecast": store}, device="cpu").run(
        work_items([T0], H12, members=[3, 5])
    )
    assert store["u10m"].shape == (2, 1, 3, 2, 3)  # ensemble, time, lead, lat, lon
    np.testing.assert_array_equal(store.coords["ensemble"], [3, 5])
    np.testing.assert_array_equal(store["u10m"][1, 0, :, 0, 0], np.arange(3))


def test_member_batches_need_a_batching_runner() -> None:
    runner = _Recording(_runner())
    pipeline = Pipeline(runner, output={"forecast": KVBackend()}, device="cpu")
    with pytest.raises(ValueError, match="does not batch"):
        pipeline.run(work_items([T0], H12, members=2, member_batch=2))
    assert runner.ran == []


def test_unreachable_final_lead_time_fails_before_running() -> None:
    runner = _Recording(_runner())
    items = [WorkItem(T0, H12), WorkItem(T1, np.timedelta64(9, "h"))]
    pipeline = Pipeline(runner, output={"forecast": KVBackend()}, device="cpu")
    with pytest.raises(ValueError, match="not a multiple"):
        pipeline.run(items)
    assert runner.ran == []


def test_scorer_only_run_scores_every_stream() -> None:
    scorer = _Scorer()
    Pipeline(
        _runner(diagnostics={"copy": Identity()}), scorer=scorer, device="cpu"
    ).run(work_items([T0], np.timedelta64(6, "h")))
    assert scorer.calls[0] == ("begin", T0)
    assert scorer.calls[-1] == ("finish", T0)
    updates = [call[1] for call in scorer.calls if call[0] == "update"]
    assert updates == ["forecast", "copy"] * 2


def test_output_or_scorer_is_required() -> None:
    with pytest.raises(ValueError, match="output mapping, a scorer"):
        Pipeline(_runner())


# -- progress and failure -----------------------------------------------------


def test_failed_item_is_not_marked_and_resume_reruns_it(tmp_path: Path) -> None:
    items = work_items([T0, T1, T2], H12)
    path = str(tmp_path / "forecast.zarr")
    failing = _Recording(_runner(), fail_at=T1)
    with pytest.raises(RuntimeError, match="injected failure"):
        Pipeline(
            failing,
            output={"forecast": ZarrBackend(path)},
            progress=ProgressStore(tmp_path / "progress"),
            device="cpu",
        ).run(items)
    assert failing.ran == [T0, T1]  # stops at the failure

    resumed = _Recording(_runner())
    store = ZarrBackend(path)
    Pipeline(
        resumed,
        output={"forecast": store},
        progress=ProgressStore(tmp_path / "progress"),
        device="cpu",
    ).run(items)
    assert resumed.ran == [T1, T2]
    for t in range(3):
        np.testing.assert_array_equal(store["u10m"][t, :, 0, 0], np.arange(3))

    again = _Recording(_runner())
    Pipeline(
        again,
        output={"forecast": ZarrBackend(path)},
        progress=ProgressStore(tmp_path / "progress"),
        device="cpu",
    ).run(items)
    assert again.ran == []


class _FailingClose(ProgressStore):
    def close(self) -> None:
        super().close()
        raise OSError("progress close failed")


def test_progress_close_failure_is_reported(tmp_path: Path) -> None:
    """A rank-local close error reaches the final failure check and raises."""
    store = KVBackend()
    with pytest.raises(OSError, match="progress close failed"):
        Pipeline(
            _runner(),
            output={"forecast": store},
            progress=_FailingClose(tmp_path),
            device="cpu",
        ).run(work_items([T0], H12))
    assert store["u10m"].shape == (1, 3, 2, 3)  # the run itself completed


def test_manifest_mismatch_is_rejected_unless_overwritten(tmp_path: Path) -> None:
    items = work_items([T0], H12)
    progress = tmp_path / "progress"
    Pipeline(
        _runner(),
        output={"forecast": KVBackend()},
        progress=ProgressStore(progress),
        device="cpu",
    ).run(items)

    backend = KVBackend()
    reseeded = Pipeline(
        _runner(seed=1),
        output={"forecast": backend},
        progress=ProgressStore(progress),
        device="cpu",
    )
    with pytest.raises(ValueError, match="differing keys: seed"):
        reseeded.run(items)
    assert reseeded._outputs is not None and reseeded._outputs._closed

    # Adding output to a scorer-only run changes the manifest, so finished items
    # are not skipped with their outputs missing.
    scored = tmp_path / "scored"
    Pipeline(
        _runner(), scorer=_Scorer(), progress=ProgressStore(scored), device="cpu"
    ).run(items)
    with pytest.raises(ValueError, match="differing keys: streams"):
        Pipeline(
            _runner(),
            output={"forecast": KVBackend()},
            progress=ProgressStore(scored),
            device="cpu",
        ).run(items)

    # Writing a new stream also changes the manifest.
    with pytest.raises(ValueError, match="differing keys: streams"):
        Pipeline(
            _runner(diagnostics={"copy": Identity()}),
            output={"forecast": KVBackend(), "copy": KVBackend()},
            progress=ProgressStore(progress),
            device="cpu",
        ).run(items)

    runner = _Recording(_runner(seed=1))
    Pipeline(
        runner,
        output={"forecast": KVBackend()},
        progress=ProgressStore(progress, overwrite=True),
        device="cpu",
    ).run(items)
    assert runner.ran == [T0]


def test_sync_joins_each_round_and_gates_progress(tmp_path: Path) -> None:
    """Single-process, every item is its own round; a failed sync is not done."""
    items = work_items([T0, T1, T2], H12)
    calls: list[np.datetime64 | None] = []

    def sync(item: WorkItem | None) -> None:
        calls.append(None if item is None else item.time)
        if item is not None and item.time == T1 and len(calls) == 2:
            raise RuntimeError("sync failed")

    def pipeline(runner: _Recording) -> Pipeline:
        return Pipeline(
            runner,
            output={"forecast": KVBackend()},
            progress=ProgressStore(tmp_path),
            sync=sync,
            device="cpu",
        )

    first = _Recording(_runner())
    with pytest.raises(RuntimeError, match="sync failed"):
        pipeline(first).run(items)
    assert first.ran == [T0, T1] and calls == [T0, T1]

    # T1 ran but its sync failed, so it is not marked done.
    resumed = _Recording(_runner())
    pipeline(resumed).run(items)
    assert resumed.ran == [T1, T2] and calls[2:] == [T1, T2]


def test_manifest_records_streams_seed_and_tags(tmp_path: Path) -> None:
    Pipeline(
        _runner(seed=3, diagnostics={"copy": Identity()}),
        output={"forecast": KVBackend()},
        output_variables={"forecast": ["u10m"]},
        progress=ProgressStore(tmp_path, tags={"model": "tiny"}),
        device="cpu",
    ).run(work_items([T0], H12))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["version"] == 1
    assert manifest["seed"] == 3
    assert manifest["tags"] == {"model": "tiny"}
    forecast = manifest["streams"]["forecast"]
    assert forecast["labels"]["variable"] == ["u10m"]
    assert len(forecast["labels"]["lead_time"]) == 3
    assert forecast["written"] and not manifest["streams"]["copy"]["written"]
    assert forecast["dtype"] == "float32"


# -- integration check 6: resume across a world-size change ------------------


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _torchrun(out: Path, *args: str) -> subprocess.CompletedProcess:
    env = {**os.environ, "OMP_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    if sys.platform == "darwin":
        env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    # Rendezvous on loopback: the host name may not resolve to a local address.
    return subprocess.run(  # noqa: S603 - fixed interpreter and script
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nnodes=1",
            "--nproc_per_node=2",
            "--master_addr=127.0.0.1",
            f"--master_port={_free_port()}",
            __file__,
            str(out),
            *args,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )


def _label(time: np.datetime64) -> str:
    return str(time.astype("datetime64[s]"))


def _rank_results(out: Path, phase: str) -> list[dict]:
    return [
        json.loads((out / f"{phase}-rank{rank}.json").read_text()) for rank in (0, 1)
    ]


def test_resume_survives_a_world_size_change(tmp_path: Path) -> None:
    items = work_items([T0, T1, T2, T3], H12)

    # Two ranks: rank 0 runs T0 and T1; rank 1 fails at T2, so both raise.
    result = _torchrun(tmp_path, "first", str(T2))
    assert result.returncode != 0, result.stderr
    first = _rank_results(tmp_path, "first")
    assert first[0]["ran"] == [_label(T0), _label(T1)]
    assert first[0]["error"] == "RuntimeError: Another rank failed"
    assert first[1]["ran"] == [_label(T2)]
    assert first[1]["error"] == "RuntimeError: injected failure"

    # One rank resumes only what is left, in the stores rank 0 created.
    runner = _Recording(_runner())
    store = ZarrBackend(str(tmp_path / "forecast.zarr"))
    Pipeline(
        runner,
        output={"forecast": store},
        progress=ProgressStore(tmp_path / "progress"),
        device="cpu",
    ).run(items)
    assert runner.ran == [T2, T3]
    for t in range(4):
        np.testing.assert_array_equal(store["u10m"][t, :, 0, 0], np.arange(3))

    # Two ranks again: nothing is left, and both idle ranks finish cleanly.
    result = _torchrun(tmp_path, "last")
    assert result.returncode == 0, result.stderr
    assert [r["ran"] for r in _rank_results(tmp_path, "last")] == [[], []]


def test_sync_rounds_across_ranks(tmp_path: Path) -> None:
    """Three items on two ranks: rank 1 idles through round 2 but joins sync."""
    # Rank 1 fails its first item: both ranks stop in round 1, nothing is done.
    result = _torchrun(tmp_path, "sync_fail", str(T2))
    assert result.returncode != 0
    failed = _rank_results(tmp_path, "sync_fail")
    assert [r["ran"] for r in failed] == [[_label(T0)], [_label(T2)]]
    assert [r["error"] for r in failed] == [
        "RuntimeError: Another rank failed",
        "RuntimeError: injected failure",
    ]
    assert [r["syncs"] for r in failed] == [[], []]

    # Rank 0's sync fails after the collective, once rank 1's sync returned:
    # rank 1 must not record its item done before the failure is agreed.
    result = _torchrun(tmp_path, "sync_save")
    assert result.returncode != 0
    saved = _rank_results(tmp_path, "sync_save")
    assert [r["error"] for r in saved] == [
        "RuntimeError: injected sync failure",
        "RuntimeError: Another rank failed",
    ]
    assert saved[1]["syncs"] == [[_label(T2), 2]]

    # The rerun redoes every item; each sync sees how many ranks had an item.
    result = _torchrun(tmp_path, "sync")
    assert result.returncode == 0, result.stderr
    done = _rank_results(tmp_path, "sync")
    assert done[0]["syncs"] == [[_label(T0), 2], [_label(T1), 1]]
    assert done[1]["syncs"] == [[_label(T2), 2], [None, 1]]


def test_setup_failure_on_one_rank_raises_on_every_rank(tmp_path: Path) -> None:
    """Placing the runner fails on rank 1 only; rank 0 must not hang."""
    result = _torchrun(tmp_path, "setup")
    assert result.returncode != 0
    errors = [r["error"] for r in _rank_results(tmp_path, "setup")]
    assert errors == [
        "RuntimeError: Another rank failed",
        "RuntimeError: injected setup failure",
    ]
    assert not (tmp_path / "progress").exists()  # failed before progress setup


def test_ranks_with_different_items_are_rejected(tmp_path: Path) -> None:
    result = _torchrun(tmp_path, "mismatch")
    assert result.returncode != 0
    errors = [r["error"] for r in _rank_results(tmp_path, "mismatch")]
    assert errors[0] == "RuntimeError: Another rank failed"
    assert errors[1].startswith("ValueError: Every rank must pass identical")
    assert not (tmp_path / "progress").exists()  # rejected before any setup


def _torchrun_main() -> int:
    """One rank of a distributed run: ``torchrun ... test_pipeline.py OUT PHASE``.

    Each rank writes ``OUT/PHASE-rank{N}.json`` with the item times it ran and
    the error it raised, then exits non-zero if it raised. An optional third
    argument injects a failure at that item time; phase ``mismatch`` gives rank
    1 a different item list, and phase ``setup`` fails placing rank 1's runner.
    Phases starting ``sync`` run three items with a ``sync`` hook that counts,
    across ranks, the ranks that had an item in each round; in ``sync_save``,
    rank 0's sync then fails.
    """
    out, phase = Path(sys.argv[1]), sys.argv[2]
    fail_at = np.datetime64(sys.argv[3]) if len(sys.argv) > 3 else None
    dist.init_process_group("gloo")
    rank, _ = rank_info()

    times = [T0, T1, T2, T3]
    if phase.startswith("sync") or (phase == "mismatch" and rank == 1):
        times = times[:3]
    syncs: list[tuple[str | None, int]] = []

    def sync(item: WorkItem | None) -> None:
        count = torch.tensor([0 if item is None else 1])
        dist.all_reduce(count)
        syncs.append((None if item is None else _label(item.time), int(count)))
        if phase == "sync_save" and rank == 0:
            raise RuntimeError("injected sync failure")

    runner = _Recording(_runner(), fail_at=fail_at)
    runner.fail_to = phase == "setup" and rank == 1
    output = rank0_first(lambda: {"forecast": ZarrBackend(str(out / "forecast.zarr"))})
    pipeline = Pipeline(
        runner,
        output=output,
        progress=ProgressStore(out / "progress"),
        device="cpu",
        sync=sync if phase.startswith("sync") else None,
    )

    error = None
    try:
        pipeline.run(work_items(times, H12))
    except Exception as err:  # noqa: BLE001 - recorded for the test
        error = f"{type(err).__name__}: {str(err).splitlines()[0].split(';')[0]}"
    result = {
        "ran": [_label(time) for time in runner.ran],
        "error": error,
        "syncs": syncs,
    }
    (out / f"{phase}-rank{rank}.json").write_text(json.dumps(result))
    dist.destroy_process_group()
    return 1 if error else 0


if __name__ == "__main__":
    sys.exit(_torchrun_main())
