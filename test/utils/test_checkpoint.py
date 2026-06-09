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

import json
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pytest
import torch

import earth2studio.run as run
from earth2studio.data import Random
from earth2studio.io import ZarrBackend
from earth2studio.models.dx import Identity
from earth2studio.models.px import Persistence
from earth2studio.perturbation import Zero
from earth2studio.utils.checkpoint import (
    Checkpoint,
    CheckpointError,
    CheckpointSerializationError,
    CheckpointStateCollision,
    CheckpointStateSchemaError,
    bind_checkpoint_state,
    default_checkpoint_path,
)
from earth2studio.utils.time import to_time_array


@dataclass
class NestedState:
    label: str = "inner"
    count: int = 0


@dataclass
class ToyState:
    calls: int = 0
    rng: torch.Tensor | None = None
    weights: np.ndarray = field(
        default_factory=lambda: np.asarray([1.0, 2.0], dtype=np.float32)
    )
    timestamp: np.datetime64 = np.datetime64("2026-06-08T00", "h")
    delta: np.timedelta64 = np.timedelta64(0, "h")
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float32
    np_dtype: np.dtype = np.dtype("float32")
    created: datetime = datetime(2024, 1, 1, tzinfo=timezone.utc)
    day: date = date(2024, 1, 2)
    window: timedelta = timedelta(hours=2)
    items: list = field(default_factory=lambda: [True, np.float32(2.0)])
    pair: tuple = ("left", 1)
    mapping: dict = field(default_factory=lambda: {"value": np.int64(3)})
    nested: NestedState = field(default_factory=NestedState)
    loose: object | None = None


@dataclass
class OtherState:
    value: int = 0


@dataclass
class BadState:
    value: object = object()


@dataclass
class RequiredState:
    value: int



def _coords(hours: int):
    return OrderedDict({"lead_time": np.asarray([np.timedelta64(hours, "h")])})


def test_bind_round_trip_hydrates_dataclass_and_catalog(tmp_path):
    checkpoint = Checkpoint("forecast", path=tmp_path, mode="overwrite")
    time = np.asarray([np.datetime64("2024-01-01T00")])

    with checkpoint.select(time=time, ensemble=0) as ckpt:
        state = bind_checkpoint_state(ToyState())
        state.calls = 7
        state.rng = torch.arange(4, dtype=torch.uint8)
        state.weights = np.asarray([3.0, 4.0], dtype=np.float32)
        state.delta = np.timedelta64(6, "h")
        state.nested.count = 2
        state.loose = NestedState(count=5)
        ckpt.write(coords=_coords(6), artifacts={"sample": 3})

    checkpoint = Checkpoint("forecast", path=tmp_path)
    with checkpoint.select(time=time, ensemble=0) as ckpt:
        restored = bind_checkpoint_state(ToyState())

        assert ckpt.exists
        assert ckpt.lead_time == np.timedelta64(6, "h")
        assert restored.calls == 7
        assert torch.equal(restored.rng, torch.arange(4, dtype=torch.uint8))
        assert np.array_equal(restored.weights, np.asarray([3.0, 4.0], dtype=np.float32))
        assert restored.delta == np.timedelta64(6, "h")
        assert restored.device == torch.device("cpu")
        assert restored.dtype == torch.float32
        assert restored.np_dtype == np.dtype("float32")
        assert restored.created == datetime(2024, 1, 1, tzinfo=timezone.utc)
        assert restored.day == date(2024, 1, 2)
        assert restored.window == timedelta(hours=2)
        assert restored.items == [True, 2.0]
        assert restored.pair == ("left", 1)
        assert restored.mapping == {"value": 3}
        assert restored.nested.count == 2
        assert restored.loose == {"label": "inner", "count": 5}
        assert ckpt.commit_id is not None
        assert ckpt.artifact("sample") == 3

    text = repr(checkpoint)
    assert 'Checkpoint("forecast")' in text
    assert "ensemble" in text
    assert "6 hours" in text


def test_duplicate_state_type_errors_but_different_selections_are_independent(tmp_path):
    checkpoint = Checkpoint("forecast", path=tmp_path)

    with checkpoint.select(time="2024-01-01") as ckpt:
        bind_checkpoint_state(ToyState())
        with pytest.raises(CheckpointStateCollision):
            bind_checkpoint_state(ToyState())
        ckpt.flush(coords=_coords(0))

    with checkpoint.select(time="2024-01-02") as ckpt:
        state = bind_checkpoint_state(ToyState())
        state.calls = 2
        ckpt.flush(coords=_coords(6))

    assert len(checkpoint.catalog) == 2


def test_write_interval_overwrite_and_manual_flush_prune_old_commits(tmp_path):
    checkpoint = Checkpoint(
        "forecast", path=tmp_path, mode="overwrite", flush_interval=2
    )

    with checkpoint.select(time="2024-01-01") as ckpt:
        state = bind_checkpoint_state(ToyState())
        state.calls = 1
        assert ckpt.write(coords=_coords(6)) is None

        state.calls = 2
        first = ckpt.write(coords=_coords(12))
        assert first is not None

        state.calls = 3
        assert ckpt.write(coords=_coords(18)) is None
        final = ckpt.flush()

    assert final.write_count == 3
    assert len(checkpoint.catalog) == 1
    commits = list((checkpoint.rank_path / "commits").iterdir())
    assert [commit.name for commit in commits] == [final.commit_id]

    with checkpoint.select(time="2024-01-01") as ckpt:
        state = bind_checkpoint_state(ToyState())
        assert ckpt.lead_time == np.timedelta64(18, "h")
        assert state.calls == 3


def test_append_keep_last_and_positional_selection(tmp_path):
    checkpoint = Checkpoint(
        "forecast", path=tmp_path, mode="append", flush_interval=1, keep_last=2
    )

    for hours in (6, 12, 18):
        with checkpoint.select(time="2024-01-01", ensemble=0) as ckpt:
            bind_checkpoint_state(ToyState()).calls = hours
            ckpt.write(coords=_coords(hours))

    assert len(checkpoint.catalog) == 2
    assert checkpoint.select(-1).lead_time == np.timedelta64(18, "h")
    assert checkpoint.select(-2).lead_time == np.timedelta64(12, "h")
    assert checkpoint.select(time=-1, ensemble=0).lead_time == np.timedelta64(18, "h")


def test_defensive_paths_and_catalog_rebuild(tmp_path):
    plain = ToyState()
    assert bind_checkpoint_state(plain) is plain
    with pytest.raises(TypeError):
        bind_checkpoint_state(object())

    with pytest.raises(ValueError):
        Checkpoint("bad", path=tmp_path, mode="bad")
    with pytest.raises(ValueError):
        Checkpoint("bad", path=tmp_path, flush_interval=0)
    with pytest.raises(ValueError):
        Checkpoint("bad", path=tmp_path, keep_last=0)

    checkpoint = Checkpoint("forecast", path=tmp_path / "catalog", mode="append")
    assert "catalog: empty" in repr(checkpoint)
    with pytest.raises(IndexError):
        checkpoint.select(-1)
    assert checkpoint.select(time="missing").artifacts == {}
    assert not checkpoint.select(time="missing")
    with pytest.raises(CheckpointError):
        checkpoint.select(time="missing")._commit_path

    with pytest.raises(CheckpointSerializationError):
        checkpoint.select(meta={1: 2})
    with pytest.raises(CheckpointSerializationError):
        checkpoint.select(meta=object())
    with pytest.raises(CheckpointSerializationError):
        checkpoint.select(meta=np.asarray([object()], dtype=object))

    with checkpoint.select(time="2024-01-01") as ckpt:
        with pytest.raises(CheckpointSerializationError):
            ckpt.flush(artifacts={1: 2})
        with pytest.raises(CheckpointSerializationError):
            ckpt.flush(artifacts={"bad": {1: 2}})
        bind_checkpoint_state(RequiredState(1))
        ckpt.flush(coords={"lead_time": torch.tensor([6])})
        ckpt.flush(coords={})

    meta_checkpoint = Checkpoint("metadata", path=tmp_path / "metadata")
    with meta_checkpoint.select(
        day=date(2024, 1, 1),
        window=timedelta(hours=2),
        device=torch.device("cpu"),
        dtype=torch.float32,
        np_dtype=np.dtype("float32"),
        values=[np.int64(1)],
        meta={"ok": np.float32(1.0)},
    ) as ckpt:
        ckpt.flush(coords={"lead_time": np.asarray([1, 2])})
    assert Checkpoint("metadata", path=tmp_path / "metadata").select(-1).exists

    assert len(checkpoint.catalog) == 2
    (checkpoint.rank_path / "catalog.json").write_text("{")
    assert len(Checkpoint("forecast", path=tmp_path / "catalog").catalog) == 2

    (checkpoint.rank_path / "catalog.json").unlink()
    bad_commit = checkpoint.rank_path / "commits" / "bad"
    bad_commit.mkdir()
    (bad_commit / "manifest.json").write_text("{")
    assert len(Checkpoint("forecast", path=tmp_path / "catalog").catalog) == 2


def test_artifacts_round_trip_and_unsupported_objects_reject(tmp_path):
    checkpoint = Checkpoint("forecast", path=tmp_path)

    with checkpoint.select(time="2024-01-01") as ckpt:
        ckpt.write(
            coords=_coords(6),
            artifacts={
                "mask": torch.tensor([True, False]),
                "scores": np.asarray([1.0, 2.0], dtype=np.float32),
                "meta": {"name": "sample"},
            },
        )

    selected = checkpoint.select(time="2024-01-01")
    assert torch.equal(selected.artifact("mask"), torch.tensor([True, False]))
    assert np.array_equal(
        selected.artifact("scores"), np.asarray([1.0, 2.0], dtype=np.float32)
    )
    assert selected.artifact("meta") == {"name": "sample"}
    with pytest.raises(KeyError):
        selected.artifact("missing")

    with checkpoint.select(time="bad") as ckpt:
        bind_checkpoint_state(BadState())
        with pytest.raises(CheckpointSerializationError):
            ckpt.flush(coords=_coords(0))

    with checkpoint.select(time="bad-array") as ckpt:
        with pytest.raises(CheckpointSerializationError):
            ckpt.flush(artifacts={"bad": np.asarray([object()], dtype=object)})


def test_schema_mismatch_errors_before_hydration(tmp_path):
    checkpoint = Checkpoint("forecast", path=tmp_path)

    with checkpoint.select(time="2024-01-01") as ckpt:
        state = bind_checkpoint_state(ToyState())
        state.calls = 4
        ckpt.flush(coords=_coords(6))

    metadata_path = next((checkpoint.rank_path / "commits").glob("*/states/*/metadata.json"))
    original_metadata = json.loads(metadata_path.read_text())

    metadata = original_metadata.copy()
    metadata["state_id"] = "bad"
    metadata_path.write_text(json.dumps(metadata))
    with Checkpoint("forecast", path=tmp_path).select(time="2024-01-01"):
        with pytest.raises(CheckpointStateSchemaError):
            bind_checkpoint_state(ToyState())

    metadata = original_metadata.copy()
    metadata["fields"] = metadata["fields"].copy()
    metadata["fields"].pop("calls")
    metadata_path.write_text(json.dumps(metadata))
    with Checkpoint("forecast", path=tmp_path).select(time="2024-01-01"):
        with pytest.raises(CheckpointStateSchemaError):
            bind_checkpoint_state(ToyState())

    metadata = original_metadata.copy()
    metadata["schema_hash"] = "bad"
    metadata_path.write_text(json.dumps(metadata))
    with Checkpoint("forecast", path=tmp_path).select(time="2024-01-01"):
        with pytest.raises(CheckpointStateSchemaError):
            bind_checkpoint_state(ToyState())


def test_default_path_and_rank_directory_detection(tmp_path, monkeypatch):
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))
    monkeypatch.setenv("RANK", "2")
    monkeypatch.setenv("WORLD_SIZE", "4")

    assert default_checkpoint_path("forecast") == tmp_path / "checkpoints" / "forecast"

    checkpoint = Checkpoint("forecast")
    with checkpoint.select(time="2024-01-01") as ckpt:
        ckpt.flush(coords=_coords(0))

    assert checkpoint.rank == 2
    assert checkpoint.world_size == 4
    assert checkpoint.rank_path.name == "rank_000002"
    assert checkpoint.catalog[0].rank == 2


def test_deterministic_workflow_records_checkpoint(tmp_path):
    coords = OrderedDict([("lat", np.arange(2)), ("lon", np.arange(3))])
    variables = ["u10m", "v10m"]
    data = Random(domain_coords=coords)
    model = Persistence(variables, coords)
    io = ZarrBackend()
    checkpoint = Checkpoint(
        "deterministic", path=tmp_path, mode="overwrite", flush_interval=2
    )

    run.deterministic(
        ["2024-01-01"],
        3,
        model,
        data,
        io,
        device=torch.device("cpu"),
        verbose=False,
        checkpoint=checkpoint,
    )

    selected = checkpoint.select(-1)
    assert selected.exists
    assert selected.lead_time == np.timedelta64(18, "h")
    assert selected.write_count == 4


def test_deterministic_workflow_resumes_from_checkpoint(tmp_path):
    coords = OrderedDict([("lat", np.arange(2)), ("lon", np.arange(3))])
    variables = ["u10m", "v10m"]
    io = ZarrBackend()
    io.add_array(
        OrderedDict(
            {
                "time": np.asarray(["2024-01-01T00"], dtype="datetime64[ns]"),
                "lead_time": np.asarray(
                    [np.timedelta64(6 * i, "h") for i in range(4)]
                ),
                **coords,
            }
        ),
        variables,
    )
    checkpoint = Checkpoint(
        "deterministic", path=tmp_path, mode="append", flush_interval=1
    )

    data = Random(domain_coords=coords)
    model = Persistence(variables, coords)
    run.deterministic(
        ["2024-01-01"],
        1,
        model,
        data,
        io,
        device=torch.device("cpu"),
        verbose=False,
        checkpoint=checkpoint,
    )
    assert checkpoint.select(-1).lead_time == np.timedelta64(6, "h")

    with checkpoint.select(-1) as ckpt:
        data = Random(domain_coords=coords)
        model = Persistence(variables, coords)
        run.deterministic(
            ["2024-01-01"],
            3,
            model,
            data,
            io,
            device=torch.device("cpu"),
            verbose=False,
            checkpoint=ckpt,
        )

    selected = checkpoint.select(-1)
    assert selected.lead_time == np.timedelta64(18, "h")
    assert selected.write_count == 4
    assert io["u10m"].shape[1] == 4



def test_diagnostic_workflow_resumes_from_checkpoint(tmp_path):
    coords = OrderedDict([("lat", np.arange(2)), ("lon", np.arange(3))])
    variables = ["u10m", "v10m"]
    io = ZarrBackend()
    io.add_array(
        OrderedDict(
            {
                "time": np.asarray(["2024-01-01T00"], dtype="datetime64[ns]"),
                "lead_time": np.asarray(
                    [np.timedelta64(6 * i, "h") for i in range(4)]
                ),
                **coords,
            }
        ),
        variables,
    )
    checkpoint = Checkpoint(
        "diagnostic", path=tmp_path, mode="append", flush_interval=1
    )

    with checkpoint.select(time=to_time_array(["2024-01-01"])) as ckpt:
        data = Random(domain_coords=coords)
        model = Persistence(variables, coords)
        diagnostic = Identity()
        run.diagnostic(
            ["2024-01-01"],
            1,
            model,
            diagnostic,
            data,
            io,
            device=torch.device("cpu"),
            verbose=False,
            checkpoint=ckpt,
        )
    assert checkpoint.select(-1).lead_time == np.timedelta64(6, "h")

    with checkpoint.select(-1) as ckpt:
        data = Random(domain_coords=coords)
        model = Persistence(variables, coords)
        diagnostic = Identity()
        run.diagnostic(
            ["2024-01-01"],
            3,
            model,
            diagnostic,
            data,
            io,
            device=torch.device("cpu"),
            verbose=False,
            checkpoint=ckpt,
        )

    selected = checkpoint.select(-1)
    assert selected.lead_time == np.timedelta64(18, "h")
    assert selected.write_count == 4
    assert io["u10m"].shape[1] == 4


def test_ensemble_workflow_resumes_each_batch_from_checkpoint(tmp_path):
    coords = OrderedDict([("lat", np.arange(2)), ("lon", np.arange(3))])
    variables = ["u10m", "v10m"]
    nensemble = 2
    io = ZarrBackend()
    io.add_array(
        OrderedDict(
            {
                "ensemble": np.arange(nensemble),
                "time": np.asarray(["2024-01-01T00"], dtype="datetime64[ns]"),
                "lead_time": np.asarray(
                    [np.timedelta64(6 * i, "h") for i in range(4)]
                ),
                **coords,
            }
        ),
        variables,
    )
    checkpoint = Checkpoint("ensemble", path=tmp_path, mode="append", flush_interval=1)

    run.ensemble(
        ["2024-01-01"],
        1,
        nensemble,
        Persistence(variables, coords),
        Random(domain_coords=coords),
        io,
        Zero(),
        batch_size=1,
        device=torch.device("cpu"),
        verbose=False,
        checkpoint=checkpoint,
    )

    run.ensemble(
        ["2024-01-01"],
        3,
        nensemble,
        Persistence(variables, coords),
        Random(domain_coords=coords),
        io,
        Zero(),
        batch_size=1,
        device=torch.device("cpu"),
        verbose=False,
        checkpoint=checkpoint,
    )

    time = to_time_array(["2024-01-01"])
    for batch_id in range(nensemble):
        selected = checkpoint.select(time=time, ensemble_batch=batch_id)
        assert selected.lead_time == np.timedelta64(18, "h")
        assert selected.write_count == 4
    assert len(checkpoint.catalog) == 8
    assert io["u10m"].shape[:3] == (nensemble, 1, 4)
