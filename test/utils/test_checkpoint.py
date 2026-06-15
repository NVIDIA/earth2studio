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
    NO_CHECKPOINT,
    Checkpoint,
    CheckpointError,
    CheckpointSerializationError,
    CheckpointState,
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
    tensor_device: torch.device = torch.device("cpu")
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


def test_checkpoint_contexts_and_no_checkpoint_session(tmp_path):
    with NO_CHECKPOINT as ckpt:
        assert not ckpt
        assert not ckpt.exists
        assert not ckpt.is_active
        assert ckpt.write(lead_time=_lead_time(0)) is None
        assert ckpt.flush() is None

    checkpoint = Checkpoint("forecast", path=tmp_path, mode="append", flush_interval=1)
    with checkpoint.select(time="2024-01-01") as selected:
        with checkpoint as active:
            assert active is selected
            assert active.write(lead_time=_lead_time(0)) is not None
            assert active.flush() is None

    assert len(checkpoint.catalog) == 1


def test_checkpoint_state_proxy_metadata_and_rebinding(tmp_path):
    proxy = CheckpointState(ToyState())
    assert repr(proxy) == repr(proxy.checkpoint_dataclass)
    assert proxy.checkpoint_state_policy == "minimal"
    assert proxy.checkpoint_device == torch.device("cpu")
    assert proxy.device == torch.device("cpu")
    assert proxy.checkpoint_flush_interval is None
    assert proxy.checkpoint_write_count == 0
    assert not proxy.checkpoint_is_flush_due
    assert proxy.checkpoint_lead_time is None
    assert proxy.checkpoint_labels == {}
    assert not proxy.checkpoint_state_loaded
    with pytest.raises(AttributeError):
        proxy.device = torch.device("cpu")
    assert NO_CHECKPOINT.labels == {}
    assert NO_CHECKPOINT.write_count == 0
    assert not NO_CHECKPOINT.is_active
    assert NO_CHECKPOINT.device == torch.device("cpu")

    checkpoint = Checkpoint(
        "forecast", path=tmp_path, flush_interval=None, device="cpu"
    )
    assert checkpoint.device == torch.device("cpu")
    with checkpoint.select(time="2024-01-01") as ckpt:
        rebound = bind_checkpoint_state(proxy)
        assert rebound is proxy
        assert ckpt.device == torch.device("cpu")
        assert proxy.device == torch.device("cpu")
        assert bind_checkpoint_state(proxy) is proxy
        assert ckpt.is_active
        assert proxy.checkpoint_labels == {"time": "2024-01-01"}
        assert ckpt.artifacts == {}
        with pytest.raises(TypeError):
            ckpt.bind(object())
        assert ckpt.write(lead_time=torch.tensor([6])) is None
        entry = ckpt.flush()

    assert entry is not None
    assert entry.lead_time == 6


class DroppingLeadTimeDiagnostic(torch.nn.Module):
    def __init__(self, variables: list[str], domain_coords: OrderedDict):
        super().__init__()
        self.variables = np.asarray(variables)
        self.domain_coords = domain_coords

    def input_coords(self):
        coords = OrderedDict(
            {
                "batch": np.empty(0),
                "time": np.empty(0),
                "lead_time": np.empty(0),
                "variable": self.variables,
            }
        )
        coords.update(self.domain_coords)
        return coords

    def output_coords(self, input_coords):
        output_coords = input_coords.copy()
        output_coords.pop("lead_time", None)
        return output_coords

    def __call__(self, x, coords):
        output_coords = coords.copy()
        lead_time_index = list(output_coords).index("lead_time")
        output_coords.pop("lead_time")
        return x.squeeze(lead_time_index), output_coords


class RecordingIO:
    def __init__(self):
        self.coords = None
        self.array_names = None
        self.writes = []

    def add_array(self, coords, array_name, **kwargs):
        self.coords = coords
        self.array_names = array_name

    def write(self, x, coords, array_name):
        self.writes.append((x, coords, array_name))


def _lead_time(hours: int):
    return np.timedelta64(hours, "h")


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
        ckpt.write(lead_time=_lead_time(6), artifacts={"sample": 3})

    checkpoint = Checkpoint("forecast", path=tmp_path)
    with checkpoint.select(time=time, ensemble=0) as ckpt:
        restored = bind_checkpoint_state(ToyState())

        assert ckpt.exists
        assert ckpt.lead_time == np.timedelta64(6, "h")
        assert restored.calls == 7
        assert torch.equal(restored.rng, torch.arange(4, dtype=torch.uint8))
        assert np.array_equal(
            restored.weights, np.asarray([3.0, 4.0], dtype=np.float32)
        )
        assert restored.delta == np.timedelta64(6, "h")
        assert restored.tensor_device == torch.device("cpu")
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
        ckpt.flush(lead_time=_lead_time(0))

    with checkpoint.select(time="2024-01-02") as ckpt:
        state = bind_checkpoint_state(ToyState())
        state.calls = 2
        ckpt.flush(lead_time=_lead_time(6))

    assert len(checkpoint.catalog) == 2


def test_write_interval_overwrite_and_manual_flush_prune_old_commits(tmp_path):
    checkpoint = Checkpoint(
        "forecast", path=tmp_path, mode="overwrite", flush_interval=2
    )

    with checkpoint.select(time="2024-01-01") as ckpt:
        state = bind_checkpoint_state(ToyState())
        state.calls = 1
        assert ckpt.write(lead_time=_lead_time(6)) is None

        state.calls = 2
        first = ckpt.write(lead_time=_lead_time(12))
        assert first is not None

        state.calls = 3
        assert ckpt.write(lead_time=_lead_time(18)) is None
        final = ckpt.flush()

    assert final.write_count == 3
    assert len(checkpoint.catalog) == 1
    commits = list((checkpoint.rank_path / "commits").iterdir())
    assert [commit.name for commit in commits] == [final.commit_id]

    with checkpoint.select(time="2024-01-01") as ckpt:
        state = bind_checkpoint_state(ToyState())
        assert ckpt.lead_time == np.timedelta64(18, "h")
        assert state.calls == 3


def test_append_history_size_and_positional_selection(tmp_path):
    checkpoint = Checkpoint(
        "forecast", path=tmp_path, mode="append", flush_interval=1, history_size=2
    )

    for hours in (6, 12, 18):
        with checkpoint.select(time="2024-01-01", ensemble=0) as ckpt:
            bind_checkpoint_state(ToyState()).calls = hours
            ckpt.write(lead_time=_lead_time(hours))

    assert len(checkpoint.catalog) == 2
    assert checkpoint.select(-1).lead_time == np.timedelta64(18, "h")
    assert checkpoint.select(-2).lead_time == np.timedelta64(12, "h")
    assert checkpoint.select(time=-1, ensemble=0).lead_time == np.timedelta64(18, "h")


def test_bind_before_new_session_is_adopted_on_enter(tmp_path):
    checkpoint = Checkpoint(
        "forecast", path=tmp_path, flush_interval=2, state_policy="state"
    )

    dataclass_state = ToyState()
    state = bind_checkpoint_state(dataclass_state)
    assert isinstance(state, CheckpointState)
    assert state.checkpoint_dataclass is dataclass_state
    assert bind_checkpoint_state(dataclass_state) is state
    assert state.checkpoint_enabled
    assert state.checkpoint_state_policy == "state"
    assert state.checkpoint_flush_interval == 2
    with pytest.raises(AttributeError):
        state.checkpoint_state_policy = "full"
    state.calls = 5

    with checkpoint.select(time="2024-01-01") as ckpt:
        assert list(ckpt.bound_states.values()) == [state]
        assert not state.checkpoint_selected
        assert not state.checkpoint_state_loaded
        assert state.checkpoint_write_count == 0
        assert not state.checkpoint_is_flush_due
        ckpt.write(lead_time=_lead_time(3))
        assert state.checkpoint_write_count == 1
        assert state.checkpoint_is_flush_due
        ckpt.flush(lead_time=_lead_time(6))

    with Checkpoint("forecast", path=tmp_path).select(time="2024-01-01"):
        restored = bind_checkpoint_state(ToyState())
        assert restored.checkpoint_selected
        assert restored.checkpoint_state_loaded
        assert restored.checkpoint_lead_time == np.timedelta64(6, "h")

    assert restored.calls == 5
    assert not restored.checkpoint_selected
    assert not restored.checkpoint_state_loaded


def test_bind_before_existing_session_warns_and_hydrates_late(tmp_path):
    checkpoint = Checkpoint("forecast", path=tmp_path)
    with checkpoint.select(time="2024-01-01") as ckpt:
        state = bind_checkpoint_state(ToyState())
        state.calls = 9
        ckpt.flush(lead_time=_lead_time(6))

    checkpoint = Checkpoint("forecast", path=tmp_path)
    state = bind_checkpoint_state(ToyState())
    assert state.calls == 0

    with pytest.warns(UserWarning, match="bound before an existing checkpoint session"):
        with checkpoint.select(time="2024-01-01") as ckpt:
            assert ckpt.exists
            assert state.checkpoint_state_loaded
            assert state.calls == 9


def test_defensive_paths_and_catalog_rebuild(tmp_path):
    plain = ToyState()
    plain_state = bind_checkpoint_state(plain)
    assert isinstance(plain_state, CheckpointState)
    assert plain_state.checkpoint_dataclass is plain
    plain_state.calls = 1
    assert plain.calls == 1
    with pytest.raises(TypeError):
        bind_checkpoint_state(object())

    with pytest.raises(ValueError):
        Checkpoint("bad", path=tmp_path, mode="bad")
    with pytest.raises(ValueError):
        Checkpoint("bad", path=tmp_path, flush_interval=0)
    with pytest.raises(ValueError):
        Checkpoint("bad", path=tmp_path, history_size=0)
    with pytest.raises(ValueError):
        Checkpoint("bad", path=tmp_path, state_policy="bad")
    with pytest.raises(ValueError):
        Checkpoint("legacy-replay", path=tmp_path, state_policy="replay")
    with pytest.raises(ValueError):
        Checkpoint("legacy-direct", path=tmp_path, state_policy="direct")

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
        ckpt.flush(lead_time=torch.tensor([6]))
        assert ckpt.flush() is None
        ckpt.write(lead_time=torch.tensor([7]))

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
        ckpt.flush(lead_time=np.asarray([1, 2]))
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
            lead_time=_lead_time(6),
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
            ckpt.flush(lead_time=_lead_time(0))

    with checkpoint.select(time="bad-array") as ckpt:
        with pytest.raises(CheckpointSerializationError):
            ckpt.flush(artifacts={"bad": np.asarray([object()], dtype=object)})


def test_schema_mismatch_errors_before_hydration(tmp_path):
    checkpoint = Checkpoint("forecast", path=tmp_path)

    with checkpoint.select(time="2024-01-01") as ckpt:
        state = bind_checkpoint_state(ToyState())
        state.calls = 4
        ckpt.flush(lead_time=_lead_time(6))

    metadata_path = next(
        (checkpoint.rank_path / "commits").glob("*/states/*/metadata.json")
    )
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
        ckpt.flush(lead_time=_lead_time(0))

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
                "lead_time": np.asarray([np.timedelta64(6 * i, "h") for i in range(4)]),
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

    with checkpoint.select(-1):
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
            checkpoint=checkpoint,
        )

    selected = checkpoint.select(-1)
    assert selected.lead_time == np.timedelta64(18, "h")
    assert selected.write_count == 4
    assert io["u10m"].shape[1] == 4


def test_deterministic_workflow_uses_model_checkpoint_state_when_io_is_filtered(
    tmp_path,
):
    coords = OrderedDict([("lat", np.arange(2)), ("lon", np.arange(3))])
    variables = ["u10m", "v10m"]
    output_coords = OrderedDict({"variable": np.asarray(["u10m"])})
    io = ZarrBackend()
    io.add_array(
        OrderedDict(
            {
                "time": np.asarray(["2024-01-01T00"], dtype="datetime64[ns]"),
                "lead_time": np.asarray([np.timedelta64(6 * i, "h") for i in range(4)]),
                **coords,
            }
        ),
        ["u10m"],
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
        output_coords=output_coords,
        device=torch.device("cpu"),
        verbose=False,
        checkpoint=checkpoint,
    )

    with checkpoint.select(-1):
        data = Random(domain_coords=coords)
        model = Persistence(variables, coords)
        run.deterministic(
            ["2024-01-01"],
            3,
            model,
            data,
            io,
            output_coords=output_coords,
            device=torch.device("cpu"),
            verbose=False,
            checkpoint=checkpoint,
        )

    selected = checkpoint.select(-1)
    assert selected.lead_time == np.timedelta64(18, "h")
    assert selected.write_count == 4
    assert io["u10m"].shape[1] == 4
    assert "v10m" not in io


def test_diagnostic_workflow_resumes_from_checkpoint(tmp_path):
    coords = OrderedDict([("lat", np.arange(2)), ("lon", np.arange(3))])
    variables = ["u10m", "v10m"]
    io = ZarrBackend()
    io.add_array(
        OrderedDict(
            {
                "time": np.asarray(["2024-01-01T00"], dtype="datetime64[ns]"),
                "lead_time": np.asarray([np.timedelta64(6 * i, "h") for i in range(4)]),
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


def test_diagnostic_checkpoint_tracks_prognostic_lead_time(tmp_path):
    coords = OrderedDict([("lat", np.arange(2)), ("lon", np.arange(3))])
    variables = ["u10m", "v10m"]
    checkpoint = Checkpoint("diagnostic", path=tmp_path, flush_interval=1)
    io = RecordingIO()

    run.diagnostic(
        ["2024-01-01"],
        1,
        Persistence(variables, coords),
        DroppingLeadTimeDiagnostic(variables, coords),
        Random(domain_coords=coords),
        io,
        device=torch.device("cpu"),
        verbose=False,
        checkpoint=checkpoint,
    )

    assert len(io.writes) == 2
    assert checkpoint.select(-1).lead_time == np.timedelta64(6, "h")


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
                "lead_time": np.asarray([np.timedelta64(6 * i, "h") for i in range(4)]),
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
