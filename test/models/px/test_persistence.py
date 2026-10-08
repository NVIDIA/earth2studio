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

from collections import OrderedDict
from contextlib import nullcontext
from copy import deepcopy

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.data import Random, fetch_data
from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.px import Persistence
from earth2studio.utils import coord_array_like
from earth2studio.utils.checkpoint import Checkpoint, CheckpointState
from earth2studio.utils.cupy import Earth2StudioAccessor, from_torch


def _history_field(model: Persistence, device: str) -> xr.DataArray:
    signature = coord_array_like(model.input_coords(), {"batch": [0]})
    x = from_torch(
        torch.arange(
            np.prod(signature.shape), device=device, dtype=torch.float32
        ).reshape(signature.shape),
        signature,
        name="weather",
        attrs={"experiment": {"name": "history"}},
    )
    x = x.assign_coords(sample=("lead_time", np.arange(model._history)), height=2.0)
    x.coords["sample"].attrs["description"] = "history sample"
    x.lead_time.attrs["description"] = "forecast lead"
    x.encoding["source"] = "history.nc"
    return x


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("history", [1, 3])
@pytest.mark.parametrize("hook", ["front", "rear", "both"])
def test_persistence_inplace_hooks_own_storage(device, history, hook):
    model = Persistence("t2m", OrderedDict(point=np.arange(2)), history=history)
    x = _history_field(model, device).drop_vars("sample")
    original = x.copy(deep=True)
    calls = []

    def mutate(field):
        calls.append(field.sizes["lead_time"])
        field.data += 1
        field.attrs["experiment"]["name"] = "hook"
        return field

    if hook in ("front", "both"):
        model.front_hook = mutate
    if hook in ("rear", "both"):
        model.rear_hook = mutate
    iterator = model.create_iterator(x)
    retained = []
    for step in range(4):
        output = next(iterator)
        xr.testing.assert_identical(x, original)
        if retained and hook in ("front", "both"):
            retained[-1][1].data += 1
            retained[-1][1].attrs["experiment"]["name"] = "hook"
        for previous, snapshot in retained:
            xr.testing.assert_identical(previous, snapshot)
        torch.testing.assert_close(
            output.e2s.to_torch()[0],
            original.isel(lead_time=slice(-1, None)).e2s.to_torch()[0]
            + step * (hook in ("front", "both"))
            + (step + 1) * (hook in ("rear", "both")),
        )
        assert output.e2s.to_torch()[0].device == torch.device(device)
        retained.append((output, output.copy(deep=True)))
    assert len(calls) == 3 * (hook in ("front", "both")) + 4 * (
        hook in ("rear", "both")
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_persistence_initial_yield_is_owned(device):
    model = Persistence("t2m", OrderedDict(point=np.arange(2)), history=3)
    x = _history_field(model, device)
    original = x.copy(deep=True)
    iterator = model.create_iterator(x)
    initial = next(iterator)
    initial.data += 100
    initial.attrs["experiment"]["name"] = "consumer"
    xr.testing.assert_identical(x, original)
    output = next(iterator)
    torch.testing.assert_close(
        output.e2s.to_torch()[0],
        original.isel(lead_time=slice(-1, None)).e2s.to_torch()[0] + 100,
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_persistence_identity_without_tensor_conversion(monkeypatch, device):
    model = Persistence("t2m", OrderedDict(point=np.arange(2)), history=3)
    # Exercise the identity path independently of an inherited checkpoint catalog.
    model.checkpoint = CheckpointState(model.checkpoint.checkpoint_dataclass)
    assert not model.checkpoint.checkpoint_enabled
    x = _history_field(model, device)

    def forbidden(*args, **kwargs):
        pytest.fail("Persistence identity must not convert field data to Torch")

    monkeypatch.setattr(Earth2StudioAccessor, "to_torch", forbidden)
    output = model(x)
    expected = x.isel(lead_time=slice(-1, None)).assign_coords(
        lead_time=x.lead_time[-1:] + model._dt
    )
    expected.lead_time.attrs = x.lead_time.attrs
    xr.testing.assert_identical(output, expected)
    assert output.encoding == x.encoding
    assert not {
        "earth2studio_kind",
        "earth2studio_schema_version",
        "earth2studio_dynamic_dims",
    }.intersection(output.attrs)
    output.data += 1
    assert not bool((output.data == expected.data).any())


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_persistence_history_metadata(device):
    model = Persistence("t2m", OrderedDict(point=np.arange(2)), history=3)
    x = _history_field(model, device)
    iterator = model.create_iterator(x)
    for step in range(4):
        output = next(iterator)
        xr.testing.assert_identical(
            output["sample"],
            x["sample"].isel(lead_time=[-1]).assign_coords(lead_time=output.lead_time),
        )
        assert output.lead_time.attrs == x.lead_time.attrs
        assert output.attrs == x.attrs
        assert output.encoding == x.encoding
        assert output.name == x.name
        assert output.lead_time.values[0] == np.timedelta64(6 * (step + 1), "h")


def test_persistence_checkpoint_copies_coordinates_once(monkeypatch, tmp_path):
    with Checkpoint("persistence", path=tmp_path, mode="append", level=2):
        model = Persistence("t2m", OrderedDict(point=np.arange(2)), history=3)
        x = _history_field(model, "cpu")

        def copy_metadata(metadata):
            # A numeric auxiliary coordinate exposes its backing array directly;
            # string indexes may allocate when xarray materializes .values.
            assert np.shares_memory(metadata["coords"]["sample"][1], x["sample"].values)
            return deepcopy(metadata)

        monkeypatch.setattr(
            "earth2studio.models.px.persistence.deepcopy", copy_metadata
        )
        model._save_checkpoint_state(x)
        for name, (_, values, _) in model.checkpoint.metadata["coords"].items():
            assert not np.shares_memory(values, x.coords[name].values)
        x.attrs["experiment"]["name"] = "mutated"
        assert model.checkpoint.metadata["attrs"]["experiment"]["name"] == "history"


@pytest.mark.parametrize("level", [None, 0, 1, 2])
def test_persistence_execution_ignores_ambient_checkpoint(monkeypatch, tmp_path, level):
    context = (
        nullcontext()
        if level is None
        else Checkpoint("persistence", path=tmp_path, mode="append", level=level)
    )
    with context:
        model = Persistence("t2m", OrderedDict(point=np.arange(2)), history=3)
        if level is None:
            # Binding otherwise inherits the most recently created checkpoint.
            model.checkpoint = CheckpointState(model.checkpoint.checkpoint_dataclass)
        x = _history_field(model, "cpu")
        model.checkpoint.x = torch.ones(1)
        model.checkpoint.metadata = {"stale": True}

        def checked_concat(*args, **kwargs):
            pytest.fail("Persistence must not assemble unused history")

        monkeypatch.setattr(xr, "concat", checked_concat)
        output = model(x)
        assert output.lead_time.values[0] == model._dt
        next_output, state = model.step(output, None)
        assert state is None
        assert next_output.lead_time.values[0] == 2 * model._dt
        torch.testing.assert_close(model.checkpoint.x, torch.ones(1))
        assert model.checkpoint.metadata == {"stale": True}


@pytest.mark.parametrize(
    "variable",
    ["t2m", ["t2m", "tcwv"]],
)
@pytest.mark.parametrize("history", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_persistence_lat_lon(
    variable: str | list[str], history: int, device: str
) -> None:
    time = np.array(
        [np.datetime64("1999-10-11T12:00"), np.datetime64("2001-06-04T00:00")]
    )
    # Construct Domain Coordinates
    dc = OrderedDict(
        {
            "lat": np.linspace(-90, 90, 360),
            "lon": np.linspace(0, 360, 720, endpoint=False),
        }
    )

    # Initialize Model
    p = Persistence(variable, dc, history=history)

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x = fetch_data(r, time, variable, lead_time, device=device)
    # Random supplies labels only; declare the configured geographic CRS.
    x.attrs["earth2studio_crs"] = p.input_coords().attrs["earth2studio_crs"]

    # Get generator
    out = p(x)
    assert torch.allclose(x.e2s.to_torch()[0][:, -1:], out.e2s.to_torch()[0])
    assert out.lead_time == np.timedelta64(6, "h")
    assert lead_time.shape[0] == history


@pytest.mark.parametrize(
    "variable",
    ["t2m", ["t2m", "tcwv"]],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_persistence_unstructured(variable, device):
    time = np.array(
        [
            np.datetime64("1999-10-11T12:00"),
            np.datetime64("2001-06-04T00:00"),
        ]
    )
    # Construct Domain Coordinates
    dc = OrderedDict(
        {"face": np.arange(6), "lat": np.random.randn(60), "lon": np.random.randn(60)}
    )
    # Initialize Model
    p = Persistence(variable, dc)

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x = fetch_data(r, time, variable, lead_time, device=device)

    # Get generator
    out = p(x)

    assert torch.allclose(x.e2s.to_torch()[0][:, -1:], out.e2s.to_torch()[0])
    assert (out.time == x.time).all()
    assert out.lead_time == np.timedelta64(6, "h")


@pytest.mark.parametrize(
    "ensemble",
    [1, 2],
)
@pytest.mark.parametrize(
    "variable",
    [["t2m"], ["t2m", "tcwv"]],
)
@pytest.mark.parametrize("history", [1, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_persistence_iter(ensemble, variable, history, device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Construct Domain Coordinates
    dc = OrderedDict({"lat": np.random.randn(60), "lon": np.random.randn(60)})
    # Initialize Model
    p = Persistence(variable, dc, history=history)

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x = fetch_data(r, time, variable, lead_time, device=device)
    x.attrs["earth2studio_crs"] = p.input_coords().attrs["earth2studio_crs"]

    # Add ensemble to front
    x = x.expand_dims(ensemble=np.arange(ensemble)).copy(deep=True)

    p_iter = p.create_iterator(x)

    # Get generator
    for i, out in enumerate(p_iter):
        assert len(out.shape) == 6
        assert torch.allclose(x.e2s.to_torch()[0][:, :, -1:], out.e2s.to_torch()[0])
        assert out.shape[0] == ensemble
        assert out.shape[2] == 1
        assert out.shape[3] == variable.shape[0]
        assert out.time == x.time
        assert out.lead_time[0] == np.timedelta64(6 * (i + 1), "h")

        if i > 5:
            break


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_persistence_checkpoint_state_round_trip(device):
    import pickle

    variable = ["t2m", "tcwv"]
    time = np.array([np.datetime64("1993-04-05T00:00")])
    domain_coords = OrderedDict({"lat": np.arange(2), "lon": np.arange(3)})
    lead_time = np.asarray([np.timedelta64(-6, "h"), np.timedelta64(0, "h")])
    data = Random(domain_coords)
    x = fetch_data(data, time, variable, lead_time, device=device)
    x = x.assign_coords(sample=("lead_time", [0, 1]), height=2.0)
    x.coords["sample"].attrs["description"] = "history sample"
    x.lead_time.attrs["description"] = "forecast lead"
    x.attrs["earth2studio_crs"] = (
        Persistence(variable, domain_coords, history=2)
        .input_coords()
        .attrs["earth2studio_crs"]
    )
    x.name = "weather"
    x.attrs["experiment"] = "checkpoint"
    x.encoding["test"] = "preserved"
    model = Persistence(variable, domain_coords, history=2)
    forecast, state = model.initialize(x)
    assert state is None
    saved = pickle.dumps((forecast, state))
    expected, _ = model.step(forecast, state)
    model = Persistence(variable, domain_coords, history=2)
    restored, state = pickle.loads(saved)  # noqa: S301 - locally generated checkpoint
    out, _ = model.step(restored, state)

    assert out.lead_time[0] == np.timedelta64(12, "h")
    assert torch.allclose(out.e2s.to_torch()[0], x.e2s.to_torch()[0][:, -1:])
    assert out.name == x.name
    assert out.attrs["experiment"] == "checkpoint"
    assert out.encoding == x.encoding
    xr.testing.assert_identical(out, expected)
    assert out.e2s.to_torch()[0].device == torch.device(device)


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(60)}),
        OrderedDict({"lat": np.random.randn(60), "phoo": np.random.randn(60)}),
        OrderedDict({"lat": np.random.randn(61), "lon": np.random.randn(61)}),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_persistence_coords(dc, device):
    variable = ["t2m"]
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Construct Domain Coordinates
    true_dc = OrderedDict({"lat": np.random.randn(60), "lon": np.random.randn(60)})
    # Initialize Model
    p = Persistence(variable, true_dc)
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x = fetch_data(r, time, variable, lead_time, device=device)

    with pytest.raises((KeyError, ValueError)):
        p(x)


def test_persistence_conformance():
    variable = ["t2m", "tcwv"]
    dc = OrderedDict(
        {
            "lat": np.linspace(-90, 90, 360),
            "lon": np.linspace(0, 360, 720, endpoint=False),
        }
    )
    p = Persistence(variable, dc, history=2)
    # P14 is skipped rather than passed: the model does not declare itself
    # stochastic, so the RNG-isolation rule has nothing to check.
    assert check_prognostic_contract(p) == [
        "P14: model does not declare itself stochastic"
    ]
