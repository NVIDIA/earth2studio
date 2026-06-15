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

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.px import Persistence
from earth2studio.utils.checkpoint import Checkpoint


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
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Get generator
    out, out_coords = p(x, coords)
    assert torch.allclose(x[:, -1:], out)  # Persistence Model
    assert out_coords["lead_time"] == np.timedelta64(6, "h")
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
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Get generator
    out, out_coords = p(x, coords)

    assert torch.allclose(x[:, -1:], out)  # Persistence Model
    assert (out_coords["time"] == coords["time"]).all()
    assert out_coords["lead_time"] == np.timedelta64(6, "h")


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
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Add ensemble to front
    x = x.unsqueeze(0).repeat(ensemble, 1, 1, 1, 1, 1)
    coords.update({"ensemble": np.arange(ensemble)})
    coords.move_to_end("ensemble", last=False)

    p_iter = p.create_iterator(x, coords)

    # Get generator
    next(p_iter)  # Skip first which should return the input
    for i, (out, out_coords) in enumerate(p_iter):
        assert len(out.shape) == 6
        assert torch.allclose(x[:, :, -1:], out)  # Persistence Model
        assert out.shape[0] == ensemble
        assert out.shape[2] == 1
        assert out.shape[3] == variable.shape[0]
        assert out_coords["time"] == coords["time"]
        assert out_coords["lead_time"][0] == np.timedelta64(6 * (i + 1), "h")

        if i > 5:
            break


def test_persistence_checkpoint_state_round_trip(tmp_path):
    variable = ["t2m", "tcwv"]
    time = np.array([np.datetime64("1993-04-05T00:00")])
    domain_coords = OrderedDict({"lat": np.arange(2), "lon": np.arange(3)})
    lead_time = np.asarray([np.timedelta64(-6, "h"), np.timedelta64(0, "h")])
    data = Random(domain_coords)
    x, coords = fetch_data(data, time, np.asarray(variable), lead_time, device="cpu")
    checkpoint = Checkpoint(
        "persistence", path=tmp_path, mode="append", state_policy="full"
    )

    with checkpoint as ckpt:
        model = Persistence(variable, domain_coords, history=2)
        iterator = model.create_iterator(x, coords)
        _, initial_coords = next(iterator)
        _, saved_coords = next(iterator)
        assert initial_coords["lead_time"][0] == np.timedelta64(0, "h")
        assert saved_coords["lead_time"][0] == np.timedelta64(6, "h")
        ckpt.write(lead_time=saved_coords["lead_time"][-1])
        ckpt.flush()

    with checkpoint.select(-1):
        model = Persistence(variable, domain_coords, history=2)
        iterator = model.create_iterator(x, coords)
        out, out_coords = next(iterator)
        assert model.checkpoint.checkpoint_state_loaded

    assert out_coords["lead_time"][0] == np.timedelta64(12, "h")
    assert torch.allclose(out, x[:, -1:])


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
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    with pytest.raises((KeyError, ValueError)):
        p(x, coords)
