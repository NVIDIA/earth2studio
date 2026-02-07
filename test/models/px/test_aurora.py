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
from collections.abc import Iterable

import numpy as np
import pytest
import torch
from aurora import Batch, Metadata

from earth2studio.data import Random, fetch_data
from earth2studio.models.px import Aurora
from earth2studio.utils import handshake_dim


class PhooAuroraModel(torch.nn.Module):
    def forward(self, x):
        return Batch(
            surf_vars={k: v[:, -1:, ...] for k, v in x.surf_vars.items()},
            static_vars=x.static_vars,
            atmos_vars={k: v[:, -1:, ...] for k, v in x.atmos_vars.items()},
            metadata=Metadata(
                lat=x.metadata.lat,
                lon=x.metadata.lon,
                time=x.metadata.time,
                atmos_levels=x.metadata.atmos_levels,
                rollout_step=x.metadata.rollout_step + 1,
            ),
        )


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("1993-04-05T00:00")]),
        np.array(
            [np.datetime64("2001-06-04T00:00")]
        ),  # Only len 1 time array is supported by Auroral model
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aurora_call(time, device):

    # Spoof model
    model = PhooAuroraModel()
    z = torch.ones(720, 1440)
    slt = torch.ones(720, 1440)
    lsm = torch.ones(720, 1440)

    p = Aurora(model, z, slt, lsm).to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size([len(time), 1, 69, 720, 1440])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)


@pytest.mark.parametrize(
    "ensemble",
    [1, 2],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aurora_iter(ensemble, device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Spoof model
    model = PhooAuroraModel()
    z = torch.ones(720, 1440)
    slt = torch.ones(720, 1440)
    lsm = torch.ones(720, 1440)

    p = Aurora(model, z, slt, lsm).to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
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

    if not isinstance(time, Iterable):
        time = [time]

    # Get generator
    next(p_iter)  # Skip first which should return the input
    for i, (out, out_coords) in enumerate(p_iter):
        assert len(out.shape) == 6
        assert out.shape == torch.Size([ensemble, len(time), 1, 69, 720, 1440])
        assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
        assert (out_coords["ensemble"] == np.arange(ensemble)).all()
        assert (out_coords["time"] == time).all()
        assert out_coords["lead_time"] == np.timedelta64(6 * (i + 1), "h")

        if i > 5:
            break


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(720)}),
        OrderedDict({"lat": np.random.randn(720), "phoo": np.random.randn(1440)}),
        OrderedDict({"lat": np.random.randn(720), "lon": np.random.randn(1)}),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_aurora_exceptions(dc, device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    model = PhooAuroraModel()
    z = torch.ones(720, 1440)
    slt = torch.ones(720, 1440)
    lsm = torch.ones(720, 1440)

    p = Aurora(model, z, slt, lsm).to(device)

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    with pytest.raises((KeyError, ValueError)):
        p(x, coords)


@pytest.fixture(scope="function")
def model() -> Aurora:
    package = Aurora.load_default_package()
    p = Aurora.load_model(package)
    return p


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_aurora_package(model, device):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Test the cached model package FCN
    p = model.to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size([len(time), 1, 69, 720, 1440])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)
