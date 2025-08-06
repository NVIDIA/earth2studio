# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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

from earth2studio.data import Random, fetch_data
from earth2studio.models.px import FCN3
from earth2studio.utils import handshake_dim


class PhooFCN3Model(torch.nn.Module):
    def forward(self, x, t, normalized_data: bool = False, replace_state: bool = False):
        return x


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("1993-04-05T00:00")]),
        np.array(
            [
                np.datetime64("1999-10-11T12:00"),
                np.datetime64("2001-06-04T00:00"),
            ]
        ),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fcn3_call(time, device):

    # Spoof model
    model = PhooFCN3Model()
    p = FCN3(model).to(device)

    # Create "domain coords"
    dc = {k: p.input_coords()[k] for k in ["lat", "lon"]}

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size([len(time), 1, 72, 721, 1440])
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
def test_fcn3_iter(ensemble, device):

    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Spoof model
    model = PhooFCN3Model()
    p = FCN3(model).to(device)

    # Create "domain coords"
    dc = {k: p.input_coords()[k] for k in ["lat", "lon"]}

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
        assert out.shape == torch.Size([ensemble, len(time), 1, 72, 721, 1440])
        assert (
            out_coords["variable"] == p.output_coords(p.input_coords())["variable"]
        ).all()
        assert (out_coords["ensemble"] == np.arange(ensemble)).all()
        assert out_coords["lead_time"][0] == np.timedelta64(6 * (i + 1), "h")

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
def test_fcn3_exceptions(dc, device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Spoof model
    model = PhooFCN3Model()
    p = FCN3(model).to(device)

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    with pytest.raises((KeyError, ValueError)):
        p(x, coords)


@pytest.fixture(scope="function")
def model(model_cache_context) -> FCN3:
    # Test only on cuda device
    with model_cache_context():
        package = FCN3.load_default_package()
        p = FCN3.load_model(package)
        return p


@pytest.mark.ci_cache
@pytest.mark.timeout(360)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fcn3_load_package(device, model):
    torch.cuda.empty_cache()
    # Test the cached model package FCN3
    model.to(device)


# Will not test while we do not have 80GB GPU cards
# in CI
# @pytest.mark.ci_cache
# @pytest.mark.timeout(360)
# @pytest.mark.parametrize("device", ["cuda:0"])
# def test_fcn3_package(device, model):
#     torch.cuda.empty_cache()
#     time = np.array([np.datetime64("1993-04-05T00:00")])
#     # Test the cached model package FCN3
#     p = model.to(device)

#     # Create "domain coords"
#     dc = {k: p.input_coords()[k] for k in ["lat", "lon"]}

#     # Initialize Data Source
#     r = Random(dc)

#     # Get Data and convert to tensor, coords
#     lead_time = p.input_coords()["lead_time"]
#     variable = p.input_coords()["variable"]
#     x, coords = fetch_data(r, time, variable, lead_time, device=device)

#     out, out_coords = p(x, coords)

#     if not isinstance(time, Iterable):
#         time = [time]

#     assert out.shape == torch.Size([len(time), 1, 72, 721, 1440])
#     assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
#     handshake_dim(out_coords, "lon", 4)
#     handshake_dim(out_coords, "lat", 3)
#     handshake_dim(out_coords, "variable", 2)
#     handshake_dim(out_coords, "lead_time", 1)
#     handshake_dim(out_coords, "time", 0)
