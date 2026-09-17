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

from earth2studio.data import Random, fetch_data
from earth2studio.models.px import FCN
from earth2studio.utils import handshake_dim
from earth2studio.utils.checkpoint import Checkpoint
from earth2studio.utils.cupy import from_torch


@pytest.fixture(autouse=True)
def cupy_for_cuda(request):
    if hasattr(request.node, "callspec") and request.node.callspec.params.get(
        "device", "cpu"
    ).startswith("cuda"):
        pytest.importorskip("cupy")


class PhooFCNModel(torch.nn.Module):
    def forward(self, x):
        return x


class IncrementFCNModel(torch.nn.Module):
    def forward(self, x):
        return x + 1


def test_fcn_coordinate_signatures():
    model = FCN(PhooFCNModel(), torch.zeros(26, 1, 1), torch.ones(26, 1, 1))
    input_coords = model.input_coords()
    output_coords = model.output_coords(input_coords)

    assert input_coords.data.nbytes == output_coords.data.nbytes == 0
    assert input_coords.attrs["earth2studio_grid_id"] == "fcn1"
    assert output_coords.coords["lead_time"] == np.timedelta64(6, "h")


def _random_source() -> Random:
    return Random(
        OrderedDict(
            {
                "lat": np.linspace(90, -90, 720, endpoint=False),
                "lon": np.linspace(0, 360, 1440, endpoint=False),
            }
        )
    )


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
def test_fcn_call(time, device):

    # Spoof model
    model = PhooFCNModel()
    center = torch.zeros(26, 1, 1)
    scale = torch.ones(26, 1, 1)

    p = FCN(model, center, scale).to(device)

    signature = p.input_coords()
    r = _random_source()

    # Get data and convert to the established tensor coordinate contract.
    lead_time = signature["lead_time"].values
    variable = signature["variable"].values
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    array = from_torch(x, coords, attrs=signature.attrs)
    result = p(array)
    out, out_coords = result.e2s.to_torch()

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size([len(time), 1, 26, 720, 1440])
    assert (
        out_coords["variable"] == p.output_coords(p.input_coords())["variable"]
    ).all()
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
def test_fcn_iter(ensemble, device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Spoof model
    model = PhooFCNModel()
    center = torch.zeros(26, 1, 1)
    scale = torch.ones(26, 1, 1)

    p = FCN(model, center, scale).to(device)

    signature = p.input_coords()
    r = _random_source()

    # Get data and convert to the established tensor coordinate contract.
    lead_time = signature["lead_time"].values
    variable = signature["variable"].values
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Add ensemble to front
    x = x.unsqueeze(0).repeat(ensemble, 1, 1, 1, 1, 1)
    coords.update({"ensemble": np.arange(ensemble)})
    coords.move_to_end("ensemble", last=False)

    p_iter = p.create_iterator(from_torch(x, coords, attrs=signature.attrs))

    if not isinstance(time, Iterable):
        time = [time]

    # Get generator
    next(p_iter)  # Skip first which should return the input
    for i, result in enumerate(p_iter):
        out, out_coords = result.e2s.to_torch()
        assert len(out.shape) == 6
        assert out.shape == torch.Size([ensemble, len(time), 1, 26, 720, 1440])
        assert (
            out_coords["variable"] == p.output_coords(p.input_coords())["variable"]
        ).all()
        assert (out_coords["ensemble"] == np.arange(ensemble)).all()
        assert out_coords["lead_time"][0] == np.timedelta64(6 * (i + 1), "h")

        if i > 5:
            break


def test_fcn_shifted_output_signature():
    model = PhooFCNModel()
    center = torch.zeros(26, 1, 1)
    scale = torch.ones(26, 1, 1)
    p = FCN(model, center, scale)
    signature = p.input_coords().assign_coords(
        lead_time=np.array([12], dtype="timedelta64[h]")
    )
    output = p.output_coords(signature)
    assert output.data.nbytes == 0
    assert output.lead_time.values[0] == np.timedelta64(18, "h")


def test_fcn_checkpoint_level_2_state_round_trip(tmp_path):
    center = torch.zeros(26, 1, 1)
    scale = torch.ones(26, 1, 1)
    source_model = FCN(IncrementFCNModel(), center, scale)
    base_coords = source_model.input_coords()
    coords = OrderedDict(
        {
            "time": np.array([np.datetime64("1993-04-05T00:00")]),
            "lead_time": base_coords["lead_time"].values,
            "variable": base_coords["variable"].values,
            "lat": base_coords["lat"].values,
            "lon": base_coords["lon"].values,
        }
    )
    x = torch.zeros(1, 1, 26, 720, 1440)

    checkpoint = Checkpoint("fcn", path=tmp_path, flush_interval=1, level=2)
    with checkpoint as ckpt:
        model = FCN(IncrementFCNModel(), center, scale)
        iterator = model.create_iterator(from_torch(x, coords, attrs=base_coords.attrs))
        next(iterator)
        saved_x, saved_coords = next(iterator).e2s.to_torch()
        assert saved_coords["lead_time"][0] == np.timedelta64(6, "h")
        assert saved_x[0, 0, 0, 0, 0] == 1
        ckpt.write(lead_time=saved_coords["lead_time"][-1])

    checkpoint = Checkpoint("fcn", path=tmp_path, level=2)
    with checkpoint.select(-1):
        model = FCN(IncrementFCNModel(), center, scale)
        assert model.checkpoint.checkpoint_state_loaded
        restart_x = torch.full_like(x, -5)
        resumed_x, resumed_coords = next(
            model.create_iterator(
                from_torch(restart_x, coords, attrs=base_coords.attrs)
            )
        ).e2s.to_torch()

    assert resumed_coords["lead_time"][0] == np.timedelta64(12, "h")
    assert resumed_x[0, 0, 0, 0, 0] == 2
    assert resumed_x.amin() == 2
    assert resumed_x.amax() == 2


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(720)}),
        OrderedDict({"lat": np.random.randn(720), "phoo": np.random.randn(1440)}),
        OrderedDict({"lat": np.random.randn(720), "lon": np.random.randn(1)}),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fcn_exceptions(dc, device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    model = PhooFCNModel()
    center = torch.zeros(26, 1, 1)
    scale = torch.ones(26, 1, 1)

    p = FCN(model, center, scale).to(device)

    r = Random(dc)

    signature = p.input_coords()
    lead_time = signature["lead_time"].values
    variable = signature["variable"].values
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    with pytest.raises((KeyError, ValueError)):
        p(from_torch(x, coords, attrs=signature.attrs))


@pytest.fixture(scope="function")
def model() -> FCN:
    # Test only on cuda device
    package = FCN.load_default_package()
    p = FCN.load_model(package)
    return p


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_fcn_package(model, device):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Test the cached model package FCN
    p = model.to(device)

    signature = p.input_coords()
    r = _random_source()

    # Get data and convert to the established tensor coordinate contract.
    lead_time = signature["lead_time"].values
    variable = signature["variable"].values
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = p(from_torch(x, coords, attrs=signature.attrs)).e2s.to_torch()

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size([len(time), 1, 26, 720, 1440])
    assert (
        out_coords["variable"] == p.output_coords(p.input_coords())["variable"]
    ).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)
