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
from earth2studio.models.px import FCN3
from earth2studio.utils import handshake_dim
from earth2studio.utils.checkpoint import Checkpoint


class PhooFCN3Preprocessor(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.register_buffer(
            "state",
            torch.randn(
                10,
            ),
        )

    def set_internal_state(self, state: torch.Tensor):
        self.state = state.to(self.state.device)

    def get_internal_state(self, tensor=True):
        return self.state

    def update_internal_state(self, replace_state=True):
        self.state = torch.randn((10,), device=self.state.device)


class PhooFCN3Model(torch.nn.Module):
    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor


class PhooFCN3ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, normalized_data: bool = False, replace_state: bool = False):
        return x

    def set_rng(self, reset: bool = True, seed: int = 333):
        return


class PhooRestartFCN3ModelWrapper(PhooFCN3ModelWrapper):
    def forward(self, x, t, normalized_data: bool = False, replace_state: bool = False):
        return x + self.model.preprocessor.state[0].to(x.device)


def _phoo_fcn3(model: torch.nn.Module, variables: np.ndarray) -> FCN3:
    fcn3 = FCN3.__new__(FCN3)
    FCN3.__init__.__wrapped__(fcn3, model, variables=variables)
    return fcn3


@pytest.fixture(scope="function")
def dummy_model():
    preprocessor = PhooFCN3Preprocessor()
    model = PhooFCN3Model(preprocessor)
    return model


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
def test_fcn3_call(time, device, dummy_model):

    # Spoof model
    model = PhooFCN3ModelWrapper(dummy_model)
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
def test_fcn3_iter(ensemble, device, dummy_model):

    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Spoof model
    model = PhooFCN3ModelWrapper(dummy_model)
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


def test_fcn3_checkpoint_state_round_trip_with_phoo_model(tmp_path):
    variables = np.array(["u10m"])
    time = np.array([np.datetime64("1993-04-05T00:00")])
    checkpoint = Checkpoint("fcn3", path=tmp_path / "fcn3", flush_interval=2, level=2)

    torch.manual_seed(123)
    model = PhooRestartFCN3ModelWrapper(PhooFCN3Model(PhooFCN3Preprocessor()))
    p = _phoo_fcn3(model, variables=variables).to("cpu")
    input_coords = p.input_coords()
    coords = OrderedDict(
        {
            "time": time,
            "lead_time": input_coords["lead_time"],
            "variable": input_coords["variable"],
            "lat": input_coords["lat"],
            "lon": input_coords["lon"],
        }
    )
    x = torch.zeros(
        len(coords["time"]),
        len(coords["lead_time"]),
        len(coords["variable"]),
        len(coords["lat"]),
        len(coords["lon"]),
    )

    with checkpoint.select(time=time) as ckpt:
        iterator = p.create_iterator(x, coords)
        _, initial_coords = next(iterator)
        ckpt.write(lead_time=initial_coords["lead_time"])
        step1, step1_coords = next(iterator)
        ckpt.write(lead_time=step1_coords["lead_time"])
        step2, step2_coords = next(iterator)
        step3, step3_coords = next(iterator)

    torch.manual_seed(999)
    with checkpoint.select(-1):
        resumed_model = PhooRestartFCN3ModelWrapper(
            PhooFCN3Model(PhooFCN3Preprocessor())
        )
        resumed = _phoo_fcn3(resumed_model, variables=variables).to("cpu")
        assert resumed.checkpoint.checkpoint_state_loaded

        resumed_iterator = resumed.create_iterator(torch.full_like(x, -100.0), coords)
        resumed_step2, resumed_step2_coords = next(resumed_iterator)
        resumed_step3, resumed_step3_coords = next(resumed_iterator)

    assert torch.allclose(resumed_step2, step2)
    assert torch.allclose(resumed_step3, step3)
    assert resumed_step2_coords["lead_time"][0] == step2_coords["lead_time"][0]
    assert resumed_step3_coords["lead_time"][0] == step3_coords["lead_time"][0]


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(720)}),
        OrderedDict({"lat": np.random.randn(720), "phoo": np.random.randn(1440)}),
        OrderedDict({"lat": np.random.randn(720), "lon": np.random.randn(1)}),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fcn3_exceptions(dc, device, dummy_model):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    # Spoof model
    model = PhooFCN3ModelWrapper(dummy_model)
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
def model() -> FCN3:
    package = FCN3.load_default_package()
    p = FCN3.load_model(package)
    return p


@pytest.mark.package
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_fcn3_load_package(device, model):
    torch.cuda.empty_cache()
    # Test the cached model package FCN3
    model.to(device)


# Will not test while we do not have 80GB GPU cards
# in CI
# @pytest.mark.package
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
