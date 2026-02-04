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

import gc
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.auto import Package
from earth2studio.models.px import Pangu3, Pangu6, Pangu24
from earth2studio.utils import handshake_dim


class PhooPanguModel(torch.nn.Module):
    """Dummy pangu model, adds time-step"""

    def __init__(self, delta_t: int = 24):
        super().__init__()
        self.delta_t = delta_t

    def forward(self, x, x_surface):
        return x + self.delta_t, x_surface + self.delta_t


@pytest.fixture(scope="class")
def onnx_test_package(tmp_path_factory):
    """Creates a bunch of spoof ONNX models to unit test with"""
    input_tensors = (torch.rand(5, 13, 721, 1440), torch.rand(4, 721, 1440))
    tmp_path = tmp_path_factory.mktemp("data")
    for name, delta_t in [
        ("pangu_weather_24.onnx", 24),
        ("pangu_weather_6.onnx", 6),
        ("pangu_weather_3.onnx", 3),
    ]:
        onnx_path = tmp_path / name
        torch.onnx.export(
            PhooPanguModel(delta_t),
            input_tensors,
            str(onnx_path),
            export_params=True,
            opset_version=10,
            input_names=["input", "input_surface"],
            output_names=["output", "output_surface"],
        )
    return Package(str(tmp_path))


class TestPanguMock:
    @pytest.mark.parametrize(
        "time",
        [
            np.array(
                [
                    np.datetime64("1999-10-11T12:00"),
                    np.datetime64("2001-06-04T00:00"),
                ]
            ),
        ],
    )
    @pytest.mark.parametrize(
        "PanguModel, delta_t",
        [(Pangu24, 24), (Pangu6, 6), (Pangu3, 3)],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_pangu_call(self, time, PanguModel, delta_t, onnx_test_package, device):

        # Use dummy package
        p = PanguModel.load_model(onnx_test_package).to(device)

        dc = p.input_coords()
        del dc["batch"]
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

        assert out.shape == torch.Size(
            [len(time), 1, len(p.output_coords(coords)["variable"]), 721, 1440]
        )
        assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
        assert (out_coords["time"] == time).all()
        assert torch.allclose(
            out, (x + delta_t)
        )  # Phoo model should add by delta t each call
        handshake_dim(out_coords, "lon", 4)
        handshake_dim(out_coords, "lat", 3)
        handshake_dim(out_coords, "variable", 2)
        handshake_dim(out_coords, "lead_time", 1)
        handshake_dim(out_coords, "time", 0)

        torch.cuda.empty_cache()

    @pytest.mark.parametrize(
        "ensemble",
        [1, 2],
    )
    @pytest.mark.parametrize(
        "PanguModel, delta_t",
        [(Pangu24, 24), (Pangu6, 6), (Pangu3, 3)],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda:0"])
    def test_pangu_iter(self, ensemble, PanguModel, delta_t, onnx_test_package, device):
        time = np.array([np.datetime64("1993-04-05T00:00")])
        # Use dummy package
        p = PanguModel.load_model(onnx_test_package).to(device)

        dc = p.input_coords()
        del dc["batch"]
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
        x0 = x.clone()
        p_iter = p.create_iterator(x, coords)

        if not isinstance(time, Iterable):
            time = [time]

        # Get generator
        next(p_iter)  # Skip first which should return the input
        old_coords = coords.copy()
        for i, (out, out_coords) in enumerate(p_iter):
            assert len(out.shape) == 6
            assert out.shape[0] == ensemble
            assert (
                out_coords["variable"] == p.output_coords(old_coords)["variable"]
            ).all()
            assert out_coords["lead_time"][0] == np.timedelta64(delta_t * (i + 1), "h")
            assert torch.allclose(
                out, (x0 + (i + 1) * delta_t), atol=1e-3, rtol=1e-3
            )  # Phoo model should add by delta t each call
            handshake_dim(out_coords, "lon", 5)
            handshake_dim(out_coords, "lat", 4)
            handshake_dim(out_coords, "variable", 3)
            handshake_dim(out_coords, "lead_time", 2)
            handshake_dim(out_coords, "time", 1)
            handshake_dim(out_coords, "ensemble", 0)
            old_coords = out_coords.copy()
            if i > 8:
                break

        torch.cuda.empty_cache()

    @pytest.mark.parametrize(
        "dc",
        [
            OrderedDict({"lat": np.random.randn(720)}),
            OrderedDict({"lat": np.random.randn(720), "phoo": np.random.randn(1440)}),
            OrderedDict({"lat": np.random.randn(720), "lon": np.random.randn(1)}),
        ],
    )
    @pytest.mark.parametrize(
        "PanguModel, delta_t",
        [(Pangu24, 24), (Pangu6, 6), (Pangu3, 3)],
    )
    @pytest.mark.parametrize("device", ["cuda:0"])
    def test_pangu_exceptions(self, dc, PanguModel, delta_t, onnx_test_package, device):
        # Test invalid coordinates error
        time = np.array([np.datetime64("1993-04-05T00:00")])
        # Use dummy package
        p = PanguModel.load_model(onnx_test_package)

        # Initialize Data Source
        r = Random(dc)

        # Get Data and convert to tensor, coords
        lead_time = p.input_coords()["lead_time"]
        variable = p.input_coords()["variable"]
        x, coords = fetch_data(r, time, variable, lead_time, device=device)

        with pytest.raises((KeyError, ValueError)):
            p(x, coords)

        torch.cuda.empty_cache()


@pytest.mark.package
@pytest.mark.parametrize(
    "PanguModel, delta_t",
    [(Pangu24, 24), (Pangu6, 6), (Pangu3, 3)],
)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_pangu_package(PanguModel, delta_t, device):
    time = np.array([np.datetime64("1993-04-05T00:00")])
    with torch.device(device):
        package = PanguModel.load_default_package()
        p = PanguModel.load_model(package).to(device)

    dc = p.input_coords()
    del dc["batch"]
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

    assert out.shape == torch.Size([len(time), 1, 69, 721, 1440])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)

    del p
    torch.cuda.empty_cache()
    gc.collect()
