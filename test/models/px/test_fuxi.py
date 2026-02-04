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

try:
    import onnx  # noqa
except ImportError:
    pytest.skip("onnx not installed which is needed for tests", allow_module_level=True)

from earth2studio.data import Random, fetch_data
from earth2studio.models.auto import Package
from earth2studio.models.px import FuXi
from earth2studio.utils import handshake_dim


class PhooFuXiModel(torch.nn.Module):
    """Dummy FuXi model, adds time-step"""

    def __init__(self, model_type: str = "short"):
        super().__init__()
        # Model cascade testing
        if model_type == "short":
            self.delta_t = 1
        elif model_type == "medium":
            self.delta_t = 2
        else:
            self.delta_t = 3

    def forward(self, x, y):
        # Remove first time-step
        assert y.shape[1] == 12
        # Add  0*y[0,0] so input y remains in ONNX graph
        output = x + self.delta_t + 0 * y[0, 0]
        return output


@pytest.fixture(scope="class")
def fuxi_test_package(tmp_path_factory):
    """Creates a bunch of spoof ONNX models to unit test with"""
    tmp_path = tmp_path_factory.mktemp("data")

    for model in ["short", "medium", "long"]:
        onnx_path = tmp_path / f"{model}.onnx"
        torch.onnx.export(
            PhooFuXiModel(model_type=model),
            args=(torch.rand(1, 2, 70, 721, 1440), torch.rand(1, 12)),
            f=str(onnx_path),
            export_params=True,
            opset_version=10,
            input_names=["input", "temb"],
            output_names=["output"],
        )
        # Empty weight file
        open(tmp_path / f"{model}", "a").close()

    return Package(str(tmp_path))


class TestFuXiMock:
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
    def test_fuxi_call(self, time, fuxi_test_package, device):

        # Use dummy package
        p = FuXi.load_model(fuxi_test_package).to(device)

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

        assert out.shape == torch.Size(
            [
                len(time),
                1,
                len(p.output_coords(p.input_coords())["variable"]),
                721,
                1440,
            ]
        )
        assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
        assert (out_coords["time"] == time).all()
        assert torch.allclose(
            out[:, :, :-1],
            (x[:, 1:, :-1] + 1),  # Ignore last field with is tp b/c mm conversion
        )  # Phoo model should add by delta t each call
        handshake_dim(out_coords, "lon", 4)
        handshake_dim(out_coords, "lat", 3)
        handshake_dim(out_coords, "variable", 2)
        handshake_dim(out_coords, "lead_time", 1)
        handshake_dim(out_coords, "time", 0)

    @pytest.mark.parametrize(
        "ensemble",
        [2],
    )
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_fuxi_iter(self, ensemble, fuxi_test_package, device):
        time = np.array([np.datetime64("1993-04-05T00:00")])
        # Use dummy package
        p = FuXi.load_model(fuxi_test_package).to(device)

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
        out, out_coords = next(p_iter)  # Skip first which should return the input
        assert torch.allclose(
            out[:, :, :-1], x[:, 1:, :-1]
        )  # Ignore last field with is tp b/c mm conversion

        step_index = 0
        for i, (out, out_coords) in enumerate(p_iter):
            # Test the model cascade
            if i < 20:
                step_index += 1
            elif i < 40:
                step_index += 2
            else:
                step_index += 3

            assert len(out.shape) == 6
            assert out.shape[0] == ensemble
            assert (
                out_coords["variable"] == p.output_coords(p.input_coords())["variable"]
            ).all()
            assert (out_coords["time"] == time).all()
            assert out_coords["lead_time"][0] == np.timedelta64(6 * (i + 1), "h")
            assert torch.allclose(
                out[:, :, :-1], (x[:, 1:, :-1] + step_index)
            )  # Phoo model should add by delta t each call
            handshake_dim(out_coords, "lon", 5)
            handshake_dim(out_coords, "lat", 4)
            handshake_dim(out_coords, "variable", 3)
            handshake_dim(out_coords, "lead_time", 2)
            handshake_dim(out_coords, "time", 1)
            handshake_dim(out_coords, "ensemble", 0)

            if i > 41:  # Long test because of model cascade
                break

        # Test forward pass reloads short model
        out, out_coords = p(x, coords)
        assert out.shape == torch.Size(
            [
                ensemble,
                len(time),
                1,
                len(p.output_coords(p.input_coords())["variable"]),
                721,
                1440,
            ]
        )
        assert (
            out_coords["variable"] == p.output_coords(p.input_coords())["variable"]
        ).all()
        assert torch.allclose(
            out[:, :, :-1],
            (x[:, 1:, :-1] + 1),  # Ignore last field with is tp b/c mm conversion
        )  # Phoo model should add by delta t each call

    @pytest.mark.parametrize(
        "dc",
        [
            OrderedDict({"lat": np.random.randn(721)}),
            OrderedDict({"lat": np.random.randn(721), "phoo": np.random.randn(1440)}),
            OrderedDict({"lat": np.random.randn(721), "lon": np.random.randn(1)}),
        ],
    )
    @pytest.mark.parametrize("device", ["cuda:0"])
    def test_fuxi_exceptions(self, dc, fuxi_test_package, device):
        # Test invalid coordinates error
        time = np.array([np.datetime64("1993-04-05T00:00")])
        # Use dummy package
        p = FuXi.load_model(fuxi_test_package).to(device)

        # Initialize Data Source
        r = Random(dc)

        # Get Data and convert to tensor, coords
        lead_time = p.input_coords()["lead_time"]
        variable = p.input_coords()["variable"]
        x, coords = fetch_data(r, time, variable, lead_time, device=device)

        with pytest.raises((KeyError, ValueError)):
            p(x, coords)


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_fuxi_package(device):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("1993-04-05T00:00")])
    with torch.device(device):
        package = FuXi.load_default_package()
        p = FuXi.load_model(package).to(device)

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

    assert out.shape == torch.Size(
        [len(time), 1, len(p.output_coords(coords)["variable"]), 721, 1440]
    )
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)
