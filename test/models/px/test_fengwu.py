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
from earth2studio.models.auto import Package
from earth2studio.models.px import FengWu
from earth2studio.utils import handshake_dim


class PhooFengWuModel(torch.nn.Module):
    """Dummy fengwu model, adds time-step"""

    def __init__(self, delta_t: int = 6):
        super().__init__()
        self.delta_t = delta_t

    def forward(self, x):
        # Remove first time-step
        return (
            torch.cat([x[:, 69:], torch.empty_like(x[:, 69:])], axis=1) + self.delta_t
        )


@pytest.fixture(scope="class")
def fengwu_test_package(tmp_path_factory):
    """Creates a bunch of spoof ONNX models to unit test with"""
    tmp_path = tmp_path_factory.mktemp("data")

    onnx_path = tmp_path / "fengwu_v1.onnx"
    torch.onnx.export(
        PhooFengWuModel(),
        torch.rand(
            2, 138, 721, 1440
        ),  # https://github.com/pytorch/pytorch/issues/165259#issuecomment-3394619898
        str(onnx_path),
        export_params=True,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_shapes=({0: "batch_size"},),
    )
    # Create fake normalization files
    np.save(tmp_path / "global_means.npy", np.zeros(69))
    np.save(tmp_path / "global_stds.npy", np.ones(69))
    return Package(str(tmp_path))


class TestFengWuMock:

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
    def test_fengwu_call(self, time, fengwu_test_package, device):

        # Use dummy package
        p = FengWu.load_model(fengwu_test_package).to(device)

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
            out, (x[:, 1:] + 6)
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
    @pytest.mark.parametrize("device", ["cpu", "cuda"])
    def test_fengwu_iter(self, ensemble, fengwu_test_package, device):
        time = np.array([np.datetime64("1993-04-05T00:00")])
        # Use dummy package
        p = FengWu.load_model(fengwu_test_package).to(device)

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

        p_iter = p.create_iterator(x, coords)

        if not isinstance(time, Iterable):
            time = [time]

        # Get generator
        out, out_coords = next(p_iter)  # Skip first which should return the input
        assert torch.allclose(out, x[:, 1:])
        for i, (out, out_coords) in enumerate(p_iter):
            assert len(out.shape) == 6
            assert out.shape[0] == ensemble
            assert (
                out_coords["variable"] == p.output_coords(p.input_coords())["variable"]
            ).all()
            assert out_coords["lead_time"][0] == np.timedelta64(6 * (i + 1), "h")
            assert torch.allclose(
                out, (x[:, 1:] + (i + 1) * 6)
            )  # Phoo model should add by delta t each call
            handshake_dim(out_coords, "lon", 5)
            handshake_dim(out_coords, "lat", 4)
            handshake_dim(out_coords, "variable", 3)
            handshake_dim(out_coords, "lead_time", 2)
            handshake_dim(out_coords, "time", 1)
            handshake_dim(out_coords, "ensemble", 0)

            if i > 3:
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
    @pytest.mark.parametrize("device", ["cuda:0"])
    def test_fengwu_exceptions(self, dc, fengwu_test_package, device):
        # Test invalid coordinates error
        time = np.array([np.datetime64("1993-04-05T00:00")])
        # Use dummy package
        p = FengWu.load_model(fengwu_test_package).to(device)

        # Initialize Data Source
        r = Random(dc)

        # Get Data and convert to tensor, coords
        lead_time = p.input_coords()["lead_time"]
        variable = p.input_coords()["variable"]
        x, coords = fetch_data(r, time, variable, lead_time, device=device)

        with pytest.raises((KeyError, ValueError, RuntimeError)):
            p(x, coords)


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_fengwu_package(device):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("1993-04-05T00:00")])
    with torch.device(device):
        package = FengWu.load_default_package()
        p = FengWu.load_model(package).to(device)

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
