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

from earth2studio.models.dx import PrecipitationAFNO
from earth2studio.utils.coords import coord_array
from earth2studio.utils.cupy import from_torch


@pytest.fixture(autouse=True)
def cupy_for_cuda(request, monkeypatch):
    if hasattr(request.node, "callspec") and request.node.callspec.params.get(
        "device", "cpu"
    ).startswith("cuda"):
        pytest.importorskip("cupy")
    if "package" not in request.node.keywords:
        # These tests supply the complete Torch core; package imports are unused.
        monkeypatch.setattr(
            PrecipitationAFNO, "__init__", PrecipitationAFNO.__init__.__wrapped__
        )


class PhooAFNOPrecip(torch.nn.Module):
    def forward(self, x):
        return x[:, :1, :, :]


@pytest.mark.parametrize(
    "x",
    [
        torch.randn(1, 20, 720, 1440),
        torch.randn(2, 20, 720, 1440),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_afno_precip(x, device):
    # Just test forward pass of AFNO with spoof model
    model = PhooAFNOPrecip()
    center = torch.zeros(20, 1, 1)
    scale = torch.ones(20, 1, 1)

    dx = PrecipitationAFNO(model, center, scale).to(device)
    x = x.to(device)

    coords = OrderedDict(
        {
            "batch": np.arange(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )
    coords = coord_array(tuple(coords), coords, attrs=dx.input_coords().attrs)

    array = from_torch(x, coords, attrs=dx.input_coords().attrs)
    result = dx(array)
    out, out_coords = result.e2s.to_torch()

    assert out.shape == torch.Size([x.shape[0], 1, 720, 1440])
    assert out_coords["variable"] == dx.output_coords(coords)["variable"]
    assert result.dims == ("batch", "variable", "lat", "lon")


def test_precipitationafno_output_signature():
    model = PhooAFNOPrecip()
    center = torch.zeros(20, 1, 1)
    scale = torch.ones(20, 1, 1)
    dx = PrecipitationAFNO(model, center, scale)
    output = dx.output_coords(dx.input_coords())
    assert output.data.nbytes == 0
    assert output["variable"].values.tolist() == ["tp:sum:6h"]
    assert "tp:sum:6h" in output.attrs["earth2studio_statistics"]


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_afno_precip_package(device):
    package = PrecipitationAFNO.load_default_package()
    dx = PrecipitationAFNO.load_model(package).to(device)
    x = torch.randn(2, 20, 720, 1440).to(device)
    coords = OrderedDict(
        {
            "batch": np.arange(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )
    coords = coord_array(tuple(coords), coords, attrs=dx.input_coords().attrs)

    array = from_torch(x, coords, attrs=dx.input_coords().attrs)
    result = dx(array)
    out, out_coords = result.e2s.to_torch()
    assert out.shape == torch.Size([x.shape[0], 1, 720, 1440])
    assert out_coords["variable"] == dx.output_coords(coords)["variable"]
    assert result.dims == ("batch", "variable", "lat", "lon")


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_afno_exceptions(device):

    model = PhooAFNOPrecip()
    center = torch.zeros(20)
    scale = torch.ones(20)

    dx = PrecipitationAFNO(model, center, scale).to(device)
    x = torch.randn(1, 20, 720, 1440).to(device)
    wrong_coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "wrong": dx.input_coords()["variable"].values,
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )
    wrong_coords = coord_array(
        tuple(wrong_coords), wrong_coords, attrs=dx.input_coords().attrs
    )

    with pytest.raises((KeyError, ValueError)):
        dx(from_torch(x, wrong_coords, attrs=dx.input_coords().attrs))

    wrong_coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lon": dx.input_coords()["lon"],
            "lat": dx.input_coords()["lat"],
        }
    )
    wrong_coords = coord_array(
        tuple(wrong_coords), wrong_coords, attrs=dx.input_coords().attrs
    )

    with pytest.raises(ValueError):
        dx(from_torch(x.transpose(-1, -2), wrong_coords, attrs=dx.input_coords().attrs))

    wrong_coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lat": np.linspace(-90, 90, 721),
            "lon": dx.input_coords()["lon"],
        }
    )
    wrong_coords = coord_array(
        tuple(wrong_coords), wrong_coords, attrs=dx.input_coords().attrs
    )
    with pytest.raises(ValueError):
        dx(
            from_torch(
                torch.zeros(1, 20, 721, 1440, device=device),
                wrong_coords,
                attrs=dx.input_coords().attrs,
            )
        )
