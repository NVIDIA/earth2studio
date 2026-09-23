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

import inspect

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.grids import LatLonGrid
from earth2studio.models.conformance import check_diagnostic_contract
from earth2studio.models.dx import ClimateNet
from earth2studio.utils import coord_array, coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch


class PhooCNet(torch.nn.Module):
    def forward(self, x):
        return x[:, :3, :, :]


@pytest.fixture(autouse=True)
def optional_device(request):
    if (
        "device" in request.fixturenames
        and request.getfixturevalue("device").startswith("cuda")
        and not torch.cuda.is_available()
    ):
        pytest.skip("CUDA unavailable")


@pytest.fixture
def diagnostic():
    model = ClimateNet.__new__(ClimateNet)
    inspect.unwrap(ClimateNet.__init__)(
        model, PhooCNet(), torch.ones(4, 1, 1), torch.full((4, 1, 1), 2.0)
    )
    declared = model.input_coords()
    assert declared.attrs["earth2studio_grid_id"] == "latlon-0.25deg"
    signature = coord_array(
        declared.dims,
        {"variable": declared.coords["variable"]},
        dynamic=("batch",),
        grid=LatLonGrid([45, -45], [0, 120, 240]),
    )
    model.input_coords = lambda: signature.copy()
    return model


@pytest.mark.parametrize(
    "batch",
    [
        1,
        2,
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cnet(batch, device, diagnostic):
    dx = diagnostic.to(device)
    coords = coord_array_like(dx.input_coords(), {"batch": np.arange(batch)})
    x = from_torch(
        torch.randn(coords.shape, device=device), coords, name="weather"
    ).rename(batch="member")
    x = x.assign_coords(
        units=("variable", ["input"] * 4), experiment=("member", np.arange(batch))
    )
    x.encoding = {"source": "fixture"}
    original = x.copy(deep=True)
    out = dx(x)

    assert out.shape == torch.Size([x.shape[0], 3, 2, 3])
    handshake_dataarray(out, dx.output_coords(x))
    np.testing.assert_allclose(
        out.sum("variable").e2s.to_torch()[0].cpu(), 1, atol=1e-6
    )
    torch.testing.assert_close(
        out.e2s.to_torch()[0],
        torch.softmax((x.e2s.to_torch()[0][..., :3, :, :] - 1) / 2, -3),
    )
    assert out.name == x.name and out.encoding == x.encoding
    assert "units" not in out.coords
    xr.testing.assert_identical(out.experiment, x.experiment)
    xr.testing.assert_identical(x, original)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_cnet_exceptions(device, diagnostic):
    dx = diagnostic.to(device)
    coords = coord_array_like(dx.input_coords(), {"batch": [0]})
    x = from_torch(torch.zeros(coords.shape, device=device), coords)
    for wrong in (
        x.rename(variable="wrong"),
        x.transpose("batch", "variable", "lon", "lat"),
        x.isel(lat=slice(1, None)),
    ):
        with pytest.raises(ValueError):
            dx(wrong)


def test_climatenet_conformance(diagnostic):
    dx = diagnostic
    assert check_diagnostic_contract(dx) == [
        "D10: model does not declare itself stochastic"
    ]


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_cnet_package(device):
    # Only cuda supported
    package = ClimateNet.load_default_package()
    dx = ClimateNet.load_model(package).to(device)
    coords = coord_array_like(dx.input_coords(), {"batch": [0, 1]})
    x = from_torch(torch.randn(coords.shape, device=device), coords)
    out = dx(x)
    assert out.shape == torch.Size([x.shape[0], 3, 721, 1440])
    # Check that we get 0-1 masks
    tensor, _ = out.e2s.to_torch()
    assert torch.all(tensor <= 1)
    assert torch.all(tensor >= 0)
    # Check variables
    handshake_dataarray(out, dx.output_coords(x))
