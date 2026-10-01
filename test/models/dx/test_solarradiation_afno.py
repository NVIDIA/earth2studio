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
from earth2studio.models.dx import SolarRadiationAFNO1H, SolarRadiationAFNO6H
from earth2studio.utils import coord_array, coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch


@pytest.fixture(autouse=True)
def optional_backend(request, monkeypatch):
    if (
        "device" in request.fixturenames
        and request.getfixturevalue("device").startswith("cuda")
        and not torch.cuda.is_available()
    ):
        pytest.skip("CUDA unavailable")
    if request.node.get_closest_marker("package") is None:
        monkeypatch.setattr(
            "earth2studio.models.dx.solarradiation_afno.cos_zenith_angle",
            lambda time, lon, lat: np.full(np.shape(lon), time.hour),
        )


class PhooAFNOSolarRadiation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.solar = []

    def forward(self, x):
        self.solar.append(float(x[0, 24, 0, 0]))
        return x[:, :1, :, :]


@pytest.fixture(params=[SolarRadiationAFNO1H, SolarRadiationAFNO6H])
def model_class(request):
    return request.param


def make_model(model_class, device="cpu"):
    model = model_class.__new__(model_class)
    inspect.unwrap(model_class.__init__)(
        model,
        PhooAFNOSolarRadiation(),
        model_class.freq,
        torch.ones(24, 1, 1),
        torch.full((24, 1, 1), 2.0),
        torch.ones(1, 1, 1),
        torch.full((1, 1, 1), 3.0),
        torch.zeros(1, 1, 4, 6),
        torch.zeros(1, 1, 4, 6),
        torch.zeros(1, 4, 4, 6),
    )
    declared = model.input_coords()
    assert declared.data.nbytes == 0
    assert declared.attrs["earth2studio_grid_id"] == "latlon-0.25deg"
    signature = coord_array(
        declared.dims,
        {"variable": declared.coords["variable"]},
        dynamic=declared.attrs["earth2studio_dynamic_dims"],
        grid=LatLonGrid(np.linspace(90, -90, 4), np.arange(6) * 60),
    )
    model.input_coords = lambda: signature.copy()
    return model.to(device)


def make_input(model, shape, device):
    batch, times, leads = shape
    coords = coord_array_like(
        model.input_coords(),
        {
            "batch": np.arange(batch),
            "time": np.datetime64("2024-01-01", "ns")
            + np.arange(times) * np.timedelta64(1, "h"),
            "lead_time": np.arange(leads) * np.timedelta64(int(model.freq[:-1]), "h"),
        },
    )
    x = from_torch(torch.randn(coords.shape, device=device), coords, name="weather")
    x = x.assign_coords(
        units=("variable", ["input"] * x.sizes["variable"]),
        valid_time=(("time", "lead_time"), x.time.values[:, None] + x.lead_time.values),
    )
    x.attrs["source"] = "fixture"
    x.encoding = {"source": "fixture"}
    return x


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 2, 2)])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_solarradiation_afno(shape, device, model_class):
    model = make_model(model_class, device)
    x = make_input(model, shape, device)
    before = x.copy(deep=True)
    out = model(x)
    assert out.shape == (*shape, 1, 4, 6)
    assert out.coords["variable"].values.tolist() == [f"ssrd:sum:{model.freq}"]
    handshake_dataarray(out, model.output_coords(x))
    tensor, _ = out.e2s.to_torch()
    assert torch.all(tensor >= 0)
    assert torch.all(tensor <= 1e6)
    source = x.e2s.to_torch()[0]
    torch.testing.assert_close(
        tensor, torch.clamp((source[..., :1, :, :] - 1) / 2 * 3 + 1, min=0)
    )
    assert tensor.device == torch.device(device)
    assert out.name == x.name and out.encoding == x.encoding
    assert out.attrs["source"] == x.attrs["source"] and "units" not in out.coords
    assert out.attrs["earth2studio_crs"] == x.attrs["earth2studio_crs"]
    xr.testing.assert_identical(out.valid_time, x.valid_time)
    xr.testing.assert_identical(x, before)
    hours = (
        ((x.valid_time.values - x.time.values[0]) / np.timedelta64(1, "h"))
        .ravel()
        .tolist()
    )
    assert model.core_model.solar == hours * shape[0]


@pytest.mark.parametrize("variable", ["wrong_var", "t2m", "sza"])
def test_solarradiation_afno_invalid_coords(variable, model_class):
    model = make_model(model_class)
    x = (
        make_input(model, (1, 1, 1), "cpu")
        .isel(variable=[0])
        .assign_coords(variable=[variable])
    )
    with pytest.raises(ValueError):
        model(x)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_solarradiation_afno_exceptions(device, model_class):
    model = make_model(model_class, device)
    x = make_input(model, (1, 1, 1), device)
    for wrong in (
        x.isel(lat=slice(1, None)),
        x.drop_vars("lat"),
        x.assign_coords(lead_time=[0]),
        x.assign_coords(time=np.array(["NaT"], dtype="datetime64[ns]")),
        x.assign_attrs(earth2studio_crs="EPSG:3857"),
    ):
        with pytest.raises(ValueError):
            model(wrong)


def test_solarradiation_afno_conformance(model_class):
    model = make_model(model_class)
    assert check_diagnostic_contract(model) == [
        "D10: model does not declare itself stochastic"
    ]


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_solarradiation_afno_package(device, model_class):
    pytest.importorskip("physicsnemo")
    package = model_class.load_default_package()
    dx = model_class.load_model(package).to(device)
    x = make_input(dx, (2, 1, 1), device)
    out = dx(x)
    assert out.shape == (2, 1, 1, 1, 721, 1440)
    handshake_dataarray(out, dx.output_coords(x))
    assert torch.all(out.e2s.to_torch()[0] >= 0)
