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
from earth2studio.models.dx import PrecipitationAFNOv2
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
            "earth2studio.models.dx.precipitation_afno_v2.cos_zenith_angle",
            lambda time, lon, lat: np.full(np.shape(lon), time.hour),
        )


class PhooAFNOPrecipV2(torch.nn.Module):
    def forward(self, x):
        return x[:, :1, :, :]


def make_model(device="cpu"):
    model = PrecipitationAFNOv2.__new__(PrecipitationAFNOv2)
    inspect.unwrap(PrecipitationAFNOv2.__init__)(
        model,
        PhooAFNOPrecipV2(),
        torch.zeros(1, 1, 4, 6),
        torch.zeros(1, 1, 4, 6),
        torch.ones(20, 1, 1),
        torch.full((20, 1, 1), 2.0),
    )
    declared = model.input_coords()
    assert declared.data.nbytes == 0
    assert (
        declared.attrs["earth2studio_grid_id"] == "latlon-0.25deg-south-pole-excluded"
    )
    signature = coord_array(
        declared.dims,
        {"variable": declared.coords["variable"]},
        dynamic=declared.attrs["earth2studio_dynamic_dims"],
        grid=LatLonGrid(np.linspace(90, -90, 4), np.arange(6, dtype=float) * 60),
    )
    model.input_coords = lambda: signature.copy()
    return model.to(device)


def make_input(model, batch, device):
    coords = coord_array_like(
        model.input_coords(),
        {
            "batch": np.arange(batch),
            "time": np.array(["2023-01-01T00:00"], dtype="datetime64[ns]"),
            "lead_time": np.array([6], dtype="timedelta64[h]"),
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


@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_afno_precip_v2(batch, device):
    dx = make_model(device)
    x = make_input(dx, batch, device)
    before = x.copy(deep=True)
    out = dx(x)
    assert out.shape == (batch, 1, 1, 1, 4, 6)
    handshake_dataarray(out, dx.output_coords(x))
    tensor, _ = out.e2s.to_torch()
    source, _ = x.e2s.to_torch()
    expected = torch.clamp(
        1e-5 * (torch.exp((source[..., :1, :, :] - 1) / 2) - 1) / 1000, min=0
    )
    torch.testing.assert_close(tensor, expected)
    assert tensor.device == torch.device(device)
    assert out.name == x.name and out.encoding == x.encoding
    assert out.attrs["source"] == x.attrs["source"] and "units" not in out.coords
    assert out.attrs["earth2studio_crs"] == x.attrs["earth2studio_crs"]
    xr.testing.assert_identical(out.valid_time, x.valid_time)
    xr.testing.assert_identical(x, before)


def test_afno_precip_v2_sza_latlon_order():
    dx = make_model()
    x = make_input(dx, 1, "cpu")
    captured = {}

    def spy_sza(lon, lat, time, lead_time):
        captured.update(lon=np.asarray(lon), lat=np.asarray(lat))
        return torch.zeros(4, 6)

    dx._compute_sza = spy_sza
    dx(x)
    np.testing.assert_array_equal(captured["lon"][0], x.coords["lon"])
    np.testing.assert_array_equal(captured["lat"][:, 0], x.coords["lat"])


def test_precipitationafnov2_conformance():
    model = make_model()
    assert check_diagnostic_contract(model) == [
        "D10: model does not declare itself stochastic"
    ]


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_afno_precip_v2_package(device):
    pytest.importorskip("physicsnemo")
    package = PrecipitationAFNOv2.load_default_package()
    dx = PrecipitationAFNOv2.load_model(package).to(device)
    x = make_input(dx, 2, device)
    out = dx(x)
    assert out.shape == (2, 1, 1, 1, 720, 1440)
    handshake_dataarray(out, dx.output_coords(x))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_afno_v2_exceptions(device):
    dx = make_model(device)
    x = make_input(dx, 1, device)
    for wrong in (
        x.rename(variable="wrong"),
        x.transpose("batch", "time", "lead_time", "variable", "lon", "lat"),
        x.isel(lat=slice(1, None)),
        x.assign_coords(lead_time=[6]),
        x.assign_coords(time=np.array(["NaT"], dtype="datetime64[ns]")),
        x.assign_attrs(earth2studio_crs="EPSG:3857"),
    ):
        with pytest.raises(ValueError):
            dx(wrong)
