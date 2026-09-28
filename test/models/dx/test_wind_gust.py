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
from earth2studio.models.dx import WindgustAFNO
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
            "earth2studio.models.dx.wind_gust.cos_zenith_angle",
            lambda time, lon, lat: np.full(np.shape(lon), time.hour),
        )


class PhooAFNOWindgust(torch.nn.Module):
    def forward(self, x):
        return x[:, :1, :, :]


def make_model(device="cpu"):
    model = WindgustAFNO.__new__(WindgustAFNO)
    inspect.unwrap(WindgustAFNO.__init__)(
        model,
        PhooAFNOWindgust(),
        torch.ones(1, 1, 4, 6),
        torch.ones(1, 1, 4, 6),
        torch.ones(17, 1, 1),
        torch.full((17, 1, 1), 2.0),
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


def make_input(model, batch, times, device):
    coords = coord_array_like(
        model.input_coords(),
        {
            "batch": np.arange(batch),
            "time": np.datetime64("2024-01-01", "ns")
            + np.arange(times) * np.timedelta64(1, "D"),
            "lead_time": np.array([0], dtype="timedelta64[h]"),
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


@pytest.mark.parametrize("batch,times", [(1, 1), (2, 2)])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_afno_windgust(batch, times, device):
    dx = make_model(device)
    x = make_input(dx, batch, times, device)
    before = x.copy(deep=True)
    out = dx(x)
    assert out.shape == (batch, times, 1, 1, 4, 6)
    assert out.coords["variable"].values.tolist() == ["fg10m:max:1h"]
    handshake_dataarray(out, dx.output_coords(x))
    tensor, _ = out.e2s.to_torch()
    source, _ = x.e2s.to_torch()
    torch.testing.assert_close(
        tensor, torch.clamp((source[..., :1, :, :] - 1) / 2, min=0)
    )
    assert tensor.device == torch.device(device)
    assert out.name == x.name and out.encoding == x.encoding
    assert out.attrs["source"] == x.attrs["source"] and "units" not in out.coords
    assert out.attrs["earth2studio_crs"] == x.attrs["earth2studio_crs"]
    xr.testing.assert_identical(out.valid_time, x.valid_time)
    xr.testing.assert_identical(x, before)


def test_afno_windgust_sza_latlon_order():
    dx = make_model()
    x = make_input(dx, 1, 1, "cpu")
    captured = {}

    def spy_sza(lon, lat, time, lead_time):
        captured.update(lon=np.asarray(lon), lat=np.asarray(lat))
        return torch.zeros(4, 6)

    dx._compute_sza = spy_sza
    dx(x)
    np.testing.assert_array_equal(captured["lon"][0], x.coords["lon"])
    np.testing.assert_array_equal(captured["lat"][:, 0], x.coords["lat"])


def test_windgust_afno_conformance():
    model = make_model()
    assert check_diagnostic_contract(model) == [
        "D10: model does not declare itself stochastic"
    ]


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_afno_windgust_package(device):
    pytest.importorskip("physicsnemo")
    package = WindgustAFNO.load_default_package()
    dx = WindgustAFNO.load_model(package).to(device)
    x = make_input(dx, 2, 1, device)
    out = dx(x)
    assert out.shape == (2, 1, 1, 1, 720, 1440)
    assert torch.all(out.e2s.to_torch()[0] >= 0)
    handshake_dataarray(out, dx.output_coords(x))


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_afno_windgust_exceptions(device):
    dx = make_model(device)
    x = make_input(dx, 1, 1, device)
    for wrong in (
        x.rename(variable="wrong"),
        x.isel(lat=slice(1, None)),
        x.assign_coords(lead_time=[0]),
        x.assign_coords(time=np.array(["NaT"], dtype="datetime64[ns]")),
        x.assign_attrs(earth2studio_crs="EPSG:3857"),
    ):
        with pytest.raises(ValueError):
            dx(wrong)
