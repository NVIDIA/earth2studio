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
import xarray as xr

import earth2studio.grids as grids
from earth2studio.grids import CurvilinearGrid, LatLonGrid, resolve_grid
from earth2studio.models.conformance import check_diagnostic_contract
from earth2studio.models.dx import (
    DerivedRH,
    DerivedRHDewpoint,
    DerivedSurfacePressure,
    DerivedTCWV,
    DerivedVPD,
    DerivedWS,
)
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.cupy import from_torch


def _grid(shape=(2, 3), endpoint=True):
    return LatLonGrid(
        np.linspace(-90, 90, shape[-2]),
        np.linspace(0, 360, shape[-1], endpoint=endpoint),
    )


def _field(tensor, coords):
    return from_torch(
        tensor,
        coord_array(
            tuple(coords),
            {key: np.asarray(value) for key, value in coords.items()},
            grid=LatLonGrid(np.asarray(coords["lat"]), np.asarray(coords["lon"])),
        ),
    )


def _model(kind, grid):
    if kind is DerivedSurfacePressure:
        return kind(
            [1000, 900],
            torch.full(grid.shape, 500.0),
            OrderedDict((dim, np.asarray(grid.coords()[dim])) for dim in grid.dims),
            corr_adjustment=(0.0, 1.0),
        )
    return kind(grid=grid)


DERIVED_MODELS = [
    DerivedWS,
    DerivedRH,
    DerivedRHDewpoint,
    DerivedVPD,
    DerivedSurfacePressure,
    DerivedTCWV,
]


@pytest.mark.parametrize("kind", DERIVED_MODELS)
@pytest.mark.parametrize(
    "leading,dtype",
    [
        ({}, torch.float32),
        ({"batch": [7, 3]}, torch.float64),
        ({"member": [2, 5], "batch": [8, 4], "time": [0, 1]}, torch.float32),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_derived_native_metadata(kind, leading, dtype, device):
    grid = _grid()
    model = _model(kind, grid).to(device=device, dtype=dtype)
    declared = model.input_coords()
    assert isinstance(declared, xr.DataArray)
    assert declared.data.nbytes == 0
    assert declared.attrs["earth2studio_dynamic_dims"] == ("batch",)
    assert declared.sizes["lat"] == 2
    assert declared.sizes["lon"] == 3
    signature = coord_array(
        (*leading, "variable", "lat", "lon"),
        {**leading, "variable": declared.coords["variable"].values},
        grid=grid,
        dtype=np.float64 if dtype == torch.float64 else np.float32,
        name="weather",
        attrs={"source": "fixture"},
    )
    tensor = torch.ones(signature.shape, device=device, dtype=dtype)
    # Finite, physically plausible probes for the thermodynamic kernels.
    for i, variable in enumerate(declared.coords["variable"].values):
        value = 290 if variable.startswith("t") else 1
        if variable.startswith("q"):
            value = 0.005
        elif variable.startswith("d"):
            value = 285
        elif variable.startswith("r"):
            value = 50
        elif variable == "sp":
            value = 101325
        elif variable == "z1000":
            value = 0
        elif variable == "z900":
            value = 1000
        tensor[..., i, :, :] = value
    x = from_torch(tensor, signature).assign_coords(
        orography=(("lat", "lon"), np.arange(6).reshape(2, 3)),
        units=("variable", ["input units"] * declared.sizes["variable"]),
        experiment="control",
    )
    if "member" in leading:
        x = x.assign_coords(member_name=("member", ["a", "b"]))
    x.coords["orography"].attrs["units"] = "m"
    x.encoding = {"source": "weather.nc"}
    original = x.copy(deep=True)
    planned = model.output_coords(x)
    assert planned.data.nbytes == 0
    assert planned.encoding == x.encoding
    out = model(x)
    result, _ = out.e2s.to_torch()
    assert result.device == tensor.device
    assert result.dtype == dtype
    assert torch.isfinite(result).all()
    assert out.dims == x.dims
    assert out.name == x.name
    assert out.encoding == x.encoding
    assert out.attrs == x.attrs
    assert "units" not in out.coords
    xr.testing.assert_identical(out.coords.to_dataset(), planned.coords.to_dataset())
    xr.testing.assert_identical(x.e2s.as_numpy(), original.e2s.as_numpy())
    xr.testing.assert_identical(out.e2s.as_numpy(), model(x).e2s.as_numpy())
    for name in ["orography", "experiment", *leading]:
        xr.testing.assert_identical(out.coords[name], x.coords[name])


@pytest.mark.parametrize("kind", DERIVED_MODELS)
@pytest.mark.parametrize(
    "grid_name", ["latlon-0.25deg", "latlon-0.25deg-south-pole-excluded"]
)
def test_derived_builtin_grid(kind, grid_name):
    grid = resolve_grid(grid_name)
    model = _model(kind, grid)
    signature = model.input_coords()
    assert signature.attrs["earth2studio_grid_id"] == grid_name
    assert signature.data.nbytes == 0
    assert signature.shape[-2:] == grid.shape
    if kind is not DerivedSurfacePressure and grid_name == "latlon-0.25deg":
        assert kind().input_coords().attrs["earth2studio_grid_id"] == grid_name


@pytest.mark.parametrize("kind", DERIVED_MODELS)
@pytest.mark.parametrize(
    "invalid", ["variable_order", "latitude", "dimension_order", "crs"]
)
def test_derived_native_invalid_input(kind, invalid):
    model = _model(kind, _grid())
    signature = coord_array_like(model.input_coords(), {"batch": [0]})
    x = from_torch(torch.ones(signature.shape), signature)
    if invalid == "variable_order":
        x = x.isel(variable=list(range(x.sizes["variable"]))[::-1])
    elif invalid == "latitude":
        x = x.assign_coords(lat=x.lat + 1)
    elif invalid == "dimension_order":
        x = x.transpose("batch", "variable", "lon", "lat")
    else:
        x.attrs["earth2studio_crs"] = "EPSG:3857"
    with pytest.raises(ValueError):
        model.output_coords(x)
    with pytest.raises(ValueError):
        model(x)


@pytest.mark.parametrize(
    "levels,shape",
    [
        ([100], (1, 2, 16, 32)),
        ([500, 850], (1, 4, 32, 64)),
        (["10m", "80m"], (1, 4, 16, 32)),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_derived_ws(levels, shape, device):
    """Test wind speed derivation"""
    model = DerivedWS(levels, grid=_grid(shape)).to(device)

    # Test input coordinates
    input_coords = model.input_coords()
    assert "variable" in input_coords.coords
    assert input_coords["variable"].shape[0] == 2 * len(levels)  # u and v components
    assert all(f"u{level}" in input_coords["variable"] for level in levels)
    assert all(f"v{level}" in input_coords["variable"] for level in levels)

    coords = OrderedDict(
        {
            "time": np.arange(1),
            "variable": input_coords["variable"],
            "lat": np.linspace(-90, 90, shape[-2]),
            "lon": np.linspace(0, 360, shape[-1]),
        }
    )

    # Check wind speed calculation
    x = torch.randn(shape).to(device)
    u = x[:, ::2]
    v = x[:, 1::2]
    expected_ws = torch.sqrt(u**2 + v**2).to(device)
    output = model(_field(x, coords))
    out, out_coords = output.e2s.to_torch()
    assert torch.allclose(out, expected_ws, rtol=1e-5)
    assert out.device == x.device
    assert out_coords["variable"].shape[0] == len(levels)

    # Test with zero wind components
    x = torch.zeros(shape).to(device)
    out, _ = model(_field(x, coords)).e2s.to_torch()
    assert torch.allclose(out, torch.zeros_like(out))

    # Test with known values
    x = torch.ones(shape).to(device)
    out, _ = model(_field(x, coords)).e2s.to_torch()
    expected_ws = torch.sqrt(torch.tensor(2.0)).expand_as(out).to(device)
    assert torch.allclose(out, expected_ws)


@pytest.mark.parametrize(
    "invalid_coords",
    [
        OrderedDict({"time": np.array([0]), "variable": np.array(["wrong_var"])}),
        OrderedDict(
            {"time": np.array([0]), "variable": np.array(["u100"])}
        ),  # Missing v component
    ],
)
def test_derived_ws_invalid_coords(invalid_coords):
    """Test wind speed derivation with invalid coordinates"""
    model = DerivedWS([100], grid=_grid((16, 32)))
    x = torch.randn(1, 1, 16, 32)
    invalid_coords.update(
        lat=model.input_coords()["lat"], lon=model.input_coords()["lon"]
    )
    field = _field(x, invalid_coords)

    with pytest.raises(ValueError):
        model(field)

    # Wrong number of variables
    x = torch.randn(1, 3, 16, 32)  # 3 variables instead of 2
    invalid_coords["variable"] = np.array(["u100", "v100", "u200"])
    field = _field(x, invalid_coords)
    with pytest.raises(ValueError):
        model(field)


@pytest.mark.parametrize(
    "levels,shape",
    [
        ([100], (1, 2, 16, 32)),
        ([500, 850], (1, 4, 32, 64)),
        ([100, 200, 300], (1, 6, 32, 64)),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_derived_rh(levels, shape, device):

    model = DerivedRH(levels, grid=_grid(shape)).to(device)

    batch_size = shape[0]
    n_levels = len(levels)
    lat_size = shape[2]
    lon_size = shape[3]

    input_coords = model.input_coords()
    coords = OrderedDict(
        {
            "batch": np.arange(batch_size),
            "variable": input_coords["variable"],
            "lat": np.linspace(-90, 90, lat_size),
            "lon": np.linspace(0, 360, lon_size),
        }
    )

    # Realistic temperature and specific humidity values
    t = 273.15 + 20
    q = 0.01
    x = torch.ones(shape).to(device)
    x[:, ::2] *= t
    x[:, 1::2] *= q

    out, out_coords = model(_field(x, coords)).e2s.to_torch()

    # Check output shape and coordinates
    assert out.shape == (batch_size, n_levels, lat_size, lon_size)
    assert "variable" in out_coords
    assert len(out_coords["variable"]) == len(levels)
    assert all(f"r{level}" in out_coords["variable"] for level in levels)

    # Very cold temperature (210K) should give near 100% RH
    x_cold = torch.ones(shape).to(device)
    x_cold[:, ::2] *= 210.0
    x_cold[:, 1::2] *= 0.001  # Low specific humidity
    out_cold, _ = model(_field(x_cold, coords)).e2s.to_torch()
    assert out_cold.device == x_cold.device
    assert torch.all(out_cold <= 101)
    assert torch.all(out_cold >= 0)
    assert torch.all(out_cold >= 98)

    # Very warm temperature (350K) should give low RH
    x_warm = torch.ones(shape).to(device)
    x_warm[:, ::2] *= 350.0
    x_warm[:, 1::2] *= 0.001  # Low specific humidity
    out_warm, _ = model(_field(x_warm, coords)).e2s.to_torch()
    assert out_warm.device == x_warm.device
    assert torch.all(out_warm <= 100)
    assert torch.all(out_warm >= 0)
    assert torch.all(out_warm <= 2)
    # Warm temperatures should give lower RH than cold temperatures for same q
    assert torch.all(out_warm < out_cold)


@pytest.mark.parametrize("t_celsius", [5.0, 20.0, 30.0])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_derived_rh_above_freezing(t_celsius, device):
    """Regression test: above freezing the mixed-phase liquid-water fraction must
    saturate at 1.0, so saturation vapor pressure equals the pure-water value
    ``es_w``. A larger clip bound (e.g. 1.2) over-weights ``es_w`` and inflates RH.
    """
    levels = [850]
    model = DerivedRH(levels, grid=_grid((16, 32))).to(device)

    input_coords = model.input_coords()
    coords = OrderedDict(
        {
            "batch": np.arange(1),
            "variable": input_coords["variable"],
            "lat": np.linspace(-90, 90, 16),
            "lon": np.linspace(0, 360, 32),
        }
    )

    t = 273.16 + t_celsius
    # Low enough q that RH stays below 100% at all test temperatures, so the
    # output is not masked by the [0, 100] clamp and genuinely exercises the blend.
    q = 0.004
    x = torch.ones((1, 2, 16, 32)).to(device)
    x[:, ::2] *= t
    x[:, 1::2] *= q

    out, _ = model(_field(x, coords)).e2s.to_torch()

    # Reference RH using pure-water saturation vapor pressure (alpha == 1.0)
    epsilon = 0.621981
    p = float(levels[0]) * 100.0  # hPa -> Pa
    e = (p * q * (1.0 / epsilon)) / (1 + q * (1.0 / epsilon - 1))
    es_w = 611.21 * np.exp(17.502 * (t - 273.16) / (t - 32.19))
    expected = np.clip(100 * e / es_w, 0, 100)

    assert torch.allclose(out, torch.full_like(out, float(expected)), atol=1e-3)


@pytest.mark.parametrize(
    "invalid_coords",
    [
        OrderedDict({"batch": np.array([0]), "variable": np.array(["wrong_var"])}),
        OrderedDict(
            {"batch": np.array([0]), "variable": np.array(["t100"])}
        ),  # Missing q component
    ],
)
def test_derived_rh_invalid_coords(invalid_coords):
    """Test relative humidity derivation with invalid coordinates"""
    model = DerivedRH([100], grid=_grid((16, 32)))
    x = torch.randn(1, 1, 16, 32)
    invalid_coords.update(
        lat=model.input_coords()["lat"], lon=model.input_coords()["lon"]
    )
    field = _field(x, invalid_coords)

    with pytest.raises(ValueError):
        model(field)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 2, 16, 32),
        (2, 2, 32, 64),
        (4, 2, 48, 96),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_derived_rh_dewpoint(shape, device):
    model = DerivedRHDewpoint(grid=_grid(shape)).to(device)

    batch_size = shape[0]
    lat_size = shape[2]
    lon_size = shape[3]

    input_coords = model.input_coords()
    coords = OrderedDict(
        {
            "batch": np.arange(batch_size),
            "variable": input_coords["variable"],
            "lat": np.linspace(-90, 90, lat_size),
            "lon": np.linspace(0, 360, lon_size),
        }
    )

    # Test with realistic temperature and dewpoint values
    t = 273.15 + 20
    d = 273.15 + 15
    x = torch.ones(shape).to(device)
    x[:, 0::2] *= t
    x[:, 1::2] *= d

    out, out_coords = model(_field(x, coords)).e2s.to_torch()

    # Check output shape and coordinates
    assert out.shape == (batch_size, 1, lat_size, lon_size)
    assert "variable" in out_coords
    assert len(out_coords["variable"]) == 1
    assert "r2m" in out_coords["variable"]

    # Test device consistency
    assert out.device == x.device

    # Test physical bounds and behavior
    # At 20°C and 15°C dewpoint, RH should be around 73%
    # https://bmcnoldy.earth.miami.edu/Humidity.ht
    expected_rh = 73.0
    assert torch.allclose(
        torch.mean(out), torch.tensor(expected_rh, device=device), rtol=0.1
    )

    # Test with saturated conditions (T = Td)
    x_sat = x.clone()
    x_sat[:, 1::2] = x_sat[:, ::2]  # Dewpoint equals air temperature
    out_sat, _ = model(_field(x_sat, coords)).e2s.to_torch()
    assert torch.all(out_sat > 99)

    # Test with very dry conditions
    x_dry = x.clone()
    x_dry[:, ::2] = 273.15 + 30
    x_dry[:, 1::2] = 273.15 - 10
    out_dry, _ = model(_field(x_dry, coords)).e2s.to_torch()
    assert torch.all(out_dry < 20)

    # Test with cold conditions (below freezing)
    x_cold = x.clone()
    x_cold[:, ::2] = 273.15 - 10
    x_cold[:, 1::2] = 273.15 - 12
    out_cold, _ = model(_field(x_cold, coords)).e2s.to_torch()
    # Should still give reasonable RH values and use ice formula
    assert torch.all(out_cold >= 0)
    assert torch.all(out_cold <= 100)
    assert torch.allclose(
        torch.mean(out_cold), torch.tensor(85.0, device=device), rtol=0.1
    )


@pytest.mark.parametrize(
    "invalid_coords",
    [
        OrderedDict({"batch": np.array([0]), "variable": np.array(["wrong_var"])}),
        OrderedDict(
            {"batch": np.array([0]), "variable": np.array(["t2m"])}
        ),  # Missing d2m
        OrderedDict(
            {"batch": np.array([0]), "variable": np.array(["d2m"])}
        ),  # Missing t2m
    ],
)
def test_derived_rh_dewpoint_invalid_coords(invalid_coords):
    """Test RH from dewpoint derivation with invalid coordinates"""
    model = DerivedRHDewpoint(grid=_grid((16, 32)))
    x = torch.randn(1, 1, 16, 32)
    invalid_coords.update(
        lat=model.input_coords()["lat"], lon=model.input_coords()["lon"]
    )
    field = _field(x, invalid_coords)

    with pytest.raises(ValueError):
        model(field)


@pytest.mark.parametrize(
    "levels,shape",
    [
        ([100], (1, 2, 16, 32)),
        ([500, 850], (1, 4, 32, 64)),
        ([100, 200, 300], (1, 6, 32, 64)),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_derived_vpd(levels, shape, device):
    model = DerivedVPD(levels, grid=_grid(shape)).to(device)

    batch_size = shape[0]
    n_levels = len(levels)
    lat_size = shape[2]
    lon_size = shape[3]

    input_coords = model.input_coords()
    coords = OrderedDict(
        {
            "batch": np.arange(batch_size),
            "variable": input_coords["variable"],
            "lat": np.linspace(-90, 90, lat_size),
            "lon": np.linspace(0, 360, lon_size),
        }
    )

    # Test with realistic temperature and RH values
    t = 273.15 + 25  # 25°C
    rh = 60  # 60% relative humidity
    x = torch.ones(shape).to(device)
    x[:, ::2] *= t  # Temperature channels
    x[:, 1::2] *= rh  # RH channels

    out, out_coords = model(_field(x, coords)).e2s.to_torch()

    # Check output shape and coordinates
    assert out.shape == (batch_size, n_levels, lat_size, lon_size)
    assert "variable" in out_coords
    assert len(out_coords["variable"]) == len(levels)
    assert all(f"vpd{level}" in out_coords["variable"] for level in levels)

    # Test device consistency
    assert out.device == x.device

    # Test physical bounds and behavior
    # At 25°C and 60% RH, VPD should be around 1.2-1.3 kPa or 12-13 mb/hPa
    # https://www.dimluxlighting.com/knowledge/vapor-pressure-deficit-vpd-calculator/
    # https://www.omnicalculator.com/biology/vapor-pressure-deficit#what-is-vapor-pressure-deficit-vpd
    # https://en.wikipedia.org/wiki/Vapour-pressure_deficit
    expected_vpd = 12.5
    assert torch.allclose(
        torch.mean(out), torch.tensor(expected_vpd, device=device), rtol=0.2
    )

    # Test with saturated conditions (RH = 100%)
    x_sat = x.clone()
    x_sat[:, 1::2] = 100
    out_sat, _ = model(_field(x_sat, coords)).e2s.to_torch()
    assert torch.allclose(out_sat, torch.zeros_like(out_sat), atol=1e-5)

    # Test with hot and dry conditions (high VPD)
    x_hot_dry = x.clone()
    x_hot_dry[:, ::2] = 273.15 + 35  # 35°C
    x_hot_dry[:, 1::2] = 20  # 20% RH
    out_hot_dry, _ = model(_field(x_hot_dry, coords)).e2s.to_torch()
    assert torch.all(out_hot_dry > out)

    # Test with cold and humid conditions (low VPD)
    x_cold_humid = x.clone()
    x_cold_humid[:, ::2] = 273.15 + 10  # 10°C
    x_cold_humid[:, 1::2] = 90  # 90% RH
    out_cold_humid, _ = model(_field(x_cold_humid, coords)).e2s.to_torch()
    assert torch.all(out_cold_humid < out)


@pytest.mark.parametrize(
    "invalid_coords",
    [
        OrderedDict({"batch": np.array([0]), "variable": np.array(["wrong_var"])}),
        OrderedDict(
            {"batch": np.array([0]), "variable": np.array(["t100"])}
        ),  # Missing rh component
    ],
)
def test_derived_vpd_invalid_coords(invalid_coords):
    """Test VPD derivation with invalid coordinates"""
    model = DerivedVPD([100], grid=_grid((16, 32)))
    x = torch.randn(1, 1, 16, 32)
    invalid_coords.update(
        lat=model.input_coords()["lat"], lon=model.input_coords()["lon"]
    )
    field = _field(x, invalid_coords)

    with pytest.raises(ValueError):
        model(field)


@pytest.mark.parametrize(
    "z_surf_constant,temperature_correction,sp_correct",
    [
        (0.0, True, 1000e2),
        (1000.0, True, 900e2),
        (500.0, False, np.exp(0.5 * (np.log(1000e2) + np.log(900e2)))),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_derived_surface_pressure(
    device: str, z_surf_constant: float, temperature_correction: bool, sp_correct: float
) -> None:
    shape = (8, 16)
    z_surface = torch.full(shape, z_surf_constant, device=device)
    z_surf_coords = OrderedDict(
        lat=np.linspace(40, 50, shape[0]), lon=np.linspace(50, 70, shape[1])
    )

    sp_model = DerivedSurfacePressure(
        p_levels=[900, 1000],
        surface_geopotential=z_surface,
        surface_geopotential_coords=z_surf_coords,
        temperature_correction=temperature_correction,
        corr_adjustment=(0.0, 1.0),  # needed to verify theoretical results
    )
    sp_model.to(device)

    z_levels = torch.empty((2, 2, *shape), device=device)
    z_levels[:, 0, :, :] = 1000
    z_levels[:, 1, :, :] = 0
    t_levels = torch.full_like(z_levels, 288.15)

    x_in = torch.concat([z_levels, t_levels], dim=1)
    coords_in = OrderedDict(
        batch=np.array([0, 1]),
        variable=np.array(["z900", "z1000", "t900", "t1000"]),
        lat=z_surf_coords["lat"],
        lon=z_surf_coords["lon"],
    )

    field = _field(x_in, coords_in).sel(variable=sp_model.input_coords()["variable"])
    x_out, coords_out = sp_model(field).e2s.to_torch()

    # check shapes
    assert x_in.ndim == x_out.ndim
    for i, dim in enumerate(coords_out):
        if dim == "variable":
            continue
        assert x_in.shape[i] == x_out.shape[i]
        assert (coords_in[dim] == coords_out[dim]).all()
    assert (coords_out["variable"] == np.array(["sp"])).all()
    variable_dim = list(coords_out).index("variable")
    assert x_out.shape[variable_dim] == 1

    # check that we get analytic solution
    assert (x_out - sp_correct).abs().max() / sp_correct < 1e-4


@pytest.mark.parametrize(
    "levels,shape",
    [
        ([1000, 850, 500], (1, 4, 16, 32)),
        ([1000, 850, 700, 500, 300], (2, 6, 32, 64)),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_derived_tcwv(levels, shape, device):
    model = DerivedTCWV(levels, grid=_grid(shape)).to(device)

    batch_size = shape[0]
    n_levels = len(levels)
    lat_size = shape[2]
    lon_size = shape[3]

    input_coords = model.input_coords()
    assert len(input_coords["variable"]) == n_levels + 1  # q levels + sp

    coords = OrderedDict(
        {
            "batch": np.arange(batch_size),
            "variable": input_coords["variable"],
            "lat": np.linspace(-90, 90, lat_size),
            "lon": np.linspace(0, 360, lon_size),
        }
    )
    x = torch.ones((batch_size, n_levels + 1, lat_size, lon_size)).to(device)
    x[:, :n_levels] *= 0.01  # Specific humidity ~0.01 kg/kg
    x[:, n_levels] *= 101325  # Surface pressure ~101325 Pa

    out, out_coords = model(_field(x, coords)).e2s.to_torch()

    assert out.shape == (batch_size, 1, lat_size, lon_size)
    assert out_coords["variable"][0] == "tcwv"
    assert out.device == x.device
    assert torch.all(out >= 0)  # TCWV should be non-negative

    # Test with zero specific humidity -> TCWV should be zero
    x_zero = torch.zeros_like(x)
    x_zero[:, n_levels] = 101325  # Keep surface pressure
    out_zero, _ = model(_field(x_zero, coords)).e2s.to_torch()
    assert torch.allclose(out_zero, torch.zeros_like(out_zero), atol=1e-6)

    # Test numerical accuracy with constant q and known pressure range
    # For constant q, TCWV = q * (p_surface - p_top) / g
    q_const = 0.01  # kg/kg
    p_surface = 101325.0  # Pa
    p_top = min(levels) * 100.0  # Top pressure level in Pa
    x_const = torch.full((batch_size, n_levels + 1, lat_size, lon_size), q_const).to(
        device
    )
    x_const[:, n_levels] = p_surface
    out_const, _ = model(_field(x_const, coords)).e2s.to_torch()
    expected_tcwv = q_const * (p_surface - p_top) / model.g
    assert torch.allclose(
        out_const, torch.full_like(out_const, expected_tcwv), rtol=0.01
    )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_derived_tcwv_partial_layer_uses_surface_pressure(device):
    levels = [1000, 850]
    model = DerivedTCWV(levels, grid=_grid(endpoint=False)).to(device)

    # Small grid and batch
    batch_size = 1
    lat_size = 2
    lon_size = 3
    n_levels = len(levels)

    input_coords = model.input_coords()
    coords = OrderedDict(
        {
            "batch": np.arange(batch_size),
            "variable": input_coords["variable"],
            "lat": np.linspace(-90, 90, lat_size),
            "lon": np.linspace(0, 360, lon_size, endpoint=False),
        }
    )

    # Specific humidity per level and surface pressure
    q1000 = 0.1
    q850 = 0.2
    # Surface pressure: half below 1000 hPa (95000 Pa), half slightly above (100001 Pa)
    x = torch.zeros(
        (batch_size, n_levels + 1, lat_size, lon_size), dtype=torch.float32
    ).to(device)
    x[:, 0, :, :] = q1000
    x[:, 1, :, :] = q850
    sp = torch.full((lat_size, lon_size), 95000.0, device=device)
    sp[:, lon_size // 2 :] = 100001.0
    x[:, 2, :, :] = sp

    out, out_coords = model(_field(x, coords)).e2s.to_torch()

    # Expected:
    # - For sp=95000 Pa: integrate from 95000 to 85000 with q850
    expected_low = q850 * (95000.0 - 85000.0) / model.g
    # - For sp=100001 Pa: small sliver from 100001->100000 with q1000
    #   plus trapezoid from 100000->85000 with (q1000+q850)/2
    expected_high = (q1000 * (100001.0 - 100000.0) / model.g) + (
        0.5 * (q1000 + q850) * (100000.0 - 85000.0) / model.g
    )

    expected = torch.full(
        (batch_size, 1, lat_size, lon_size), expected_low, device=device
    )
    expected[:, :, :, lon_size // 2 :] = expected_high

    assert out.shape == (batch_size, 1, lat_size, lon_size)
    assert out_coords["variable"][0] == "tcwv"
    assert torch.allclose(out, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "invalid_coords",
    [
        OrderedDict({"batch": np.array([0]), "variable": np.array(["wrong_var"])}),
        OrderedDict(
            {"batch": np.array([0]), "variable": np.array(["q1000"])}
        ),  # Missing sp
    ],
)
def test_derived_tcwv_invalid_coords(invalid_coords):
    """Test TCWV derivation with invalid coordinates"""
    model = DerivedTCWV([1000, 850, 500], grid=_grid((16, 32)))
    x = torch.randn(1, 1, 16, 32)
    invalid_coords.update(
        lat=model.input_coords()["lat"], lon=model.input_coords()["lon"]
    )
    field = _field(x, invalid_coords)

    with pytest.raises(ValueError):
        model(field)


def test_derivedws_conformance():
    model = DerivedWS([100], grid=_grid())
    assert check_diagnostic_contract(model) == [
        "D10: model does not declare itself stochastic"
    ]


def test_derivedrh_conformance():
    # Native equality treats identical NaN locations as deterministic. Numerical
    # behavior at unphysical temperatures is covered separately below.
    model = DerivedRH([100], grid=_grid())
    assert check_diagnostic_contract(model) == [
        "D10: model does not declare itself stochastic"
    ]


def test_derivedrhdewpoint_conformance():
    model = DerivedRHDewpoint(grid=_grid())
    assert check_diagnostic_contract(model) == [
        "D10: model does not declare itself stochastic"
    ]


def test_derivedvpd_conformance():
    model = DerivedVPD([100], grid=_grid())
    assert check_diagnostic_contract(model) == [
        "D10: model does not declare itself stochastic"
    ]


def test_derivedsurfacepressure_conformance():
    shape = (8, 16)
    z_surface = torch.zeros(shape)
    z_surf_coords = OrderedDict(
        lat=np.linspace(40, 50, shape[0]), lon=np.linspace(50, 70, shape[1])
    )
    model = DerivedSurfacePressure(
        p_levels=[900, 1000],
        surface_geopotential=z_surface,
        surface_geopotential_coords=z_surf_coords,
    )
    assert check_diagnostic_contract(model) == [
        "D10: model does not declare itself stochastic"
    ]


def test_derivedtcwv_conformance():
    model = DerivedTCWV([1000, 850, 500], grid=_grid())
    assert check_diagnostic_contract(model) == [
        "D10: model does not declare itself stochastic"
    ]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_derived_rh_unphysical_temperature_nan_is_repeatable(device):
    model = DerivedRH([850], grid=_grid()).to(device)
    signature = coord_array_like(model.input_coords(), {"batch": [0]})
    tensor = torch.full(signature.shape, 0.004, device=device)
    tensor[:, 0] = torch.tensor([32.18, 32.20, 290.0], device=device)
    field = from_torch(tensor, signature)
    first = model(field)
    result, _ = first.e2s.to_torch()
    assert torch.isnan(result[..., 0]).all()
    assert torch.isfinite(result[..., 1:]).all()
    xr.testing.assert_identical(first.e2s.as_numpy(), model(field).e2s.as_numpy())


def test_derived_qualified_variable_statistics():
    model = DerivedWS(["10m:mean:6h"], grid=_grid())
    signature = coord_array_like(model.input_coords(), {"batch": [0]})
    assert set(signature.attrs["earth2studio_statistics"]) == {
        "u10m:mean:6h",
        "v10m:mean:6h",
    }
    field = from_torch(torch.ones(signature.shape), signature)
    output = model(field)
    assert output.coords["variable"].values.tolist() == ["ws10m:mean:6h"]
    assert set(output.attrs["earth2studio_statistics"]) == {"ws10m:mean:6h"}
    assert (
        output.attrs["earth2studio_statistics"]
        == model.output_coords(field).attrs["earth2studio_statistics"]
    )
    wrong = field.copy(deep=True)
    wrong.attrs.pop("earth2studio_statistics")
    with pytest.raises(ValueError, match="statistics"):
        model(wrong)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("half_model", [False, True])
def test_derived_rh_float16_pressure_precision(device, half_model):
    model = DerivedRH([850], grid=_grid()).to(device)
    if half_model:
        model.half()
    signature = coord_array_like(model.input_coords(), {"batch": [0]})
    tensor = torch.full(signature.shape, 0.004, dtype=torch.float16, device=device)
    tensor[:, 0] = 290.0
    field = from_torch(tensor, signature)
    result, _ = model(field).e2s.to_torch()

    # Evaluate the physical reference in double precision using the actual
    # quantized humidity, without narrowing 85000 Pa to float16 (infinity).
    q = tensor[0, 1, 0, 0].item()
    epsilon = 0.621981
    e = 85000.0 * q / epsilon / (1 + q * (1 / epsilon - 1))
    es = 611.21 * np.exp(17.502 * (290.0 - 273.16) / (290.0 - 32.19))
    expected = 100 * e / es
    assert 28 < expected < 29
    assert result.dtype == tensor.dtype
    assert result.device == tensor.device
    torch.testing.assert_close(
        result, torch.full_like(result, expected), rtol=1e-3, atol=0.02
    )


@pytest.mark.parametrize("shift_indexes", [False, True])
def test_derived_registered_curvilinear_exact_coordinates(monkeypatch, shift_indexes):
    latitude = np.array([[40.0, 40.1], [41.0, 41.1]])
    longitude = np.array([[10.0, 11.0], [10.1, 11.1]])
    registered = CurvilinearGrid(latitude, longitude)
    name = "derived-test-curvilinear"
    monkeypatch.setitem(grids._GRID_REGISTRY, name, registered)
    configured = CurvilinearGrid(
        latitude,
        longitude,
        y=np.array([10, 11]) if shift_indexes else np.arange(2),
        x=np.array([20, 21]) if shift_indexes else np.arange(2),
    )
    # The fingerprint intentionally cannot distinguish these index layouts.
    assert configured.fingerprint() == registered.fingerprint()
    model = DerivedWS([100], grid=configured)
    declared = model.input_coords()
    for coord in configured.coords():
        xr.testing.assert_identical(declared.coords[coord], configured.coords()[coord])
    assert declared.attrs.get("earth2studio_grid_id") == (
        None if shift_indexes else name
    )
    signature = coord_array(
        ("variable", "y", "x"),
        {"variable": ["u100", "v100"]},
        grid=configured if shift_indexes else name,
    )
    out = model(from_torch(torch.ones(signature.shape), signature))
    for coord in configured.coords():
        xr.testing.assert_identical(out.coords[coord], configured.coords()[coord])
    np.testing.assert_allclose(out.values, np.sqrt(2), rtol=1e-6)


def test_derived_explicit_grid_name_skips_registry_scan(monkeypatch):
    class NoScanRegistry(dict):
        def __iter__(self):
            pytest.fail("An explicit grid name must not scan the registry")

    monkeypatch.setattr(grids, "_GRID_REGISTRY", NoScanRegistry(grids._GRID_REGISTRY))
    name = "latlon-0.25deg"
    model = DerivedWS(grid=name)
    assert model.grid == name
    assert model.input_coords().attrs["earth2studio_grid_id"] == name
