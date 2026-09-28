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

import io

import numpy as np
import pytest
import torch

import earth2studio.models.dx.orbit2_precip as orbit_module
from earth2studio.models.conformance import check_diagnostic_contract
from earth2studio.models.dx import OrbitGlobalPrecip
from earth2studio.utils.coords import coord_array_like
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.imports import OptionalDependencyFailure


@pytest.fixture(autouse=True)
def offline_backend(monkeypatch):
    if orbit_module.LogTransform is not None:
        return
    monkeypatch.delitem(
        OptionalDependencyFailure.failures, orbit_module.__file__, raising=False
    )
    monkeypatch.setattr(orbit_module, "PRECIP_VARIABLES", ["total_precipitation_24hr"])

    class LogTransform:
        def __init__(self, **kwargs):
            pass

        def __call__(self, x):
            return torch.log1p(torch.clamp(x * 1000, min=0))

    monkeypatch.setattr(orbit_module, "LogTransform", LogTransform)
    # Exercise preprocessing, the mock network, and physical-unit postprocessing
    # without the optional backend's tile scheduler.
    original = OrbitGlobalPrecip.__init__

    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.do_tiling = False
        self.model.lat_out = 2880
        self.model.lon_out = 5760

    monkeypatch.setattr(OrbitGlobalPrecip, "__init__", initialize)


class PhooORBIT2Precip(torch.nn.Module):
    def __init__(
        self,
        lat,
        lon,
        div,
        overlap,
    ):
        super().__init__()
        self.lat_out = int((lat / div + overlap) * 4)
        self.lon_out = int((lon / div + overlap * 2) * 4)

    def forward(self, x, in_variables, out_variables):
        x_out = torch.zeros(
            x.shape[0], len(out_variables), self.lat_out, self.lon_out, device=x.device
        )
        return x_out


@pytest.mark.parametrize(
    "x",
    [
        torch.randn(1, 20, 721, 1440),
        torch.randn(2, 20, 721, 1440),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_orbit2_precip(x, device):
    x = x.to(device)
    land_sea_mask = np.zeros((720, 1440))
    orography = np.zeros((720, 1440))
    lattitude = np.zeros((720, 1440))
    landcover = np.zeros((720, 1440))
    normalize_mean_lowres = {
        "land_sea_mask": np.array([0.33852854], dtype=np.float32),
        "orography": np.array([386.0143], dtype=np.float32),
        "landcover": np.array([13.248313], dtype=np.float32),
        "lattitude": np.array([-0.5]),
        "2m_temperature": np.array([280.4026], dtype=np.float32),
        "temperature_200": np.array([218.02412], dtype=np.float32),
        "temperature_500": np.array([252.92754], dtype=np.float32),
        "temperature_850": np.array([274.50055], dtype=np.float32),
        "10m_u_component_of_wind": np.array([-0.04986294], dtype=np.float32),
        "u_component_of_wind_200": np.array([14.208759], dtype=np.float32),
        "u_component_of_wind_500": np.array([6.563433], dtype=np.float32),
        "u_component_of_wind_850": np.array([1.4184277], dtype=np.float32),
        "10m_v_component_of_wind": np.array([0.18893285], dtype=np.float32),
        "v_component_of_wind_200": np.array([-0.0450691], dtype=np.float32),
        "v_component_of_wind_500": np.array([-0.02393754], dtype=np.float32),
        "v_component_of_wind_850": np.array([0.14214961], dtype=np.float32),
        "specific_humidity_200": np.array([1.9402956e-05], dtype=np.float32),
        "specific_humidity_500": np.array([0.00085243], dtype=np.float32),
        "specific_humidity_850": np.array([0.0045715], dtype=np.float32),
        "volumetric_soil_water_layer_1": np.array([0.08639744], dtype=np.float32),
        "total_precipitation_24hr": np.array([0.00239384], dtype=np.float32),
        "2m_temperature_max": np.array([281.84592], dtype=np.float32),
        "2m_temperature_min": np.array([279.05597], dtype=np.float32),
    }
    normalize_std_lowres = {
        "land_sea_mask": np.array([0.46153313], dtype=np.float32),
        "orography": np.array([864.79724], dtype=np.float32),
        "landcover": np.array([3.6126225], dtype=np.float32),
        "lattitude": np.array([51.96072235]),
        "2m_temperature": np.array([20.601086], dtype=np.float32),
        "temperature_200": np.array([7.216472], dtype=np.float32),
        "temperature_500": np.array([13.08831], dtype=np.float32),
        "temperature_850": np.array([15.682292], dtype=np.float32),
        "10m_u_component_of_wind": np.array([5.542934], dtype=np.float32),
        "u_component_of_wind_200": np.array([17.66932], dtype=np.float32),
        "u_component_of_wind_500": np.array([11.970395], dtype=np.float32),
        "u_component_of_wind_850": np.array([8.181024], dtype=np.float32),
        "10m_v_component_of_wind": np.array([4.7573], dtype=np.float32),
        "v_component_of_wind_200": np.array([11.869816], dtype=np.float32),
        "v_component_of_wind_500": np.array([9.163726], dtype=np.float32),
        "v_component_of_wind_850": np.array([6.2524633], dtype=np.float32),
        "specific_humidity_200": np.array([2.2676879e-05], dtype=np.float32),
        "specific_humidity_500": np.array([0.0010795], dtype=np.float32),
        "specific_humidity_850": np.array([0.00411547], dtype=np.float32),
        "volumetric_soil_water_layer_1": np.array([0.14186133], dtype=np.float32),
        "total_precipitation_24hr": np.array([0.00580253], dtype=np.float32),
        "2m_temperature_max": np.array([20.402489], dtype=np.float32),
        "2m_temperature_min": np.array([20.929356], dtype=np.float32),
    }
    normalize_mean_highres = normalize_mean_lowres
    normalize_std_highres = normalize_std_lowres

    buf_normalize_mean_lowres = io.BytesIO()
    np.savez(buf_normalize_mean_lowres, **normalize_mean_lowres)
    buf_normalize_mean_lowres.seek(0)
    normalize_mean_lowres_npz = np.load(buf_normalize_mean_lowres)

    buf_normalize_std_lowres = io.BytesIO()
    np.savez(buf_normalize_std_lowres, **normalize_std_lowres)
    buf_normalize_std_lowres.seek(0)
    normalize_std_lowres_npz = np.load(buf_normalize_std_lowres)

    buf_normalize_mean_highres = io.BytesIO()
    np.savez(buf_normalize_mean_highres, **normalize_mean_highres)
    buf_normalize_mean_highres.seek(0)
    normalize_mean_highres_npz = np.load(buf_normalize_mean_highres)

    buf_normalize_std_highres = io.BytesIO()
    np.savez(buf_normalize_std_highres, **normalize_std_highres)
    buf_normalize_std_highres.seek(0)
    normalize_std_highres_npz = np.load(buf_normalize_std_highres)

    do_tiling = True
    div = 4
    overlap = 4
    model = PhooORBIT2Precip(x.shape[-2] - 1, x.shape[-1], div, overlap)

    dx = OrbitGlobalPrecip(
        model,
        land_sea_mask,
        orography,
        lattitude,
        landcover,
        normalize_mean_lowres_npz,
        normalize_std_lowres_npz,
        normalize_mean_highres_npz,
        normalize_std_highres_npz,
        do_tiling,
        div,
        overlap,
    ).to(device)

    coords = coord_array_like(dx.input_coords(), {"batch": np.arange(x.shape[0])})
    assert coords.data.nbytes == 0
    out = dx(from_torch(x, coords))
    out_coords = out.coords

    assert out.shape == torch.Size([x.shape[0], 1, 2880, 5760])
    assert out_coords["variable"] == dx.output_coords(coords)["variable"]
    assert out.dims == ("batch", "variable", "lat", "lon")


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("model_size", ["9.5m", "126m"])
def test_orbit2_precip_package(device, model_size):
    package = OrbitGlobalPrecip.load_default_package()
    dx = OrbitGlobalPrecip.load_model(
        package, "global", model_size, "precipitation"
    ).to(device)
    x = torch.randn(1, 20, 721, 1440).to(device)
    coords = coord_array_like(dx.input_coords(), {"batch": np.arange(x.shape[0])})
    out = dx(from_torch(x, coords))
    out_coords = out.coords
    assert out.shape == torch.Size([x.shape[0], 1, 2880, 5760])
    assert out_coords["variable"] == dx.output_coords(coords)["variable"]
    assert out.dims == ("batch", "variable", "lat", "lon")


def test_orbitglobalprecip_conformance():
    land_sea_mask = np.zeros((720, 1440))
    orography = np.zeros((720, 1440))
    lattitude = np.zeros((720, 1440))
    landcover = np.zeros((720, 1440))
    normalize_mean_lowres = {
        "land_sea_mask": np.array([0.33852854], dtype=np.float32),
        "orography": np.array([386.0143], dtype=np.float32),
        "landcover": np.array([13.248313], dtype=np.float32),
        "lattitude": np.array([-0.5]),
        "2m_temperature": np.array([280.4026], dtype=np.float32),
        "temperature_200": np.array([218.02412], dtype=np.float32),
        "temperature_500": np.array([252.92754], dtype=np.float32),
        "temperature_850": np.array([274.50055], dtype=np.float32),
        "10m_u_component_of_wind": np.array([-0.04986294], dtype=np.float32),
        "u_component_of_wind_200": np.array([14.208759], dtype=np.float32),
        "u_component_of_wind_500": np.array([6.563433], dtype=np.float32),
        "u_component_of_wind_850": np.array([1.4184277], dtype=np.float32),
        "10m_v_component_of_wind": np.array([0.18893285], dtype=np.float32),
        "v_component_of_wind_200": np.array([-0.0450691], dtype=np.float32),
        "v_component_of_wind_500": np.array([-0.02393754], dtype=np.float32),
        "v_component_of_wind_850": np.array([0.14214961], dtype=np.float32),
        "specific_humidity_200": np.array([1.9402956e-05], dtype=np.float32),
        "specific_humidity_500": np.array([0.00085243], dtype=np.float32),
        "specific_humidity_850": np.array([0.0045715], dtype=np.float32),
        "volumetric_soil_water_layer_1": np.array([0.08639744], dtype=np.float32),
        "total_precipitation_24hr": np.array([0.00239384], dtype=np.float32),
        "2m_temperature_max": np.array([281.84592], dtype=np.float32),
        "2m_temperature_min": np.array([279.05597], dtype=np.float32),
    }
    normalize_std_lowres = {
        "land_sea_mask": np.array([0.46153313], dtype=np.float32),
        "orography": np.array([864.79724], dtype=np.float32),
        "landcover": np.array([3.6126225], dtype=np.float32),
        "lattitude": np.array([51.96072235]),
        "2m_temperature": np.array([20.601086], dtype=np.float32),
        "temperature_200": np.array([7.216472], dtype=np.float32),
        "temperature_500": np.array([13.08831], dtype=np.float32),
        "temperature_850": np.array([15.682292], dtype=np.float32),
        "10m_u_component_of_wind": np.array([5.542934], dtype=np.float32),
        "u_component_of_wind_200": np.array([17.66932], dtype=np.float32),
        "u_component_of_wind_500": np.array([11.970395], dtype=np.float32),
        "u_component_of_wind_850": np.array([8.181024], dtype=np.float32),
        "10m_v_component_of_wind": np.array([4.7573], dtype=np.float32),
        "v_component_of_wind_200": np.array([11.869816], dtype=np.float32),
        "v_component_of_wind_500": np.array([9.163726], dtype=np.float32),
        "v_component_of_wind_850": np.array([6.2524633], dtype=np.float32),
        "specific_humidity_200": np.array([2.2676879e-05], dtype=np.float32),
        "specific_humidity_500": np.array([0.0010795], dtype=np.float32),
        "specific_humidity_850": np.array([0.00411547], dtype=np.float32),
        "volumetric_soil_water_layer_1": np.array([0.14186133], dtype=np.float32),
        "total_precipitation_24hr": np.array([0.00580253], dtype=np.float32),
        "2m_temperature_max": np.array([20.402489], dtype=np.float32),
        "2m_temperature_min": np.array([20.929356], dtype=np.float32),
    }
    normalize_mean_highres = normalize_mean_lowres
    normalize_std_highres = normalize_std_lowres

    buf_normalize_mean_lowres = io.BytesIO()
    np.savez(buf_normalize_mean_lowres, **normalize_mean_lowres)
    buf_normalize_mean_lowres.seek(0)
    normalize_mean_lowres_npz = np.load(buf_normalize_mean_lowres)

    buf_normalize_std_lowres = io.BytesIO()
    np.savez(buf_normalize_std_lowres, **normalize_std_lowres)
    buf_normalize_std_lowres.seek(0)
    normalize_std_lowres_npz = np.load(buf_normalize_std_lowres)

    buf_normalize_mean_highres = io.BytesIO()
    np.savez(buf_normalize_mean_highres, **normalize_mean_highres)
    buf_normalize_mean_highres.seek(0)
    normalize_mean_highres_npz = np.load(buf_normalize_mean_highres)

    buf_normalize_std_highres = io.BytesIO()
    np.savez(buf_normalize_std_highres, **normalize_std_highres)
    buf_normalize_std_highres.seek(0)
    normalize_std_highres_npz = np.load(buf_normalize_std_highres)

    do_tiling = True
    div = 4
    overlap = 4
    model = PhooORBIT2Precip(720, 1440, div, overlap)

    dx = OrbitGlobalPrecip(
        model,
        land_sea_mask,
        orography,
        lattitude,
        landcover,
        normalize_mean_lowres_npz,
        normalize_std_lowres_npz,
        normalize_mean_highres_npz,
        normalize_std_highres_npz,
        do_tiling,
        div,
        overlap,
    )

    assert check_diagnostic_contract(dx) == [
        "D10: model does not declare itself stochastic"
    ]


def test_orbit2_precip_exceptions():
    x = torch.randn(1, 20, 720, 1440)
    land_sea_mask = np.zeros((720, 1440))
    orography = np.zeros((720, 1440))
    lattitude = np.zeros((720, 1440))
    landcover = np.zeros((720, 1440))
    normalize_mean_lowres = {
        "land_sea_mask": np.array([0.33852854], dtype=np.float32),
        "orography": np.array([386.0143], dtype=np.float32),
        "landcover": np.array([13.248313], dtype=np.float32),
        "lattitude": np.array([-0.5]),
        "2m_temperature": np.array([280.4026], dtype=np.float32),
        "temperature_200": np.array([218.02412], dtype=np.float32),
        "temperature_500": np.array([252.92754], dtype=np.float32),
        "temperature_850": np.array([274.50055], dtype=np.float32),
        "10m_u_component_of_wind": np.array([-0.04986294], dtype=np.float32),
        "u_component_of_wind_200": np.array([14.208759], dtype=np.float32),
        "u_component_of_wind_500": np.array([6.563433], dtype=np.float32),
        "u_component_of_wind_850": np.array([1.4184277], dtype=np.float32),
        "10m_v_component_of_wind": np.array([0.18893285], dtype=np.float32),
        "v_component_of_wind_200": np.array([-0.0450691], dtype=np.float32),
        "v_component_of_wind_500": np.array([-0.02393754], dtype=np.float32),
        "v_component_of_wind_850": np.array([0.14214961], dtype=np.float32),
        "specific_humidity_200": np.array([1.9402956e-05], dtype=np.float32),
        "specific_humidity_500": np.array([0.00085243], dtype=np.float32),
        "specific_humidity_850": np.array([0.0045715], dtype=np.float32),
        "volumetric_soil_water_layer_1": np.array([0.08639744], dtype=np.float32),
        "total_precipitation_24hr": np.array([0.00239384], dtype=np.float32),
        "2m_temperature_max": np.array([281.84592], dtype=np.float32),
        "2m_temperature_min": np.array([279.05597], dtype=np.float32),
    }
    normalize_std_lowres = {
        "land_sea_mask": np.array([0.46153313], dtype=np.float32),
        "orography": np.array([864.79724], dtype=np.float32),
        "landcover": np.array([3.6126225], dtype=np.float32),
        "lattitude": np.array([51.96072235]),
        "2m_temperature": np.array([20.601086], dtype=np.float32),
        "temperature_200": np.array([7.216472], dtype=np.float32),
        "temperature_500": np.array([13.08831], dtype=np.float32),
        "temperature_850": np.array([15.682292], dtype=np.float32),
        "10m_u_component_of_wind": np.array([5.542934], dtype=np.float32),
        "u_component_of_wind_200": np.array([17.66932], dtype=np.float32),
        "u_component_of_wind_500": np.array([11.970395], dtype=np.float32),
        "u_component_of_wind_850": np.array([8.181024], dtype=np.float32),
        "10m_v_component_of_wind": np.array([4.7573], dtype=np.float32),
        "v_component_of_wind_200": np.array([11.869816], dtype=np.float32),
        "v_component_of_wind_500": np.array([9.163726], dtype=np.float32),
        "v_component_of_wind_850": np.array([6.2524633], dtype=np.float32),
        "specific_humidity_200": np.array([2.2676879e-05], dtype=np.float32),
        "specific_humidity_500": np.array([0.0010795], dtype=np.float32),
        "specific_humidity_850": np.array([0.00411547], dtype=np.float32),
        "volumetric_soil_water_layer_1": np.array([0.14186133], dtype=np.float32),
        "total_precipitation_24hr": np.array([0.00580253], dtype=np.float32),
        "2m_temperature_max": np.array([20.402489], dtype=np.float32),
        "2m_temperature_min": np.array([20.929356], dtype=np.float32),
    }
    normalize_mean_highres = normalize_mean_lowres
    normalize_std_highres = normalize_std_lowres

    buf_normalize_mean_lowres = io.BytesIO()
    np.savez(buf_normalize_mean_lowres, **normalize_mean_lowres)
    buf_normalize_mean_lowres.seek(0)
    normalize_mean_lowres_npz = np.load(buf_normalize_mean_lowres)

    buf_normalize_std_lowres = io.BytesIO()
    np.savez(buf_normalize_std_lowres, **normalize_std_lowres)
    buf_normalize_std_lowres.seek(0)
    normalize_std_lowres_npz = np.load(buf_normalize_std_lowres)

    buf_normalize_mean_highres = io.BytesIO()
    np.savez(buf_normalize_mean_highres, **normalize_mean_highres)
    buf_normalize_mean_highres.seek(0)
    normalize_mean_highres_npz = np.load(buf_normalize_mean_highres)

    buf_normalize_std_highres = io.BytesIO()
    np.savez(buf_normalize_std_highres, **normalize_std_highres)
    buf_normalize_std_highres.seek(0)
    normalize_std_highres_npz = np.load(buf_normalize_std_highres)

    do_tiling = True
    div = 4
    overlap = 4
    model = PhooORBIT2Precip(x.shape[-2] - 1, x.shape[-1], div, overlap)

    dx = OrbitGlobalPrecip(
        model,
        land_sea_mask,
        orography,
        lattitude,
        landcover,
        normalize_mean_lowres_npz,
        normalize_std_lowres_npz,
        normalize_mean_highres_npz,
        normalize_std_highres_npz,
        do_tiling,
        div,
        overlap,
    )

    signature = coord_array_like(dx.input_coords(), {"batch": [0]})
    field = from_torch(torch.zeros(signature.shape), signature)
    with pytest.raises(ValueError):
        dx(field.rename(variable="wrong"))
    with pytest.raises(ValueError):
        dx(field.transpose("batch", "variable", "lon", "lat"))
    with pytest.raises(ValueError):
        dx(field.isel(lat=slice(None, None, -1)))
