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
from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.utils.imports import pytest_require

pytestmark = pytest_require(groups=["orbit"])

from earth2studio.models.dx import OrbitGlobalPrecip  # noqa: E402
from earth2studio.utils import handshake_dim  # noqa: E402


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

    coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )

    out, out_coords = dx(x, coords)

    assert out.shape == torch.Size([x.shape[0], 1, 2880, 5760])
    assert out_coords["variable"] == dx.output_coords(coords)["variable"]
    handshake_dim(out_coords, "lon", 3)
    handshake_dim(out_coords, "lat", 2)
    handshake_dim(out_coords, "variable", 1)
    handshake_dim(out_coords, "batch", 0)


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize("model_size", ["9.5m", "126m"])
def test_orbit2_precip_package(device, model_size):
    package = OrbitGlobalPrecip.load_default_package()
    dx = OrbitGlobalPrecip.load_model(
        package, "global", model_size, "precipitation"
    ).to(device)
    x = torch.randn(1, 20, 721, 1440).to(device)
    coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )

    out, out_coords = dx(x, coords)
    assert out.shape == torch.Size([x.shape[0], 1, 2880, 5760])
    assert out_coords["variable"] == dx.output_coords(coords)["variable"]
    handshake_dim(out_coords, "lon", 3)
    handshake_dim(out_coords, "lat", 2)
    handshake_dim(out_coords, "variable", 1)
    handshake_dim(out_coords, "batch", 0)


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

    wrong_coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "wrong": dx.input_coords()["variable"],
            "lat": dx.input_coords()["lat"],
            "lon": dx.input_coords()["lon"],
        }
    )

    with pytest.raises((KeyError, ValueError)):
        dx(x, wrong_coords)

    wrong_coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lon": dx.input_coords()["lon"],
            "lat": dx.input_coords()["lat"],
        }
    )

    with pytest.raises(ValueError):
        dx(x, wrong_coords)

    wrong_coords = OrderedDict(
        {
            "batch": np.ones(x.shape[0]),
            "variable": dx.input_coords()["variable"],
            "lat": np.linspace(-90, 90, 721),
            "lon": dx.input_coords()["lon"],
        }
    )
    with pytest.raises(ValueError):
        dx(x, wrong_coords)
