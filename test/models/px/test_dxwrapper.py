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

from earth2studio.data import Random, fetch_data
from earth2studio.io import XarrayBackend
from earth2studio.models.dx import (
    CorrDiffTaiwan,
    DerivedSurfacePressure,
    DerivedWS,
    PrecipitationAFNOv2,
    SolarRadiationAFNO1H,
)
from earth2studio.models.px import FCN3, DiagnosticWrapper, Persistence
from earth2studio.run import deterministic
from earth2studio.utils.coords import map_coords


class PhooFCN3Preprocessor(torch.nn.Module):

    def __init__(
        self,
    ):
        super().__init__()
        self.register_buffer(
            "state",
            torch.randn(
                10,
            ),
        )

    def set_internal_state(self, state: torch.Tensor):
        self.state = state.to(self.state.device)

    def get_internal_state(self, tensor=True):
        return self.state

    def update_internal_state(self, replace_state=True):
        self.state = torch.randn((10,), device=self.state.device)


class PhooFCN3Model(torch.nn.Module):
    def __init__(self, preprocessor):
        super().__init__()
        self.preprocessor = preprocessor


class PhooFCN3ModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, t, normalized_data: bool = False, replace_state: bool = False):
        return x

    def set_rng(self, reset: bool = True, seed: int = 333):
        return


class PhooAFNOPrecipV2(torch.nn.Module):
    def forward(self, x):
        return x[:, :1, :, :]


class PhooAFNOSolarRadiation(torch.nn.Module):
    """Mock model for testing."""

    def forward(self, x):
        # x: (batch, variables, lat, lon)
        # The model expects input shape (batch, variables, lat, lon)
        # where variables includes the input variables plus sza, sincos_latlon, orography, and landsea_mask
        # We'll return a tensor of the same shape but with only one variable
        return torch.zeros_like(x[:, :1, :, :])


class PhooCorrDiff(torch.nn.Module):
    img_out_channels = 4
    img_resolution = 448
    sigma_min = 0
    sigma_max = float("inf")

    def __init__(self):
        super().__init__()
        self.register_buffer("device_buffer", torch.empty(0))

    @property
    def device(self) -> torch.device:
        return self.device_buffer.device

    @device.setter
    def device(self, value) -> None:
        dev = torch.device(value)
        self.device_buffer = torch.empty(0, device=dev)

    def forward(self, x, img_lr, class_labels=None, force_fp32=False, **model_kwargs):
        return x[:, :4]

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


@pytest.mark.parametrize("device", ["cuda:0"])  # Removing CPU here too slow atm "cpu",
@pytest.mark.parametrize("model_type", ["precip", "solar"])
@pytest.mark.parametrize(
    "times",
    [
        [np.datetime64("2025-08-21T00:00:00")],
        [np.datetime64("2025-08-21T00:00:00"), np.datetime64("2025-08-22T00:00:00")],
    ],
)
def test_dxwrapper_call(device, model_type, times):
    # Spoof models
    fcn3_model = PhooFCN3ModelWrapper(PhooFCN3Model(PhooFCN3Preprocessor()))
    px_model = FCN3(fcn3_model)

    if model_type == "precip":
        precipafnov2_model = PhooAFNOPrecipV2()
        center = torch.zeros(20, 1, 1)
        scale = torch.ones(20, 1, 1)
        landsea_mask = torch.zeros(1, 1, 720, 1440)
        orography = torch.zeros(1, 1, 720, 1440)

        dx_model = PrecipitationAFNOv2(
            precipafnov2_model, landsea_mask, orography, center, scale
        ).to(device)
    elif model_type == "solar":
        era5_mean = torch.zeros(24, 1, 1)
        era5_std = torch.ones(24, 1, 1)
        ssrd_mean = torch.zeros(1, 1, 1)
        ssrd_std = torch.ones(1, 1, 1)
        orography = torch.zeros(1, 1, 721, 1440)
        landsea_mask = torch.zeros(1, 1, 721, 1440)
        sincos_latlon = torch.zeros(1, 4, 721, 1440)

        dx_model = SolarRadiationAFNO1H(
            core_model=PhooAFNOSolarRadiation(),
            freq="1h",
            era5_mean=era5_mean,
            era5_std=era5_std,
            ssrd_mean=ssrd_mean,
            ssrd_std=ssrd_std,
            orography=orography,
            landsea_mask=landsea_mask,
            sincos_latlon=sincos_latlon,
        ).to(device)

    px_out_coords = px_model.output_coords(px_model.input_coords())
    sp_model = DerivedSurfacePressure(
        p_levels=[50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
        surface_geopotential=torch.zeros(721, 1440),
        surface_geopotential_coords=OrderedDict(
            {"lat": px_out_coords["lat"], "lon": px_out_coords["lon"]}
        ),
    )
    ws_model = DerivedWS(levels=["100m"])

    wrapped_model = DiagnosticWrapper(
        px_model=px_model, dx_model=[sp_model, ws_model]
    ).to(device=device)
    wrapped_model = DiagnosticWrapper(px_model=wrapped_model, dx_model=[dx_model]).to(
        device=device
    )

    dc = {k: wrapped_model.input_coords()[k] for k in ["lat", "lon"]}
    data = Random(dc)

    (x, coords) = fetch_data(
        data,
        times,
        variable=wrapped_model.input_coords()["variable"],
        device=device,
    )
    (x, input_coords) = map_coords(x, coords, wrapped_model.input_coords())
    (x, coords) = wrapped_model(x, input_coords)

    coord_shape = tuple(coord.shape[0] for coord in coords.values())
    expected_shape = tuple(
        coord.shape[0] for coord in wrapped_model.output_coords(input_coords).values()
    )
    assert x.shape == coord_shape
    assert x.shape == expected_shape
    assert tuple(coords) == ("time", "lead_time", "variable", "lat", "lon")


@pytest.mark.parametrize("device", ["cuda:0"])  # Removing CPU here too slow atm "cpu",
@pytest.mark.parametrize(
    "times,number_of_samples",
    [
        ([np.datetime64("2025-08-21T00:00:00")], 2),
        (
            [
                np.datetime64("2025-08-21T00:00:00"),
                np.datetime64("2025-08-22T00:00:00"),
            ],
            1,
        ),
    ],
)
def test_dxwrapper_iter(device, times, number_of_samples):
    # Spoof models
    model = PhooCorrDiff()
    in_center = torch.zeros(12, 1, 1)
    in_scale = torch.ones(12, 1, 1)
    out_center = torch.zeros(4, 1, 1)
    out_scale = torch.ones(4, 1, 1)
    lat = torch.as_tensor(np.linspace(19.5, 27, 450, endpoint=True))
    lon = torch.as_tensor(np.linspace(117, 125, 450, endpoint=False))
    out_lon, out_lat = torch.meshgrid(lon, lat)
    corrdiff_model = CorrDiffTaiwan(
        model,
        model,
        in_center,
        in_scale,
        out_center,
        out_scale,
        out_lat,
        out_lon,
        number_of_samples=number_of_samples,
    ).to(device)

    # Create persistence prognostic model
    lat = np.linspace(-90, 90, 721)
    lon = np.linspace(0, 360, 1440, endpoint=False)
    domain_coords = OrderedDict({"lat": lat, "lon": lon})
    px_model = Persistence(
        variable=corrdiff_model.input_coords()["variable"],
        domain_coords=domain_coords,
        dt=np.timedelta64(6, "h"),
    ).to(device)

    wrapped_model = DiagnosticWrapper(
        px_model=px_model,
        dx_model=corrdiff_model,
    ).to(device=device)

    dc = {k: wrapped_model.input_coords()[k] for k in ["lat", "lon"]}
    data = Random(dc)

    (x, coords) = fetch_data(
        data,
        times,
        variable=px_model.input_coords()["variable"],
        device=device,
    )
    (x, coords) = map_coords(x, coords, wrapped_model.input_coords())
    # Get generator
    p_iter = wrapped_model.create_iterator(x, coords)
    for i, (out, out_coords) in enumerate(p_iter):

        coord_shape = tuple(coord.shape[0] for coord in out_coords.values())
        expected_shape = tuple(
            coord.shape[0] for coord in wrapped_model.output_coords(coords).values()
        )
        assert out.shape == coord_shape
        assert out.shape == expected_shape

        if i == 2:
            break


@pytest.mark.parametrize("device", ["cuda:0"])
@pytest.mark.parametrize(
    "times,number_of_samples",
    [
        ([np.datetime64("2025-08-21T00:00:00")], 1),
    ],
)
def test_dxwrapper_run(device, times, number_of_samples):

    model = PhooCorrDiff()
    in_center = torch.zeros(12, 1, 1)
    in_scale = torch.ones(12, 1, 1)
    out_center = torch.zeros(4, 1, 1)
    out_scale = torch.ones(4, 1, 1)
    lat = torch.as_tensor(np.linspace(19.5, 27, 450, endpoint=True))
    lon = torch.as_tensor(np.linspace(117, 125, 450, endpoint=False))
    out_lon, out_lat = torch.meshgrid(lon, lat)
    corrdiff_model = CorrDiffTaiwan(
        model,
        model,
        in_center,
        in_scale,
        out_center,
        out_scale,
        out_lat,
        out_lon,
        number_of_samples=number_of_samples,
    ).to(device)

    # Create persistence prognostic model
    lat = np.linspace(-90, 90, 721)
    lon = np.linspace(0, 360, 1440, endpoint=False)
    domain_coords = OrderedDict({"lat": lat, "lon": lon})
    px_model = Persistence(
        variable=corrdiff_model.input_coords()["variable"],
        domain_coords=domain_coords,
        dt=np.timedelta64(6, "h"),
    ).to(device)

    wrapped_model = DiagnosticWrapper(
        px_model=px_model,
        dx_model=corrdiff_model,
    ).to(device=device)

    dc = {k: wrapped_model.input_coords()[k] for k in ["lat", "lon"]}
    data = Random(dc)

    (x, coords) = fetch_data(
        data,
        times,
        variable=px_model.input_coords()["variable"],
        device=device,
    )
    (x, coords) = map_coords(x, coords, wrapped_model.input_coords())
    io = XarrayBackend()
    deterministic(times, 2, wrapped_model, data, io, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_prepare_output1(device):
    """Test strategy 1: Direct concatenation when all coords match"""
    from earth2studio.models.px.dxwrapper import PrepareOutputTensorDefault

    prepare_output = PrepareOutputTensorDefault()

    # Create matching coordinate systems
    px_coords = OrderedDict(
        [
            ("time", np.array([np.datetime64("2024-01-01")])),
            ("variable", np.array(["t2m", "u10m"])),
            ("lat", np.linspace(-90, 90, 10)),
            ("lon", np.linspace(0, 360, 20)),
        ]
    )

    dx_coords = [
        OrderedDict(
            [
                ("time", np.array([np.datetime64("2024-01-01")])),
                ("variable", np.array(["precip"])),
                ("lat", np.linspace(-90, 90, 10)),
                ("lon", np.linspace(0, 360, 20)),
            ]
        )
    ]

    # Create tensors
    px_x = torch.randn(1, 2, 10, 20, device=device)
    dx_x = [torch.randn(1, 1, 10, 20, device=device)]

    x_out, coords_out = prepare_output(px_x, px_coords, dx_x, dx_coords)

    # Verify shape and concatenation
    assert x_out.shape == (1, 3, 10, 20)
    assert list(coords_out["variable"]) == ["t2m", "u10m", "precip"]
    assert x_out.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_prepare_output2(device):
    """Test strategy 2: Subregion extraction when dx is a lat/lon subregion of px"""
    from earth2studio.models.px.dxwrapper import PrepareOutputTensorDefault

    prepare_output = PrepareOutputTensorDefault()

    # Create px coords with larger spatial domain
    lat_px = np.linspace(-90, 90, 20)
    lon_px = np.linspace(0, 360, 40)
    px_coords = OrderedDict(
        [
            ("time", np.array([np.datetime64("2024-01-01")])),
            ("variable", np.array(["t2m", "u10m"])),
            ("lat", lat_px),
            ("lon", lon_px),
        ]
    )

    # Create dx coords with subregion (middle section)
    lat_dx = lat_px[5:15]  # Contiguous subregion
    lon_dx = lon_px[10:30]  # Contiguous subregion
    dx_coords = [
        OrderedDict(
            [
                ("time", np.array([np.datetime64("2024-01-01")])),
                ("variable", np.array(["precip"])),
                ("lat", lat_dx),
                ("lon", lon_dx),
            ]
        )
    ]

    # Create tensors
    px_x = torch.randn(1, 2, 20, 40, device=device)
    dx_x = [torch.randn(1, 1, 10, 20, device=device)]

    x_out, coords_out = prepare_output(px_x, px_coords, dx_x, dx_coords)

    # Verify shape and concatenation
    assert x_out.shape == (1, 3, 10, 20)
    assert list(coords_out["variable"]) == ["t2m", "u10m", "precip"]

    # Verify that the sliced region was extracted correctly
    # The first 2 variables should match the subregion of px_x
    expected_slice = px_x[:, :, 5:15, 10:30]
    assert torch.allclose(x_out[:, :2, :, :], expected_slice)
    assert x_out.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_prepare_output3(device):

    from earth2studio.models.px.dxwrapper import PrepareOutputTensorDefault

    prepare_output = PrepareOutputTensorDefault()

    # Create incompatible coordinate systems (different time dimension)
    px_coords = OrderedDict(
        [
            ("time", np.array([np.datetime64("2024-01-01")])),
            ("variable", np.array(["t2m", "u10m"])),
            ("lat", np.linspace(-90, 90, 10)),
            ("lon", np.linspace(0, 360, 20)),
        ]
    )

    dx_coords = [
        OrderedDict(
            [
                (
                    "time",
                    np.array(
                        [np.datetime64("2024-01-01"), np.datetime64("2024-01-02")]
                    ),
                ),
                ("variable", np.array(["precip"])),
                ("lat", np.linspace(-90, 90, 10)),
                ("lon", np.linspace(0, 360, 20)),
            ]
        )
    ]

    # Create tensors
    px_x = torch.randn(1, 2, 10, 20, device=device)
    dx_x = [torch.randn(2, 1, 10, 20, device=device)]

    # Test forward pass
    x_out, coords_out = prepare_output(px_x, px_coords, dx_x, dx_coords)

    # Verify only dx outputs are used
    assert x_out.shape == (2, 1, 10, 20)
    assert list(coords_out["variable"]) == ["precip"]
    assert x_out.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_prepare_output_subregion(device):
    from earth2studio.models.px.dxwrapper import PrepareOutputTensorDefault

    prepare_output = PrepareOutputTensorDefault()

    # Create px coords
    lat_px = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    lon_px = np.linspace(0, 360, 40)
    px_coords = OrderedDict(
        [
            ("time", np.array([np.datetime64("2024-01-01")])),
            ("variable", np.array(["t2m"])),
            ("lat", lat_px),
            ("lon", lon_px),
        ]
    )

    # Create dx coords with non-contiguous lat indices (skip some values)
    lat_dx = np.array([0, 20, 40, 60, 80])  # Non-contiguous in the array
    lon_dx = lon_px[10:30]  # Contiguous
    dx_coords = [
        OrderedDict(
            [
                ("time", np.array([np.datetime64("2024-01-01")])),
                ("variable", np.array(["precip"])),
                ("lat", lat_dx),
                ("lon", lon_dx),
            ]
        )
    ]
    px_x = torch.randn(1, 1, 10, 40, device=device)
    dx_x = [torch.randn(1, 1, 5, 20, device=device)]

    x_out, coords_out = prepare_output(px_x, px_coords, dx_x, dx_coords)

    assert x_out.shape == (1, 1, 5, 20)
    assert list(coords_out["variable"]) == ["precip"]
    assert torch.equal(x_out, dx_x[0])
    assert x_out.device.type == device.split(":")[0]

    # Create a dx with a lat lon domain thats out of bounds of prognostic
    dx_coords = [
        OrderedDict(
            [
                ("time", np.array([np.datetime64("2024-01-01")])),
                ("variable", np.array(["precip"])),
                ("lat", np.linspace(-10, 80, 8)),
                ("lon", np.linspace(200, 360, 10)),
            ]
        )
    ]
    px_x = torch.randn(1, 1, 10, 20, device=device)
    dx_x = [torch.randn(1, 1, 8, 10, device=device)]

    x_out, coords_out = prepare_output(px_x, px_coords, dx_x, dx_coords)

    assert x_out.shape == (1, 1, 8, 10)
    assert list(coords_out["variable"]) == ["precip"]
    assert torch.equal(x_out, dx_x[0])
    assert x_out.device.type == device.split(":")[0]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_prepare_output_tensor_multiple_dx_models(device):
    from earth2studio.models.px.dxwrapper import PrepareOutputTensorDefault

    prepare_output = PrepareOutputTensorDefault()

    px_coords = OrderedDict(
        [
            ("time", np.array([np.datetime64("2024-01-01")])),
            ("variable", np.array(["t2m", "u10m"])),
            ("lat", np.linspace(-90, 90, 10)),
            ("lon", np.linspace(0, 360, 20)),
        ]
    )

    dx_coords = [
        OrderedDict(
            [
                ("time", np.array([np.datetime64("2024-01-01")])),
                ("variable", np.array(["precip"])),
                ("lat", np.linspace(-90, 90, 10)),
                ("lon", np.linspace(0, 360, 20)),
            ]
        ),
        OrderedDict(
            [
                ("time", np.array([np.datetime64("2024-01-01")])),
                ("variable", np.array(["solar"])),
                ("lat", np.linspace(-90, 90, 10)),
                ("lon", np.linspace(0, 360, 20)),
            ]
        ),
    ]
    px_x = torch.randn(1, 2, 10, 20, device=device)
    dx_x = [
        torch.randn(1, 1, 10, 20, device=device),
        torch.randn(1, 1, 10, 20, device=device),
    ]

    x_out, coords_out = prepare_output(px_x, px_coords, dx_x, dx_coords)

    # Verify shape and concatenation with multiple dx models
    assert x_out.shape == (1, 4, 10, 20)
    assert list(coords_out["variable"]) == ["t2m", "u10m", "precip", "solar"]
    assert x_out.device.type == device.split(":")[0]
