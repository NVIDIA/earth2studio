# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
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
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.io import XarrayBackend
from earth2studio.models.dx import (
    CorrDiffTaiwan,
    DerivedSurfacePressure,
    PrecipitationAFNOv2,
    SolarRadiationAFNO1H,
    TCTrackerVitart,
)
from earth2studio.models.px import FCN3, DiagnosticWrapper
from earth2studio.run import deterministic
from earth2studio.utils.coords import map_coords


class PhooFCN3Model(torch.nn.Module):
    def forward(self, x, t, normalized_data: bool = False, replace_state: bool = False):
        return x


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

    def forward(self, x, img_lr, class_labels=None, force_fp32=False, **model_kwargs):
        return x[:, :4]

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("model_type", ["precip", "solar"])
@pytest.mark.parametrize(
    "times",
    [
        [np.datetime64("2025-08-21T00:00:00")],
        [np.datetime64("2025-08-21T00:00:00"), np.datetime64("2025-08-22T00:00:00")],
    ],
)
def test_fcn3_precip(device, model_type, times):
    # Spoof models
    fcn3_model = PhooFCN3Model()
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

    wrapped_model = DiagnosticWrapper(
        px_model=px_model, dx_models=[sp_model, dx_model]
    ).to(device=device)

    dc = {k: wrapped_model.input_coords()[k] for k in ["lat", "lon"]}
    data = Random(dc)

    (x, coords) = fetch_data(
        data,
        times,
        variable=wrapped_model.input_coords()["variable"],
        device=device,
    )
    (x, coords) = map_coords(x, coords, wrapped_model.input_coords())

    (x, coords) = wrapped_model(x, coords)

    expected_shape = tuple(len(v) for v in coords.values())
    expected_vars = np.concatenate(
        [
            model.output_coords(model.input_coords())["variable"]
            for model in [px_model, sp_model, dx_model]
        ]
    )
    assert (
        x.shape
        == expected_shape
        == (len(times), 1, len(expected_vars), *landsea_mask.shape[-2:])
    )
    assert tuple(coords) == ("time", "lead_time", "variable", "lat", "lon")
    assert (coords["variable"] == expected_vars).all()

    io = XarrayBackend()
    deterministic(times, 2, wrapped_model, data, io, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    "times",
    [
        [np.datetime64("2025-08-21T00:00:00")],
        [np.datetime64("2025-08-21T00:00:00"), np.datetime64("2025-08-22T00:00:00")],
    ],
)
@pytest.mark.parametrize("number_of_samples", [1, 2])
@pytest.mark.parametrize("keep_px_output", [False, True])
def test_fcn3_corrdiff(device, times, number_of_samples, keep_px_output):
    # Spoof models
    px_model = FCN3(PhooFCN3Model())
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

    wrapped_model = DiagnosticWrapper(
        px_model=px_model,
        dx_models=corrdiff_model,
        keep_px_output=keep_px_output,
        interpolate_coords=True,
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

    with pytest.raises(ValueError) if keep_px_output else nullcontext():
        (x, coords) = wrapped_model(x, coords)
    if keep_px_output:
        return

    expected_shape = tuple(len(v) for v in coords.values())
    assert x.shape == expected_shape == (len(times), 1, number_of_samples, 4, 448, 448)
    expected_vars = corrdiff_model.output_coords(corrdiff_model.input_coords())[
        "variable"
    ]
    assert (coords["variable"] == expected_vars).all()

    io = XarrayBackend()
    deterministic(times, 2, wrapped_model, data, io, device=device)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize(
    "times",
    [
        [np.datetime64("2025-08-21T00:00:00")],
        [np.datetime64("2025-08-21T00:00:00"), np.datetime64("2025-08-22T00:00:00")],
    ],
)
@pytest.mark.parametrize("keep_px_output", [False, True])
def test_fcn3_tc_tracker(device, times, keep_px_output):
    # Spoof models
    fcn3_model = PhooFCN3Model()
    px_model = FCN3(fcn3_model)

    tc_tracker = TCTrackerVitart()

    wrapped_model = DiagnosticWrapper(
        px_model=px_model, dx_models=tc_tracker, keep_px_output=keep_px_output
    ).to(device=device)

    dc = {k: wrapped_model.input_coords()[k] for k in ["lat", "lon"]}
    data = Random(dc)

    (x, coords) = fetch_data(
        data,
        times,
        variable=wrapped_model.input_coords()["variable"],
        device=device,
    )
    (x, coords) = map_coords(x, coords, wrapped_model.input_coords())

    with pytest.raises(ValueError) if keep_px_output else nullcontext():
        (x, coords) = wrapped_model(x, coords)
    if keep_px_output:
        return

    expected_shape = tuple(len(v) for v in coords.values())
    assert x.shape == expected_shape
    assert all(x.shape[dim] == (len(times), 1, -1, 1, 4)[dim] for dim in (0, 1, 3, 4))
    expected_vars = tc_tracker.output_coords(tc_tracker.input_coords())["variable"]
    assert (coords["variable"] == expected_vars).all()
