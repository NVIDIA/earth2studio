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
from collections.abc import Iterable

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.px.stormscope import (
    StormScopeGOES,
    StormScopeMRMS,
)
from earth2studio.utils import handshake_dim


# Spoof diffusion model with same call signature as EDMPrecond-wrapped models
class PhooStormScopeDiffusionModel(torch.nn.Module):
    def __init__(self, nvar=8):
        super().__init__()
        self.sigma_min = 0.0
        self.sigma_max = 88.0
        self.nvar = nvar

    def forward(self, x, noise, class_labels=None, condition=None):
        # Return denoised output (same shape as x, but only nvar channels)
        return x[:, : self.nvar, :, :]

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


def create_spoof_model(
    nvar=8,
    nvar_cond=1,
    h=64,
    w=128,
    device="cpu",
    sliding_window=False,
    input_interp=True,
    conditioning_interp=True,
):
    """Create a spoof StormScope model for testing"""
    # Create simple lat/lon grids
    y = np.arange(h)
    x = np.arange(w)
    lat = torch.linspace(25, 50, h).unsqueeze(1).repeat(1, w)
    lon = torch.linspace(-120, -80, w).unsqueeze(0).repeat(h, 1)

    # Create spoof models
    diffusion = PhooStormScopeDiffusionModel(nvar=nvar)

    # Model spec for staged denoising
    model_spec = [
        {
            "model": diffusion,
            "sigma_min": 0.0,
            "sigma_max": 88.0,
        }
    ]

    # Normalization constants
    means = torch.zeros(1, nvar, 1, 1)
    stds = torch.ones(1, nvar, 1, 1)

    # Variables
    variables = np.array([f"var{i:02d}" for i in range(nvar)])

    # Conditioning setup
    conditioning_means = torch.zeros(1, nvar_cond, 1, 1) if nvar_cond > 0 else None
    conditioning_stds = torch.ones(1, nvar_cond, 1, 1) if nvar_cond > 0 else None
    conditioning_variables = (
        np.array([f"cond{i:02d}" for i in range(nvar_cond)]) if nvar_cond > 0 else None
    )

    # Create random conditioning data source
    dc = OrderedDict(
        [("lat", np.linspace(90, -90, num=181)), ("lon", np.linspace(0, 360, num=360))]
    )
    conditioning_data_source = Random(dc) if nvar_cond > 0 else None

    # Input/output times
    if sliding_window:
        input_times = np.array([-2, -1, 0]) * np.timedelta64(1, "h")
        output_times = np.array([1]) * np.timedelta64(1, "h")
    else:
        input_times = np.array([0]) * np.timedelta64(1, "h")
        output_times = np.array([1]) * np.timedelta64(1, "h")

    # Create base model
    model = StormScopeGOES(
        model_spec=model_spec,
        means=means,
        stds=stds,
        latitudes=lat,
        longitudes=lon,
        variables=variables,
        conditioning_means=conditioning_means,
        conditioning_stds=conditioning_stds,
        conditioning_variables=conditioning_variables,
        conditioning_data_source=conditioning_data_source,
        sampler_args={"num_steps": 2},  # Small number for testing
        input_times=input_times,
        output_times=output_times,
        y_coords=y,
        x_coords=x,
    ).to(device)

    if input_interp:
        model.build_input_interpolator(lat, lon)
    if conditioning_interp:
        model.build_conditioning_interpolator(dc["lat"], dc["lon"])

    return model


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2020-04-05T00:00")]),
        np.array(
            [
                np.datetime64("2020-10-11T12:00"),
                np.datetime64("2020-06-04T00:00"),
            ]
        ),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("batch", [1, 2])
def test_stormscope_call(time, device, batch):
    """Test basic StormScope call functionality"""
    nvar = 8
    nvar_cond = 1
    h, w = 64, 128

    model = create_spoof_model(nvar=nvar, nvar_cond=nvar_cond, h=h, w=w, device=device)

    # Create random data source matching model grid
    dc = OrderedDict([("y", model.y), ("x", model.x)])
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Add batch dimension
    x = x.unsqueeze(0).repeat(batch, 1, 1, 1, 1, 1)
    coords.update({"batch": np.arange(batch)})
    coords.move_to_end("batch", last=False)

    # Test forward pass
    out, out_coords = model(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    # Check output shape and coordinates
    assert out.shape == torch.Size([batch, len(time), 1, nvar, h, w])
    assert (out_coords["variable"] == model.output_coords(coords)["variable"]).all()
    assert np.all(out_coords["time"] == time)
    handshake_dim(out_coords, "x", 5)
    handshake_dim(out_coords, "y", 4)
    handshake_dim(out_coords, "variable", 3)
    handshake_dim(out_coords, "lead_time", 2)
    handshake_dim(out_coords, "time", 1)
    handshake_dim(out_coords, "batch", 0)


@pytest.mark.parametrize(
    "batch",
    [1, 2],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormscope_iter(batch, device):
    """Test StormScope iterator mode"""
    time = np.array([np.datetime64("2020-04-05T00:00")])
    nvar = 8
    nvar_cond = 1
    h, w = 32, 64

    model = create_spoof_model(nvar=nvar, nvar_cond=nvar_cond, h=h, w=w, device=device)

    # Create random data source
    dc = OrderedDict([("y", model.y), ("x", model.x)])
    r = Random(dc)

    # Get Data
    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Add batch dimension
    x = x.unsqueeze(0).repeat(batch, 1, 1, 1, 1, 1)
    coords.update({"batch": np.arange(batch)})
    coords.move_to_end("batch", last=False)

    p_iter = model.create_iterator(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    # Get generator
    next(p_iter)  # Skip first which should return the input
    for i, (out, out_coords) in enumerate(p_iter):
        assert len(out.shape) == 6
        assert out.shape == torch.Size([batch, len(time), 1, nvar, h, w])
        assert (
            out_coords["variable"]
            == model.output_coords(model.input_coords())["variable"]
        ).all()
        assert (out_coords["batch"] == np.arange(batch)).all()
        assert out_coords["lead_time"][0] == np.timedelta64(i + 1, "h")

        if i > 3:
            break


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormscope_interpolation(device):
    """Test StormScope interpolation methods"""
    nvar = 8
    nvar_cond = 1
    h, w = 64, 128

    model = create_spoof_model(nvar=nvar, nvar_cond=nvar_cond, h=h, w=w, device=device)

    # Create input data with different grid (coarser)
    h_input, w_input = 32, 64
    input_lat = torch.linspace(25, 50, h_input).unsqueeze(1).repeat(1, w_input)
    input_lon = torch.linspace(-120, -80, w_input).unsqueeze(0).repeat(h_input, 1)

    # Build input interpolator
    model.build_input_interpolator(input_lat, input_lon, max_dist_km=20.0)
    assert model.input_interp is not None
    assert model.valid_mask.shape == torch.Size([h, w])

    # Create random data on the input grid
    time = np.array([np.datetime64("2020-04-05T00:00")])
    dc = OrderedDict([("y", np.arange(h_input)), ("x", np.arange(w_input))])
    r = Random(dc)

    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Test that prep_input can handle the different grid
    x_prep, x_prep_coords = model.prep_input(x, coords)
    assert x_prep.shape[-2:] == (h, w)  # Should be interpolated to model grid
    assert (x_prep_coords["y"] == model.y).all()
    assert (x_prep_coords["x"] == model.x).all()

    # Build conditioning interpolator (with global grid)
    cond_lat = np.linspace(90, -90, num=181)
    cond_lon = np.linspace(0, 360, num=360)
    cond_lat_grid, cond_lon_grid = np.meshgrid(cond_lat, cond_lon, indexing="ij")
    model.build_conditioning_interpolator(
        cond_lat_grid, cond_lon_grid, max_dist_km=30.0
    )
    assert model.conditioning_interp is not None
    assert model.conditioning_valid_mask.shape == torch.Size([h, w])


@pytest.mark.parametrize("sliding_window", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("batch", [1, 2])
def test_stormscope_next_input(sliding_window, device, batch):
    """Test StormScope next_input method for sliding window"""
    nvar = 8
    nvar_cond = 1
    h, w = 32, 64

    model = create_spoof_model(
        nvar=nvar,
        nvar_cond=nvar_cond,
        h=h,
        w=w,
        device=device,
        sliding_window=sliding_window,
    )

    time = np.array([np.datetime64("2020-04-05T00:00")])
    dc = OrderedDict([("y", model.y), ("x", model.x)])
    r = Random(dc)

    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)
    x = x.unsqueeze(0).repeat(batch, 1, 1, 1, 1, 1)
    coords.update({"batch": np.arange(batch)})
    coords.move_to_end("batch", last=False)

    # Simulate a prediction
    pred_coords = model.output_coords(coords)
    pred = torch.randn(
        batch,
        len(time),
        len(pred_coords["lead_time"]),
        nvar,
        h,
        w,
        device=device,
    )

    # Get next input
    next_x, next_coords = model.next_input(pred, pred_coords, x, coords)

    if sliding_window:
        # Should have same number of lead times as input
        assert len(next_coords["lead_time"]) == len(coords["lead_time"])
        # Lead times should be shifted forward
        expected_lead_time = model.input_times + pred_coords["lead_time"][-1]
        assert np.allclose(
            next_coords["lead_time"].astype("timedelta64[h]").astype(int),
            expected_lead_time.astype("timedelta64[h]").astype(int),
        )
        # Should contain old input data and new prediction
        assert next_x.shape[2] == len(next_coords["lead_time"])
    else:
        # Without sliding window, next input is just the prediction
        assert torch.allclose(next_x, pred)
        assert next_coords["lead_time"][0] == pred_coords["lead_time"][0]


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormscope_call_with_conditioning(device):
    """Test StormScope call_with_conditioning method"""
    nvar = 8
    nvar_cond = 1
    h, w = 32, 64

    # Create model without conditioning data source
    model = create_spoof_model(nvar=nvar, nvar_cond=nvar_cond, h=h, w=w, device=device)
    model.conditioning_data_source = None  # Explicitly remove

    time = np.array([np.datetime64("2020-04-05T00:00")])
    dc = OrderedDict([("y", model.y), ("x", model.x)])
    r = Random(dc)

    # Get input data
    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Manually create conditioning data
    conditioning = torch.randn(
        len(time), len(lead_time), nvar_cond, h, w, device=device
    )
    conditioning_coords = OrderedDict(
        {
            "time": coords["time"],
            "lead_time": coords["lead_time"],
            "variable": model.conditioning_variables,
            "y": model.y,
            "x": model.x,
        }
    )

    # Add batch dimension
    batch_size = 2
    x = x.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1, 1)
    coords.update({"batch": np.arange(batch_size)})
    coords.move_to_end("batch", last=False)

    conditioning = conditioning.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1, 1)
    conditioning_coords.update({"batch": np.arange(batch_size)})
    conditioning_coords.move_to_end("batch", last=False)

    # Test call_with_conditioning
    out, out_coords = model.call_with_conditioning(
        x, coords, conditioning, conditioning_coords
    )

    # Check output shape
    assert out.shape == torch.Size([batch_size, len(time), 1, nvar, h, w])
    assert (out_coords["variable"] == variable).all()


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormscope_mrms(device):
    """Test StormScopeMRMS specific functionality"""
    # Create MRMS model (has refc variable and specific preprocessing)
    h, w = 32, 64
    y = np.arange(h)
    x = np.arange(w)
    lat = torch.linspace(25, 50, h).unsqueeze(1).repeat(1, w)
    lon = torch.linspace(-120, -80, w).unsqueeze(0).repeat(h, 1)

    nvar_cond = 8  # GOES conditioning
    diffusion = PhooStormScopeDiffusionModel(nvar=1)  # MRMS has 1 variable (refc)

    model_spec = [{"model": diffusion, "sigma_min": 0.0, "sigma_max": 88.0}]

    means = torch.zeros(1, 1, 1, 1)
    stds = torch.ones(1, 1, 1, 1)
    variables = np.array(["refc"])

    conditioning_means = torch.zeros(1, nvar_cond, 1, 1)
    conditioning_stds = torch.ones(1, nvar_cond, 1, 1)
    conditioning_variables = np.array([f"abi{i:02d}c" for i in range(1, nvar_cond + 1)])

    dc_cond = OrderedDict([("y", y), ("x", x)])
    conditioning_data_source = Random(dc_cond)

    model = StormScopeMRMS(
        model_spec=model_spec,
        means=means,
        stds=stds,
        latitudes=lat,
        longitudes=lon,
        variables=variables,
        conditioning_means=conditioning_means,
        conditioning_stds=conditioning_stds,
        conditioning_variables=conditioning_variables,
        conditioning_data_source=conditioning_data_source,
        sampler_args={"num_steps": 2},
        y_coords=y,
        x_coords=x,
    ).to(device)

    # Test prep_input with low reflectivity values (MRMS-specific preprocessing)
    time = np.array([np.datetime64("2020-04-05T00:00")])
    dc = OrderedDict([("y", model.y), ("x", model.x)])
    r = Random(dc)

    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Set some values to very low reflectivity
    x[:, :, :, :, :] = -25.0  # Should be imputed to -10

    x_prep, x_prep_coords = model.prep_input(x, coords)

    # Check that low reflectivity values were imputed
    assert torch.all(x_prep <= -10.0)  # All values should be >= -10 after imputation

    # Test forward pass
    out, out_coords = model(x, coords)
    assert out.shape == torch.Size([1, 1, 1, h, w])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormscope_exceptions(device):
    """Test StormScope exception handling"""
    nvar = 8
    h, w = 32, 64

    # Test 1: Model without conditioning data source should warn
    model = create_spoof_model(nvar=nvar, nvar_cond=1, h=h, w=w, device=device)
    model.conditioning_data_source = None

    time = np.array([np.datetime64("2020-04-05T00:00")])
    dc = OrderedDict([("y", model.y), ("x", model.x)])
    r = Random(dc)

    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Should raise error when trying to fetch conditioning without data source
    with pytest.raises(RuntimeError):
        model(x, coords)

    # Test 2: Using non-native grid without interpolator should fail
    model2 = create_spoof_model(nvar=nvar, nvar_cond=0, h=h, w=w, device=device)

    h_input, w_input = 16, 32
    dc2 = OrderedDict([("y", np.arange(h_input)), ("x", np.arange(w_input))])
    r2 = Random(dc2)
    x2, coords2 = fetch_data(r2, time, variable, lead_time, device=device)

    with pytest.raises(ValueError):
        # Should fail because we're passing data on wrong grid without interpolator
        model2.prep_input(x2, coords2)

    # Test 3: Invalid coordinate order for call_with_conditioning
    model3 = create_spoof_model(nvar=nvar, nvar_cond=1, h=h, w=w, device=device)
    model3.conditioning_data_source = None

    # Create coordinates with missing required dimensions
    bad_coords = OrderedDict({"variable": variable, "y": model3.y, "x": model3.x})
    conditioning_coords = OrderedDict(
        {
            "time": time,
            "lead_time": lead_time,
            "variable": model3.conditioning_variables,
            "y": model3.y,
            "x": model3.x,
        }
    )

    x_test = torch.randn(1, 1, nvar, h, w, device=device)
    conditioning_test = torch.randn(1, 1, 1, h, w, device=device)

    with pytest.raises(ValueError):
        model3.call_with_conditioning(
            x_test, bad_coords, conditioning_test, conditioning_coords
        )


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormscope_staged_denoising(device):
    """Test StormScope staged denoising with multiple experts"""
    nvar = 8
    h, w = 32, 64
    y = np.arange(h)
    x = np.arange(w)
    lat = torch.linspace(25, 50, h).unsqueeze(1).repeat(1, w)
    lon = torch.linspace(-120, -80, w).unsqueeze(0).repeat(h, 1)

    # Create multiple experts for different sigma ranges
    diffusion1 = PhooStormScopeDiffusionModel(nvar=nvar)
    diffusion2 = PhooStormScopeDiffusionModel(nvar=nvar)

    model_spec = [
        {"model": diffusion1, "sigma_min": 10.0, "sigma_max": 88.0},
        {"model": diffusion2, "sigma_min": 0.0, "sigma_max": 10.0},
    ]

    means = torch.zeros(1, nvar, 1, 1)
    stds = torch.ones(1, nvar, 1, 1)
    variables = np.array([f"var{i:02d}" for i in range(nvar)])

    model = StormScopeGOES(
        model_spec=model_spec,
        means=means,
        stds=stds,
        latitudes=lat,
        longitudes=lon,
        variables=variables,
        conditioning_means=None,
        conditioning_stds=None,
        conditioning_variables=None,
        conditioning_data_source=None,
        sampler_args={"num_steps": 2},
        y_coords=y,
        x_coords=x,
    ).to(device)

    # Check that model_spec is sorted correctly
    assert model.model_spec[0]["sigma_max"] >= model.model_spec[1]["sigma_max"]

    # Test expert selection
    t_high = torch.tensor(50.0, device=device)
    expert_high = model._select_expert(t_high)
    assert expert_high == model.stage_models[0]

    t_low = torch.tensor(5.0, device=device)
    expert_low = model._select_expert(t_low)
    assert expert_low == model.stage_models[1]


@pytest.mark.package
def test_stormscope_package_loading():
    """Test StormScope GOES package loading and a minimal inference step."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.cuda.empty_cache()

    package = StormScopeGOES.load_default_package()
    model = StormScopeGOES.load_model(package, conditioning_data_source=None)
    model = model.to(device)
    model.eval()
    model.sampler_args = {"num_steps": 2, "S_churn": 0}

    batch_size = 1
    time = np.array([np.datetime64("2025-07-01T00:00")])
    coords = model.input_coords()
    coords["batch"] = np.arange(batch_size)
    coords["time"] = time
    lead_times = coords["lead_time"]
    variables = coords["variable"]

    h, w = len(model.y), len(model.x)
    x = torch.randn(
        batch_size,
        len(time),
        len(lead_times),
        len(variables),
        h,
        w,
        device=device,
    )

    if (
        model.conditioning_variables is not None
        and len(model.conditioning_variables) > 0
    ):
        conditioning = torch.randn(
            batch_size,
            len(time),
            len(lead_times),
            len(model.conditioning_variables),
            h,
            w,
            device=device,
        )
        conditioning_coords = OrderedDict(
            {
                "time": coords["time"],
                "lead_time": lead_times,
                "variable": model.conditioning_variables,
                "y": model.y,
                "x": model.x,
            }
        )
        conditioning_coords.update({"batch": np.arange(batch_size)})
        conditioning_coords.move_to_end("batch", last=False)

        out, out_coords = model.call_with_conditioning(
            x, coords, conditioning, conditioning_coords
        )
    else:
        out, out_coords = model(x, coords)

    expected_coords = model.output_coords(coords)
    expected_shape = (
        batch_size,
        len(time),
        len(model.output_times),
        len(model.variables),
        h,
        w,
    )
    assert out.shape == torch.Size(expected_shape)
    assert np.array_equal(out_coords["lead_time"], expected_coords["lead_time"])
    assert (out_coords["variable"] == expected_coords["variable"]).all()
