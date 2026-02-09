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
from earth2studio.models.px import Atlas
from earth2studio.utils import handshake_coords, handshake_dim


class PhooAtlasModel(torch.nn.Module):
    """Dummy Atlas model for testing.

    This model simulates the Atlas forward pass by adding a time delta
    to the input state to represent a prognostic step.
    """

    def __init__(self, delta_t: int = 6, n_vars: int = 75):
        super().__init__()
        self.delta_t = delta_t
        self.n_vars = n_vars

    def forward(self, x):
        """Simple forward that adds delta_t to second time step."""
        # x shape: [batch, 2*n_vars, lat, lon] for two lead times
        # Take second lead time (latest) and add delta_t
        return x[:, self.n_vars :] + self.delta_t


class PhooAutoencoder(torch.nn.Module):
    """Dummy autoencoder for testing."""

    def __init__(self):
        super().__init__()

    def forward(self, x, prediction_latent):
        return x


class PhooNormalizer(torch.nn.Module):
    """Dummy normalizer for testing."""

    def __init__(self):
        super().__init__()

    def normalize(self, x):
        return x

    def unnormalize(self, x):
        return x


class PhooProcessor(torch.nn.Module):
    """Dummy processor for testing."""

    def __init__(self):
        super().__init__()
        self.normalizer_in = PhooNormalizer()
        self.normalizer_out = PhooNormalizer()
        self.downsample_grid_shape = (181, 360)

    def forward(self, x):
        return x

    def preprocess_input(self, x, current_date):
        return x, x

    def intep(self, x, downsample_grid_shape):
        return x

    def postprocess(self, x, x_cur):
        return x


class PhooSinterpolant(torch.nn.Module):
    """Dummy stochastic interpolant for testing."""

    def __init__(self):
        super().__init__()

    def sample(self, model, x, steps, cond, verbose, compute_normalization):
        return x


@pytest.fixture()
def atlas_test_components():
    """Create dummy Atlas model components for testing."""
    n_vars = 75

    # Create dummy components
    autoencoders = torch.nn.ModuleList([PhooAutoencoder() for _ in range(n_vars)])
    autoencoder_processors = torch.nn.ModuleList(
        [PhooProcessor() for _ in range(n_vars)]
    )
    model = PhooAtlasModel(delta_t=6, n_vars=n_vars)
    model_processor = PhooProcessor()
    sinterpolant = PhooSinterpolant()

    return {
        "autoencoders": autoencoders,
        "autoencoder_processors": autoencoder_processors,
        "model": model,
        "model_processor": model_processor,
        "sinterpolant": sinterpolant,
        "sinterpolant_sample_steps": 60,
    }


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("1993-04-05T00:00")]),
        np.array(
            [
                np.datetime64("1999-10-11T12:00"),
                np.datetime64("2001-06-04T00:00"),
            ]
        ),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_atlas_call(time, device, batch_size, atlas_test_components):
    """Test Atlas __call__ method with different times and devices."""
    p = Atlas(**atlas_test_components).to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Add batch dimension
    x = x.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1, 1)
    coords.update({"batch": np.arange(batch_size)})
    coords.move_to_end("batch", last=False)

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size(
        [
            batch_size,
            len(time),
            1,
            len(p.output_coords(p.input_coords())["variable"]),
            721,
            1440,
        ]
    )
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    assert out_coords["lead_time"][0] == np.timedelta64(6, "h")

    handshake_dim(out_coords, "lon", 5)
    handshake_dim(out_coords, "lat", 4)
    handshake_dim(out_coords, "variable", 3)
    handshake_dim(out_coords, "lead_time", 2)
    handshake_dim(out_coords, "time", 1)
    handshake_dim(out_coords, "batch", 0)


@pytest.mark.parametrize(
    "ensemble",
    [1, 2],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_atlas_iter(ensemble, atlas_test_components, device):
    """Test Atlas iterator for autoregressive predictions."""
    time = np.array([np.datetime64("1993-04-05T00:00")])

    p = Atlas(**atlas_test_components).to(device)

    dc = p.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Add ensemble to front
    x = x.unsqueeze(0).repeat(ensemble, 1, 1, 1, 1, 1)
    coords.update({"ensemble": np.arange(ensemble)})
    coords.move_to_end("ensemble", last=False)

    p_iter = p.create_iterator(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    # Get generator
    out, out_coords = next(p_iter)  # Skip first which should return the input
    # First output should be the latest lead time from input
    assert torch.allclose(out, x[:, :, 1:])

    for i, (out, out_coords) in enumerate(p_iter):
        assert len(out.shape) == 6
        assert out.shape[0] == ensemble
        assert (
            out_coords["variable"] == p.output_coords(p.input_coords())["variable"]
        ).all()
        assert (out_coords["time"] == time).all()
        assert out_coords["lead_time"][0] == np.timedelta64(6 * (i + 1), "h")

        handshake_dim(out_coords, "lon", 5)
        handshake_dim(out_coords, "lat", 4)
        handshake_dim(out_coords, "variable", 3)
        handshake_dim(out_coords, "lead_time", 2)
        handshake_dim(out_coords, "time", 1)
        handshake_dim(out_coords, "ensemble", 0)

        if i > 3:
            break


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(720)}),
        OrderedDict({"lat": np.random.randn(720), "phoo": np.random.randn(1440)}),
        OrderedDict({"lat": np.random.randn(720), "lon": np.random.randn(1)}),
    ],
)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_atlas_exceptions(dc, atlas_test_components, device):
    """Test that Atlas raises exceptions for invalid coordinates."""
    time = np.array([np.datetime64("1993-04-05T00:00")])

    p = Atlas(**atlas_test_components).to(device)

    # Initialize Data Source with invalid coordinates
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    with pytest.raises((KeyError, ValueError, RuntimeError)):
        p(x, coords)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_atlas_prep_next_input(atlas_test_components, batch_size, device):
    """Test Atlas prep_next_input method for autoregressive stepping.

    The prep_next_input method should:
    1. Take the prediction at t+6h and place it as the latest input
    2. Shift the previous latest input (t=0) to the earlier position (t-6h)
    3. Update lead times by +6h
    """
    p = Atlas(**atlas_test_components).to(device)

    # Create input data with 2 lead times
    time_steps = 2
    n_vars = 75
    lat = 721
    lon = 1440

    # Input state at t-6h and t=0
    x = torch.randn(batch_size, 1, time_steps, n_vars, lat, lon, device=device)
    coords = p.input_coords()
    coords["batch"] = np.arange(batch_size)
    coords["time"] = np.array([np.datetime64("2020-01-01T00:00")])

    # Prediction at t+6h (output has shape [batch, 1, n_vars, lat, lon])
    x_pred = torch.randn(batch_size, 1, 1, n_vars, lat, lon, device=device)
    coords_pred = p.output_coords(coords)
    coords_pred["batch"] = coords["batch"]
    coords_pred["time"] = coords["time"]

    # Call prep_next_input
    x_next, coords_next = p.prep_next_input(x_pred, coords_pred, x, coords)

    # Check that x_next has the correct shape
    assert x_next.shape == x.shape

    # Check that the latest lead time contains the prediction
    assert torch.allclose(x_next[:, :, -1:], x_pred[:, :, :1])

    # Check that the earlier lead time contains the previous latest
    assert torch.allclose(x_next[:, :, :-1], x[:, :, 1:])

    # Check that lead times are updated correctly
    expected_lead_time = coords["lead_time"] + p.DT
    assert np.array_equal(coords_next["lead_time"], expected_lead_time)

    # Check other coordinates remain unchanged
    assert np.array_equal(coords_next["batch"], coords["batch"])
    assert np.array_equal(coords_next["time"], coords["time"])
    assert np.array_equal(coords_next["variable"], coords["variable"])
    assert np.array_equal(coords_next["lat"], coords["lat"])
    assert np.array_equal(coords_next["lon"], coords["lon"])


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_atlas_prep_next_input_with_ensemble(atlas_test_components, device):
    """Test prep_next_input with ensemble dimension."""
    p = Atlas(**atlas_test_components).to(device)

    # Create input data with ensemble dimension
    ensemble_size = 3
    batch_size = 2
    time_steps = 2
    n_vars = 75
    lat = 721
    lon = 1440

    # Input state at t-6h and t=0 with ensemble
    x = torch.randn(
        ensemble_size, batch_size, 1, time_steps, n_vars, lat, lon, device=device
    )
    coords = p.input_coords()
    coords.update({"ensemble": np.arange(ensemble_size)})
    coords.move_to_end("ensemble", last=False)
    coords["batch"] = np.arange(batch_size)
    coords["time"] = np.array([np.datetime64("2020-01-01T00:00")] * batch_size)

    # Prediction at t+6h
    x_pred = torch.randn(
        ensemble_size, batch_size, 1, 1, n_vars, lat, lon, device=device
    )
    coords_pred = p.output_coords(coords)
    coords_pred.update({"ensemble": coords["ensemble"]})
    coords_pred.move_to_end("ensemble", last=False)
    coords_pred["batch"] = coords["batch"]
    coords_pred["time"] = coords["time"]

    # Call prep_next_input
    x_next, coords_next = p.prep_next_input(x_pred, coords_pred, x, coords)

    # Check shapes
    assert x_next.shape == x.shape

    # Check that sliding window works correctly with ensemble dimension
    assert torch.allclose(x_next[:, :, :, -1:], x_pred[:, :, :, :1])
    assert torch.allclose(x_next[:, :, :, :-1], x[:, :, :, 1:])

    # Check ensemble coordinate is preserved
    assert np.array_equal(coords_next["ensemble"], coords["ensemble"])


def test_atlas_input_coords(atlas_test_components):
    """Test that input_coords returns expected coordinate system."""
    p = Atlas(**atlas_test_components)
    coords = p.input_coords()

    # Check expected keys
    assert "batch" in coords
    assert "time" in coords
    assert "lead_time" in coords
    assert "variable" in coords
    assert "lat" in coords
    assert "lon" in coords

    # Check lead_time has two steps: -6h and 0h
    assert len(coords["lead_time"]) == 2
    assert coords["lead_time"][0] == np.timedelta64(-6, "h")
    assert coords["lead_time"][1] == np.timedelta64(0, "h")

    # Check variable count
    assert len(coords["variable"]) == 75

    # Check spatial dimensions
    assert len(coords["lat"]) == 721
    assert len(coords["lon"]) == 1440

    # Check spatial range
    assert coords["lat"][0] == pytest.approx(90.0, abs=1e-5)
    assert coords["lat"][-1] == pytest.approx(-90.0, abs=1e-5)
    assert coords["lon"][0] == pytest.approx(0.0, abs=1e-5)
    assert coords["lon"][-1] == pytest.approx(360.0 - (360.0 / 1440.0), abs=1e-5)


def test_atlas_output_coords(atlas_test_components):
    """Test that output_coords returns expected coordinate system."""
    p = Atlas(**atlas_test_components)
    input_coords = p.input_coords()
    output_coords = p.output_coords(input_coords)

    # Check expected keys
    assert "batch" in output_coords
    assert "time" in output_coords
    assert "lead_time" in output_coords
    assert "variable" in output_coords
    assert "lat" in output_coords
    assert "lon" in output_coords

    # Check lead_time is single step at +6h
    assert len(output_coords["lead_time"]) == 1
    assert output_coords["lead_time"][0] == np.timedelta64(6, "h")

    # Check variable count matches input
    assert len(output_coords["variable"]) == len(input_coords["variable"])

    # Check spatial dimensions match input
    assert len(output_coords["lat"]) == len(input_coords["lat"])
    assert len(output_coords["lon"]) == len(input_coords["lon"])


@pytest.mark.package
@pytest.mark.parametrize("device", ["cuda:0"])
def test_atlas_package(device):
    """Test that Atlas loads from package and runs a forward pass."""
    torch.cuda.empty_cache()

    model = Atlas.load_model(Atlas.load_default_package()).to(device)
    model.sinterpolant_sample_steps = 4  # reduce sample steps for testing

    batch_size = 1
    time = np.array([np.datetime64("2020-01-01T00:00")])
    input_coords = model.input_coords()
    lead_time = input_coords["lead_time"]
    variable = input_coords["variable"]
    lat = len(input_coords["lat"])
    lon = len(input_coords["lon"])

    x = torch.randn(
        batch_size,
        len(time),
        len(lead_time),
        len(variable),
        lat,
        lon,
        device=device,
    )

    input_coords["batch"] = np.arange(batch_size)
    input_coords["time"] = time

    output, output_coords = model(x, input_coords)
    expected_coords = model.output_coords(input_coords)

    assert output.shape == (
        batch_size,
        len(time),
        len(expected_coords["lead_time"]),
        len(variable),
        lat,
        lon,
    )
    for key in expected_coords:
        handshake_coords(output_coords, expected_coords, key)
