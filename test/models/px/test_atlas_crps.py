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
from collections import OrderedDict
from collections.abc import Iterable

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.data import Random, fetch_data
from earth2studio.grids import LatLonGrid
from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.px import AtlasCRPS
from earth2studio.utils import coord_array, coord_array_like
from earth2studio.utils.cupy import from_torch


class PhooAtlasCRPSModel(torch.nn.Module):
    """Dummy AtlasCRPS model for testing.

    This model simulates the CRPS latent transformer by adding a time delta to the
    current state to represent a prognostic step.
    """

    def __init__(self, delta_t: int = 6, n_vars: int = 75):
        super().__init__()
        self.delta_t = delta_t
        self.n_vars = n_vars

    def forward(self, x_1, x_2):
        """Simple forward that adds delta_t to the current state stream."""
        return x_2 * 0.1 + x_1 * 0.2


class PhooAutoencoder(torch.nn.Module):
    """Dummy autoencoder for testing."""

    def __init__(self):
        super().__init__()

    def forward(self, x, residual_latent):
        return residual_latent


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

    def preprocess_conditioning(self, high_res, low_res):
        return low_res

    def intep(self, x, downsample_grid_shape):
        return x

    def postprocess(self, x, x_cur):
        return x


@pytest.fixture()
def atlas_crps_test_components():
    """Create dummy AtlasCRPS model components for testing."""
    n_vars = 75

    return {
        "model": PhooAtlasCRPSModel(delta_t=6, n_vars=n_vars),
        "model_processor": PhooProcessor(),
        "autoencoder": PhooAutoencoder(),
        "autoencoder_processor": PhooProcessor(),
    }


@pytest.fixture(autouse=True)
def model_domain(request, monkeypatch):
    if (
        "device" in request.fixturenames
        and request.getfixturevalue("device").startswith("cuda")
        and not torch.cuda.is_available()
    ):
        pytest.skip("CUDA unavailable")
    if request.node.originalname == "test_atlas_crps_iter":
        declared = AtlasCRPS.input_coords

        def small(self):
            signature = declared(self)
            return coord_array(
                signature.dims,
                {
                    "lead_time": signature.lead_time,
                    "variable": signature.coords["variable"],
                },
                dynamic=("batch", "time"),
                grid=LatLonGrid([45, -45], [0, 120, 240]),
            )

        monkeypatch.setattr(AtlasCRPS, "input_coords", small)
    elif request.node.originalname not in (
        "test_atlas_crps_input_coords",
        "test_atlas_crps_output_coords",
    ):
        pytest.importorskip("physicsnemo")
    if request.node.originalname in (
        "test_atlas_crps_iter",
        "test_atlas_crps_input_coords",
        "test_atlas_crps_output_coords",
    ):
        monkeypatch.setattr(AtlasCRPS, "__init__", inspect.unwrap(AtlasCRPS.__init__))


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
def test_atlas_crps_call(time, device, batch_size, atlas_crps_test_components):
    """Test AtlasCRPS __call__ method with different times and devices."""
    p = AtlasCRPS(**atlas_crps_test_components).to(device)

    dc = {d: p.input_coords().coords[d].values for d in ("lat", "lon")}
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x = fetch_data(
        r, time, variable, lead_time, device=device, delta_t=np.timedelta64(1, "h")
    )
    x.attrs.update(earth2studio_grid_id="latlon-0.25deg", earth2studio_crs="EPSG:4326")
    x = x.expand_dims(batch=np.arange(batch_size))
    coords = x
    out = p(x)
    out_coords = out.coords

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

    assert out.dims == ("batch", "time", "lead_time", "variable", "lat", "lon")


@pytest.mark.parametrize(
    "ensemble",
    [1, 2],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_atlas_crps_iter(ensemble, atlas_crps_test_components, device):
    """Test AtlasCRPS iterator for autoregressive predictions."""
    time = np.array([np.datetime64("1993-04-05T00:00")])

    p = AtlasCRPS(**atlas_crps_test_components).to(device)

    dc = {d: p.input_coords().coords[d].values for d in ("lat", "lon")}
    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x = fetch_data(
        r, time, variable, lead_time, device=device, delta_t=np.timedelta64(1, "h")
    )
    x.attrs.update(earth2studio_crs="EPSG:4326", source="fixture")
    x = x.expand_dims(ensemble=np.arange(ensemble))
    x = x.expand_dims(sample=1, axis=1).rename("weather")
    x.encoding = {"source": "fixture"}
    original = x.copy(deep=True)
    tensor = x.e2s.to_torch()[0][0, 0, 0].to(device)
    core_coords = {d: x.coords[d].values for d in x.dims if d in x.coords}
    expected1, latent = p._forward(tensor, core_coords)
    core_coords["lead_time"] = core_coords["lead_time"] + p.DT
    expected2, _ = p._forward(
        torch.cat((tensor[-1:], expected1), dim=0), core_coords, latent
    )
    p_iter = p.create_iterator(x)

    if not isinstance(time, Iterable):
        time = [time]

    # Get generator
    out = next(p_iter)
    # First output should be the latest lead time from input
    xr.testing.assert_identical(out, x.isel(lead_time=slice(-1, None)))
    initial = out
    retained = []

    for i, out in enumerate(p_iter):
        out_coords = out.coords
        assert len(out.shape) == 7
        assert out.shape[0] == ensemble
        assert (
            out_coords["variable"] == p.output_coords(p.input_coords())["variable"]
        ).all()
        assert (out_coords["time"] == time).all()
        assert out_coords["lead_time"][0] == np.timedelta64(6 * (i + 1), "h")

        assert out.dims == x.dims and "sample" not in out.coords
        assert out.name == x.name and out.encoding == x.encoding
        assert out.e2s.to_torch()[0].device == torch.device(device)
        retained.append((out, out.copy(deep=True)))
        if i < 2:
            torch.testing.assert_close(
                out.e2s.to_torch()[0][0, 0, 0], (expected1, expected2)[i]
            )

        if i > 3:
            break
    for out, saved in retained:
        xr.testing.assert_identical(out, saved)
    xr.testing.assert_identical(x, original)
    xr.testing.assert_identical(initial, original.isel(lead_time=slice(-1, None)))
    p_iter.close()
    calls = []

    def front(value):
        calls.append("front")
        value.data += 1
        return value

    def rear(value):
        calls.append("rear")
        value = value.rename(None)
        value.attrs.pop("source", None)
        value.encoding.clear()
        return value

    p.front_hook, p.rear_hook = front, rear
    p(x)
    assert calls == []
    iterator = p.create_iterator(x)
    next(iterator)
    first = next(iterator)
    saved = first.copy(deep=True)
    second = next(iterator)
    assert calls == ["front", "rear", "front", "rear"]
    assert (
        second.name is None and "source" not in second.attrs and second.encoding == {}
    )
    xr.testing.assert_identical(first, saved)
    xr.testing.assert_identical(x, original)
    iterator.close()
    p.clear_hooks()

    def preprocess(value, dates):
        value.add_(1)
        return value, value

    p.model_processor.preprocess_input = preprocess
    p(x)
    xr.testing.assert_identical(x, original)


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(720)}),
        OrderedDict({"lat": np.random.randn(720), "phoo": np.random.randn(1440)}),
        OrderedDict({"lat": np.random.randn(720), "lon": np.random.randn(1)}),
    ],
)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_atlas_crps_exceptions(dc, atlas_crps_test_components, device):
    """Test that AtlasCRPS raises exceptions for invalid coordinates."""
    time = np.array([np.datetime64("1993-04-05T00:00")])

    p = AtlasCRPS(**atlas_crps_test_components).to(device)

    # Initialize Data Source with invalid coordinates
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x = fetch_data(
        r, time, variable, lead_time, device=device, delta_t=np.timedelta64(1, "h")
    )

    with pytest.raises((KeyError, ValueError, RuntimeError)):
        p(x)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_atlas_crps_prep_next_input(atlas_crps_test_components, batch_size, device):
    """Test AtlasCRPS prep_next_input method for autoregressive stepping.

    The prep_next_input method should:
    1. Take the prediction at t+6h and place it as the latest input
    2. Shift the previous latest input (t=0) to the earlier position (t-6h)
    3. Update lead times by +6h
    """
    p = AtlasCRPS(**atlas_crps_test_components).to(device)

    # Create input data with 2 lead times
    time_steps = 2
    n_vars = 75
    lat = 721
    lon = 1440

    # Input state at t-6h and t=0
    x = torch.randn(batch_size, 1, time_steps, n_vars, lat, lon, device=device)
    coords = coord_array_like(
        p.input_coords(),
        {
            "batch": np.arange(batch_size),
            "time": np.array([np.datetime64("2020-01-01T00:00")]),
        },
    )
    x = from_torch(x, coords)

    # Prediction at t+6h (output has shape [batch, 1, n_vars, lat, lon])
    x_pred = torch.randn(batch_size, 1, 1, n_vars, lat, lon, device=device)
    coords_pred = p.output_coords(coords)
    x_pred = from_torch(x_pred, coords_pred)

    # Call prep_next_input
    x_next = p.prep_next_input(x_pred, x)
    coords_next = x_next.coords

    # Check that x_next has the correct shape
    assert x_next.shape == x.shape

    # Check that the latest lead time contains the prediction
    xr.testing.assert_equal(x_next.isel(lead_time=slice(-1, None)), x_pred)

    # Check that the earlier lead time contains the previous latest
    xr.testing.assert_equal(
        x_next.isel(lead_time=slice(0, 1)), x.isel(lead_time=slice(-1, None))
    )

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
def test_atlas_crps_prep_next_input_with_ensemble(atlas_crps_test_components, device):
    """Test prep_next_input with ensemble dimension."""
    p = AtlasCRPS(**atlas_crps_test_components).to(device)

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
    coords = coord_array_like(
        p.input_coords(),
        {
            "batch": np.arange(batch_size),
            "time": np.array([np.datetime64("2020-01-01T00:00")]),
        },
    )
    coords = coord_array(
        ("ensemble", *coords.dims),
        {"ensemble": np.arange(ensemble_size), **dict(coords.coords)},
        attrs=coords.attrs,
    )
    x = from_torch(x, coords)

    # Prediction at t+6h
    x_pred = torch.randn(
        ensemble_size, batch_size, 1, 1, n_vars, lat, lon, device=device
    )
    coords_pred = p.output_coords(coords)
    x_pred = from_torch(x_pred, coords_pred)

    # Call prep_next_input
    x_next = p.prep_next_input(x_pred, x)
    coords_next = x_next.coords

    # Check shapes
    assert x_next.shape == x.shape

    # Check that sliding window works correctly with ensemble dimension
    xr.testing.assert_equal(x_next.isel(lead_time=slice(-1, None)), x_pred)
    xr.testing.assert_equal(
        x_next.isel(lead_time=slice(0, 1)), x.isel(lead_time=slice(-1, None))
    )

    # Check ensemble coordinate is preserved
    assert np.array_equal(coords_next["ensemble"], coords["ensemble"])


def test_atlas_crps_conformance(atlas_crps_test_components):
    """Check the mock AtlasCRPS model against the Earth2Studio model contract.

    AtlasCRPS does not currently declare `stochastic` or implement `set_rng()`
    (see dev/spec/MODEL_CONTRACT_SPEC.md's Migration table: it has no seeding
    mechanism today and needs one added, forked). Until that lands, the contract
    checker treats it as a non-stochastic model, so this test only exercises the
    structural/coordinate rules against the deterministic mock.

    Note: `torch-harmonics` (required by the `atlas` extra) failed to build in
    every environment (a broken local C++ toolchain, unrelated to this wrapper),
    so this assertion could not be executed against real dependencies everywhere;
    it is expected to hold based on static review of AtlasCRPS's hook wiring and
    the deterministic Phoo forward pass above.
    """
    p = AtlasCRPS(**atlas_crps_test_components)
    assert check_prognostic_contract(p) == [
        "P14: model does not declare itself stochastic"
    ]


def test_atlas_crps_input_coords(atlas_crps_test_components):
    """Test that input_coords returns expected coordinate system."""
    p = AtlasCRPS(**atlas_crps_test_components)
    coords = p.input_coords()
    assert (
        coords.data.nbytes == 0
        and coords.attrs["earth2studio_grid_id"] == "latlon-0.25deg"
    )

    # Check expected keys
    assert coords.dims == ("batch", "time", "lead_time", "variable", "lat", "lon")

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


def test_atlas_crps_output_coords(atlas_crps_test_components):
    """Test that output_coords returns expected coordinate system."""
    p = AtlasCRPS(**atlas_crps_test_components)
    input_coords = p.input_coords()
    output_coords = p.output_coords(input_coords)

    # Check expected keys
    assert output_coords.dims == input_coords.dims

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
def test_atlas_crps_package(device):
    """Test that AtlasCRPS loads from package and runs a forward pass."""
    torch.cuda.empty_cache()

    model = AtlasCRPS.load_model(AtlasCRPS.load_default_package()).to(device)

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

    input_coords = coord_array_like(
        input_coords, {"batch": np.arange(batch_size), "time": time}
    )
    x = from_torch(x, input_coords)
    output = model(x)
    output_coords = output.coords
    expected_coords = model.output_coords(input_coords)

    assert output.shape == (
        batch_size,
        len(time),
        len(expected_coords["lead_time"]),
        len(variable),
        lat,
        lon,
    )
    for key in expected_coords.coords:
        np.testing.assert_array_equal(output_coords[key], expected_coords.coords[key])
