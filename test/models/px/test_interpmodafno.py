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
from collections.abc import Iterable

import numpy as np
import pytest
import torch

from earth2studio.data import Random, fetch_data
from earth2studio.models.px import SFNO, InterpModAFNO
from earth2studio.models.px.persistence import Persistence
from earth2studio.utils import handshake_dim


class PhooSFNOModel(torch.nn.Module):
    """Mock SFNO model for testing."""

    def forward(self, x, t, normalized_data=True):
        return x


class PhooInterpolationModel(torch.nn.Module):
    """Mock interpolation model for testing."""

    def forward(self, x, t_norm):
        return x[:, :73]


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
def test_forecast_interpolation_call(time, device):
    """Test basic forward pass of InterpModAFNO model."""
    # Set up base SFNO model
    sfno_model = PhooSFNOModel()
    center = torch.zeros(1, 73, 1, 1)
    scale = torch.ones(1, 73, 1, 1)
    base_model = SFNO(sfno_model).to(device)

    # Set up interpolation model
    interp_model = PhooInterpolationModel()
    geop = torch.zeros(1, 1, 720, 1440)  # Mock geopotential height
    lsm = torch.zeros(1, 1, 720, 1440)  # Mock land-sea mask

    model = InterpModAFNO(
        interp_model=interp_model,
        center=center,
        scale=scale,
        geop=geop,
        lsm=lsm,
        px_model=base_model,
        num_interp_steps=6,
    ).to(device)

    # Create domain coordinates
    dc = {k: model.input_coords()[k] for k in ["lat", "lon"]}

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Run forward pass
    out, out_coords = model(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    # Verify output shape and coordinates
    assert out.shape == torch.Size([len(time), 1, 73, 720, 1440])
    assert (out_coords["variable"] == model.output_coords(coords)["variable"]).all()
    assert (out_coords["time"] == time).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)


@pytest.mark.parametrize(
    "ensemble",
    [1, 2],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_forecast_interpolation_iter(ensemble, device):
    """Test iteration functionality of InterpModAFNO model."""
    time = np.array([np.datetime64("1993-04-05T00:00")])

    # Set up base SFNO model
    sfno_model = PhooSFNOModel()
    center = torch.zeros(1, 73, 1, 1)
    scale = torch.ones(1, 73, 1, 1)
    base_model = SFNO(sfno_model).to(device)

    # Set up interpolation model
    interp_model = PhooInterpolationModel()
    geop = torch.zeros(1, 1, 720, 1440)  # Mock geopotential height
    lsm = torch.zeros(1, 1, 720, 1440)  # Mock land-sea mask

    model = InterpModAFNO(
        interp_model=interp_model,
        center=center,
        scale=scale,
        geop=geop,
        lsm=lsm,
        px_model=base_model,
        num_interp_steps=6,
    ).to(device)

    # Create domain coordinates
    dc = {k: model.input_coords()[k] for k in ["lat", "lon"]}

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Add ensemble to front
    x = x.unsqueeze(0).repeat(ensemble, 1, 1, 1, 1, 1)
    coords.update({"ensemble": np.arange(ensemble)})
    coords.move_to_end("ensemble", last=False)

    # Create iterator
    model_iter = model.create_iterator(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    # Get generator
    next(model_iter)  # Skip first which should return the input

    # Test interpolation steps
    for i, (out, out_coords) in enumerate(model_iter):

        # Check output shape
        assert len(out.shape) == 6
        assert out.shape == torch.Size([ensemble, len(time), 1, 73, 720, 1440])

        # Check coordinates
        assert (
            out_coords["variable"]
            == model.output_coords(model.input_coords())["variable"]
        ).all()
        assert (out_coords["ensemble"] == np.arange(ensemble)).all()

        # Check lead time - should be 1 hour increments due to interpolation
        assert out_coords["lead_time"][0] == np.timedelta64(1 * (i + 1), "h")

        # Break after testing a few steps
        if i > 10:
            break


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict({"lat": np.random.randn(720)}),
        OrderedDict({"lat": np.random.randn(720), "phoo": np.random.randn(1440)}),
        OrderedDict({"lat": np.random.randn(720), "lon": np.random.randn(1)}),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_forecast_interpolation_exceptions(dc, device):
    """Test exception handling for invalid inputs in InterpModAFNO model."""
    time = np.array([np.datetime64("1993-04-05T00:00")])

    # Set up base SFNO model
    sfno_model = PhooSFNOModel()
    center = torch.zeros(1, 73, 1, 1)
    scale = torch.ones(1, 73, 1, 1)
    base_model = SFNO(sfno_model).to(device)

    # Set up interpolation model
    interp_model = PhooInterpolationModel()
    geop = torch.zeros(1, 1, 720, 1440)  # Mock geopotential height
    lsm = torch.zeros(1, 1, 720, 1440)  # Mock land-sea mask

    model = InterpModAFNO(
        interp_model=interp_model,
        center=center,
        scale=scale,
        geop=geop,
        lsm=lsm,
        px_model=base_model,
        num_interp_steps=6,
    ).to(device)

    # Initialize Data Source with invalid domain coordinates
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Expect an exception when running the model with invalid inputs
    with pytest.raises((KeyError, ValueError)):
        model(x, coords)

    model = InterpModAFNO(
        interp_model=interp_model,
        center=center,
        scale=scale,
        geop=geop,
        lsm=lsm,
        num_interp_steps=6,
    ).to(device)
    with pytest.raises(ValueError):
        model.input_coords()


@pytest.fixture(scope="function")
def model(model_cache_context) -> InterpModAFNO:

    from earth2studio.models.px.interpmodafno import VARIABLES

    base_model = Persistence(
        variable=VARIABLES,
        domain_coords={
            "lat": np.linspace(90.0, -90.0, 720, endpoint=False),
            "lon": np.linspace(0, 360, 1440, endpoint=False),
        },
    )
    # Test only on cuda device
    with model_cache_context():
        # Load the interpolation model
        interp_package = InterpModAFNO.load_default_package()
        model = InterpModAFNO.load_model(interp_package, px_model=base_model)
        return model


@pytest.mark.ci_cache
@pytest.mark.timeout(360)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_forecast_interpolation_package(device, model):
    """Test loading and using the InterpModAFNO model from a package."""
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("1993-04-05T00:00")])

    # Test the cached model package
    model = model.to(device)

    # Create domain coordinates
    dc = {k: model.input_coords()[k] for k in ["lat", "lon"]}

    # Initialize Data Source
    r = Random(dc)

    # Get Data and convert to tensor, coords
    lead_time = model.input_coords()["lead_time"]
    variable = model.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    # Run forward pass
    out, out_coords = model(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    # Verify output shape and coordinates
    assert out.shape == torch.Size([len(time), 1, 73, 720, 1440])
    assert (out_coords["variable"] == model.output_coords(coords)["variable"]).all()
    handshake_dim(out_coords, "lon", 4)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)
