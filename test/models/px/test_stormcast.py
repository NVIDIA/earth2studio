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

from earth2studio.data import HRRR, Random, fetch_data
from earth2studio.models.px import StormCast
from earth2studio.utils import handshake_dim


# Spoof models with same call signature
class PhooStormCastRegressionModel(torch.nn.Module):
    def __init__(self, out_vars=3):
        super().__init__()
        self.out_vars = out_vars

    def forward(self, x):
        return x[:, : self.out_vars, :, :]


class PhooStormCastDiffusionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma_min = 0.0
        self.sigma_max = 88.0

    def forward(self, x, noise, class_labels=None, condition=None):
        return x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)


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
def test_stormcast_call(time, device):

    # Spoof models
    regression = PhooStormCastRegressionModel()
    diffusion = PhooStormCastDiffusionModel()

    # Init data sources
    X_START, X_END = 579, 1219
    Y_START, Y_END = 273, 785
    nvar, nvar_cond = 3, 5
    dc = OrderedDict(
        [
            ("hrrr_y", HRRR.HRRR_Y[Y_START:Y_END]),
            ("hrrr_x", HRRR.HRRR_X[X_START:X_END]),
        ]
    )
    lat, lon = np.meshgrid(dc["hrrr_y"], dc["hrrr_x"], indexing="ij")

    r = Random(dc)
    r_condition = Random(
        OrderedDict(
            [
                ("lat", np.linspace(90, -90, num=181, endpoint=True)),
                ("lon", np.linspace(0, 360, num=360)),
            ]
        )
    )

    # Spoof variable names
    variables = np.array(["u%02d" % i for i in range(nvar)])

    # Init model with explicit conditioning data in constructor
    means = torch.zeros(1, nvar, 1, 1)
    stds = torch.ones(1, nvar, 1, 1)
    invariants = torch.randn(1, 2, lat.shape[0], lat.shape[1])
    conditioning_means = torch.randn(1, nvar_cond, 1, 1, device=device)
    conditioning_stds = torch.randn(1, nvar_cond, 1, 1, device=device)
    conditioning_variables = np.array(["u%02d" % i for i in range(nvar_cond)])
    p = StormCast(
        regression,
        diffusion,
        means,
        stds,
        invariants,
        variables=variables,
        conditioning_means=conditioning_means,
        conditioning_stds=conditioning_stds,
        conditioning_variables=conditioning_variables,
        conditioning_data_source=r_condition,
        sampler_args={"num_steps": 2},
    ).to(device)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size([len(time), 1, nvar, lat.shape[0], lat.shape[1]])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert np.all(out_coords["time"] == time)
    handshake_dim(out_coords, "hrrr_x", 4)
    handshake_dim(out_coords, "hrrr_y", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)


@pytest.mark.parametrize(
    "ensemble",
    [1, 2],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormcast_iter(ensemble, device):

    time = np.array([np.datetime64("2020-04-05T00:00")])

    # Spoof models
    regression = PhooStormCastRegressionModel()
    diffusion = PhooStormCastDiffusionModel()

    # Init data sources
    X_START, X_END = 32, 64
    Y_START, Y_END = 32, 128
    nvar, nvar_cond = 3, 5
    dc = OrderedDict(
        [
            ("hrrr_y", HRRR.HRRR_Y[Y_START:Y_END]),
            ("hrrr_x", HRRR.HRRR_X[X_START:X_END]),
        ]
    )
    lat, lon = np.meshgrid(dc["hrrr_y"], dc["hrrr_x"], indexing="ij")

    r = Random(dc)
    r_condition = Random(
        OrderedDict(
            [
                ("lat", np.linspace(90, -90, num=181, endpoint=True)),
                ("lon", np.linspace(0, 360, num=360)),
            ]
        )
    )

    # Init model with explicit conditioning data in constructor
    variables = np.array(["u%02d" % i for i in range(nvar)])
    means = torch.zeros(1, nvar, 1, 1)
    stds = torch.ones(1, nvar, 1, 1)
    invariants = torch.randn(1, 2, lat.shape[0], lat.shape[1])
    conditioning_means = torch.randn(1, nvar_cond, 1, 1, device=device)
    conditioning_stds = torch.randn(1, nvar_cond, 1, 1, device=device)
    conditioning_variables = np.array(["u%02d" % i for i in range(nvar_cond)])
    p = StormCast(
        regression,
        diffusion,
        means,
        stds,
        invariants,
        hrrr_lat_lim=(Y_START, Y_END),
        hrrr_lon_lim=(X_START, X_END),
        variables=variables,
        conditioning_means=conditioning_means,
        conditioning_stds=conditioning_stds,
        conditioning_variables=conditioning_variables,
        conditioning_data_source=r_condition,
        sampler_args={"num_steps": 2},
    ).to(device)

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
    next(p_iter)  # Skip first which should return the input
    for i, (out, out_coords) in enumerate(p_iter):
        assert len(out.shape) == 6
        assert out.shape == torch.Size(
            [ensemble, len(time), 1, nvar, lat.shape[0], lat.shape[1]]
        )
        assert (
            out_coords["variable"] == p.output_coords(p.input_coords())["variable"]
        ).all()
        assert (out_coords["ensemble"] == np.arange(ensemble)).all()
        assert out_coords["lead_time"][0] == np.timedelta64(i + 1, "h")

        if i > 5:
            break


@pytest.mark.parametrize(
    "dc",
    [
        OrderedDict(
            {
                "lat": np.random.randn(312, 640),
                "lon": np.random.randn(312, 640),
            }
        ),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_stormcast_exceptions(dc, device):
    time = np.array([np.datetime64("2020-04-05T00:00")])

    regression = PhooStormCastRegressionModel()
    diffusion = PhooStormCastDiffusionModel()

    # Build model with correct coords but no conditioning info
    X_START, X_END = 579, 1219
    Y_START, Y_END = 273, 785
    dc = OrderedDict(
        [
            ("hrrr_y", HRRR.HRRR_Y[Y_START:Y_END]),
            ("hrrr_x", HRRR.HRRR_X[X_START:X_END]),
        ]
    )
    lat, lon = np.meshgrid(dc["hrrr_y"], dc["hrrr_x"], indexing="ij")

    r = Random(dc)
    means = torch.zeros(1, 99, 1, 1)
    stds = torch.ones(1, 99, 1, 1)
    invariants = torch.randn(1, 2, 512, 640)
    p = StormCast(
        regression,
        diffusion,
        means,
        stds,
        invariants,
    ).to(device)

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    with pytest.raises(RuntimeError):
        # Calling with no conditioning info should fail
        p(x, coords)

    # Create iterator and consume first batch (initial condition)
    p_iter = p.create_iterator(x, coords)
    next(p_iter)
    with pytest.raises(ValueError):
        # Using the generator with no built-in conditioning should fail
        next(p_iter)


@pytest.fixture(scope="function")
def model() -> StormCast:
    package = StormCast.load_default_package()
    p = StormCast.load_model(package)
    return p


@pytest.mark.package
@pytest.mark.parametrize(
    "cond_dims",
    [["time", "variable", "lat", "lon"], ["variable", "time", "lat", "lon"]],
)
@pytest.mark.parametrize("device", ["cuda:0"])
def test_stormcast_package(cond_dims, device, model):
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("2020-04-05T00:00")])
    # Test the cached model package StormCast
    p = model.to(device)

    X_START, X_END = 579, 1219
    Y_START, Y_END = 273, 785
    dc = OrderedDict(
        [
            ("hrrr_y", HRRR.HRRR_Y[Y_START:Y_END]),
            ("hrrr_x", HRRR.HRRR_X[X_START:X_END]),
        ]
    )
    r = Random(dc)

    # returns dimensions in a different order
    class _RandomWithSpecifiedOrder(Random):
        def __call__(self, time, variable):
            x = super().__call__(time, variable)
            return x.transpose(*cond_dims)

    r_condition = _RandomWithSpecifiedOrder(
        OrderedDict(
            [
                ("lat", np.linspace(90, -90, num=721, endpoint=True)),
                ("lon", np.linspace(0, 360, num=1440)),
            ]
        )
    )

    # Manually set the condition data source (necessary as NGC package doesn't specify)
    p.conditioning_data_source = r_condition

    # Decrease the number of edm sampling steps to speed up the test
    p.sampler_args = {"num_steps": 2}

    # Get Data and convert to tensor, coords
    lead_time = p.input_coords()["lead_time"]
    variable = p.input_coords()["variable"]
    x, coords = fetch_data(r, time, variable, lead_time, device=device)

    out, out_coords = p(x, coords)

    if not isinstance(time, Iterable):
        time = [time]

    assert out.shape == torch.Size([len(time), 1, 99, 512, 640])
    assert (out_coords["variable"] == p.output_coords(coords)["variable"]).all()
    assert np.all(out_coords["time"] == time)
    handshake_dim(out_coords, "hrrr_x", 4)
    handshake_dim(out_coords, "hrrr_y", 3)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "time", 0)
