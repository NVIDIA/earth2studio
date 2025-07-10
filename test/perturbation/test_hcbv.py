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

import datetime
from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.data import Constant, Random
from earth2studio.models.batch import batch_coords, batch_func
from earth2studio.models.px.persistence import Persistence
from earth2studio.perturbation import (
    Brown,
    Gaussian,
    HemisphericCentredBredVector,
)
from earth2studio.utils.type import CoordSystem


# Fake PX model
@pytest.fixture
def model():
    class FooModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("scale", torch.Tensor([0.1]))
            self.index = 0
            self._input_coords = None

        def input_coords(self):
            return self._input_coords

        @batch_coords()
        def output_coords(self, input_coords: CoordSystem):
            output_coords = input_coords.copy()
            output_coords["lead_time"] = np.array([np.timedelta64(1, "s")])
            return output_coords

        @batch_func()
        def forward(self, x, coords):
            self.index += 1
            return self.scale * x, coords

    return FooModel()


@pytest.mark.parametrize(
    "time",
    [
        datetime.datetime(year=1971, month=6, day=1, hour=6),
        datetime.datetime(year=2021, month=11, day=23, hour=12),
    ],
)
@pytest.mark.parametrize("variable", [["tcwv"], ["tp", "u200", "z500"]])
@pytest.mark.parametrize(
    "amplitude,steps,batch",
    [[1.0, 5, 2], [2.0, 3, 4], [0.3, 2, 2]],
)
@pytest.mark.parametrize(
    "seeding_perturbation_method",
    [Brown(), Gaussian()],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
def test_hem_centered_bred(
    model, time, variable, amplitude, steps, batch, seeding_perturbation_method, device
):
    if amplitude > 1.0:
        amplitude = torch.Tensor([amplitude] * len(variable))[:, None, None]

    # Domain coordinates
    dc = OrderedDict(
        [
            ("lat", np.arange(16)),
            ("lon", np.arange(16)),
        ]
    )
    fc = OrderedDict(
        [
            ("batch", np.empty(0)),
            ("lead_time", np.array([np.timedelta64(0, "h")])),
            ("variable", np.array(variable)),
        ]
    )
    fc.update(dc)
    model._input_coords = fc
    model = model.to(device)
    model.index = 0

    # Initialize Data Source and input tensor
    data_source = Random(dc)
    x = torch.randn(batch, 1, 1, len(variable), 16, 16).to(device)
    coords = OrderedDict(
        [
            ("ensemble", np.arange(batch)),
            ("time", np.array([time])),
            ("lead_time", np.array([np.timedelta64(0, "h")])),
            ("variable", np.array(variable)),
        ]
    )
    coords.update(dc)

    prtb = HemisphericCentredBredVector(
        model=model,
        data=data_source,
        seeding_perturbation_method=seeding_perturbation_method,
        noise_amplitude=amplitude,
        integration_steps=steps,
    )
    xout, coords = prtb(x, coords)
    dx = xout - x

    # Don't have a good statistical test for this at the moment
    assert dx.shape == x.shape
    assert dx.device == x.device
    assert (
        model.index == steps * 2
    )  # x2 here because model forward twice each bred step

    # Validate that the clip variables are non-negative
    for var in range(len(variable)):
        if variable[var] == "tcwv" or variable[var] == "tp":
            assert (xout[:, :, :, var, :, :] >= 0).all()
        else:
            assert not (xout[:, :, :, var, :, :] >= 0).all()

    # Test data source which needs a map_coords
    dc_diff = OrderedDict(
        [
            ("lat", np.arange(20)),
            ("lon", np.arange(20)),
        ]
    )
    data_source = Random(dc_diff)
    prtb = HemisphericCentredBredVector(
        model=model,
        data=data_source,
        seeding_perturbation_method=seeding_perturbation_method,
        noise_amplitude=amplitude,
        integration_steps=steps,
    )
    xout, coords = prtb(x, coords)
    # Don't have a good statistical test for this at the moment
    assert xout.shape == x.shape
    assert xout.device == x.device


@pytest.mark.parametrize(
    "steps",
    [2, 5],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="cuda missing"
            ),
        ),
    ],
)
def test_hem_centered_bred_batching(steps, device):
    time = datetime.datetime(year=1971, month=6, day=1, hour=6)
    variable = ["u10m"]
    seeding_perturbation_method = Gaussian()

    # Domain coordinates
    dc = OrderedDict(
        [
            ("lat", np.arange(16)),
            ("lon", np.arange(16)),
        ]
    )
    fc = OrderedDict(
        [
            ("batch", np.empty(0)),
            ("lead_time", np.array([np.timedelta64(0, "h")])),
            ("variable", np.array(variable)),
        ]
    )
    fc.update(dc)
    model = Persistence(variable, dc)
    # model._input_coords = fc
    model = model.to(device)

    # Run twice with batch size 1 and two repeated calls give the centered perturbations
    # given a constant data source
    data_source = Constant(dc, 0)
    x = torch.randn(1, 1, 1, len(variable), 16, 16).to(device)
    x2 = torch.randn(1, 1, 1, len(variable), 16, 16).to(device)
    coords = OrderedDict(
        [
            ("ensemble", np.arange(1)),
            ("time", np.array([time])),
            ("lead_time", np.array([np.timedelta64(0, "h")])),
            ("variable", np.array(variable)),
        ]
    )
    coords.update(dc)
    torch.manual_seed(42)
    prtb = HemisphericCentredBredVector(
        model=model,
        data=data_source,
        seeding_perturbation_method=seeding_perturbation_method,
        noise_amplitude=1.0,
        integration_steps=steps,
    )
    xout1, coords = prtb(x, coords)
    xout2, coords = prtb(x, coords)

    assert torch.allclose(xout1, -xout2)

    # Test that two repeat odd runs is equal to a single even
    # Should be invariant to x input, this uses a data source
    torch.manual_seed(42)
    prtb = HemisphericCentredBredVector(
        model=model,
        data=data_source,
        seeding_perturbation_method=seeding_perturbation_method,
        noise_amplitude=1.0,
        integration_steps=steps,
    )
    coords["ensemble"] = np.arange(3)
    xout1a, coords = prtb(x, coords)
    xout1b, coords = prtb(x2, coords)

    torch.manual_seed(42)
    prtb = HemisphericCentredBredVector(
        model=model,
        data=data_source,
        seeding_perturbation_method=seeding_perturbation_method,
        noise_amplitude=1.0,
        integration_steps=steps,
    )
    coords["ensemble"] = np.arange(6)
    xout2, coords = prtb(x, coords)

    # Note that the last two batches are not gaurenteed to be the same here
    # since the batch size 3 run requires two different generators
    assert torch.allclose(torch.cat([xout1a, xout1b])[:4], xout2[:4])
