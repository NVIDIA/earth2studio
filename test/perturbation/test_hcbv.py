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

from earth2studio.data import Random
from earth2studio.models.batch import batch_coords, batch_func
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
def test_hem_cen_bred_vec(
    model, time, variable, amplitude, steps, batch, seeding_perturbation_method, device
):
    if amplitude > 1.0:
        amplitude = torch.Tensor([amplitude] * len(variable))[:, None, None]

    # Domain coordinates
    dc = OrderedDict(
        [
            ("lat", [f"{nn}" for nn in range(16)]),
            ("lon", [f"{nn}" for nn in range(16)]),
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
    assert model.index == steps * 2

    # Validate that the clip variables are non-negative
    for var in range(len(variable)):
        if variable[var] == "tcwv" or variable[var] == "tp":
            assert (xout[:, :, :, var, :, :] >= 0).all()
        else:
            assert not (xout[:, :, :, var, :, :] >= 0).all()
