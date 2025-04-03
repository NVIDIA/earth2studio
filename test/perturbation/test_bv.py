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
from typing import Any

import numpy as np
import pytest
import torch
from xarray import DataArray, concat

from earth2studio.data import fetch_data
from earth2studio.perturbation import BredVector, Brown, Gaussian, HemisphericCentredBredVector


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

        def output_coords(self, dummy):
            return {"lead_time": np.array([np.timedelta64(1, "s")])}

        def forward(self, x, coords):
            self.index += 1
            return self.scale * x, coords

    return FooModel()


@pytest.mark.parametrize(
    "x, coords",
    [
        [
            torch.randn(2, 16, 16, 16),
            OrderedDict([("a", []), ("variable", []), ("lat", []), ("lon", [])]),
        ],
        [
            torch.randn(2, 8, 32, 16),
            OrderedDict([("a", []), ("variable", []), ("lat", []), ("lon", [])]),
        ],
    ],
)
@pytest.mark.parametrize(
    "amplitude,steps,ensemble",
    [[1.0, 5, False], [1.0, 3, True], ["tensor", 2, True]],
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
def test_bred_vec(
    model, x, coords, amplitude, steps, ensemble, seeding_perturbation_method, device
):

    model = model.to(device)
    model.index = 0
    x = x.to(device)

    if amplitude == "tensor":
        amplitude = torch.randn(
            [x.shape[list(coords).index("variable")], 1, 1], device=device
        )

    prtb = BredVector(model, amplitude, steps, ensemble, seeding_perturbation_method)
    xout, coords = prtb(x, coords)
    dx = xout - x

    # Don't have a good statistical test for this at the moment
    assert dx.shape == x.shape
    assert dx.device == x.device
    assert model.index == steps + 1



class ConstRandom:
    def __init__(
        self,
        shape,
        coords,  #: OrderedDict[str, np.ndarray],
    ):
        self.xx = torch.randn(shape)
        self.coords = coords

    def __call__(
        self,
        time: Any = [np.datetime64("1066-11-14")],
        variable: Any = 0,
    ) -> DataArray:
        da = DataArray(data=self.xx, dims=list(self.coords), coords=self.coords)
        return concat([da] * len(time), dim="time")


@pytest.mark.parametrize(
    "shape",
    [(1, 16, 16, 16), (1, 8, 32, 8)],
)
@pytest.mark.parametrize(
    "amplitude,steps,batch_size",
    [[1.0, 5, 2], [1.0, 3, 4], [0.3, 2, 2]],
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
    model, shape, amplitude, steps, batch_size, seeding_perturbation_method, device
):
    coords = OrderedDict(
        [
            ("time", [f"{nn}" for nn in range(shape[0])]),
            ("variable", [f"{nn}" for nn in range(shape[1])]),
            ("lat", [f"{nn}" for nn in range(shape[2])]),
            ("lon", [f"{nn}" for nn in range(shape[3])]),
        ]
    )
    model._input_coords = coords.copy()
    model._input_coords.update([("lead_time", np.array([np.timedelta64(1, "s")]))])
    data_source = ConstRandom(coords=coords, shape=shape)
    x, _ = fetch_data(
        source=data_source,
        time=[np.datetime64("1784-10-08")],
        variable=coords["variable"],
        lead_time=np.array([np.timedelta64(1, "s")]),
        device=device,
    )

    batch_size = 2
    model = model.to(device)
    model.index = 0

    x = torch.stack([x] * batch_size)

    prtb = HemisphericCentredBredVector(
        model=model,
        data=data_source,
        time=["1066-11-14"],
        seeding_perturbation_method=seeding_perturbation_method,
        noise_amplitude=amplitude,
        integration_steps=steps,
    )
    xout, coords = prtb(x, coords)
    dx = xout - x

    # Don't have a good statistical test for this at the moment
    assert dx.shape == x.shape
    assert dx.device == x.device
    assert model.index == steps * batch_size

