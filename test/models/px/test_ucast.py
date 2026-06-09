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

from earth2studio.models.px import UCast
from earth2studio.models.px.ucast import VARIABLES
from earth2studio.utils import handshake_dim


class ZeroResidualUCast(torch.nn.Module):
    """Tiny test double for the U-CAST core model."""

    def forward(
        self,
        inputs: torch.Tensor,
        dynamical_condition: torch.Tensor | None = None,
        static_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert dynamical_condition is not None
        assert static_condition is not None
        assert inputs.shape[1] == len(VARIABLES) * 2
        assert dynamical_condition.shape[1] == 4
        assert static_condition.shape[1] == 2
        return torch.zeros(
            inputs.shape[0],
            len(VARIABLES),
            inputs.shape[-2],
            inputs.shape[-1],
            device=inputs.device,
            dtype=inputs.dtype,
        )


@pytest.fixture
def ucast_model() -> UCast:
    n_variables = len(VARIABLES)
    return UCast(
        model=ZeroResidualUCast(),
        center=torch.zeros(n_variables),
        scale=torch.ones(n_variables),
        residual_scale=torch.ones(n_variables),
        static_condition=torch.zeros(2, 121, 240),
        sst_fill_value=0.0,
        stochastic=False,
    )


def _input(ucast_model: UCast) -> tuple[torch.Tensor, OrderedDict]:
    coords = ucast_model.input_coords()
    del coords["batch"]
    coords["time"] = np.array([np.datetime64("2020-01-01T00:00")])
    x = torch.randn(
        coords["time"].shape[0],
        coords["lead_time"].shape[0],
        coords["variable"].shape[0],
        coords["lat"].shape[0],
        coords["lon"].shape[0],
    )
    return x, coords


def test_ucast_call(ucast_model: UCast) -> None:
    x, coords = _input(ucast_model)

    out, out_coords = ucast_model(x, coords)

    assert out.shape == torch.Size([1, 1, len(VARIABLES), 121, 240])
    assert torch.allclose(out, x[:, -1:])
    assert (out_coords["time"] == coords["time"]).all()
    assert out_coords["lead_time"][0] == np.timedelta64(12, "h")
    assert (out_coords["variable"] == np.array(VARIABLES)).all()
    handshake_dim(out_coords, "time", 0)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "lon", 4)


def test_ucast_iter_ensemble(ucast_model: UCast) -> None:
    x, coords = _input(ucast_model)
    x = x.unsqueeze(0).repeat(2, 1, 1, 1, 1, 1)
    coords.update({"ensemble": np.arange(2)})
    coords.move_to_end("ensemble", last=False)

    iterator = ucast_model.create_iterator(x, coords)
    initial, initial_coords = next(iterator)
    assert torch.allclose(initial, x[:, :, -1:])
    assert initial_coords["lead_time"][0] == np.timedelta64(0, "h")

    for step in range(3):
        out, out_coords = next(iterator)
        assert out.shape == torch.Size([2, 1, 1, len(VARIABLES), 121, 240])
        assert torch.allclose(out, x[:, :, -1:])
        assert out_coords["lead_time"][0] == np.timedelta64(12 * (step + 1), "h")
        handshake_dim(out_coords, "ensemble", 0)
        handshake_dim(out_coords, "time", 1)
        handshake_dim(out_coords, "lead_time", 2)
        handshake_dim(out_coords, "variable", 3)
        handshake_dim(out_coords, "lat", 4)
        handshake_dim(out_coords, "lon", 5)


@pytest.mark.parametrize(
    "coords_update",
    [
        {"lat": np.linspace(90, -90, 121)},
        {"lon": np.linspace(0, 360, 241, endpoint=False)},
    ],
)
def test_ucast_invalid_coords(ucast_model: UCast, coords_update: dict) -> None:
    x, coords = _input(ucast_model)
    coords.update(coords_update)

    with pytest.raises((KeyError, ValueError)):
        ucast_model(x, coords)
