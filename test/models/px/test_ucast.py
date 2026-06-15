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
from earth2studio.models.px import UCast
from earth2studio.models.px.ucast import VARIABLES
from earth2studio.utils import handshake_dim
from earth2studio.utils.checkpoint import Checkpoint


class PhooUCastModel(torch.nn.Module):
    """Test double for the U-CAST core model."""

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


class SstResidualUCastModel(PhooUCastModel):
    """Test double that predicts a non-zero SST residual."""

    def forward(
        self,
        inputs: torch.Tensor,
        dynamical_condition: torch.Tensor | None = None,
        static_condition: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = super().forward(inputs, dynamical_condition, static_condition)
        out[:, VARIABLES.index("sst")] = 1.0
        return out


def _make_ucast_model(stochastic: bool = False) -> UCast:
    n_variables = len(VARIABLES)
    return UCast(
        model=PhooUCastModel(),
        center=torch.zeros(n_variables),
        scale=torch.ones(n_variables),
        residual_scale=torch.ones(n_variables),
        static_condition=torch.zeros(2, 121, 240),
        sst_fill_value=0.0,
        stochastic=stochastic,
    )


@pytest.fixture(scope="function")
def ucast_model() -> UCast:
    return _make_ucast_model()


@pytest.fixture(scope="function")
def model() -> UCast:
    package = UCast.load_default_package()
    return UCast.load_model(package)


def _input(
    ucast_model: UCast,
    time: np.ndarray,
    device: str = "cpu",
) -> tuple[torch.Tensor, OrderedDict]:
    dc = ucast_model.input_coords()
    del dc["batch"]
    del dc["time"]
    del dc["lead_time"]
    del dc["variable"]

    ds = Random(dc)
    lead_time = ucast_model.input_coords()["lead_time"]
    variable = ucast_model.input_coords()["variable"]
    return fetch_data(ds, time, variable, lead_time, device=device)


def _check_output_coords(
    out_coords: OrderedDict,
    coords: OrderedDict,
    lead_time: np.timedelta64,
) -> None:
    np.testing.assert_array_equal(out_coords["time"], coords["time"])
    np.testing.assert_array_equal(out_coords["lead_time"], np.array([lead_time]))
    np.testing.assert_array_equal(out_coords["variable"], np.array(VARIABLES))
    assert out_coords["lat"][0] == 90
    assert out_coords["lat"][-1] == -90
    handshake_dim(out_coords, "time", 0)
    handshake_dim(out_coords, "lead_time", 1)
    handshake_dim(out_coords, "variable", 2)
    handshake_dim(out_coords, "lat", 3)
    handshake_dim(out_coords, "lon", 4)


@pytest.mark.parametrize(
    "time",
    [
        np.array([np.datetime64("2020-01-01T00:00")]),
        np.array(
            [
                np.datetime64("2020-01-01T00:00"),
                np.datetime64("2020-01-02T00:00"),
            ]
        ),
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU"),
        ),
    ],
)
def test_ucast_call(ucast_model: UCast, time: np.ndarray, device: str) -> None:
    ucast_model = ucast_model.to(device)
    x, coords = _input(ucast_model, time, device=device)

    out, out_coords = ucast_model(x, coords)

    assert out.shape == torch.Size([len(time), 1, len(VARIABLES), 121, 240])
    assert torch.allclose(out, x[:, -1:])
    _check_output_coords(out_coords, coords, np.timedelta64(12, "h"))


@pytest.mark.parametrize("ensemble", [1, 2])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU"),
        ),
    ],
)
def test_ucast_iter(ucast_model: UCast, ensemble: int, device: str) -> None:
    ucast_model = ucast_model.to(device)
    time = np.array([np.datetime64("2020-01-01T00:00")])
    x, coords = _input(ucast_model, time, device=device)

    x = x.unsqueeze(0).repeat(ensemble, *([1] * x.ndim))
    coords.update({"ensemble": np.arange(ensemble)})
    coords.move_to_end("ensemble", last=False)

    iterator = ucast_model.create_iterator(x, coords)
    assert isinstance(iterator, Iterable)

    initial, initial_coords = next(iterator)
    assert torch.allclose(initial, x[:, :, -1:])
    np.testing.assert_array_equal(
        initial_coords["lead_time"], np.array([np.timedelta64(0, "h")])
    )

    for i, (out, out_coords) in enumerate(iterator):
        assert out.shape == torch.Size(
            [ensemble, len(time), 1, len(VARIABLES), 121, 240]
        )
        assert torch.allclose(out, x[:, :, -1:])
        assert (out_coords["ensemble"] == np.arange(ensemble)).all()
        assert (out_coords["time"] == time).all()
        assert out_coords["lead_time"][0] == np.timedelta64(12 * (i + 1), "h")
        handshake_dim(out_coords, "ensemble", 0)
        handshake_dim(out_coords, "time", 1)
        handshake_dim(out_coords, "lead_time", 2)
        handshake_dim(out_coords, "variable", 3)
        handshake_dim(out_coords, "lat", 4)
        handshake_dim(out_coords, "lon", 5)

        if i >= 4:
            break


def test_ucast_checkpoint_state_round_trip(tmp_path) -> None:
    time = np.array([np.datetime64("2020-01-01T00:00")])
    source_model = _make_ucast_model()
    x, coords = _input(source_model, time)
    x = x.unsqueeze(0)
    coords.update({"ensemble": np.array([0])})
    coords.move_to_end("ensemble", last=False)

    checkpoint = Checkpoint(
        "ucast",
        path=tmp_path,
        mode="append",
        flush_interval=1,
        state_policy="full",
    )
    with checkpoint as ckpt:
        model = _make_ucast_model()
        iterator = model.create_iterator(x, coords)
        initial, initial_coords = next(iterator)
        first, first_coords = next(iterator)
        ckpt.write(lead_time=first_coords["lead_time"][-1])
        ckpt.flush()

    np.testing.assert_array_equal(
        initial_coords["lead_time"], np.array([np.timedelta64(0, "h")])
    )
    np.testing.assert_array_equal(
        first_coords["lead_time"], np.array([np.timedelta64(12, "h")])
    )
    assert torch.allclose(initial, x[:, :, -1:])
    assert torch.allclose(first, x[:, :, -1:])

    checkpoint = Checkpoint(
        "ucast",
        path=tmp_path,
        mode="append",
        flush_interval=1,
        state_policy="full",
    )
    with checkpoint.select(-1):
        model = _make_ucast_model()
        resumed, resumed_coords = next(model.create_iterator(x, coords))
        assert model.checkpoint.checkpoint_state_loaded

    np.testing.assert_array_equal(
        resumed_coords["lead_time"], np.array([np.timedelta64(24, "h")])
    )
    assert torch.allclose(resumed, x[:, :, -1:])


def test_ucast_iter_uses_internal_normalized_state() -> None:
    n_variables = len(VARIABLES)
    ucast_model = UCast(
        model=SstResidualUCastModel(),
        center=torch.zeros(n_variables),
        scale=torch.ones(n_variables),
        residual_scale=torch.ones(n_variables),
        static_condition=torch.zeros(2, 121, 240),
        sst_fill_value=-1.0,
        stochastic=False,
    )
    time = np.array([np.datetime64("2020-01-01T00:00")])
    x, coords = _input(ucast_model, time)
    sst_index = VARIABLES.index("sst")
    x[:, -1, sst_index, 0, 0] = torch.nan

    iterator = ucast_model.create_iterator(x, coords)
    next(iterator)
    first, _ = next(iterator)
    second, _ = next(iterator)

    assert first[0, 0, sst_index, 0, 0] == ucast_model.sst_fill_value
    assert second[0, 0, sst_index, 0, 0] == ucast_model.sst_fill_value


@pytest.mark.parametrize(
    "coords_update",
    [
        {"lead_time": np.array([np.timedelta64(-6, "h"), np.timedelta64(0, "h")])},
        {"variable": np.array(["bad_variable", *VARIABLES[1:]])},
        {"lat": np.linspace(-90, 90, 121)},
        {"lon": np.linspace(0, 360, 241, endpoint=False)},
    ],
)
def test_ucast_exceptions(ucast_model: UCast, coords_update: dict) -> None:
    time = np.array([np.datetime64("2020-01-01T00:00")])
    x, coords = _input(ucast_model, time)
    coords.update(coords_update)

    with pytest.raises((KeyError, ValueError)):
        ucast_model(x, coords)


@pytest.mark.package
@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU"),
        ),
    ],
)
def test_ucast_package(model: UCast, device: str) -> None:
    torch.cuda.empty_cache()
    time = np.array([np.datetime64("2020-01-01T00:00")])
    ucast_model = model.to(device)
    _, coords = _input(ucast_model, time, device=device)
    x = ucast_model.center.reshape(1, 1, len(VARIABLES), 1, 1).expand(
        len(time), coords["lead_time"].shape[0], len(VARIABLES), 121, 240
    )
    x = x.contiguous()

    out, out_coords = ucast_model(x, coords)

    assert out.shape == torch.Size([len(time), 1, len(VARIABLES), 121, 240])
    assert torch.isfinite(out).all()
    _check_output_coords(out_coords, coords, np.timedelta64(12, "h"))
