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
import xarray as xr

from earth2studio.data import Random, fetch_data
from earth2studio.models.conformance import check_prognostic_contract
from earth2studio.models.px import UCast
from earth2studio.models.px.ucast import VARIABLES
from earth2studio.utils import coord_array_like
from earth2studio.utils.cupy import from_torch


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


@pytest.fixture(scope="function")
def ucast_model() -> UCast:
    n_variables = len(VARIABLES)
    return UCast(
        model=PhooUCastModel(),
        center=torch.zeros(n_variables),
        scale=torch.ones(n_variables),
        residual_scale=torch.ones(n_variables),
        static_condition=torch.zeros(2, 121, 240),
        sst_fill_value=0.0,
        stochastic=False,
    )


@pytest.fixture(scope="function")
def model() -> UCast:
    package = UCast.load_default_package()
    return UCast.load_model(package)


def _input(
    ucast_model: UCast,
    time: np.ndarray,
    device: str = "cpu",
) -> xr.DataArray:
    signature = ucast_model.input_coords()
    assert isinstance(signature, xr.DataArray)
    assert signature.data.nbytes == 0
    ds = Random(OrderedDict((d, signature[d].values) for d in ("lat", "lon")))
    x = fetch_data(ds, time, signature["variable"].values, signature.lead_time.values)
    x.attrs = {
        k: v
        for k, v in signature.attrs.items()
        if k
        not in (
            "earth2studio_kind",
            "earth2studio_schema_version",
            "earth2studio_dynamic_dims",
        )
    }
    x.name = "weather"
    x.attrs["experiment"] = "ucast"
    x.encoding = {"test": "retained"}
    return x


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
    assert out_coords.dims == ("time", "lead_time", "variable", "lat", "lon")


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
    x = _input(ucast_model, time, device=device)
    x.attrs["nested"] = {"items": [1]}
    x.encoding["nested"] = {"items": [2]}
    original = x.copy(deep=True)
    out = ucast_model(x)

    assert out.shape == torch.Size([len(time), 1, len(VARIABLES), 121, 240])
    np.testing.assert_allclose(out.e2s.as_numpy(), x[:, -1:])
    _check_output_coords(out, x, np.timedelta64(12, "h"))
    xr.testing.assert_identical(x, original)
    assert out.name == x.name and out.attrs == x.attrs and out.encoding == x.encoding
    out.attrs["nested"].clear()
    out.encoding["nested"].clear()
    assert x.attrs["nested"] == {"items": [1]}
    assert x.encoding["nested"] == {"items": [2]}


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
    x = (
        _input(ucast_model, time, device=device)
        .expand_dims(ensemble=np.arange(ensemble))
        .copy(deep=True)
    )
    x = x.assign_coords(member=("ensemble", np.arange(ensemble)))
    x.attrs["removed"] = True
    events = []

    def front(field):
        assert field.dims == x.dims
        events.append("front")
        return field

    def rear(field):
        events.append("rear")
        field.attrs.pop("removed", None)
        field.encoding.clear()
        return field.drop_vars("member", errors="ignore")

    ucast_model.front_hook, ucast_model.rear_hook = front, rear
    iterator = ucast_model.create_iterator(x)
    assert isinstance(iterator, Iterable)

    initial = next(iterator)
    assert events == []
    xr.testing.assert_identical(initial, x.isel(lead_time=slice(-1, None)))
    np.testing.assert_array_equal(
        initial["lead_time"], np.array([np.timedelta64(0, "h")])
    )

    retained = initial.copy(deep=True)
    for i, out in enumerate(iterator):
        out_coords = out
        assert out.name == x.name and "removed" not in out.attrs
        assert out.encoding == {} and "member" not in out.coords
        assert events == ["front", "rear"] * (i + 1)
        assert out.shape == torch.Size(
            [ensemble, len(time), 1, len(VARIABLES), 121, 240]
        )
        np.testing.assert_allclose(out.e2s.as_numpy(), x[:, :, -1:])
        xr.testing.assert_identical(initial, retained)
        assert (out_coords["ensemble"] == np.arange(ensemble)).all()
        assert (out_coords["time"] == time).all()
        assert out_coords["lead_time"][0] == np.timedelta64(12 * (i + 1), "h")
        assert out_coords.dims == (
            "ensemble",
            "time",
            "lead_time",
            "variable",
            "lat",
            "lon",
        )

        if i >= 4:
            break


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
    x = _input(ucast_model, time)
    sst_index = VARIABLES.index("sst")
    x[:, -1, sst_index, 0, 0] = torch.nan

    iterator = ucast_model.create_iterator(x)
    next(iterator)
    first = next(iterator)
    second = next(iterator)

    assert first[0, 0, sst_index, 0, 0] == float(ucast_model.sst_fill_value)
    assert second[0, 0, sst_index, 0, 0] == float(ucast_model.sst_fill_value)

    # A metadata-only hook must retain the normalized SST recurrence; a value
    # edit must feed the next prediction instead of being hidden by that cache.
    ucast_model.front_hook = lambda field: field.assign_coords(note="metadata")
    iterator = ucast_model.create_iterator(x)
    next(iterator)
    hooked = next(iterator)
    np.testing.assert_allclose(hooked, first)
    np.testing.assert_allclose(next(iterator), second)
    ucast_model.front_hook = lambda field: field + 2
    iterator = ucast_model.create_iterator(x)
    next(iterator)
    changed = next(iterator)
    assert changed[0, 0, 0, 1, 1] == x[0, -1, 0, 1, 1] + 2
    ucast_model.clear_hooks()

    def edit_temperature(field):
        field.loc[{"variable": "t2m"}] += 1
        return field

    for hook in ("front_hook", "rear_hook"):
        setattr(ucast_model, hook, edit_temperature)
        iterator = ucast_model.create_iterator(x)
        next(iterator)
        for _ in range(3):
            forecast = next(iterator)
            assert forecast[0, 0, sst_index, 0, 0] == float(ucast_model.sst_fill_value)
        ucast_model.clear_hooks()

    def edit_sst(field):
        field.loc[{"variable": "sst", "lat": 90.0, "lon": 0.0}] = 10.0
        return field

    ucast_model.rear_hook = edit_sst
    iterator = ucast_model.create_iterator(x)
    next(iterator)
    assert next(iterator)[0, 0, sst_index, 0, 0] == 10.0
    ucast_model.clear_hooks()
    assert next(iterator)[0, 0, sst_index, 0, 0] == 11.0

    def mask_sst(field):
        field.loc[{"variable": "sst", "lat": 90.0, "lon": 0.0}] = np.nan
        return field

    ucast_model.front_hook = mask_sst
    assert next(iterator)[0, 0, sst_index, 0, 0] == float(ucast_model.sst_fill_value)
    ucast_model.clear_hooks()


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
    x = _input(ucast_model, time)
    with pytest.raises((KeyError, ValueError)):
        ucast_model(x.assign_coords(coords_update))


def test_ucast_conformance(ucast_model: UCast) -> None:
    # ucast_model is built with stochastic=False (dropout disabled at
    # inference), so P14 is skipped rather than passed: the RNG-isolation
    # rule has nothing to check for a model that does not declare itself
    # stochastic.
    assert check_prognostic_contract(ucast_model) == [
        "P14: model does not declare itself stochastic"
    ]


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
    coords = _input(ucast_model, time, device=device)
    x = ucast_model.center.reshape(1, 1, len(VARIABLES), 1, 1).expand(
        len(time), coords["lead_time"].shape[0], len(VARIABLES), 121, 240
    )
    x = x.contiguous()

    out = ucast_model(from_torch(x, coord_array_like(coords)))

    assert out.shape == torch.Size([len(time), 1, len(VARIABLES), 121, 240])
    assert np.isfinite(out.e2s.as_numpy()).all()
    _check_output_coords(out, coords, np.timedelta64(12, "h"))
