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

import pytest
import xarray as xr

from earth2studio.models.px.utils import PrognosticMixin


class CoupledModel(PrognosticMixin):
    def input_coords(self) -> tuple[xr.DataArray, xr.DataArray]:
        return (xr.DataArray(), xr.DataArray())

    def forcing_coords(self) -> xr.DataArray:
        return xr.DataArray([0], dims="lead_time")

    def initialize(
        self, atmosphere: xr.DataArray, ocean: xr.DataArray, forcing: xr.DataArray
    ) -> tuple[tuple[xr.DataArray, xr.DataArray], int]:
        return (atmosphere + forcing, ocean + forcing), 1

    def step(
        self,
        atmosphere: xr.DataArray,
        ocean: xr.DataArray,
        forcing: xr.DataArray,
        *,
        state: int,
    ) -> tuple[tuple[xr.DataArray, xr.DataArray], int]:
        return (atmosphere + forcing, ocean + forcing), state + 1


def test_variadic_call() -> None:
    model = CoupledModel()
    x, ocean, forcing = (xr.DataArray(value) for value in (1, 10, 2))
    outputs = model(x, ocean, forcing)
    expected, _ = model.initialize(x, ocean, forcing)
    for actual, reference in zip(outputs, expected):
        xr.testing.assert_identical(actual, reference)
    assert tuple(value.item() for value in outputs) == (3, 12)


def test_missing_initial_forcing() -> None:
    with pytest.raises(ValueError, match="forcing"):
        model = CoupledModel()
        x = (xr.DataArray(1), xr.DataArray(2))
        model(*x)


class SimpleModel(PrognosticMixin):
    def input_coords(self) -> xr.DataArray:
        return xr.DataArray()

    def initialize(self, x: xr.DataArray) -> tuple[xr.DataArray, None]:
        return x + 1, None

    def step(self, y: xr.DataArray, *, state: None) -> tuple[xr.DataArray, None]:
        return y + 1, state


def test_unforced_call_does_not_apply_hooks() -> None:
    model = SimpleModel()
    events = []

    def front(y: xr.DataArray) -> xr.DataArray:
        events.append("front")
        return y + 10

    def rear(y: xr.DataArray) -> xr.DataArray:
        events.append("rear")
        y += 100
        return y

    model.front_hook = front
    model.rear_hook = rear
    x = xr.DataArray(0)
    assert model(x).item() == 1
    assert not events
    assert x.item() == 0


def test_create_iterator_requires_implementation() -> None:
    with pytest.raises(NotImplementedError, match="create_iterator"):
        SimpleModel().create_iterator(xr.DataArray(0))


class StaticModel(SimpleModel):
    def forcing_coords(self) -> xr.DataArray:
        return xr.DataArray()

    def initialize(
        self, x: xr.DataArray, static: xr.DataArray
    ) -> tuple[xr.DataArray, xr.DataArray]:
        return x + static, static

    def step(
        self, y: xr.DataArray, *, state: xr.DataArray
    ) -> tuple[xr.DataArray, xr.DataArray]:
        return y + state, state


def test_static_forcing_is_only_supplied_at_initialization() -> None:
    model = StaticModel()
    y, state = model.initialize(xr.DataArray(1), xr.DataArray(2))
    assert y.item() == 3
    y, state = model.step(y, state=state)
    assert y.item() == 5
    assert model.default_sources() == (None, None)
