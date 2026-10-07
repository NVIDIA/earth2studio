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


def test_variadic_call_and_iterator() -> None:
    model = CoupledModel()
    x, ocean, forcing = (xr.DataArray(value) for value in (1, 10, 2))
    expected = model(x, ocean, forcing)
    iterator = model.create_iterator(x, ocean, forcing)
    first = next(iterator)
    for actual, reference in zip(first, expected):
        xr.testing.assert_identical(actual, reference)
    second = iterator.send((xr.DataArray(3),))
    assert tuple(value.item() for value in second) == (6, 15)
    assert tuple(value.item() for value in first) == (3, 12)
    with pytest.raises(ValueError, match="forcing"):
        next(iterator)


@pytest.mark.parametrize("iterator", [False, True])
def test_missing_initial_forcing(iterator: bool) -> None:
    with pytest.raises(ValueError, match="forcing"):
        model = CoupledModel()
        x = (xr.DataArray(1), xr.DataArray(2))
        if iterator:
            next(model.create_iterator(*x))
        else:
            model(*x)


class SimpleModel(PrognosticMixin):
    def input_coords(self) -> xr.DataArray:
        return xr.DataArray()

    def initialize(self, x: xr.DataArray) -> tuple[xr.DataArray, None]:
        return x + 1, None

    def step(self, y: xr.DataArray, *, state: None) -> tuple[xr.DataArray, None]:
        return y + 1, state


def test_unforced_iterator_hooks_and_ownership() -> None:
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
    iterator = model.create_iterator(x)
    first = next(iterator)
    assert first.item() == 101
    assert events == ["rear"]
    assert next(iterator).item() == 112
    assert events == ["rear", "front", "rear"]
    assert first.item() == 101
    assert x.item() == 0


def test_front_hook_preserves_previous_yield() -> None:
    model = SimpleModel()

    def front(y: xr.DataArray) -> xr.DataArray:
        y += 10
        return y

    model.front_hook = front
    iterator = model.create_iterator(xr.DataArray(0))
    first = next(iterator)
    assert next(iterator).item() == 12
    assert first.item() == 1


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
    iterator = model.create_iterator(xr.DataArray(1), xr.DataArray(2))
    assert next(iterator).item() == 3
    assert next(iterator).item() == 5
    assert model.default_sources() == (None, None)


@pytest.mark.parametrize("forcing", [(), (xr.DataArray(1), xr.DataArray(2))])
def test_iterator_rejects_wrong_forcing_count(
    forcing: tuple[xr.DataArray, ...],
) -> None:
    iterator = CoupledModel().create_iterator(*(xr.DataArray(1) for _ in range(3)))
    next(iterator)
    with pytest.raises(ValueError, match="forcing"):
        iterator.send(forcing)
