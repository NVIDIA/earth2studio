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


class SimpleModel(PrognosticMixin):
    def input_coords(self):
        return xr.DataArray(0)

    def initialize(self, x):
        return x + 1, None

    def step(self, y, state):
        return y + 1, state

    def __call__(self, x):
        return self._default_call(x)

    def create_iterator(self, x):
        return self._default_create_iterator(x)


class ForcedModel(PrognosticMixin):
    def input_coords(self):
        return xr.DataArray(0), xr.DataArray(0)

    def forcing_coords(self):
        return xr.DataArray(0), xr.DataArray([0], dims="lead_time")

    def initialize(self, atmosphere, ocean, static, forcing):
        return (atmosphere + forcing, ocean + forcing), static

    def step(self, atmosphere, ocean, forcing, state):
        return (atmosphere + forcing + state, ocean + forcing + state), state

    def __call__(self, atmosphere, ocean, static, forcing):
        return self._default_call(atmosphere, ocean, static, forcing)

    def create_iterator(self, atmosphere, ocean, static, forcing):
        return self._default_create_iterator(atmosphere, ocean, static, forcing)


@pytest.mark.parametrize("method", ["__call__", "create_iterator"])
def test_public_methods_require_overrides(method):
    with pytest.raises(NotImplementedError, match=method):
        getattr(PrognosticMixin(), method)(xr.DataArray(0))


def test_default_call_and_forecasts_only_iterator():
    model = SimpleModel()
    x = xr.DataArray(0)
    it = model.create_iterator(x)
    first = next(it)
    xr.testing.assert_equal(first, model(x))
    assert first.item() == 1
    assert next(it).item() == 2
    assert first.item() == 1
    assert x.item() == 0


@pytest.mark.parametrize("multiple", [False, True])
def test_in_place_hooks_preserve_recurrence_and_earlier_yields(multiple):
    def front(y):
        for slot in y if isinstance(y, tuple) else (y,):
            slot.values[...] += 10
        return y

    def rear(y):
        for slot in y if isinstance(y, tuple) else (y,):
            slot.values[...] += 100
        return y

    model = ForcedModel() if multiple else SimpleModel()
    model.front_hook = front
    model.rear_hook = rear
    x = xr.DataArray(0)
    args = (x, x, x, xr.DataArray(1)) if multiple else (x,)
    direct = model(*args)
    assert (direct[0] if multiple else direct).item() == 1
    it = model.create_iterator(*args)
    first = next(it)
    second = it.send(xr.DataArray(1)) if multiple else next(it)
    for slot in first if multiple else (first,):
        assert slot.item() == 101
    for slot in second if multiple else (second,):
        assert slot.item() == 112
    assert x.item() == 0


@pytest.mark.parametrize("as_tuple", [False, True])
def test_mixed_static_and_dynamic_forcing(as_tuple):
    model = ForcedModel()
    it = model.create_iterator(*(xr.DataArray(v) for v in (0, 10, 2, 1)))
    first = next(it)
    assert tuple(slot.item() for slot in first) == (1, 11)
    forcing = xr.DataArray(3)
    second = it.send((forcing,) if as_tuple else forcing)
    assert tuple(slot.item() for slot in second) == (6, 16)
    with pytest.raises(ValueError, match="forcing"):
        next(it)


@pytest.mark.parametrize("iterator", [False, True])
def test_missing_initial_forcing(iterator):
    model = ForcedModel()
    with pytest.raises(ValueError, match="forcing"):
        if iterator:
            next(model._default_create_iterator(xr.DataArray(0)))
        else:
            model._default_call(xr.DataArray(0))


def test_unforced_iterator_rejects_extra_forcing():
    it = SimpleModel().create_iterator(xr.DataArray(0))
    next(it)
    with pytest.raises(ValueError, match="forcing"):
        it.send(xr.DataArray(1))
