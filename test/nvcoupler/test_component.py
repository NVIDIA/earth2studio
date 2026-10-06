# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr

from earth2studio.nvcoupler.clock import Clock
from earth2studio.nvcoupler.component import (
    CallableComponent,
    ConditioningKwargAdapter,
    Exchange,
)
from earth2studio.nvcoupler.errors import CouplingError
from earth2studio.nvcoupler.field import Field, State
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, grid_coords

T0 = np.datetime64("2024-01-01")


def _imports(value: float = 3.0) -> State:
    array = xr.DataArray(
        np.full((32, 64), value, dtype=np.float32),
        dims=("lat", "lon"),
        coords=grid_coords(32, 64),
    )
    return State(
        "imports",
        [Field(array, "sea_surface_temperature", "K")],
    )


def test_exchange_inject_is_dataarray_native() -> None:
    state = atmos_ic()
    output = Exchange(
        state,
        _imports(5.0),
        {"sea_surface_temperature": "sst"},
    ).inject()
    assert isinstance(output, xr.DataArray)
    assert np.all(output.sel(variable="sst") == 5.0)
    assert np.all(output.sel(variable="z1000") == 0.0)
    assert np.all(state.sel(variable="sst") == 2.0)


def test_exchange_rejects_unknown_model_variable() -> None:
    with pytest.raises(CouplingError, match="not in state variables"):
        Exchange(
            atmos_ic(),
            _imports(),
            {"sea_surface_temperature": "missing"},
        ).inject()


def test_conditioning_adapter_passes_labeled_array() -> None:
    captured: dict[str, xr.DataArray] = {}

    class Model:
        def call_with_conditioning(
            self, state: xr.DataArray, conditioning: xr.DataArray
        ) -> xr.DataArray:
            captured["conditioning"] = conditioning
            return state

    state = atmos_ic()
    output = ConditioningKwargAdapter()(Model(), Exchange(state, _imports(7.0)))
    assert output.identical(state)
    assert list(captured["conditioning"].coords["variable"].values) == [
        "sea_surface_temperature"
    ]
    assert np.all(captured["conditioning"] == 7.0)


def test_callable_component_runs_with_dataarray() -> None:
    component = fake_atmos()
    component.realize(Clock(T0, "2024-01-02", "6h"))
    component.initialize(atmos_ic())
    component.import_state.add(_imports(4.0)["sea_surface_temperature"])
    component.run(T0 + np.timedelta64(6, "h"))
    assert np.allclose(component.state.sel(variable="z1000"), 1.4)
    assert component.run_count == 1


def test_callable_component_requires_dataarray_ic() -> None:
    component = CallableComponent(
        "identity",
        lambda array: array,
        timestep="6h",
    )
    with pytest.raises(CouplingError, match="needs an initial condition"):
        component.initialize()
