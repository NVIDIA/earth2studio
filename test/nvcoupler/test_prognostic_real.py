# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr

from earth2studio.models.px import Persistence
from earth2studio.nvcoupler.clock import Clock
from earth2studio.nvcoupler.component import PrognosticComponent
from earth2studio.nvcoupler.errors import CouplingError
from earth2studio.nvcoupler.testing import grid_coords

T0 = np.datetime64("2024-01-01")
VARIABLES = ["t2m", "z1000"]


def _state(model: Persistence) -> xr.DataArray:
    signature = model.input_coords()
    coords = {
        name: ([0] if name == "batch" else np.asarray(signature.coords[name]))
        for name in signature.dims
    }
    shape = tuple(len(coords[name]) for name in signature.dims)
    return xr.DataArray(
        np.arange(np.prod(shape), dtype=np.float32).reshape(shape),
        dims=signature.dims,
        coords=coords,
    )


def test_persistence_model_seam_is_dataarray_native() -> None:
    model = Persistence(VARIABLES, grid_coords(8, 16))
    component = PrognosticComponent("persist", model)
    component.realize(Clock(T0, T0 + np.timedelta64(12, "h"), "6h"))
    initial = _state(model)
    component.initialize(initial)
    component.run(T0 + np.timedelta64(6, "h"))
    assert isinstance(component.state, xr.DataArray)
    assert component.state.dims == initial.dims
    assert np.array_equal(
        component.state.coords["lead_time"],
        model.input_coords().coords["lead_time"],
    )
    for name in component.export_names:
        assert component.export_state[name].array.dims == ("lat", "lon")


def test_timestep_and_exports_are_inferred() -> None:
    component = PrognosticComponent(
        "persist", Persistence(VARIABLES, grid_coords(8, 16))
    )
    assert component.timestep == np.timedelta64(6, "h")
    assert component.export_names == [
        "air_temperature_2m",
        "geopotential_at_1000hpa",
    ]


def test_multi_history_requires_next_input_policy() -> None:
    model = Persistence(VARIABLES, grid_coords(8, 16), history=2)
    component = PrognosticComponent("persist", model)
    component.realize(Clock(T0, T0 + np.timedelta64(6, "h"), "6h"))
    component.initialize(_state(model))
    with pytest.raises(CouplingError, match="next_input"):
        component.run(T0 + np.timedelta64(6, "h"))
