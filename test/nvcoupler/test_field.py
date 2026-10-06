# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr

from earth2studio.nvcoupler.dictionary import DEFAULT_DICTIONARY
from earth2studio.nvcoupler.errors import CouplingError, UnknownFieldError
from earth2studio.nvcoupler.field import Field, State
from earth2studio.nvcoupler.testing import grid_coords


def _array(value: float = 1.0) -> xr.DataArray:
    return xr.DataArray(
        np.full((4, 8), value, dtype=np.float32),
        dims=("lat", "lon"),
        coords=grid_coords(4, 8),
    )


def test_field_requires_single_variable_dataarray() -> None:
    with pytest.raises(TypeError, match="DataArray"):
        Field(np.ones((2, 2)), "air_temperature_2m", "K")  # type: ignore[arg-type]
    with pytest.raises(CouplingError, match="must not contain"):
        Field(
            xr.concat([_array()], xr.IndexVariable("variable", ["t2m"])),
            "air_temperature_2m",
            "K",
        )


def test_field_clone_is_independent() -> None:
    field = Field(_array(), "air_temperature_2m", "K")
    clone = field.clone()
    clone.array.data[0, 0] = 99
    assert field.array.data[0, 0] == 1
    assert field.grid_signature() == clone.grid_signature()


def test_state_stack_and_from_dataarray_round_trip() -> None:
    state = State(
        "state",
        [
            Field(_array(2), "sea_surface_temperature", "K"),
            Field(_array(3), "air_temperature_2m", "K"),
        ],
    )
    stacked = state.stack(["sea_surface_temperature", "air_temperature_2m"])
    assert stacked.dims == ("variable", "lat", "lon")
    rebuilt = State.from_dataarray("rebuilt", stacked, DEFAULT_DICTIONARY)
    assert set(rebuilt) == {"sea_surface_temperature", "air_temperature_2m"}
    assert np.allclose(rebuilt["air_temperature_2m"].array, 3)


def test_state_stack_rejects_grid_mismatch() -> None:
    state = State(
        "state",
        [
            Field(_array(), "sea_surface_temperature", "K"),
            Field(
                xr.DataArray(
                    np.ones((3, 8)),
                    dims=("lat", "lon"),
                    coords=grid_coords(3, 8),
                ),
                "air_temperature_2m",
                "K",
            ),
        ],
    )
    with pytest.raises(CouplingError, match="cannot be stacked"):
        state.stack()


def test_from_dataarray_alias_and_strict_handling() -> None:
    array = xr.concat(
        [_array(), _array(2)],
        xr.IndexVariable("variable", ["sst", "unknown"]),
    )
    with pytest.raises(UnknownFieldError):
        State.from_dataarray("state", array, DEFAULT_DICTIONARY)
    state = State.from_dataarray("state", array, DEFAULT_DICTIONARY, strict=False)
    assert list(state) == ["sea_surface_temperature"]
