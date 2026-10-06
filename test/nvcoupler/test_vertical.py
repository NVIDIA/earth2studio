# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr

from earth2studio.nvcoupler.errors import VerticalMismatchError
from earth2studio.nvcoupler.vertical import (
    HybridLevels,
    PressureLevels,
    interp_to_pressure,
)


def _pressure_array() -> xr.DataArray:
    levels = np.array([100.0, 500.0, 1000.0])
    return xr.DataArray(
        np.log(levels * 100.0)[:, None, None],
        dims=("level", "lat", "lon"),
        coords={"level": levels, "lat": [0.0], "lon": [0.0]},
    )


def test_pressure_interpolation_is_dataarray_native() -> None:
    output = interp_to_pressure(
        _pressure_array(),
        PressureLevels((100.0, 500.0, 1000.0)),
        PressureLevels((200.0, 750.0)),
    )
    assert isinstance(output, xr.DataArray)
    assert output.dims == ("level", "lat", "lon")
    assert np.allclose(output[:, 0, 0], np.log([20000.0, 75000.0]))


def test_pressure_interpolation_clamps_endpoints() -> None:
    output = interp_to_pressure(
        _pressure_array(),
        PressureLevels((100.0, 500.0, 1000.0)),
        PressureLevels((50.0, 1100.0)),
    )
    assert np.allclose(output[:, 0, 0], _pressure_array()[[0, -1], 0, 0])


def test_hybrid_interpolation() -> None:
    source = HybridLevels(
        (1000.0, 5000.0, 10000.0),
        (0.0, 0.4, 0.9),
    )
    pressure = np.array(source.a) + np.array(source.b) * 100000.0
    array = xr.DataArray(
        np.log(pressure)[:, None, None],
        dims=("level", "lat", "lon"),
        coords={"level": np.arange(3), "lat": [0.0], "lon": [0.0]},
    )
    ps = xr.DataArray(
        [[100000.0]], dims=("lat", "lon"), coords={"lat": [0.0], "lon": [0.0]}
    )
    output = interp_to_pressure(array, source, PressureLevels((200.0, 500.0)), ps)
    assert np.allclose(output[:, 0, 0], np.log([20000.0, 50000.0]))


def test_hybrid_requires_surface_pressure() -> None:
    source = HybridLevels((1000.0, 5000.0), (0.0, 0.9))
    array = xr.DataArray(
        np.ones((2, 1, 1)),
        dims=("level", "lat", "lon"),
        coords={"level": [0, 1], "lat": [0.0], "lon": [0.0]},
    )
    with pytest.raises(VerticalMismatchError, match="requires"):
        interp_to_pressure(array, source, PressureLevels((500.0,)))


def test_level_validation() -> None:
    with pytest.raises(ValueError, match="increasing"):
        PressureLevels((1000.0, 500.0))
    with pytest.raises(VerticalMismatchError, match="does not match"):
        interp_to_pressure(
            _pressure_array(),
            PressureLevels((100.0, 400.0, 1000.0)),
            PressureLevels((500.0,)),
        )
    with pytest.raises(VerticalMismatchError, match="no 'level'"):
        interp_to_pressure(
            _pressure_array().isel(level=0, drop=True),
            PressureLevels((100.0,)),
            PressureLevels((100.0,)),
        )
