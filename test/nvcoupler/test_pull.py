# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr

from earth2studio.nvcoupler.component import Exchange
from earth2studio.nvcoupler.errors import CouplingError
from earth2studio.nvcoupler.field import Field, State
from earth2studio.nvcoupler.pull import PullAdapter, StateDataSource
from earth2studio.nvcoupler.testing import grid_coords

T0 = np.datetime64("2024-01-01")


def _field(name: str, value: float) -> Field:
    return Field(
        xr.DataArray(
            np.full((4, 8), value),
            dims=("lat", "lon"),
            coords=grid_coords(4, 8),
        ),
        name,
        "m s-1" if "wind" in name else "K",
        valid_time=T0,
    )


def _imports() -> State:
    return State(
        "imports",
        [
            _field("eastward_wind_10m", 3.0),
            _field("air_temperature_2m", 290.0),
        ],
    )


def test_state_data_source_returns_labeled_array() -> None:
    source = StateDataSource(
        _imports(),
        raw_to_std={
            "u10m": "eastward_wind_10m",
            "t2m": "air_temperature_2m",
        },
    )
    output = source([T0], ["u10m", "t2m"])
    assert output.dims == ("time", "variable", "lat", "lon")
    assert np.all(output.sel(variable="u10m") == 3.0)
    assert np.all(output.sel(variable="t2m") == 290.0)


def test_state_data_source_reports_unknown_and_stale_requests() -> None:
    with pytest.raises(CouplingError, match="wire a connector"):
        StateDataSource(_imports())([T0], ["msl"])
    source = StateDataSource(
        _imports(),
        raw_to_std={"u10m": "eastward_wind_10m"},
        strict_time=True,
    )
    with pytest.raises(CouplingError, match="run-sequence ordering"):
        source([T0 + np.timedelta64(1, "h")], ["u10m"])


def test_pull_adapter_installs_live_source() -> None:
    class Model:
        conditioning_data_source = None

        def __call__(self, state: xr.DataArray) -> xr.DataArray:
            forcing = self.conditioning_data_source([T0], ["u10m", "t2m"])
            return state + float(forcing.sel(variable="u10m").mean())

    state = xr.DataArray([0.0], dims=("variable",), coords={"variable": ["x"]})
    output = PullAdapter()(
        Model(),
        Exchange(
            state,
            _imports(),
            {
                "eastward_wind_10m": "u10m",
                "air_temperature_2m": "t2m",
            },
        ),
    )
    assert float(output.item()) == 3.0


def test_pull_adapter_requires_data_source_attribute() -> None:
    with pytest.raises(CouplingError, match="ConditioningKwargAdapter"):
        PullAdapter()(lambda state: state, Exchange(xr.DataArray(0), _imports()))
