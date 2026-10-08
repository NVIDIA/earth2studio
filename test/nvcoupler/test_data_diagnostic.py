# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr

from earth2studio.nvcoupler.clock import Clock
from earth2studio.nvcoupler.component import DataComponent, DiagnosticComponent
from earth2studio.nvcoupler.errors import CouplingError
from earth2studio.nvcoupler.field import Field
from earth2studio.nvcoupler.testing import grid_coords

T0 = np.datetime64("2024-01-01")


class Source:
    def __init__(self, value: float = 3.0):
        self.value = value
        self.calls: list[tuple[np.ndarray, np.ndarray]] = []

    def __call__(self, time, variable) -> xr.DataArray:
        time = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        variable = np.atleast_1d(variable)
        self.calls.append((time, variable))
        return xr.DataArray(
            np.full((len(time), len(variable), 4, 8), self.value),
            dims=("time", "variable", "lat", "lon"),
            coords={
                "time": time,
                "variable": variable,
                **grid_coords(4, 8),
            },
        )


class Diagnostic:
    def input_coords(self) -> xr.DataArray:
        return xr.DataArray(
            np.empty((1, 1, 4, 8)),
            dims=("batch", "variable", "lat", "lon"),
            coords={
                "batch": [0],
                "variable": ["z1000"],
                **grid_coords(4, 8),
            },
        )

    def output_coords(self, input_coords: xr.DataArray) -> xr.DataArray:
        return input_coords.assign_coords(variable=["z500"])

    def __call__(self, array: xr.DataArray) -> xr.DataArray:
        return (0.5 * array).assign_coords(variable=["z500"])

    def to(self, device):
        return self


def test_data_component_fetches_and_publishes_dataarray() -> None:
    source = Source()
    component = DataComponent("data", source, ["sea_surface_temperature"], "6h")
    component.realize(Clock(T0, T0 + np.timedelta64(12, "h"), "6h"))
    component.initialize()
    field = component.export_state["sea_surface_temperature"]
    assert isinstance(field.array, xr.DataArray)
    assert field.array.dims == ("lat", "lon")
    assert np.allclose(field.array, 3.0)
    assert list(source.calls[0][1]) == ["sst"]


def test_data_component_requires_realization_before_fetch() -> None:
    component = DataComponent("data", Source(), ["sea_surface_temperature"], "6h")
    with pytest.raises(CouplingError, match="realized"):
        component.initialize()


def test_diagnostic_consumes_import_state() -> None:
    component = DiagnosticComponent("diag", Diagnostic(), timestep="6h")
    component.realize(Clock(T0, T0 + np.timedelta64(6, "h"), "6h"))
    component.initialize()
    source = xr.DataArray(
        np.full((4, 8), 10.0),
        dims=("lat", "lon"),
        coords=grid_coords(4, 8),
    )
    component.import_state.add(Field(source, "geopotential_at_1000hpa", "m2 s-2"))
    component.run(T0 + np.timedelta64(6, "h"))
    output = component.export_state["geopotential_at_500hpa"]
    assert isinstance(output.array, xr.DataArray)
    assert np.allclose(output.array, 5.0)


def test_diagnostic_reports_missing_import() -> None:
    component = DiagnosticComponent("diag", Diagnostic(), timestep="6h")
    with pytest.raises(CouplingError, match="missing imports"):
        component.run(T0)
