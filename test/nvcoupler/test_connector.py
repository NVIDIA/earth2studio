# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr

from earth2studio.grids import PointGrid
from earth2studio.nvcoupler.clock import Clock
from earth2studio.nvcoupler.component import CallableComponent
from earth2studio.nvcoupler.connector import Connector
from earth2studio.nvcoupler.errors import CouplingError, IncompatibleFieldError
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

T0 = np.datetime64("2024-01-01")


def _pair(with_mask: bool = False):
    atmosphere = fake_atmos()
    ocean = fake_ocean(with_mask=with_mask)
    clock = Clock(T0, "2024-01-05", "6h")
    atmosphere.realize(clock)
    ocean.realize(clock)
    atmosphere.initialize(atmos_ic())
    ocean.initialize(ocean_ic(sst0=3.0))
    return atmosphere, ocean


def test_match_and_regrid_dataarray() -> None:
    atmosphere, ocean = _pair()
    connector = Connector(ocean, atmosphere)
    assert connector.match() == ["sea_surface_temperature"]
    connector.execute(T0)
    field = atmosphere.import_state["sea_surface_temperature"]
    assert isinstance(field.array, xr.DataArray)
    assert field.array.dims == ("lat", "lon")
    assert field.array.shape == (32, 64)
    assert np.allclose(field.array, 3.0)


def test_no_matching_fields_raises() -> None:
    atmosphere, ocean = _pair()
    with pytest.raises(IncompatibleFieldError, match="no fields match"):
        Connector(atmosphere, ocean).match()


def test_zero_and_nearest_mask_fill() -> None:
    atmosphere, ocean = _pair(with_mask=True)
    Connector(ocean, atmosphere, fill="zero").execute(T0)
    zeroed = atmosphere.import_state["sea_surface_temperature"]
    assert zeroed.mask is None
    assert np.any(np.asarray(zeroed.array) == 0)

    atmosphere, ocean = _pair(with_mask=True)
    Connector(ocean, atmosphere, fill="nearest").execute(T0)
    filled = atmosphere.import_state["sea_surface_temperature"]
    assert filled.mask is None
    assert np.allclose(filled.array, 3.0)


def _point_destination(grid: PointGrid) -> CallableComponent:
    return CallableComponent(
        "stations",
        lambda array: array,
        timestep="6h",
        imports=["sea_surface_temperature"],
        grid=grid,
    )


@pytest.mark.parametrize("sample", ["nearest", "bilinear"])
def test_pointgrid_sampling(sample: str) -> None:
    _, ocean = _pair()
    grid = PointGrid(
        latitude=np.array([10.0, -20.0]),
        longitude=np.array([45.0, 180.0]),
        x=np.array(["a", "b"]),
    )
    stations = _point_destination(grid)
    stations.realize(ocean.clock)
    Connector(ocean, stations, sample=sample).execute(T0)
    sampled = stations.import_state["sea_surface_temperature"].array
    assert sampled.dims == ("x",)
    assert list(sampled.coords["x"].values) == ["a", "b"]
    assert np.allclose(sampled, 3.0)


def test_pointgrid_requires_sampling_policy() -> None:
    _, ocean = _pair()
    stations = _point_destination(PointGrid(np.array([0.0]), np.array([0.0])))
    stations.realize(ocean.clock)
    with pytest.raises(CouplingError, match="requires sample"):
        Connector(ocean, stations).execute(T0)


def test_custom_regridder_is_dataarray_callable() -> None:
    atmosphere, ocean = _pair()

    def regrid(array: xr.DataArray) -> xr.DataArray:
        return xr.DataArray(
            np.full((32, 64), float(array.mean())),
            dims=("lat", "lon"),
            coords={
                "lat": atmosphere.state.coords["lat"],
                "lon": atmosphere.state.coords["lon"],
            },
        )

    Connector(ocean, atmosphere, regridder=regrid).execute(T0)
    assert np.allclose(atmosphere.import_state["sea_surface_temperature"].array, 3.0)
