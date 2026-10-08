# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import xarray as xr

from earth2studio.nvcoupler.clock import Clock
from earth2studio.nvcoupler.driver import Driver
from earth2studio.nvcoupler.errors import CouplingError
from earth2studio.nvcoupler.mediator import TrailingAverageMediator
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

T0 = "2024-01-01"
T96 = "2024-01-05"


def _driver(stop: str = T96) -> Driver:
    components = {
        "atmos": fake_atmos(),
        "ocean": fake_ocean(),
        "med": TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"]),
    }
    driver = Driver(
        components,
        clock=Clock(T0, stop, "6h"),
        connectors=[("atmos", "med"), ("ocean", "atmos"), ("med", "ocean")],
    )
    driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
    return driver


def test_driver_runs_dataarray_components() -> None:
    driver = _driver()
    datasets = driver.run()
    assert all(isinstance(value, xr.Dataset) for value in datasets.values())
    assert np.allclose(
        datasets["atmos"]["geopotential_at_1000hpa"].isel(time=-1),
        19.2336,
        atol=1e-4,
    )
    assert np.allclose(
        datasets["ocean"]["sea_surface_temperature"].isel(time=-1),
        2.180147,
        atol=1e-6,
    )
    assert driver.components["atmos"].run_count == 16
    assert driver.components["ocean"].run_count == 2


def test_steps_and_probe_return_labeled_data() -> None:
    driver = _driver("2024-01-03")
    seen = list(driver.steps())
    assert len(seen) == 8
    assert all(
        isinstance(states["atmos"], type(driver.components["atmos"].export_state))
        for _, states in seen
    )
    transfer = driver.probe("ocean->atmos")
    assert isinstance(transfer["sea_surface_temperature"].array, xr.DataArray)


def test_missing_initial_condition_raises() -> None:
    driver = Driver(
        {"atmos": fake_atmos(), "ocean": fake_ocean()},
        clock=Clock(T0, T96, "6h"),
        connectors=[("ocean", "atmos")],
        allow_unfed_imports=True,
    )
    with pytest.raises(CouplingError, match="initial condition"):
        driver.initialize({"atmos": atmos_ic()})


def test_run_requires_initialize_and_non_exhausted_clock() -> None:
    driver = Driver(
        {"atmos": fake_atmos()},
        clock=Clock(T0, "2024-01-02", "6h"),
        allow_unfed_imports=True,
    )
    with pytest.raises(CouplingError, match="initialize"):
        driver.run()
    driver.initialize({"atmos": atmos_ic()})
    driver.run()
    with pytest.raises(CouplingError, match="exhausted"):
        driver.run()


def test_reset_requires_reinitialize() -> None:
    driver = _driver("2024-01-02")
    first = driver.run()
    driver.reset()
    with pytest.raises(CouplingError, match="initialize"):
        driver.run()
    driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
    second = driver.run()
    xr.testing.assert_identical(first["atmos"], second["atmos"])
