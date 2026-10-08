# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import xarray as xr

from earth2studio.nvcoupler.api import couple
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic


def test_dataarrays_cross_component_boundaries() -> None:
    driver = couple(
        fake_atmos(),
        fake_ocean(),
        start="2024-01-01",
        stop="2024-01-03",
    )
    driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
    datasets = driver.run()
    assert isinstance(
        driver.probe("ocean->atmos")["sea_surface_temperature"].array,
        xr.DataArray,
    )
    assert datasets["atmos"]["geopotential_at_1000hpa"].dims == (
        "time",
        "lat",
        "lon",
    )
    assert np.isfinite(datasets["atmos"].to_array()).all()
