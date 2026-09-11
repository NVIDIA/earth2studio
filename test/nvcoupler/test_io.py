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

"""Streaming IO tests: the coupled toy system writing into ZarrBackends."""

import numpy as np
import pytest

from earth2studio.io import ZarrBackend
from earth2studio.nvcoupler.clock import Clock
from earth2studio.nvcoupler.driver import Driver
from earth2studio.nvcoupler.errors import CouplingError
from earth2studio.nvcoupler.mediator import TrailingAverageMediator
from earth2studio.nvcoupler.testing import atmos_ic, fake_atmos, fake_ocean, ocean_ic

T0 = "2024-01-01"
T96 = "2024-01-05"

LAGGED_DSL = """
@6h
  atmos -> med
  ocean -> atmos
  atmos
@48h
  med.compute
  med -> ocean
  ocean
@
"""


def make_driver(io=None, collect=True):
    components = {
        "atmos": fake_atmos(gain=1.0),
        "ocean": fake_ocean(gain=1.0),
        "med": TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"]),
    }
    driver = Driver(
        components, LAGGED_DSL, Clock(T0, T96, "6h"), collect=collect, io=io
    )
    driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
    return driver


def test_zarr_streaming_matches_to_xarray():
    io = {"atmos": ZarrBackend(), "ocean": ZarrBackend()}
    driver = make_driver(io=io)
    ds = driver.run()

    z = io["atmos"]["geopotential_at_1000hpa"][:]
    assert z.shape == (17, 32, 64)  # IC + 16 rings
    assert np.array_equal(z, ds["atmos"]["geopotential_at_1000hpa"].values)
    assert np.allclose(z[0], 0.0)
    assert np.allclose(z[-1], 19.2336, atol=1e-4)

    sst = io["ocean"]["sea_surface_temperature"][:]
    assert sst.shape == (3, 16, 32)  # IC + 2 rings
    assert np.array_equal(sst, ds["ocean"]["sea_surface_temperature"].values)
    assert np.allclose(sst[-1], 2.180147, atol=1e-6)

    # time coords are each component's ring times INCLUDING t0
    atmos_times = io["atmos"]["time"][:].astype("datetime64[ns]")
    expected = np.datetime64(T0, "ns") + np.arange(17) * np.timedelta64(6, "h")
    assert np.array_equal(atmos_times, expected)
    ocean_times = io["ocean"]["time"][:].astype("datetime64[ns]")
    expected = np.datetime64(T0, "ns") + np.arange(3) * np.timedelta64(48, "h")
    assert np.array_equal(ocean_times, expected)

    # spatial coords round-trip
    assert np.allclose(io["atmos"]["lat"][:], np.linspace(90.0, -90.0, 32))
    assert np.allclose(io["ocean"]["lat"][:], np.linspace(90.0, -90.0, 16))


def test_mediator_io_deferred_setup():
    """A mediator exports nothing at t0, so its arrays are allocated at first
    compute; the t0 row stays unwritten (zarr's default fill value)."""
    io = {"med": ZarrBackend()}
    driver = make_driver(io=io)
    driver.run()
    zm = io["med"]["geopotential_at_1000hpa_48h_mean"][:]
    assert zm.shape == (3, 32, 64)
    # unwritten t0 row must be NaN (poisonous), not zarr's 0.0 default —
    # never-written data must not masquerade as physical values
    assert np.all(np.isnan(zm[0]))
    assert np.allclose(zm[1], 4.2, atol=1e-6)
    assert np.allclose(zm[2], 13.8147, atol=1e-4)


def test_io_independent_of_collect():
    io = {"atmos": ZarrBackend()}
    driver = make_driver(io=io, collect=False)
    result = driver.run()
    assert result == {}  # nothing collected in memory
    z = io["atmos"]["geopotential_at_1000hpa"][:]
    assert z.shape == (17, 32, 64)
    assert np.allclose(z[-1], 19.2336, atol=1e-4)


def test_zarr_rows_land_under_ring_times_for_time_coord_fields():
    """A component whose fields keep size-1 batch/time coords (DLESyM-split
    style) must stream rows under the RING times — the field's own stale IC
    time coord is squeezed out, never overwriting the ring time."""
    from collections import OrderedDict

    import torch

    from earth2studio.nvcoupler.component import CallableComponent

    def step(x, coords):
        return x + 1.0, coords

    comp = CallableComponent(
        "comp", step, timestep="6h", exports=["geopotential_at_1000hpa"]
    )
    coords = OrderedDict(
        {
            "batch": np.array([0]),
            "time": np.array([np.datetime64(T0, "ns")]),
            "variable": np.array(["z1000"]),
            "lat": np.linspace(90.0, -90.0, 8),
            "lon": np.linspace(0.0, 360.0, 16, endpoint=False),
        }
    )
    io = {"comp": ZarrBackend()}
    driver = Driver(
        {"comp": comp}, "@6h\n  comp\n@", Clock(T0, "2024-01-01T12", "6h"), io=io
    )
    driver.initialize({"comp": (torch.zeros(1, 1, 1, 8, 16), coords)})
    driver.run()

    z = io["comp"]["geopotential_at_1000hpa"][:]
    assert z.shape == (3, 8, 16)  # (ring time, lat, lon) — batch/time squeezed
    assert np.allclose(z, np.arange(3)[:, None, None])
    times = io["comp"]["time"][:].astype("datetime64[ns]")
    expected = np.datetime64(T0, "ns") + np.arange(3) * np.timedelta64(6, "h")
    assert np.array_equal(times, expected)


def test_unknown_io_key_raises():
    with pytest.raises(CouplingError, match="not component names"):
        Driver(
            {"atmos": fake_atmos()},
            "@6h\n  atmos\n@",
            Clock(T0, "2024-01-02", "6h"),
            io={"atmso": ZarrBackend()},
        )
