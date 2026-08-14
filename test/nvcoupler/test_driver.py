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

"""End-to-end coupled-system tests on the synthetic atmos/ocean toys.

Hand-computed expectations (gain = 1, atmos z0 = 0, sst0 = 2, dt = 6 h):

- atmos step: z += 1 + 0.1 * sst. With sst = 2 held for the first 48 h,
  z(t) = 1.2 * t/6h, so z(42h) = 8.4 and z(48h) = 9.6.
- mediator window 1 accumulates z at t = 0..42h (lagged transfer before each
  atmos run): mean = 1.2 * (0+..+7)/8 = 4.2.
- ocean at 48 h: sst = 2 + 0.01 * 4.2 = 2.042.
- atmos then steps at 1.2042 for 48..96 h: z(96h) = 9.6 + 8 * 1.2042 = 19.2336.
- mediator window 2: mean of z(48..90h) = (9.6 + 18.0294)/2 = 13.8147.
- ocean at 96 h: sst = 2.042 + 0.138147 = 2.180147.
"""

from collections import OrderedDict
from contextlib import contextmanager

import numpy as np
import pytest
import torch

from earth2studio.nvcoupler.clock import Clock
from earth2studio.nvcoupler.component import CallableComponent
from earth2studio.nvcoupler.driver import Driver
from earth2studio.nvcoupler.errors import CouplingError, UnmatchedImportError
from earth2studio.nvcoupler.mediator import TrailingAverageMediator
from earth2studio.nvcoupler.testing import (
    ATMOS_GRID,
    atmos_ic,
    fake_atmos,
    fake_ocean,
    ocean_ic,
)

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


@contextmanager
def capture_loguru(level="WARNING"):
    """Collect loguru messages emitted inside the block (repo loguru pattern)."""
    from loguru import logger

    messages: list[str] = []
    handler_id = logger.add(messages.append, level=level, format="{message}")
    try:
        yield messages
    finally:
        logger.remove(handler_id)


def make_driver(dsl=LAGGED_DSL, gain_atmos=1.0, gain_ocean=1.0, stop=T96):
    components = {
        "atmos": fake_atmos(gain=gain_atmos),
        "ocean": fake_ocean(gain=gain_ocean),
        "med": TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"]),
    }
    driver = Driver(components, dsl, Clock(T0, stop, "6h"))
    driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
    return driver


def test_cadence_and_hand_computed_values():
    driver = make_driver()
    driver.run()
    atmos = driver.components["atmos"]
    ocean = driver.components["ocean"]
    med = driver.components["med"]
    assert atmos.run_count == 16
    assert ocean.run_count == 2
    assert med.run_count == 2

    z = atmos.export_state["geopotential_at_1000hpa"]
    assert torch.allclose(z.data, torch.full(ATMOS_GRID, 19.2336), atol=1e-4)
    assert z.valid_time == np.datetime64("2024-01-05")

    sst = ocean.export_state["sea_surface_temperature"]
    assert torch.allclose(sst.data, torch.full((16, 32), 2.180147), atol=1e-6)

    zmean = med.export_state["geopotential_at_1000hpa_48h_mean"]
    assert torch.allclose(zmean.data, torch.full(ATMOS_GRID, 13.8147), atol=1e-4)
    assert med.samples_last_window["geopotential_at_1000hpa_48h_mean"] == 8


def test_lagged_vs_sequential_within_slot():
    # Both variants isolate the sst hand-off to the 48h slot; they differ only
    # in whether atmos receives the sst produced in that same slot (sequential)
    # or the one from before the ocean ran (lagged).
    lagged = """
@6h
  atmos -> med
  atmos
@48h
  med.compute
  ocean -> atmos
  med -> ocean
  ocean
@
"""
    sequential = """
@6h
  atmos -> med
  atmos
@48h
  med.compute
  med -> ocean
  ocean
  ocean -> atmos
@
"""
    z_lagged = make_driver(lagged).run()["atmos"]["geopotential_at_1000hpa"].values[-1]
    z_sequential = (
        make_driver(sequential).run()["atmos"]["geopotential_at_1000hpa"].values[-1]
    )
    # lagged: atmos runs 48-96h forced by the IC sst (2.0) -> z96 = 19.2
    # sequential: forced by the fresh sst (2.042) -> z96 = 19.2336
    assert np.allclose(z_lagged, 19.2, atol=1e-4)
    assert np.allclose(z_sequential, 19.2336, atol=1e-4)
    # difference = 8 steps * 0.1 * (2.042 - 2.0) = one coupling window's worth
    assert np.allclose(z_sequential - z_lagged, 8 * 0.1 * 0.042, atol=1e-5)


def test_gradient_flows_across_the_exchange():
    gain_atmos = torch.tensor(1.0, requires_grad=True)
    gain_ocean = torch.tensor(1.0, requires_grad=True)
    driver = make_driver(gain_atmos=gain_atmos, gain_ocean=gain_ocean)
    with torch.enable_grad():
        states = driver.rollout(16)  # full 96h
    loss = states["atmos"]["geopotential_at_1000hpa"].data.sum()
    loss.backward()
    # atmos gain obviously in the graph; ocean gain reaches the loss only
    # THROUGH the exchange: ocean sst -> connector regrid -> atmos injection
    assert gain_atmos.grad is not None and gain_atmos.grad != 0
    assert gain_ocean.grad is not None and gain_ocean.grad != 0


def test_steps_iteration_and_probe():
    driver = make_driver(stop="2024-01-03")  # 48h
    seen = []
    for time, states in driver.steps():
        seen.append(time)
        assert "atmos" in states and "ocean" in states
    assert len(seen) == 8
    transfer = driver.probe("ocean->atmos")
    assert "sea_surface_temperature" in transfer
    assert transfer["sea_surface_temperature"].data.shape == ATMOS_GRID


def test_to_xarray_output():
    ds = make_driver().run()
    z = ds["atmos"]["geopotential_at_1000hpa"]
    assert z.dims == ("time", "lat", "lon")
    assert z.shape == (17, 32, 64)  # IC + 16 rings
    assert np.allclose(z.values[0], 0.0)
    assert np.allclose(z.values[-1], 19.2336, atol=1e-4)
    sst = ds["ocean"]["sea_surface_temperature"]
    assert sst.shape == (3, 16, 32)  # IC + 2 rings
    assert np.allclose(sst.values[-1], 2.180147, atol=1e-6)


def test_missing_ic_raises():
    driver = Driver(
        {
            "atmos": fake_atmos(),
            "ocean": fake_ocean(),
            "med": TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"]),
        },
        LAGGED_DSL,
        Clock(T0, T96, "6h"),
    )
    with pytest.raises(CouplingError, match="initial condition"):
        driver.initialize({"atmos": atmos_ic()})


def test_run_before_initialize_raises():
    driver = Driver(
        {"atmos": fake_atmos()}, "@6h\n  atmos\n@", Clock(T0, "2024-01-02", "6h")
    )
    with pytest.raises(CouplingError, match="initialize"):
        driver.run()


def test_unconsumed_export_warning():
    driver = Driver(
        {"atmos": fake_atmos()},
        "@6h\n  atmos\n@",
        Clock(T0, "2024-01-02", "6h"),
        allow_unfed_imports=True,  # atmos's sst import is deliberately unfed
    )
    with capture_loguru() as messages:
        driver.initialize({"atmos": atmos_ic()})
    assert any("no connector consumes" in m for m in messages)


# -- finding 2b: unfed imports must fail loudly at initialize -------------------
FORGOT_OCEAN_TO_ATMOS_DSL = """
@6h
  atmos -> med
  atmos
@48h
  med.compute
  med -> ocean
  ocean
@
"""


def make_forgetful_driver(**kwargs):
    from earth2studio.nvcoupler.mediator import TrailingAverageMediator

    components = {
        "atmos": fake_atmos(),
        "ocean": fake_ocean(),
        "med": TrailingAverageMediator("med", ["geopotential_at_1000hpa_48h_mean"]),
    }
    return Driver(components, FORGOT_OCEAN_TO_ATMOS_DSL, Clock(T0, T96, "6h"), **kwargs)


def test_forgotten_connector_raises_at_initialize():
    driver = make_forgetful_driver()
    with pytest.raises(
        UnmatchedImportError, match=r"'atmos' imports 'sea_surface_temperature'"
    ):
        driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})


def test_allow_unfed_imports_warns_and_runs():
    driver = make_forgetful_driver(allow_unfed_imports=True)
    with capture_loguru() as messages:
        driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
    assert any(
        "atmos" in m and "sea_surface_temperature" in m and "no connector" in m
        for m in messages
    )
    driver.run()  # runs on stale IC forcing, by explicit opt-in
    assert driver.components["atmos"].run_count == 16


# -- finding 2c: collected records must not pin autograd graphs ------------------
def test_records_are_detached_during_gradient_rollout():
    gain_atmos = torch.tensor(1.0, requires_grad=True)
    gain_ocean = torch.tensor(1.0, requires_grad=True)
    driver = make_driver(gain_atmos=gain_atmos, gain_ocean=gain_ocean)
    with torch.enable_grad():
        states = driver.rollout(16)
    # exchange path stays attached...
    loss = states["atmos"]["geopotential_at_1000hpa"].data.sum()
    loss.backward()
    assert gain_atmos.grad is not None and gain_ocean.grad is not None
    # ...but records (off the exchange path) are detached clones
    for records in driver._records.values():
        for _, fields in records:
            for field in fields.values():
                assert not field.data.requires_grad
                assert field.data.grad_fn is None


# -- finding 2d: exhausted clock fails loudly; reset() enables a rerun -----------
def make_atmos_only_driver():
    driver = Driver(
        {"atmos": fake_atmos()},
        "@6h\n  atmos\n@",
        Clock(T0, "2024-01-02", "6h"),
        allow_unfed_imports=True,
    )
    driver.initialize({"atmos": atmos_ic()})
    return driver


def test_second_run_raises_clock_exhausted():
    driver = make_atmos_only_driver()
    driver.run()
    with pytest.raises(CouplingError, match="exhausted.*reset"):
        driver.run()


def test_steps_after_exhaustion_raises():
    driver = make_atmos_only_driver()
    for _ in driver.steps():
        pass
    with pytest.raises(CouplingError, match="exhausted.*reset"):
        driver.steps()


def test_rollout_exhaustion_raises_coupling_error_not_stopiteration():
    driver = make_atmos_only_driver()
    driver.run()
    with pytest.raises(CouplingError, match="exhausted.*reset"):
        driver.rollout(1)
    # mid-iteration exhaustion: more steps requested than remain
    driver.reset()
    driver.initialize({"atmos": atmos_ic()})
    with pytest.raises(CouplingError, match=r"rollout\(5\)"):
        driver.rollout(5)  # 24h clock has only 4 steps


def test_reset_requires_reinitialize_then_reruns_identically():
    driver = make_atmos_only_driver()
    first = driver.run()
    driver.reset()
    assert all(not records for records in driver._records.values())
    with pytest.raises(CouplingError, match="initialize"):
        driver.run()  # reset invalidates initialization
    driver.initialize({"atmos": atmos_ic()})
    second = driver.run()
    z1 = first["atmos"]["geopotential_at_1000hpa"].values
    z2 = second["atmos"]["geopotential_at_1000hpa"].values
    assert np.array_equal(z1, z2)
    assert z1.shape[0] == 5  # IC + 4 rings, not stale/duplicated


# -- finding 2a: fields carrying their own size-1 time/batch coords ---------------
def make_time_coord_component(n_time: int = 1):
    """A component whose published fields keep size-1 (or larger) time and
    batch dims — the DLESyM-split style of model coords."""

    def step(x, coords):
        return x + 1.0, coords

    comp = CallableComponent(
        "comp",
        step,
        timestep="6h",
        exports=["geopotential_at_1000hpa"],
    )
    coords = OrderedDict(
        {
            "batch": np.array([0]),
            "time": np.array([np.datetime64(T0, "ns")] * n_time),
            "variable": np.array(["z1000"]),
            "lat": np.linspace(90.0, -90.0, 8),
            "lon": np.linspace(0.0, 360.0, 16, endpoint=False),
        }
    )
    x = torch.zeros(1, n_time, 1, 8, 16)
    return comp, (x, coords)


def test_to_xarray_with_field_time_coord_uses_ring_times():
    comp, ic = make_time_coord_component()
    driver = Driver({"comp": comp}, "@6h\n  comp\n@", Clock(T0, "2024-01-01T12", "6h"))
    driver.initialize({"comp": ic})
    ds = driver.run()
    z = ds["comp"]["geopotential_at_1000hpa"]
    # squeezed batch/time dims, ring-time axis prepended
    assert z.dims == ("time", "lat", "lon")
    assert z.shape == (3, 8, 16)
    # RING times, not the stale IC time carried on every field
    expected = np.datetime64(T0, "ns") + np.arange(3) * np.timedelta64(6, "h")
    assert np.array_equal(z["time"].values, expected)
    assert np.allclose(z.values, np.arange(3)[:, None, None])


def test_non_size1_time_dim_raises():
    comp, ic = make_time_coord_component(n_time=2)
    driver = Driver({"comp": comp}, "@6h\n  comp\n@", Clock(T0, "2024-01-01T12", "6h"))
    driver.initialize({"comp": ic})
    with pytest.raises(CouplingError, match="'time' dimension of size 2"):
        driver.run()
