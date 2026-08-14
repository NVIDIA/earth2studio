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

"""Tests for the UX layer: couple() auto-wiring, coupled(), describe().

The auto-wired toy system must reproduce the hand-computed expectations of
test_driver.py: z(96h) = 19.2336 and sst(96h) = 2.180147 — with the derived
import carried by a windowed connector instead of a mediator.
"""

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.nvcoupler.api import couple, coupled, describe, describe_html
from earth2studio.nvcoupler.component import CallableComponent
from earth2studio.nvcoupler.errors import (
    AmbiguousCouplingError,
    UnmatchedImportError,
)
from earth2studio.nvcoupler.mediator import AccumulationMediator
from earth2studio.nvcoupler.testing import (
    ATMOS_GRID,
    OCEAN_GRID,
    atmos_ic,
    fake_atmos,
    fake_ocean,
    ocean_ic,
)

T0 = "2024-01-01"
T96 = "2024-01-05"


def test_couple_synthesizes_windowed_connector():
    # No mediator: couple() carries the derived import on a windowed
    # connector built from the CellMethod of geopotential_at_1000hpa_48h_mean.
    driver = couple(fake_atmos(), fake_ocean(), start=T0, stop=T96)

    assert not any(
        isinstance(c, AccumulationMediator) for c in driver.components.values()
    )
    conn = driver._connectors[("atmos", "ocean")]
    assert conn.window == np.timedelta64(48, "h").astype("timedelta64[ns]")
    assert conn.reduce == "mean"

    # dt defaults to the GCD of 6h and 48h
    assert driver.clock.dt == np.timedelta64(6, "h").astype("timedelta64[ns]")

    # sequence is derived from the graph in the canonical lagged shape
    assert driver.sequence_derived
    expected = "\n".join(
        [
            "@6h",
            "  atmos -> ocean",
            "  ocean -> atmos",
            "  atmos",
            "@48h",
            "  ocean",
            "@",
        ]
    )
    assert str(driver.sequence) == expected

    driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
    driver.run()
    z = driver.components["atmos"].export_state["geopotential_at_1000hpa"]
    assert torch.allclose(z.data, torch.full(ATMOS_GRID, 19.2336), atol=1e-4)
    sst = driver.components["ocean"].export_state["sea_surface_temperature"]
    assert torch.allclose(sst.data, torch.full(OCEAN_GRID, 2.180147), atol=1e-4)


def test_couple_synthesizes_mediator_when_pair_also_transfers_plainly():
    """When the (src, dst) pair already carries a plain transfer, the derived
    import cannot share the connector — a mediator is genuinely needed."""

    def step(x, coords):
        return x, coords

    greedy = CallableComponent(
        "greedy",
        step,
        timestep="48h",
        imports=["geopotential_at_1000hpa", "geopotential_at_1000hpa_48h_mean"],
        exports=["sea_surface_temperature"],
        variable_aliases={
            "z1000": "geopotential_at_1000hpa",
            "z48m": "geopotential_at_1000hpa_48h_mean",
        },
    )
    driver = couple(fake_atmos(), greedy, start=T0, stop=T96)
    mediators = [
        c for c in driver.components.values() if isinstance(c, AccumulationMediator)
    ]
    assert len(mediators) == 1
    med = mediators[0]
    assert med.export_names == ["geopotential_at_1000hpa_48h_mean"]
    # plain z1000 rides the direct connector; the mean goes via the mediator
    assert ("atmos", "greedy") in driver._connectors
    assert ("atmos", med.name) in driver._connectors
    assert (med.name, "greedy") in driver._connectors
    assert driver._connectors[("atmos", "greedy")].window is None


def test_ambiguous_exports_raise():
    def step(x, coords):
        return x, coords

    ocean2 = CallableComponent(
        "ocean2",
        step,
        timestep="48h",
        imports=["geopotential_at_1000hpa_48h_mean"],
        exports=["sea_surface_temperature"],
        variable_aliases={"z48m": "geopotential_at_1000hpa_48h_mean"},
    )
    with pytest.raises(AmbiguousCouplingError, match="sea_surface_temperature"):
        couple(fake_atmos(), fake_ocean(), ocean2, start=T0, stop=T96)


def test_unmatched_import_raises_with_available_exports():
    def step(x, coords):
        return x, coords

    # imports air_temperature_2m: no exporter, and no cell_method to
    # synthesize a mediator from
    lonely = CallableComponent(
        "lonely",
        step,
        timestep="6h",
        imports=["air_temperature_2m"],
        exports=["mean_sea_level_pressure"],
    )
    other = CallableComponent(
        "other", step, timestep="6h", exports=["geopotential_at_1000hpa"]
    )
    with pytest.raises(UnmatchedImportError) as err:
        couple(other, lonely, start=T0, stop=T96)
    msg = str(err.value)
    assert "air_temperature_2m" in msg
    assert "geopotential_at_1000hpa" in msg  # available exports listed


def test_unmatched_derived_import_without_base_exporter_raises():
    def step(x, coords):
        return x, coords

    # ocean imports the 48h mean but nobody exports the base field
    with pytest.raises(UnmatchedImportError, match="geopotential_at_1000hpa_48h_mean"):
        couple(
            fake_ocean(),
            CallableComponent(
                "other", step, timestep="6h", exports=["mean_sea_level_pressure"]
            ),
            start=T0,
            stop=T96,
        )


def test_describe_pre_initialize():
    driver = couple(fake_atmos(), fake_ocean(), start=T0, stop=T96)
    text = describe(driver)  # must work BEFORE initialize
    assert "atmos" in text
    assert "ocean" in text
    assert "6h" in text
    assert "2D" in text  # 48h cadence formatted by fmt_timedelta
    assert "->" in text
    assert "sea_surface_temperature" in text
    assert "constant" in text  # time policy column
    assert "lagged" in text
    # run sequence section present
    assert "@6h" in text and "@48h" in text

    html = describe_html(driver)
    assert "atmos" in html and "ocean" in html
    assert "nvc-box" in html and "&rarr;" in html

    # still works after initialize
    driver.initialize({"atmos": atmos_ic(), "ocean": ocean_ic()})
    assert "atmos" in describe(driver)


def test_describe_labels_mediator_delivery_sequential():
    """Mode is per exchange: med -> ocean follows med.compute in the same
    slot, so ocean consumes state produced in this very iteration."""
    from earth2studio.nvcoupler.clock import Clock
    from earth2studio.nvcoupler.driver import Driver
    from earth2studio.nvcoupler.mediator import TrailingAverageMediator

    driver = Driver(
        {
            "atmos": fake_atmos(),
            "ocean": fake_ocean(),
            "med": TrailingAverageMediator(
                "med", ["geopotential_at_1000hpa_48h_mean"]
            ),
        },
        clock=Clock(T0, T96, "6h"),
        connectors=[("atmos", "med"), ("ocean", "atmos"), ("med", "ocean")],
    )
    text = describe(driver)
    rows = [
        line.strip()
        for line in text.splitlines()
        if " -> " in line and ("lagged" in line or "sequential" in line)
    ]

    def mode_of(name: str) -> str:
        row = next(r for r in rows if r.startswith(name))
        return "sequential" if "sequential" in row else "lagged"

    assert mode_of("atmos -> med") == "lagged"
    assert mode_of("ocean -> atmos") == "lagged"
    assert mode_of("med -> ocean") == "sequential"


def test_coupled_end_to_end():
    ds = coupled(
        T0,
        T96,
        [fake_atmos(), fake_ocean()],
        ics={"atmos": atmos_ic(), "ocean": ocean_ic()},
        verbose=False,
    )
    assert isinstance(ds["atmos"], xr.Dataset)
    z = ds["atmos"]["geopotential_at_1000hpa"]
    assert z.dims == ("time", "lat", "lon")
    assert z.shape == (17, 32, 64)
    assert np.allclose(z.values[-1], 19.2336, atol=1e-4)
    sst = ds["ocean"]["sea_surface_temperature"]
    assert sst.shape == (3, 16, 32)
    assert np.allclose(sst.values[-1], 2.180147, atol=1e-4)


def test_coupled_accepts_nsteps_and_dict():
    ds = coupled(
        T0,
        16,  # 16 x 6h = 96h
        {"atmos": fake_atmos(), "ocean": fake_ocean()},
        ics={"atmos": atmos_ic(), "ocean": ocean_ic()},
        verbose=False,
    )
    assert np.allclose(
        ds["atmos"]["geopotential_at_1000hpa"].values[-1], 19.2336, atol=1e-4
    )
