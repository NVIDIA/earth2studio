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

"""State connectors and per-component devices."""

from collections.abc import Mapping

import numpy as np
import pytest
import torch
from test_coupling_toys import (
    ATMOS_GRID,
    OCEAN_GRID,
    TO_ATMOS,
    TO_OCEAN,
    Double,
    Fill,
    H,
    Mixed,
    _Toy,
    atmosphere,
    ocean,
    run,
    series,
    toy_runner,
)

from earth2studio.coupling import Connector, CoupledRunner, CouplingError


def _state_coupled() -> CoupledRunner:
    # The atmosphere starts from SST 5; the ocean's SST 2 replaces it.
    return CoupledRunner(
        {"atmos": Mixed(), "ocean": ocean()},
        [
            Connector(
                "ocean",
                "atmos",
                state=True,
                variables=["sst"],
                time="hold",
                regridder=TO_ATMOS,
            ),
            Connector("atmos", "ocean", regridder=TO_OCEAN),
        ],
        sources={
            "atmos": Fill(ATMOS_GRID, {"z1000": 0.0, "sst": 5.0}),
            "ocean": Fill(OCEAN_GRID, {"sst": 2.0}),
        },
    )


# -- state connectors ----------------------------------------------------------


def test_state_overwrite_matches_forcing_coupling(
    ic_sources: Mapping[str, Fill],
) -> None:
    streams = run(_state_coupled(), 96)
    z = {
        lead: float(x.sel(variable="z1000").mean())
        for lead, x in zip(
            [int(x["lead_time"].values[0] // H) for x in streams["atmos"]],
            streams["atmos"],
        )
    }
    expected = series(run(toy_runner(ic_sources), 96)["atmos"])
    assert list(z) == list(expected)
    for lead, value in expected.items():
        np.testing.assert_allclose(z[lead], value, atol=1e-4)
    # The atmosphere publishes its own state: SST 5 initially, then the ocean's.
    sst = [float(x.sel(variable="sst").mean()) for x in streams["atmos"]]
    assert sst[0] == 5.0
    np.testing.assert_allclose(sst[-1], 2.042, atol=1e-6)


def test_state_overwrite_is_described() -> None:
    plan = _state_coupled().describe(np.timedelta64(12, "h"))
    assert "ocean -> atmos state input[0] (hold, regridded): sst" in plan
    assert "state input[0] <- ocean at +6h from +0h" in plan


@pytest.mark.parametrize(
    "connectors, match",
    [
        (
            [Connector("ocean", "atmos", state=True, variables=["sst"], slot=1)],
            "has no state slot 1",
        ),
        (
            [Connector("ocean", "atmos", state=True, variables=["sic"])],
            r"holds no state variables \['sic'\]",
        ),
        (
            [
                Connector(
                    "ocean", "atmos", state=True, variables=["sst"], regridder=TO_ATMOS
                ),
                Connector(
                    "ocean",
                    "atmos",
                    state=True,
                    variables={"sst": "sst"},
                    regridder=TO_ATMOS,
                ),
            ],
            r"both overwrite \['sst'\]",
        ),
        (
            [Connector("ocean", "atmos", state=True, variables=["sst"])],
            "grids differ; pass a regridder",
        ),
    ],
)
def test_state_connectors_are_validated(
    connectors: list[Connector], match: str
) -> None:
    with pytest.raises(CouplingError, match=match):
        CoupledRunner({"atmos": Mixed(), "ocean": ocean()}, connectors)


def test_state_connectors_name_their_variables() -> None:
    with pytest.raises(CouplingError, match="must name the state variables"):
        Connector("ocean", "atmos", state=True)
    with pytest.raises(CouplingError, match="mapping or a sequence"):
        Connector("ocean", "atmos", variables="sst")


def test_diagnostics_have_no_state() -> None:
    with pytest.raises(CouplingError, match="has no state slot 0"):
        CoupledRunner(
            {"atmos": atmosphere(), "double": Double()},
            [
                Connector("atmos", "double"),
                Connector("atmos", "double", state=True, variables=["z1000"]),
            ],
        )


# -- devices -------------------------------------------------------------------


class _Moved(_Toy):
    def to(self, device: torch.device) -> "_Moved":
        self.moved = device
        return self


def test_pinned_components_stay_on_their_device(
    ic_sources: Mapping[str, Fill],
) -> None:
    atmos = _Moved("z1000", "sst", ATMOS_GRID, 6, 1.0, 0.1)
    impact = _Moved("sst", "z1000:mean:48h", ATMOS_GRID, 48, 0.0, 0.01, 48)
    runner = CoupledRunner(
        {"atmos": atmos, "ocean": impact},
        [Connector("ocean", "atmos", time="hold"), Connector("atmos", "ocean")],
        sources={"atmos": ic_sources["atmos"], "ocean": ic_sources["atmos"]},
        devices={"ocean": "cpu"},
    )
    runner.to("meta")
    assert atmos.moved == torch.device("meta")
    assert not hasattr(impact, "moved")
    assert runner.device_of("ocean") == torch.device("cpu")
    assert runner.device_of("atmos") == torch.device("meta")
    with pytest.raises(CouplingError, match="unknown component 'oceans'"):
        CoupledRunner({"ocean": ocean()}, devices={"oceans": "cpu"})
