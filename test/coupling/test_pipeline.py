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

"""CoupledRunner under Pipeline: one store per component, any number of items."""

from collections.abc import Mapping

import numpy as np
from conftest import ATMOS_GRID, OCEAN_GRID, T0, Fill, atmosphere, ocean

from earth2studio.coupling import Connector, CoupledRunner
from earth2studio.grids import NearestRegridder
from earth2studio.io import KVBackend
from earth2studio.run import Pipeline, work_items


def test_pipeline_writes_one_store_per_component(
    ic_sources: Mapping[str, Fill],
) -> None:
    runner = CoupledRunner(
        {"atmos": atmosphere(), "ocean": ocean()},
        [
            Connector(
                "ocean",
                "atmos",
                time="hold",
                regridder=NearestRegridder(OCEAN_GRID, ATMOS_GRID),
            ),
            Connector(
                "atmos", "ocean", regridder=NearestRegridder(ATMOS_GRID, OCEAN_GRID)
            ),
        ],
        sources=dict(ic_sources),
    )
    atmos, ocean_store = KVBackend(), KVBackend()
    times = [T0, T0 + np.timedelta64(24, "h")]
    Pipeline(runner, output={"atmos": atmos, "ocean": ocean_store}, device="cpu").run(
        work_items(times, np.timedelta64(96, "h"))
    )
    z, sst = atmos["z1000"], ocean_store["sst"]
    assert tuple(z.shape) == (2, 17, 2, 4)
    assert tuple(sst.shape) == (2, 3, 1, 2)
    for i in range(2):
        np.testing.assert_allclose(z[i, -1].numpy(), 19.2336, atol=1e-4)
        np.testing.assert_allclose(
            sst[i, :, 0, 0].numpy(), [2.0, 2.042, 2.180147], atol=1e-6
        )
    leads = atmos["lead_time"].values.astype("timedelta64[h]").astype(int)
    np.testing.assert_array_equal(leads, np.arange(0, 97, 6))
