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

"""Multi-lead-time chunks, batch passthrough and device staging."""

import numpy as np
import pytest
import torch
import xarray as xr
from coupling_toys import (
    ATMOS_SHAPE,
    OCEAN_SHAPE,
    ToyAtmosphere,
    ToyOcean,
    ToyPrognostic,
    toy_grid,
    toy_source,
)

from earth2studio.coupling import Connector, Coupler, CouplingError
from earth2studio.run import WorkItem
from earth2studio.utils import coord_array_like

T0 = np.datetime64("2026-01-01T00")
GRID = toy_grid(*ATMOS_SHAPE)


def _hours(*values: int) -> np.ndarray:
    return np.array(values, dtype="timedelta64[h]")


class _Chunked(ToyPrognostic):
    """12 h steps publishing two lead times (+6h, +12h) at once, like DLWP."""

    def __init__(self):
        super().__init__(["a"], step="12h")

    def output_coords(self, input_coords):
        lead = input_coords["lead_time"].values[-1] + _hours(6, 12)
        return coord_array_like(input_coords, {"lead_time": lead})

    def _advance(self, x, forcing=None):
        last = x.isel(lead_time=[-1])
        chunk = xr.concat([last + 0.5, last + 1.0], "lead_time")
        leads = last["lead_time"].values[-1] + _hours(6, 12)
        return chunk.assign_coords(lead_time=leads).transpose(*x.dims)


def test_multi_lead_chunks_are_published_and_served_per_lead():
    a = _Chunked()
    b = ToyPrognostic(
        ["b"], forcing=["a"], advance=lambda x, f: x + f.isel(lead_time=-1).values
    )
    coupler = Coupler(
        {"a": a, "b": b},
        [Connector("a", "b")],
        {"a": toy_source(GRID, 0.0), "b": toy_source(GRID, 0.0)},
    )
    item = WorkItem(T0, np.timedelta64(24, "h"))
    schema = coupler.output_coords(item.final_lead_time)["a"]
    np.testing.assert_array_equal(schema["lead_time"], _hours(0, 6, 12, 18, 24))
    steps = list(coupler.run_item(item))
    # The chunked model appears at 12 h and 24 h with both leads of its step.
    assert [("a" in s) for s in steps] == [True, False, True, False, True]
    np.testing.assert_array_equal(steps[2]["a"]["lead_time"], _hours(6, 12))
    np.testing.assert_array_equal(steps[4]["a"]["lead_time"], _hours(18, 24))
    # b wants a at its own input time: 0h (IC), then 6h is held from the IC
    # since a has not advanced yet, then the 12 h and 18 h frames exist.
    assert b.calls == [("a", [0]), ("a", [6]), ("a", [12]), ("a", [18])]
    # a's frames: 0 (IC), 0.5 (6h), 1.0 (12h); the 18 h frame appears only after
    # b has advanced at 24 h, so b holds the 12 h value there.
    values = [float(s["b"].isel(lat=1, lon=1).values.ravel()[0]) for s in steps]
    np.testing.assert_allclose(values, [0, 0, 0, 1.0, 2.0])


def test_batch_dimension_passes_through_unchanged():
    ic = xr.DataArray(
        np.stack([np.full((1, 1, *ATMOS_SHAPE), v, dtype=np.float32) for v in (0, 10)]),
        dims=("batch", "lead_time", "variable", "lat", "lon"),
        coords={
            "lead_time": _hours(0),
            "variable": ["z1000"],
            **GRID.coords(only_index=True),
        },
    )
    coupler = Coupler(
        {"atmos": ToyAtmosphere(), "ocean": ToyOcean()},
        [Connector("ocean", "atmos"), Connector("atmos", "ocean")],
        {"ocean": toy_source(toy_grid(*OCEAN_SHAPE), 2.0)},
    )
    steps = list(
        coupler.run_item(WorkItem(T0, np.timedelta64(6, "h")), ics={"atmos": ic})
    )
    out = steps[1]["atmos"]
    assert out.dims == ("batch", "time", "lead_time", "variable", "lat", "lon")
    np.testing.assert_allclose(out.isel(lat=1, lon=1).values.ravel(), [1.2, 11.2])
    assert coupler.supports_member_batching is False


def test_devices_pin_models_and_stage_fields():
    class _Module(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

    class _TorchToy(ToyPrognostic):
        def __init__(self):
            super().__init__(["z1000"], forcing=["sst"])
            self.module = _Module()

        def parameters(self):
            return self.module.parameters()

    coupler = Coupler(
        {"atmos": ToyAtmosphere(), "ocean": ToyOcean()},
        [Connector("ocean", "atmos"), Connector("atmos", "ocean")],
        {
            "atmos": toy_source(GRID, 0.0),
            "ocean": toy_source(toy_grid(*OCEAN_SHAPE), 2.0),
        },
        devices={"ocean": "cpu"},
    )
    assert coupler.device_of("ocean") == torch.device("cpu")
    assert coupler.device_of("atmos") == torch.device("cpu")  # no parameters
    coupler.to("cpu")
    steps = list(coupler.run_item(WorkItem(T0, np.timedelta64(48, "h"))))
    assert isinstance(steps[8]["ocean"].data, np.ndarray)
    with pytest.raises(CouplingError, match="devices names unknown"):
        Coupler(
            {"atmos": ToyAtmosphere()},
            sources={"atmos": (toy_source(GRID, 0), toy_source(GRID, 2))},
            devices={"zz": "cpu"},
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda missing")
def test_cpu_pinned_model_receives_numpy_from_a_cuda_model():
    coupler = Coupler(
        {"atmos": ToyAtmosphere(), "ocean": ToyOcean()},
        [Connector("ocean", "atmos"), Connector("atmos", "ocean")],
        {
            "atmos": toy_source(GRID, 0.0),
            "ocean": toy_source(toy_grid(*OCEAN_SHAPE), 2.0),
        },
        devices={"atmos": "cuda:0", "ocean": "cpu"},
    )
    steps = list(coupler.run_item(WorkItem(T0, np.timedelta64(48, "h"))))
    assert isinstance(steps[8]["ocean"].data, np.ndarray)
    assert not isinstance(steps[8]["atmos"].data, np.ndarray)
