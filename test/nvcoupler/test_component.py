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

from collections import OrderedDict

import numpy as np
import pytest
import torch

from earth2studio.nvcoupler.clock import Clock
from earth2studio.nvcoupler.component import (
    ConditioningKwargAdapter,
    ExtraTensorAdapter,
    PrognosticComponent,
    VariableOverwriteAdapter,
)
from earth2studio.nvcoupler.errors import CadenceError, CouplingError
from earth2studio.nvcoupler.field import Field, State
from earth2studio.nvcoupler.testing import (
    atmos_ic,
    fake_atmos,
    fake_ocean,
    grid_coords,
    ocean_ic,
)


def _sst_import(value=3.0, nlat=32, nlon=64):
    return State(
        "imports",
        [
            Field(
                torch.full((nlat, nlon), value),
                grid_coords(nlat, nlon),
                "sea_surface_temperature",
                "K",
            )
        ],
    )


class TestVariableOverwriteAdapter:
    def test_inject_overwrites_slice(self):
        x, coords = atmos_ic(z0=0.0, sst0=2.0)
        out = VariableOverwriteAdapter.inject(
            x, coords, _sst_import(5.0), {"sea_surface_temperature": "sst"}
        )
        assert torch.all(out[1] == 5.0)  # sst slice overwritten
        assert torch.all(out[0] == 0.0)  # z1000 untouched
        assert torch.all(x[1] == 2.0)  # original not mutated

    def test_missing_variable_raises(self):
        x, coords = atmos_ic()
        with pytest.raises(CouplingError, match="not a state variable"):
            VariableOverwriteAdapter.inject(
                x, coords, _sst_import(), {"sea_surface_temperature": "nope"}
            )

    def test_gradient_flows_through_injection(self):
        x, coords = atmos_ic()
        sst = torch.full((32, 64), 5.0, requires_grad=True)
        imports = State(
            "imports",
            [Field(sst, grid_coords(32, 64), "sea_surface_temperature", "K")],
        )
        out = VariableOverwriteAdapter.inject(
            x, coords, imports, {"sea_surface_temperature": "sst"}
        )
        out.sum().backward()
        assert sst.grad is not None and torch.all(sst.grad == 1.0)


class TestOtherAdapters:
    def test_conditioning_kwarg(self):
        captured = {}

        class Model:
            def call_with_conditioning(
                self, x, coords, conditioning, conditioning_coords
            ):
                captured["conditioning"] = conditioning
                captured["coords"] = conditioning_coords
                return x, coords

        x, coords = atmos_ic()
        adapter = ConditioningKwargAdapter()
        adapter(Model(), x, coords, _sst_import(7.0), {})
        assert torch.all(captured["conditioning"] == 7.0)
        assert list(captured["coords"]) == ["variable", "lat", "lon"]

    def test_extra_tensor(self):
        captured = {}

        def model(x, coords, coupling):
            captured["coupling"] = coupling
            return x, coords

        x, coords = atmos_ic()
        ExtraTensorAdapter()(model, x, coords, _sst_import(9.0), {})
        assert torch.all(captured["coupling"] == 9.0)

    def test_multiple_imports_require_explicit_field_order(self):
        # channel order is model-sensitive; alphabetical stacking would run
        # fine and predict garbage, so >1 import without field_order= must fail
        from earth2studio.nvcoupler.field import Field
        from earth2studio.nvcoupler.testing import grid_coords

        imports = State(
            "imports",
            [
                Field(
                    torch.ones(8, 16),
                    grid_coords(8, 16),
                    "sea_surface_temperature",
                    "K",
                ),
                Field(torch.ones(8, 16), grid_coords(8, 16), "air_temperature_2m", "K"),
            ],
        )
        x, coords = atmos_ic()
        model = lambda x, coords, coupling: (x, coords)  # noqa: E731
        with pytest.raises(CouplingError, match="field_order"):
            ExtraTensorAdapter()(model, x, coords, imports, {})
        # explicit order works, and is honored (not alphabetical)
        captured = {}

        def capture(x, coords, coupling):
            captured["coupling"] = coupling
            return x, coords

        ExtraTensorAdapter(
            field_order=["sea_surface_temperature", "air_temperature_2m"]
        )(capture, x, coords, imports, {})
        assert captured["coupling"].shape[0] == 2
        # unknown name in field_order is rejected
        with pytest.raises(CouplingError, match="not in the"):
            ExtraTensorAdapter(field_order=["nope"])(model, x, coords, imports, {})


class TestCallableComponent:
    def test_toy_atmos_step_arithmetic(self):
        atmos = fake_atmos()
        clock = Clock("2024-01-01", "2024-01-02", "6h")
        atmos.realize(clock)
        atmos.initialize(*atmos_ic(z0=0.0, sst0=2.0))
        # export seeded at t0 for lagged coupling
        assert atmos.export_state["geopotential_at_1000hpa"].valid_time == clock.start

        t1 = clock.advance()
        atmos.run(t1)
        # z = 0 + 1 + 0.1*2 = 1.2 (no import set; state sst used)
        z = atmos.export_state["geopotential_at_1000hpa"]
        assert torch.allclose(z.data, torch.full((32, 64), 1.2))
        assert z.valid_time == t1 and z.source == "atmos"

        # inject an import and step again: z = 1.2 + 1 + 0.1*10 = 3.2
        atmos.import_state.add(_sst_import(10.0)["sea_surface_temperature"])
        atmos.run(clock.advance())
        assert torch.allclose(
            atmos.export_state["geopotential_at_1000hpa"].data,
            torch.full((32, 64), 3.2),
        )

    def test_ocean_mask_export(self):
        ocean = fake_ocean(with_mask=True)
        ocean.realize(Clock("2024-01-01", "2024-01-03", "48h"))
        ocean.initialize(*ocean_ic())
        sst = ocean.export_state["sea_surface_temperature"]
        assert sst.mask is not None
        assert not sst.mask[0, 0] and sst.mask[-1, -1]

    def test_cadence_validation(self):
        atmos = fake_atmos(timestep="6h")
        with pytest.raises(CadenceError):
            atmos.realize(Clock("2024-01-01", "2024-01-02", "4h"))

    def test_run_before_initialize_raises(self):
        atmos = fake_atmos()
        atmos.realize(Clock("2024-01-01", "2024-01-02", "6h"))
        with pytest.raises(CouplingError, match="not initialized"):
            atmos.run(np.datetime64("2024-01-01T06:00"))

    def test_advertise(self):
        ocean = fake_ocean()
        imports, exports = ocean.advertise()
        assert imports == ["geopotential_at_1000hpa_48h_mean"]
        assert exports == ["sea_surface_temperature"]


class MockPrognostic:
    """Minimal PrognosticModel: two variables, 6h step, +1 per step."""

    def __init__(self):
        self._in = OrderedDict(
            {
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(["z1000", "sst"]),
                **grid_coords(8, 16),
            }
        )

    def input_coords(self):
        return OrderedDict({k: v.copy() for k, v in self._in.items()})

    def output_coords(self, input_coords):
        out = OrderedDict({k: v.copy() for k, v in input_coords.items()})
        out["lead_time"] = input_coords["lead_time"] + np.timedelta64(6, "h")
        return out

    def __call__(self, x, coords):
        return x + 1.0, self.output_coords(coords)

    def to(self, device):
        return self


class TestPrognosticComponent:
    def test_timestep_and_exports_inferred(self):
        comp = PrognosticComponent("mock", MockPrognostic())
        assert comp.timestep == np.timedelta64(6, "h")
        assert set(comp.export_names) == {
            "geopotential_at_1000hpa",
            "sea_surface_temperature",
        }

    def test_rollout_and_publish(self):
        comp = PrognosticComponent(
            "mock", MockPrognostic(), imports=["sea_surface_temperature"]
        )
        clock = Clock("2024-01-01", "2024-01-02", "6h")
        comp.realize(clock)
        ic = MockPrognostic().input_coords()
        comp.initialize(torch.zeros(1, 2, 8, 16), ic)
        for t in clock:
            comp.run(t)
        assert comp.run_count == 4
        z = comp.export_state["geopotential_at_1000hpa"]
        # exports are exchange-shaped: singleton lead_time squeezed away
        assert list(z.coords) == ["lat", "lon"]
        assert z.data.shape == (8, 16)
        assert torch.all(z.data == 4.0)
        assert z.valid_time == np.datetime64("2024-01-02")
        # the internal model state keeps the full model dims
        x, coords = comp.state
        assert list(coords) == ["lead_time", "variable", "lat", "lon"]
        assert x.shape == (1, 2, 8, 16)

    def test_multi_window_needs_next_input(self):
        class TwoWindow(MockPrognostic):
            def output_coords(self, input_coords):
                out = super().output_coords(input_coords)
                out["lead_time"] = np.array(
                    [np.timedelta64(6, "h"), np.timedelta64(12, "h")]
                )
                return out

            def __call__(self, x, coords):
                return torch.cat([x, x]), self.output_coords(coords)

        comp = PrognosticComponent("two", TwoWindow(), timestep="6h")
        comp.realize(Clock("2024-01-01", "2024-01-02", "6h"))
        comp.initialize(torch.zeros(1, 2, 8, 16), MockPrognostic().input_coords())
        with pytest.raises(CouplingError, match="next_input"):
            comp.run(np.datetime64("2024-01-01T06:00"))

    def test_publish_missing_export_raises(self):
        comp = PrognosticComponent(
            "mock", MockPrognostic(), exports=["air_temperature_2m"]
        )
        comp.realize(Clock("2024-01-01", "2024-01-02", "6h"))
        with pytest.raises(CouplingError, match="advertises export"):
            comp.initialize(torch.zeros(1, 2, 8, 16), MockPrognostic().input_coords())
