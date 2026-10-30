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

from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.models.conformance import (
    ContractException,
    _evaluate_diagnostic,
    _evaluate_prognostic,
    check_diagnostic_contract,
    check_prognostic_contract,
    iter_contract_rules,
)
from earth2studio.models.dx import Identity
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import coord_array, coord_array_like, handshake_dataarray
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem

DT = np.timedelta64(6, "h")


class ToyPrognostic(torch.nn.Module, PrognosticMixin):
    def input_coords(self) -> CoordinateSystem:
        return coord_array(
            ("batch", "lead_time", "variable", "lat", "lon"),
            {
                "lead_time": [DT * 0],
                "variable": ["t2m"],
                "lat": [1, 0],
                "lon": [0, 1, 2],
            },
            dynamic=("batch",),
        )

    def output_coords(self, x: CoordinateSystem) -> CoordinateSystem:
        lead = x.lead_time.values
        if lead.dtype.kind != "m" or lead.size != 1 or np.isnat(lead).any():
            raise ValueError("expected one finite lead")
        handshake_dataarray(
            x.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
        return coord_array_like(x, {"lead_time": lead + DT})

    def _forward(self, x: xr.DataArray) -> xr.DataArray:
        return from_torch(x.e2s.to_torch()[0] + 1, self.output_coords(x))

    def initialize(self, x: xr.DataArray) -> tuple[xr.DataArray, None]:
        return self._forward(x), None

    def step(self, y: xr.DataArray, state: None) -> tuple[xr.DataArray, None]:
        return self._forward(y), None

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        return self._default_call(x)

    def create_iterator(
        self, x: xr.DataArray
    ) -> Generator[xr.DataArray, xr.DataArray | tuple[xr.DataArray, ...] | None, None]:
        return self._default_create_iterator(x)


@dataclass
class NoiseState:
    rng: torch.Tensor


class StochasticToy(ToyPrognostic):
    stochastic = True

    def set_rng(self, seed: int, reset: bool = True) -> None:
        if reset or not hasattr(self, "rng"):
            self.rng = torch.Generator().manual_seed(seed).get_state()

    def initialize(self, x: xr.DataArray) -> tuple[xr.DataArray, NoiseState]:
        return self.step(x, NoiseState(self.rng.clone()))

    def step(
        self, y: xr.DataArray, state: NoiseState
    ) -> tuple[xr.DataArray, NoiseState]:
        generator = torch.Generator().set_state(state.rng)
        noise = torch.randn(y.shape, generator=generator)
        output = from_torch(y.e2s.to_torch()[0] + noise, self.output_coords(y))
        return output, NoiseState(generator.get_state())


class ForcedToy(ToyPrognostic):
    def forcing_coords(self) -> tuple[CoordinateSystem, CoordinateSystem]:
        static = coord_array(("variable",), {"variable": ["z"]})
        dynamic = coord_array_like(self.input_coords(), {"lead_time": [-DT, DT * 0]})
        return static, dynamic

    def initialize(
        self, x: xr.DataArray, static: xr.DataArray, forcing: xr.DataArray
    ) -> tuple[xr.DataArray, xr.DataArray]:
        handshake_dataarray(static, self.forcing_coords()[0])
        handshake_dataarray(forcing, self.forcing_coords()[1])
        return self._forward(x), static.copy(deep=True)

    def step(
        self, y: xr.DataArray, forcing: xr.DataArray, state: xr.DataArray
    ) -> tuple[xr.DataArray, xr.DataArray]:
        expected = coord_array_like(y)
        handshake_dataarray(forcing, expected)
        return self._forward(y), state.copy(deep=True)

    def __call__(
        self, x: xr.DataArray, static: xr.DataArray, forcing: xr.DataArray
    ) -> xr.DataArray:
        return self._default_call(x, static, forcing)

    def create_iterator(
        self, x: xr.DataArray, static: xr.DataArray, forcing: xr.DataArray
    ) -> Generator[xr.DataArray, xr.DataArray | tuple[xr.DataArray, ...] | None, None]:
        return self._default_create_iterator(x, static, forcing)


class MultiDiagnostic(Identity):
    stochastic = False

    def input_coords(self) -> tuple[CoordinateSystem, CoordinateSystem]:
        return ToyPrognostic().input_coords(), coord_array(
            ("variable",), {"variable": ["z"]}
        )

    def output_coords(
        self, atmosphere: CoordinateSystem, static: CoordinateSystem
    ) -> tuple[CoordinateSystem, ...]:
        inputs = (atmosphere, static)
        for x, declaration in zip(inputs, self.input_coords()):
            handshake_dataarray(x, declaration)
        return tuple(coord_array_like(x) for x in inputs)

    def __call__(
        self, atmosphere: xr.DataArray, static: xr.DataArray
    ) -> tuple[xr.DataArray, xr.DataArray]:
        return atmosphere.copy(deep=True), static.copy(deep=True)


class MultiPrognostic(ToyPrognostic):
    def input_coords(self) -> tuple[CoordinateSystem, CoordinateSystem]:
        fine = super().input_coords()
        return fine, coord_array_like(fine, {"lat": [0]})

    def output_coords(
        self, fine: CoordinateSystem, coarse: CoordinateSystem
    ) -> tuple[CoordinateSystem, ...]:
        inputs = (fine, coarse)
        outputs = []
        for x, declared in zip(inputs, self.input_coords()):
            lead = x.lead_time.values
            handshake_dataarray(x.assign_coords(lead_time=lead - lead[-1]), declared)
            outputs.append(coord_array_like(x, {"lead_time": lead + DT}))
        return tuple(outputs)

    def initialize(
        self, fine: xr.DataArray, coarse: xr.DataArray
    ) -> tuple[tuple[xr.DataArray, ...], None]:
        coords = self.output_coords(fine, coarse)
        return (
            tuple(
                from_torch(x.e2s.to_torch()[0] + 1, c)
                for x, c in zip((fine, coarse), coords)
            ),
            None,
        )

    def step(
        self, fine: xr.DataArray, coarse: xr.DataArray, state: None
    ) -> tuple[tuple[xr.DataArray, ...], None]:
        return self.initialize(fine, coarse)

    def __call__(
        self, fine: xr.DataArray, coarse: xr.DataArray
    ) -> tuple[xr.DataArray, ...]:
        return self._default_call(fine, coarse)

    def create_iterator(
        self, fine: xr.DataArray, coarse: xr.DataArray
    ) -> Generator[
        tuple[xr.DataArray, ...], xr.DataArray | tuple[xr.DataArray, ...] | None, None
    ]:
        return self._default_create_iterator(fine, coarse)


class ChunkedToy(ToyPrognostic):
    def output_coords(self, x: CoordinateSystem) -> CoordinateSystem:
        super().output_coords(x)
        return coord_array_like(
            x, {"lead_time": x.lead_time.values[-1] + np.array([DT, 2 * DT])}
        )

    def _forward(self, x: xr.DataArray) -> xr.DataArray:
        last = x.isel(lead_time=[-1])
        data = last.e2s.to_torch()[0]
        return from_torch(
            torch.cat((data + 1, data + 2), dim=last.get_axis_num("lead_time")),
            self.output_coords(last),
        )


class StochasticDiagnostic(MultiDiagnostic):
    stochastic = True

    def set_rng(self, seed: int, reset: bool = True) -> None:
        if reset or not hasattr(self, "rng"):
            self.rng = torch.Generator().manual_seed(seed)

    def __call__(
        self, atmosphere: xr.DataArray, static: xr.DataArray
    ) -> tuple[xr.DataArray, ...]:
        return tuple(
            from_torch(
                x.e2s.to_torch()[0] + torch.randn(x.shape, generator=self.rng),
                coord_array_like(x),
            )
            for x in (atmosphere, static)
        )


def violations(model, diagnostic=False, **kwargs):
    checker = check_diagnostic_contract if diagnostic else check_prognostic_contract
    with pytest.raises(ContractException) as error:
        checker(model, **kwargs)
    return {v.split(":")[0] for v in error.value.violations}


@pytest.mark.parametrize(
    "factory", [ToyPrognostic, ForcedToy, MultiPrognostic, ChunkedToy]
)
def test_forecasts_only_reference(factory):
    model = factory()
    hooks = model.front_hook, model.rear_hook
    assert check_prognostic_contract(model) == [
        "P14: model does not declare itself stochastic",
        "P21: checkpoint serialization requires component-specific tests",
    ]
    assert (model.front_hook, model.rear_hook) == hooks


def test_stochastic_explicit_state():
    assert check_prognostic_contract(StochasticToy()) == [
        "P21: checkpoint serialization requires component-specific tests"
    ]


def test_stochastic_diagnostic():
    assert check_diagnostic_contract(StochasticDiagnostic()) == []


def test_reset_false_preserves_seed():
    class Bad(StochasticToy):
        def set_rng(self, seed: int, reset: bool = True) -> None:
            super().set_rng(seed, reset=True)

    assert "P12" in violations(Bad())


def test_step_requires_valid_forcing():
    class Bad(ForcedToy):
        def step(
            self, y: xr.DataArray, forcing: xr.DataArray, state: xr.DataArray
        ) -> tuple[xr.DataArray, xr.DataArray]:
            return self._forward(y), state.copy(deep=True)

    assert "P22" in violations(Bad())


def test_missing_protocol_members_collected():
    assert "P1" in violations(object())


def test_missing_forcing_declaration_collected():
    model = ToyPrognostic()
    model.forcing_coords = None
    assert "P2" in violations(model)


def test_duplicate_output_slots():
    class Bad(MultiPrognostic):
        def output_coords(
            self, fine: CoordinateSystem, coarse: CoordinateSystem
        ) -> tuple[CoordinateSystem, CoordinateSystem]:
            fine, _ = super().output_coords(fine, coarse)
            return fine, coord_array_like(fine, {"variable": ["other"]})

    assert "P18" in violations(Bad(), rollout=False)


def test_legacy_initial_yield_rejected():
    class Legacy(ToyPrognostic):
        def create_iterator(
            self, x: xr.DataArray
        ) -> Generator[
            xr.DataArray, xr.DataArray | tuple[xr.DataArray, ...] | None, None
        ]:
            yield x
            yield from self._default_create_iterator(x)

    assert {"P7", "P8"} <= violations(Legacy())


def test_initialization_must_advance_lead():
    class Bad(ToyPrognostic):
        def output_coords(self, x: CoordinateSystem) -> CoordinateSystem:
            super().output_coords(x)
            return coord_array_like(x)

    assert "P7" in violations(Bad())


@pytest.mark.parametrize(
    "method", ["__call__", "initialize", "step", "create_iterator"]
)
@pytest.mark.parametrize("variadic", ["positional", "keyword"])
def test_variadic_wrappers_rejected(method, variadic):
    model = ToyPrognostic()
    if variadic == "positional":
        setattr(model, method, lambda *args: None)
    elif method == "step":
        setattr(model, method, lambda y, state, **kwargs: None)
    else:
        setattr(model, method, lambda x, **kwargs: None)
    assert "P24" in violations(model, rollout=False)


def test_keyword_only_state_rejected():
    class Bad(ToyPrognostic):
        def step(self, y: xr.DataArray, *, state: None) -> tuple[xr.DataArray, None]:
            return super().step(y, state)

    assert "P24" in violations(Bad(), rollout=False)


def test_state_mutation_and_replay():
    class Bad(ToyPrognostic):
        def initialize(self, x: xr.DataArray) -> tuple[xr.DataArray, dict[str, int]]:
            return self._forward(x), {"count": 0}

        def step(
            self, y: xr.DataArray, state: dict[str, int]
        ) -> tuple[xr.DataArray, dict[str, int]]:
            state["count"] += 1
            return self._forward(y) + state["count"], state

    assert "P19" in violations(Bad())


def test_call_equivalence():
    class Bad(ToyPrognostic):
        def __call__(self, x: xr.DataArray) -> xr.DataArray:
            return self._forward(x) + 3

    assert "P20" in violations(Bad())


def test_missing_forcing_rejected():
    class Bad(ForcedToy):
        def create_iterator(
            self, x: xr.DataArray, static: xr.DataArray, forcing: xr.DataArray
        ) -> Generator[
            xr.DataArray, xr.DataArray | tuple[xr.DataArray, ...] | None, None
        ]:
            y, state = self.initialize(x, static, forcing)
            while True:
                yield y
                y = self._forward(y)

    assert "P22" in violations(Bad())


def test_source_slot_count():
    class Bad(ForcedToy):
        def default_sources(self) -> tuple[None]:
            return (None,)

    assert "P23" in violations(Bad(), rollout=False)


def test_synchronous_source_recommendation():
    class Source:
        def __call__(self, time, variable):
            raise AssertionError("Conformance must not fetch data")

    class Model(ToyPrognostic):
        def default_sources(self):
            return Source()

    check_prognostic_contract(Model(), rollout=False)


def test_invalid_source_recommendation():
    class Model(ToyPrognostic):
        def default_sources(self):
            return lambda: None

    assert "P23" in violations(Model(), rollout=False)


def test_input_mutation():
    class Bad(ToyPrognostic):
        def _forward(self, x: xr.DataArray) -> xr.DataArray:
            x.data += 1
            return super()._forward(x)

    assert "P15" in violations(Bad())


def test_aliased_yields():
    class Bad(ToyPrognostic):
        def create_iterator(
            self, x: xr.DataArray
        ) -> Generator[
            xr.DataArray, xr.DataArray | tuple[xr.DataArray, ...] | None, None
        ]:
            y, _ = self.initialize(x)
            while True:
                yield y
                y.data += 1
                y = y.assign_coords(lead_time=y.lead_time + DT)

    assert "P16" in violations(Bad())


@pytest.mark.parametrize("missing", ["front", "rear"])
def test_hook_order_and_scope(missing):
    class Bad(ToyPrognostic):
        def create_iterator(
            self, x: xr.DataArray
        ) -> Generator[
            xr.DataArray, xr.DataArray | tuple[xr.DataArray, ...] | None, None
        ]:
            y, state = self.initialize(x)
            while True:
                if missing != "rear":
                    y = self.rear_hook(y)
                yield y
                if missing != "front":
                    y = self.front_hook(y)
                y, state = self.step(y, state)

    assert "P10" in violations(Bad())


def test_hooks_on_initialize():
    class Bad(ToyPrognostic):
        def initialize(self, x: xr.DataArray) -> tuple[xr.DataArray, None]:
            return self.rear_hook(self._forward(x)), None

    assert "P10" in violations(Bad())


def test_broken_rebasing():
    class Bad(ToyPrognostic):
        def output_coords(self, x: CoordinateSystem) -> CoordinateSystem:
            return super().output_coords(x).assign_coords(lead_time=[DT])

    assert "P6" in violations(Bad(), rollout=False)


def test_planning_mutation():
    class Bad(ToyPrognostic):
        def output_coords(self, x: CoordinateSystem) -> CoordinateSystem:
            x.attrs["changed"] = True
            return super().output_coords(x)

    assert "P4" in violations(Bad(), rollout=False)


def test_invalid_metadata():
    class Bad(ToyPrognostic):
        def _forward(self, x: xr.DataArray) -> xr.DataArray:
            y = super()._forward(x)
            y.attrs["earth2studio_crs"] = "bad"
            return y

    assert "P9" in violations(Bad())


def test_global_rng_isolation():
    class Bad(StochasticToy):
        def set_rng(self, seed: int, reset: bool = True) -> None:
            torch.manual_seed(seed)
            super().set_rng(seed, reset)

    assert "P14" in violations(Bad(), rollout=False)


def test_diagnostic_multislot():
    skipped = check_diagnostic_contract(MultiDiagnostic())
    assert skipped == ["D10: model does not declare itself stochastic"]


def test_diagnostic_variadic():
    class Bad(Identity):
        def __call__(self, *x: xr.DataArray) -> xr.DataArray:
            return x[0]

    assert "D11" in violations(Bad(), diagnostic=True, forward=False)


def test_diagnostic_mutation():
    class Bad(Identity):
        def __call__(self, x: xr.DataArray) -> xr.DataArray:
            x.attrs["changed"] = True
            return x

    assert "D6" in violations(Bad(), diagnostic=True)


def test_legacy_declarations_reported():
    class Bad(Identity):
        def input_coords(self) -> dict[str, np.ndarray]:
            return {"batch": np.empty(0)}

    assert "D2" in violations(Bad(), diagnostic=True)


def test_rule_inventory_and_reachability():
    rules = {f"P{i}" for i in range(1, 25)} | {f"D{i}" for i in range(1, 12)}
    assert set(dict(iter_contract_rules())) == rules
    evaluated = _evaluate_prognostic(StochasticToy()).evaluated
    evaluated |= _evaluate_prognostic(ForcedToy()).evaluated
    evaluated |= _evaluate_diagnostic(Identity()).evaluated
    assert evaluated == rules


def test_rollout_disabled():
    report = _evaluate_prognostic(ToyPrognostic(), rollout=False)
    assert not report.violations
    assert "P19: rollout checks disabled" in report.skipped


@pytest.mark.parametrize("nsteps", [0, -1, True])
def test_invalid_nsteps(nsteps):
    with pytest.raises(ValueError):
        check_prognostic_contract(ToyPrognostic(), nsteps=nsteps)
