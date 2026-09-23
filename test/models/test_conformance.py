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
from collections.abc import Generator, Iterator

import numpy as np
import pytest
import torch
import xarray as xr

from earth2studio.models.batch import batch_func
from earth2studio.models.conformance import (
    ContractException,
    _evaluate_diagnostic,
    _evaluate_prognostic,
    _expected_shape,
    check_diagnostic_contract,
    check_prognostic_contract,
    iter_contract_rules,
)
from earth2studio.models.dx import Identity
from earth2studio.models.px import Persistence
from earth2studio.models.px.fcn import FCN
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
)
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem, CoordSystem

DOMAIN = OrderedDict(
    {
        "lat": np.linspace(90, -90, 8, endpoint=False),
        "lon": np.linspace(0, 360, 16, endpoint=False),
    }
)
DT = np.timedelta64(6, "h")


class ToyPrognostic(torch.nn.Module, PrognosticMixin):
    """A minimal, fully conformant prognostic model used as the contract reference."""

    def input_coords(self) -> CoordinateSystem:
        return coord_array(
            ("batch", "lead_time", "variable", *DOMAIN),
            {
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(["t2m"]),
                **DOMAIN,
            },
            dynamic=("batch",),
        )

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        lead = input_coords.lead_time.values
        if lead.dtype.kind != "m" or lead.size != 1 or np.isnat(lead).any():
            raise ValueError("lead_time must contain one finite timedelta")
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
        return coord_array_like(input_coords, {"lead_time": lead + DT})

    def _step(self, x: xr.DataArray) -> xr.DataArray:
        return from_torch(x.e2s.to_torch()[0] + 1, self.output_coords(x))

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        # Hooks belong to the iterator; the single-step path deliberately skips them
        return self._step(x)

    def _default_generator(
        self, x: xr.DataArray
    ) -> Generator[xr.DataArray, None, None]:
        x = x.copy(deep=True)
        yield x.copy(deep=True)
        while True:
            x = self.front_hook(x)
            x = self._step(x)
            x = self.rear_hook(x)
            yield x.copy(deep=True)

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        yield from self._default_generator(x)


def _violations(model, **kwargs) -> set[str]:
    """Return the rule identifiers a model violates."""
    with pytest.raises(ContractException) as error:
        check_prognostic_contract(model, **kwargs)
    return {violation.split(":")[0] for violation in error.value.violations}


def native_model():
    model = FCN(torch.nn.Identity(), torch.zeros(1, 1, 1), torch.ones(1, 1, 1))
    model.input_coords = lambda: coord_array(
        ("batch", "lead_time", "variable", "lat", "lon"),
        {
            "lead_time": np.array([0], dtype="timedelta64[h]"),
            "variable": ["t2m"],
            "lat": [1, 0],
            "lon": [0, 1, 2],
        },
        dynamic=("batch",),
    )
    return model


@pytest.mark.parametrize("factory", [ToyPrognostic, native_model])
def test_conformance_reference_model(factory):
    model = factory()
    original = model.front_hook, model.rear_hook
    assert check_prognostic_contract(model) == [
        "P14: model does not declare itself stochastic"
    ]
    assert (model.front_hook, model.rear_hook) == original


def test_conformance_diagnostic():
    assert check_diagnostic_contract(Identity()) == [
        "D4: model declares fewer than three dimensions",
        "D10: model does not declare itself stochastic",
    ]

    class InvalidMetadata(Identity):
        def __call__(self, x):
            output = super().__call__(x)
            output.attrs[key] = value
            return output

    for key, value in (
        ("earth2studio_crs", "invalid"),
        ("earth2studio_grid_id", "wrong-grid"),
        ("earth2studio_statistics", {"t2m": "sum:6h"}),
        ("earth2studio_kind", "coordinate_array"),
        ("earth2studio_schema_version", 1),
        ("earth2studio_dynamic_dims", ()),
    ):
        assert "D5" in _diagnostic_violations(InvalidMetadata())


class StochasticIdentity(Identity):
    """A conformant stochastic diagnostic: declared, seedable, and reproducible."""

    stochastic = True

    def __init__(self) -> None:
        super().__init__()
        self.generator: torch.Generator | None = None

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Seed the generator used to draw noise."""
        if reset or self.generator is None:
            self.generator = torch.Generator().manual_seed(seed)

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        tensor, _ = x.e2s.to_torch()
        noise = torch.randn(x.shape, generator=self.generator)
        return from_torch(tensor + noise, self.output_coords(x))


def _diagnostic_violations(model, **kwargs) -> set[str]:
    """Return the rule identifiers a diagnostic violates."""
    with pytest.raises(ContractException) as error:
        check_diagnostic_contract(model, **kwargs)
    return {violation.split(":")[0] for violation in error.value.violations}


def test_conformance_stochastic_diagnostic():
    assert check_diagnostic_contract(StochasticIdentity()) == [
        "D4: model declares fewer than three dimensions"
    ]


def test_conformance_detects_undeclared_diagnostic_stochasticity():
    class Undeclared(StochasticIdentity):
        stochastic = False

    assert _diagnostic_violations(Undeclared()) == {"D9"}


def test_conformance_detects_diagnostic_missing_set_rng():
    class NoSeeding(Identity):
        stochastic = True

    assert _diagnostic_violations(NoSeeding(), forward=False) == {"D8"}


def test_conformance_detects_diagnostic_unreached_randomness():
    class Unseeded(StochasticIdentity):
        def set_rng(self, seed: int, reset: bool = True) -> None:
            """Accept a seed and ignore it."""
            self.generator = torch.Generator().manual_seed(0)

    assert _diagnostic_violations(Unseeded()) == {"D9"}


@pytest.mark.parametrize("metadata", [False, True])
def test_conformance_detects_diagnostic_input_mutation(metadata):
    class MutatingIdentity(Identity):
        def __call__(self, x: xr.DataArray) -> xr.DataArray:
            if metadata:
                x.attrs["changed"] = True
            else:
                x.data += 1
            return x

    with pytest.raises(ContractException) as error:
        check_diagnostic_contract(MutatingIdentity())
    assert {v.split(":")[0] for v in error.value.violations} == {"D6"}


def test_conformance_reports_shape_changing_output_as_violation():
    """A diagnostic whose output shape moves between calls is reported, not a crash.

    The TC trackers accumulate a path buffer across calls, so a second call on
    one input returns a larger tensor — but each individual call's tensor and
    declared coordinates agree with each other, exactly like the real trackers
    (``output_coords()`` grows the ``value`` dimension to match). It is only
    the *comparison across two calls* that disagrees. ``torch.allclose`` raises
    on non-broadcastable shapes rather than returning False, which would
    surface as a checker crash instead of the ``D9`` violation it is.
    """

    class Accumulating(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def input_coords(self) -> CoordSystem:
            return OrderedDict({"batch": np.empty(0), "value": np.arange(1)})

        def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
            output_coords = input_coords.copy()
            output_coords["value"] = np.arange(max(self.calls, 1))
            return output_coords

        @batch_func()
        def __call__(
            self, x: torch.Tensor, coords: CoordSystem
        ) -> tuple[torch.Tensor, CoordSystem]:
            self.calls += 1
            out = x.repeat_interleave(self.calls, dim=-1)
            return out, self.output_coords(coords)

    assert "D9" in _diagnostic_violations(Accumulating())


def test_conformance_detects_input_mutation():
    """The StormCast v1 bug: writing results into the caller's initial condition."""

    class MutatesInput(ToyPrognostic):
        def _step(self, x: xr.DataArray) -> xr.DataArray:
            x.data += 1
            return from_torch(x.e2s.to_torch()[0], self.output_coords(x))

    assert "P15" in _violations(MutatesInput())


@pytest.mark.parametrize("native", [False, True])
def test_conformance_detects_aliased_yields(native):
    """Yields that share one buffer read stale under a deferred write."""

    class AliasedYields(ToyPrognostic):
        def _default_generator(
            self, x: xr.DataArray
        ) -> Generator[xr.DataArray, None, None]:
            buffer = x.copy(deep=True)
            yield buffer
            while True:
                buffer.data += 1
                buffer = buffer.assign_coords(lead_time=buffer.lead_time + DT)
                buffer = self.front_hook(buffer)
                buffer = self.rear_hook(buffer)
                yield buffer

    model = AliasedYields()
    if native:
        model = native_model()

        def iterator(x):
            state = x.copy(deep=True)
            yield state
            while True:
                state.data += 1
                state = state.assign_coords(lead_time=state.lead_time + DT)
                yield state

        model.create_iterator = iterator
    assert "P16" in _violations(model)


@pytest.mark.parametrize("history", [1, 2])
def test_conformance_persistence(history):
    model = Persistence("t2m", DOMAIN, history=history)
    assert check_prognostic_contract(model, rollout=False) == [
        "P14: model does not declare itself stochastic",
        "P7: rollout checks disabled",
        "P8: rollout checks disabled",
        "P9: rollout checks disabled",
        "P10: rollout checks disabled",
        "P13: rollout checks disabled",
        "P15: rollout checks disabled",
        "P16: rollout checks disabled",
    ]
    assert check_prognostic_contract(model) == [
        "P14: model does not declare itself stochastic"
    ]


def test_conformance_curvilinear_coords():
    """A 2D lat/lon pair spans two tensor dimensions, not two sizes of one."""
    lat, lon = np.meshgrid(np.linspace(0, 1, 4), np.linspace(0, 1, 6), indexing="ij")
    model = Persistence("t2m", OrderedDict({"lat": lat, "lon": lon}))
    assert model.input_coords().shape == (0, 1, 1, 4, 6)
    assert check_prognostic_contract(model) == [
        "P14: model does not declare itself stochastic"
    ]


def test_expected_shape_rejects_inconsistent_groups():
    """A multidimensional coordinate not followed by its partners implies no shape."""
    grid = np.zeros((4, 6))
    assert _expected_shape(OrderedDict({"lat": grid})) is None
    assert _expected_shape(OrderedDict({"lat": grid, "lon": np.zeros((4, 5))})) is None


def test_conformance_detects_mutation():
    class Mutating(ToyPrognostic):
        def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
            input_coords["lead_time"] = input_coords["lead_time"] + DT
            return ToyPrognostic.output_coords(self, input_coords)

    assert "P4" in _violations(Mutating(), rollout=False)


def test_conformance_detects_missing_validation():
    class Permissive(ToyPrognostic):
        def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
            return coord_array_like(
                input_coords, {"lead_time": input_coords.lead_time + DT}
            )

    assert _violations(Permissive(), rollout=False) == {"P5"}


def test_conformance_detects_broken_rebasing():
    class Unrebased(ToyPrognostic):
        def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
            return ToyPrognostic.output_coords(self, input_coords).assign_coords(
                lead_time=np.array([DT])
            )

    assert "P6" in _violations(Unrebased(), rollout=False)


@pytest.mark.parametrize("native", [False, True])
def test_conformance_detects_missing_zeroth_yield(native):
    class NoInitialCondition(ToyPrognostic):
        def _default_generator(
            self, x: xr.DataArray
        ) -> Generator[xr.DataArray, None, None]:
            while True:
                x = self._step(x)
                yield x

    model = NoInitialCondition()
    if native:
        model = native_model()

        def iterator(x):
            yield x + 1
            while True:
                x = model(x)
                yield x

        model.create_iterator = iterator
    assert "P7" in _violations(model)


def test_conformance_detects_shape_mismatch():
    class Ragged(ToyPrognostic):
        def _step(self, x: xr.DataArray) -> xr.DataArray:
            return super()._step(x).isel(lon=slice(None, -1))

    assert "P9" in _violations(Ragged())

    model = native_model()
    original_iterator = model.create_iterator

    def broken_later_coordinates(x):
        for index, field in enumerate(original_iterator(x)):
            if index >= 2:
                field = field.assign_coords(lat=field.lat + 100)
            yield field

    model.create_iterator = broken_later_coordinates
    assert _violations(model) == {"P9"}

    def broken_later_metadata(x):
        for index, field in enumerate(original_iterator(x)):
            if index >= 2:
                field = field.copy(deep=True)
                field.attrs[key] = value
            yield field

    key, value = "user_note", "hook metadata remains flexible"
    model.create_iterator = broken_later_metadata
    assert check_prognostic_contract(model) == [
        "P14: model does not declare itself stochastic"
    ]

    for key, value in (
        ("earth2studio_crs", "invalid"),
        ("earth2studio_kind", "coordinate_array"),
    ):
        model.create_iterator = broken_later_metadata
        assert _violations(model) == {"P9"}

    original_forward = model._step

    def broken_call(x):
        output = original_forward(x)
        output.attrs[key] = value
        return output

    model.create_iterator = original_iterator
    model._step = broken_call
    for key, value in (
        ("earth2studio_crs", "invalid"),
        ("earth2studio_kind", "coordinate_array"),
    ):
        assert "P9" in _violations(model)


@pytest.mark.parametrize(
    "native,interval,missing",
    [(False, 1, "front"), (True, 1, "front"), (True, 2, "rear"), (True, 2, None)],
)
def test_conformance_detects_dropped_hook(native, interval, missing):
    """The real bug class: a generator that applies only one of the two hooks.

    Three in-repo wrappers apply ``rear_hook`` and never ``front_hook``, so a front
    hook a caller sets is silently discarded.
    """

    class RearHookOnly(ToyPrognostic):
        def _default_generator(
            self, x: xr.DataArray
        ) -> Generator[xr.DataArray, None, None]:
            yield x
            while True:
                x = self._step(x)
                x = self.rear_hook(x)
                yield x

    if not native:
        assert _violations(RearHookOnly()) == {"P10"}
        return
    model = native_model()
    model.front_hook_interval = interval
    original = model.front_hook, model.rear_hook

    def iterator(x):
        yield x
        cycle = 0
        while True:
            if not (cycle == 1 and missing == "front"):
                x = model.front_hook(x)
            for output in range(interval):
                x = model(x)
                if not (cycle == 1 and output == interval - 1 and missing == "rear"):
                    x = model.rear_hook(x)
                yield x
            cycle += 1

    model.create_iterator = iterator
    if missing is None:
        assert check_prognostic_contract(model, nsteps=1) == [
            "P14: model does not declare itself stochastic"
        ]
    else:
        assert _violations(model, nsteps=1) == {"P10"}
    assert (model.front_hook, model.rear_hook) == original


@pytest.mark.parametrize("interval", [0, True, 1.5])
def test_conformance_invalid_hook_interval(interval):
    model = native_model()
    model.front_hook_interval = interval
    original = model.front_hook, model.rear_hook
    assert _violations(model) == {"P10"}
    assert (model.front_hook, model.rear_hook) == original


@pytest.mark.parametrize(
    "temporal_dims,unit",
    [
        (("time", "lead_time"), None),
        (("time", "lead_time"), "ms"),
        (("analysis_time", "forecast_offset"), "us"),
    ],
)
def test_conformance_temporal_declarations(temporal_dims, unit):
    time_dim, lead_dim = temporal_dims
    timestamp = np.datetime64("2025-03-04T06:07:08.123456", "us")
    temporal = (
        {
            time_dim: np.array([], dtype=f"datetime64[{unit}]"),
            lead_dim: np.array([], dtype=f"timedelta64[{unit}]"),
        }
        if unit
        else {}
    )
    signature = coord_array(
        ("member", time_dim, lead_dim, "variable", "lat", "lon"),
        {**temporal, "variable": ["t2m"], "lat": [1, 0], "lon": [0, 1, 2]},
        dynamic=("member", time_dim, lead_dim),
    )
    original = signature.copy(deep=True)
    probes = []

    class TemporalDiagnostic(Identity):
        def input_coords(self):
            return signature

        def output_coords(self, x):
            handshake_dataarray(x, signature)
            for dim, kind in ((time_dim, "M"), (lead_dim, "m")):
                assert x.coords[dim].dtype.kind == kind
                if unit:
                    assert x.coords[dim].dtype == signature.coords[dim].dtype
            probes.append(x)
            return coord_array_like(x)

    assert check_diagnostic_contract(TemporalDiagnostic(), time=timestamp) == [
        "D10: model does not declare itself stochastic"
    ]
    assert len(probes) == 1
    probe = probes[0]
    assert probe.data.nbytes == 0 and probe.attrs["earth2studio_dynamic_dims"] == ()
    np.testing.assert_array_equal(probe.member, [0])
    np.testing.assert_array_equal(
        probe.coords[time_dim],
        np.array([timestamp], dtype=f"datetime64[{unit}]" if unit else None),
    )
    np.testing.assert_array_equal(probe.coords[lead_dim], [np.timedelta64(0, "ns")])
    assert signature.sizes == original.sizes and signature.attrs == original.attrs
    xr.testing.assert_identical(
        signature.coords.to_dataset(), original.coords.to_dataset()
    )


def test_conformance_detects_hooks_on_call():
    class CallAppliesHooks(ToyPrognostic):
        def __call__(self, x: xr.DataArray) -> xr.DataArray:
            x = self.front_hook(x)
            return self.rear_hook(self._step(x))

    assert _violations(CallAppliesHooks()) == {"P10"}


def _step_once(model: ToyPrognostic) -> None:
    """Draw one forecast step, the only path that applies hooks."""
    coords = coord_array_like(model.input_coords(), {"batch": np.arange(1)})
    iterator = model.create_iterator(from_torch(torch.zeros(coords.shape), coords))
    next(iterator)
    next(iterator)


def test_hook_assignment_and_composition():
    """front_hook/rear_hook are single callable slots: a caller composes multiple
    transformations into one function and assigns that, so ordering is visible at
    the assignment site rather than spread across separate registration calls."""
    model = ToyPrognostic()
    order: list[str] = []

    def first(x):
        order.append("first")
        return x

    def second(x):
        order.append("second")
        return x

    def combined(x):
        return second(first(x))

    model.front_hook = combined
    _step_once(model)
    assert order == ["first", "second"]


def test_clear_hooks_restores_default():
    model = ToyPrognostic()
    order: list[str] = []

    x = xr.DataArray([1.0], dims="member", coords={"member": [7]})
    assert model.front_hook(x) is x
    assert model.rear_hook(x) is x
    assert model.front_hook_interval == 1
    assert model.stochastic is False
    model.front_hook = model.rear_hook = lambda x: order.append("assigned") or x
    model.front_hook(x)
    model.rear_hook(x)
    assert order == ["assigned", "assigned"]
    order.clear()
    model.clear_hooks()
    model.clear_hooks()
    assert "front_hook" not in vars(model) and "rear_hook" not in vars(model)
    assert model.front_hook(x) is x
    assert model.rear_hook(x) is x
    _step_once(model)
    assert order == []


class StochasticToy(ToyPrognostic):
    """A conformant stochastic model: declared, seedable, and reproducible."""

    stochastic = True

    def __init__(self) -> None:
        super().__init__()
        self.generator: torch.Generator | None = None

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Seed the generator used to draw per-step noise."""
        if reset or self.generator is None:
            self.generator = torch.Generator().manual_seed(seed)

    def _step(self, x: xr.DataArray) -> xr.DataArray:
        noise = torch.randn(x.shape, generator=self.generator)
        return from_torch(x.e2s.to_torch()[0] + noise, self.output_coords(x))


def test_conformance_stochastic_model():
    assert check_prognostic_contract(StochasticToy()) == []


def test_conformance_detects_undeclared_stochasticity():
    class Undeclared(StochasticToy):
        stochastic = False

    assert _violations(Undeclared()) == {"P13"}


def test_conformance_detects_undeclarable_type():
    class NotABool(ToyPrognostic):
        stochastic = "yes"

    assert "P11" in _violations(NotABool(), rollout=False)


def test_conformance_detects_missing_set_rng():
    class NoSeeding(ToyPrognostic):
        stochastic = True

    assert _violations(NoSeeding(), rollout=False) == {"P12"}


def test_conformance_detects_set_rng_signature():
    class WrongSignature(StochasticToy):
        def set_rng(self, reset: bool = True, seed: int = 0) -> None:
            """Seed the generator behind a leading reset argument."""
            super().set_rng(seed, reset)

    assert "P12" in _violations(WrongSignature(), rollout=False)


def test_conformance_detects_unreached_randomness():
    """set_rng that misses a source of randomness duplicates ensemble members."""

    class Unseeded(StochasticToy):
        def set_rng(self, seed: int, reset: bool = True) -> None:
            """Accept a seed and ignore it."""
            self.generator = torch.Generator().manual_seed(0)

    assert _violations(Unseeded()) == {"P13"}


def test_conformance_detects_global_seeding_in_set_rng():
    """The common violation: set_rng implemented as a bare torch.manual_seed."""

    class GlobalSeeder(StochasticToy):
        def set_rng(self, seed: int, reset: bool = True) -> None:
            """Seed the global generator, reaching every other RNG consumer."""
            torch.manual_seed(seed)

        def _step(self, x: xr.DataArray) -> xr.DataArray:
            return from_torch(
                x.e2s.to_torch()[0] + torch.randn(x.shape), self.output_coords(x)
            )

    # Caught without a rollout: the seeding half of P14 needs no forward pass
    assert "P14" in _violations(GlobalSeeder(), rollout=False)


def test_conformance_detects_global_seeding_mid_step():
    """A clean set_rng does not excuse reseeding the global RNG while stepping."""

    class ReseedsPerStep(StochasticToy):
        def _step(self, x: xr.DataArray) -> xr.DataArray:
            torch.manual_seed(0)
            return super()._step(x)

    violations = _violations(ReseedsPerStep())
    assert "P14" in violations


def test_conformance_accepts_forked_global_seeding():
    """The escape hatch for models whose randomness lives in an external package.

    ``aifs2ens`` seeds the global RNG per step because ``anemoi`` exposes no
    generator. Forking keeps the call and removes its blast radius, so the model
    conforms without an upstream change.
    """

    class ForkedSeeder(StochasticToy):
        def set_rng(self, seed: int, reset: bool = True) -> None:
            """Record the seed; the draw itself is seeded inside a fork."""
            if reset or getattr(self, "_seed", None) is None:
                self._seed = seed

        def _step(self, x: xr.DataArray) -> xr.DataArray:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self._seed)
                noise = torch.randn(x.shape)
            return from_torch(x.e2s.to_torch()[0] + noise, self.output_coords(x))

    assert check_prognostic_contract(ForkedSeeder()) == []


def test_conformance_detects_global_seeding_diagnostic():
    """D10 is the diagnostic half of the same rule."""

    class GlobalSeeder(StochasticIdentity):
        def set_rng(self, seed: int, reset: bool = True) -> None:
            """Seed the global generator rather than a local one."""
            torch.manual_seed(seed)

    assert "D10" in _diagnostic_violations(GlobalSeeder(), forward=False)


def test_conformance_unseeded_global_draws_are_not_violations():
    """Advancing the global RNG is not the harm; reseeding it is.

    A model that never has ``set_rng`` called still draws from the global generator
    by the spec's own fallback, so consuming from it must stay legal.
    """

    class GlobalDraws(ToyPrognostic):
        def _step(self, x: xr.DataArray) -> xr.DataArray:
            return from_torch(
                x.e2s.to_torch()[0] + torch.randn(x.shape) * 0, self.output_coords(x)
            )

    assert check_prognostic_contract(GlobalDraws()) == [
        "P14: model does not declare itself stochastic"
    ]


def test_contract_rules_documented():
    rules = dict(iter_contract_rules())
    assert set(rules) == {f"P{i}" for i in range(1, 17)} | {
        f"D{i}" for i in range(1, 11)
    }
    assert all(summary.endswith(".") for summary in rules.values())


def test_all_rules_are_reachable():
    """Every rule documented in _RULES must be recorded (as a pass, a failure, or
    a skip — any outcome counts, this only proves the code path exists) by at
    least one of these checks, so the prose spec and the checker cannot silently
    drift apart: a rule added to one without the other fails this test."""
    evaluated = set(_evaluate_prognostic(StochasticToy()).evaluated)
    for diagnostic in (Identity(), StochasticIdentity()):
        evaluated |= _evaluate_diagnostic(diagnostic).evaluated

    all_rules = {f"P{i}" for i in range(1, 17)} | {f"D{i}" for i in range(1, 11)}
    assert evaluated == all_rules
