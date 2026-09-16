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

from earth2studio.models.batch import batch_coords, batch_func
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
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import handshake_coords, handshake_dim
from earth2studio.utils.type import CoordSystem

DOMAIN = OrderedDict(
    {
        "lat": np.linspace(90, -90, 8, endpoint=False),
        "lon": np.linspace(0, 360, 16, endpoint=False),
    }
)
DT = np.timedelta64(6, "h")


class ToyPrognostic(torch.nn.Module, PrognosticMixin):
    """A minimal, fully conformant prognostic model used as the contract reference."""

    def input_coords(self) -> CoordSystem:
        coords = OrderedDict(
            {
                "batch": np.empty(0),
                "lead_time": np.array([np.timedelta64(0, "h")]),
                "variable": np.array(["t2m"]),
            }
        )
        coords.update(DOMAIN)
        return coords

    @batch_coords()
    def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
        target = self.input_coords()
        for index, key in enumerate(target):
            if key == "batch":
                continue
            handshake_dim(input_coords, key, index)
            if key != "lead_time":
                handshake_coords(input_coords, target, key)

        output_coords = target.copy()
        output_coords["batch"] = input_coords["batch"]
        output_coords["lead_time"] = input_coords["lead_time"] + DT
        return output_coords

    def _step(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        return x + 1, self.output_coords(coords)

    @batch_func()
    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        # Hooks belong to the iterator; the single-step path deliberately skips them
        return self._step(x, coords)

    @batch_func()
    def _default_generator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
        coords = coords.copy()
        yield x, coords.copy()
        while True:
            x, coords = self.front_hook(x, coords)
            x, coords = self._step(x, coords)
            x, coords = self.rear_hook(x, coords)
            yield x, coords.copy()

    def create_iterator(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        yield from self._default_generator(x, coords)


def _violations(model, **kwargs) -> set[str]:
    """Return the rule identifiers a model violates."""
    with pytest.raises(ContractException) as error:
        check_prognostic_contract(model, **kwargs)
    return {violation.split(":")[0] for violation in error.value.violations}


def test_conformance_reference_model():
    assert check_prognostic_contract(ToyPrognostic()) == [
        "P14: model does not declare itself stochastic"
    ]


def test_conformance_diagnostic():
    assert check_diagnostic_contract(Identity()) == [
        "D4: model declares fewer than three dimensions",
        "D10: model does not declare itself stochastic",
    ]


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

    @batch_func()
    def __call__(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        noise = torch.randn(x.shape, generator=self.generator)
        return x + noise, self.output_coords(coords)


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


def test_conformance_detects_diagnostic_input_mutation():
    class MutatingIdentity(Identity):
        @batch_func()
        def __call__(
            self, x: torch.Tensor, coords: CoordSystem
        ) -> tuple[torch.Tensor, CoordSystem]:
            x.add_(1)
            return x, self.output_coords(coords)

    with pytest.raises(ContractException) as error:
        check_diagnostic_contract(MutatingIdentity())
    assert {v.split(":")[0] for v in error.value.violations} == {"D6"}


def test_conformance_reports_shape_changing_output_as_violation():
    """A diagnostic whose output shape moves between calls is reported, not a crash.

    The TC trackers accumulate a path buffer across calls, so a second call on
    one input returns a larger tensor. ``torch.allclose`` raises on
    non-broadcastable shapes rather than returning False, which would surface as
    a checker crash instead of the ``D9`` violation it is. ``D5`` comes along
    because the grown tensor no longer matches its own declared coordinates.
    """

    class Accumulating(Identity):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        @batch_func()
        def __call__(
            self, x: torch.Tensor, coords: CoordSystem
        ) -> tuple[torch.Tensor, CoordSystem]:
            self.calls += 1
            return x.repeat_interleave(self.calls, dim=-1), self.output_coords(coords)

    assert "D9" in _diagnostic_violations(Accumulating())


def test_conformance_detects_input_mutation():
    """The StormCast v1 bug: writing results into the caller's initial condition."""

    class MutatesInput(ToyPrognostic):
        def _step(
            self, x: torch.Tensor, coords: CoordSystem
        ) -> tuple[torch.Tensor, CoordSystem]:
            x.add_(1)
            return x, self.output_coords(coords)

    assert "P15" in _violations(MutatesInput())


def test_conformance_detects_aliased_yields():
    """Yields that share one buffer read stale under a deferred write."""

    class AliasedYields(ToyPrognostic):
        @batch_func()
        def _default_generator(
            self, x: torch.Tensor, coords: CoordSystem
        ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
            coords = coords.copy()
            buffer = x.clone()
            yield buffer, coords.copy()
            while True:
                buffer.add_(1)
                coords = self.output_coords(coords)
                buffer, coords = self.front_hook(buffer, coords)
                buffer, coords = self.rear_hook(buffer, coords)
                yield buffer, coords.copy()

    assert "P16" in _violations(AliasedYields())


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
    assert _expected_shape(model.input_coords()) == (0, 1, 1, 4, 6)
    assert check_prognostic_contract(model) == [
        "P14: model does not declare itself stochastic"
    ]


def test_expected_shape_rejects_inconsistent_groups():
    """A multidimensional coordinate not followed by its partners implies no shape."""
    grid = np.zeros((4, 6))
    assert _expected_shape(OrderedDict({"lat": grid})) is None
    assert _expected_shape(OrderedDict({"lat": grid, "lon": np.zeros((4, 5))})) is None


def test_conformance_detects_mutation():
    """Detection requires an undecorated output_coords.

    ``batch_coords`` copies before delegating, so a decorated implementation cannot
    reach its caller's dictionary and satisfies this rule for free.
    """

    class Mutating(ToyPrognostic):
        def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
            input_coords["lead_time"] = input_coords["lead_time"] + DT
            return ToyPrognostic.output_coords(self, input_coords)

    assert "P4" in _violations(Mutating(), rollout=False)


def test_conformance_detects_missing_validation():
    class Permissive(ToyPrognostic):
        @batch_coords()
        def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
            output_coords = self.input_coords()
            output_coords["batch"] = input_coords["batch"]
            output_coords["lead_time"] = input_coords["lead_time"] + DT
            return output_coords

    assert _violations(Permissive(), rollout=False) == {"P5"}


def test_conformance_detects_broken_rebasing():
    class Unrebased(ToyPrognostic):
        @batch_coords()
        def output_coords(self, input_coords: CoordSystem) -> CoordSystem:
            output_coords = ToyPrognostic.output_coords.__wrapped__(self, input_coords)
            output_coords["lead_time"] = np.array([DT])
            return output_coords

    assert "P6" in _violations(Unrebased(), rollout=False)


def test_conformance_detects_missing_zeroth_yield():
    class NoInitialCondition(ToyPrognostic):
        @batch_func()
        def _default_generator(
            self, x: torch.Tensor, coords: CoordSystem
        ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
            coords = coords.copy()
            while True:
                x, coords = self._step(x, coords)
                yield x, coords.copy()

    assert "P7" in _violations(NoInitialCondition())


def test_conformance_detects_shape_mismatch():
    class Ragged(ToyPrognostic):
        def _step(
            self, x: torch.Tensor, coords: CoordSystem
        ) -> tuple[torch.Tensor, CoordSystem]:
            return x[..., :-1], self.output_coords(coords)

    assert "P9" in _violations(Ragged())


def test_conformance_detects_dropped_hook():
    """The real bug class: a generator that applies only one of the two hooks.

    Three in-repo wrappers apply ``rear_hook`` and never ``front_hook``, so a front
    hook a caller sets is silently discarded.
    """

    class RearHookOnly(ToyPrognostic):
        @batch_func()
        def _default_generator(
            self, x: torch.Tensor, coords: CoordSystem
        ) -> Generator[tuple[torch.Tensor, CoordSystem], None, None]:
            coords = coords.copy()
            yield x, coords.copy()
            while True:
                x, coords = self._step(x, coords)
                x, coords = self.rear_hook(x, coords)
                yield x, coords.copy()

    assert _violations(RearHookOnly()) == {"P10"}


def test_conformance_detects_hooks_on_call():
    class CallAppliesHooks(ToyPrognostic):
        @batch_func()
        def __call__(
            self, x: torch.Tensor, coords: CoordSystem
        ) -> tuple[torch.Tensor, CoordSystem]:
            x, coords = self.front_hook(x, coords)
            return self.rear_hook(*self._step(x, coords))

    assert _violations(CallAppliesHooks()) == {"P10"}


def _step_once(model: ToyPrognostic) -> None:
    """Draw one forecast step, the only path that applies hooks."""
    coords = model.input_coords()
    coords["batch"] = np.arange(1)
    iterator = model.create_iterator(torch.zeros(1, 1, 1, 8, 16), coords)
    next(iterator)
    next(iterator)


def test_hook_assignment_and_composition():
    """front_hook/rear_hook are single callable slots: a caller composes multiple
    transformations into one function and assigns that, so ordering is visible at
    the assignment site rather than spread across separate registration calls."""
    model = ToyPrognostic()
    order: list[str] = []

    def first(x, coords):
        order.append("first")
        return x, coords

    def second(x, coords):
        order.append("second")
        return x, coords

    def combined(x, coords):
        x, coords = first(x, coords)
        return second(x, coords)

    model.front_hook = combined
    _step_once(model)
    assert order == ["first", "second"]


def test_clear_hooks_restores_default():
    model = ToyPrognostic()
    order: list[str] = []

    model.rear_hook = lambda x, coords: (order.append("assigned"), (x, coords))[1]
    model.clear_hooks()
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

    def _step(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        noise = torch.randn(x.shape, generator=self.generator)
        return x + noise, self.output_coords(coords)


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

        def _step(
            self, x: torch.Tensor, coords: CoordSystem
        ) -> tuple[torch.Tensor, CoordSystem]:
            return x + torch.randn(x.shape), self.output_coords(coords)

    # Caught without a rollout: the seeding half of P14 needs no forward pass
    assert "P14" in _violations(GlobalSeeder(), rollout=False)


def test_conformance_detects_global_seeding_mid_step():
    """A clean set_rng does not excuse reseeding the global RNG while stepping."""

    class ReseedsPerStep(StochasticToy):
        def _step(
            self, x: torch.Tensor, coords: CoordSystem
        ) -> tuple[torch.Tensor, CoordSystem]:
            torch.manual_seed(0)
            return super()._step(x, coords)

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

        def _step(
            self, x: torch.Tensor, coords: CoordSystem
        ) -> tuple[torch.Tensor, CoordSystem]:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self._seed)
                noise = torch.randn(x.shape)
            return x + noise, self.output_coords(coords)

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
        def _step(
            self, x: torch.Tensor, coords: CoordSystem
        ) -> tuple[torch.Tensor, CoordSystem]:
            return x + torch.randn(x.shape) * 0, self.output_coords(coords)

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
