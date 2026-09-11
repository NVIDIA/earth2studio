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

"""Conformance checks for the Earth2Studio model contract.

The rule identifiers reported here (``P1``-``P15``, ``D1``-``D9``) match the rule
table in ``dev/spec/MODEL_CONTRACT_SPEC.md``.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
from inspect import Parameter, signature
from itertools import islice
from typing import Any, cast

import numpy as np
import torch

from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.models.px.base import PrognosticModel
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils.type import CoordSystem


class ContractViolation(AssertionError):
    """Raised when a model violates the Earth2Studio model contract.

    Parameters
    ----------
    model : Any
        The model that was checked
    violations : list[str]
        One message per failed rule, each prefixed with its rule identifier
    """

    def __init__(self, model: Any, violations: list[str]) -> None:
        self.violations = violations
        body = "\n".join(f"  - {violation}" for violation in violations)
        super().__init__(f"{type(model).__name__} violates the model contract:\n{body}")


class _Report:
    """Collects rule outcomes so every rule is evaluated before failing."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.violations: list[str] = []
        self.skipped: list[str] = []

    def require(self, rule: str, condition: bool, message: str) -> bool:
        """Record a rule outcome and report whether it held."""
        if not condition:
            self.violations.append(f"{rule}: {message}")
        return condition

    def skip(self, rule: str, message: str) -> None:
        """Record that a rule could not be evaluated."""
        self.skipped.append(f"{rule}: {message}")

    def raise_for_violations(self) -> None:
        """Raise if any rule failed."""
        if self.violations:
            raise ContractViolation(self.model, self.violations)


def _expected_shape(coords: CoordSystem) -> tuple[int, ...] | None:
    """Tensor shape implied by a coordinate system, or None if it is not derivable.

    A multidimensional coordinate spans several tensor dimensions at once, and by the
    convention :func:`earth2studio.utils.coords.convert_multidim_to_singledim` uses,
    the following ``n - 1`` entries describe the same grid and carry the same shape.
    A group that does not satisfy that convention has no derivable shape.
    """
    values = list(coords.values())
    shape: list[int] = []
    index = 0
    while index < len(values):
        value = values[index]
        if not isinstance(value, np.ndarray) or value.ndim == 0:
            return None
        if value.ndim == 1:
            shape.append(value.shape[0])
            index += 1
            continue
        group = values[index : index + value.ndim]
        if len(group) != value.ndim or any(
            not isinstance(member, np.ndarray) or member.shape != value.shape
            for member in group
        ):
            return None
        shape.extend(value.shape)
        index += value.ndim
    return tuple(shape)


def _concretize(coords: CoordSystem, batch_size: int = 1) -> CoordSystem:
    """Replace the open-ended coordinates of a declaration with concrete values.

    Model declarations use zero-length arrays to mean "any size" on batch-like
    dimensions. Contract checks need a runnable coordinate system, so each such
    dimension is pinned to ``batch_size``.

    Parameters
    ----------
    coords : CoordSystem
        Declared coordinate system, typically from ``input_coords()``
    batch_size : int, optional
        Size to give open-ended dimensions, by default 1

    Returns
    -------
    CoordSystem
        Coordinate system with every dimension concretely sized
    """
    concrete = OrderedDict()
    for key, value in coords.items():
        if isinstance(value, np.ndarray) and value.size == 0:
            if key == "time":
                concrete[key] = np.array(
                    [np.datetime64("2024-01-01T00:00:00")] * batch_size
                )
            else:
                concrete[key] = np.arange(batch_size)
        else:
            concrete[key] = value
    return concrete


def _sample_tensor(coords: CoordSystem, device: Any) -> torch.Tensor:
    """Build a probe tensor matching a coordinate system.

    Values are pseudo-random rather than zero so that a model writing into its input
    is detectable, and seeded so that repeated calls return an identical tensor.
    """
    shape = _expected_shape(coords)
    if shape is None:
        raise ValueError("Coordinate system does not imply a tensor shape")
    generator = torch.Generator().manual_seed(0)
    return torch.randn(shape, generator=generator).to(device)


def _swap_last_dims(coords: CoordSystem) -> CoordSystem:
    """Return a copy of a coordinate system with its final two dimensions swapped."""
    keys = list(coords)
    keys[-1], keys[-2] = keys[-2], keys[-1]
    return OrderedDict((key, coords[key]) for key in keys)


def _check_coord_declaration(
    report: _Report,
    model: Any,
    input_coords: CoordSystem,
    *,
    coords_rule: str,
    readonly_rule: str,
    invalid_rule: str,
) -> CoordSystem | None:
    """Evaluate the declaration rules shared by prognostic and diagnostic models.

    The rule identifiers differ between the two protocols, so each caller supplies
    the identifier its spec section uses.
    """
    report.require(
        coords_rule,
        isinstance(input_coords, OrderedDict),
        "input_coords() must return an OrderedDict; dimension order is part of the "
        f"contract, got {type(input_coords).__name__}",
    )
    report.require(
        coords_rule,
        all(isinstance(value, np.ndarray) for value in input_coords.values()),
        "every input_coords() value must be a numpy array",
    )
    report.require(
        coords_rule,
        next(iter(input_coords), None) == "batch",
        "the leading dimension of input_coords() must be 'batch'",
    )

    concrete = _concretize(input_coords)
    reference = OrderedDict((key, value.copy()) for key, value in concrete.items())
    try:
        output_coords = model.output_coords(concrete)
    except Exception as error:  # noqa: BLE001 - reported as a violation
        report.require(
            readonly_rule,
            False,
            f"output_coords() raised on its own input_coords(): {error!r}",
        )
        return None

    report.require(
        readonly_rule,
        list(concrete) == list(reference)
        and all(
            np.array_equal(concrete[key], value) for key, value in reference.items()
        ),
        "output_coords() mutated the coordinate system it was given; it must treat "
        "its argument as read-only",
    )

    if len(input_coords) >= 3:
        try:
            model.output_coords(_swap_last_dims(concrete))
        except (ValueError, KeyError):
            pass
        except Exception as error:  # noqa: BLE001 - reported as a violation
            report.require(
                invalid_rule,
                False,
                f"output_coords() raised {type(error).__name__} for a misordered "
                "coordinate system; it must raise ValueError",
            )
        else:
            report.require(
                invalid_rule,
                False,
                "output_coords() accepted a coordinate system whose final two "
                "dimensions were swapped; invalid input must raise ValueError",
            )
    else:
        report.skip(invalid_rule, "model declares fewer than three dimensions")

    return output_coords


def check_prognostic_contract(
    model: PrognosticModel,
    *,
    rollout: bool = True,
    nsteps: int = 2,
    device: Any = "cpu",
) -> list[str]:
    """Check a prognostic model against the Earth2Studio model contract.

    Every rule is evaluated before the check fails, so a single call reports all
    violations rather than only the first.

    Parameters
    ----------
    model : PrognosticModel
        Model to check
    rollout : bool, optional
        Whether to run the rules that require a forward pass (``P7``-``P10``), by
        default True. Set to False for models too expensive to step in CI.
    nsteps : int, optional
        Forecast steps to draw from the iterator when ``rollout`` is True, by
        default 2
    device : Any, optional
        Device to run the rollout on, by default ``"cpu"``

    Returns
    -------
    list[str]
        Rules that could not be evaluated, each with the reason it was skipped

    Raises
    ------
    ContractViolation
        If the model violates any evaluated rule
    """
    report = _Report(model)
    report.require(
        "P1",
        isinstance(model, PrognosticModel),
        "model does not structurally satisfy the PrognosticModel protocol",
    )

    input_coords = model.input_coords()
    lead_time = input_coords.get("lead_time")
    if lead_time is None:
        report.require(
            "P3", False, "input_coords() must declare a 'lead_time' dimension"
        )
    else:
        report.require(
            "P3",
            np.issubdtype(lead_time.dtype, np.timedelta64),
            f"input_coords()['lead_time'] must hold timedeltas, got {lead_time.dtype}",
        )
        report.require(
            "P3",
            lead_time.size > 0 and lead_time[-1] == np.timedelta64(0, "h"),
            "input_coords()['lead_time'] must be relative and end at zero, so that "
            f"the final entry is the analysis time, got {lead_time}",
        )
        report.require(
            "P3",
            lead_time.size < 2 or bool(np.all(np.diff(lead_time) > np.timedelta64(0))),
            f"input_coords()['lead_time'] must be strictly increasing, got {lead_time}",
        )

    output_coords = _check_coord_declaration(
        report,
        model,
        input_coords,
        coords_rule="P2",
        readonly_rule="P4",
        invalid_rule="P5",
    )
    if output_coords is not None:
        report.require(
            "P2",
            next(iter(output_coords), None) == "batch",
            "the leading dimension of output_coords() must be 'batch'",
        )
        _check_rebasing(report, model, input_coords, output_coords)

    stochastic = _check_stochasticity(report, model)

    if not rollout:
        report.skip("P7-P10, P13-P15", "rollout checks disabled")
    elif _expected_shape(_concretize(input_coords)) is None:
        report.skip(
            "P7-P10, P13-P15",
            "input_coords() does not imply a tensor shape, so no probe input can be "
            "built",
        )
    elif output_coords is not None:
        _check_rollout(
            report, model, input_coords, output_coords, nsteps, device, stochastic
        )

    report.raise_for_violations()
    return report.skipped


def _check_rebasing(
    report: _Report,
    model: PrognosticModel,
    input_coords: CoordSystem,
    output_coords: CoordSystem,
) -> None:
    """Evaluate the lead-time rebasing rule (``P6``)."""
    if "lead_time" not in input_coords or "lead_time" not in output_coords:
        report.skip("P6", "model does not declare a lead_time dimension")
        return

    offset = np.timedelta64(24, "h")
    shifted = _concretize(input_coords)
    shifted["lead_time"] = shifted["lead_time"] + offset
    try:
        rebased = model.output_coords(shifted)
    except Exception as error:  # noqa: BLE001 - reported as a violation
        report.require(
            "P6",
            False,
            f"output_coords() rejected an input rebased by {offset}: {error!r}",
        )
        return

    report.require(
        "P6",
        np.array_equal(rebased["lead_time"], output_coords["lead_time"] + offset),
        "shifting input lead_time by a constant must shift output lead_time by the "
        f"same constant; expected {output_coords['lead_time'] + offset}, got "
        f"{rebased['lead_time']}",
    )


def _check_immutability(
    report: _Report,
    x: torch.Tensor,
    pristine_x: torch.Tensor,
    coords: CoordSystem,
    pristine_coords: CoordSystem,
    path: str,
    rule: str = "P14",
) -> None:
    """Evaluate the input immutability rule (``P14``, ``D6``) for one call path."""
    report.require(
        rule,
        x.shape == pristine_x.shape and torch.equal(x, pristine_x),
        f"{path} modified the input tensor in place; a caller's initial condition "
        "must survive the call, and mutating it only moves the defensive copy onto "
        "every caller",
    )
    report.require(
        rule,
        list(coords) == list(pristine_coords)
        and all(
            np.array_equal(coords[key], value) for key, value in pristine_coords.items()
        ),
        f"{path} modified the input coordinate system in place",
    )


def _check_call_immutability(
    report: _Report, model: Any, coords: CoordSystem, device: Any, rule: str = "P14"
) -> None:
    """Evaluate input immutability for the single-step ``__call__`` path."""
    x = _sample_tensor(coords, device)
    pristine_x = x.clone()
    call_coords = OrderedDict((key, value.copy()) for key, value in coords.items())
    pristine_coords = OrderedDict((key, value.copy()) for key, value in coords.items())
    model(x, call_coords)
    _check_immutability(
        report, x, pristine_x, call_coords, pristine_coords, "__call__", rule
    )


def _check_rollout(
    report: _Report,
    model: PrognosticModel,
    input_coords: CoordSystem,
    output_coords: CoordSystem,
    nsteps: int,
    device: Any,
    stochastic: bool,
) -> None:
    """Evaluate the rules that require stepping the model.

    Covers ``P7``-``P10`` and ``P13``-``P15``.
    """
    coords = _concretize(input_coords)
    x = _sample_tensor(coords, device)
    pristine_x = x.clone()
    pristine_coords = OrderedDict((key, value.copy()) for key, value in coords.items())

    # Seed first: a stochastic model may require set_rng before it can be stepped,
    # and a caller driving a rollout would seed it too
    set_rng = getattr(model, "set_rng", None)
    if stochastic and callable(set_rng):
        set_rng(0)

    # Snapshot during iteration, not after: yields that alias one buffer have already
    # converged on the final step's values by the time the rollout finishes
    steps: list[tuple[torch.Tensor, CoordSystem]] = []
    snapshots: list[torch.Tensor] = []
    for values, step_coords in islice(model.create_iterator(x, coords), nsteps + 1):
        steps.append((values, step_coords))
        snapshots.append(values.clone())

    if not report.require("P7", len(steps) > 0, "create_iterator() yielded nothing"):
        return

    _check_immutability(
        report, x, pristine_x, coords, pristine_coords, "create_iterator()"
    )
    for index, ((values, _), snapshot) in enumerate(zip(steps, snapshots)):
        if not report.require(
            "P15",
            values.shape == snapshot.shape and torch.equal(values, snapshot),
            f"yield {index} changed after later steps were produced, so the yields "
            "alias one buffer; a caller holding a yield across steps — an async IO "
            "write, a resume buffer — reads the wrong values",
        ):
            break

    zeroth_x, zeroth_coords = steps[0]
    expected = coords.copy()
    expected["lead_time"] = coords["lead_time"][-1:]
    report.require(
        "P7",
        list(zeroth_coords) == list(expected),
        "the 0th yield must carry the input dimensions in order; expected "
        f"{list(expected)}, got {list(zeroth_coords)}",
    )
    report.require(
        "P7",
        "lead_time" in zeroth_coords
        and np.array_equal(zeroth_coords["lead_time"], expected["lead_time"]),
        "the 0th yield is the initial condition, so its lead_time must be the final "
        f"input lead_time {expected['lead_time']}, got "
        f"{zeroth_coords.get('lead_time')}",
    )

    if len(steps) > 1:
        first_coords = steps[1][1]
        report.require(
            "P8",
            list(first_coords) == list(output_coords),
            "the 1st yield must carry the dimensions declared by output_coords(); "
            f"expected {list(output_coords)}, got {list(first_coords)}",
        )
        report.require(
            "P8",
            "lead_time" in first_coords
            and np.array_equal(first_coords["lead_time"], output_coords["lead_time"]),
            "the 1st yield must land on the lead_time declared by output_coords(); "
            f"expected {output_coords['lead_time']}, got "
            f"{first_coords.get('lead_time')}",
        )
    else:
        report.skip("P8", "iterator exhausted before the first forecast step")

    for index, (step_x, step_coords) in enumerate(steps):
        step_shape = _expected_shape(step_coords)
        if step_shape is None:
            report.skip("P9", f"yield {index} implies no tensor shape")
            continue
        report.require(
            "P9",
            tuple(step_x.shape) == step_shape,
            f"yield {index} shape {tuple(step_x.shape)} does not match its "
            f"coordinate system {step_shape}",
        )

    _check_call_immutability(report, model, coords, device)
    _check_hook_scope(report, model, coords, device)
    _check_reproducibility(report, model, coords, device, nsteps, stochastic)


def _rollout_values(
    model: PrognosticModel, x: torch.Tensor, coords: CoordSystem, nsteps: int
) -> torch.Tensor:
    """Concatenate the forecast steps of a rollout into one tensor.

    The input is cloned per rollout so that a model violating ``P14`` cannot make
    successive rollouts disagree for the wrong reason.
    """
    steps = islice(model.create_iterator(x.clone(), coords.copy()), 1, nsteps + 1)
    return torch.cat([values.flatten().clone() for values, _ in steps])


def _check_stochasticity(
    report: _Report,
    model: Any,
    declare_rule: str = "P11",
    set_rng_rule: str = "P12",
) -> bool:
    """Evaluate the stochasticity declaration rules.

    Shared by both protocols: ``P11``/``P12`` for prognostics, ``D7``/``D8`` for
    diagnostics. The rule identifiers differ, the requirement does not.

    Returns
    -------
    bool
        Whether the model declares itself stochastic
    """
    stochastic = getattr(model, "stochastic", False)
    report.require(
        declare_rule,
        isinstance(stochastic, bool),
        "model must declare a boolean 'stochastic' attribute; randomness that is "
        f"not declared cannot be planned for, got {stochastic!r}",
    )
    stochastic = bool(stochastic)

    set_rng = getattr(model, "set_rng", None)
    if not stochastic:
        return False

    if not report.require(
        set_rng_rule,
        callable(set_rng),
        "a stochastic model must implement set_rng(seed, reset=True) so a caller "
        "can make its output reproducible",
    ):
        return True

    parameters = [
        name
        for name, parameter in signature(cast(Callable, set_rng)).parameters.items()
        if parameter.kind
        in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
    ]
    report.require(
        set_rng_rule,
        parameters[:1] == ["seed"],
        "set_rng() must take the seed as its first positional argument, got "
        f"{parameters or 'no positional arguments'}",
    )
    return True


def _check_diagnostic_reproducibility(
    report: _Report,
    model: DiagnosticModel,
    coords: CoordSystem,
    device: Any,
    stochastic: bool,
) -> None:
    """Evaluate the diagnostic reproducibility rule (``D9``).

    A diagnostic has no rollout, so the property is over a single call: seeding
    determines the output, and two seeds do not give the same one.
    """
    x = _sample_tensor(coords, device)

    def run() -> torch.Tensor:
        out, _ = model(x.clone(), coords.copy())
        return out.detach().flatten().clone()

    if not stochastic:
        report.require(
            "D9",
            torch.allclose(run(), run()),
            "model declares stochastic=False but two calls on one input disagree; "
            "declare stochastic=True and implement set_rng()",
        )
        return

    set_rng = getattr(model, "set_rng", None)
    if not callable(set_rng):
        report.skip("D9", "stochastic model does not implement set_rng")
        return

    set_rng(0)
    first = run()
    set_rng(0)
    report.require(
        "D9",
        torch.allclose(first, run()),
        "reseeding with the same seed must reproduce the output exactly",
    )
    set_rng(1)
    report.require(
        "D9",
        not torch.allclose(first, run()),
        "two seeds produced identical output, so set_rng() does not reach every "
        "source of randomness; ensemble members would be duplicates",
    )


def _check_reproducibility(
    report: _Report,
    model: PrognosticModel,
    coords: CoordSystem,
    device: Any,
    nsteps: int,
    stochastic: bool,
) -> None:
    """Evaluate the reproducibility rule (``P13``)."""
    x = _sample_tensor(coords, device)

    if not stochastic:
        report.require(
            "P13",
            torch.allclose(
                _rollout_values(model, x, coords, nsteps),
                _rollout_values(model, x, coords, nsteps),
            ),
            "model declares stochastic=False but two rollouts from one input "
            "disagree; declare stochastic=True and implement set_rng()",
        )
        return

    set_rng = getattr(model, "set_rng", None)
    if not callable(set_rng):
        report.skip("P13", "stochastic model does not implement set_rng")
        return

    set_rng(0)
    first = _rollout_values(model, x, coords, nsteps)
    set_rng(0)
    repeated = _rollout_values(model, x, coords, nsteps)
    report.require(
        "P13",
        torch.allclose(first, repeated),
        "reseeding with the same seed must reproduce a rollout exactly; a resumed "
        "run cannot otherwise match the run it resumes",
    )

    set_rng(1)
    reseeded = _rollout_values(model, x, coords, nsteps)
    report.require(
        "P13",
        not torch.allclose(first, reseeded),
        "two seeds produced an identical rollout, so set_rng() does not reach every "
        "source of randomness; ensemble members would be duplicates",
    )


def _check_hook_scope(
    report: _Report, model: PrognosticModel, coords: CoordSystem, device: Any
) -> None:
    """Evaluate the hook scope rule (``P10``)."""
    # Bound separately so the narrowing does not shadow the PrognosticModel methods
    hooks = model if isinstance(model, PrognosticMixin) else None
    if hooks is None:
        report.skip("P10", "model does not use PrognosticMixin hooks")
        return

    calls: list[str] = []

    def front(
        x: torch.Tensor, hook_coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        calls.append("front")
        return x, hook_coords

    def rear(
        x: torch.Tensor, hook_coords: CoordSystem
    ) -> tuple[torch.Tensor, CoordSystem]:
        calls.append("rear")
        return x, hook_coords

    hooks.add_front_hook(front)
    hooks.add_rear_hook(rear)
    try:
        x = _sample_tensor(coords, device)
        next(islice(model.create_iterator(x, coords.copy()), 1, 2))
        iterator_calls = set(calls)

        calls.clear()
        model(_sample_tensor(coords, device), coords.copy())
        call_calls = set(calls)
    finally:
        hooks.clear_hooks()

    report.require(
        "P10",
        iterator_calls == {"front", "rear"},
        "create_iterator() must apply both hook chains on every forecast step, "
        f"applied {iterator_calls or 'neither'}; a hook a caller sets must not be "
        "silently dropped",
    )
    report.require(
        "P10",
        call_calls == set(),
        "__call__ is the single-step primitive and must not apply hooks, applied "
        f"{call_calls}; a caller holding one step can transform the tensor directly, "
        "and two step paths with different semantics is the ambiguity to avoid",
    )


def check_diagnostic_contract(
    model: DiagnosticModel,
    *,
    forward: bool = True,
    device: Any = "cpu",
) -> list[str]:
    """Check a diagnostic model against the Earth2Studio model contract.

    Parameters
    ----------
    model : DiagnosticModel
        Model to check
    forward : bool, optional
        Whether to run the rule that requires a forward pass (``D5``), by default
        True
    device : Any, optional
        Device to run the forward pass on, by default ``"cpu"``

    Returns
    -------
    list[str]
        Rules that could not be evaluated, each with the reason it was skipped

    Raises
    ------
    ContractViolation
        If the model violates any evaluated rule
    """
    report = _Report(model)
    report.require(
        "D1",
        isinstance(model, DiagnosticModel),
        "model does not structurally satisfy the DiagnosticModel protocol",
    )

    input_coords = model.input_coords()
    output_coords = _check_coord_declaration(
        report,
        model,
        input_coords,
        coords_rule="D2",
        readonly_rule="D3",
        invalid_rule="D4",
    )

    stochastic = _check_stochasticity(report, model, "D7", "D8")

    coords = _concretize(input_coords)
    if not forward:
        report.skip("D5, D6, D9", "forward check disabled")
    elif _expected_shape(coords) is None or output_coords is None:
        report.skip(
            "D5, D6, D9",
            "input_coords() does not imply a tensor shape, so no probe input can be "
            "built",
        )
    else:
        out, out_coords = model(_sample_tensor(coords, device), coords.copy())
        report.require(
            "D5",
            list(out_coords) == list(output_coords),
            "__call__ must return the dimensions declared by output_coords(); "
            f"expected {list(output_coords)}, got {list(out_coords)}",
        )
        out_shape = _expected_shape(out_coords)
        if out_shape is not None:
            report.require(
                "D5",
                tuple(out.shape) == out_shape,
                f"__call__ shape {tuple(out.shape)} does not match its coordinate "
                f"system {out_shape}",
            )
        _check_call_immutability(report, model, coords, device, rule="D6")
        _check_diagnostic_reproducibility(report, model, coords, device, stochastic)

    report.raise_for_violations()
    return report.skipped


def iter_contract_rules() -> Iterator[tuple[str, str]]:
    """Yield the rule identifiers and summaries documented by the contract spec.

    Yields
    ------
    Iterator[tuple[str, str]]
        Pairs of rule identifier and one-line summary
    """
    yield from _RULES.items()


_RULES = {
    "P1": "A prognostic model structurally satisfies the PrognosticModel protocol.",
    "P2": "Coordinate systems are ordered dictionaries of arrays led by 'batch'.",
    "P3": "Input lead_time is relative, strictly increasing, and ends at zero.",
    "P4": "output_coords() treats its argument as read-only.",
    "P5": "output_coords() raises ValueError for an invalid coordinate system.",
    "P6": "Shifting input lead_time shifts output lead_time by the same offset.",
    "P7": "create_iterator() yields the initial condition as its 0th step.",
    "P8": "The 1st yield matches the coordinates declared by output_coords().",
    "P9": "Every yielded tensor shape matches its coordinate system.",
    "P10": "create_iterator() applies both hook chains; __call__ applies neither.",
    "P11": "The model declares a boolean 'stochastic' attribute.",
    "P12": "A stochastic model implements set_rng(seed, reset=True).",
    "P13": "Seeding determines a rollout, and different seeds give different rollouts.",
    "P14": "Stepping the model does not modify its input tensor or coordinates.",
    "P15": "A yielded tensor does not change once a later step is produced.",
    "D1": "A diagnostic model structurally satisfies the DiagnosticModel protocol.",
    "D2": "Coordinate systems are ordered dictionaries of arrays led by 'batch'.",
    "D3": "output_coords() treats its argument as read-only.",
    "D4": "output_coords() raises ValueError for an invalid coordinate system.",
    "D5": "__call__ returns the coordinates declared by output_coords().",
    "D6": "__call__ does not modify its input tensor or coordinates.",
    "D7": "The model declares a boolean 'stochastic' attribute.",
    "D8": "A stochastic model implements set_rng(seed, reset=True).",
    "D9": "Seeding determines the output, and different seeds give different output.",
}
