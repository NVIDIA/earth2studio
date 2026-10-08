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

The rule identifiers reported here (``P1``-``P24``, ``D1``-``D11``) match the rule
table in ``dev/spec/MODEL_CONTRACT_SPEC.md``.

Prognostic probes use forecasts-only iteration and explicit continuation state.
Execution and coordinate planning take separate positional input slots.
Probes never call recommended sources. Model-specific numerical
correctness, semantic source ordering, and absence of internal fetching still
require wrapper tests and review.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import fields, is_dataclass
from inspect import Parameter, signature
from typing import Any

import numpy as np
import pandas as pd
import torch
import xarray as xr

from earth2studio.data.base import DataSource, ForecastSource
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.models.px.base import PrognosticModel
from earth2studio.utils import coord_array_like, handshake_metadata
from earth2studio.utils.coords import E2S_DYNAMIC_DIMS, E2S_KIND, E2S_SCHEMA_VERSION
from earth2studio.utils.cupy import from_torch

# Time stamped onto an open-ended 'time' dimension when a declaration leaves it
# unsized. Models valid only over a restricted period take the time to probe with
# through the ``time`` argument of the public check functions.
_PROBE_TIME = np.datetime64("2024-01-01T00:00:00")

_RULES = {
    "P1": "A prognostic model structurally satisfies the PrognosticModel protocol.",
    "P2": "Declarations are allocation-free DataArray signatures.",
    "P3": "Input lead_time is relative, strictly increasing, and ends at zero.",
    "P4": "output_coords() treats its argument as read-only.",
    "P5": "output_coords() raises ValueError for an invalid coordinate system.",
    "P6": "Shifting input lead_time shifts output lead_time by the same offset.",
    "P7": "Iteration yields forecasts only, starting with initialization output.",
    "P8": "The first yield matches planned output coordinates.",
    "P9": "Every output matches planned coordinates and structural metadata.",
    "P10": "Hooks are iterator-only and both affect recurrence.",
    "P11": "The model declares a boolean 'stochastic' attribute.",
    "P12": "A stochastic model implements set_rng(seed, reset=True).",
    "P13": "Seeding determines a rollout, and different seeds give different rollouts.",
    "P14": "After set_rng(), seeding and stepping leave global RNG state unperturbed.",
    "P15": "Execution does not mutate borrowed inputs or coordinates.",
    "P16": "Advances preserve earlier yields except explicit in-place hook edits.",
    "P17": "Execution accepts one positional DataArray per declared slot.",
    "P18": "Output slots have distinct non-variable coordinates.",
    "P19": "Step is pure and replays exactly from outputs and state.",
    "P20": "Call, initialization and the first hook-free yield agree.",
    "P21": "Outputs and serializable state contain the complete continuation.",
    "P22": "Models require supplied forcing and never fetch internally.",
    "P23": "Default sources match input-then-forcing slots.",
    "P24": "Execution methods have fixed named parameters.",
    "D1": "A diagnostic model structurally satisfies the DiagnosticModel protocol.",
    "D2": "Declarations are allocation-free DataArray signatures.",
    "D3": "output_coords() treats its argument as read-only.",
    "D4": "output_coords() raises ValueError for an invalid coordinate system.",
    "D5": "Outputs match planned coordinates and structural metadata.",
    "D6": "Execution does not mutate borrowed inputs or coordinates.",
    "D7": "The model declares a boolean 'stochastic' attribute.",
    "D8": "A stochastic model implements set_rng(seed, reset=True).",
    "D9": "Seeding determines the output, and different seeds give different output.",
    "D10": "After set_rng(), seeding and calling leave global RNG state unperturbed.",
    "D11": "The public call has fixed named parameters.",
}
_ROLLOUT_RULES = (
    "P7",
    "P8",
    "P9",
    "P10",
    "P13",
    "P15",
    "P16",
    "P19",
    "P20",
    "P21",
    "P22",
)


def _validate_rule(rule: str) -> None:
    """Raise if a rule identifier is not one declared in ``_RULES``.

    A plain assertion would work here but is stripped under ``python -O``; this is
    an internal-consistency guard against a typo'd rule ID (e.g. ``"P1O"``) shipping
    silently as an unrecognized violation string, not a user input check, so it must
    hold even under optimized execution.
    """
    if rule not in _RULES:
        raise ValueError(f"{rule!r} is not a rule ID declared in _RULES")


class ContractException(Exception):
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
    """Collects rule outcomes so every rule is evaluated before failing.

    Every identifier passed to :meth:`require` or :meth:`skip` is validated against
    ``_RULES``, so a typo'd rule ID (e.g. ``"P1O"``) fails loudly instead of silently
    shipping as an unrecognized violation string. ``evaluated`` accumulates every
    rule this report has recorded an outcome for, which is what lets a completeness
    test confirm every documented rule is reachable (see
    ``test_all_rules_are_reachable`` in ``test/models/test_conformance.py``).
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        self.violations: list[str] = []
        self.skipped: list[str] = []
        self.evaluated: set[str] = set()

    def require(self, rule: str, condition: bool, message: str) -> bool:
        """Record a rule outcome and report whether it held."""
        _validate_rule(rule)
        self.evaluated.add(rule)
        if not condition:
            self.violations.append(f"{rule}: {message}")
        return condition

    def skip(self, rule: str | tuple[str, ...], message: str) -> None:
        """Record that one or more rules could not be evaluated."""
        rules = (rule,) if isinstance(rule, str) else rule
        for single_rule in rules:
            _validate_rule(single_rule)
            self.evaluated.add(single_rule)
            self.skipped.append(f"{single_rule}: {message}")

    def probe(self, rule: str, action: Callable[[], Any]) -> Any:
        """Collect execution errors without aborting independent probes."""
        try:
            return action()
        except Exception as error:  # noqa: BLE001 - model failures become violations
            self.require(rule, False, f"probe raised {error!r}")
            return None

    def raise_for_violations(self) -> None:
        """Raise if any rule failed."""
        if self.violations:
            raise ContractException(self.model, self.violations)


def _slots(value: Any) -> tuple[xr.DataArray, ...]:
    slots = value if isinstance(value, tuple) else (value,)
    if not slots or not all(isinstance(x, xr.DataArray) for x in slots):
        raise ValueError("expected a DataArray or a nonempty tuple of DataArrays")
    return slots


def _group(slots: tuple[xr.DataArray, ...]) -> Any:
    return slots[0] if len(slots) == 1 else slots


def _same_coords(first: xr.DataArray, second: xr.DataArray) -> bool:
    return (
        first.dims == second.dims
        and first.sizes == second.sizes
        and first.coords.to_dataset().identical(second.coords.to_dataset())
        and _same_values(first.attrs, second.attrs)
        and _same_values(first.encoding, second.encoding)
    )


def _concretize(
    coords: xr.DataArray,
    batch_size: int = 1,
    time: np.datetime64 = _PROBE_TIME,
) -> xr.DataArray:
    """Replace the open-ended coordinates of a declaration with concrete values.

    Model declarations use zero-length arrays to mean "any size" on batch-like
    dimensions. Contract checks need a runnable coordinate system, so each such
    dimension is pinned to ``batch_size``.

    Parameters
    ----------
    coords : xr.DataArray
        Declared coordinate system, typically from ``input_coords()``
    batch_size : int, optional
        Size to give open-ended dimensions, by default 1
    time : np.datetime64, optional
        Timestamp to fill an open-ended 'time' dimension with, by default
        2024-01-01T00:00

    Returns
    -------
    xr.DataArray
        Coordinate system with every dimension concretely sized
    """
    replacements = {}
    for dim in coords.attrs.get(E2S_DYNAMIC_DIMS, ()):
        coordinate = coords.coords.get(dim)
        dtype = coordinate.dtype if coordinate is not None else None
        if dtype is not None and dtype.kind == "M":
            replacements[dim] = np.full(batch_size, time, dtype=dtype)
        elif dtype is not None and dtype.kind == "m":
            replacements[dim] = np.zeros(batch_size, dtype=dtype)
        elif dim == "time":
            replacements[dim] = np.array([time] * batch_size)
        elif dim == "lead_time":
            replacements[dim] = np.zeros(batch_size, dtype="timedelta64[ns]")
        else:
            replacements[dim] = np.arange(batch_size)
    return coord_array_like(coords, replacements)


def _sample_tensor(coords: xr.DataArray, device: Any) -> xr.DataArray:
    """Build a probe tensor matching a coordinate system.

    Values are pseudo-random rather than zero so that a model writing into its input
    is detectable, and seeded so that repeated calls return an identical tensor.
    """
    generator = torch.Generator().manual_seed(0)
    tensor = torch.randn(coords.shape, generator=generator).to(device)
    return from_torch(tensor, coords)


def _swap_last_dims(coords: xr.DataArray) -> xr.DataArray:
    """Return a copy of a coordinate system with its final two dimensions swapped."""
    keys = list(coords.dims)
    keys[-1], keys[-2] = keys[-2], keys[-1]
    return coords.transpose(*keys)


def _mismatched_coord_keys(
    expected: xr.DataArray,
    actual: xr.DataArray,
) -> list[str]:
    """Keys present in both coordinate systems whose values differ."""
    if not isinstance(actual, xr.DataArray):
        return ["DataArray type"]
    mismatched = []
    if expected.dims != actual.dims or expected.sizes != actual.sizes:
        mismatched.append("dimensions")
    if not expected.coords.to_dataset().identical(actual.coords.to_dataset()):
        mismatched.append("coordinates")
    if any(
        key in actual.attrs for key in (E2S_KIND, E2S_SCHEMA_VERSION, E2S_DYNAMIC_DIMS)
    ):
        mismatched.append("signature metadata on field")
    # Hook-owned user attributes are flexible; physical grid/statistic metadata
    # must agree with the independently planned declaration, including absence.
    try:
        handshake_metadata(
            actual,
            expected,
            (
                "earth2studio_grid_id",
                "earth2studio_crs",
                "earth2studio_statistics",
                "type",
                "dims",
                "shape",
                "topology",
                "crs",
                "level",
                "nside",
                "ordering",
                "layout",
                "origin",
                "clockwise",
            ),
        )
    except ValueError:
        mismatched.append("structural metadata")
    return mismatched


def _declaration(
    report: _Report, slots: tuple[xr.DataArray, ...], rule: str, dynamic: bool = True
) -> None:
    for x in slots:
        report.require(
            rule,
            x.attrs.get(E2S_KIND) == "coordinate_array" and x.data.nbytes == 0,
            "declarations must be allocation-free coordinate signatures",
        )
        dims = tuple(x.attrs.get(E2S_DYNAMIC_DIMS, ()))
        report.require(
            rule,
            x.dims[: len(dims)] == dims and all(x.sizes[d] == 0 for d in dims),
            "dynamic dimensions must be an empty leading prefix",
        )
        if not dynamic:
            report.require(
                rule, not dims, "planning must retain concrete leading dimensions"
            )


def _check_coord_declaration(
    report: _Report,
    model: Any,
    declared: tuple[xr.DataArray, ...],
    *,
    coords_rule: str,
    readonly_rule: str,
    invalid_rule: str,
    time: np.datetime64 = _PROBE_TIME,
) -> tuple[tuple[xr.DataArray, ...], tuple[xr.DataArray, ...]] | None:
    """Evaluate the declaration rules shared by prognostic and diagnostic models.

    The rule identifiers differ between the two protocols, so each caller supplies
    the identifier its spec section uses.
    """
    _declaration(report, declared, coords_rule)
    concrete = tuple(_concretize(x, time=time) for x in declared)
    reference = deepcopy(concrete)
    try:
        output_coords = _slots(model.output_coords(*concrete))
    except Exception as error:  # noqa: BLE001 - reported as a violation
        report.require(
            readonly_rule,
            False,
            f"output_coords() raised on its own input_coords(): {error!r}",
        )
        return None

    report.require(
        readonly_rule,
        all(_same_coords(a, b) for a, b in zip(concrete, reference)),
        "output_coords() mutated its input coordinates or metadata",
    )
    _declaration(report, output_coords, coords_rule, dynamic=False)
    tested = False
    for index, input_coords in enumerate(declared):
        dynamic = tuple(input_coords.attrs.get(E2S_DYNAMIC_DIMS, ()))
        fixed_ndim = input_coords.ndim - len(dynamic)
        if fixed_ndim < 2:
            continue
        tested = True
        bad = list(deepcopy(reference))
        bad[index] = _swap_last_dims(bad[index])
        try:
            model.output_coords(*bad)
        except ValueError:
            report.require(invalid_rule, True, "")
        except Exception as error:  # noqa: BLE001 - reported as a violation
            report.require(
                invalid_rule,
                False,
                f"output_coords() raised {type(error).__name__} for a misordered coordinate system; it must raise ValueError",
            )
        else:
            report.require(
                invalid_rule, False, "output_coords() accepted swapped fixed dimensions"
            )
    if not tested:
        report.skip(invalid_rule, "model declares fewer than two fixed dimensions")
    return reference, output_coords


def check_prognostic_contract(
    model: PrognosticModel,
    rollout: bool = True,
    nsteps: int = 2,
    device: Any = "cpu",
    time: np.datetime64 = _PROBE_TIME,
) -> list[str]:
    """Check a prognostic model against the Earth2Studio model contract.

    Every rule is evaluated before the check fails, so a single call reports all
    violations rather than only the first.

    Parameters
    ----------
    model : PrognosticModel
        Model to check
    rollout : bool, optional
        Whether to run execution, replay, forcing and hook checks, by default True.
        False retains declaration, signature, source and seeding checks.
    nsteps : int, optional
        Forecast yields to check, by default 2. At least two are probed to check
        continuation and ownership; initial conditions are never counted.
    device : Any, optional
        Device to run the rollout on, by default ``"cpu"``
    time : np.datetime64, optional
        Timestamp to probe the model at, by default 2024-01-01T00:00. Models valid
        only over a restricted period need a time inside it.

    Returns
    -------
    list[str]
        Rules that could not be evaluated, each with the reason it was skipped

    Raises
    ------
    ContractException
        If the model violates any evaluated rule
    ValueError
        If nsteps is not a positive integer.
    """
    report = _evaluate_prognostic(
        model, rollout=rollout, nsteps=nsteps, device=device, time=time
    )
    report.raise_for_violations()
    return report.skipped


def _evaluate_prognostic(
    model: PrognosticModel,
    rollout: bool = True,
    nsteps: int = 2,
    device: Any = "cpu",
    time: np.datetime64 = _PROBE_TIME,
) -> _Report:
    """Run every prognostic rule and return the report without raising.

    Split from :func:`check_prognostic_contract` so a test can inspect
    ``report.evaluated`` (e.g. to confirm every documented rule is reachable)
    without needing a model that fails nothing.
    """
    if type(nsteps) is not int or nsteps < 1:
        raise ValueError("nsteps must be a positive integer")
    report = _Report(model)
    report.require(
        "P1",
        isinstance(model, PrognosticModel),
        "model does not structurally satisfy the PrognosticModel protocol",
    )
    declared = report.probe("P2", lambda: _slots(model.input_coords()))
    if declared is None:
        return report
    for input_coords in declared:
        lead_time = input_coords.coords.get("lead_time")
        if lead_time is not None:
            lead_time = np.asarray(lead_time)
        if lead_time is None:
            report.require(
                "P3", False, "input_coords() must declare a 'lead_time' dimension"
            )
        else:
            valid_lead = report.require(
                "P3",
                lead_time.ndim == 1 and np.issubdtype(lead_time.dtype, np.timedelta64),
                f"input_coords()['lead_time'] must hold timedeltas, got {lead_time.dtype}",
            )
            report.require(
                "P3",
                valid_lead
                and lead_time.size > 0
                and not np.isnat(lead_time).any()
                and lead_time[-1] == np.timedelta64(0, "h"),
                "input_coords()['lead_time'] must be relative and end at zero, so that "
                f"the final entry is the analysis time, got {lead_time}",
            )
            report.require(
                "P3",
                valid_lead
                and (
                    lead_time.size < 2
                    or bool(np.all(np.diff(lead_time) > np.timedelta64(0)))
                ),
                f"input_coords()['lead_time'] must be strictly increasing, got {lead_time}",
            )

    planned = report.probe(
        "P2",
        lambda: _check_coord_declaration(
            report,
            model,
            declared,
            coords_rule="P2",
            readonly_rule="P4",
            invalid_rule="P5",
            time=time,
        ),
    )
    raw_forcing = report.probe("P2", lambda: model.forcing_coords())
    forcing = report.probe(
        "P2", lambda: () if raw_forcing is None else _slots(raw_forcing)
    )
    if planned is None or forcing is None:
        return report
    inputs, outputs = planned
    _declaration(report, forcing, "P2")
    forcing = tuple(_concretize(f, time=time) for f in forcing)
    initial_count = len(inputs) + len(forcing)
    arities = dict.fromkeys(
        ("__call__", "initialize", "create_iterator"), initial_count
    )
    arities["step"] = len(outputs) + sum("lead_time" in f.dims for f in forcing) + 1
    _signatures(report, model, arities, "P24")
    report.require(
        "P17",
        not any(v.startswith("P24:") for v in report.violations),
        "execution arity must match separate declared array slots",
    )
    report.probe("P23", lambda: _sources(report, model, initial_count, "P23"))
    for index, x in enumerate(outputs):
        for other in outputs[:index]:
            a = x.coords.to_dataset().drop_dims("variable", errors="ignore")
            b = other.coords.to_dataset().drop_dims("variable", errors="ignore")
            report.require(
                "P18",
                not a.identical(b),
                "merge output slots with identical non-variable coordinates",
            )
    report.require("P18", True, "")

    report.probe("P6", lambda: _check_rebasing(report, model, inputs, outputs))
    stochastic = _check_stochasticity(report, model, device=device)
    if not rollout:
        report.skip(_ROLLOUT_RULES, "rollout checks disabled")
        if stochastic:
            report.skip("P14", "rollout checks disabled")
    else:
        report.probe(
            "P9",
            lambda: _check_rollout(
                report, model, inputs, outputs, forcing, nsteps, device, stochastic
            ),
        )
    return report


def _check_rebasing(
    report: _Report,
    model: PrognosticModel,
    input_coords: tuple[xr.DataArray, ...],
    output_coords: tuple[xr.DataArray, ...],
) -> None:
    """Evaluate the lead-time rebasing rule (``P6``)."""
    offset = np.timedelta64(24, "h")
    shifted = tuple(
        coord_array_like(x, {"lead_time": x.lead_time.values + offset})
        for x in input_coords
    )
    rebased = _slots(model.output_coords(*shifted))
    report.require(
        "P6",
        len(rebased) == len(output_coords)
        and all(
            np.array_equal(a.lead_time.values, b.lead_time.values + offset)
            for a, b in zip(rebased, output_coords)
        ),
        "shifting input lead_time by a constant must shift output lead_time by the "
        "same constant",
    )


def _reject(
    report: _Report, rule: str, action: Callable[[], Any], missing: bool = False
) -> None:
    try:
        action()
    except ValueError:
        report.require(rule, True, "")
    except TypeError as error:
        report.require(
            rule, missing, f"invalid data must raise ValueError, got {error!r}"
        )
    except Exception as error:  # noqa: BLE001 - report the wrong exception
        report.require(
            rule, False, f"invalid data raised {error!r}, expected ValueError"
        )
    else:
        report.require(rule, False, "invalid or missing input was accepted")


def _signatures(
    report: _Report, model: Any, arities: dict[str, int], rule: str
) -> None:
    for name, count in arities.items():

        def check() -> None:
            parameters = list(signature(getattr(model, name)).parameters.values())
            report.require(
                rule,
                all(
                    p.kind not in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
                    for p in parameters
                ),
                f"{name} must declare explicit parameters without *args or **kwargs",
            )
            positional = [
                p
                for p in parameters
                if p.kind
                in (Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD)
            ]
            report.require(
                rule,
                len(positional) == count,
                f"{name} needs {count} positional parameters for its declared slots",
            )
            if name == "step":
                report.require(
                    rule,
                    bool(positional)
                    and positional[-1].name == "state"
                    and positional[-1].kind == Parameter.POSITIONAL_OR_KEYWORD,
                    "step must end with positional-or-keyword state",
                )

        report.probe(rule, check)


def _sources(report: _Report, model: Any, count: int, rule: str) -> None:
    sources = model.default_sources()
    if sources is None:
        report.require(rule, True, "")
        return
    slots = sources if isinstance(sources, tuple) else (sources,)
    report.require(
        rule,
        len(slots) == count,
        "default_sources must match the input-then-forcing slot count",
    )
    report.require(
        rule,
        all(s is None or _source_callable(s) for s in slots),
        "default_sources entries must be raw data sources or None",
    )


def _source_callable(source: Any) -> bool:
    if isinstance(source, (DataSource, ForecastSource)):
        return True
    if not callable(source):
        return False
    # Synchronous sources need not implement the optional async fetch path.
    try:
        parameters = signature(source).parameters
        if not {"time", "variable"}.issubset(parameters):
            return False
        signature(source).bind(
            **{
                name: None
                for name in ("time", "lead_time", "variable")
                if name in parameters
            }
        )
    except (TypeError, ValueError):
        return False
    return True


def _matches(
    report: _Report, value: Any, expected: tuple[xr.DataArray, ...], rule: str
) -> None:
    actual = _slots(value)
    report.require(
        rule,
        (len(expected) > 1) == isinstance(value, tuple)
        and len(actual) == len(expected),
        "return one array for one output, otherwise a tuple in declared order",
    )
    for x, coords in zip(actual, expected):
        report.require(
            rule,
            not _mismatched_coord_keys(coords, x),
            "output differs from planned dimensions, coordinates or structural metadata",
        )


def _check_rollout(
    report: _Report,
    model: Any,
    inputs: tuple[xr.DataArray, ...],
    outputs: tuple[xr.DataArray, ...],
    forcing: tuple[xr.DataArray, ...],
    nsteps: int,
    device: Any,
    stochastic: bool,
) -> None:
    """Evaluate the rules that require stepping the model.

    Covers ``P7``-``P10``, ``P13``-``P16``, ``P19``, ``P20`` and ``P22``.
    Serialization coverage (``P21``) requires component-specific checkpoint tests.
    """
    report.skip("P21", "checkpoint serialization requires component-specific tests")
    args = tuple(_sample_tensor(x, device) for x in (*inputs, *forcing))
    with _hook_free(model):
        _seed(model)
        pair = report.probe(
            "P19",
            lambda: _call_readonly(report, model.initialize, deepcopy(args), "P15"),
        )
        _seed(model)
        called = report.probe(
            "P9", lambda: _call_readonly(report, model, deepcopy(args), "P15")
        )
        if called is not None:
            report.probe("P9", lambda: _matches(report, called, outputs, "P9"))
        _seed(model)
        snapshots = report.probe(
            "P9",
            lambda: _rollout_values(
                model, args, inputs, outputs, forcing, max(2, nsteps), device, report
            ),
        )
        if pair is not None:
            report.probe("P9", lambda: _matches(report, pair[0], outputs, "P9"))
            report.require(
                "P20",
                _same_values(called, pair[0]),
                "call differs from initialization forecast",
            )
            if snapshots is not None:
                report.require(
                    "P7",
                    _same_values(snapshots[0], pair[0]),
                    "first yield must be initialization forecast, not initial conditions",
                )
                report.require(
                    "P20",
                    _same_values(called, snapshots[0]),
                    "call differs from first hook-free forecast",
                )
            report.probe(
                "P19",
                lambda: _continuation(
                    report,
                    model,
                    pair,
                    forcing,
                    _next_plan(model, inputs, outputs),
                    device,
                ),
            )

        def run() -> list[Any]:
            return _rollout_values(
                model, args, inputs, outputs, forcing, max(2, nsteps), device
            )

        report.probe(
            "P13", lambda: _check_reproducibility(report, model, run, "P13", stochastic)
        )
        if stochastic:
            _seed(model)
            report.probe(
                "P14",
                lambda: _check_rng_isolation(
                    report, run, "stepping a seeded model", "P14", device
                ),
            )
        report.probe(
            "P10",
            lambda: _check_hook_scope(
                report, model, args, inputs, outputs, forcing, device
            ),
        )
        report.probe(
            "P22", lambda: _forcing_errors(report, model, args, forcing, device)
        )


def _same_values(first: Any, second: Any) -> bool:
    if type(first) is not type(second):
        return False
    if isinstance(first, xr.DataArray):
        return first.e2s.as_numpy().identical(second.e2s.as_numpy()) and _same_values(
            first.encoding, second.encoding
        )
    if isinstance(first, (pd.DataFrame, pd.Series, pd.Index)):
        return first.equals(second)
    if isinstance(first, torch.Tensor):
        return torch.equal(first, second)
    if isinstance(first, np.ndarray):
        return np.array_equal(first, second)
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(
            _same_values(a, b) for a, b in zip(first, second)
        )
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _same_values(v, second[k]) for k, v in first.items()
        )
    if is_dataclass(first):
        return all(
            _same_values(getattr(first, f.name), getattr(second, f.name))
            for f in fields(first)
        )
    return bool(first == second)


def _seed(model: Any, seed: int = 0) -> None:
    if getattr(model, "stochastic", False) and callable(
        getattr(model, "set_rng", None)
    ):
        model.set_rng(seed)


def _fork_devices(device: Any) -> list[int]:
    """CUDA device indices to fork RNG state for, empty when checking on CPU."""
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return []
    index = resolved.index
    return [index if index is not None else torch.cuda.current_device()]


def _rng_state(device: Any) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Snapshot the global RNG state that a caller owns.

    CUDA state is only read when the check runs on a CUDA device, so that a CPU check
    does not initialize a CUDA context. ``torch.manual_seed`` seeds the CPU generator
    as well as every device, so the CPU half of the snapshot catches it either way.
    """
    cuda_states = [torch.cuda.get_rng_state(d) for d in _fork_devices(device)]
    return torch.get_rng_state(), cuda_states


def _rng_state_unchanged(
    before: tuple[torch.Tensor, list[torch.Tensor]],
    after: tuple[torch.Tensor, list[torch.Tensor]],
) -> bool:
    """Whether two global RNG snapshots are identical."""
    cpu_before, cuda_before = before
    cpu_after, cuda_after = after
    if not torch.equal(cpu_before, cpu_after):
        return False
    return len(cuda_before) == len(cuda_after) and all(
        torch.equal(one, other) for one, other in zip(cuda_before, cuda_after)
    )


def _check_rng_isolation(
    report: _Report,
    action: Callable[[], Any],
    path: str,
    rule: str,
    device: Any,
) -> None:
    with torch.random.fork_rng(devices=_fork_devices(device)):
        before = _rng_state(device)
        action()
        after = _rng_state(device)
    report.require(
        rule,
        _rng_state_unchanged(before, after),
        f"{path} left the global RNG state perturbed, which silently reseeds every "
        "other consumer in the process — a second model in a cascade, a perturbation "
        "method, a dataloader. Seed a local torch.Generator, or confine global "
        "seeding to a torch.random.fork_rng() block",
    )


def _check_stochasticity(
    report: _Report,
    model: Any,
    declare_rule: str = "P11",
    set_rng_rule: str = "P12",
    isolation_rule: str = "P14",
    device: Any = "cpu",
) -> bool:
    """Evaluate the stochasticity declaration rules.

    Shared by both protocols: ``P11``/``P12``/``P14`` for prognostics,
    ``D7``/``D8``/``D10`` for diagnostics. The rule identifiers differ, the
    requirement does not.

    The seeding half of the isolation rule is evaluated here because it needs no
    forward pass; the stepping half runs with the rules that do.

    Returns
    -------
    bool
        Whether the model declares itself stochastic
    """
    stochastic = getattr(model, "stochastic", None)
    report.require(
        declare_rule,
        isinstance(stochastic, bool),
        "model must declare a boolean 'stochastic' attribute; randomness that is "
        f"not declared cannot be planned for, got {stochastic!r}",
    )
    stochastic = bool(stochastic)

    set_rng = getattr(model, "set_rng", None)
    if not stochastic:
        report.require(set_rng_rule, True, "")
        report.skip(isolation_rule, "model does not declare itself stochastic")
        return False

    if not report.require(
        set_rng_rule,
        callable(set_rng),
        "a stochastic model must implement set_rng(seed, reset=True) so a caller "
        "can make its output reproducible",
    ):
        report.skip(isolation_rule, "stochastic model does not implement set_rng")
        return False

    def check() -> None:
        params = signature(set_rng).parameters
        report.require(
            set_rng_rule,
            list(params) == ["seed", "reset"] and params["reset"].default is True,
            "expected set_rng(seed, reset=True)",
        )
        _check_rng_isolation(
            report, lambda: set_rng(0), "set_rng()", isolation_rule, device
        )

    report.probe(set_rng_rule, check)
    return True


def _check_reproducibility(
    report: _Report, model: Any, run: Callable[[], Any], rule: str, stochastic: bool
) -> None:
    _seed(model)
    first = run()
    _seed(model)
    report.require(
        rule,
        _same_values(first, run()),
        "identical seeds/inputs must reproduce output exactly",
    )
    if stochastic:
        _seed(model)
        model.set_rng(1, reset=False)
        report.require(
            "P12" if rule == "P13" else "D8",
            _same_values(first, run()),
            "reset=False must preserve an existing seed",
        )
        _seed(model, 1)
        report.require(
            rule,
            not _same_values(first, run()),
            "different seeds produced identical output",
        )


@contextmanager
def _hook_free(model: Any) -> Iterator[None]:
    originals = {
        name: getattr(model, name)
        for name in ("front_hook", "rear_hook")
        if hasattr(model, name)
    }
    try:
        for name in originals:
            setattr(model, name, lambda y: y)
        yield
    finally:
        for name, hook in originals.items():
            setattr(model, name, hook)


def _next_plan(
    model: Any, inputs: tuple[xr.DataArray, ...], outputs: tuple[xr.DataArray, ...]
) -> tuple[xr.DataArray, ...]:
    end = max(x.lead_time.values[-1] for x in outputs)
    shifted = tuple(
        coord_array_like(
            x, {"lead_time": x.lead_time.values - x.lead_time.values[-1] + end}
        )
        for x in inputs
    )
    return _slots(model.output_coords(*shifted))


def _forcing(
    forcing: tuple[xr.DataArray, ...], outputs: tuple[xr.DataArray, ...], device: Any
) -> tuple[xr.DataArray, ...]:
    leads = np.unique(np.concatenate([x.lead_time.values for x in outputs]))
    return tuple(
        _sample_tensor(
            coord_array_like(f, {"lead_time": f.lead_time.values[-1] + leads}), device
        )
        for f in forcing
        if "lead_time" in f.dims
    )


def _rollout_values(
    model: Any,
    args: tuple[xr.DataArray, ...],
    inputs: tuple[xr.DataArray, ...],
    outputs: tuple[xr.DataArray, ...],
    forcing: tuple[xr.DataArray, ...],
    count: int,
    device: Any,
    report: _Report | None = None,
) -> list[Any]:
    borrowed = deepcopy(args)
    pristine = deepcopy(borrowed)
    retained, snapshots = [], []
    expected = outputs
    with closing(model.create_iterator(*borrowed)) as iterator:
        if not callable(getattr(iterator, "send", None)):
            raise ValueError("create_iterator must return a generator supporting send")
        value = next(iterator)
        for index in range(count):
            if index:
                new_forcing = _forcing(forcing, expected, device)
                saved_forcing = deepcopy(new_forcing)
                value = iterator.send(_group(new_forcing) if new_forcing else None)
                if report is not None:
                    report.require(
                        "P15",
                        _same_values(new_forcing, saved_forcing),
                        "iterator mutated supplied forcing",
                    )
                expected = _next_plan(model, inputs, expected)
            retained.append(value)
            snapshots.append(deepcopy(value))
            if report is not None:
                _matches(report, value, expected, "P9")
    if report is not None:
        origin = max(x.lead_time.values[-1] for x in inputs)
        first_leads = [x.lead_time.values for x in _slots(snapshots[0])]
        report.require(
            "P7",
            all(leads.size > 0 and np.all(leads > origin) for leads in first_leads),
            "initialization must yield future leads, not the initial condition",
        )
        report.require(
            "P15",
            _same_values(borrowed, pristine),
            "iterator mutated initialization inputs",
        )
        report.require(
            "P16",
            _same_values(retained, snapshots),
            "advancing the iterator changed an earlier yield",
        )
        _matches(report, snapshots[0], outputs, "P8")
    return snapshots


def _check_immutability(
    report: _Report,
    x: Any,
    pristine_x: Any,
    path: str,
    rule: str = "P15",
) -> None:
    """Evaluate the input immutability rule (``P15``, ``D6``) for one call path."""
    report.require(
        rule,
        _same_values(x, pristine_x),
        f"{path} modified input values, coordinates, metadata, or state in place",
    )


def _call_readonly(
    report: _Report,
    function: Callable[..., Any],
    args: tuple[Any, ...],
    rule: str,
) -> Any:
    saved = deepcopy(args)
    result = function(*args)
    _check_immutability(
        report,
        args,
        saved,
        function.__name__ if hasattr(function, "__name__") else "__call__",
        rule,
    )
    return result


def _continuation(
    report: _Report,
    model: Any,
    pair: Any,
    forcing: tuple[xr.DataArray, ...],
    expected: tuple[xr.DataArray, ...],
    device: Any,
) -> None:
    y, state = pair
    new_forcing = _forcing(forcing, _slots(y), device)
    pristine = deepcopy(pair)
    first = _call_readonly(report, model.step, (*_slots(y), *new_forcing, state), "P19")
    _matches(report, first[0], expected, "P9")
    # Reseeding the model must not affect a rollout whose RNG position is in state.
    _seed(model, 17)
    second = model.step(*_slots(pristine[0]), *deepcopy(new_forcing), state=pristine[1])
    report.require(
        "P19",
        _same_values(first, second),
        "step does not replay exactly from (y, state)",
    )


def _check_hook_scope(
    report: _Report,
    model: Any,
    args: tuple[xr.DataArray, ...],
    inputs: tuple[xr.DataArray, ...],
    outputs: tuple[xr.DataArray, ...],
    forcing: tuple[xr.DataArray, ...],
    device: Any,
) -> None:
    if not all(hasattr(model, name) for name in ("front_hook", "rear_hook")):
        report.skip("P10", "model exposes no iterator hook slots")
        return
    events = []

    def front(y: Any) -> Any:
        events.append("front")
        return _group(tuple(x + 0.25 for x in _slots(y)))

    def rear(y: Any) -> Any:
        events.append("rear")
        return _group(tuple(x + 0.5 for x in _slots(y)))

    with _hook_free(model):
        model.front_hook, model.rear_hook = front, rear
        _seed(model)
        model(*deepcopy(args))
        _seed(model)
        y, state = model.initialize(*deepcopy(args))
        f = _forcing(forcing, outputs, device)
        model.step(*_slots(deepcopy(y)), *deepcopy(f), state=deepcopy(state))
        report.require(
            "P10", not events, "call, initialize and step must not apply hooks"
        )
        # Compare to manually transformed recurrence, including rear output feedback.
        with _hook_free(model):
            _seed(model)
            y, state = model.initialize(*deepcopy(args))
            first = rear(y)
            second, _ = model.step(
                *_slots(front(deepcopy(first))), *deepcopy(f), state=state
            )
            second = rear(second)
        events.clear()
        _seed(model)
        actual = _rollout_values(model, args, inputs, outputs, forcing, 2, device)
        report.require(
            "P10", events == ["rear", "front", "rear"], f"wrong hook order: {events}"
        )
        report.require(
            "P10",
            _same_values(actual, [first, second]),
            "hook return values must feed subsequent recurrence",
        )


def _forcing_errors(
    report: _Report,
    model: Any,
    args: tuple[xr.DataArray, ...],
    forcing: tuple[xr.DataArray, ...],
    device: Any,
) -> None:
    if not forcing:
        report.require("P22", True, "")
        return
    _seed(model)
    _reject(report, "P22", lambda: model.initialize(*deepcopy(args[:-1])), missing=True)
    for index in range(len(forcing)):
        bad = list(deepcopy(args))
        bad[len(args) - len(forcing) + index] = xr.DataArray(0.0)
        _seed(model)
        _reject(report, "P22", lambda: model.initialize(*bad))
    if any("lead_time" in f.dims for f in forcing):
        _seed(model)
        y, state = model.initialize(*deepcopy(args))
        frames = _forcing(forcing, _slots(y), device)
        _reject(
            report,
            "P22",
            lambda: model.step(
                *_slots(deepcopy(y)), *deepcopy(frames[:-1]), state=deepcopy(state)
            ),
            missing=True,
        )
        for index in range(len(frames)):
            bad_frames = list(deepcopy(frames))
            bad_frames[index] = xr.DataArray(0.0)
            _reject(
                report,
                "P22",
                lambda: model.step(
                    *_slots(deepcopy(y)), *bad_frames, state=deepcopy(state)
                ),
            )
        _seed(model)
        with closing(model.create_iterator(*deepcopy(args))) as iterator:
            next(iterator)
            _reject(report, "P22", lambda: iterator.send(None))


def check_diagnostic_contract(
    model: DiagnosticModel,
    forward: bool = True,
    device: Any = "cpu",
    time: np.datetime64 = _PROBE_TIME,
) -> list[str]:
    """Check a diagnostic model against the Earth2Studio model contract.

    Parameters
    ----------
    model : DiagnosticModel
        Model to check
    forward : bool, optional
        Whether to run the rules that require a forward pass (``D5``, ``D6``,
        ``D9``, ``D10``), by default True
    device : Any, optional
        Device to run the forward pass on, by default ``"cpu"``
    time : np.datetime64, optional
        Timestamp to probe the model at, by default 2024-01-01T00:00. Models valid
        only over a restricted period need a time inside it.

    Returns
    -------
    list[str]
        Rules that could not be evaluated, each with the reason it was skipped

    Raises
    ------
    ContractException
        If the model violates any evaluated rule
    """
    report = _evaluate_diagnostic(model, forward=forward, device=device, time=time)
    report.raise_for_violations()
    return report.skipped


def _evaluate_diagnostic(
    model: DiagnosticModel,
    forward: bool = True,
    device: Any = "cpu",
    time: np.datetime64 = _PROBE_TIME,
) -> _Report:
    """Run every diagnostic rule and return the report without raising.

    Split from :func:`check_diagnostic_contract` so a test can inspect
    ``report.evaluated`` (e.g. to confirm every documented rule is reachable)
    without needing a model that fails nothing.
    """
    report = _Report(model)
    report.require(
        "D1",
        isinstance(model, DiagnosticModel),
        "model does not structurally satisfy the DiagnosticModel protocol",
    )
    declared = report.probe("D2", lambda: _slots(model.input_coords()))
    if declared is None:
        return report
    _signatures(report, model, {"__call__": len(declared)}, "D11")
    report.probe("D1", lambda: _sources(report, model, len(declared), "D1"))
    planned = report.probe(
        "D2",
        lambda: _check_coord_declaration(
            report,
            model,
            declared,
            coords_rule="D2",
            readonly_rule="D3",
            invalid_rule="D4",
            time=time,
        ),
    )
    stochastic = _check_stochasticity(report, model, "D7", "D8", "D10", device)
    if not forward or planned is None:
        report.skip(
            ("D5", "D6", "D9"),
            "forward check disabled" if not forward else "coordinate planning failed",
        )
        if stochastic:
            report.skip(
                "D10",
                (
                    "forward check disabled"
                    if not forward
                    else "coordinate planning failed"
                ),
            )
        return report
    inputs, outputs = planned
    args = tuple(_sample_tensor(x, device) for x in inputs)

    def run() -> Any:
        return deepcopy(model(*deepcopy(args)))

    def check() -> None:
        _seed(model)
        y = _call_readonly(report, model, deepcopy(args), "D6")
        _matches(report, y, outputs, "D5")

    report.probe("D5", check)
    report.probe(
        "D9", lambda: _check_reproducibility(report, model, run, "D9", stochastic)
    )
    if stochastic:
        _seed(model)
        report.probe(
            "D10",
            lambda: _check_rng_isolation(
                report, run, "calling a seeded model", "D10", device
            ),
        )
    return report


def iter_contract_rules() -> Iterator[tuple[str, str]]:
    """Yield the rule identifiers and summaries documented by the contract spec.

    Yields
    ------
    Iterator[tuple[str, str]]
        Pairs of rule identifier and one-line summary
    """
    yield from _RULES.items()
