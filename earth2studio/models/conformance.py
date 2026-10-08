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

"""Conformance probes for the DataArray model protocols (P1–P24 and D1–D11).

Prognostic probes use forecasts-only iteration and explicit continuation state.
Execution takes separate positional slots; coordinate planning takes one grouped
signature. Probes never call recommended sources. Model-specific numerical
correctness, semantic source ordering, and absence of internal fetching still
require wrapper tests and review.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from copy import deepcopy
from dataclasses import fields, is_dataclass
from inspect import Parameter, signature
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.data.base import DataSource, ForecastSource
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.models.px.base import PrognosticModel
from earth2studio.utils import coord_array_like, handshake_metadata
from earth2studio.utils.coords import E2S_DYNAMIC_DIMS, E2S_KIND, E2S_SCHEMA_VERSION
from earth2studio.utils.cupy import from_torch

_PROBE_TIME = np.datetime64("2024-01-01T00:00:00")
_RULES = {
    "P1": "Implements the prognostic interface.",
    "P2": "Declarations are allocation-free DataArray signatures.",
    "P3": "Input history is finite, increasing, relative and ends at zero.",
    "P4": "Coordinate planning does not mutate its inputs.",
    "P5": "Invalid coordinates raise ValueError.",
    "P6": "Output leads shift with the input leads.",
    "P7": "Iteration yields forecasts only, starting with initialization output.",
    "P8": "The first yield matches planned output coordinates.",
    "P9": "Every output matches planned coordinates and structural metadata.",
    "P10": "Hooks are iterator-only and both affect recurrence.",
    "P11": "Declares a boolean stochastic attribute.",
    "P12": "Stochastic models implement set_rng(seed, reset=True).",
    "P13": "Seeding determines rollouts and different seeds differ.",
    "P14": "Seeded execution and seeding preserve global RNG state.",
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
    "D1": "Implements the diagnostic interface including default_sources.",
    "D2": "Declarations are allocation-free DataArray signatures.",
    "D3": "Coordinate planning does not mutate its inputs.",
    "D4": "Invalid coordinates raise ValueError.",
    "D5": "Outputs match planned coordinates and structural metadata.",
    "D6": "Execution does not mutate borrowed inputs or coordinates.",
    "D7": "Declares a boolean stochastic attribute.",
    "D8": "Stochastic models implement set_rng(seed, reset=True).",
    "D9": "Seeding determines output and different seeds differ.",
    "D10": "Seeded execution and seeding preserve global RNG state.",
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


class ContractException(Exception):
    """Report model contract violations with their rule identifiers.

    Parameters
    ----------
    model : Any
        Model being checked.
    violations : list[str]
        Rule-prefixed failure descriptions.
    """

    def __init__(self, model: Any, violations: list[str]) -> None:
        self.violations = violations
        body = "\n".join(f"  - {v}" for v in violations)
        super().__init__(f"{type(model).__name__} violates the model contract:\n{body}")


class _Report:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.violations: list[str] = []
        self.skipped: list[str] = []
        self.evaluated: set[str] = set()

    def require(self, rule: str, condition: bool, message: str) -> bool:
        """Record a rule outcome and return its truth value."""
        if rule not in _RULES:
            raise ValueError(f"Unknown rule {rule}")
        self.evaluated.add(rule)
        if not condition:
            self.violations.append(f"{rule}: {message}")
        return condition

    def skip(self, rules: str | tuple[str, ...], reason: str) -> None:
        """Record unevaluated rules with an explicit reason."""
        for rule in (rules,) if isinstance(rules, str) else rules:
            self.require(rule, True, "")
            self.skipped.append(f"{rule}: {reason}")

    def probe(self, rule: str, action: Callable[[], Any]) -> Any:
        """Collect execution errors without aborting independent probes."""
        try:
            return action()
        except Exception as error:  # noqa: BLE001 - model failures become violations
            self.require(rule, False, f"probe raised {error!r}")
            return None

    def raise_for_violations(self) -> None:
        """Raise the collected failures."""
        if self.violations:
            raise ContractException(self.model, self.violations)


def _slots(value: Any) -> tuple[xr.DataArray, ...]:
    slots = value if isinstance(value, tuple) else (value,)
    if not slots or not all(isinstance(x, xr.DataArray) for x in slots):
        raise ValueError("expected a DataArray or a nonempty tuple of DataArrays")
    return slots


def _group(slots: tuple[xr.DataArray, ...]) -> Any:
    return slots[0] if len(slots) == 1 else slots


def _same(first: Any, second: Any) -> bool:
    if type(first) is not type(second):
        return False
    if isinstance(first, xr.DataArray):
        return first.e2s.as_numpy().identical(second.e2s.as_numpy()) and _same(
            first.encoding, second.encoding
        )
    if isinstance(first, torch.Tensor):
        return torch.equal(first, second)
    if isinstance(first, np.ndarray):
        return np.array_equal(first, second)
    if isinstance(first, (tuple, list)):
        return len(first) == len(second) and all(
            _same(a, b) for a, b in zip(first, second)
        )
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(
            _same(v, second[k]) for k, v in first.items()
        )
    if is_dataclass(first):
        return all(
            _same(getattr(first, f.name), getattr(second, f.name))
            for f in fields(first)
        )
    return bool(first == second)


def _same_coords(first: xr.DataArray, second: xr.DataArray) -> bool:
    return (
        first.dims == second.dims
        and first.sizes == second.sizes
        and first.coords.to_dataset().identical(second.coords.to_dataset())
        and _same(first.attrs, second.attrs)
        and _same(first.encoding, second.encoding)
    )


def _concretize(coords: xr.DataArray, time: np.datetime64) -> xr.DataArray:
    replacements = {}
    for dim in coords.attrs.get(E2S_DYNAMIC_DIMS, ()):
        coordinate = coords.coords.get(dim)
        dtype = coordinate.dtype if coordinate is not None else None
        if dtype is not None and dtype.kind == "M":
            replacements[dim] = np.full(1, time, dtype=dtype)
        elif dtype is not None and dtype.kind == "m":
            replacements[dim] = np.zeros(1, dtype=dtype)
        elif dim == "time":
            replacements[dim] = np.array([time])
        elif dim == "lead_time":
            replacements[dim] = np.zeros(1, dtype="timedelta64[ns]")
        else:
            replacements[dim] = np.arange(1)
    return coord_array_like(coords, replacements)


def _sample(coords: xr.DataArray, device: Any) -> xr.DataArray:
    tensor = torch.randn(coords.shape, generator=torch.Generator().manual_seed(0))
    return from_torch(tensor.to(device), coords)


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


def _plan(
    report: _Report,
    model: Any,
    declared: tuple[xr.DataArray, ...],
    prefix: str,
    time: np.datetime64,
) -> tuple[tuple[xr.DataArray, ...], tuple[xr.DataArray, ...]] | None:
    coord_rule, readonly, invalid = (
        ("P2", "P4", "P5") if prefix == "P" else ("D2", "D3", "D4")
    )
    _declaration(report, declared, coord_rule)
    concrete = tuple(_concretize(x, time) for x in declared)
    pristine = deepcopy(concrete)
    output = _slots(model.output_coords(_group(concrete)))
    report.require(
        readonly,
        all(_same_coords(a, b) for a, b in zip(concrete, pristine)),
        "output_coords mutated its inputs",
    )
    _declaration(report, output, coord_rule, dynamic=False)
    tested = False
    for index, x in enumerate(concrete):
        if x.ndim - len(declared[index].attrs.get(E2S_DYNAMIC_DIMS, ())) < 2:
            continue
        tested = True
        dims = list(x.dims)
        dims[-2], dims[-1] = dims[-1], dims[-2]
        bad = list(deepcopy(pristine))
        bad[index] = bad[index].transpose(*dims)
        _reject(report, invalid, lambda: model.output_coords(_group(tuple(bad))))
    if not tested:
        report.skip(invalid, "model declares fewer than two fixed dimensions")
    return pristine, output


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
                all(p.kind != Parameter.VAR_POSITIONAL for p in parameters),
                f"{name} must declare fixed named array parameters",
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
        all(s is None or isinstance(s, (DataSource, ForecastSource)) for s in slots),
        "default_sources entries must be raw data sources or None",
    )


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
        match = (
            x.dims == coords.dims
            and x.sizes == coords.sizes
            and x.coords.to_dataset().identical(coords.coords.to_dataset())
        )
        match = match and not any(
            k in x.attrs for k in (E2S_KIND, E2S_SCHEMA_VERSION, E2S_DYNAMIC_DIMS)
        )
        try:
            handshake_metadata(
                x,
                coords,
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
            match = False
        report.require(
            rule,
            match,
            "output differs from planned dimensions, coordinates or structural metadata",
        )


def _seed(model: Any, seed: int = 0) -> None:
    if getattr(model, "stochastic", False) and callable(
        getattr(model, "set_rng", None)
    ):
        model.set_rng(seed)


def _isolation(
    report: _Report, action: Callable[[], Any], rule: str, device: Any
) -> None:
    resolved = torch.device(device)
    devices = (
        [resolved.index if resolved.index is not None else torch.cuda.current_device()]
        if resolved.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        before = [
            torch.get_rng_state(),
            *(torch.cuda.get_rng_state(d) for d in devices),
        ]
        action()
        after = [torch.get_rng_state(), *(torch.cuda.get_rng_state(d) for d in devices)]
    report.require(
        rule,
        all(torch.equal(a, b) for a, b in zip(before, after)),
        "seeded execution perturbed global Torch RNG state",
    )


def _stochasticity(report: _Report, model: Any, prefix: str, device: Any) -> bool:
    declaration, seeding, isolation = (
        ("P11", "P12", "P14") if prefix == "P" else ("D7", "D8", "D10")
    )
    stochastic = getattr(model, "stochastic", None)
    report.require(
        declaration,
        isinstance(stochastic, bool),
        "stochastic must be explicitly declared as a bool",
    )
    if not stochastic:
        report.require(seeding, True, "")
        report.skip(isolation, "model does not declare itself stochastic")
        return False
    method = getattr(model, "set_rng", None)
    if not report.require(seeding, callable(method), "stochastic models need set_rng"):
        report.skip(isolation, "stochastic model does not implement set_rng")
        return False

    def check() -> None:
        params = signature(method).parameters
        report.require(
            seeding,
            list(params) == ["seed", "reset"] and params["reset"].default is True,
            "expected set_rng(seed, reset=True)",
        )
        _isolation(report, lambda: method(0), isolation, device)

    report.probe(seeding, check)
    return True


def _reproducibility(
    report: _Report, model: Any, run: Callable[[], Any], rule: str, stochastic: bool
) -> None:
    _seed(model)
    first = run()
    _seed(model)
    report.require(
        rule,
        _same(first, run()),
        "identical seeds/inputs must reproduce output exactly",
    )
    if stochastic:
        _seed(model)
        model.set_rng(1, reset=False)
        report.require(
            "P12" if rule == "P13" else "D8",
            _same(first, run()),
            "reset=False must preserve an existing seed",
        )
        _seed(model, 1)
        report.require(
            rule, not _same(first, run()), "different seeds produced identical output"
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
    return _slots(model.output_coords(_group(shifted)))


def _forcing(
    forcing: tuple[xr.DataArray, ...], outputs: tuple[xr.DataArray, ...], device: Any
) -> tuple[xr.DataArray, ...]:
    leads = np.unique(np.concatenate([x.lead_time.values for x in outputs]))
    return tuple(
        _sample(
            coord_array_like(f, {"lead_time": f.lead_time.values[-1] + leads}), device
        )
        for f in forcing
        if "lead_time" in f.dims
    )


def _iterate(
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
                        _same(new_forcing, saved_forcing),
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
            "P15", _same(borrowed, pristine), "iterator mutated initialization inputs"
        )
        report.require(
            "P16",
            _same(retained, snapshots),
            "advancing the iterator changed an earlier yield",
        )
        _matches(report, snapshots[0], outputs, "P8")
    return snapshots


def _call_readonly(
    report: _Report,
    function: Callable[..., Any],
    args: tuple[Any, ...],
    rule: str,
    **kwargs: Any,
) -> Any:
    saved = deepcopy((args, kwargs))
    result = function(*args, **kwargs)
    report.require(
        rule, _same((args, kwargs), saved), "execution mutated borrowed arrays or state"
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
    first = _call_readonly(
        report, model.step, (*_slots(y), *new_forcing), "P19", state=state
    )
    _matches(report, first[0], expected, "P9")
    # Reseeding the model must not affect a rollout whose RNG position is in state.
    _seed(model, 17)
    second = model.step(*_slots(pristine[0]), *deepcopy(new_forcing), state=pristine[1])
    report.require(
        "P19", _same(first, second), "step does not replay exactly from (y, state)"
    )

    def checkpoint() -> None:
        # Only deserialize the trusted pair just produced by this local model.
        restored = pickle.loads(pickle.dumps(pristine))  # noqa: S301
        replay = model.step(
            *_slots(restored[0]), *deepcopy(new_forcing), state=restored[1]
        )
        report.require(
            "P21",
            _same(first, replay),
            "serialized continuation does not reproduce the next step",
        )

    report.probe("P21", checkpoint)


def _hooks(
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
        actual = _iterate(model, args, inputs, outputs, forcing, 2, device)
        report.require(
            "P10", events == ["rear", "front", "rear"], f"wrong hook order: {events}"
        )
        report.require(
            "P10",
            _same(actual, [first, second]),
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


def _rollout(
    report: _Report,
    model: Any,
    inputs: tuple[xr.DataArray, ...],
    outputs: tuple[xr.DataArray, ...],
    forcing: tuple[xr.DataArray, ...],
    nsteps: int,
    device: Any,
    stochastic: bool,
) -> None:
    args = tuple(_sample(x, device) for x in (*inputs, *forcing))
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
            lambda: _iterate(
                model, args, inputs, outputs, forcing, max(2, nsteps), device, report
            ),
        )
        if pair is not None:
            report.probe("P9", lambda: _matches(report, pair[0], outputs, "P9"))
            report.require(
                "P20",
                _same(called, pair[0]),
                "call differs from initialization forecast",
            )
            if snapshots is not None:
                report.require(
                    "P7",
                    _same(snapshots[0], pair[0]),
                    "first yield must be initialization forecast, not initial conditions",
                )
                report.require(
                    "P20",
                    _same(called, snapshots[0]),
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
            return _iterate(
                model, args, inputs, outputs, forcing, max(2, nsteps), device
            )

        report.probe(
            "P13", lambda: _reproducibility(report, model, run, "P13", stochastic)
        )
        if stochastic:
            _seed(model)
            report.probe("P14", lambda: _isolation(report, run, "P14", device))
        report.probe(
            "P10", lambda: _hooks(report, model, args, inputs, outputs, forcing, device)
        )
        report.probe(
            "P22", lambda: _forcing_errors(report, model, args, forcing, device)
        )


def _evaluate_prognostic(
    model: PrognosticModel,
    rollout: bool = True,
    nsteps: int = 2,
    device: Any = "cpu",
    time: np.datetime64 = _PROBE_TIME,
) -> _Report:
    if type(nsteps) is not int or nsteps < 1:
        raise ValueError("nsteps must be a positive integer")
    report = _Report(model)
    report.require(
        "P1", isinstance(model, PrognosticModel), "missing prognostic protocol members"
    )
    declared = report.probe("P2", lambda: _slots(model.input_coords()))
    if declared is None:
        return report
    for x in declared:
        lead = x.coords.get("lead_time")
        valid = (
            lead is not None
            and lead.ndim == 1
            and lead.dtype.kind == "m"
            and lead.size > 0
        )
        if valid:
            values = lead.values
            valid = (
                not np.isnat(values).any()
                and values[-1] == np.timedelta64(0)
                and bool(np.all(np.diff(values) > np.timedelta64(0)))
            )
        report.require(
            "P3",
            bool(valid),
            "input lead_time must be finite, increasing and end at zero",
        )
    planned = report.probe("P2", lambda: _plan(report, model, declared, "P", time))
    raw_forcing = report.probe("P2", lambda: model.forcing_coords())
    forcing = report.probe(
        "P2", lambda: () if raw_forcing is None else _slots(raw_forcing)
    )
    if planned is None or forcing is None:
        return report
    inputs, outputs = planned
    _declaration(report, forcing, "P2")
    forcing = tuple(_concretize(f, time) for f in forcing)
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

    def rebase() -> None:
        offset = np.timedelta64(24, "h")
        shifted = tuple(
            coord_array_like(x, {"lead_time": x.lead_time.values + offset})
            for x in inputs
        )
        actual = _slots(model.output_coords(_group(shifted)))
        report.require(
            "P6",
            len(actual) == len(outputs)
            and all(
                np.array_equal(a.lead_time.values, b.lead_time.values + offset)
                for a, b in zip(actual, outputs)
            ),
            "output lead times did not rebase with inputs",
        )

    report.probe("P6", rebase)
    stochastic = _stochasticity(report, model, "P", device)
    if not rollout:
        report.skip(_ROLLOUT_RULES, "rollout checks disabled")
        if stochastic:
            report.skip("P14", "rollout checks disabled")
    else:
        report.probe(
            "P9",
            lambda: _rollout(
                report, model, inputs, outputs, forcing, nsteps, device, stochastic
            ),
        )
    return report


def check_prognostic_contract(
    model: PrognosticModel,
    rollout: bool = True,
    nsteps: int = 2,
    device: Any = "cpu",
    time: np.datetime64 = _PROBE_TIME,
) -> list[str]:
    """Check a DataArray prognostic against the explicit-state model contract.

    Parameters
    ----------
    model : PrognosticModel
        Model with fixed execution signatures and single or multiple slots.
    rollout : bool, optional
        Run execution, replay, forcing and hook probes, by default True.
        False retains declaration, signature, source and seeding checks.
    nsteps : int, optional
        Forecast yields to check, by default 2. At least two are probed to check
        continuation and ownership; initial conditions are never counted.
    device : Any, optional
        Probe array device, by default "cpu". Move the model there beforehand.
    time : np.datetime64, optional
        Concrete timestamp for open temporal dimensions, by default 2024-01-01.

    Returns
    -------
    list[str]
        Unevaluated rules with explicit reasons.

    Raises
    ------
    ContractException
        If any evaluated rule fails. Independent probes continue after failures.
    ValueError
        If nsteps is not a positive integer.
    """
    report = _evaluate_prognostic(model, rollout, nsteps, device, time)
    report.raise_for_violations()
    return report.skipped


def _evaluate_diagnostic(
    model: DiagnosticModel,
    forward: bool = True,
    device: Any = "cpu",
    time: np.datetime64 = _PROBE_TIME,
) -> _Report:
    report = _Report(model)
    report.require(
        "D1", isinstance(model, DiagnosticModel), "missing diagnostic protocol members"
    )
    declared = report.probe("D2", lambda: _slots(model.input_coords()))
    if declared is None:
        return report
    _signatures(report, model, {"__call__": len(declared)}, "D11")
    report.probe("D1", lambda: _sources(report, model, len(declared), "D1"))
    planned = report.probe("D2", lambda: _plan(report, model, declared, "D", time))
    stochastic = _stochasticity(report, model, "D", device)
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
    args = tuple(_sample(x, device) for x in inputs)

    def run() -> Any:
        return deepcopy(model(*deepcopy(args)))

    def check() -> None:
        _seed(model)
        y = _call_readonly(report, model, deepcopy(args), "D6")
        _matches(report, y, outputs, "D5")

    report.probe("D5", check)
    report.probe("D9", lambda: _reproducibility(report, model, run, "D9", stochastic))
    if stochastic:
        _seed(model)
        report.probe("D10", lambda: _isolation(report, run, "D10", device))
    return report


def check_diagnostic_contract(
    model: DiagnosticModel,
    forward: bool = True,
    device: Any = "cpu",
    time: np.datetime64 = _PROBE_TIME,
) -> list[str]:
    """Check a single- or multi-slot DataArray diagnostic model.

    Parameters
    ----------
    model : DiagnosticModel
        Model with fixed named input parameters.
    forward : bool, optional
        Run execution checks in addition to planning and seeding, by default True.
    device : Any, optional
        Probe device, by default "cpu". Move the model there beforehand.
    time : np.datetime64, optional
        Concrete timestamp for dynamic temporal dimensions, by default 2024-01-01.

    Returns
    -------
    list[str]
        Unevaluated rules with explicit reasons.

    Raises
    ------
    ContractException
        If any evaluated rule fails.
    """
    report = _evaluate_diagnostic(model, forward, device, time)
    report.raise_for_violations()
    return report.skipped


def iter_contract_rules() -> Iterator[tuple[str, str]]:
    """Yield the supported rule identifiers and descriptions.

    Yields
    ------
    tuple[str, str]
        Rule identifier and its one-line summary.
    """
    yield from _RULES.items()
