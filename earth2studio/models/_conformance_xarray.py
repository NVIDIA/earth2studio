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

"""Native labelled-array probes for the public model conformance checker."""

from collections.abc import Callable
from contextlib import closing
from itertools import islice
from typing import Any

import numpy as np
import torch
import xarray as xr

from earth2studio.models.conformance import (
    _check_rng_isolation,
    _check_stochasticity,
    _Report,
)
from earth2studio.models.dx.base import DiagnosticModel
from earth2studio.models.px.base import PrognosticModel
from earth2studio.utils import coord_array_like
from earth2studio.utils.coords import E2S_DYNAMIC_DIMS, E2S_KIND, E2S_SCHEMA_VERSION
from earth2studio.utils.cupy import from_torch


def _coordinates_equal(first: xr.DataArray, second: xr.DataArray) -> bool:
    """Compare ordered dimensions and all coordinate values and metadata."""
    return (
        first.dims == second.dims
        and first.sizes == second.sizes
        and first.coords.to_dataset().identical(second.coords.to_dataset())
    )


def _identical(first: xr.DataArray, second: xr.DataArray) -> bool:
    """Compare field values, labels, and metadata on CPU or CUDA."""
    return (
        first.e2s.as_numpy().identical(second.e2s.as_numpy())
        and first.encoding == second.encoding
    )


def _field_matches(field: xr.DataArray, signature: xr.DataArray) -> bool:
    if not isinstance(field, xr.DataArray) or not _coordinates_equal(field, signature):
        return False
    if any(
        key in field.attrs for key in (E2S_KIND, E2S_SCHEMA_VERSION, E2S_DYNAMIC_DIMS)
    ):
        return False
    # User attrs can change through hooks. Grid representation, CRS and statistics
    # describe the field itself and must agree with its independent declaration.
    keys = {
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
    }
    return all(field.attrs.get(key) == signature.attrs.get(key) for key in keys)


def _declaration(
    report: _Report, model: Any, prognostic: bool, time: np.datetime64
) -> tuple[xr.DataArray, xr.DataArray | None]:
    """Validate signatures and construct a concretely sized probe signature."""
    rule, readonly, invalid = ("P2", "P4", "P5") if prognostic else ("D2", "D3", "D4")
    signature = model.input_coords()
    dynamic = tuple(signature.attrs.get("earth2studio_dynamic_dims", ()))
    report.require(
        rule,
        signature.attrs.get(E2S_KIND) == "coordinate_array"
        and signature.data.nbytes == 0,
        "input_coords() must return an allocation-free coordinate signature",
    )
    report.require(
        rule,
        tuple(signature.dims[: len(dynamic)]) == dynamic
        and all(signature.sizes[dim] == 0 for dim in dynamic),
        "dynamic dimensions must be an empty leading prefix",
    )
    replacements = {}
    for dim in dynamic:
        coordinate = signature.coords.get(dim)
        dtype = coordinate.dtype if coordinate is not None else None
        if dtype is not None and dtype.kind == "M":
            replacements[dim] = np.array([time], dtype=dtype)
        elif dtype is not None and dtype.kind == "m":
            replacements[dim] = np.array([0], dtype=dtype)
        elif dim == "time":
            replacements[dim] = np.array([time])
        elif dim == "lead_time":
            replacements[dim] = np.array([0], dtype="timedelta64[ns]")
        else:
            replacements[dim] = np.array([0])
    concrete = coord_array_like(signature, replacements)
    original = concrete.copy(deep=True)
    try:
        output = model.output_coords(concrete)
    except Exception as error:
        report.require(
            readonly, False, f"output_coords() rejected its own signature: {error!r}"
        )
        return concrete, None
    report.require(
        readonly,
        _coordinates_equal(concrete, original) and concrete.attrs == original.attrs,
        "output_coords() mutated its input coordinates or metadata",
    )
    if not report.require(
        rule,
        isinstance(output, xr.DataArray),
        "output_coords() must return a DataArray",
    ):
        return concrete, None
    report.require(
        rule,
        output.attrs.get(E2S_KIND) == "coordinate_array" and output.data.nbytes == 0,
        "output_coords() must return an allocation-free coordinate signature",
    )
    if signature.ndim >= 3 and signature.ndim - len(dynamic) >= 2:
        dims = list(concrete.dims)
        dims[-2:] = dims[-2:][::-1]
        try:
            model.output_coords(concrete.transpose(*dims))
        except ValueError:
            report.require(invalid, True, "invalid dimension order rejected")
        except Exception as error:
            report.require(
                invalid,
                False,
                f"invalid dimensions raised {type(error).__name__}, expected ValueError",
            )
        else:
            report.require(
                invalid, False, "output_coords() accepted swapped fixed dimensions"
            )
    else:
        report.skip(invalid, "model declares fewer than three dimensions")
    return concrete, output


def _lead_times(
    report: _Report, model: Any, coords: xr.DataArray, output: xr.DataArray
) -> None:
    """Check relative input history and translation of forecast lead times."""
    lead = coords.coords.get("lead_time")
    valid = (
        lead is not None
        and lead.dims == ("lead_time",)
        and lead.size > 0
        and np.issubdtype(lead.dtype, np.timedelta64)
    )
    if not report.require(
        "P3",
        valid,
        "input lead_time must be a nonempty one-dimensional timedelta coordinate",
    ):
        return
    values = np.asarray(lead)
    report.require(
        "P3",
        not np.isnat(values).any()
        and values[-1] == np.timedelta64(0, "h")
        and bool(np.all(np.diff(values) > np.timedelta64(0, "h"))),
        "input lead_time must be finite, strictly increasing, and end at zero",
    )
    offset = np.timedelta64(24, "h")
    try:
        shifted = model.output_coords(coords.assign_coords(lead_time=values + offset))
        report.require(
            "P6",
            np.array_equal(shifted.lead_time, output.lead_time + offset),
            "output lead_time did not shift with input lead_time",
        )
    except Exception as error:
        report.require("P6", False, f"shifted lead_time rejected: {error!r}")


def _hooks(report: _Report, model: Any, x: xr.DataArray) -> None:
    """Check exact iterator hook ordering and single-step hook exclusion."""
    if not all(hasattr(model, name) for name in ("front_hook", "rear_hook")):
        report.skip("P10", "model does not expose prognostic hooks")
        return
    interval = getattr(model, "front_hook_interval", 1)
    if not report.require(
        "P10",
        type(interval) is int and interval > 0,
        "front_hook_interval must be a positive integer",
    ):
        return
    calls: list[str] = []

    def front(field: xr.DataArray) -> xr.DataArray:
        calls.append("front")
        return field

    def rear(field: xr.DataArray) -> xr.DataArray:
        calls.append("rear")
        return field

    original = model.front_hook, model.rear_hook
    model.front_hook, model.rear_hook = front, rear
    try:
        with closing(model.create_iterator(x.copy(deep=True))) as iterator:
            list(islice(iterator, 2 * interval + 1))
        report.require(
            "P10",
            calls == (["front"] + ["rear"] * interval) * 2,
            f"iterator hook order was {calls}",
        )
        calls.clear()
        model(x.copy(deep=True))
        report.require("P10", not calls, "single-step call invoked iterator hooks")
    finally:
        model.front_hook, model.rear_hook = original


def _reproducibility(
    report: _Report,
    model: Any,
    run: Callable[[], list[xr.DataArray]],
    prognostic: bool,
    stochastic: bool,
    device: Any,
) -> None:
    """Compare independent seeded runs and probe seeded execution RNG isolation."""
    rule = "P13" if prognostic else "D9"
    seed = getattr(model, "set_rng", None)
    if stochastic and not callable(seed):
        report.skip(rule, "stochastic model does not implement set_rng")
        return

    def same(first: list[xr.DataArray], second: list[xr.DataArray]) -> bool:
        return len(first) == len(second) and all(
            _identical(a, b) for a, b in zip(first, second)
        )

    if stochastic and callable(seed):
        seed(0)
    first = run()
    if stochastic and callable(seed):
        seed(0)
    report.require(
        rule, same(first, run()), "repeated runs with the same input and seed disagree"
    )
    if stochastic and callable(seed):
        seed(1)
        report.require(
            rule, not same(first, run()), "different seeds produced identical results"
        )
        seed(0)
        _check_rng_isolation(
            report,
            run,
            "executing a seeded model",
            "P14" if prognostic else "D10",
            device,
        )


def evaluate(
    model: Any,
    prognostic: bool,
    forward: bool,
    nsteps: int,
    device: Any,
    time: np.datetime64,
) -> _Report:
    """Evaluate the DataArray execution contract without a tensor-pair adapter."""
    report = _Report(model)
    report.require(
        "P1" if prognostic else "D1",
        isinstance(model, PrognosticModel if prognostic else DiagnosticModel),
        "model does not satisfy its public protocol",
    )
    coords, output = _declaration(report, model, prognostic, time)
    if prognostic and output is not None:
        _lead_times(report, model, coords, output)
    stochastic = _check_stochasticity(
        report,
        model,
        *(("P11", "P12", "P14") if prognostic else ("D7", "D8", "D10")),
        device=device,
    )
    rules = (
        ("P7", "P8", "P9", "P10", "P13", "P15", "P16")
        if prognostic
        else ("D5", "D6", "D9")
    )
    if not forward or output is None:
        report.skip(
            rules, "rollout checks disabled" if prognostic else "forward check disabled"
        )
        return report
    tensor = torch.randn(coords.shape, generator=torch.Generator().manual_seed(0)).to(
        device
    )
    x = from_torch(tensor, coords)
    pristine = x.copy(deep=True)
    seed = getattr(model, "set_rng", None)
    if stochastic and callable(seed):
        seed(0)
    result = model(x)
    report.require(
        "P15" if prognostic else "D6",
        _identical(x, pristine),
        "__call__ mutated input values, coordinates, or metadata",
    )
    report.require(
        "P9" if prognostic else "D5",
        _field_matches(result, output),
        "__call__ returned coordinates or structural metadata differing from its declaration",
    )
    if prognostic:
        x = pristine.copy(deep=True)
        retained: list[xr.DataArray] = []
        snapshots: list[xr.DataArray] = []
        expected = output
        with closing(model.create_iterator(x)) as iterator:
            for index, step in enumerate(islice(iterator, nsteps + 1)):
                report.require(
                    "P9",
                    isinstance(step, xr.DataArray),
                    "iterator must yield DataArrays",
                )
                if not isinstance(step, xr.DataArray):
                    break
                if index:
                    report.require(
                        "P9",
                        _field_matches(step, expected),
                        f"forecast {index} differs from its planned coordinates or structural metadata",
                    )
                    # Reconstruct the original declared history at the previous
                    # planned endpoint, never from potentially corrupted yields.
                    # This also supports models whose output variables/grid differ
                    # from their input and cores emitting multiple leads per yield.
                    if index < nsteps:
                        history = coord_array_like(
                            coords,
                            {
                                "lead_time": coords.lead_time.values
                                - coords.lead_time.values[-1]
                                + expected.lead_time.values[-1]
                            },
                        )
                        expected = model.output_coords(history)
                retained.append(step)
                snapshots.append(step.copy(deep=True))
        report.require("P15", _identical(x, pristine), "iterator mutated its input")
        report.require(
            "P7",
            bool(snapshots)
            and _identical(snapshots[0], pristine.isel(lead_time=slice(-1, None))),
            "initial yield must contain the final input history entry",
        )
        report.require(
            "P8",
            len(snapshots) > 1 and _field_matches(snapshots[1], output),
            "first forecast differs from output_coords()",
        )
        report.require(
            "P16",
            all(_identical(a, b) for a, b in zip(retained, snapshots)),
            "a retained yield changed after advancing the iterator",
        )
        _hooks(report, model, pristine)

    def run() -> list[xr.DataArray]:
        if not prognostic:
            return [model(pristine.copy(deep=True)).copy(deep=True)]
        with closing(model.create_iterator(pristine.copy(deep=True))) as iterator:
            return [step.copy(deep=True) for step in islice(iterator, 1, nsteps + 1)]

    _reproducibility(report, model, run, prognostic, stochastic, device)
    return report
