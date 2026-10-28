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

"""Model declarations parsed once for the coupled scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from earth2studio.coupling.errors import CouplingError
from earth2studio.run._fields import _publishes_initial_condition
from earth2studio.utils.type import CoordinateSystem, FrameSchema


def _ns(values: Any) -> np.ndarray:
    """Lead times as int64 nanoseconds."""
    return np.atleast_1d(np.asarray(values).astype("timedelta64[ns]").astype(np.int64))


def _slots(signature: Any) -> tuple:
    if signature is None:
        return ()
    return signature if isinstance(signature, tuple) else (signature,)


@dataclass(frozen=True)
class _Slot:
    """One positional argument of a model: an input or forcing declaration."""

    index: int
    signature: CoordinateSystem | FrameSchema
    forcing: bool
    per_call: bool = False  # a diagnostic input, read at each call's valid time

    @property
    def frame(self) -> bool:
        """Whether the slot takes a DataFrame."""
        return not isinstance(self.signature, xr.DataArray)

    @property
    def static(self) -> bool:
        """Whether the slot is gridded without lead times, supplied only initially."""
        return (
            not self.frame
            and not self.per_call
            and "lead_time" not in self.signature.dims
        )

    @property
    def labels(self) -> tuple[str, ...]:
        """Variable labels the slot declares."""
        values = (
            self.signature["variable"]
            if self.frame
            else self.signature["variable"].values
        )
        return tuple(str(v) for v in np.asarray(values))

    @property
    def leads(self) -> np.ndarray:
        """Declared lead times in nanoseconds."""
        return _ns(self.signature["lead_time"].values)

    def __str__(self) -> str:
        return f"{'forcing' if self.forcing else 'input'}[{self.index}]"


@dataclass
class _Node:
    """A named model's declarations, read once at construction.

    Stateful nodes (prognostic and stateful assimilation models) call
    ``initialize`` then ``step``: chunk ``k`` covers the output lead times
    ``out_leads + k * cadence`` and is due at its last lead time, with chunk 0
    from ``initialize``. Diagnostic nodes are called once per valid time at which
    their inputs arrive and leave the timing fields empty.
    """

    name: str
    model: Any
    stateful: bool
    inputs: tuple[_Slot, ...]
    forcing: tuple[_Slot, ...]
    outputs: tuple[CoordinateSystem, ...]
    imports: tuple[_Slot, ...]  # slots a connector may feed
    out_leads: np.ndarray
    ic_lead: int = 0
    cadence: int = 0
    spacing: int = 0
    ic_slots: tuple[int, ...] = ()  # output slots publishing an initial condition

    def exports(self, slot: int) -> tuple[str, ...]:
        """Variables output slot ``slot`` publishes."""
        return tuple(str(v) for v in self.outputs[slot]["variable"].values)

    @property
    def variables(self) -> tuple[str, ...]:
        """Variables published by any output slot."""
        return tuple(v for i in range(len(self.outputs)) for v in self.exports(i))

    @property
    def streams(self) -> tuple[str, ...]:
        """Stream name per output slot: the component's name, or ``name.i``."""
        if len(self.outputs) == 1:
            return (self.name,)
        return tuple(f"{self.name}.{i}" for i in range(len(self.outputs)))

    def chunk(self, k: int) -> np.ndarray:
        """Lead times of output chunk ``k``."""
        return self.out_leads + k * self.cadence

    def due(self, k: int) -> int:
        """Lead time at which chunk ``k`` is complete."""
        return int(self.chunk(k)[-1])

    def slots_for(self, k: int) -> tuple[_Slot, ...]:
        """Forcing slots passed to the call producing chunk ``k``."""
        return (
            self.forcing if k == 0 else tuple(s for s in self.forcing if not s.static)
        )

    def required(self, slot: _Slot, index: int) -> np.ndarray:
        """Lead times ``slot`` needs for a call.

        ``index`` is the chunk number for a stateful node and the valid time for
        a diagnostic, which reads every input at that time. ``initialize`` takes
        the whole forcing window; ``step`` takes the window's final lead time
        shifted by each lead time of the output it advances. DataFrame slots are
        requested at the analysis time, the chunk's end.
        """
        if not self.stateful:
            return np.array([index], dtype=np.int64)
        if slot.frame:
            return np.array([self.due(index)], dtype=np.int64)
        if slot.static:
            return np.zeros(1, dtype=np.int64)
        return slot.leads if index == 0 else slot.leads[-1] + self.chunk(index - 1)

    def state_leads(self, slot: _Slot, k: int) -> np.ndarray:
        """Lead times of state slot ``slot`` passed to the call producing chunk ``k``.

        ``initialize`` takes the input slot's window; ``step`` the previous chunk.
        """
        return slot.leads if k == 0 else self.chunk(k - 1)


def _outputs(name: str, model: Any, inputs: tuple[_Slot, ...]) -> tuple:
    outputs = _slots(model.output_coords(*(s.signature for s in inputs)))
    if not outputs:
        raise CouplingError(f"Component {name!r} declares no outputs (C1)")
    for i, output in enumerate(outputs):
        if not isinstance(output, xr.DataArray) or "variable" not in output.dims:
            raise CouplingError(
                f"Component {name!r} output slot {i} must be gridded with a "
                "'variable' dimension (C1)"
            )
    return outputs


def _stateful(name: str, model: Any) -> _Node:
    inputs = tuple(
        _Slot(i, s, False) for i, s in enumerate(_slots(model.input_coords()))
    )
    forcing_coords = getattr(model, "forcing_coords", None)
    signatures = _slots(forcing_coords()) if callable(forcing_coords) else ()
    forcing = tuple(_Slot(i, s, True) for i, s in enumerate(signatures))
    if not inputs or inputs[0].frame or inputs[0].static:
        raise CouplingError(
            f"Component {name!r} needs a gridded first input slot with lead times"
        )
    outputs = _outputs(name, model, inputs)
    if any("lead_time" not in output.dims for output in outputs):
        raise CouplingError(f"Component {name!r} output has no lead times (C1)")
    out_leads = _ns(outputs[0]["lead_time"].values)
    for i, output in enumerate(outputs[1:], start=1):
        if not np.array_equal(_ns(output["lead_time"].values), out_leads):
            raise CouplingError(
                f"Component {name!r} output slot {i} has different lead times "
                "from slot 0; split models whose outputs advance at different "
                "cadences into separate components (C1)"
            )
    ic_lead = int(inputs[0].leads[-1])
    cadence = int(out_leads[-1]) - ic_lead
    if cadence <= 0:
        raise CouplingError(f"Component {name!r} does not advance in time")
    spacing = int(out_leads[1] - out_leads[0]) if len(out_leads) > 1 else cadence
    # Output slot i publishes an initial condition from input slot i when that
    # output declares that input's variables on its grid, as state slots come first.
    ic_slots = tuple(
        i
        for i, slot in enumerate(inputs[: len(outputs)])
        if not slot.frame
        and not slot.static
        and _publishes_initial_condition(slot.signature, outputs[i])
    )
    return _Node(
        name,
        model,
        True,
        inputs,
        forcing,
        outputs,
        tuple(s for s in forcing if not s.frame and not s.static),
        out_leads,
        ic_lead,
        cadence,
        spacing,
        ic_slots,
    )


def _diagnostic(name: str, model: Any) -> _Node:
    # Inputs without lead times, as most diagnostics declare, take the call's
    # valid time as a leading dimension.
    inputs = tuple(
        _Slot(i, s, False, per_call=True)
        for i, s in enumerate(_slots(model.input_coords()))
    )
    for slot in inputs:
        if slot.frame:
            raise CouplingError(
                f"Diagnostic {name!r} {slot} takes a DataFrame; only gridded "
                "diagnostic inputs can be coupled"
            )
        if "lead_time" in slot.signature.dims and slot.signature.sizes["lead_time"] > 1:
            raise CouplingError(
                f"Diagnostic {name!r} {slot} declares several lead times; "
                "coupled diagnostics take one valid time per call"
            )
    outputs = _outputs(name, model, inputs)
    return _Node(
        name, model, False, inputs, (), outputs, inputs, np.zeros(0, dtype=np.int64)
    )


def _node(name: str, model: Any) -> _Node:
    """Read ``model``'s declarations by the protocol it implements."""
    try:
        if callable(getattr(model, "initialize", None)) and callable(
            getattr(model, "step", None)
        ):
            return _stateful(name, model)
        if callable(model) and callable(getattr(model, "input_coords", None)):
            return _diagnostic(name, model)
    except CouplingError:
        raise
    except Exception as error:  # noqa: BLE001 - surface the model's own failure
        raise CouplingError(
            f"Component {name!r} declarations could not be read: {error}"
        ) from error
    raise CouplingError(
        f"Component {name!r} is neither a stateful model (initialize/step) nor a "
        "diagnostic model"
    )
