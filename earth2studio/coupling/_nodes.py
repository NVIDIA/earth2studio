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

"""Adapters presenting each model protocol to the coupled scheduler.

The scheduler and buffers see only nodes: their slots, what they export, when
they run and which lead times each slot needs. All protocol-specific behavior
lives here, so a new kind of model needs only a new adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

from earth2studio.coupling.errors import CouplingError
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

    @property
    def frame(self) -> bool:
        """Whether the slot takes a DataFrame."""
        return not isinstance(self.signature, xr.DataArray)

    @property
    def static(self) -> bool:
        """Whether the slot is gridded without lead times, supplied only initially."""
        return not self.frame and "lead_time" not in self.signature.dims

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


class _Node:
    """A named model as the scheduler sees it."""

    stateful: bool
    kind: str

    def __init__(self, name: str, model: Any) -> None:
        self.name = name
        self.model = model
        self.inputs: tuple[_Slot, ...] = ()
        self.forcing: tuple[_Slot, ...] = ()
        self.outputs: tuple[CoordinateSystem, ...] = ()

    def _set_outputs(self, outputs: Any) -> None:
        outputs = _slots(outputs)
        if not outputs:
            raise CouplingError(f"Component {self.name!r} declares no outputs (C1)")
        for i, output in enumerate(outputs):
            if not isinstance(output, xr.DataArray) or "variable" not in output.dims:
                raise CouplingError(
                    f"Component {self.name!r} output slot {i} must be gridded with "
                    "a 'variable' dimension (C1)"
                )
        self.outputs = outputs

    @property
    def imports(self) -> tuple[_Slot, ...]:
        """Slots a connector may feed."""
        raise NotImplementedError

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


class _Stateful(_Node):
    """Prognostic or stateful assimilation model: ``initialize`` then ``step``.

    Chunk ``k`` covers the output lead times ``out_leads + k * cadence``; it is
    due once its last lead time is reached. Chunk 0 comes from ``initialize``.
    """

    stateful = True
    kind = "stateful"

    def __init__(self, name: str, model: Any) -> None:
        super().__init__(name, model)
        self.inputs = tuple(
            _Slot(i, s, False) for i, s in enumerate(_slots(model.input_coords()))
        )
        forcing_coords = getattr(model, "forcing_coords", None)
        signatures = _slots(forcing_coords()) if callable(forcing_coords) else ()
        self.forcing = tuple(_Slot(i, s, True) for i, s in enumerate(signatures))
        if not self.inputs or self.inputs[0].frame or self.inputs[0].static:
            raise CouplingError(
                f"Component {name!r} needs a gridded first input slot with lead times"
            )
        self._set_outputs(model.output_coords(*(s.signature for s in self.inputs)))
        if any("lead_time" not in output.dims for output in self.outputs):
            raise CouplingError(f"Component {name!r} output has no lead times (C1)")
        self.out_leads = _ns(self.outputs[0]["lead_time"].values)
        for i, output in enumerate(self.outputs[1:], start=1):
            if not np.array_equal(_ns(output["lead_time"].values), self.out_leads):
                raise CouplingError(
                    f"Component {name!r} output slot {i} has different lead times "
                    "from slot 0; split models whose outputs advance at different "
                    "cadences into separate components (C1)"
                )
        self.ic_lead = int(self.inputs[0].leads[-1])
        self.cadence = int(self.out_leads[-1]) - self.ic_lead
        if self.cadence <= 0:
            raise CouplingError(f"Component {name!r} does not advance in time")
        self.spacing = (
            int(self.out_leads[1] - self.out_leads[0])
            if len(self.out_leads) > 1
            else self.cadence
        )
        # Output slot i publishes an initial condition from input slot i when
        # that input holds the same variables, as state slots come first.
        self.ic_slots = tuple(
            i
            for i, slot in enumerate(self.inputs[: len(self.outputs)])
            if not slot.frame
            and not slot.static
            and set(slot.labels) == set(self.exports(i))
        )

    @property
    def imports(self) -> tuple[_Slot, ...]:
        return tuple(s for s in self.forcing if not s.frame and not s.static)

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

    def required(self, slot: _Slot, k: int) -> np.ndarray:
        """Lead times ``slot`` needs for the call producing chunk ``k``.

        ``initialize`` takes the whole forcing window; ``step`` takes the window's
        final lead time shifted by each lead time of the output it advances.
        DataFrame slots are requested at the analysis time, the chunk's end.
        """
        if slot.frame:
            return np.array([self.due(k)], dtype=np.int64)
        if slot.static:
            return np.zeros(1, dtype=np.int64)
        return slot.leads if k == 0 else slot.leads[-1] + self.chunk(k - 1)


class _Diagnostic(_Node):
    """Diagnostic model: called once per valid time at which its inputs arrive."""

    stateful = False
    kind = "diagnostic"

    def __init__(self, name: str, model: Any) -> None:
        super().__init__(name, model)
        self.inputs = tuple(
            _Slot(i, s, False) for i, s in enumerate(_slots(model.input_coords()))
        )
        for slot in self.inputs:
            if slot.frame:
                raise CouplingError(
                    f"Diagnostic {name!r} {slot} takes a DataFrame; only gridded "
                    "diagnostic inputs can be coupled"
                )
            if (
                "lead_time" in slot.signature.dims
                and slot.signature.sizes["lead_time"] > 1
            ):
                raise CouplingError(
                    f"Diagnostic {name!r} {slot} declares several lead times; "
                    "coupled diagnostics take one valid time per call"
                )
        self._set_outputs(model.output_coords(*(s.signature for s in self.inputs)))

    @property
    def imports(self) -> tuple[_Slot, ...]:
        return self.inputs

    def required(self, slot: _Slot, valid: int) -> np.ndarray:
        """A diagnostic reads every input at the valid time it is called for."""
        return np.array([valid], dtype=np.int64)


def _node(name: str, model: Any) -> _Node:
    """Adapt ``model`` by the protocol it implements."""
    try:
        if callable(getattr(model, "initialize", None)) and callable(
            getattr(model, "step", None)
        ):
            return _Stateful(name, model)
        if callable(model) and callable(getattr(model, "input_coords", None)):
            return _Diagnostic(name, model)
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
