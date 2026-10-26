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

"""What the coupler reads off a model's declarations.

Everything the coupler needs to schedule and feed a model comes from the model
contract (``dev/spec/MODEL_CONTRACT_SPEC.md``): imports from ``forcing_coords()``
or a diagnostic's ``input_coords()``, exports from ``output_coords()``, cadence
from the lead-time advance, and grids from the signatures. No coupler-specific
base class or protocol is involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from earth2studio.grids import GridDefinition, infer_grid
from earth2studio.nvcoupler._time import DeltaLike, as_timedelta
from earth2studio.nvcoupler.errors import CouplingError
from earth2studio.utils.type import CoordinateSystem

Kind = Literal["prognostic", "diagnostic"]
SlotKind = Literal["input", "forcing", "state"]


def _slots(signature: Any) -> tuple[CoordinateSystem, ...]:
    if signature is None:
        return ()
    return signature if isinstance(signature, tuple) else (signature,)


def _labels(signature: CoordinateSystem) -> list[str]:
    if "variable" not in signature.coords:
        return []
    return [str(label) for label in signature.coords["variable"].values]


def _lead_times(signature: CoordinateSystem) -> np.ndarray | None:
    """Concrete lead times of a slot, or ``None`` when absent or dynamic."""
    if "lead_time" not in signature.dims or signature.sizes["lead_time"] == 0:
        return None
    return np.asarray(signature.coords["lead_time"].values).astype("timedelta64[ns]")


def _grid(signature: CoordinateSystem) -> GridDefinition | None:
    try:
        return infer_grid(signature)
    except (ValueError, KeyError, TypeError):
        return None


@dataclass
class _ModelSpec:
    """A model and what its declarations say about coupling it."""

    name: str
    model: Any
    kind: Kind
    inputs: tuple[CoordinateSystem, ...]
    forcing: tuple[CoordinateSystem, ...]
    outputs: tuple[CoordinateSystem, ...]
    cadence: np.timedelta64 | None
    imports: dict[str, int] = field(default_factory=dict)  # label -> import slot
    exports: dict[str, int] = field(default_factory=dict)  # label -> output slot
    state: dict[str, int] = field(default_factory=dict)  # label -> output slot

    @property
    def import_kind(self) -> SlotKind:
        """Which signatures imports feed: forcing slots or diagnostic inputs."""
        return "forcing" if self.kind == "prognostic" else "input"

    @property
    def import_slots(self) -> tuple[CoordinateSystem, ...]:
        """Signatures imports are delivered to."""
        return self.forcing if self.kind == "prognostic" else self.inputs

    def slot(self, kind: SlotKind, index: int) -> CoordinateSystem:
        """Return the signature of one slot."""
        if kind == "forcing":
            return self.forcing[index]
        if kind == "state":
            return self.outputs[index]
        return self.inputs[index]

    def stream(self, index: int) -> str:
        """Name of the stream carrying output slot ``index``."""
        return self.name if index == 0 else f"{self.name}.{index}"

    def required_cadence(self) -> np.timedelta64:
        """Cadence, raising when the coupler could not determine one."""
        if self.cadence is None:
            raise CouplingError(
                f"Diagnostic {self.name!r} has no cadence: it is fed by no coupled "
                "model, so pass cadence={" + repr(self.name) + ": '6h'} (say) to "
                "the Coupler"
            )
        return self.cadence


def _kind(model: Any) -> Kind:
    stateful = all(callable(getattr(model, m, None)) for m in ("initialize", "step"))
    declares = all(
        callable(getattr(model, m, None)) for m in ("input_coords", "output_coords")
    )
    if not declares:
        raise TypeError(
            f"{type(model).__name__} declares no input_coords()/output_coords(); "
            "coupled models implement PrognosticModel or DiagnosticModel"
        )
    if stateful:
        return "prognostic"
    if callable(model):
        return "diagnostic"
    raise TypeError(
        f"{type(model).__name__} is neither a PrognosticModel (initialize/step) "
        "nor a callable DiagnosticModel"
    )


def inspect_model(
    name: str, model: Any, cadence: DeltaLike | None = None
) -> _ModelSpec:
    """Read a model's declarations into a :class:`_ModelSpec`.

    Parameters
    ----------
    name : str
        The model's name in the coupled system.
    model : PrognosticModel | DiagnosticModel
        The model.
    cadence : DeltaLike, optional
        Explicit cadence. Required for a diagnostic not fed by a coupled model;
        a prognostic's cadence is always read from its declaration.

    Returns
    -------
    _ModelSpec
        The model's imports, exports, state variables, cadence and signatures.
    """
    kind = _kind(model)
    inputs = _slots(model.input_coords())
    if not inputs:
        raise CouplingError(f"Model {name!r} declares no input slots")
    forcing = _slots(model.forcing_coords()) if kind == "prognostic" else ()
    try:
        outputs = _slots(model.output_coords(*inputs))
    except Exception as error:  # the model's own validation
        raise CouplingError(
            f"Model {name!r}: output_coords() rejected its own input_coords(): {error}"
        ) from error
    if not outputs:
        raise CouplingError(f"Model {name!r} declares no output slots")

    spec = _ModelSpec(name, model, kind, inputs, forcing, outputs, None)
    for index, signature in enumerate(outputs):
        for label in _labels(signature):
            spec.exports.setdefault(label, index)
    for index, signature in enumerate(spec.import_slots):
        for label in _labels(signature):
            if label in spec.imports:
                raise CouplingError(
                    f"Model {name!r} declares {label!r} in two "
                    f"{spec.import_kind} slots; the coupler cannot tell them apart"
                )
            spec.imports[label] = index

    if kind == "prognostic":
        if cadence is not None:
            raise CouplingError(
                f"Model {name!r} is prognostic; its cadence comes from its "
                "declaration and cannot be overridden"
            )
        spec.cadence = _prognostic_cadence(name, inputs, outputs)
        input_labels = {label for sig in inputs for label in _labels(sig)}
        for index, signature in enumerate(outputs):
            for label in _labels(signature):
                if label in input_labels and label not in spec.imports:
                    spec.state.setdefault(label, index)
    elif cadence is not None:
        spec.cadence = as_timedelta(cadence)
        if spec.cadence <= np.timedelta64(0, "ns"):
            raise CouplingError(f"Model {name!r}: cadence must be positive")
    return spec


def _prognostic_cadence(
    name: str,
    inputs: tuple[CoordinateSystem, ...],
    outputs: tuple[CoordinateSystem, ...],
) -> np.timedelta64:
    """The advance of one step: final output lead minus final input lead."""
    final_in = [leads[-1] for sig in inputs if (leads := _lead_times(sig)) is not None]
    final_out = [
        leads[-1] for sig in outputs if (leads := _lead_times(sig)) is not None
    ]
    if not final_in or not final_out:
        raise CouplingError(
            f"Prognostic {name!r} must declare concrete lead_time coordinates on "
            "its input and output slots so the coupler can read its cadence"
        )
    cadence = max(final_out) - max(final_in)
    if cadence <= np.timedelta64(0, "ns"):
        raise CouplingError(
            f"Prognostic {name!r} does not advance lead time (input ends at "
            f"{max(final_in)}, output at {max(final_out)})"
        )
    return cadence.astype("timedelta64[ns]")
