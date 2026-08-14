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

"""Mediators: components that compute derived exchange fields.

The NUOPC_Mediator analog. A Mediator sits between components of different
cadence, accumulating fast-component fields as they arrive (every connector
transfer into it) and, when its alarm rings, exporting a windowed reduction
— the trailing 48 h mean an ocean model was trained on, a precipitation sum
a flood model needs, a temperature max for impact indices.

Reductions are running torch ops (add / maximum / minimum), so memory is one
accumulator per field regardless of window length, and gradients flow
through mean and sum (max/min propagate to the extremal sample).
"""

from collections import OrderedDict
from typing import Any

import numpy as np
import torch

from .component import Component
from .dictionary import CellMethod
from .errors import CouplingError
from .field import Field, State

# Wire in the same valid_time twice (e.g. two connectors or a re-executed
# slot) and the second arrival is ignored rather than double-counted.


class _AccumulatingState(State):
    """Import state that forwards every added field to the owning mediator."""

    def __init__(self, name: str, mediator: "Mediator"):
        super().__init__(name)
        self._mediator = mediator

    def add(self, field: Field, replace: bool = True) -> None:
        super().add(field, replace=replace)
        self._mediator.accumulate(field)


class Mediator(Component):
    """Base class: accumulate imports, reduce on ring.

    Subclasses implement :meth:`accumulate` (called on every field arriving
    in the import state) and :meth:`compute` (called when the mediator's
    alarm rings; must populate ``export_state``).
    """

    requires_ic = False  # mediators need no initial condition

    def __init__(self, name: str, timestep: Any, imports=(), exports=(), **kwargs: Any):
        super().__init__(name, timestep, imports, exports, **kwargs)
        self.import_state = _AccumulatingState(f"{name}.imports", self)

    def initialize(self, x: torch.Tensor | None = None, coords=None) -> None:
        """Mediators need no initial condition."""

    def accumulate(self, field: Field) -> None:
        raise NotImplementedError

    def compute(self, time: np.datetime64) -> None:
        raise NotImplementedError

    def run(self, time: np.datetime64) -> None:
        self.compute(time)
        self.run_count += 1


class AccumulationMediator(Mediator):
    """Windowed reduction of fast-component fields onto a slow cadence.

    Parameters
    ----------
    name : str
    fields : list[str]
        *Derived* standard names to produce (e.g.
        ``geopotential_at_1000hpa_48h_mean``). Each must be a dictionary
        entry carrying a :class:`CellMethod`; the cell method supplies the
        base field to import, the reduction, and the window (= the
        mediator's timestep unless ``window`` overrides it).
    window : optional
        Override the alarm interval; defaults to the (common) cell-method
        window of `fields`.

    This is the generalization of PhysicsNeMo's TrailingAverageCoupler and
    DLESyM's ``_make_ocean_coupling`` chunk-mean, plus the sum/max/min
    reductions impact chains need.
    """

    def __init__(self, name: str, fields: list[str], window: Any = None, **kwargs: Any):
        from .dictionary import DEFAULT_DICTIONARY

        dictionary = kwargs.get("dictionary") or DEFAULT_DICTIONARY
        methods: dict[str, CellMethod] = {}
        for derived in fields:
            entry = dictionary.resolve(derived)
            if entry.cell_method is None:
                raise CouplingError(
                    f"AccumulationMediator {name!r}: {derived!r} has no "
                    "cell_method in the field dictionary — register a "
                    "FieldEntry(cell_method=CellMethod(base, method, window)) "
                    "describing how it derives from a base field"
                )
            methods[entry.standard_name] = entry.cell_method
        windows = {cm.window for cm in methods.values()}
        if window is None:
            if len(windows) != 1:
                raise CouplingError(
                    f"AccumulationMediator {name!r}: fields have differing "
                    f"windows {sorted(str(w) for w in windows)}; split them "
                    "across mediators or pass window= explicitly"
                )
            window = next(iter(windows))
        # dedupe: several derived fields may reduce the same base import
        imports = list(dict.fromkeys(cm.base for cm in methods.values()))
        super().__init__(name, window, imports=imports, exports=list(methods), **kwargs)
        self.methods = methods  # derived std name -> CellMethod
        # base std name -> ALL derived fields reducing it (e.g. the 24h max
        # and 24h mean of t2m accumulate from the same delivered field)
        self._base_to_derived: dict[str, list[str]] = {}
        for derived, cm in methods.items():
            self._base_to_derived.setdefault(cm.base, []).append(derived)
        self._acc: dict[str, torch.Tensor] = {}
        self._count: dict[str, int] = {}
        self._coords: dict[str, OrderedDict] = {}
        self._last_time: dict[str, np.datetime64] = {}

    def accumulate(self, field: Field) -> None:
        for derived in self._base_to_derived.get(field.standard_name, ()):
            if (
                field.valid_time is not None
                and self._last_time.get(derived) == field.valid_time
            ):
                continue  # duplicate arrival for the same time
            self._last_time[derived] = field.valid_time
            method = self.methods[derived].method
            if derived not in self._acc:
                self._acc[derived] = field.data
                self._count[derived] = 1
            else:
                acc = self._acc[derived]
                if method in ("mean", "sum"):
                    self._acc[derived] = acc + field.data
                elif method == "max":
                    self._acc[derived] = torch.maximum(acc, field.data)
                else:  # min
                    self._acc[derived] = torch.minimum(acc, field.data)
                self._count[derived] += 1
            self._coords[derived] = OrderedDict(field.coords)

    def compute(self, time: np.datetime64) -> None:
        for derived, cm in self.methods.items():
            if derived not in self._acc:
                raise CouplingError(
                    f"Mediator {self.name!r}: no samples of {cm.base!r} "
                    f"accumulated before compute at {time} — is a connector "
                    "feeding this mediator in a faster slot?"
                )
            data = self._acc[derived]
            if cm.method == "mean":
                data = data / self._count[derived]
            entry = self.dictionary.resolve(derived)
            self.export_state.add(
                Field(
                    data=data,
                    coords=self._coords[derived],
                    standard_name=derived,
                    units=entry.canonical_units,
                    valid_time=time,
                    source=self.name,
                )
            )
        self.samples_last_window = dict(self._count)
        self._acc.clear()
        self._count.clear()
        self._last_time.clear()


class TrailingAverageMediator(AccumulationMediator):
    """AccumulationMediator restricted to mean reductions — the exact
    semantics of DLESyM's ocean coupling and PhysicsNeMo's
    TrailingAverageCoupler."""

    def __init__(self, name: str, fields: list[str], window: Any = None, **kwargs: Any):
        super().__init__(name, fields, window, **kwargs)
        bad = [f for f, cm in self.methods.items() if cm.method != "mean"]
        if bad:
            raise CouplingError(
                f"TrailingAverageMediator {name!r}: fields {bad} are not mean "
                "reductions — use AccumulationMediator"
            )
