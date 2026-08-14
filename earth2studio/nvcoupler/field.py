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

"""Field and State: the exchange currency of the coupler (ESMF analogs).

A :class:`Field` is one physical quantity — a torch tensor plus its
CoordSystem, canonical identity (standard name + units), validity time, and
optional mask / vertical-coordinate metadata. A :class:`State` is a named
bag of Fields keyed by standard name; every component owns an import State
and an export State, and Connectors move Fields between them.

Field data stays a torch tensor end-to-end (never round-tripped through
numpy) so autograd graphs survive the exchange — a hard requirement for
coupled fine-tuning.
"""

from collections import OrderedDict
from collections.abc import Iterable, Iterator, MutableMapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from earth2studio.utils.coords import cat_coords, split_coords
from earth2studio.utils.type import CoordSystem

from .dictionary import FieldDictionary
from .errors import CouplingError

if TYPE_CHECKING:
    from .vertical import VerticalCoordinate

# Dims regarded as spatial when choosing where to (re)insert a variable axis
_SPATIAL_DIMS = ("level", "face", "lat", "lon", "hpx", "height", "width", "y", "x")


@dataclass
class Field:
    """One exchanged quantity.

    Parameters
    ----------
    data : torch.Tensor
        Field values; dimension order given by `coords` insertion order.
        Must NOT contain a "variable" dimension — a Field is one variable.
    coords : CoordSystem
        earth2studio coordinate dictionary describing `data`.
    standard_name : str
        Canonical name from the FieldDictionary.
    units : str
        Units of `data` (checked, not converted, in v1).
    valid_time : np.datetime64, optional
        Time the data is valid for.
    source : str, optional
        Name of the producing component (provenance).
    mask : torch.Tensor, optional
        Boolean validity mask broadcastable to `data` (True = valid),
        e.g. ocean points for SST.
    vertical : VerticalCoordinate, optional
        Vertical coordinate description when `coords` contains a "level"
        dimension (see :mod:`earth2studio.nvcoupler.vertical`).
    """

    data: torch.Tensor
    coords: CoordSystem
    standard_name: str
    units: str
    valid_time: np.datetime64 | None = None
    source: str | None = None
    mask: torch.Tensor | None = None
    vertical: "VerticalCoordinate | None" = None

    def __post_init__(self) -> None:
        if "variable" in self.coords:
            raise CouplingError(
                f"Field {self.standard_name!r} coords must not contain a "
                "'variable' dimension; use State.from_tensor to split a "
                "multi-variable tensor into Fields"
            )
        ndim_coords = len(self.coords)
        if self.data.ndim != ndim_coords:
            raise CouplingError(
                f"Field {self.standard_name!r}: data has {self.data.ndim} dims "
                f"but coords describe {ndim_coords} "
                f"({list(self.coords.keys())})"
            )

    def to(self, device: Any) -> "Field":
        return replace(
            self,
            data=self.data.to(device),
            mask=self.mask.to(device) if self.mask is not None else None,
        )

    def clone(self) -> "Field":
        return replace(
            self,
            data=self.data.clone(),
            coords=OrderedDict({k: v.copy() for k, v in self.coords.items()}),
            mask=self.mask.clone() if self.mask is not None else None,
        )

    def grid_signature(self) -> tuple:
        """Hashable signature of the spatial grid, for regridder caching."""
        parts: list[tuple] = []
        for key, value in self.coords.items():
            if key in _SPATIAL_DIMS:
                parts.append((key, value.shape, value.tobytes()))
        return tuple(parts)

    def __repr__(self) -> str:
        dims = ", ".join(f"{k}: {len(v) if v.ndim else 0}" for k, v in self.coords.items())
        t = f", valid_time={self.valid_time}" if self.valid_time is not None else ""
        return f"Field({self.standard_name!r} [{self.units}], {dims}{t})"


class State(MutableMapping):
    """A named collection of Fields keyed by standard name (ESMF_State analog)."""

    def __init__(self, name: str, fields: Iterable[Field] = ()):
        self.name = name
        self._fields: dict[str, Field] = {}
        for f in fields:
            self.add(f)

    # -- MutableMapping interface -------------------------------------------
    def __getitem__(self, key: str) -> Field:
        try:
            return self._fields[key]
        except KeyError:
            raise KeyError(
                f"State {self.name!r} has no field {key!r}; "
                f"present: {sorted(self._fields)}"
            ) from None

    def __setitem__(self, key: str, value: Field) -> None:
        if key != value.standard_name:
            raise CouplingError(
                f"State key {key!r} must equal the field's standard_name "
                f"{value.standard_name!r}"
            )
        self._fields[key] = value

    def __delitem__(self, key: str) -> None:
        del self._fields[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    # -- convenience ---------------------------------------------------------
    def add(self, field: Field, replace: bool = True) -> None:
        if not replace and field.standard_name in self._fields:
            raise CouplingError(
                f"Field {field.standard_name!r} already in state {self.name!r}"
            )
        self._fields[field.standard_name] = field

    def subset(self, names: Iterable[str]) -> "State":
        return State(self.name, (self[n] for n in names))

    def to(self, device: Any) -> "State":
        return State(self.name, (f.to(device) for f in self._fields.values()))

    def as_tensor(
        self, names: list[str] | None = None
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Stack fields along a new "variable" dimension.

        All selected fields must share identical coords (same grid); use a
        Connector to bring fields onto one grid first. The variable axis is
        inserted immediately before the first spatial dimension, matching the
        earth2studio convention (batch, time, lead_time, variable, spatial...).
        """
        names = list(names) if names is not None else sorted(self._fields)
        if not names:
            raise CouplingError(f"State {self.name!r}: no fields to stack")
        fields = [self[n] for n in names]
        ref = fields[0].coords
        dims = list(ref.keys())
        insert_at = next(
            (i for i, d in enumerate(dims) if d in _SPATIAL_DIMS), len(dims)
        )
        tensors, coord_list = [], []
        for f in fields:
            c = OrderedDict()
            for i, (k, v) in enumerate(f.coords.items()):
                if i == insert_at:
                    c["variable"] = np.array([f.standard_name])
                c[k] = v
            if "variable" not in c:
                c["variable"] = np.array([f.standard_name])
            tensors.append(f.data.unsqueeze(insert_at))
            coord_list.append(c)
        # cat_coords validates all non-variable dims match across fields
        return cat_coords(tuple(tensors), tuple(coord_list), dim="variable")

    @classmethod
    def from_tensor(
        cls,
        name: str,
        x: torch.Tensor,
        coords: CoordSystem,
        dictionary: FieldDictionary,
        valid_time: np.datetime64 | None = None,
        source: str | None = None,
        strict: bool = True,
    ) -> "State":
        """Split a multi-variable tensor into a State of Fields.

        Raw variable names in ``coords["variable"]`` are resolved to standard
        names (and canonical units) through the dictionary. Unknown names
        raise unless ``strict=False``, in which case they are skipped.
        """
        if "variable" not in coords:
            raise CouplingError(
                f"from_tensor for state {name!r}: coords have no 'variable' dim"
            )
        tensors, reduced_coords, values = split_coords(x, coords, dim="variable")
        state = cls(name)
        for tensor, raw_name in zip(tensors, values):
            if raw_name not in dictionary:
                if strict:
                    dictionary.resolve(str(raw_name))  # raises UnknownFieldError
                continue
            entry = dictionary.resolve(str(raw_name))
            state.add(
                Field(
                    data=tensor,
                    coords=OrderedDict(reduced_coords),
                    standard_name=entry.standard_name,
                    units=entry.canonical_units,
                    valid_time=valid_time,
                    source=source,
                )
            )
        return state

    def __repr__(self) -> str:
        return f"State({self.name!r}, fields={sorted(self._fields)})"
