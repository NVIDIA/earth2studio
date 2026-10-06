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

"""DataArray exchange objects used by the coupler."""

from collections.abc import Iterable, Iterator, MutableMapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np
import xarray as xr

from earth2studio.grids import infer_grid

from .dictionary import FieldDictionary
from .errors import CouplingError

if TYPE_CHECKING:
    from .vertical import VerticalCoordinate

# Dims regarded as spatial when choosing where to (re)insert a variable axis.
# ``point`` is a scattered sample-location dimension.
_SPATIAL_DIMS = (
    "level",
    "face",
    "lat",
    "lon",
    "hpx",
    "height",
    "width",
    "y",
    "x",
    "point",
)


@dataclass
class Field:
    """One exchanged quantity.

    Parameters
    ----------
    array : xarray.DataArray
        Labeled field values. Must not contain a ``variable`` dimension.
    standard_name : str
        Canonical name from the FieldDictionary.
    units : str
        Units of ``array`` (checked, not converted).
    valid_time : np.datetime64, optional
        Time the data is valid for.
    source : str, optional
        Name of the producing component (provenance).
    mask : xarray.DataArray, optional
        Boolean validity mask broadcastable to ``array`` (True = valid),
        e.g. ocean points for SST.
    vertical : VerticalCoordinate, optional
        Vertical coordinate description when `coords` contains a "level"
        dimension (see :mod:`earth2studio.nvcoupler.vertical`).
    """

    array: xr.DataArray
    standard_name: str
    units: str
    valid_time: np.datetime64 | None = None
    source: str | None = None
    mask: xr.DataArray | None = None
    vertical: "VerticalCoordinate | None" = None

    def __post_init__(self) -> None:
        if not isinstance(self.array, xr.DataArray):
            raise TypeError(
                f"Field {self.standard_name!r} array must be an xarray.DataArray"
            )
        if "variable" in self.array.dims:
            raise CouplingError(
                f"Field {self.standard_name!r} array must not contain a "
                "'variable' dimension; use State.from_dataarray to split a "
                "multi-variable DataArray into Fields"
            )
        if self.mask is not None:
            try:
                self.mask.broadcast_like(self.array)
            except ValueError as error:
                raise CouplingError(
                    f"Field {self.standard_name!r} mask is not broadcastable "
                    f"to dimensions {self.array.dims}"
                ) from error

    @property
    def data(self) -> Any:
        """The NumPy/CuPy payload of :attr:`array`."""
        return self.array.data

    def with_array(self, array: xr.DataArray) -> "Field":
        return replace(self, array=array)

    def to_backend(self, backend: str) -> "Field":
        if backend == "numpy":
            array = self.array.e2s.as_numpy()
            mask = self.mask.e2s.as_numpy() if self.mask is not None else None
        elif backend == "cupy":
            array = self.array.e2s.as_cupy()
            mask = self.mask.e2s.as_cupy() if self.mask is not None else None
        else:
            raise CouplingError(
                f"Unsupported array backend {backend!r}; choose 'numpy' or 'cupy'"
            )
        return replace(self, array=array, mask=mask)

    def clone(self) -> "Field":
        return replace(
            self,
            array=self.array.copy(deep=True),
            mask=self.mask.copy(deep=True) if self.mask is not None else None,
        )

    def grid_signature(self) -> tuple:
        """Hashable signature of the spatial grid, for regridder caching."""
        try:
            return (infer_grid(self.array).fingerprint(),)
        except ValueError:
            pass
        parts: list[tuple] = []
        for key in self.array.dims:
            if key in _SPATIAL_DIMS:
                value = np.asarray(self.array.coords[key])
                parts.append((key, value.shape, value.tobytes()))
        return tuple(parts)

    def __repr__(self) -> str:
        dims = ", ".join(f"{name}: {size}" for name, size in self.array.sizes.items())
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

    def stack(self, names: list[str] | None = None) -> xr.DataArray:
        """Stack fields along a ``variable`` dimension.

        All selected fields must share identical coords (same grid); use a
        Connector to bring fields onto one grid first. The variable axis is
        inserted immediately before the first spatial dimension, matching the
        earth2studio convention (batch, time, lead_time, variable, spatial...).
        """
        names = list(names) if names is not None else sorted(self._fields)
        if not names:
            raise CouplingError(f"State {self.name!r}: no fields to stack")
        fields = [self[n] for n in names]
        dims = list(fields[0].array.dims)
        insert_at = next(
            (i for i, d in enumerate(dims) if d in _SPATIAL_DIMS), len(dims)
        )
        try:
            stacked = xr.concat(
                [field.array for field in fields],
                xr.IndexVariable("variable", names),
                join="exact",
            )
        except (ValueError, KeyError) as error:
            raise CouplingError(
                f"State {self.name!r}: fields cannot be stacked because their "
                f"dimensions or coordinates differ: {error}"
            ) from error
        order = list(stacked.dims)
        order.remove("variable")
        order.insert(insert_at, "variable")
        return stacked.transpose(*order)

    @classmethod
    def from_dataarray(
        cls,
        name: str,
        array: xr.DataArray,
        dictionary: FieldDictionary,
        valid_time: np.datetime64 | None = None,
        source: str | None = None,
        strict: bool = True,
    ) -> "State":
        """Split a multi-variable DataArray into a State of Fields.

        Raw variable names in ``coords["variable"]`` are resolved to standard
        names (and canonical units) through the dictionary. Unknown names
        raise unless ``strict=False``, in which case they are skipped.
        """
        if "variable" not in array.dims:
            raise CouplingError(
                f"from_dataarray for state {name!r}: array has no 'variable' dim"
            )
        state = cls(name)
        for raw_name in np.asarray(array.coords["variable"]):
            if raw_name not in dictionary:
                if strict:
                    dictionary.resolve(str(raw_name))  # raises UnknownFieldError
                continue
            entry = dictionary.resolve(str(raw_name))
            state.add(
                Field(
                    array=array.sel(variable=raw_name, drop=True),
                    standard_name=entry.standard_name,
                    units=entry.canonical_units,
                    valid_time=valid_time,
                    source=source,
                )
            )
        return state

    def __repr__(self) -> str:
        return f"State({self.name!r}, fields={sorted(self._fields)})"
