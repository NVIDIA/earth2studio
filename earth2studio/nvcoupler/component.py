# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DataArray-native components for coupled inference workflows."""

from __future__ import annotations

import abc
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import xarray as xr

from earth2studio.grids import GridDefinition, infer_grid, resolve_grid
from earth2studio.utils import handshake_dataarray

from .clock import Clock, DeltaLike, as_datetime, as_timedelta, is_multiple
from .dictionary import DEFAULT_DICTIONARY, FieldDictionary
from .errors import CadenceError, CouplingError
from .field import State

StepFn = Callable[[xr.DataArray], xr.DataArray]
NextInputFn = Callable[[xr.DataArray, xr.DataArray], xr.DataArray]


def _stacking_order(
    field_order: list[str] | None, imports: State, who: str
) -> list[str]:
    if field_order is not None:
        missing = [name for name in field_order if name not in imports]
        if missing:
            raise CouplingError(
                f"{who}: field_order names {missing} are not in the import "
                f"state (present: {sorted(imports)})"
            )
        return list(field_order)
    if len(imports) > 1:
        raise CouplingError(
            f"{who}: {len(imports)} imported fields but no field_order= "
            f"given; pass an explicit ordering of {sorted(imports)}"
        )
    return list(imports)


def _squeeze_exchange(array: xr.DataArray) -> xr.DataArray:
    dims = [
        name
        for name in ("batch", "time", "lead_time")
        if name in array.dims and array.sizes[name] == 1
    ]
    return array.squeeze(dims, drop=True) if dims else array


def _dimension_values(signature: xr.DataArray, name: str) -> np.ndarray:
    if name not in signature.coords:
        raise CouplingError(
            f"Model signature has no coordinate values for dimension {name!r}"
        )
    return np.asarray(signature.coords[name])


def _conform_to_signature(
    array: xr.DataArray, signature: xr.DataArray, time: np.datetime64
) -> xr.DataArray:
    extra = set(array.dims) - set(signature.dims)
    if extra:
        raise CouplingError(
            f"Input dimensions {sorted(extra)} are absent from model signature "
            f"{list(signature.dims)}"
        )
    output = array
    for name in signature.dims:
        if name in output.dims:
            continue
        if name == "time":
            values = np.array([as_datetime(time)])
        elif name in signature.coords and signature.sizes[name]:
            values = np.asarray(signature.coords[name])
        else:
            values = np.arange(1)
        output = output.expand_dims({name: values})
    output = output.transpose(*signature.dims)
    output.attrs.update(signature.attrs)
    auxiliaries = {
        name: coordinate.variable
        for name, coordinate in signature.coords.items()
        if name not in output.coords
        and all(
            dim in output.dims and output.sizes[dim] == coordinate.sizes[dim]
            for dim in coordinate.dims
        )
    }
    return output.assign_coords(auxiliaries) if auxiliaries else output


def _align_conditioning(
    conditioning: xr.DataArray, reference: xr.DataArray
) -> xr.DataArray:
    output = conditioning
    if "variable" not in reference.dims:
        return output
    leading = reference.dims[: reference.dims.index("variable")]
    for name in leading:
        if name not in output.dims:
            output = output.expand_dims({name: reference.coords[name]})
    order = (
        *(name for name in reference.dims if name in output.dims),
        *(name for name in output.dims if name not in reference.dims),
    )
    return output.transpose(*order)


@dataclass(frozen=True)
class Exchange:
    """One component state plus the fields imported for its next step."""

    state: xr.DataArray
    imports: State
    std_to_raw: Mapping[str, str] = dc_field(default_factory=dict)
    time: np.datetime64 | None = None

    def inject(self) -> xr.DataArray:
        """Return state with imported fields replacing matching variables."""
        if not self.imports:
            return self.state
        if "variable" not in self.state.dims:
            raise CouplingError(
                "Exchange.inject requires a 'variable' dimension in state"
            )
        replacements = {
            self.std_to_raw.get(standard_name, standard_name): field.array
            for standard_name, field in self.imports.items()
        }
        variables = [str(value) for value in self.state.coords["variable"].values]
        unknown = sorted(set(replacements) - set(variables))
        if unknown:
            raise CouplingError(
                f"Imported model variables {unknown} are not in state variables "
                f"{variables}; use a conditioning adapter instead"
            )
        pieces = []
        for raw_name in variables:
            target = self.state.sel(variable=raw_name, drop=True)
            value = replacements.get(raw_name, target)
            try:
                value = value.broadcast_like(target).transpose(*target.dims)
            except ValueError as error:
                raise CouplingError(
                    f"Imported field for {raw_name!r} is incompatible with "
                    f"state dimensions {target.dims}"
                ) from error
            pieces.append(value)
        output = xr.concat(
            pieces, xr.IndexVariable("variable", np.asarray(variables)), join="exact"
        )
        output = output.transpose(*self.state.dims)
        output.attrs = dict(self.state.attrs)
        return output

    def stacked(
        self, field_order: list[str] | None = None, *, who: str = "Exchange.stacked"
    ) -> xr.DataArray:
        return self.imports.stack(_stacking_order(field_order, self.imports, who))


@runtime_checkable
class ImportAdapter(Protocol):
    def __call__(self, model: Any, exchange: Exchange) -> xr.DataArray: ...


class VariableOverwriteAdapter:
    """Overwrite matching state variables, then call a DataArray model."""

    def __call__(self, model: Any, exchange: Exchange) -> xr.DataArray:
        return model(exchange.inject())


class ConditioningKwargAdapter:
    """Call a model conditioning method with a stacked import DataArray."""

    def __init__(
        self,
        field_order: list[str] | None = None,
        method: str = "call_with_conditioning",
    ):
        self.field_order = field_order
        self.method = method

    def __call__(self, model: Any, exchange: Exchange) -> xr.DataArray:
        conditioning = exchange.stacked(self.field_order, who=type(self).__name__)
        conditioning = conditioning.assign_coords(
            variable=np.array(
                [
                    exchange.std_to_raw.get(str(name), str(name))
                    for name in conditioning.coords["variable"].values
                ]
            )
        )
        conditioning = _align_conditioning(conditioning, exchange.state)
        return getattr(model, self.method)(exchange.state, conditioning)


class Component(abc.ABC):
    """A scheduled participant in a coupled inference workflow."""

    requires_ic: bool = True

    def __init__(
        self,
        name: str,
        timestep: DeltaLike,
        imports: Iterable[str] = (),
        exports: Iterable[str] = (),
        dictionary: FieldDictionary | None = None,
        variable_aliases: Mapping[str, str] | None = None,
        export_masks: Mapping[str, xr.DataArray | np.ndarray] | None = None,
        import_vertical: Mapping[str, Any] | None = None,
        export_vertical: Mapping[str, Any] | None = None,
        grid: str | GridDefinition | None = None,
    ):
        self.name = name
        self.timestep = as_timedelta(timestep)
        self.dictionary = FieldDictionary(dictionary or DEFAULT_DICTIONARY)
        self._raw_to_std = dict(variable_aliases or {})
        for raw, standard in self._raw_to_std.items():
            if raw not in self.dictionary:
                self.dictionary.add_alias(standard, raw)
        self._std_to_raw = {standard: raw for raw, standard in self._raw_to_std.items()}
        self.import_names = [self.dictionary.standard_name(name) for name in imports]
        self.export_names = [self.dictionary.standard_name(name) for name in exports]
        self.export_masks = dict(export_masks or {})
        self.import_vertical = dict(import_vertical or {})
        self.export_vertical = dict(export_vertical or {})
        self.grid = resolve_grid(grid) if isinstance(grid, str) else grid
        self.import_state = State(f"{name}.imports")
        self.export_state = State(f"{name}.exports")
        self.clock: Clock | None = None
        self.run_count = 0

    def advertise(self) -> tuple[list[str], list[str]]:
        return list(self.import_names), list(self.export_names)

    def realize(self, clock: Clock) -> None:
        if not is_multiple(self.timestep, clock.dt):
            raise CadenceError(
                f"Component {self.name!r} timestep",
                str(self.timestep),
                str(clock.dt),
            )
        self.clock = clock

    @abc.abstractmethod
    def initialize(self, state: xr.DataArray | None = None) -> None: ...

    @abc.abstractmethod
    def run(self, time: np.datetime64) -> None: ...

    def finalize(self) -> None:
        pass

    def _exchange(self, state: xr.DataArray, time: np.datetime64) -> Exchange:
        imports = self.import_state.subset(
            name for name in self.import_names if name in self.import_state
        )
        return Exchange(
            state,
            imports,
            self.resolve_std_to_raw(state),
            time,
        )

    def grid_definition(self) -> GridDefinition | None:
        if self.grid is not None:
            return self.grid
        state = getattr(self, "_state", None)
        if state is None:
            return None
        try:
            return infer_grid(state)
        except ValueError:
            return None

    def resolve_std_to_raw(self, array: xr.DataArray) -> dict[str, str]:
        mapping: dict[str, str] = {}
        if "variable" in array.coords:
            for raw_value in array.coords["variable"].values:
                raw = str(raw_value)
                if raw in self.dictionary:
                    mapping[self.dictionary.standard_name(raw)] = raw
        mapping.update(self._std_to_raw)
        return mapping

    def publish(self, array: xr.DataArray, valid_time: np.datetime64 | None) -> None:
        state = State.from_dataarray(
            f"{self.name}.exports",
            _squeeze_exchange(array),
            self.dictionary,
            valid_time=valid_time,
            source=self.name,
            strict=False,
        )
        for standard_name in self.export_names:
            if standard_name not in state:
                variables = list(array.coords.get("variable", ()))
                raise CouplingError(
                    f"Component {self.name!r} advertises export "
                    f"{standard_name!r} but output variables are {variables}"
                )
            field = state[standard_name]
            if standard_name in self.export_masks:
                mask = self.export_masks[standard_name]
                if not isinstance(mask, xr.DataArray):
                    mask = xr.DataArray(mask, dims=field.array.dims)
                field.mask = mask
            if standard_name in self.export_vertical:
                field.vertical = self.export_vertical[standard_name]
            self.export_state.add(field)

    def __repr__(self) -> str:
        from .clock import fmt_timedelta

        return (
            f"{type(self).__name__}({self.name!r}, "
            f"dt={fmt_timedelta(self.timestep)}, imports={self.import_names}, "
            f"exports={self.export_names})"
        )


class CallableComponent(Component):
    def __init__(
        self,
        name: str,
        fn: StepFn,
        timestep: DeltaLike,
        imports: Iterable[str] = (),
        exports: Iterable[str] = (),
        import_adapter: ImportAdapter | None = None,
        **kwargs: Any,
    ):
        super().__init__(name, timestep, imports, exports, **kwargs)
        self.fn = fn
        self.import_adapter = import_adapter or VariableOverwriteAdapter()
        self._state: xr.DataArray | None = None

    def initialize(self, state: xr.DataArray | None = None) -> None:
        if state is None:
            raise CouplingError(f"Component {self.name!r} needs an initial condition")
        self._state = state
        self.publish(state, self.clock.start if self.clock is not None else None)

    def run(self, time: np.datetime64) -> None:
        if self._state is None:
            raise CouplingError(f"Component {self.name!r} not initialized")
        self._state = self.import_adapter(self.fn, self._exchange(self._state, time))
        self.publish(self._state, time)
        self.run_count += 1

    @property
    def state(self) -> xr.DataArray:
        if self._state is None:
            raise CouplingError(f"Component {self.name!r} not initialized")
        return self._state


class PrognosticComponent(Component):
    def __init__(
        self,
        name: str,
        model: Any,
        timestep: DeltaLike | None = None,
        imports: Iterable[str] = (),
        exports: Iterable[str] | None = None,
        import_adapter: ImportAdapter | None = None,
        next_input: NextInputFn | None = None,
        **kwargs: Any,
    ):
        self.model = model
        signature = model.input_coords()
        output_signature = model.output_coords(signature)
        if not isinstance(signature, xr.DataArray) or not isinstance(
            output_signature, xr.DataArray
        ):
            raise TypeError("Prognostic models must use the DataArray protocol")
        if timestep is None:
            timestep = (
                _dimension_values(output_signature, "lead_time")[-1]
                - _dimension_values(signature, "lead_time")[-1]
            ).astype("timedelta64[ns]")
        if exports is None:
            dictionary = kwargs.get("dictionary") or DEFAULT_DICTIONARY
            aliases = kwargs.get("variable_aliases") or {}
            exports = [
                aliases.get(str(raw), dictionary.standard_name(str(raw)))
                for raw in output_signature.coords["variable"].values
                if str(raw) in aliases or str(raw) in dictionary
            ]
        kwargs.setdefault("grid", _safe_infer_grid(signature))
        super().__init__(name, timestep, imports, exports, **kwargs)
        self.import_adapter = import_adapter or VariableOverwriteAdapter()
        self.next_input = next_input or self._default_next_input
        self._state: xr.DataArray | None = None

    def _default_next_input(
        self, previous: xr.DataArray, output: xr.DataArray
    ) -> xr.DataArray:
        input_lead = _dimension_values(self.model.input_coords(), "lead_time")
        if "lead_time" not in output.dims or output.sizes["lead_time"] != len(
            input_lead
        ):
            raise CouplingError(
                f"Component {self.name!r}: model output cannot rebuild its "
                "input window; supply next_input="
            )
        return output.assign_coords(lead_time=input_lead)

    def initialize(self, state: xr.DataArray | None = None) -> None:
        if state is None:
            raise CouplingError(f"Component {self.name!r} needs an initial condition")
        handshake_dataarray(state, self.model.input_coords())
        self._state = state
        self.publish(state, self.clock.start if self.clock is not None else None)

    def run(self, time: np.datetime64) -> None:
        if self._state is None:
            raise CouplingError(f"Component {self.name!r} not initialized")
        output = self.import_adapter(self.model, self._exchange(self._state, time))
        self._state = self.next_input(self._state, output)
        self.publish(output, time)
        self.run_count += 1

    def to(self, device: Any) -> "PrognosticComponent":
        self.model = self.model.to(device)
        return self

    @property
    def state(self) -> xr.DataArray:
        if self._state is None:
            raise CouplingError(f"Component {self.name!r} not initialized")
        return self._state


class DataComponent(Component):
    requires_ic = False

    def __init__(
        self,
        name: str,
        source: Any,
        exports: Iterable[str],
        timestep: DeltaLike,
        variable_map: Mapping[str, str] | None = None,
        target_grid: str | GridDefinition | None = None,
        device: Any = "cpu",
        **kwargs: Any,
    ):
        super().__init__(
            name,
            timestep,
            imports=(),
            exports=exports,
            grid=target_grid,
            **kwargs,
        )
        self.source = source
        self.device = device
        self._variable_map = {
            self.dictionary.standard_name(standard): raw
            for standard, raw in (variable_map or {}).items()
        }
        self._state: xr.DataArray | None = None

    def _raw_name(self, standard_name: str) -> str:
        if standard_name in self._variable_map:
            return self._variable_map[standard_name]
        if standard_name in self._std_to_raw:
            return self._std_to_raw[standard_name]
        entry = self.dictionary.resolve(standard_name)
        return min(entry.aliases) if entry.aliases else standard_name

    def _fetch(self, time: np.datetime64) -> xr.DataArray:
        from earth2studio.data import fetch_data

        return fetch_data(
            self.source,
            time=np.array([as_datetime(time)]),
            variable=np.array([self._raw_name(name) for name in self.export_names]),
            device=self.device,
            target_grid=self.grid,
        )

    def initialize(self, state: xr.DataArray | None = None) -> None:
        if state is None:
            if self.clock is None:
                raise CouplingError(
                    f"DataComponent {self.name!r} must be realized before fetching"
                )
            state = self._fetch(self.clock.start)
        self._state = state
        self.publish(state, self.clock.start if self.clock is not None else None)

    def run(self, time: np.datetime64) -> None:
        self._state = self._fetch(time)
        self.publish(self._state, as_datetime(time))
        self.run_count += 1


class DiagnosticComponent(Component):
    requires_ic = False

    def __init__(
        self,
        name: str,
        model: Any,
        timestep: DeltaLike,
        imports: Iterable[str] | None = None,
        exports: Iterable[str] | None = None,
        **kwargs: Any,
    ):
        self.model = model
        signature = model.input_coords()
        output_signature = model.output_coords(signature)
        if not isinstance(signature, xr.DataArray) or not isinstance(
            output_signature, xr.DataArray
        ):
            raise TypeError("Diagnostic models must use the DataArray protocol")
        self._input_raw = [str(value) for value in signature.coords["variable"].values]
        dictionary = kwargs.get("dictionary") or DEFAULT_DICTIONARY
        aliases = kwargs.get("variable_aliases") or {}
        if imports is None:
            imports = [
                aliases.get(raw, dictionary.standard_name(raw))
                for raw in self._input_raw
            ]
        if exports is None:
            exports = [
                aliases.get(str(raw), dictionary.standard_name(str(raw)))
                for raw in output_signature.coords["variable"].values
                if str(raw) in aliases or str(raw) in dictionary
            ]
        kwargs.setdefault("grid", _safe_infer_grid(signature))
        super().__init__(name, timestep, imports, exports, **kwargs)

    def initialize(self, state: xr.DataArray | None = None) -> None:
        if state is not None:
            output = self.model(state)
            self.publish(output, self.clock.start if self.clock is not None else None)

    def run(self, time: np.datetime64) -> None:
        missing = [name for name in self.import_names if name not in self.import_state]
        if missing:
            raise CouplingError(
                f"DiagnosticComponent {self.name!r} is missing imports {missing}"
            )
        standards = [self.dictionary.standard_name(raw) for raw in self._input_raw]
        array = self.import_state.stack(standards).assign_coords(
            variable=np.asarray(self._input_raw)
        )
        array = _conform_to_signature(
            array, self.model.input_coords(), as_datetime(time)
        )
        output = self.model(array)
        self.publish(output, as_datetime(time))
        self.run_count += 1

    def to(self, device: Any) -> "DiagnosticComponent":
        self.model = self.model.to(device)
        return self


def _safe_infer_grid(array: xr.DataArray) -> GridDefinition | None:
    try:
        return infer_grid(array)
    except ValueError:
        return None
