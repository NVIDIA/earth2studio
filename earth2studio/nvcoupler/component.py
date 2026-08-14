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

"""Components: the NUOPC_Model analog wrapping steppable things.

A Component owns an internal model state, an import State (fields other
components provide) and an export State (fields it offers), and a timestep
defining its cadence. The Driver calls the NUOPC-style phases:
advertise -> realize(clock) -> initialize(x, coords) -> run(time)* -> finalize.

The critical seam is the :class:`ImportAdapter`: real models receive coupled
fields in different call shapes (state-variable overwrite, conditioning
kwarg, extra input tensor), so the adapter — not the component — owns the
model invocation. The adapter receives everything about the step bundled in
an :class:`Exchange`.
"""

import abc
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from earth2studio.utils.type import CoordSystem

from .clock import Clock, DeltaLike, as_datetime, as_timedelta, is_multiple
from .dictionary import DEFAULT_DICTIONARY, FieldDictionary
from .errors import CadenceError, CouplingError
from .field import State

StepFn = Callable[[torch.Tensor, CoordSystem], tuple[torch.Tensor, CoordSystem]]
NextInputFn = Callable[
    [torch.Tensor, CoordSystem, torch.Tensor, CoordSystem],
    tuple[torch.Tensor, CoordSystem],
]


# ---------------------------------------------------------------------------
# Exchange and import adapters
# ---------------------------------------------------------------------------
def _broadcast_to_slice(data: torch.Tensor, slice_shape: torch.Size) -> torch.Tensor:
    """Left-pad `data` with singleton dims and expand to `slice_shape`."""
    if data.ndim > len(slice_shape):
        raise CouplingError(
            f"Imported field with shape {tuple(data.shape)} has more dims than "
            f"the model state slice {tuple(slice_shape)}"
        )
    data = data.reshape((1,) * (len(slice_shape) - data.ndim) + tuple(data.shape))
    return data.expand(slice_shape)


def _stacking_order(field_order: list[str] | None, imports: State, who: str) -> list[str]:
    """Resolve the channel order for stacking imported fields.

    Models are channel-order-sensitive; silently stacking in alphabetical
    order would feed permuted inputs that run fine and predict garbage. With
    more than one import the order must therefore be explicit.
    """
    if field_order is not None:
        missing = [n for n in field_order if n not in imports]
        if missing:
            raise CouplingError(
                f"{who}: field_order names {missing} are not in the "
                f"import state (present: {sorted(imports)})"
            )
        return list(field_order)
    if len(imports) > 1:
        raise CouplingError(
            f"{who}: {len(imports)} imported fields but no field_order= "
            "given — the model's channel order cannot be inferred. Pass "
            f"field_order=[...] with an explicit ordering of {sorted(imports)}"
        )
    return list(imports)


@dataclass(frozen=True)
class Exchange:
    """Everything an :class:`ImportAdapter` needs for one coupled model step.

    Attributes
    ----------
    x : torch.Tensor
        The component's current model state tensor.
    coords : CoordSystem
        Coordinates of ``x``.
    imports : State
        The imported Fields available this step (already subset to the
        component's advertised imports).
    std_to_raw : Mapping[str, str]
        Standard field name -> this model's raw variable name.
    time : np.datetime64 | None
        The valid time the step advances to.

    The two accessors cover the common delivery shapes: :meth:`inject` for
    state-variable overwrite and :meth:`stacked` for channel-stacked forcing
    tensors. Both are pure torch and autograd-safe.
    """

    x: torch.Tensor
    coords: CoordSystem
    imports: State
    std_to_raw: Mapping[str, str] = dc_field(default_factory=dict)
    time: np.datetime64 | None = None

    def inject(self) -> torch.Tensor:
        """Return ``x`` with each imported field overwritten into its matching
        variable slice (resolved through ``std_to_raw``).

        The overwrite composes a cloned tensor via index_copy_ on the clone,
        which keeps the autograd graph intact.
        """
        x, imports = self.x, self.imports
        if not imports:
            return x
        if "variable" not in self.coords:
            raise CouplingError(
                "Exchange.inject requires a 'variable' dim in the model state "
                "coords; use ConditioningKwargAdapter or ExtraTensorAdapter "
                "for models without one"
            )
        var_axis = list(self.coords).index("variable")
        variables = self.coords["variable"]
        x = x.clone()
        for std_name, field in imports.items():
            raw = self.std_to_raw.get(std_name, std_name)
            pos = np.flatnonzero(variables == raw)
            if pos.size == 0:
                raise CouplingError(
                    f"Imported field {std_name!r} (model name {raw!r}) is not a "
                    f"state variable of the model (variables: {list(variables)}). "
                    "If the model takes forcing as a conditioning kwarg or an "
                    "extra tensor, pass the matching ImportAdapter."
                )
            slice_shape = x.select(var_axis, int(pos[0])).shape
            data = _broadcast_to_slice(field.data, slice_shape).unsqueeze(var_axis)
            index = torch.tensor([int(pos[0])], device=x.device)
            x.index_copy_(var_axis, index, data.to(dtype=x.dtype, device=x.device))
        return x

    def stacked(
        self, field_order: list[str] | None = None, *, who: str = "Exchange.stacked"
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Stack the imported fields into one tensor with a leading
        'variable' dim.

        ``field_order`` fixes the channel order; with more than one import it
        is required (see :func:`_stacking_order`). ``who`` names the caller in
        error messages.
        """
        names = _stacking_order(field_order, self.imports, who)
        return self.imports.as_tensor(names)


@runtime_checkable
class ImportAdapter(Protocol):
    """Runs one model step with the import State injected.

    The adapter owns the model call because coupled models disagree on how
    forcing arrives: DLESyM/PhysicsNeMo expect an extra input tensor,
    StormScope expects a conditioning kwarg, and prescribed-forcing setups
    overwrite state variables. Implementations must be autograd-safe (no
    in-place mutation of tensors that may carry grad).
    """

    def __call__(
        self, model: Any, exchange: Exchange
    ) -> tuple[torch.Tensor, CoordSystem]: ...


class VariableOverwriteAdapter:
    """Default adapter: overwrite matching variable slices in x, then call
    ``model(x, coords)``.

    Suits toys and prescribed-forcing-as-state setups where imported fields
    are also state variables of the model (e.g. an atmosphere carrying sst
    as an input channel). See :meth:`Exchange.inject`.
    """

    def __call__(
        self, model: Any, exchange: Exchange
    ) -> tuple[torch.Tensor, CoordSystem]:
        return model(exchange.inject(), exchange.coords)


class ConditioningKwargAdapter:
    """Pass imports as a conditioning tensor kwarg (StormScope pattern).

    Calls ``model.call_with_conditioning(x, coords, conditioning=...,
    conditioning_coords=...)`` with the stacked import fields.
    """

    def __init__(
        self,
        field_order: list[str] | None = None,
        method: str = "call_with_conditioning",
    ):
        self.field_order = field_order
        self.method = method

    def __call__(
        self, model: Any, exchange: Exchange
    ) -> tuple[torch.Tensor, CoordSystem]:
        conditioning, conditioning_coords = exchange.stacked(
            self.field_order, who=type(self).__name__
        )
        fn = getattr(model, self.method)
        return fn(
            exchange.x,
            exchange.coords,
            conditioning=conditioning,
            conditioning_coords=conditioning_coords,
        )


class ExtraTensorAdapter:
    """Pass imports as an extra positional tensor (DLESyM / PhysicsNeMo
    4-tensor pattern): ``model(x, coords, coupling)`` by default.
    """

    def __init__(self, field_order: list[str] | None = None, kwarg: str | None = None):
        self.field_order = field_order
        self.kwarg = kwarg

    def __call__(
        self, model: Any, exchange: Exchange
    ) -> tuple[torch.Tensor, CoordSystem]:
        coupling, _ = exchange.stacked(self.field_order, who=type(self).__name__)
        if self.kwarg is not None:
            return model(exchange.x, exchange.coords, **{self.kwarg: coupling})
        return model(exchange.x, exchange.coords, coupling)


# ---------------------------------------------------------------------------
# Component base
# ---------------------------------------------------------------------------
class Component(abc.ABC):
    """NUOPC_Model analog: a steppable participant in the coupled system."""

    # Whether initialize() needs an (x, coords) initial condition. Subclasses
    # whose initialize() is safely callable with no arguments (mediators,
    # data components, diagnostics) set this False so the Driver can
    # initialize them without an ics entry.
    requires_ic: bool = True

    def __init__(
        self,
        name: str,
        timestep: DeltaLike,
        imports: Iterable[str] = (),
        exports: Iterable[str] = (),
        dictionary: FieldDictionary | None = None,
        variable_aliases: Mapping[str, str] | None = None,
        export_masks: Mapping[str, torch.Tensor] | None = None,
        import_vertical: Mapping[str, Any] | None = None,
        export_vertical: Mapping[str, Any] | None = None,
    ):
        self.name = name
        self.timestep = as_timedelta(timestep)
        self.dictionary = FieldDictionary(dictionary or DEFAULT_DICTIONARY)
        # variable_aliases: raw model variable name -> standard name
        self._raw_to_std: dict[str, str] = dict(variable_aliases or {})
        for raw, std in self._raw_to_std.items():
            if raw not in self.dictionary:
                self.dictionary.add_alias(std, raw)
        self._std_to_raw = {std: raw for raw, std in self._raw_to_std.items()}
        self.import_names = [self.dictionary.standard_name(n) for n in imports]
        self.export_names = [self.dictionary.standard_name(n) for n in exports]
        self.export_masks = dict(export_masks or {})
        # std name -> VerticalCoordinate this component expects imports on /
        # publishes exports on (only fields with a "level" dim need these)
        self.import_vertical: dict[str, Any] = dict(import_vertical or {})
        self.export_vertical: dict[str, Any] = dict(export_vertical or {})
        self.import_state = State(f"{name}.imports")
        self.export_state = State(f"{name}.exports")
        self.clock: Clock | None = None
        self.run_count = 0

    # -- NUOPC phases ---------------------------------------------------------
    def advertise(self) -> tuple[list[str], list[str]]:
        return list(self.import_names), list(self.export_names)

    def realize(self, clock: Clock) -> None:
        if not is_multiple(self.timestep, clock.dt):
            raise CadenceError(
                f"Component {self.name!r} timestep", str(self.timestep), str(clock.dt)
            )
        self.clock = clock

    @abc.abstractmethod
    def initialize(self, x: torch.Tensor, coords: CoordSystem) -> None:
        """Set internal state from an initial condition and seed export_state
        (so lagged coupling has data at t0)."""

    @abc.abstractmethod
    def run(self, time: np.datetime64) -> None:
        """Advance one component timestep; exports become valid at `time`."""

    def finalize(self) -> None:
        pass

    # -- helpers ---------------------------------------------------------------
    def _exchange(
        self, x: torch.Tensor, coords: CoordSystem, time: np.datetime64
    ) -> Exchange:
        """Bundle the current step's state and imports for an ImportAdapter."""
        imports = self.import_state.subset(
            [n for n in self.import_names if n in self.import_state]
        )
        return Exchange(x, coords, imports, self.resolve_std_to_raw(coords), time)

    def grid_coords(self) -> CoordSystem | None:
        """Spatial coordinates of this component's grid (None = no grid of
        its own, e.g. mediators — connectors then pass fields through)."""
        coords = getattr(self, "_coords", None)
        if coords is None:
            return None
        from .field import _SPATIAL_DIMS  # local import to avoid cycle at module load

        spatial = OrderedDict((k, v) for k, v in coords.items() if k in _SPATIAL_DIMS)
        return spatial or None

    def resolve_std_to_raw(self, coords: CoordSystem) -> dict[str, str]:
        """Map standard names to this model's raw variable names, derived from
        the actual variable coordinate plus any explicit variable_aliases."""
        mapping: dict[str, str] = {}
        for raw in coords.get("variable", ()):  # type: ignore[union-attr]
            raw = str(raw)
            if raw in self.dictionary:
                mapping[self.dictionary.standard_name(raw)] = raw
        mapping.update(self._std_to_raw)
        return mapping

    def publish(
        self, x: torch.Tensor, coords: CoordSystem, valid_time: np.datetime64
    ) -> None:
        """Populate export_state from a model output tensor."""
        state = State.from_tensor(
            f"{self.name}.exports",
            x,
            coords,
            self.dictionary,
            valid_time=valid_time,
            source=self.name,
            strict=False,
        )
        for std_name in self.export_names:
            if std_name not in state:
                raise CouplingError(
                    f"Component {self.name!r} advertises export {std_name!r} but "
                    f"its output variables are {list(coords.get('variable', []))}"
                )
            field = state[std_name]
            if std_name in self.export_masks:
                field.mask = self.export_masks[std_name]
            if std_name in self.export_vertical:
                field.vertical = self.export_vertical[std_name]
            self.export_state.add(field)

    def __repr__(self) -> str:
        from .clock import fmt_timedelta

        return (
            f"{type(self).__name__}({self.name!r}, dt={fmt_timedelta(self.timestep)}, "
            f"imports={self.import_names}, exports={self.export_names})"
        )


# ---------------------------------------------------------------------------
# Concrete components
# ---------------------------------------------------------------------------
class CallableComponent(Component):
    """Wraps a plain ``fn(x, coords) -> (x, coords)`` step function.

    The workhorse for synthetic components and non-ML models (any Python
    process model can join the coupled system through this class).
    """

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
        self.import_adapter: ImportAdapter = (
            import_adapter or VariableOverwriteAdapter()
        )
        self._x: torch.Tensor | None = None
        self._coords: CoordSystem | None = None

    def initialize(self, x: torch.Tensor, coords: CoordSystem) -> None:
        self._x, self._coords = x, OrderedDict(coords)
        start = self.clock.start if self.clock is not None else None
        self.publish(x, self._coords, valid_time=start)

    def run(self, time: np.datetime64) -> None:
        if self._x is None:
            raise CouplingError(f"Component {self.name!r} not initialized")
        y, ycoords = self.import_adapter(
            self.fn, self._exchange(self._x, self._coords, time)
        )
        self._x, self._coords = y, OrderedDict(ycoords)
        self.publish(y, ycoords, valid_time=time)
        self.run_count += 1

    @property
    def state(self) -> tuple[torch.Tensor, CoordSystem]:
        return self._x, self._coords


class PrognosticComponent(Component):
    """Wraps an earth2studio PrognosticModel (``models/px/base.py``).

    Owns the model state tensor and steps by calling the model directly
    (rather than ``create_iterator``) so imports can be injected between
    steps. Models that manage multi-window inputs internally need a
    ``next_input`` hook mapping (prev_x, prev_coords, out, out_coords) to the
    next step's input; the default handles single-window models by reusing
    the output with the model's input lead_time coordinates.
    """

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
        in_coords = model.input_coords()
        out_coords = model.output_coords(in_coords)
        if timestep is None:
            timestep = (
                out_coords["lead_time"][-1] - in_coords["lead_time"][-1]
            ).astype("timedelta64[ns]")
        if exports is None:
            exports = []
            dictionary = kwargs.get("dictionary") or DEFAULT_DICTIONARY
            aliases = kwargs.get("variable_aliases") or {}
            for raw in out_coords["variable"]:
                raw = str(raw)
                if raw in aliases:
                    exports.append(aliases[raw])
                elif raw in dictionary:
                    exports.append(dictionary.standard_name(raw))
        super().__init__(name, timestep, imports, exports, **kwargs)
        self.import_adapter: ImportAdapter = (
            import_adapter or VariableOverwriteAdapter()
        )
        self.next_input = next_input or self._default_next_input
        self._x: torch.Tensor | None = None
        self._coords: CoordSystem | None = None

    def _default_next_input(
        self,
        prev_x: torch.Tensor,
        prev_coords: CoordSystem,
        out: torch.Tensor,
        out_coords: CoordSystem,
    ) -> tuple[torch.Tensor, CoordSystem]:
        in_lead = self.model.input_coords()["lead_time"]
        out_lead = out_coords.get("lead_time", np.empty(0))
        if len(out_lead) != len(in_lead):
            raise CouplingError(
                f"Component {self.name!r}: model outputs {len(out_lead)} lead "
                f"times but takes {len(in_lead)} as input — supply a "
                "next_input hook to manage the sliding input window"
            )
        coords = OrderedDict(out_coords)
        coords["lead_time"] = in_lead.copy()
        return out, coords

    def initialize(self, x: torch.Tensor, coords: CoordSystem) -> None:
        self._x, self._coords = x, OrderedDict(coords)
        start = self.clock.start if self.clock is not None else None
        self.publish(x, self._coords, valid_time=start)

    def run(self, time: np.datetime64) -> None:
        if self._x is None:
            raise CouplingError(f"Component {self.name!r} not initialized")
        y, ycoords = self.import_adapter(
            self.model, self._exchange(self._x, self._coords, time)
        )
        self._x, self._coords = self.next_input(self._x, self._coords, y, ycoords)
        self.publish(y, ycoords, valid_time=time)
        self.run_count += 1

    def publish(
        self, x: torch.Tensor, coords: CoordSystem, valid_time: np.datetime64
    ) -> None:
        """Publish exchange-shaped exports: singleton batch/time/lead_time
        dims (and their stale size-1 coord values, e.g. the model's 'time')
        are squeezed away so exported Fields carry plain spatial coords like
        every other component's. The internal model state (`self._x`,
        `self._coords`) keeps the full model dims."""
        x, coords = _squeeze_singletons(x, coords)
        super().publish(x, coords, valid_time)

    def to(self, device: Any) -> "PrognosticComponent":
        self.model = self.model.to(device)
        if self._x is not None:
            self._x = self._x.to(device)
        return self

    @property
    def state(self) -> tuple[torch.Tensor, CoordSystem]:
        return self._x, self._coords


def _squeeze_singletons(
    x: torch.Tensor,
    coords: CoordSystem,
    dims: tuple[str, ...] = ("batch", "time", "lead_time"),
) -> tuple[torch.Tensor, CoordSystem]:
    """Drop size-1 batch/time/lead_time dims so published Fields carry the
    plain spatial coords (lat, lon) the rest of the coupler expects."""
    out = OrderedDict(coords)
    keys = list(out)
    for axis in range(len(keys) - 1, -1, -1):
        key = keys[axis]
        if key in dims and x.shape[axis] == 1:
            x = x.squeeze(axis)
            del out[key]
    return x, out


class DataComponent(Component):
    """Prescribed-forcing component wrapping an earth2studio DataSource.

    The NUOPC "data component" analog: instead of stepping a model it
    fetches fields from a data source (ERA5, GFS, an OISST archive, ...) at
    its own cadence and publishes them as exports. Swapping a modeled ocean
    for ``DataComponent("ocean", source=wb2, exports=
    ["sea_surface_temperature"], timestep="24h")`` turns a two-way coupled
    system into a prescribed-SST run with no other changes — the connectors,
    mediators, and run sequence are untouched.

    Parameters
    ----------
    name : str
    source : DataSource
        earth2studio data source; called through
        :func:`earth2studio.data.utils.fetch_data`.
    exports : Iterable[str]
        Standard names (or aliases) to fetch and export.
    timestep : DeltaLike
        Fetch cadence (e.g. "24h" for daily analysis fields).
    variable_map : Mapping[str, str], optional
        standard name -> raw source variable name, for sources whose
        vocabulary is not in the field dictionary (the raw name is also
        registered as an alias so exports resolve). Exports without an entry
        fall back to the dictionary's aliases.
    interp_to : CoordSystem, optional
        Forwarded to fetch_data for source-side regridding; usually left
        None so the Connector regrids onto each destination grid instead.
    device : torch.device | str
        Device fetched tensors are loaded to, by default "cpu".
    """

    # initialize() with no arguments fetches at clock.start — no IC needed
    requires_ic = False

    def __init__(
        self,
        name: str,
        source: Any,
        exports: Iterable[str],
        timestep: DeltaLike,
        variable_map: Mapping[str, str] | None = None,
        interp_to: CoordSystem | None = None,
        device: Any = "cpu",
        **kwargs: Any,
    ):
        super().__init__(name, timestep, imports=(), exports=exports, **kwargs)
        self.source = source
        self.interp_to = interp_to
        self.device = device
        self._variable_map: dict[str, str] = {}
        for std, raw in (variable_map or {}).items():
            std = self.dictionary.standard_name(std)
            self._variable_map[std] = raw
            if raw not in self.dictionary:
                self.dictionary.add_alias(std, raw)
        self._coords: CoordSystem | None = None

    def _raw_name(self, std_name: str) -> str:
        """Source variable name for a standard name: variable_map first, then
        explicit variable_aliases, then the dictionary's aliases."""
        if std_name in self._variable_map:
            return self._variable_map[std_name]
        if std_name in self._std_to_raw:
            return self._std_to_raw[std_name]
        entry = self.dictionary.resolve(std_name)
        return min(entry.aliases) if entry.aliases else std_name

    def _fetch(self, time: np.datetime64) -> tuple[torch.Tensor, CoordSystem]:
        # Local import: keeps nvcoupler importable without pulling the whole
        # data-source dependency stack until a DataComponent actually runs.
        from earth2studio.data.utils import fetch_data

        raw_names = [self._raw_name(n) for n in self.export_names]
        x, coords = fetch_data(
            self.source,
            time=np.array([as_datetime(time)]),
            variable=np.array(raw_names),
            device=self.device,
            interp_to=self.interp_to,
        )
        return _squeeze_singletons(x, coords)

    def initialize(
        self, x: torch.Tensor | None = None, coords: CoordSystem | None = None
    ) -> None:
        """Seed exports at the clock start (lagged coupling needs t0 data).

        Needs no initial condition: with ``x``/``coords`` omitted the source
        is queried at ``clock.start``. An explicit (x, coords) pair — with a
        "variable" dim — is published as-is instead (e.g. to avoid a fetch
        in tests or restarts).
        """
        if x is not None and coords is not None:
            data, dcoords = x, OrderedDict(coords)
        else:
            if self.clock is None:
                raise CouplingError(
                    f"DataComponent {self.name!r} cannot fetch initial data "
                    "before realize(clock) — the driver calls realize first, "
                    "or pass an explicit (x, coords) initial condition"
                )
            data, dcoords = self._fetch(self.clock.start)
        self._coords = dcoords
        start = self.clock.start if self.clock is not None else None
        self.publish(data, dcoords, valid_time=start)

    def run(self, time: np.datetime64) -> None:
        x, coords = self._fetch(time)
        self._coords = coords
        self.publish(x, coords, valid_time=as_datetime(time))
        self.run_count += 1


class DiagnosticComponent(Component):
    """Wraps an earth2studio DiagnosticModel (``models/dx/base.py``).

    A stateless single-step transform: each run it stacks its imported
    Fields into the model's expected variable order, calls
    ``model(x, coords)``, and publishes the outputs — no internal time
    state. Import/export lists default to the model's own
    ``input_coords()``/``output_coords()`` variables resolved through the
    field dictionary, so registered diagnostics wire up with just a name,
    the model, and a cadence.
    """

    # stateless transform: no-arg initialize() derives the grid from the
    # model's input_coords(), so no IC tensor is needed
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
        in_coords = model.input_coords()
        out_coords = model.output_coords(in_coords)
        self._input_raw = [str(v) for v in in_coords["variable"]]
        dictionary = kwargs.get("dictionary") or DEFAULT_DICTIONARY
        aliases = dict(kwargs.get("variable_aliases") or {})
        if imports is None:
            # every model input must resolve — a missing entry means the
            # coupler cannot know what to wire in, so resolve() raises with
            # suggestions rather than silently dropping the variable
            imports = [
                aliases[raw] if raw in aliases else dictionary.standard_name(raw)
                for raw in self._input_raw
            ]
        if exports is None:
            exports = []
            for raw in out_coords["variable"]:
                raw = str(raw)
                if raw in aliases:
                    exports.append(aliases[raw])
                elif raw in dictionary:
                    exports.append(dictionary.standard_name(raw))
        super().__init__(name, timestep, imports, exports, **kwargs)
        self._coords: CoordSystem | None = OrderedDict(in_coords)

    def initialize(
        self, x: torch.Tensor | None = None, coords: CoordSystem | None = None
    ) -> None:
        """No state tensor to set; records the model grid for grid_coords().

        An optional (x, coords) input — in the model's own vocabulary — is
        pushed through the model once to seed exports at t0 (lagged chains).
        """
        self._coords = OrderedDict(self.model.input_coords())
        if x is not None and coords is not None:
            y, ycoords = self.model(x, OrderedDict(coords))
            y, ycoords = _squeeze_singletons(y, ycoords)
            start = self.clock.start if self.clock is not None else None
            self.publish(y, ycoords, valid_time=start)

    def _conform_to_input(
        self, x: torch.Tensor, coords: CoordSystem, time: np.datetime64
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Add singleton dims (batch, time, lead_time, ...) the model's
        input_coords declare but the stacked import Fields lack."""
        in_coords = self.model.input_coords()
        out: CoordSystem = OrderedDict()
        for key, value in in_coords.items():
            if key in coords:
                out[key] = coords[key]
                continue
            x = x.unsqueeze(len(out))
            if key == "time":
                out[key] = np.array([as_datetime(time)])
            elif key == "lead_time":
                out[key] = np.array([np.timedelta64(0, "h")], dtype="timedelta64[ns]")
            else:
                out[key] = np.asarray(value)  # e.g. batch: np.empty(0)
        for key, value in coords.items():
            if key not in out:
                out[key] = value
        return x, out

    def run(self, time: np.datetime64) -> None:
        missing = [n for n in self.import_names if n not in self.import_state]
        if missing:
            raise CouplingError(
                f"DiagnosticComponent {self.name!r} is missing imports "
                f"{missing} at {time} (present: {sorted(self.import_state)}) — "
                "check the run sequence connects its source before this "
                "component runs"
            )
        # stack imports in the model's raw variable order, then relabel the
        # variable coordinate back to the model's vocabulary
        std_order = [self.dictionary.standard_name(raw) for raw in self._input_raw]
        x, coords = self.import_state.as_tensor(std_order)
        coords = OrderedDict(coords)
        coords["variable"] = np.array(self._input_raw)
        x, coords = self._conform_to_input(x, coords, time)
        y, ycoords = self.model(x, coords)
        y, ycoords = _squeeze_singletons(y, ycoords)
        self.publish(y, ycoords, valid_time=as_datetime(time))
        self.run_count += 1

    def to(self, device: Any) -> "DiagnosticComponent":
        self.model = self.model.to(device)
        return self
