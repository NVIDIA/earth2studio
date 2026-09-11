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

"""DLESyM split adapter: one coupled model, two nvcoupler components.

The NUOPC "split a monolithic executable into gridded components" move.
:class:`earth2studio.models.px.DLESyM` internally couples an atmosphere and
an ocean HEALPix U-Net inside a single ``__call__``; :func:`split_dlesym`
re-exposes those sub-models as two :class:`~earth2studio.nvcoupler.component.Component`
instances so the exchange (SST down, windowed z1000/ws10m up) runs through
explicit Connectors on the coupler's clock instead of being hidden inside
``DLESyM._forward``.

Both components step the full parent cadence (96 h, one native ``__call__``)
and close over the parent DLESyM instance for normalization (``center`` /
``scale`` buffers), insolation (``_make_insolation_tensor``), the coupling
tensor construction (``_make_atmos_coupling`` / ``_make_ocean_coupling`` are
called directly, not re-implemented), constants and the precomputed
lead-time / variable index tables:

- **Atmos** runs first. It consumes the ``sea_surface_temperature`` import
  with DLESyM's persisted-at-t0 semantics (``_make_atmos_coupling``: the
  lead-0 SST repeated for every internal sub-step), produces the 16
  6-hourly atmos outputs, and exports its prognostic fields plus the
  ocean-coupling variables already chunk-averaged into the two 48 h ocean
  windows (the exact ``_make_ocean_coupling`` math) as derived
  window-mean fields.
- **Ocean** consumes those window means, rebuilds the DLESyM ocean coupling
  tensor (lead, batch, window-major variables, face, height, width), runs
  the ocean model to produce SST at 48 h / 96 h, and exports the 96 h SST —
  which the run sequence feeds back to the atmosphere *lagged* across steps,
  exactly like ``_next_step_inputs`` carrying the ocean output into the next
  window.

Exchange happens on the shared HEALPix (face, height, width) grid, so the
connectors are identity transfers.

All exchange-path tensor math is pure torch (normalize / chunk-mean /
re-normalize round-trips through physical units are exact up to float
rounding), so autograd survives the split.

Honesty note
------------
Execution against real DLESyM weights is **untested** — this module has only
been exercised against structural mocks. A weights-equivalence test (needs
the ``dlesym`` optional dependencies plus the NGC package) would assert, for
an n-step rollout from the same initial condition:

1. the atmos component's prognostic exports at each 96 h ring equal the
   final-lead slice of ``DLESyM.retrieve_valid_atmos_outputs`` from the
   native iterator;
2. the ocean component's SST export equals the 96 h slice of
   ``DLESyM.retrieve_valid_ocean_outputs``;
3. the coupling tensor delivered to ``ocean_model`` equals
   ``DLESyM._make_ocean_coupling(atmos_outputs)`` bit-for-bit modulo the
   denormalize/renormalize round trip;
4. the lagged SST the atmosphere sees at step k equals the native step k-1
   ocean output at 96 h.
"""

from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from earth2studio.utils.type import CoordSystem

from .clock import as_datetime, as_timedelta
from .component import Component, _broadcast_to_slice
from .dictionary import DEFAULT_DICTIONARY, CellMethod, FieldDictionary, FieldEntry
from .driver import Driver
from .errors import CouplingError
from .field import Field, State
from .sequence import ConnectAction, RunAction, RunSequence, Slot

# Module-level dictionary copy: the extension point for DLESyM vocabulary.
# The standard DLESyM-V1-ERA5 variables (z1000, ws10m, sst, tau300-700, ...)
# and the 48 h window-mean entries (geopotential_at_1000hpa_48h_mean,
# wind_speed_10m_48h_mean) are already in DEFAULT_DICTIONARY; split_dlesym
# auto-registers anything a non-default DLESyM config adds.
DLESYM_DICTIONARY = FieldDictionary(DEFAULT_DICTIONARY)

_HOUR_NS = 3_600_000_000_000


class _DLESyMSplitComponent(Component):
    """Shared machinery for the two halves of a split DLESyM."""

    def __init__(
        self,
        name: str,
        parent: Any,
        dictionary: FieldDictionary,
        imports: list[str],
        exports: list[str],
    ):
        step = as_timedelta(parent.atmos_output_times[-1])
        super().__init__(name, step, imports, exports, dictionary=dictionary)
        self.parent = parent
        self._x: torch.Tensor | None = None
        self._coords: CoordSystem | None = None
        self._times: np.ndarray | None = None
        self._batch: int = 1

    # -- shared helpers ------------------------------------------------------
    def _validate_ic(self, x: torch.Tensor, coords: CoordSystem) -> None:
        dims = list(coords)
        expected = ["batch", "time", "lead_time", "variable", "face", "height", "width"]
        if x.ndim != 7 or dims != expected:
            raise CouplingError(
                f"Component {self.name!r} expects the DLESyM input layout "
                f"{expected}, got dims {dims} for tensor of shape "
                f"{tuple(x.shape)}"
            )
        lead = coords["lead_time"]
        rel = lead - lead[-1]
        want = self.parent.full_input_times - self.parent.full_input_times[-1]
        if len(rel) != len(want) or not np.array_equal(
            rel.astype("timedelta64[ns]"), want.astype("timedelta64[ns]")
        ):
            raise CouplingError(
                f"Component {self.name!r}: initial condition lead_time window "
                f"{list(lead)} does not match DLESyM full_input_times "
                f"{list(self.parent.full_input_times)}"
            )

    def _anchor_times(self, time: np.datetime64) -> np.ndarray:
        """Absolute window-end times for insolation, one per (batch, time)
        element, matching DLESyM's anchor + lead_time[-1] arithmetic."""
        if self.clock is None or self._times is None:
            raise CouplingError(f"Component {self.name!r} not realized/initialized")
        offset = (as_datetime(time) - self.clock.start) - self.timestep
        anchor = self._times + offset
        if self._batch > 1:
            anchor = np.concatenate([anchor] * self._batch)
        return anchor

    def _publish_from_tensor(
        self,
        x: torch.Tensor,
        variables: np.ndarray,
        names: list[str],
        valid_time: np.datetime64 | None,
    ) -> None:
        """Publish instantaneous fields from a (batch, time, variable, face,
        height, width) tensor onto export_state."""
        if self._coords is None:
            raise CouplingError(f"Component {self.name!r} not initialized")
        coords: CoordSystem = OrderedDict(
            (k, v) for k, v in self._coords.items() if k != "lead_time"
        )
        coords["variable"] = np.asarray(variables)
        coords.move_to_end("variable", last=False)
        coords.move_to_end("time", last=False)
        coords.move_to_end("batch", last=False)
        state = State.from_tensor(
            f"{self.name}.publish",
            x,
            coords,
            self.dictionary,
            valid_time=valid_time,
            source=self.name,
            strict=False,
        )
        for std in names:
            if std not in state:
                raise CouplingError(
                    f"Component {self.name!r} advertises export {std!r} but the "
                    f"model output variables are {list(variables)}"
                )
            self.export_state.add(state[std])

    def _center_scale(
        self, var_idx: list[int], shape: tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-variable center/scale views aligned to a var axis of `shape`.

        Mirrors the normalization constants of ``DLESyM._normalize_input`` /
        ``_denormalize_output`` restricted to a variable subset; the parent
        methods cannot be called directly because they broadcast over the
        FULL variable dimension while the split components carry tensors
        holding only their own (atmos or ocean) variables.
        """
        c = self.parent.center.reshape(-1)[var_idx].view(shape)
        s = self.parent.scale.reshape(-1)[var_idx].view(shape)
        return c, s

    def run(self, time: np.datetime64) -> None:  # pragma: no cover - abstract-ish
        raise NotImplementedError


class DLESyMAtmosComponent(_DLESyMSplitComponent):
    """The atmosphere half of a split DLESyM.

    Imports ``sea_surface_temperature`` (persisted at the window end, the
    ``_make_atmos_coupling`` semantics), exports its prognostic fields at the
    96 h window end plus the ocean-coupling variables chunk-averaged into the
    ocean output windows (carrying a leading ``window`` dimension of length
    ``len(ocean_output_times)``).
    """

    def __init__(
        self,
        parent: Any,
        dictionary: FieldDictionary,
        derived_names: Mapping[str, str],
        name: str = "atmos",
    ):
        d = FieldDictionary(dictionary)
        imports = [d.standard_name(v) for v in parent.atmos_coupling_variables]
        self._prognostic_exports = [d.standard_name(v) for v in parent.atmos_variables]
        # raw ocean-coupling variable -> derived window-mean standard name
        self._derived = dict(derived_names)
        exports = self._prognostic_exports + list(self._derived.values())
        super().__init__(name, parent, d, imports, exports)
        # window-lead index of "now" (lead 0) in full_input_times
        self._zero_idx = parent.atmos_input_lt_idx[-1]
        # next-step window: index of (t + step) in atmos_output_times per
        # full_input_times entry (DLESyM's _next_step_inputs slice)
        out_t = parent.atmos_output_times.astype("timedelta64[ns]")
        step = as_timedelta(parent.atmos_output_times[-1])
        self._next_idx = []
        for t in parent.full_input_times.astype("timedelta64[ns]"):
            hits = np.flatnonzero(out_t == t + step)
            if hits.size == 0:
                raise CouplingError(
                    f"DLESyM atmos window time {t} + {step} is not an atmos "
                    f"output time {list(parent.atmos_output_times)} — the "
                    "input window cannot be rebuilt from one step's outputs"
                )
            self._next_idx.append(int(hits[0]))
        # position of each full-variable index within the atmos output var dim
        self._atmos_pos = {vi: k for k, vi in enumerate(parent.atmos_var_idx)}

    def initialize(self, x: torch.Tensor, coords: CoordSystem) -> None:
        self._validate_ic(x, coords)
        self._x = x
        self._coords = OrderedDict(coords)
        self._times = np.asarray(coords["time"], dtype="datetime64[ns]")
        self._batch = x.shape[0]
        start = self.clock.start if self.clock is not None else None
        inst = x[:, :, self._zero_idx]
        self._publish_from_tensor(
            inst, coords["variable"], self._prognostic_exports, start
        )

    def _inject_imports(self, x: torch.Tensor) -> torch.Tensor:
        """Overwrite the coupling variables at the persisted (lead-0)
        coupling indices with imported fields, non-destructively."""
        p = self.parent
        updates = []
        for raw, vi in zip(p.atmos_coupling_variables, p.atmos_coupling_var_idx):
            std = self.dictionary.standard_name(raw)
            if std in self.import_state:
                updates.append((vi, self.import_state[std]))
        if not updates:
            return x
        x = x.clone()
        for lead_idx in sorted(set(p.atmos_coupled_input_lt_idx)):
            for vi, field in updates:
                slot = x[:, :, lead_idx, vi]
                data = _broadcast_to_slice(
                    field.data.to(device=x.device, dtype=x.dtype), slot.shape
                )
                x[:, :, lead_idx, vi] = data
        return x

    def run(self, time: np.datetime64) -> None:
        if self._x is None or self._coords is None:
            raise CouplingError(f"Component {self.name!r} not initialized")
        ic_coords = self._coords
        p = self.parent
        x = self._inject_imports(self._x)
        b, t = x.shape[0], x.shape[1]

        # Mirrors DLESyM._normalize_input; not called directly because the
        # parent method assumes its center/scale buffers already live on the
        # input's device/dtype, while the split component moves them here.
        xn = (x - p.center.to(device=x.device, dtype=x.dtype)) / p.scale.to(
            device=x.device, dtype=x.dtype
        )
        xf = xn.reshape(-1, *xn.shape[2:])  # (B, lead, var, face, h, w)

        atmos_state = xf[:, p.atmos_input_lt_idx][
            ..., p.atmos_var_idx, :, :, :
        ].permute(0, 3, 1, 2, 4, 5)
        insolation = p._make_insolation_tensor(
            anchor_times=self._anchor_times(time), timedeltas=p.atmos_sol_times
        )
        # the parent's persisted-at-t0 coupling selection, verbatim
        coupling = p._make_atmos_coupling(xf, ic_coords)
        inputs = [
            y.to(device=xf.device, dtype=xf.dtype)
            for y in [atmos_state, insolation, p.atmos_constants, coupling]
        ]
        out = p.atmos_model(inputs)  # (B, face, n_lead, n_atmos_var, h, w), normalized

        # -- derived exports: the parent's own chunk-mean math, denormalized --
        # _make_ocean_coupling returns (lead=1, B, window-major variables,
        # face, h, w): window w, coupling var k lives at index w * C + k.
        n_windows = len(p.ocean_output_times)
        n_coupling = len(p.ocean_coupling_var_idx)
        mc = p._make_ocean_coupling(out, ic_coords)[0]
        window_coords: CoordSystem = OrderedDict(
            {
                "batch": np.arange(out.shape[0]),
                "window": np.arange(n_windows),
                "face": np.asarray(ic_coords["face"]).copy(),
                "height": np.asarray(ic_coords["height"]).copy(),
                "width": np.asarray(ic_coords["width"]).copy(),
            }
        )
        for k, (raw, vi) in enumerate(
            zip(p.ocean_coupling_variables, p.ocean_coupling_var_idx)
        ):
            c, s = self._center_scale([vi], (1,))
            mean_k = torch.stack(
                [mc[:, w * n_coupling + k] for w in range(n_windows)], dim=1
            )  # (B, window, face, h, w)
            data = mean_k * s.to(device=out.device, dtype=out.dtype) + c.to(
                device=out.device, dtype=out.dtype
            )
            derived = self._derived[raw]
            entry = self.dictionary.resolve(derived)
            self.export_state.add(
                Field(
                    data=data,
                    coords=OrderedDict(
                        (k2, v.copy()) for k2, v in window_coords.items()
                    ),
                    standard_name=derived,
                    units=entry.canonical_units,
                    valid_time=as_datetime(time),
                    source=self.name,
                )
            )

        # -- denormalize outputs and rebuild the sliding input window --------
        # mirrors DLESyM._denormalize_output restricted to atmos_var_idx
        # (the atmos output tensor lacks the ocean variables)
        c_a, s_a = self._center_scale(
            p.atmos_var_idx, (1, 1, 1, len(p.atmos_var_idx), 1, 1)
        )
        out_phys = out * s_a.to(device=out.device, dtype=out.dtype) + c_a.to(
            device=out.device, dtype=out.dtype
        )
        out_bt = out_phys.permute(0, 2, 3, 1, 4, 5).reshape(
            b, t, out.shape[2], out.shape[3], out.shape[1], *out.shape[-2:]
        )  # (b, t, lead, var_atmos, face, h, w)
        window = out_bt[:, :, self._next_idx]  # (b, t, n_window_lead, A, f, h, w)
        n_lead = window.shape[2]
        pieces = []
        for vi in range(x.shape[3]):
            if vi in self._atmos_pos:
                pieces.append(window[:, :, :, self._atmos_pos[vi]])
            else:
                # non-atmos variables (sst): persist the current lead-0 value;
                # it is refreshed from the SST import before the next step
                carry = x[:, :, self._zero_idx, vi].unsqueeze(2)
                pieces.append(carry.expand(b, t, n_lead, *carry.shape[-3:]))
        self._x = torch.stack(pieces, dim=3)

        # -- prognostic exports at the window end (96 h) -----------------------
        inst = out_bt[:, :, -1]  # (b, t, var_atmos, face, h, w)
        self._publish_from_tensor(
            inst,
            np.array(p.atmos_variables),
            self._prognostic_exports,
            as_datetime(time),
        )
        self.run_count += 1

    @property
    def state(self) -> tuple[torch.Tensor, CoordSystem]:
        return self._x, self._coords


class DLESyMOceanComponent(_DLESyMSplitComponent):
    """The ocean half of a split DLESyM.

    Imports the atmosphere's window-mean coupling fields, rebuilds the
    DLESyM ocean coupling tensor (lead=1, batch, window-major variables,
    face, height, width), and exports ``sea_surface_temperature`` valid at
    the 96 h window end.
    """

    def __init__(
        self,
        parent: Any,
        dictionary: FieldDictionary,
        derived_names: Mapping[str, str],
        name: str = "ocean",
    ):
        d = FieldDictionary(dictionary)
        self._derived = dict(derived_names)
        imports = [self._derived[v] for v in parent.ocean_coupling_variables]
        self._sst_exports = [d.standard_name(v) for v in parent.ocean_variables]
        super().__init__(name, parent, d, imports, self._sst_exports)
        step = as_timedelta(parent.atmos_output_times[-1])
        out_t = parent.ocean_output_times.astype("timedelta64[ns]")
        self._next_idx = []
        for t in parent.ocean_input_times.astype("timedelta64[ns]"):
            hits = np.flatnonzero(out_t == t + step)
            if hits.size == 0:
                raise CouplingError(
                    f"DLESyM ocean window time {t} + {step} is not an ocean "
                    f"output time {list(parent.ocean_output_times)} — the "
                    "input window cannot be rebuilt from one step's outputs"
                )
            self._next_idx.append(int(hits[0]))

    def initialize(self, x: torch.Tensor, coords: CoordSystem) -> None:
        self._validate_ic(x, coords)
        p = self.parent
        window = x[:, :, p.ocean_input_lt_idx][:, :, :, p.ocean_var_idx]
        self._x = window  # (b, t, n_ocean_lead, n_ocean_var, face, h, w), physical
        self._coords = OrderedDict(coords)
        self._times = np.asarray(coords["time"], dtype="datetime64[ns]")
        self._batch = x.shape[0]
        start = self.clock.start if self.clock is not None else None
        inst = window[:, :, -1]
        self._publish_from_tensor(
            inst, np.array(p.ocean_variables), self._sst_exports, start
        )

    def _build_coupling(self, ref: torch.Tensor) -> torch.Tensor:
        """Reassemble _make_ocean_coupling's tensor from imported window-mean
        fields: (lead=1, batch, window-major variables, face, h, w).

        Mirrors ``DLESyM._make_ocean_coupling``'s window-major layout but
        cannot call it: the parent method chunk-averages the atmos model's
        normalized output tensor, which never crosses the coupling seam —
        here only the already-averaged, physical-unit import Fields exist,
        so the tensor is rebuilt (renormalized) from those instead.
        """
        p = self.parent
        n_windows = len(p.ocean_output_times)
        per_var: list[torch.Tensor] = []
        for raw, vi in zip(p.ocean_coupling_variables, p.ocean_coupling_var_idx):
            derived = self._derived[raw]
            if derived not in self.import_state:
                raise CouplingError(
                    f"Component {self.name!r} needs import {derived!r} before it "
                    "can run — schedule the atmos component and the "
                    "atmos -> ocean connector earlier in the same slot"
                )
            field = self.import_state[derived]
            data = field.data.to(device=ref.device, dtype=ref.dtype)
            if data.ndim != 5 or data.shape[1] != n_windows:
                raise CouplingError(
                    f"Component {self.name!r}: import {derived!r} must have "
                    f"shape (batch, window={n_windows}, face, height, width), "
                    f"got {tuple(data.shape)}"
                )
            c, s = self._center_scale([vi], (1,))
            per_var.append(
                (data - c.to(device=ref.device, dtype=ref.dtype))
                / s.to(device=ref.device, dtype=ref.dtype)
            )
        blocks = [
            torch.stack([v[:, w] for v in per_var], dim=1) for w in range(n_windows)
        ]  # each (B, C, face, h, w)
        return torch.cat(blocks, dim=1).unsqueeze(0)

    def run(self, time: np.datetime64) -> None:
        if self._x is None:
            raise CouplingError(f"Component {self.name!r} not initialized")
        p = self.parent
        window = self._x
        b, t = window.shape[0], window.shape[1]

        # mirrors DLESyM._normalize_input restricted to ocean_var_idx (the
        # ocean window tensor lacks the atmos variables)
        c_o, s_o = self._center_scale(
            p.ocean_var_idx, (1, 1, 1, len(p.ocean_var_idx), 1, 1, 1)
        )
        xn = (window - c_o.to(device=window.device, dtype=window.dtype)) / s_o.to(
            device=window.device, dtype=window.dtype
        )
        xf = xn.reshape(-1, *xn.shape[2:])  # (B, lead, var, face, h, w)
        ocean_state = xf.permute(0, 3, 1, 2, 4, 5)
        insolation = p._make_insolation_tensor(
            anchor_times=self._anchor_times(time), timedeltas=p.ocean_sol_times
        )
        coupling = self._build_coupling(xf)
        inputs = [
            y.to(device=xf.device, dtype=xf.dtype)
            for y in [ocean_state, insolation, p.ocean_constants, coupling]
        ]
        out = p.ocean_model(inputs)  # (B, face, n_ocean_lead, n_ocean_var, h, w)

        # mirrors DLESyM._denormalize_output restricted to ocean_var_idx
        c2, s2 = self._center_scale(
            p.ocean_var_idx, (1, 1, 1, len(p.ocean_var_idx), 1, 1)
        )
        out_phys = out * s2.to(device=out.device, dtype=out.dtype) + c2.to(
            device=out.device, dtype=out.dtype
        )
        out_bt = out_phys.permute(0, 2, 3, 1, 4, 5).reshape(
            b, t, out.shape[2], out.shape[3], out.shape[1], *out.shape[-2:]
        )  # (b, t, lead, var, face, h, w)
        self._x = out_bt[:, :, self._next_idx]

        inst = out_bt[:, :, -1]  # SST at the 96 h window end
        self._publish_from_tensor(
            inst, np.array(p.ocean_variables), self._sst_exports, as_datetime(time)
        )
        self.run_count += 1

    @property
    def state(self) -> tuple[torch.Tensor, CoordSystem]:
        return self._x, self._coords


def split_dlesym(
    dlesym: Any, dictionary: FieldDictionary | None = None
) -> tuple[DLESyMAtmosComponent, DLESyMOceanComponent]:
    """Expose a DLESyM's atmos/ocean sub-models as two nvcoupler components.

    Parameters
    ----------
    dlesym : DLESyM
        A constructed :class:`earth2studio.models.px.DLESyM` (or any object
        exposing the same attributes: ``atmos_model`` / ``ocean_model``,
        ``center`` / ``scale`` / ``*_constants`` buffers, the ``*_variables``
        and ``*_input_times`` / ``*_output_times`` config, the precomputed
        ``*_lt_idx`` / ``*_var_idx`` index tables, ``_make_insolation_tensor``
        and the ``_make_atmos_coupling`` / ``_make_ocean_coupling`` methods).
    dictionary : FieldDictionary, optional
        Vocabulary to extend; defaults to a copy of
        :data:`DLESYM_DICTIONARY`. Unknown raw variables and missing
        window-mean entries are auto-registered on a private copy.

    Returns
    -------
    tuple[DLESyMAtmosComponent, DLESyMOceanComponent]
    """
    d = FieldDictionary(dictionary or DLESYM_DICTIONARY)
    for raw in list(dlesym.atmos_variables) + list(dlesym.ocean_variables):
        if raw not in d:
            d.register(FieldEntry(raw, "", f"DLESyM variable {raw!r}"))

    n_windows = len(dlesym.ocean_output_times)
    n_lead = len(dlesym.atmos_output_times)
    if n_windows == 0 or n_lead % n_windows != 0:
        raise CouplingError(
            f"DLESyM atmos output times ({n_lead}) do not chunk evenly into "
            f"{n_windows} ocean windows — cannot replicate _make_ocean_coupling"
        )
    step = as_timedelta(dlesym.atmos_output_times[-1])
    window_ns = step.astype(np.int64) // n_windows
    if window_ns % _HOUR_NS != 0:
        raise CouplingError(
            f"DLESyM ocean coupling window {window_ns} ns is not a whole "
            "number of hours; cannot name the derived window-mean fields"
        )
    window_h = int(window_ns // _HOUR_NS)
    window = np.timedelta64(window_h, "h")

    derived: dict[str, str] = {}
    for raw in dlesym.ocean_coupling_variables:
        std = d.standard_name(raw)
        name = f"{std}_{window_h}h_mean"
        if name not in d:
            base = d.resolve(std)
            d.register(
                FieldEntry(
                    name,
                    base.canonical_units,
                    f"trailing {window_h} h mean of {std} (DLESyM ocean coupling)",
                    frozenset(),
                    CellMethod(std, "mean", window),
                )
            )
        derived[raw] = name

    atmos = DLESyMAtmosComponent(dlesym, d, derived)
    ocean = DLESyMOceanComponent(dlesym, d, derived)
    return atmos, ocean


def build_dlesym_driver(
    dlesym: Any,
    start: Any,
    stop: Any,
    dictionary: FieldDictionary | None = None,
    collect: bool = True,
) -> Driver:
    """Wire a split DLESyM into a Driver matching the native internal loop.

    The run sequence reproduces ``DLESyM._forward`` + ``_next_step_inputs``
    ordering: within a step the atmosphere runs first and its window means
    flow to the ocean (sequential coupling); the ocean's SST flows to the
    atmosphere *before* it runs, i.e. lagged across steps::

        @96h
          ocean -> atmos     # lagged SST (previous step's 96 h output)
          atmos
          atmos -> ocean     # window-mean coupling, same step
          ocean
        @

    Initialize with the same DLESyM-layout initial condition for both halves::

        driver = build_dlesym_driver(model, "2024-01-01", "2024-01-09")
        driver.initialize({"atmos": (x, coords), "ocean": (x, coords)})
        driver.run()

    ``stop - start`` must be a multiple of the 96 h step.
    """
    atmos, ocean = split_dlesym(dlesym, dictionary)
    from .clock import Clock

    clock = Clock(start, stop, dt=atmos.timestep)
    sequence = RunSequence(
        [
            Slot(
                atmos.timestep,
                [
                    ConnectAction(ocean.name, atmos.name),
                    RunAction(atmos.name),
                    ConnectAction(atmos.name, ocean.name),
                    RunAction(ocean.name),
                ],
            )
        ]
    )
    return Driver(
        {atmos.name: atmos, ocean.name: ocean},
        sequence,
        clock,
        collect=collect,
    )
