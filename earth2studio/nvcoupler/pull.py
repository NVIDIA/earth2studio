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

"""Pull-pattern coupling: adapters for models that fetch their own forcing.

Some earth2studio models (StormCast is the canonical case) do not accept
coupled fields as call arguments — they *pull* them, calling
``fetch_data(self.conditioning_data_source, time, variables, ...)`` inside
their own ``__call__``. The only injection point such a model exposes is the
settable data-source attribute.

:class:`PullAdapter` exploits exactly that point, without wrapping or
modifying the model: before each step it installs a :class:`StateDataSource`
— a tiny in-memory object satisfying the DataSource protocol that answers
fetches from the component's import State. The model runs its unmodified
production code path (fetch → interpolate → concatenate) and believes it is
reading GFS; it is reading the coupler. This is the same masquerade the
existing serve workflows play with ``InferenceOutputSource``, minus the
store-and-replay staging: the "source" is this step's live exchange.

Honest limitation: the pull path goes through the model's own
``fetch_data``/xarray machinery, so field data crosses a numpy boundary —
pull-coupled components are inference-only (no gradients through the
exchange). Push-pattern adapters keep autograd intact.
"""

from collections import OrderedDict
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import xarray as xr

from earth2studio.utils.type import CoordSystem

from .errors import CouplingError
from .field import State

if TYPE_CHECKING:
    from .component import Exchange
    from .dictionary import FieldDictionary

# Fetch times farther than this from a served field's valid_time trigger a
# warning: the pull is likely misaligned with the exchange cadence.
DEFAULT_TIME_TOLERANCE = np.timedelta64(0, "h")


class StateDataSource:
    """A DataSource serving the current contents of an import State.

    Answers ``__call__(time, variable)`` with an ``xr.DataArray`` of dims
    ``(time, variable, lat, lon)`` built from the State's Fields. Variables
    may be requested by the model's raw names (resolved through the owning
    component's dictionary/aliases) or by standard name.

    The source is a snapshot view: whatever the connector last delivered is
    what every requested time receives. Cadence alignment is the run
    sequence's job — a sequential ``global -> stormcast`` connect before the
    pull guarantees the served fields are valid at the pulled time.
    """

    def __init__(
        self,
        state: State,
        raw_to_std: Mapping[str, str] | None = None,
        strict_time: bool = False,
        dictionary: "FieldDictionary | None" = None,
    ):
        from .dictionary import DEFAULT_DICTIONARY

        self.state = state
        self.raw_to_std = dict(raw_to_std or {})
        self.strict_time = strict_time
        self.dictionary = dictionary or DEFAULT_DICTIONARY

    def _resolve(self, name: str) -> str:
        if name in self.state:
            return name
        if name in self.raw_to_std and self.raw_to_std[name] in self.state:
            return self.raw_to_std[name]
        # pulled conditioning names are usually model-raw (u10m, t2m): the
        # Exchange map only covers state variables, so fall back to aliases
        if name in self.dictionary:
            std = self.dictionary.standard_name(name)
            if std in self.state:
                return std
        raise CouplingError(
            f"Pull-coupled model requested variable {name!r}, but the import "
            f"state holds {sorted(self.state)} (raw-name map: "
            f"{self.raw_to_std}). Add the field to the component's imports "
            "and wire a connector delivering it before the model runs."
        )

    def __call__(self, time: Any, variable: Any) -> xr.DataArray:
        times = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        variables = np.atleast_1d(np.asarray(variable))
        fields = [self.state[self._resolve(str(v))] for v in variables]

        grid = fields[0].coords
        if list(grid.keys()) != ["lat", "lon"]:
            raise CouplingError(
                "StateDataSource serves (lat, lon) fields; got dims "
                f"{list(grid.keys())} for {fields[0].standard_name!r} — "
                "exchange-shaped Fields are expected (leading singleton dims "
                "are squeezed by the publishing component)"
            )
        for f in fields:
            if self.strict_time and f.valid_time is not None:
                if any(t != f.valid_time for t in times):
                    raise CouplingError(
                        f"Pull for {f.standard_name!r} at {times} but the "
                        f"served field is valid at {f.valid_time} — check the "
                        "run-sequence ordering (the connector must run before "
                        "the pulling component in the same slot)"
                    )

        # IO boundary of the pull path: the model's own fetch machinery is
        # xarray-based, so this conversion is unavoidable (inference-only).
        data = np.stack(
            [f.data.detach().cpu().numpy() for f in fields], axis=0
        )[np.newaxis].repeat(len(times), axis=0)
        return xr.DataArray(
            data,
            dims=["time", "variable", "lat", "lon"],
            coords={
                "time": times,
                "variable": variables,
                "lat": np.asarray(grid["lat"]),
                "lon": np.asarray(grid["lon"]),
            },
        )


class PullAdapter:
    """ImportAdapter for pull-pattern models (StormCast-style).

    Before each model call, installs a :class:`StateDataSource` over the
    current import State on the model's data-source attribute
    (``conditioning_data_source`` by default), then calls
    ``model(x, coords)`` unchanged. The model's internal fetch receives this
    step's coupled forcing.

    Parameters
    ----------
    attribute : str
        Name of the model's settable data-source attribute.
    strict_time : bool
        Raise if the model pulls times that do not match the served fields'
        valid_time (default False: serve the snapshot and let the run
        sequence own alignment).
    """

    def __init__(
        self,
        attribute: str = "conditioning_data_source",
        strict_time: bool = False,
        dictionary: "FieldDictionary | None" = None,
    ):
        self.attribute = attribute
        self.strict_time = strict_time
        self.dictionary = dictionary

    def __call__(
        self, model: Any, exchange: "Exchange"
    ) -> tuple[torch.Tensor, "CoordSystem"]:
        if not hasattr(model, self.attribute):
            raise CouplingError(
                f"PullAdapter: model {type(model).__name__!r} has no "
                f"attribute {self.attribute!r} — this adapter is for models "
                "that fetch forcing from a settable data source (e.g. "
                "StormCast's conditioning_data_source). For models taking "
                "conditioning as an argument use ConditioningKwargAdapter."
            )
        raw_to_std = {raw: std for std, raw in exchange.std_to_raw.items()}
        setattr(
            model,
            self.attribute,
            StateDataSource(
                exchange.imports,
                raw_to_std,
                strict_time=self.strict_time,
                dictionary=self.dictionary,
            ),
        )
        return model(exchange.x, OrderedDict(exchange.coords))
