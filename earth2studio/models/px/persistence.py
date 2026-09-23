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

from collections.abc import Generator, Iterator
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import xarray as xr
from earth2studio.models._array_utils import _resolve_domain

from earth2studio.grids import GridDefinition
from earth2studio.models.px.utils import PrognosticMixin
from earth2studio.utils import (
    coord_array,
    coord_array_like,
    handshake_dataarray,
    handshake_nonempty,
    handshake_time,
)
from earth2studio.utils.checkpoint import bind_checkpoint_state
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem, CoordSystem


@dataclass
class _PersistenceCheckpointState:
    x: torch.Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Persistence(torch.nn.Module, PrognosticMixin):
    """Persistence model that generates a forecast by applying the identity operator on
    the initial condition and indexing the lead time by 6 hours. Primarily used in
    testing.

    Parameters
    ----------
    variable : Union[str, List[str]]
        The variable or list of variables predicted by the model.
    domain_coords : CoordSystem | GridDefinition | str
        Domain coordinates, grid definition, or registered grid name.
    history : int, optional
        Specifies the number of previous time steps to include as input, by default set
        to 1.
    dt : np.timedelta64, optional
        Time-step size of model between inputs and output, by default np.timedelta64(6, "h")

    Badges
    ------
    region:global provider:nvidia backend:pytorch
    """

    def __init__(
        self,
        variable: str | list[str],
        domain_coords: CoordSystem | GridDefinition | str,
        history: int = 1,
        dt: np.timedelta64 = np.timedelta64(6, "h"),
    ) -> None:
        super().__init__()

        if isinstance(variable, str):
            variable = [variable]

        if history < 1 or np.isnat(dt) or dt <= np.timedelta64(0, "s"):
            raise ValueError("Persistence requires positive history and time step")
        dims, coordinates, grid = _resolve_domain(domain_coords)
        self._input_coords = coord_array(
            ("batch", "lead_time", "variable", *dims),
            {
                "lead_time": np.array([-dt * i for i in reversed(range(history))]),
                "variable": np.array(variable),
                **coordinates,
            },
            dynamic=("batch",),
            grid=grid,
        )

        self._history = history
        self._dt = dt
        self.checkpoint = bind_checkpoint_state(_PersistenceCheckpointState())

    def __str__(
        self,
    ) -> str:
        return "persistence"

    def input_coords(self) -> CoordinateSystem:
        """Return the allocation-free configured history and domain signature."""
        return self._input_coords.copy()

    def output_coords(self, input_coords: CoordinateSystem) -> CoordinateSystem:
        """Validate the relative history and advance its final lead by one step."""
        handshake_time(input_coords, "lead_time")
        lead = np.asarray(input_coords.lead_time)
        handshake_dataarray(
            input_coords.assign_coords(lead_time=lead - lead[-1]), self.input_coords()
        )
        final = input_coords.isel(lead_time=slice(-1, None))
        return coord_array_like(
            final.assign_coords(
                lead_time=final.lead_time.variable.copy(data=lead[-1:] + self._dt)
            )
        )

    def _restore_checkpoint_state(self, x: xr.DataArray) -> tuple[xr.DataArray, bool]:
        if (
            self.checkpoint.checkpoint_level == 2
            and self.checkpoint.checkpoint_state_loaded
            and self.checkpoint.x is not None
            and self.checkpoint.metadata
        ):
            tensor, _ = x.e2s.to_torch()
            metadata = deepcopy(self.checkpoint.metadata)
            signature = coord_array(
                metadata["dims"],
                metadata["coords"],
                sizes=metadata["sizes"],
                attrs=metadata["attrs"],
            )
            restored = from_torch(
                self.checkpoint.x.to(tensor.device),
                signature,
                name=metadata["name"],
                attrs=metadata["attrs"],
            )
            restored.encoding = metadata["encoding"]
            return restored, True
        return x, False

    def _save_checkpoint_state(self, x: xr.DataArray) -> None:
        if self.checkpoint.checkpoint_enabled and self.checkpoint.checkpoint_level == 2:
            tensor, _ = x.e2s.to_torch()
            self.checkpoint.x = tensor.detach().clone().to(self.checkpoint.device)
            self.checkpoint.metadata = deepcopy(
                {
                    "dims": tuple(x.dims),
                    "sizes": dict(x.sizes),
                    "name": x.name,
                    "coords": {
                        name: (
                            tuple(value.dims),
                            value.values,
                            dict(value.attrs),
                        )
                        for name, value in x.coords.items()
                    },
                    "attrs": dict(x.attrs),
                    "encoding": dict(x.encoding),
                }
            )
        else:
            self.checkpoint.x = None
            self.checkpoint.metadata = {}

    @torch.inference_mode()
    def _forward(self, x: xr.DataArray) -> xr.DataArray:
        handshake_nonempty(x)
        signature = self.output_coords(x)
        result = x.isel(lead_time=slice(-1, None)).assign_coords(signature.coords)
        result.attrs = dict(signature.attrs)
        for key in (
            "earth2studio_kind",
            "earth2studio_schema_version",
            "earth2studio_dynamic_dims",
        ):
            result.attrs.pop(key, None)
        # Identity has no tensor kernel; the rear hook and caller own this storage.
        return result.copy(deep=True)

    def _advance_history(self, x: xr.DataArray, output: xr.DataArray) -> xr.DataArray:
        return xr.concat([x.isel(lead_time=slice(1, None)), output], dim="lead_time")

    def __call__(self, x: xr.DataArray) -> xr.DataArray:
        """Persist the final history field for one time step on the same device."""
        x, _ = self._restore_checkpoint_state(x)
        output = self._forward(x)
        if self.checkpoint.checkpoint_enabled and self.checkpoint.checkpoint_level == 2:
            x = self._advance_history(x, output)
        self._save_checkpoint_state(x)
        return output

    def _default_generator(
        self, x: xr.DataArray
    ) -> Generator[xr.DataArray, None, None]:
        x, restored = self._restore_checkpoint_state(x)
        handshake_nonempty(x)
        self.output_coords(x)
        if not restored:
            self._save_checkpoint_state(x)
            yield x.isel(lead_time=slice(-1, None)).copy(deep=True)

        while True:
            # Hooks may mutate data and metadata in place, including restored state.
            x = self.front_hook(x.copy(deep=True))
            output = self.rear_hook(self._forward(x))
            x = self._advance_history(x, output)
            self._save_checkpoint_state(x)
            yield output.copy(deep=False)

    def create_iterator(self, x: xr.DataArray) -> Iterator[xr.DataArray]:
        """Yield the final input field, then forecasts; checkpoints resume next step."""
        yield from self._default_generator(x)
