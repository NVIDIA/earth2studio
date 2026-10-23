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
from __future__ import annotations

from collections.abc import Generator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pandas as pd
import xarray as xr

from earth2studio.utils.type import CoordinateSystem, FrameSchema

if TYPE_CHECKING:
    from earth2studio.data.base import (
        DataFrameSource,
        DataSource,
        ForecastFrameSource,
        ForecastSource,
    )


# --8<-- [start:assimilation-model-interface]
@runtime_checkable
class AssimilationModel(Protocol):
    """Data assimilation model interface.

    Assimilation models declare their arguments with ``input_coords()``.
    Inputs include observations and any background fields.
    Gridded slots use allocation-free DataArray
    signatures (``CoordinateSystem``); tabular slots use ``FrameSchema`` and accept
    pandas or cuDF DataFrames. One slot may be declared directly, multiple slots
    as an ordered tuple, and no slots as ``None``.

    Concrete execution methods declare fixed named parameters. The protocol uses
    variadic arguments only to describe models with different numbers of slots.
    Return a single output object directly, or multiple objects in declared order,
    e.g. ``return y1, y2``.
    Outputs may be gridded fields or tabular analyses. This interface does not
    imply a trajectory: use ``StatefulAssimilationModel`` for explicit continuation.
    Models never fetch data internally; ``default_sources()`` recommends providers.
    """

    stochastic: bool

    def set_rng(self, seed: int, reset: bool = True) -> None:
        """Configure the model-local random stream.

        Parameters
        ----------
        seed : int
            Seed for independent analyses or new trajectories.
        reset : bool, optional
            Restart the stream when True; otherwise initialize only if absent.
            Defaults to True. Deterministic models implement a no-op.

        Notes
        -----
        Calls advance the stream. Seeding and seeded execution leave global RNG
        state unchanged. Stateful continuation uses its explicit RNG state and is
        unaffected by reseeding the model.
        """
        pass

    def __call__(
        self, *x: pd.DataFrame | xr.DataArray | None
    ) -> xr.DataArray | pd.DataFrame | tuple[xr.DataArray | pd.DataFrame, ...]:
        """Compute an independent analysis without retaining rollout state.

        Parameters
        ----------
        *x : pd.DataFrame | xr.DataArray | None
            One argument per ``input_coords()`` slot, in declared order.
            Stateful models append their forcing slots. Optional observation
            slots accept ``None`` only where the concrete model documents it.

        Returns
        -------
        xr.DataArray | pd.DataFrame | tuple[xr.DataArray | pd.DataFrame, ...]
            Analysis matching ``output_coords()``, on the model's device.
        """
        pass

    def input_coords(
        self,
    ) -> (
        FrameSchema
        | CoordinateSystem
        | tuple[FrameSchema | CoordinateSystem, ...]
        | None
    ):
        """Describe the model's input slots.

        Returns
        -------
        FrameSchema | CoordinateSystem | tuple[FrameSchema | CoordinateSystem, ...] | None
            Field signatures and observation schemas in argument order, or
            ``None`` for no inputs. Stateful models declare initialization inputs
            here and additional conditioning in ``forcing_coords()``.
        """
        pass

    def output_coords(
        self, *input_coords: FrameSchema | CoordinateSystem, **kwargs: Any
    ) -> FrameSchema | CoordinateSystem | tuple[FrameSchema | CoordinateSystem, ...]:
        """Describe the analysis outputs.

        Parameters
        ----------
        *input_coords : FrameSchema | CoordinateSystem
            One signature or schema per input slot as separate arguments.
            Concrete models declare fixed named parameters matching their slots.
        **kwargs : Any
            Model-specific request metadata, such as analysis time.

        Returns
        -------
        FrameSchema | CoordinateSystem | tuple[FrameSchema | CoordinateSystem, ...]
            Allocation-free DataArray signatures or tabular schemas in output-slot
            order. Outputs are actual analyses, never priming ``None`` values.

        Raises
        ------
        ValueError
            If input coordinates or model-specific arguments are invalid.
        """
        pass

    def default_sources(
        self,
    ) -> (
        DataSource
        | ForecastSource
        | DataFrameSource
        | ForecastFrameSource
        | tuple[
            DataSource | ForecastSource | DataFrameSource | ForecastFrameSource | None,
            ...,
        ]
        | None
    ):
        """Recommend providers in input slot order.

        Models never fetch from these themselves. Drivers read recommendations
        through ``earth2studio.models.utils.recommended_sources``.

        Returns
        -------
        DataSource | ForecastSource | DataFrameSource | ForecastFrameSource | tuple[DataSource | ForecastSource | DataFrameSource | ForecastFrameSource | None, ...] | None
            One source directly, or one entry per input slot. Stateful models append
            providers for their forcing slots.
            A ``None`` tuple entry leaves that slot to the caller; a top-level
            ``None`` means no recommendations. Callers align data to each slot's
            declared grid, variables, and time window before execution.
        """
        pass

    def to(self, device: Any) -> AssimilationModel:
        """Move the assimilation model onto its inference device.

        Parameters
        ----------
        device : Any
            Inference device, typically ``torch.device`` or str.

        Returns
        -------
        AssimilationModel
            The model on the requested device, typically supplied by ``torch.nn.Module``.
        """
        pass


# --8<-- [end:assimilation-model-interface]


# --8<-- [start:stateful-assimilation-model-interface]
@runtime_checkable
class StatefulAssimilationModel(AssimilationModel, Protocol):
    """Assimilation model with explicit, replayable trajectory continuation.

    Extends the independent-call interface with initialization and stepping, allowing
    DataFrames and optional observation slots alongside gridded fields. Concrete
    methods use fixed named parameters; inheritance is not required. ``__call__``
    returns the analysis from ``initialize`` without retaining continuation state.
    Input/output coordinate hooks describe slots, not iterator callbacks. If a
    model offers front/rear execution hooks, they apply only during iteration,
    and their returned analyses feed the recurrence.
    """

    def forcing_coords(
        self,
    ) -> (
        FrameSchema
        | CoordinateSystem
        | tuple[FrameSchema | CoordinateSystem, ...]
        | None
    ):
        """Describe observations and other caller-supplied step conditioning.

        Gridded signatures use lead times relative to the state being processed.
        Frame schemas describe observation columns and allowed values; each model
        documents its observation window and any required DataFrame metadata.
        Forcing follows initialization inputs and is supplied again for each step;
        it is never implicitly fetched or reused.

        Returns
        -------
        FrameSchema | CoordinateSystem | tuple[FrameSchema | CoordinateSystem, ...] | None
            One signature or schema, multiple declarations in slot order, or
            ``None`` for no forcing slots. Optional slots still have a declaration.
        """
        pass

    def initialize(
        self, *x: xr.DataArray | pd.DataFrame | None
    ) -> tuple[
        xr.DataArray | pd.DataFrame | tuple[xr.DataArray | pd.DataFrame, ...], Any
    ]:
        """Compute the first analysis and initialize its continuation state.

        Parameters
        ----------
        *x : xr.DataArray | pd.DataFrame | None
            Same arguments as ``__call__``: input slots followed by forcing slots,
            including complete initialization windows and static forcing. Optional
            observations retain their declared slot as ``None``.

        Returns
        -------
        tuple[xr.DataArray | pd.DataFrame | tuple[xr.DataArray | pd.DataFrame, ...], Any]
            First computed analysis and serializable model-defined state (or
            ``None``). State contains everything else needed to continue: history,
            static forcing, latents, and RNG position. Applies no execution hooks
            and does not mutate caller inputs.
        """
        pass

    def step(
        self, *y: xr.DataArray | pd.DataFrame | None, state: Any
    ) -> tuple[
        xr.DataArray | pd.DataFrame | tuple[xr.DataArray | pd.DataFrame, ...], Any
    ]:
        """Advance the trajectory by one core computation.

        Parameters
        ----------
        *y : xr.DataArray | pd.DataFrame | None
            Previous output slots as separate arguments, followed by new dynamic
            forcing in declared order. Omit initialization-only static slots;
            preserve optional observations with ``None``. Gridded forcing supplies
            the final declared lead shifted by the output being advanced; older
            frames are retained in state. Each model documents observation windows
            relative to analysis time and any required DataFrame metadata.
        state : Any
            State returned with the previous analysis. Concrete methods declare
            this as a positional-or-keyword parameter after output and forcing
            slots; it is keyword-only here because it follows ``*y``.

        Returns
        -------
        tuple[xr.DataArray | pd.DataFrame | tuple[xr.DataArray | pd.DataFrame, ...], Any]
            Next complete analysis and new state. Together, outputs and state are
            the complete continuation. Identical arguments replay identically,
            including stochastic draws, without mutating inputs or prior state.
            One step is one iterator yield; applies no execution hooks.

        Raises
        ------
        ValueError
            If required forcing is missing, slots are invalid, or outputs do not
            continue the supplied state.
        """
        pass

    def create_iterator(self, *x: xr.DataArray | pd.DataFrame | None) -> Generator[
        xr.DataArray | pd.DataFrame | tuple[xr.DataArray | pd.DataFrame, ...],
        xr.DataArray
        | pd.DataFrame
        | tuple[xr.DataArray | pd.DataFrame | None, ...]
        | None,
        None,
    ]:
        """Iterate successive computed analyses using explicit continuation.

        Parameters
        ----------
        *x : xr.DataArray | pd.DataFrame | None
            Same input and forcing arguments as ``initialize`` and ``__call__``.

        Yields
        ------
        xr.DataArray | pd.DataFrame | tuple[xr.DataArray | pd.DataFrame, ...]
            Complete analyses, starting with ``initialize``'s output. There is no
            initial-condition or ``None`` priming yield. Drivers publish initial
            fields separately; n analyses require n yields.

        Receives
        --------
        xr.DataArray | pd.DataFrame | tuple[xr.DataArray | pd.DataFrame | None, ...] | None
            Dynamic forcing for the next step, a single slot directly or a tuple
            in declared order, omitting initialization-only static slots. Preserve
            optional slots with ``None`` entries. ``next(it)`` is ``send(None)``:
            valid only when no required new forcing is needed. Required gridded
            conditioning is never implicitly fetched or reused.
        """
        pass

    def to(self, device: Any) -> StatefulAssimilationModel:
        """Move to the inference device and return the stateful model.

        Parameters
        ----------
        device : Any
            Inference device, typically ``torch.device`` or str.

        Returns
        -------
        StatefulAssimilationModel
            Model on the requested device, typically supplied by ``torch.nn.Module``.
        """
        pass


# --8<-- [end:stateful-assimilation-model-interface]
