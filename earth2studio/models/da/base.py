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

    Like prognostic models, assimilation models declare initial state with
    ``input_coords()`` and caller-supplied conditioning with ``forcing_coords()``.
    Forcing includes observations. Gridded slots use allocation-free DataArray
    signatures (``CoordinateSystem``); tabular slots use ``FrameSchema`` and accept
    pandas or cuDF DataFrames. One slot may be declared directly, multiple slots
    as an ordered tuple, and no slots as ``None``.

    Concrete execution methods declare fixed named parameters. The protocol uses
    variadic arguments only to describe models with different numbers of slots.
    A single output is returned directly; multiple outputs form an ordered tuple.
    Models never fetch data internally; ``default_sources()`` recommends providers.
    """

    def __call__(
        self, *x: pd.DataFrame | xr.DataArray | None
    ) -> pd.DataFrame | xr.DataArray | tuple[pd.DataFrame | xr.DataArray, ...]:
        """Compute an independent analysis without retaining rollout state.

        Parameters
        ----------
        *x : pd.DataFrame | xr.DataArray | None
            One argument per ``input_coords()`` slot followed by one per
            ``forcing_coords()`` slot, in declared order. Optional observation
            slots accept ``None`` only where the concrete model documents it.

        Returns
        -------
        pd.DataFrame | xr.DataArray | tuple[pd.DataFrame | xr.DataArray, ...]
            Analysis matching ``output_coords()``, on the model's device.
        """
        pass

    def create_generator(self, *x: pd.DataFrame | xr.DataArray) -> Generator[
        pd.DataFrame | xr.DataArray | tuple[pd.DataFrame | xr.DataArray, ...] | None,
        pd.DataFrame
        | xr.DataArray
        | tuple[pd.DataFrame | xr.DataArray | None, ...]
        | None,
        None,
    ]:
        """Create a bidirectional generator for successive analyses.

        Unlike the prognostic iterator, prime this generator with ``next(gen)``
        before supplying forcing. The priming yield is the initial state or
        ``None`` and performs no assimilation. Each subsequent ``send`` supplies
        forcing for one analysis. The concrete model documents whether it evolves
        the state, consumes successive initial frames, or processes independent
        observation batches.

        Parameters
        ----------
        *x : pd.DataFrame | xr.DataArray
            Initial state, one argument per ``input_coords()`` slot. Models with
            no initial state take no arguments.

        Yields
        ------
        pd.DataFrame | xr.DataArray | tuple[pd.DataFrame | xr.DataArray, ...] | None
            Initial state or ``None`` on priming, then analyses in output-slot order.

        Receives
        --------
        pd.DataFrame | xr.DataArray | tuple[pd.DataFrame | xr.DataArray | None, ...] | None
            One forcing value directly, or a tuple in ``forcing_coords()`` order.
            Preserve optional slots with ``None`` entries. ``send(None)`` after
            priming is valid only when the model requires no forcing for that step.
            Observation times and request metadata refer to the current analysis;
            gridded forcing must match the current window declared by the model.
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
        """Describe the initial state or background slots.

        Returns
        -------
        FrameSchema | CoordinateSystem | tuple[FrameSchema | CoordinateSystem, ...] | None
            Initial-state signatures in argument order, or ``None`` for models
            driven solely by observations. Passed to ``create_generator`` and
            before forcing in ``__call__``.
        """
        pass

    def forcing_coords(
        self,
    ) -> (
        FrameSchema
        | CoordinateSystem
        | tuple[FrameSchema | CoordinateSystem, ...]
        | None
    ):
        """Describe observations and other caller-supplied conditioning.

        Gridded signatures use lead times relative to the state being processed.
        Frame schemas describe observation columns and allowed values; each model
        documents its observation window and required DataFrame request metadata.
        Forcing is supplied on every generator ``send`` and is not implicitly
        fetched or reused. ``None`` means no forcing slots, not an optional slot.

        Returns
        -------
        FrameSchema | CoordinateSystem | tuple[FrameSchema | CoordinateSystem, ...] | None
            One signature or schema, an ordered tuple (which may mix both), or
            ``None`` for a model without forcing.
        """
        pass

    def output_coords(
        self, *input_coords: FrameSchema | CoordinateSystem, **kwargs: Any
    ) -> FrameSchema | CoordinateSystem | tuple[FrameSchema | CoordinateSystem, ...]:
        """Describe the analysis outputs.

        Parameters
        ----------
        *input_coords : FrameSchema | CoordinateSystem
            One signature or DataArray per initial-state slot as separate arguments.
            Concrete models declare fixed named parameters matching their slots.
        **kwargs : Any
            Model-specific request metadata, such as ``request_time`` from an
            observation DataFrame. Observation-only models need no positional inputs.

        Returns
        -------
        FrameSchema | CoordinateSystem | tuple[FrameSchema | CoordinateSystem, ...]
            Allocation-free output signatures or schemas in output-slot order.

        Raises
        ------
        ValueError
            If input coordinates or request metadata are invalid.
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
        """Recommend providers in input-then-forcing slot order.

        Models never fetch from these themselves. Drivers read recommendations
        through ``earth2studio.models.utils.recommended_sources``.

        Returns
        -------
        DataSource | ForecastSource | DataFrameSource | ForecastFrameSource | tuple[DataSource | ForecastSource | DataFrameSource | ForecastFrameSource | None, ...] | None
            One source directly, or one entry per input slot then per forcing slot.
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
