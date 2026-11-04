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

"""Data-assimilation runners on the 1.0 :class:`AssimilationModel` protocol.

Two evaluation modes for independent DA models such as HealDA:

* ``runner: assimilation`` -- :class:`AssimilationRunner`.  Each work item
  produces one analysis at its time, yielded as the ``analysis`` stream with a
  single ``lead_time=0`` entry so the standard scoring and report machinery
  read it as "analysis error at each cycle time".
* ``runner: assimilation_forecast`` -- the DA analysis initializes a
  prognostic rollout.  :class:`AnalysisSource` adapts the DA model to the
  ``DataSource`` protocol, so the DA-initialized forecast is the ordinary
  forecast runner reading its initial condition from that source.  A
  :class:`~earth2studio.data.CompositeSource` fills prognostic input variables
  the analysis lacks from the configured data source.

:class:`ObsSources` resolves the observation inputs, one ``DataFrameSource``
per model input slot, and declares them for predownload through
``frame_requests``.

The runners support independent (stateless) DA models only: each analysis is
one ``model(*obs)`` call, so work items distribute freely across ranks.
Cycling ``StatefulAssimilationModel`` runs are out of scope, and a
``cfg.model.da.runner`` entry is an error.
"""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import hydra
import numpy as np
import torch
import xarray as xr
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from earth2studio.data import CompositeSource, RegriddedSource, fetch_dataframe
from earth2studio.grids import (
    GridDefinition,
    infer_grid,
    resolve_regridder,
    same_geometry,
)
from earth2studio.models.da.base import AssimilationModel
from earth2studio.models.utils import recommended_sources
from earth2studio.run import DataRequest, WorkItem
from earth2studio.run.runner import (
    _module_device,
    _seed_model,
    _signature_for_stream,
)
from earth2studio.utils.coords import E2S_DYNAMIC_DIMS, coord_array
from earth2studio.utils.type import CoordinateSystem, FrameSchema

from .. import models as _models
from ..data import (
    FrameRequest,
    PredownloadedFrameSource,
    frame_store_path,
    resolve_source,
)
from . import forecast as build_forecast
from .ensemble import EnsembleRunner

ZERO_LEAD = np.array([np.timedelta64(0, "h")])
"""The single lead time of an analysis product."""


# -- model helpers -----------------------------------------------------------


def load_assimilation(node: DictConfig | dict) -> AssimilationModel:
    """Load a DA model from its config node (``cfg.model.da``).

    Uses :func:`src.models.load_assimilation` when the models module provides
    it, else the generic :func:`src.models.load_from_node`.
    """
    loader = getattr(_models, "load_assimilation", None)
    if loader is not None:
        return loader(node)
    model: AssimilationModel = _models.load_from_node(node, "assimilation model")
    return model


def input_schemas(model: AssimilationModel) -> tuple[FrameSchema, ...]:
    """Return the model's tabular input slots, in argument order.

    Raises
    ------
    NotImplementedError
        If the model declares gridded (background) inputs, which these
        observation-only runners cannot supply.
    """
    declared = model.input_coords()
    slots: tuple[Any, ...]
    if declared is None:
        slots = ()
    elif isinstance(declared, tuple):
        slots = declared
    else:
        slots = (declared,)
    gridded = [i for i, slot in enumerate(slots) if isinstance(slot, xr.DataArray)]
    if gridded:
        raise NotImplementedError(
            f"{type(model).__name__} declares gridded inputs at slots {gridded}; "
            "the assimilation runners supply tabular observations only."
        )
    return tuple(slots)


def analysis_coords(model: AssimilationModel) -> CoordinateSystem:
    """Return the signature of the model's gridded analysis output.

    The first output slot is the analysis.  Models fix its ``time`` axis to a
    placeholder; this function makes the axis dynamic so supervision can
    label it.
    """
    outputs = model.output_coords(*input_schemas(model))
    signature = outputs[0] if isinstance(outputs, tuple) else outputs
    if not isinstance(signature, xr.DataArray):
        raise ValueError(
            f"{type(model).__name__} output_coords() must start with a gridded "
            f"analysis signature, got {type(signature).__name__}"
        )
    dynamic = tuple(signature.attrs.get(E2S_DYNAMIC_DIMS, ()))
    if "time" not in signature.dims or "time" in dynamic:
        return signature
    dims = tuple(signature.dims)
    if dims[0] != "time":
        raise ValueError(
            f"Analysis signature must lead with 'time', got dimensions {dims}"
        )
    coordinates = {
        name: coord.variable
        for name, coord in signature.coords.items()
        if "time" not in coord.dims
    }
    attrs = {k: v for k, v in signature.attrs.items() if k != E2S_DYNAMIC_DIMS}
    return coord_array(
        dims,
        coordinates,
        dynamic=("time",),
        sizes={dim: signature.sizes[dim] for dim in dims[1:]},
        dtype=signature.dtype,
        name=signature.name,
        attrs=attrs,
    )


def analysis_variables(model: AssimilationModel) -> list[str]:
    """Return the variable names of the model's gridded analysis."""
    return [str(v) for v in analysis_coords(model).coords["variable"].values]


# -- observation sources -----------------------------------------------------


def _fields_for(source: Any, schema: FrameSchema) -> np.ndarray | None:
    """Columns to request from ``source`` for a slot described by ``schema``.

    Schema keys other than ``variable`` are columns; ``variable`` is the list
    of requested variables and is only a column when the source's ``SCHEMA``
    says so (long-layout sources such as the UFS GSI readers).  Sources
    without a ``SCHEMA``, such as predownloaded frame stores, return every
    stored column.
    """
    declared = getattr(source, "SCHEMA", None)
    if declared is None:
        return None
    names = set(getattr(declared, "names", ()))
    fields = [k for k in schema if k != "variable" or k in names]
    return np.array(fields, dtype=str)


class ObsSources:
    """One ``DataFrameSource`` (or ``None``) per DA model input slot.

    Parameters
    ----------
    sources : Sequence[Any]
        One source per slot, positionally matched to ``input_schemas(model)``;
        ``None`` disables the slot and the model receives ``None``.
    schemas : Sequence[FrameSchema]
        The model's input schemas, giving each slot's variables and columns.
    names : Sequence[str], optional
        One logical name per slot, used to name predownloaded frame stores
        (``<output.path>/<name>.parquet``), by default ``obs0, obs1, ...``.
    """

    def __init__(
        self,
        sources: Sequence[Any],
        schemas: Sequence[FrameSchema],
        names: Sequence[str] | None = None,
    ) -> None:
        if len(sources) != len(schemas):
            raise ValueError(
                f"ObsSources: {len(sources)} sources for a model with "
                f"{len(schemas)} observation inputs; one entry is required per "
                "input (use 'enabled: false' to pass None)."
            )
        self.sources: tuple[Any, ...] = tuple(sources)
        self.schemas: tuple[FrameSchema, ...] = tuple(schemas)
        if names is None:
            names = [f"obs{i}" for i in range(len(sources))]
        if len(names) != len(sources):
            raise ValueError(
                f"ObsSources: {len(names)} names for {len(sources)} sources."
            )
        self.names: tuple[str, ...] = tuple(str(n) for n in names)

    @property
    def enabled(self) -> bool:
        """Whether at least one slot has a source."""
        return any(s is not None for s in self.sources)

    @property
    def roles(self) -> dict[str, Any]:
        """Slot name to bound source, for predownload store naming."""
        return dict(zip(self.names, self.sources))

    @classmethod
    def from_config(
        cls,
        model: AssimilationModel,
        obs_cfg: DictConfig | Mapping[str, Any] | None,
        output_path: str | None = None,
        prefer_predownloaded: bool = True,
    ) -> ObsSources:
        """Resolve the sources of every input slot of ``model``.

        ``obs_cfg`` (``cfg.model.da.obs_sources``) is an ordered mapping; its
        entries match the model's input slots positionally, in declaration
        order.  Each entry is a Hydra ``_target_`` block plus
        two convenience keys: ``enabled: false`` passes ``None`` for the slot
        and ``time_tolerance_hours: [lo, hi]`` becomes the source's
        ``time_tolerance`` pair of hour timedeltas.  A ``null`` entry, or no
        mapping at all, falls back to ``model.default_sources()``.

        With ``prefer_predownloaded``, a predownloaded frame store at
        ``<output_path>/<name>.parquet`` takes precedence over the live source.

        Parameters
        ----------
        model : AssimilationModel
            Loaded DA model, for its input schemas and default sources.
        obs_cfg : DictConfig | Mapping | None
            Mapping of slot name to source node, or None for model defaults.
        output_path : str, optional
            Run output directory holding predownloaded frame stores.
        prefer_predownloaded : bool, optional
            Read predownloaded frame stores when present, by default True.

        Returns
        -------
        ObsSources
        """
        schemas = input_schemas(model)
        entries: list[tuple[str, Any]]
        if not obs_cfg:
            entries = [(f"obs{i}", None) for i in range(len(schemas))]
        else:
            mapping: Any = (
                OmegaConf.to_container(obs_cfg, resolve=True)
                if isinstance(obs_cfg, DictConfig)
                else obs_cfg
            )
            entries = list(mapping.items())
        if len(entries) != len(schemas):
            raise ValueError(
                f"model.da.obs_sources lists {len(entries)} entries for a model "
                f"with {len(schemas)} observation inputs; give one per input "
                "slot in input_coords() order."
            )
        defaults: tuple[Any, ...] | None = None
        sources: list[Any] = []
        names: list[str] = []
        for i, (name, node) in enumerate(entries):
            name = str(name)
            names.append(name)
            spec = dict(node) if node is not None else None
            if spec is not None and not spec.pop("enabled", True):
                logger.info(f"Observation slot '{name}' disabled; passing None.")
                sources.append(None)
                continue
            if output_path is not None and prefer_predownloaded:
                store = frame_store_path(output_path, name)
                if os.path.isdir(store):
                    logger.info(
                        f"Observation slot '{name}': reading predownloaded frame "
                        f"store {store}"
                    )
                    sources.append(PredownloadedFrameSource(store))
                    continue
            if spec is None:
                if defaults is None:
                    defaults = _default_sources(model, len(schemas))
                if defaults[i] is None:
                    raise ValueError(
                        f"Observation slot '{name}' has no configured source and "
                        f"{type(model).__name__} recommends none for it."
                    )
                logger.info(f"Observation slot '{name}': using the model's default")
                sources.append(defaults[i])
                continue
            target = spec.pop("_target_", None)
            if target is None:
                raise ValueError(
                    f"Observation source '{name}' needs a '_target_' key "
                    f"(got keys {sorted(spec)})."
                )
            tolerance = spec.pop("time_tolerance_hours", None)
            if tolerance is not None:
                lo, hi = tolerance
                spec["time_tolerance"] = (
                    np.timedelta64(int(lo), "h"),
                    np.timedelta64(int(hi), "h"),
                )
            # Construct directly so the numpy tolerance tuple is not re-wrapped
            # in an OmegaConf container by hydra.utils.instantiate.
            sources.append(hydra.utils.get_class(target)(**spec))
            logger.success(f"Observation slot '{name}': {target}")
        return cls(sources, schemas, names=names)

    def fetch(
        self, time: np.datetime64, device: torch.device | str = "cpu"
    ) -> tuple[Any, ...]:
        """Fetch every slot's observations for one analysis time.

        Parameters
        ----------
        time : np.datetime64
            Analysis valid time, recorded as ``request_time`` in each frame's
            ``attrs`` by :func:`~earth2studio.data.fetch_dataframe`.
        device : torch.device | str, optional
            ``"cpu"`` yields pandas frames, CUDA yields cudf, by default "cpu".

        Returns
        -------
        tuple
            One DataFrame or ``None`` per slot, in slot order.
        """
        times = np.array([time], dtype="datetime64[ns]")
        frames: list[Any] = []
        for source, schema in zip(self.sources, self.schemas):
            if source is None:
                frames.append(None)
                continue
            frames.append(
                fetch_dataframe(
                    source,
                    time=times,
                    variable=np.asarray(schema["variable"]),
                    fields=_fields_for(source, schema),
                    device=torch.device(device),
                )
            )
        return tuple(frames)

    def requests(self, times: np.ndarray) -> tuple[FrameRequest, ...]:
        """Describe the frame fetches of every enabled slot at ``times``."""
        times = np.asarray(times, dtype="datetime64[ns]")
        return tuple(
            FrameRequest(
                name=name,
                source=source,
                time=times,
                variable=np.asarray(schema["variable"]),
                fields=_fields_for(source, schema),
            )
            for name, source, schema in zip(self.names, self.sources, self.schemas)
            if source is not None
        )


def _default_sources(model: AssimilationModel, nslots: int) -> tuple[Any, ...]:
    recommended = recommended_sources(model)
    if recommended is None:
        return (None,) * nslots
    if not isinstance(recommended, tuple):
        recommended = (recommended,)
    if len(recommended) != nslots:
        raise ValueError(
            f"{type(model).__name__}.default_sources() lists {len(recommended)} "
            f"sources for {nslots} input slots"
        )
    return recommended


# -- analysis computation ----------------------------------------------------


def analyze(
    model: AssimilationModel,
    obs: ObsSources,
    time: np.datetime64,
    obs_device: torch.device | str = "cpu",
) -> xr.DataArray:
    """Run one independent analysis at ``time``.

    Parameters
    ----------
    model : AssimilationModel
        Device-placed DA model.
    obs : ObsSources
        Observation sources for every input slot.
    time : np.datetime64
        Analysis valid time.
    obs_device : torch.device | str, optional
        Destination device of the observation frames, by default "cpu".

    Returns
    -------
    xr.DataArray
        Gridded analysis with dimensions ``[time, variable, <spatial>]`` and
        ``time == [time]``, on the model's device.
    """
    if not obs.enabled:
        raise ValueError(
            "All observation sources are disabled; at least one must be enabled "
            "to produce an analysis."
        )
    frames = obs.fetch(time, device=obs_device)
    output = model(*frames)
    analysis = output[0] if isinstance(output, tuple) else output
    if not isinstance(analysis, xr.DataArray):
        raise ValueError(
            f"{type(model).__name__} returned {type(analysis).__name__} as its "
            "first output; the analysis must be a gridded xr.DataArray."
        )
    stamp = np.array([time], dtype="datetime64[ns]")
    if "time" not in analysis.dims:
        analysis = analysis.expand_dims(time=stamp, axis=0)
    else:
        analysis = analysis.assign_coords(time=stamp)
    return analysis


def _to_host(x: xr.DataArray) -> xr.DataArray:
    """Return ``x`` NumPy-backed, moving CuPy data to the host."""
    data = x.data
    if hasattr(data, "__cuda_array_interface__"):
        return x.copy(data=np.asarray(data.get()))
    return x


def with_zero_lead(x: xr.DataArray) -> xr.DataArray:
    """Insert a singleton ``lead_time=[0h]`` axis after ``time``."""
    axis = list(x.dims).index("time") + 1
    return x.expand_dims({"lead_time": ZERO_LEAD}, axis=axis)


# -- runner -------------------------------------------------------------------


class AssimilationRunner:
    """Produce one DA analysis per work item as the ``analysis`` stream.

    Implements the :class:`earth2studio.run.Runner` protocol.  The runner
    fetches no gridded data (``data_requests`` is empty) and declares its
    observations through :meth:`frame_requests` for predownload.  Items carry
    at most one ensemble member; the runner seeds stochastic models per member
    with :func:`~earth2studio.run.work.member_seed`.

    Parameters
    ----------
    model : AssimilationModel
        Independent DA model.
    obs : ObsSources
        Observation sources for the model's input slots.
    seed : int, optional
        Base seed for member seeding of stochastic models, by default 0.
    obs_device : torch.device | str, optional
        Destination device of the observation frames (``"cpu"`` for pandas,
        CUDA for cudf), by default "cpu".
    """

    supports_member_batching = False
    step = None
    """No model step: only ``nsteps: 0`` campaigns are valid."""

    def __init__(
        self,
        model: AssimilationModel,
        obs: ObsSources,
        seed: int = 0,
        obs_device: torch.device | str = "cpu",
    ) -> None:
        if not obs.enabled:
            raise ValueError(
                "All observation sources are disabled; enable at least one slot."
            )
        self.model = model
        self.obs = obs
        self.seed = int(seed)
        self.obs_device = obs_device
        self.device = _module_device(model)
        self.signature = _signature_for_stream(analysis_coords(model), ZERO_LEAD)
        self.source_roles: dict[str, Any] = {}
        self.frame_roles: dict[str, Any] = obs.roles

    def to(self, device: torch.device) -> AssimilationRunner:
        """Move the DA model to ``device``."""
        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        return self

    def output_coords(
        self, final_lead_time: np.timedelta64
    ) -> Mapping[str, CoordinateSystem]:
        """Return the ``analysis`` stream signature: ``lead_time=[0h]`` then fields."""
        if np.timedelta64(final_lead_time) != np.timedelta64(0, "s"):
            raise ValueError(
                f"AssimilationRunner produces analyses only; final lead time "
                f"{final_lead_time} needs runner 'assimilation_forecast' or nsteps=0."
            )
        return {"analysis": self.signature}

    def data_requests(self, item: WorkItem) -> tuple[DataRequest, ...]:
        """No gridded fetches: observations arrive through :meth:`frame_requests`."""
        return ()

    def frame_requests(self, item: WorkItem) -> tuple[FrameRequest, ...]:
        """Describe the observation fetch of every enabled slot for ``item``."""
        return self.obs.requests(np.array([item.time]))

    def run_item(self, item: WorkItem) -> Iterator[Mapping[str, xr.DataArray]]:
        """Yield the seeded analysis at the item's time with ``lead_time=[0h]``."""
        if len(item.member_ids) > 1:
            raise ValueError("AssimilationRunner does not batch ensemble members")
        member = item.member_ids[0] if item.member_ids else 0
        _seed_model(self.seed, item.time, member, self.model)
        analysis = analyze(self.model, self.obs, item.time, self.obs_device)
        yield {"analysis": with_zero_lead(analysis)}


# -- DA analyses as a data source -------------------------------------------


class AnalysisSource:
    """``DataSource`` producing DA analyses at the requested valid times.

    One independent model call per requested time, so a history prognostic
    requesting ``[t - 12h, t]`` runs two analyses.  Fields return
    NumPy-backed on the host so they compose with ordinary data sources in a
    :class:`~earth2studio.data.CompositeSource`; the forecast runner moves
    the initial condition to its device.  The source seeds stochastic models
    per time with :func:`~earth2studio.run.work.member_seed` (member 0), so
    the same analysis initializes every member of an ensemble.

    Parameters
    ----------
    model : AssimilationModel
        Independent DA model.
    obs : ObsSources
        Observation sources for the model's input slots.
    seed : int, optional
        Base seed for stochastic models, by default 0.
    obs_device : torch.device | str, optional
        Destination device of the observation frames, by default "cpu".
    """

    time_step = None

    def __init__(
        self,
        model: AssimilationModel,
        obs: ObsSources,
        seed: int = 0,
        obs_device: torch.device | str = "cpu",
    ) -> None:
        if not obs.enabled:
            raise ValueError(
                "All observation sources are disabled; enable at least one slot."
            )
        self.model = model
        self.obs = obs
        self.seed = int(seed)
        self.obs_device = obs_device
        self.signature = analysis_coords(model)
        self.variables = [str(v) for v in self.signature.coords["variable"].values]
        self.device = torch.device("cpu")

    @property
    def grid(self) -> GridDefinition:
        """Grid of the analysis."""
        return infer_grid(self.signature)

    def to(self, device: torch.device) -> AnalysisSource:
        """Move the DA model to ``device``; fields still return on the host."""
        self.model = self.model.to(torch.device(device))
        return self

    def frame_requests(self, times: np.ndarray) -> tuple[FrameRequest, ...]:
        """Describe the observation fetches behind analyses at ``times``."""
        return self.obs.requests(np.asarray(times))

    def __call__(self, time: Any, variable: Any) -> xr.DataArray:
        """Return analyses at ``time`` restricted to ``variable``.

        Parameters
        ----------
        time : datetime-like or array
            Analysis valid times.
        variable : str or sequence of str
            Analysis variables to return.

        Returns
        -------
        xr.DataArray
            Field with dimensions ``[time, variable, <spatial>]``.
        """
        times = np.atleast_1d(np.asarray(time, dtype="datetime64[ns]"))
        variables = [str(v) for v in np.atleast_1d(np.asarray(variable))]
        missing = sorted(set(variables) - set(self.variables))
        if missing:
            raise KeyError(
                f"{type(self.model).__name__} analyses do not provide {missing}; "
                f"available: {self.variables}"
            )
        parts: list[xr.DataArray] = []
        for t in times:
            _seed_model(self.seed, t, 0, self.model)
            parts.append(_to_host(analyze(self.model, self.obs, t, self.obs_device)))
        field = parts[0] if len(parts) == 1 else xr.concat(parts, dim="time")
        return field.sel(variable=variables)

    async def fetch(self, time: Any, variable: Any) -> xr.DataArray:
        """Async version of :meth:`__call__`."""
        return self(time, variable)


class GridAlignedSource:
    """Data source regridded onto a target grid, decided on the first fetch.

    A source's native grid is only known from its fields, so the first fetch
    infers it: fields already on ``target`` pass through; otherwise a
    :class:`~earth2studio.data.RegriddedSource` built with
    :func:`~earth2studio.grids.resolve_regridder` regrids every fetch from
    then on.

    Parameters
    ----------
    source : DataSource
        Source to align.
    target : GridDefinition
        Grid to return fields on.
    method : str, optional
        Registered regridding method, by default "nearest".
    """

    def __init__(
        self, source: Any, target: GridDefinition, method: str = "nearest"
    ) -> None:
        self.source = source
        self.target = target
        self.method = method
        self._aligned: Any = None

    @property
    def device(self) -> torch.device:
        """Device of the wrapped source."""
        return torch.device(getattr(self.source, "device", "cpu"))

    @property
    def time_step(self) -> np.timedelta64 | None:
        """Cadence of the wrapped source."""
        return getattr(self.source, "time_step", None)

    def __call__(self, time: Any, variable: Any) -> xr.DataArray:
        """Fetch from the wrapped source on the target grid."""
        if self._aligned is not None:
            return self._aligned(time, variable)
        field = self.source(time, variable)
        grid = infer_grid(field)
        if same_geometry(grid, self.target):
            self._aligned = self.source
            return field
        regridder = resolve_regridder(grid, self.target, self.method)
        logger.info(
            f"Fill source {type(self.source).__name__} is regridded onto the "
            f"analysis grid with {type(regridder).__name__}"
        )
        self._aligned = RegriddedSource(self.source, regridder)
        return regridder(field)

    async def fetch(self, time: Any, variable: Any) -> xr.DataArray:
        """Async version of :meth:`__call__`."""
        return self(time, variable)


def initial_condition_source(
    cfg: DictConfig,
    analysis: AnalysisSource,
    variables: Sequence[str],
    prefer_predownloaded: bool = True,
    fallback: Any = None,
) -> tuple[Any, Any, list[str]]:
    """Compose the initial-condition source of a DA-initialized forecast.

    Parameters
    ----------
    cfg : DictConfig
        Campaign config; reads ``model.fill_missing_variables`` (default
        true), ``ic_source``, ``data_source`` and ``output.path``.
    analysis : AnalysisSource
        Source of DA analyses.
    variables : Sequence[str]
        Variables the prognostic's initial condition needs.
    prefer_predownloaded : bool, optional
        Read a predownloaded ``data.zarr`` fill store when present, by default True.
    fallback : Any, optional
        Fill source when the config has neither ``ic_source`` nor
        ``data_source``, typically the prognostic's recommended source, by
        default None.

    Returns
    -------
    tuple[Any, Any, list[str]]
        ``(source, fill, missing)``: the initial-condition source, the
        resolved fill source (``None`` when the analysis covers every
        variable) and the filled variables.

    Raises
    ------
    ValueError
        If the analysis lacks variables and the config disables filling or
        names no fill source.
    """
    missing = [str(v) for v in variables if str(v) not in analysis.variables]
    if not missing:
        logger.info("The DA analysis provides every prognostic input variable.")
        return analysis, None, []
    if not cfg.model.get("fill_missing_variables", True):
        raise ValueError(
            f"The DA analysis does not provide the prognostic input variables "
            f"{missing}. Allow filling them from the data source "
            "(model.fill_missing_variables=true) or use a prognostic whose "
            "inputs the DA model covers."
        )
    fill = resolve_source(
        cfg,
        "data",
        byo=cfg.get("ic_source"),
        live=cfg.get("data_source"),
        prefer_predownloaded=prefer_predownloaded,
    )
    if fill is None:
        fill = fallback
    if fill is None:
        raise ValueError(
            f"Filling {missing} needs 'ic_source' or 'data_source'; neither is "
            "set and the prognostic recommends no source."
        )
    logger.warning(
        f"DA analysis lacks {len(missing)} of {len(variables)} prognostic input "
        f"variables; FILLING {missing} from {type(fill).__name__}."
    )
    aligned = GridAlignedSource(fill, analysis.grid)
    source = CompositeSource(
        {"analysis": analysis, "fill": aligned},
        {"analysis": analysis.variables, "fill": missing},
    )
    return source, fill, missing


# -- builders ----------------------------------------------------------------


def _da_components(
    cfg: DictConfig, prefer_predownloaded: bool
) -> tuple[AssimilationModel, ObsSources, DictConfig]:
    if "da" not in cfg.model:
        raise ValueError(
            "DA runners need a 'model.da' node (see cfg/model/healda.yaml)."
        )
    da_cfg = cfg.model.da
    if "runner" in da_cfg:
        raise ValueError(
            "model.da.runner is no longer supported: the assimilation runners "
            "run independent DA models only. Remove the key."
        )
    model = load_assimilation(da_cfg)
    obs = ObsSources.from_config(
        model,
        da_cfg.get("obs_sources"),
        output_path=str(cfg.output.path),
        prefer_predownloaded=prefer_predownloaded,
    )
    return model, obs, da_cfg


def assimilation(
    cfg: DictConfig, prefer_predownloaded: bool = True
) -> AssimilationRunner:
    """Build the analysis-mode runner: one DA analysis per work item.

    Parameters
    ----------
    cfg : DictConfig
        Campaign config with ``model.da`` (``architecture``, ``load_args``,
        ``obs_sources``, optional ``obs_device``) and ``random_seed``.
    prefer_predownloaded : bool, optional
        Read predownloaded observation frame stores when present, by default True.

    Returns
    -------
    AssimilationRunner
        Runner yielding the ``analysis`` stream.
    """
    model, obs, da_cfg = _da_components(cfg, prefer_predownloaded)
    return AssimilationRunner(
        model,
        obs,
        seed=int(cfg.get("random_seed", 42)),
        obs_device=da_cfg.get("obs_device", "cpu"),
    )


def assimilation_forecast(
    cfg: DictConfig, prefer_predownloaded: bool = True
) -> EnsembleRunner:
    """Build the DA-initialized forecast runner.

    Builds the forecast runner of :func:`src.runners.forecast` on
    ``cfg.model.forecast`` with an :class:`AnalysisSource` as its
    initial-condition source.  When ``cfg.model.fill_missing_variables``
    (default true) allows it, the configured data source fills prognostic
    input variables the analysis lacks.  The
    returned runner gains ``frame_requests`` (observations behind every
    analysis an item needs) and ``frame_roles``; ``source_roles`` carries the
    ``data`` role only when a fill source is in use, so predownload caches
    exactly the fill variables.

    Parameters
    ----------
    cfg : DictConfig
        Campaign config with ``model.da`` and ``model.forecast`` nodes.
    prefer_predownloaded : bool, optional
        Read predownloaded stores when present, by default True.

    Returns
    -------
    EnsembleRunner
        Forecast runner initialized from DA analyses.
    """
    if "forecast" not in cfg.model:
        raise ValueError(
            "runner 'assimilation_forecast' needs a 'model.forecast' node "
            "(see cfg/model/healda_ucast.yaml)."
        )
    model, obs, da_cfg = _da_components(cfg, prefer_predownloaded)
    seed = int(cfg.get("random_seed", 42))
    analysis = AnalysisSource(
        model, obs, seed=seed, obs_device=da_cfg.get("obs_device", "cpu")
    )
    runner = build_forecast(
        cfg, prefer_predownloaded, model_node=cfg.model.forecast, source=analysis
    )
    needed = [str(v) for v in runner.input_signature["variable"].values]
    recommended = recommended_sources(runner.prognostic)
    if isinstance(recommended, tuple):
        recommended = recommended[0]
    source, fill, missing = initial_condition_source(
        cfg, analysis, needed, prefer_predownloaded, fallback=recommended
    )
    runner.source = source
    roles = {k: v for k, v in runner.source_roles.items() if k != "data"}
    if fill is not None:
        roles["data"] = fill
    runner.source_roles = roles
    runner.frame_roles = obs.roles  # type: ignore[attr-defined]
    ic_leads = runner.input_signature["lead_time"].values
    original_requests = runner.data_requests

    def frame_requests(item: WorkItem) -> tuple[FrameRequest, ...]:
        times = np.array([item.time + lt for lt in ic_leads])
        return analysis.frame_requests(times)

    def data_requests(item: WorkItem) -> tuple[DataRequest, ...]:
        requests = []
        for request in original_requests(item):
            if request.source is not source:
                requests.append(request)
            elif fill is not None:
                requests.append(
                    DataRequest(
                        fill, request.time, np.array(missing), request.lead_time
                    )
                )
        return tuple(requests)

    runner.frame_requests = frame_requests  # type: ignore[attr-defined]
    runner.data_requests = data_requests  # type: ignore[method-assign]
    return runner
