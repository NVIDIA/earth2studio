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

"""Shared plumbing for data-assimilation (DA) models.

Everything specific to the :class:`~earth2studio.models.da.base.AssimilationModel`
protocol lives here so the DA pipelines in ``src/pipelines/assimilation.py``
stay thin:

* :func:`load_assimilation` — load a DA model from a config node (mirrors
  :func:`src.models.load_prognostic`, but takes the node directly since DA
  configs nest the model under ``cfg.model.da``).
* :class:`ObsSourceSet` — resolve and fetch the model's observation inputs.
  DA models take a *tuple* of observation DataFrames (one per entry in
  ``model.input_coords()``), each from its own ``DataFrameSource``.
* :class:`AssimilationRunner` / :class:`StatelessAssimilationRunner` —
  produce an analysis at a requested time.  The runner seam is what keeps
  stateless models (HealDA: independent per-time calls) and future stateful
  models (StormCastSDA: sequential cycling with a background state) behind
  the same pipeline-facing API.
* :func:`analysis_to_tensor` / :func:`insert_zero_lead_time` /
  :func:`analysis_spatial_ref` — convert DA model output (an
  ``xr.DataArray``, possibly cupy-backed on CUDA) into the ``(tensor,
  coords)`` shape the shared :meth:`Pipeline.run` loop expects.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

import hydra
import numpy as np
import torch
import xarray as xr
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from earth2studio.data import fetch_dataframe
from earth2studio.models.da.base import AssimilationModel
from earth2studio.utils.coords import CoordSystem

from .distributed import run_on_rank0_first
from .output import _spatial_dims


def load_assimilation(da_cfg: DictConfig) -> AssimilationModel:
    """Load a data-assimilation model from its config node.

    Mirrors :func:`src.models.load_prognostic` but takes the model node
    directly (DA configs nest it under ``cfg.model.da``, next to a
    ``forecast`` node when a DA-initialized forecast is configured).  The
    node must contain an ``architecture`` key with the fully-qualified
    class name of an :class:`AssimilationModel`; the class is expected to
    expose the standard ``load_default_package`` / ``load_model``
    classmethods from the ``AutoModelMixin`` protocol.

    Any extra keyword arguments under ``load_args`` are forwarded to
    ``load_model`` (e.g. HealDA's ``lat_lon`` / ``output_resolution``).

    Parameters
    ----------
    da_cfg : DictConfig
        DA model config node (typically ``cfg.model.da``).

    Returns
    -------
    AssimilationModel
        Loaded (but not yet device-placed) assimilation model.
    """
    cls = hydra.utils.get_class(da_cfg.architecture)

    if da_cfg.get("package_path"):
        from earth2studio.models.auto import Package

        pkg = Package(da_cfg.package_path)
    else:
        pkg = run_on_rank0_first(cls.load_default_package)

    load_kwargs: dict[str, Any] = dict(da_cfg.get("load_args", {}))
    model: AssimilationModel = cls.load_model(package=pkg, **load_kwargs)

    logger.success(f"Loaded assimilation model: {cls.__name__}")
    return model


class ObsSourceSet:
    """Resolves and fetches a DA model's observation inputs.

    A DA model's ``input_coords()`` returns a tuple of schemas, one per
    observation argument that ``__call__`` accepts.  This class holds one
    ``DataFrameSource`` (or ``None`` for a disabled slot) per schema, in
    the same order, and fetches all of them for a given analysis time via
    :func:`earth2studio.data.fetch_dataframe` — which attaches the
    ``request_time`` attrs entry that DA models rely on.

    Parameters
    ----------
    sources : list
        One ``DataFrameSource`` instance (or ``None``) per model input
        argument, positionally matched to ``model.input_coords()``.
    schemas : tuple
        The model's input schemas (``model.input_coords()``), used to
        derive the ``variable`` and ``fields`` arguments of each fetch.
    names : list[str], optional
        One logical name per source (the ``obs_sources`` config keys).
        Used to derive predownload store names (``obs_<name>``).
        Defaults to ``obs0, obs1, ...``.
    """

    def __init__(
        self,
        sources: list[Any],
        schemas: tuple[Any, ...],
        names: list[str] | None = None,
    ) -> None:
        if len(sources) != len(schemas):
            raise ValueError(
                f"ObsSourceSet: got {len(sources)} sources for a model with "
                f"{len(schemas)} observation inputs — one config entry is "
                "required per input (use 'enabled: false' to pass None)."
            )
        self._sources = list(sources)
        self._schemas = tuple(schemas)
        if names is None:
            names = [f"obs{i}" for i in range(len(sources))]
        if len(names) != len(sources):
            raise ValueError(
                f"ObsSourceSet: got {len(names)} names for {len(sources)} sources."
            )
        self._names = [str(n) for n in names]

    @classmethod
    def from_config(
        cls,
        obs_cfg: DictConfig,
        schemas: tuple[Any, ...],
        output_path: str | None = None,
    ) -> ObsSourceSet:
        """Build the source set from a config mapping.

        *obs_cfg* is an ordered mapping (e.g. ``cfg.model.da.obs_sources``)
        whose entries are matched **positionally, in declaration order**
        against the model's ``input_coords()`` tuple.  Each entry is a
        Hydra ``_target_`` block plus two optional convenience keys handled
        here rather than passed to the constructor:

        * ``enabled`` — ``false`` disables the slot (``None`` is passed to
          the model, matching the protocol's optional-argument semantics).
        * ``time_tolerance_hours`` — ``[lo, hi]`` pair converted to a
          ``time_tolerance`` tuple of ``np.timedelta64`` hours, since raw
          timedeltas are awkward to express in YAML.

        When *output_path* is provided and a predownloaded frame store
        exists at ``<output_path>/obs_<name>.parquet`` (written by
        ``predownload.py``), that store is used in place of the live
        source — the runtime analogue of the ``data.zarr`` substitution
        performed by :func:`src.data.resolve_ic_source`.

        Parameters
        ----------
        obs_cfg : DictConfig
            Mapping of name → observation source config.
        schemas : tuple
            The model's ``input_coords()`` tuple.
        output_path : str, optional
            Run output directory to check for predownloaded frame
            stores.  ``None`` (e.g. during predownload itself) always
            constructs the live sources.

        Returns
        -------
        ObsSourceSet
        """
        import os

        from .data import PredownloadedFrameSource, frame_store_path

        sources: list[Any] = []
        names: list[str] = []
        for name, node in obs_cfg.items():
            names.append(str(name))
            # Resolve to a plain dict and construct the source directly
            # (rather than via hydra.utils.instantiate) so the numpy
            # time_tolerance tuple below is passed through unchanged — routing
            # it through instantiate would re-wrap it in an OmegaConf ListConfig.
            spec = OmegaConf.to_container(node, resolve=True)
            if not spec.pop("enabled", True):
                logger.info(f"Observation source '{name}' disabled — passing None.")
                sources.append(None)
                continue
            if output_path is not None:
                store = frame_store_path(output_path, f"obs_{name}")
                if os.path.isdir(store):
                    logger.info(
                        f"Observation source '{name}': using predownloaded "
                        f"frame store {store}"
                    )
                    sources.append(PredownloadedFrameSource(store))
                    continue
            target = spec.pop("_target_", None)
            if target is None:
                raise ValueError(
                    f"Observation source '{name}' must have a '_target_' key "
                    f"(got keys: {sorted(spec)})."
                )
            tol = spec.pop("time_tolerance_hours", None)
            if tol is not None:
                lo, hi = tol
                spec["time_tolerance"] = (
                    np.timedelta64(int(lo), "h"),
                    np.timedelta64(int(hi), "h"),
                )
            source_cls = hydra.utils.get_class(target)
            sources.append(source_cls(**spec))
            logger.success(f"Loaded observation source: {name}")
        return cls(sources, schemas, names=names)

    def predownload_frame_stores(self, times: list[np.datetime64]) -> list[Any]:
        """Declare one :class:`PredownloadFrameStore` per enabled source.

        Each store fetches the full observation window for every entry
        of *times* (the analysis times the campaign will request) with
        the ``variable`` / ``fields`` arguments derived from the model
        schema — exactly what :meth:`fetch` will ask for at inference
        time.  Disabled slots (``None`` sources) declare nothing.

        Parameters
        ----------
        times : list[np.datetime64]
            Analysis times the campaign will request.

        Returns
        -------
        list[PredownloadFrameStore]
        """
        from .pipelines.base import PredownloadFrameStore

        stores: list[Any] = []
        for name, source, schema in zip(self._names, self._sources, self._schemas):
            if source is None:
                continue
            stores.append(
                PredownloadFrameStore(
                    name=f"obs_{name}",
                    source=source,
                    times=list(times),
                    variables=[str(v) for v in schema["variable"]],
                    fields=[str(k) for k in schema.keys()],
                )
            )
        return stores

    def fetch(
        self,
        time: np.datetime64,
        device: torch.device | str = "cpu",
    ) -> tuple[Any, ...]:
        """Fetch all observation inputs for one analysis time.

        Parameters
        ----------
        time : np.datetime64
            Analysis valid time.  Becomes ``request_time`` in each
            DataFrame's attrs (via ``fetch_dataframe``).
        device : torch.device | str
            Device for the returned DataFrames — ``"cpu"`` yields pandas,
            CUDA devices yield cudf.  Defaults to ``"cpu"``.

        Returns
        -------
        tuple
            One DataFrame (or ``None`` for disabled slots) per model input,
            in ``input_coords()`` order.
        """
        times = np.array([time], dtype="datetime64[ns]")
        frames: list[Any] = []
        for source, schema in zip(self._sources, self._schemas):
            if source is None:
                frames.append(None)
                continue
            frames.append(
                fetch_dataframe(
                    source,
                    time=times,
                    variable=np.array(schema["variable"]),
                    fields=np.array(list(schema.keys())),
                    device=device,
                )
            )
        return tuple(frames)


class AssimilationRunner(ABC):
    """Produces an analysis field at a requested time.

    This is the seam between the DA pipelines and the shape of the
    underlying model:

    * :class:`StatelessAssimilationRunner` (the default) calls the model
      independently per analysis time — correct for models like HealDA
      whose ``init_coords()`` is ``None``.  Work items can be freely
      distributed across ranks.
    * A future cycling runner for stateful models (e.g. StormCastSDA)
      would hold ``model.create_generator(background)`` open and step it
      sequentially between requested times.  Such a runner constrains work
      distribution (analysis times must be processed in order on one
      rank), which is why the runner — not the pipeline — owns the
      stepping semantics.

    Pipelines select a runner via the optional ``cfg.model.da.runner``
    Hydra block (see :func:`build_runner`); the stateless runner is used
    when no override is configured.
    """

    @abstractmethod
    def analysis(self, time: np.datetime64) -> xr.DataArray:
        """Return the analysis valid at *time* as an ``xr.DataArray``.

        The returned array carries the model's output dims (e.g.
        ``(time, variable, lat, lon)``) and may be cupy-backed when the
        model runs on CUDA — use :func:`analysis_to_tensor` to convert.
        """
        ...


class StatelessAssimilationRunner(AssimilationRunner):
    """Runner for stateless DA models — one independent call per time.

    Parameters
    ----------
    model : AssimilationModel
        Loaded, device-placed DA model.  Must be stateless
        (``init_coords()`` returns ``None``).
    obs_set : ObsSourceSet
        Observation sources matched to the model's inputs.
    obs_device : torch.device | str
        Device for observation DataFrames (``"cpu"`` → pandas, CUDA →
        cudf).  Defaults to ``"cpu"`` since cudf is an optional
        dependency; DA models move observation data themselves.
    """

    def __init__(
        self,
        model: AssimilationModel,
        obs_set: ObsSourceSet,
        obs_device: torch.device | str = "cpu",
    ) -> None:
        init_coords = getattr(model, "init_coords", lambda: None)()
        if init_coords is not None:
            raise NotImplementedError(
                f"{type(model).__name__} requires initialization state "
                "(init_coords() is not None) — it is a stateful/cycling DA "
                "model, which the stateless runner cannot drive.  A cycling "
                "AssimilationRunner that steps model.create_generator() "
                "sequentially is required."
            )
        self._model = model
        self._obs_set = obs_set
        self._obs_device = obs_device

    def analysis(self, time: np.datetime64) -> xr.DataArray:
        obs = self._obs_set.fetch(time, device=self._obs_device)
        if all(o is None for o in obs):
            raise ValueError(
                "All observation sources are disabled — at least one must "
                "be enabled to produce an analysis."
            )
        return self._model(*obs)


def build_runner(
    da_cfg: DictConfig,
    model: AssimilationModel,
    obs_set: ObsSourceSet,
) -> AssimilationRunner:
    """Build the :class:`AssimilationRunner` for a DA model config node.

    When ``da_cfg.runner`` is present it is Hydra-instantiated with
    ``model`` and ``obs_set`` passed as keyword arguments — the hook for
    wiring in cycling runners for stateful models.  Otherwise the
    stateless runner is used, with the optional ``da_cfg.obs_device``
    forwarded.
    """
    runner_cfg = da_cfg.get("runner")
    if runner_cfg is not None:
        runner = hydra.utils.instantiate(runner_cfg, model=model, obs_set=obs_set)
        if not isinstance(runner, AssimilationRunner):
            raise TypeError(
                f"cfg runner resolved to {type(runner).__name__}, which is "
                "not an AssimilationRunner subclass."
            )
        return runner
    return StatelessAssimilationRunner(
        model, obs_set, obs_device=da_cfg.get("obs_device", "cpu")
    )


def analysis_spatial_ref(model: AssimilationModel) -> CoordSystem:
    """Extract the spatial coord system of a DA model's analysis output.

    Takes the first entry of ``model.output_coords(model.input_coords())``
    (the gridded analysis) and keeps only its spatial dims — structural
    dims (``time``, ``variable``) are stripped, so this works for both
    lat/lon and native (e.g. ``npix``) output grids.
    """
    output_coords = model.output_coords(model.input_coords())
    analysis_coords = output_coords[0]
    ref: CoordSystem = OrderedDict()
    for dim in _spatial_dims(analysis_coords):
        ref[dim] = np.asarray(analysis_coords[dim])
    return ref


def analysis_to_tensor(
    da: xr.DataArray,
    device: torch.device,
) -> tuple[torch.Tensor, CoordSystem]:
    """Convert a DA model's output ``xr.DataArray`` to ``(tensor, coords)``.

    Handles both numpy-backed (CPU) and cupy-backed (CUDA) arrays — cupy
    is converted zero-copy via the CUDA array interface.  Data is cast to
    float32 (HealDA's lat/lon regrid path upcasts to float64) and moved
    to *device*.
    """
    data = da.data
    if hasattr(data, "__cuda_array_interface__"):
        x = torch.as_tensor(data)
    else:
        x = torch.from_numpy(np.ascontiguousarray(data))
    x = x.to(device=device, dtype=torch.float32)

    coords: CoordSystem = OrderedDict(
        (str(dim), np.asarray(da.coords[dim].values)) for dim in da.dims
    )
    return x, coords


def insert_zero_lead_time(
    x: torch.Tensor,
    coords: CoordSystem,
) -> tuple[torch.Tensor, CoordSystem]:
    """Insert a singleton ``lead_time=[0ns]`` dim after ``time``.

    Analysis products have no forecast lead, but the output store schema
    (and scoring) key on a ``lead_time`` axis — a single ``0`` entry maps
    the analysis onto the standard layout (the same convention as
    :class:`~src.pipelines.forecast.DiagnosticPipeline`).
    """
    dims = list(coords.keys())
    time_axis = dims.index("time")
    x = x.unsqueeze(time_axis + 1)
    new_coords: CoordSystem = OrderedDict()
    for dim in dims[: time_axis + 1]:
        new_coords[dim] = coords[dim]
    new_coords["lead_time"] = np.array([np.timedelta64(0, "ns")])
    for dim in dims[time_axis + 1 :]:
        new_coords[dim] = coords[dim]
    return x, new_coords


def analysis_variables(model: AssimilationModel) -> list[str]:
    """Return the variable names of a DA model's gridded analysis output."""
    output_coords = model.output_coords(model.input_coords())
    return [str(v) for v in output_coords[0]["variable"]]
