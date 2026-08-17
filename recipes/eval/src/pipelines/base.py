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

"""Pipeline abstraction for eval recipe inference loops.

The :class:`Pipeline` ABC defines a standard interface that decouples the
*per-work-item inference logic* from the shared scaffolding (work iteration,
output coordinate filtering, ensemble dimension injection, and zarr writes).

To add a custom pipeline, subclass :class:`Pipeline`, implement the three
required methods (:meth:`setup`, :meth:`build_total_coords`,
:meth:`run_item`), and point ``cfg.pipeline`` at its fully qualified class
path.

Pipelines may also declare:

* :meth:`Pipeline.predownload_stores` — what data needs to be pre-fetched
  for this pipeline (consumed by ``predownload.py``).
* A custom ensemble-injection path by overriding :meth:`_inject_ensemble`
  (e.g. models that already carry ensemble along a batch dimension).
"""

from __future__ import annotations

import glob
import os
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, cast

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from earth2studio.data import DataSource
from earth2studio.utils.coords import CoordSystem, map_coords
from src.data import CompositeSource, PredownloadedSource
from src.distributed import get_rank
from src.output import OutputManager, build_output_coords
from src.regrid import Regridder
from src.work import WorkItem, write_marker

_STRUCTURAL_DIMS = frozenset({"batch", "time", "lead_time", "variable", "ensemble"})


def _predownload_merged_verification(cfg: DictConfig) -> bool:
    """Whether ``data.zarr`` is expected to hold verification data."""
    pd_cfg = cfg.get("predownload", {}) or {}
    verif_cfg = pd_cfg.get("verification", {}) or {}
    return bool(verif_cfg.get("enabled", False)) and verif_cfg.get("source") is None


def default_verification_zarr_paths(cfg: DictConfig) -> list[str]:
    """Default verification zarr discovery: verification.zarr → data.zarr → glob.

    The logic used by :meth:`Pipeline.verification_zarr_paths`'s default
    implementation, exposed as a module-level helper so consumers that
    don't have a :class:`Pipeline` instance (e.g. narrow report tests
    that omit ``cfg.pipeline``) can reuse the same discovery.

    Returns whichever set matches first:

    1. ``<output.path>/verification.zarr`` → ``[verification.zarr]``
    2. ``<output.path>/data.zarr`` → ``[data.zarr]`` — only when predownload
       was run in the merged-verification mode (``verification.enabled=true``
       with a null ``verification.source``).  When predownload cached IC
       data only, ``data.zarr`` is skipped even if present, because its
       time axis does not cover the full verification window.
    3. ``<output.path>/data_*.zarr`` glob → sorted list (possibly empty)
    """
    verif_path = os.path.join(cfg.output.path, "verification.zarr")
    if os.path.exists(verif_path):
        return [verif_path]
    if _predownload_merged_verification(cfg):
        data_path = os.path.join(cfg.output.path, "data.zarr")
        if os.path.exists(data_path):
            return [data_path]
    return sorted(
        p
        for p in glob.glob(os.path.join(cfg.output.path, "data_*.zarr"))
        if os.path.isdir(p)
    )


# ======================================================================
# Predownload declarations
# ======================================================================


@dataclass(frozen=True)
class PredownloadStore:
    """Declarative spec for a zarr store that ``predownload.py`` should populate.

    Each pipeline returns a list of these from :meth:`Pipeline.predownload_stores`;
    ``predownload.py`` then iterates, downloading each store's data with resume
    support and per-timestamp progress markers.

    Parameters
    ----------
    name : str
        Logical store name.  Controls the on-disk location
        (``<output.path>/<name>.zarr``) and the progress-marker namespace.
        Typical values: ``"data"``, ``"verification"``, or pipeline-specific
        names like ``"data_goes"``, ``"data_mrms"``, ``"cond_gfs"``.
    source : DataSource
        Data source to fetch from.  May be a :class:`~src.regrid.RegriddedSource`
        that wraps a raw Earth2Studio source with a regridder — in that case the
        stored data is already on the target grid.
    times : list[np.datetime64]
        Timestamps to fetch (analysis times; no lead-time expansion).
    variables : list[str]
        Variable names to fetch.
    spatial_ref : CoordSystem
        Spatial coordinate system of the **stored** data (post-regrid if the
        source is a :class:`~src.regrid.RegriddedSource`).  Used to build the
        zarr schema.
    role : str
        Informational tag: ``"ic"``, ``"verification"``, ``"conditioning"``,
        or ``"data"``.  Shown in logs and can be used to sort / filter
        downstream.  Defaults to ``"data"``.
    extra_coords : dict[str, np.ndarray]
        Optional non-dimensional coordinate arrays to attach to the store
        (e.g. 2D ``lat``/``lon`` alongside ``y``/``x`` for Lambert-conformal
        grids).  Not currently used by the default store writer; reserved
        for future zarr attribute attachment.
    """

    name: str
    source: DataSource
    times: list[np.datetime64]
    variables: list[str]
    spatial_ref: CoordSystem
    role: str = "data"
    extra_coords: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class PredownloadFrameStore:
    """Declarative spec for a DataFrame store that ``predownload.py`` populates.

    The tabular twin of :class:`PredownloadStore`, used for observation
    inputs of data-assimilation pipelines (``DataFrameSource`` protocol
    rather than gridded ``DataSource``).  ``predownload.py`` fetches one
    DataFrame per timestamp via
    :func:`earth2studio.data.fetch_dataframe` and writes it as a parquet
    file under ``<output.path>/<name>.parquet/`` (one file per time,
    named by :func:`src.data.frame_filename`).  At inference time
    :class:`~src.data.PredownloadedFrameSource` serves the store in place
    of the live source.

    Parameters
    ----------
    name : str
        Logical store name.  Controls the on-disk location
        (``<output.path>/<name>.parquet``) and the progress-marker
        namespace.  DA pipelines use ``"obs_<slot>"`` (e.g.
        ``"obs_conv"``, ``"obs_sat"``).
    source : object
        Live ``DataFrameSource`` to fetch from.
    times : list[np.datetime64]
        Analysis times to fetch.  Each becomes the ``request_time`` of
        one fetch — sources with a ``time_tolerance`` window expand it
        internally, so the stored frame holds the full observation
        window for that analysis time.
    variables : list[str]
        Variable names to request (the schema's ``variable`` entry).
    fields : list[str]
        DataFrame columns to request (the schema's keys).
    role : str
        Informational tag shown in logs.  Defaults to ``"observation"``.
    """

    name: str
    source: object
    times: list[np.datetime64]
    variables: list[str]
    fields: list[str]
    role: str = "observation"


# ======================================================================
# Pipeline ABC
# ======================================================================


class Pipeline(ABC):
    """Abstract base class for eval recipe inference pipelines.

    A pipeline encapsulates three concerns that vary between inference modes:

    1. **Setup** — load models, move them to device, and cache any metadata
       needed during inference (e.g. input coordinate systems).
    2. **Output coordinates** — define the full shape of the zarr output store
       (ensemble, time, lead_time, spatial dims).
    3. **Per-item inference** — given a single :class:`WorkItem`, fetch data,
       run models, and *yield* ``(tensor, coords)`` pairs that should be
       written to the output store.

    Everything else — iterating over work items, building the output variable
    filter, optionally regridding model outputs, injecting the ensemble
    dimension, and calling :meth:`OutputManager.write` — is handled by the
    shared :meth:`run` method and does **not** need to be reimplemented.

    Optional hooks
    --------------
    * :meth:`predownload_stores` — declarative predownload spec consumed by
      ``predownload.py``.  Default returns an empty list.
    * :meth:`_inject_ensemble` — how the ensemble dim is expressed in the
      output.  Default prepends a new ``ensemble`` axis; override for
      pipelines that already carry ensemble along another dim (e.g.
      ``batch``).

    Subclassing guide
    -----------------
    Implement these three abstract methods:

    * :meth:`setup` — called once before inference begins.
    * :meth:`build_total_coords` — called once to create the zarr store schema.
    * :meth:`run_item` — called once per work item; yield output tensors.

    The base class provides :attr:`_spatial_ref` which subclasses must set
    during :meth:`setup`.  It is used by :meth:`run` to build the output
    coordinate filter (variable + lat/lon sub-selection).
    """

    _spatial_ref: CoordSystem
    """Reference coordinate system whose spatial dimensions define the output
    grid.  Must be set by :meth:`setup`.  When an output regridder is
    configured, :meth:`effective_spatial_ref` returns the *regridded* target
    coords; ``_spatial_ref`` always refers to the model's native grid."""

    _output_regridder: Regridder | None = None
    """Optional output-side regridder.  When set (by subclasses during
    :meth:`setup`), model outputs are regridded between :meth:`run_item` and
    :meth:`OutputManager.write`, and the zarr store uses the regridder's
    target coords.  Concrete regridders are added in later phases."""

    needs_data_source: bool = True
    """Whether this pipeline consumes the single ``DataSource`` that
    ``main.py`` resolves from ``cfg.ic_source`` / predownloaded cache /
    ``cfg.data_source``.

    Single-source pipelines (``forecast``, ``diagnostic``, ``dlesym``)
    leave this ``True`` — ``main.py`` hands them the resolved source,
    which they use inside :meth:`run_item` for ``fetch_data`` calls.

    Multi-source pipelines (``stormscope``, which needs separate IC
    sources per model plus a conditioning source) should set this to
    ``False`` — ``main.py`` then skips source instantiation and passes
    ``None`` as the ``data_source`` argument.  BYO is handled inside
    the pipeline's own config block (e.g. overriding
    ``cfg.model.goes.ic_source._target_``)."""

    supports_online_scoring: bool = False
    """Whether this pipeline may be driven by the online scorer.

    Online scoring rests on one invariant:

        For a fixed IC, the sequence of coords yielded by :meth:`run_item`
        is identical across ensemble members.

    Every rank in an ensemble group calls :meth:`run_item` on the same IC
    with a different ``ensemble_id`` and the per-lead-step group reduction
    is what synchronizes them — so a member-dependent yield sequence
    deadlocks the group rather than producing a wrong answer.  (The scorer
    cross-checks each step's lead time across the group when
    ``scoring.online.validate_coords`` is set, turning that deadlock into a
    clean error.)

    The flag defaults to ``False``: support is an explicit, per-pipeline
    claim rather than something inherited for free.  A pipeline sets it to
    ``True`` only once its yield sequence and seeding have been validated
    (grouping invariance, online-vs-offline parity, member reproducibility
    — see the recipe's online-scoring validation gates)."""

    _members_per_rank: int = 1
    """Ensemble members this rank carries per rollout (``K``).  Set by
    :meth:`run` from its ``member_batch`` argument before the first
    :meth:`run_item` / :meth:`run_item_batched` call.  Consulted by the
    default :meth:`seed_member` for sample-indexed stochastic components."""

    _run_item_includes_batch_dim: bool = False
    """Whether :meth:`run_item` yields tensors with a leading ``batch`` dim.

    Most models' ``__call__`` outputs carry only the structural dims
    ``(time, lead_time, variable, <spatial...>)``.  Some — notably
    StormScope's diffusion sampler — retain an internal ``batch`` axis
    with size 1.  Setting this flag to ``True`` lets :meth:`run` squeeze
    that axis before output filtering / regridding / ensemble injection,
    so subclasses don't need to do it manually."""

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------

    @abstractmethod
    def setup(self, cfg: DictConfig, device: torch.device) -> None:
        """Load models and prepare pipeline state.

        Called exactly once before any inference begins.  Implementations
        should:

        * Load and ``.to(device)`` all models.
        * Cache any coordinate metadata needed by :meth:`run_item`
          (e.g. ``input_coords()`` lookups).
        * Set ``self._spatial_ref`` to the model's native spatial coord
          system.
        * Optionally build an output regridder and assign it to
          ``self._output_regridder`` (concrete regridder classes land in
          later phases).

        Parameters
        ----------
        cfg : DictConfig
            Full Hydra config.
        device : torch.device
            Target device for inference.
        """
        ...

    @abstractmethod
    def build_total_coords(
        self,
        times: np.ndarray,
        ensemble_size: int,
    ) -> CoordSystem:
        """Build the full output coordinate system for the zarr store.

        The returned ``CoordSystem`` defines every dimension and its values
        for the output arrays.  It is passed directly to
        :meth:`OutputManager.validate_output_store` to create or validate
        the zarr store.

        Typical dimensions: ``[ensemble], time, lead_time, <spatial...>``.
        The ``ensemble`` dimension should be included only when
        ``ensemble_size > 1``.  Spatial dims should come from
        :meth:`effective_spatial_ref` so that output regridders, when
        configured, are honored.

        Parameters
        ----------
        times : np.ndarray
            All initial-condition times (sorted, unique).
        ensemble_size : int
            Total number of ensemble members.  When 1 the ensemble
            dimension should be omitted.

        Returns
        -------
        CoordSystem
            Full coordinate system for the output store.
        """
        ...

    @abstractmethod
    def run_item(
        self,
        item: WorkItem,
        data_source: DataSource,
        device: torch.device,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Run inference for a single work item and yield output chunks.

        This is the core method that varies between pipelines.  It should:

        1. Fetch input data for the work item's time.
        2. Run the pipeline's model(s).
        3. ``yield`` one or more ``(tensor, coords)`` pairs on the **model's
           native grid** (:attr:`_spatial_ref`).

        Each yielded pair is automatically:

        * filtered to the requested output variables and native spatial grid,
        * regridded to the output grid (if an output regridder is configured),
        * decorated with an ensemble dimension if needed, and
        * written to the output store.

        Implementations do **not** need to handle any of that.

        Parameters
        ----------
        item : WorkItem
            The work item to process (contains time, ensemble_id, seed).
        data_source : DataSource
            Source for fetching input data.
        device : torch.device
            Target device for tensors.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            ``(data, coords)`` pairs to be written to the output store.
            These should include all variables the pipeline produces
            (the caller handles sub-selection to output variables).
        """
        ...

    # ------------------------------------------------------------------
    # Optional hooks with defaults
    # ------------------------------------------------------------------

    def run_item_batched(
        self,
        items: list[WorkItem],
        data_source: DataSource,
        device: torch.device,
    ) -> Iterator[tuple[torch.Tensor, CoordSystem]]:
        """Run several ensemble members of one IC through a single rollout.

        The member-batched twin of :meth:`run_item`, used by online scoring
        when a rank carries more than one member
        (``scoring.online.members_per_rank > 1``).  All *items* share an
        initial-condition time and differ only in ``ensemble_id`` / ``seed``.

        Why it exists: with one member per rank, an ensemble group spans
        ``M`` ranks — typically several nodes — and the CRPS member
        exchange runs over the inter-node fabric.  Batching ``K`` members
        onto each rank shrinks the group to ``M/K`` ranks, which is what
        lets a group fit inside a node and move that exchange onto NVLink
        (~10x cheaper).  It also unblocks running an ensemble at all when
        ``world_size < M``.

        Implementations must yield ``(tensor, coords)`` pairs whose coords
        carry an ``ensemble`` axis holding *exactly* the items'
        ``ensemble_id`` values, in order — the online scorer writes
        per-member statistics at those absolute indices and validates them.

        Parameters
        ----------
        items : list[WorkItem]
            Members of one IC to run together, in ensemble-id order.
        data_source : DataSource
            Source for fetching input data.
        device : torch.device
            Target device for tensors.

        Yields
        ------
        tuple[torch.Tensor, CoordSystem]
            ``(data, coords)`` pairs with a leading ``ensemble`` dimension.

        Raises
        ------
        NotImplementedError
            By default.  Pipelines opt in by overriding this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run_item_batched, so it "
            "cannot carry more than one ensemble member per rank.  Either set "
            "scoring.online.members_per_rank=1 (and launch at least "
            "`ensemble_size` ranks), or implement the batched rollout."
        )

    def supports_member_batching(self) -> bool:
        """Whether :meth:`run_item_batched` is implemented.

        Checked by ``main.py`` before work is distributed, so an
        unsupported ``members_per_rank`` fails at startup rather than after
        the models have loaded.
        """
        return type(self).run_item_batched is not Pipeline.run_item_batched

    def stochastic_components(self) -> list[Any]:
        """Models this pipeline drives whose output depends on RNG state.

        Consulted by :meth:`seed_member`.  Default: empty — a pipeline
        with no stochastic component needs no per-member seeding and its
        ensemble is degenerate by construction.
        """
        return []

    def seed_member(self, item: WorkItem) -> None:
        """Place every stochastic component in the state for member ``item.ensemble_id``.

        Call this from :meth:`run_item` / :meth:`run_item_batched` before
        the first model invocation.  ``torch.manual_seed(item.seed)`` runs
        unconditionally first, so perturbation methods and any incidental
        global-RNG use are covered regardless of what the model itself
        does.  Each entry from :meth:`stochastic_components` is then
        dispatched by the mechanism it exposes:

        * has ``set_rng`` — ``component.set_rng(seed=item.seed, reset=True)``.
        * has ``number_of_samples`` and ``seed`` — sample-indexed draw;
          ``component.seed = item.seed`` and ``component.number_of_samples``
          is set from :attr:`_members_per_rank`.  Member ``m`` of IC ``t``
          is then a reproducible, independent draw indexed by ``(t, m)`` —
          it is not required to equal the ``m``-th sample of a
          single-process, ``M``-sample call.
        * neither — global-RNG component; already covered by the
          unconditional ``torch.manual_seed`` above.

        Override only for a mechanism genuinely new to these three.
        """
        torch.manual_seed(item.seed)
        for component in self.stochastic_components():
            if hasattr(component, "set_rng"):
                component.set_rng(seed=item.seed, reset=True)
            elif hasattr(component, "number_of_samples") and hasattr(
                component, "seed"
            ):
                component.seed = item.seed
                component.number_of_samples = self._members_per_rank

    def known_missing_leads(self) -> set[np.timedelta64]:
        """Lead times this pipeline structurally never yields.

        Consulted by the online scorer's ``finish_item`` so its
        missing-lead-time warning fires only for a genuine gap (a crashed
        or partially resumed IC) rather than every single IC.  Default:
        empty — most pipelines yield every lead time in their own output
        schema.  ``DLESyMPipeline`` overrides this to declare
        ``lead_time=0``, which its ``run_item`` deliberately skips (the
        model's IC-window yield falls outside the output schema).
        """
        return set()

    def predownload_stores(self, cfg: DictConfig) -> list[PredownloadStore]:
        """Declare zarr stores that ``predownload.py`` should populate.

        Default implementation returns an empty list, meaning the pipeline
        requires no pre-fetched data.  Concrete pipelines should override
        to declare their IC, verification, and conditioning needs.

        Parameters
        ----------
        cfg : DictConfig
            Full Hydra config.  Pipelines read ``data_source``,
            ``ic_source`` / ``verification_source`` (for BYO skips),
            ``start_times``, ``nsteps``, and ``output.variables`` from
            this.

        Returns
        -------
        list[PredownloadStore]
            One entry per zarr store to populate.
        """
        return []

    def predownload_frame_stores(self, cfg: DictConfig) -> list[PredownloadFrameStore]:
        """Declare DataFrame stores that ``predownload.py`` should populate.

        The tabular counterpart of :meth:`predownload_stores`, for
        pipelines whose inputs include ``DataFrameSource`` observations
        (the DA pipelines).  Default implementation returns an empty
        list.

        Parameters
        ----------
        cfg : DictConfig
            Full Hydra config.

        Returns
        -------
        list[PredownloadFrameStore]
            One entry per parquet store to populate.
        """
        return []

    def verification_zarr_paths(self, cfg: DictConfig) -> list[str]:
        """Return local zarr paths that together provide verification data.

        Default resolution (first non-empty match wins):

        1. ``<output.path>/verification.zarr`` — separate verification store.
        2. ``<output.path>/data.zarr`` — merged IC + verification store.
        3. ``<output.path>/data_*.zarr`` glob — per-model stores (e.g.
           StormScope's ``data_goes.zarr`` + ``data_mrms.zarr``).

        Pipelines that use non-standard store names can override this.
        The hook is consulted by :meth:`verification_source` (for scoring)
        and by the report package (for visualization), so overriding in
        one place reaches both consumers.

        Parameters
        ----------
        cfg : DictConfig
            Full Hydra config.

        Returns
        -------
        list[str]
            Zarr store paths that exist on disk.  Empty when no
            predownload store is available.
        """
        return default_verification_zarr_paths(cfg)

    def verification_source(self, cfg: DictConfig) -> DataSource:
        """Return the verification data source for scoring.

        Resolution order:

        1. ``cfg.verification_source`` — user-provided ``DataSource`` (BYO).
        2. :meth:`verification_zarr_paths` — local predownloaded zarrs.

           * One path → :class:`~src.data.PredownloadedSource`.
           * Multiple paths → :class:`~src.data.CompositeSource` dispatching
             variables to whichever store provides them.

        Parameters
        ----------
        cfg : DictConfig
            Full Hydra config.

        Returns
        -------
        DataSource

        Raises
        ------
        FileNotFoundError
            If no override is provided and no predownloaded zarr exists.
        """
        if cfg.get("verification_source") is not None:
            logger.info("Using user-provided verification_source (BYO).")
            return hydra.utils.instantiate(cfg.verification_source)

        paths = self.verification_zarr_paths(cfg)
        if not paths:
            data_path = os.path.join(cfg.output.path, "data.zarr")
            hint = (
                " (found 'data.zarr' but it holds IC-only data — set "
                "predownload.verification.enabled=true to cache the "
                "forecast-window valid times used by scoring.)"
                if os.path.exists(data_path)
                else ""
            )
            raise FileNotFoundError(
                f"No verification data found in '{cfg.output.path}'.{hint}\n"
                "Run predownload.py with predownload.verification.enabled=true, "
                "or set cfg.verification_source to a BYO override."
            )
        if len(paths) == 1:
            logger.info(f"Using verification store: {paths[0]}")
            return PredownloadedSource(paths[0])

        logger.info(
            "Verification: merging per-model stores "
            f"({', '.join(os.path.basename(p) for p in paths)})"
        )
        stores = {os.path.basename(p).removesuffix(".zarr"): p for p in paths}
        return CompositeSource.from_predownloaded_stores(stores)

    def _inject_ensemble(
        self,
        x: torch.Tensor,
        coords: CoordSystem,
        item: WorkItem,
        has_ensemble: bool,
    ) -> tuple[torch.Tensor, CoordSystem]:
        """Inject the ensemble dimension before writing.

        Default: ``unsqueeze(0)`` on the tensor and prepend an ``ensemble``
        key carrying ``[item.ensemble_id]`` to the coord system.  Override
        for pipelines that already carry ensemble members along another
        dimension (e.g. ``batch`` for StormScope).

        Parameters
        ----------
        x : torch.Tensor
            Output tensor from :meth:`run_item`, post regrid and filter.
        coords : CoordSystem
            Matching coord system.
        item : WorkItem
            The work item being written (supplies ``ensemble_id``).
        has_ensemble : bool
            True when the output zarr has an ``ensemble`` dimension.

        Returns
        -------
        tuple[torch.Tensor, CoordSystem]
            Possibly-modified ``(x, coords)`` pair.
        """
        if not has_ensemble or "ensemble" in coords:
            # A batched rollout already carries its members on an
            # ``ensemble`` axis, so there is nothing to inject.
            return x, coords
        x = x.unsqueeze(0)
        coords = CoordSystem({"ensemble": np.array([item.ensemble_id])} | dict(coords))
        return x, coords

    def effective_spatial_ref(self) -> CoordSystem:
        """Return the spatial coord system used for the output zarr.

        If an output regridder is configured, returns its target coords
        (the coarsened grid); otherwise returns :attr:`_spatial_ref`
        (the model's native grid).
        """
        if self._output_regridder is None:
            return self._spatial_ref

        merged: CoordSystem = OrderedDict()
        # Preserve any non-spatial keys (e.g. a lingering 'variable' from
        # model output_coords) — in practice spatial_ref only has spatial
        # dims, so this loop is usually a no-op.
        for d, v in self._spatial_ref.items():
            if d in _STRUCTURAL_DIMS:
                merged[d] = v
        for d, v in self._output_regridder.target_coords().items():
            merged[d] = v
        return merged

    # ------------------------------------------------------------------
    # Shared machinery
    # ------------------------------------------------------------------

    @torch.no_grad()  # not using inference_mode() due to problem with DLWP
    def run(
        self,
        work_items: list[WorkItem],
        data_source: DataSource | None,
        output_mgr: OutputManager | None,
        output_variables: list[str],
        device: torch.device,
        cfg: DictConfig | None = None,
        scorer: Any = None,
        member_batch: int = 1,
    ) -> None:
        """Iterate work items, filter+regrid outputs, and write/score them.

        Shared outer loop.  For each work item:

        1. Calls :meth:`run_item`.
        2. Filters each yielded chunk to the configured output variables
           and the model's native spatial grid.
        3. If an output regridder is configured, applies it.
        4. Hands the chunk to the online scorer, when one is supplied.
        5. Injects the ensemble dim via :meth:`_inject_ensemble` and writes
           the chunk via :class:`OutputManager`, when one is supplied.

        Steps 4 and 5 are independent: an online run with
        ``output.retain=none`` scores without ever materializing
        ``forecast.zarr``, while ``output.retain=all`` does both.

        When ``cfg.resume`` is true, a completion marker is written after
        each work item so that resumed runs can skip it.  Online runs track
        their own, IC-granular markers inside the scorer instead — an
        ensemble group's unit of durability is the IC, not the member.

        Parameters
        ----------
        work_items : list[WorkItem]
            Work items assigned to this rank.
        data_source : DataSource | None
            Source for fetching input data.  ``None`` is allowed for
            pipelines that set :attr:`needs_data_source` to ``False``
            (they manage their own sources internally).
        output_mgr : OutputManager | None
            Context-managed output handler (store already validated), or
            ``None`` to skip raw output entirely.
        output_variables : list[str]
            Variable names to sub-select before writing.
        device : torch.device
            Target device for inference.
        cfg : DictConfig | None
            Full Hydra config.  Required when ``resume=true`` so that
            completion markers can be written.
        scorer : src.online.OnlineScorer | None
            When set, each yielded chunk is accumulated into sufficient
            statistics as it is produced.  Typed loosely to keep
            :mod:`src.online` (and its distributed machinery) off the
            import path of runs that don't use it.
        member_batch : int
            Ensemble members to run together per rollout.  ``1`` (the
            default) drives :meth:`run_item` once per work item.  Greater
            than 1 groups *work_items* into consecutive blocks of that size
            — which :func:`src.work.build_group_work_items` lays out as one
            block per IC — and drives :meth:`run_item_batched` instead.
        """
        if not work_items:
            logger.warning("No work items for this rank — skipping inference.")
            return
        if output_mgr is None and scorer is None:
            raise ValueError(
                "Pipeline.run needs an output manager, an online scorer, or "
                "both — otherwise the run produces nothing."
            )
        if member_batch > 1 and len(work_items) % member_batch:
            raise ValueError(
                f"member_batch={member_batch} does not divide the "
                f"{len(work_items)} assigned work items, so the batches would "
                "straddle initial conditions."
            )

        resume = cfg.get("resume", False) if cfg is not None else False
        self._members_per_rank = member_batch

        # Filter at the model's native grid — cheaper than regridding
        # all model-output channels only to throw some away.
        native_filter = build_output_coords(self._spatial_ref, output_variables)
        has_ensemble = output_mgr is not None and "ensemble" in output_mgr.io.coords

        # Move output regridder to device once.
        if self._output_regridder is not None:
            self._output_regridder = self._output_regridder.to(device)

        # Rank only gates tqdm output below.
        rank = get_rank()

        # None is only passed when needs_data_source=False, in which case
        # the subclass's run_item ignores the argument.
        ds = cast(DataSource, data_source)
        batches = [
            work_items[i : i + member_batch]
            for i in range(0, len(work_items), member_batch)
        ]
        for batch in tqdm(batches, desc="Work items", position=0, disable=rank != 0):
            # Every member of a batch shares an IC; the first stands in for
            # the batch wherever a single item is needed (markers, seeds).
            item = batch[0]
            if scorer is not None:
                scorer.begin_item(item)

            steps = (
                self.run_item(item, ds, device)
                if member_batch == 1
                else self.run_item_batched(batch, ds, device)
            )
            for x_step, coords_step in steps:
                if self._run_item_includes_batch_dim and "batch" in coords_step:
                    batch_axis = list(coords_step.keys()).index("batch")
                    x_step = x_step.squeeze(batch_axis)
                    coords_step = OrderedDict(
                        (k, v) for k, v in coords_step.items() if k != "batch"
                    )
                x_out, coords_out = map_coords(x_step, coords_step, native_filter)

                if self._output_regridder is not None:
                    x_out, coords_out = self._output_regridder.apply_with_coords(
                        x_out, coords_out
                    )

                if scorer is not None:
                    scorer.update(x_out, coords_out)

                if output_mgr is not None:
                    x_write, coords_write = self._inject_ensemble(
                        x_out, coords_out, item, has_ensemble
                    )
                    output_mgr.write(x_write, coords_write)

            if scorer is not None:
                scorer.finish_item(item)

            if resume and output_mgr is not None:
                output_mgr.flush()
                for done in batch:
                    write_marker(done, cfg)

        logger.success("Inference complete.")
