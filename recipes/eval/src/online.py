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

"""Online (in-line) scoring — score forecasts as they are produced.

The default recipe flow is *store-then-score*: ``main.py`` writes every
(member, IC, lead, variable) field to ``forecast.zarr`` and ``score.py``
re-reads it.  At campaign scale that raw store is the binding constraint
(~170 TB for a 50-member, 730-IC, 14-day global campaign), even though the
scores derived from it are ~20 MB.

Online scoring removes the raw store from the critical path.  Ranks are
partitioned into **ensemble groups** (:class:`~src.work.EnsembleGroup`); a
group owns one IC at a time and its ranks carry disjoint slices of the
ensemble.  At every lead step each rank reads the verification field for
``IC + lead`` from the local predownloaded store, forms its local
contributions, participates in one group reduction, and the group root
appends an IC slab of **sufficient statistics** to ``stats.zarr``.

Sufficient statistics, not scores
---------------------------------
Nothing final is written during inference.  ``stats.zarr`` holds mergeable
per-``(IC, lead, variable)`` sums — squared errors, ensemble moments,
weighted rank counts, anomaly cross-moments — from which
:func:`finalize_stats` derives a ``scores.zarr`` that is schema-compatible
with the offline scorer's output (``{metric}__{variable}`` arrays).  Two
properties follow:

* Non-linear aggregations stay correct.  ``RMSE = sqrt(mean_t(MSE))`` and
  ACC are not linear in ICs, so the *sums* rather than the per-IC scores
  are the durable artifact.
* Metrics derivable from the stored sums, and bootstrap CIs over ICs,
  remain possible after the fact without re-running inference. Spatial
  weighting (e.g. cosine-latitude weighting) is baked into the sums at
  run time, so choose the set of spatial regions up front:
  ``scoring.regions`` adds a ``region`` axis to every sum (boxes on the
  scored grid, masked into the weights), and evaluating a region outside
  that set requires re-running inference (or offline scoring against a
  retained forecast store).
* Scope limited to a fixed set of metrics amenable to the above patterns.
  Users with custom metrics must still run offline.

Numerics
--------
All spatial reductions accumulate in ``float64``.  Group reductions are
formed on the **residual** ``d_i = f_i - y`` rather than on ``f_i``: the
ensemble variance is shift-invariant, so this is algebraically identical
while keeping the summands at forecast-error magnitude instead of field
magnitude, which removes essentially all of the catastrophic cancellation
in ``sum f^2``.

Scope
-----
Per-member MSE, ensemble-mean MSE, ensemble variance / spread, rank
histogram, bias / correlation / ACC, fair CRPS (via the member exchange),
optional per-member MAE and log spectral distance, optional regional splits,
and ``members_per_rank > 1`` on top of a member-batched rollout.
"""

from __future__ import annotations

import os
import shutil
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Protocol

import hydra
import numpy as np
import torch
import torch.distributed as dist
import xarray as xr
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from earth2studio.data import DataSource
from earth2studio.utils.coords import CoordSystem

from .distributed import run_on_rank0_first
from .output import OutputManager
from .regions import (
    NON_SPATIAL,
    build_spatial_weights,
    parse_regions,
)
from .scoring import _apply_valid_ranges
from .work import (
    EnsembleGroup,
    WorkItem,
    ensemble_group_for_rank,
    plan_ensemble_groups,
    write_online_marker,
)

# Dimensions that never take part in the spatial reduction.
_NON_SPATIAL = NON_SPATIAL  # shared spatial-dim filter (src.regions)

# Group products materialized once per lead step, in this fixed order, for
# the union of every configured statistic's `requires()`.  Ordering matters:
# each entry issues a collective, and every rank in the group must issue
# them in the same sequence.
_PRODUCT_ORDER = ("ens_moments", "rank_counts", "pairwise")

# Statistic layouts.  Each maps to one coordinate group in `stats.zarr`.
_LAYOUT_SCALAR = "scalar"  # (time, lead_time)
_LAYOUT_MEMBER = "member"  # (time, ensemble, lead_time)
_LAYOUT_RANK = "rank_bin"  # (time, rank_bin, lead_time)

_LAYOUT_DIMS: dict[str, tuple[str, ...]] = {
    _LAYOUT_SCALAR: ("time", "lead_time"),
    _LAYOUT_MEMBER: ("time", "ensemble", "lead_time"),
    _LAYOUT_RANK: ("time", "rank_bin", "lead_time"),
}


def _layout_dims(layout: str, n_regions: int) -> tuple[str, ...]:
    """Store dims for *layout* — a ``region`` axis follows ``time`` when
    the run configures regional splits, so per-IC slabs stay one chunk."""
    dims = _LAYOUT_DIMS[layout]
    if n_regions:
        return (dims[0], "region", *dims[1:])
    return dims


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OnlineSettings:
    """Parsed ``scoring.online`` configuration block.

    Parameters
    ----------
    ensemble_group_size : int | None
        Ranks per ensemble group ``G``.  ``None`` derives
        ``G = ensemble_size / members_per_rank``.
    members_per_rank : int
        Ensemble members per rank ``K``.  ``> 1`` requires a pipeline with
        a member-batched rollout (``run_item_batched``).
    stats_store : str
        Filename of the statistics store under ``output.path``.
    scores_store : str | None
        Where :func:`finalize_stats` writes.  ``None`` falls back to
        ``scoring.output.store_name``.  Set it explicitly to park the
        online scores beside an offline store built from the same run's
        retained fields, which is how an online campaign gets validated.
    moment_comm_dtype : torch.dtype
        Dtype of the field-sized ensemble-moment reduction.  ``float64``
        (the default) keeps the ensemble variance exact; ``float32`` halves
        the wire volume at the cost of precision in the variance's
        difference of sums.
    climatology : Any
        Optional Hydra config for a ``DataSource`` supplying climatology on
        the verification grid.  Required for ACC; ``None`` disables it.
    verification_cache_size : int
        Number of verification valid times held per rank.
    validate_coords : bool
        Whether to cross-check each step's lead time across the group.  A
        pipeline whose ``run_item`` yields different coords for different
        members would otherwise deadlock rather than fail.
    nccl_timeout_s : int
        Watchdog timeout on the group process group.  A dead rank hangs its
        whole group, so this converts a hang into a clean failure that
        ``resume=true`` can pick up from.
    crps : bool
        Whether to accumulate fair CRPS.  This is the only statistic that
        needs field-sized member-to-member communication — everything else
        rides on spatially-reduced scalars — so it is the one knob that
        materially changes an online run's cost.
    mae : bool
        Accumulate per-member weighted absolute error (finalizes to
        ``mae``, matching ``earth2studio.statistics.mae``).  Off by
        default.
    lsd : bool
        Compute ``earth2studio.statistics.log_spectral_distance`` per
        member and lead (finalizes to ``lsd``, in dB).  Configured as a
        bool, or as ``{wavenumber_cutoff: N}`` to also set the cutoff.
        Spectra are global by construction, so with ``regions`` configured
        the values land only in whole-grid (``null``-box) regions; at
        least one must exist.
    lsd_wavenumber_cutoff : int | None
        Optional radial-mode cutoff forwarded to the earth2studio metric.
    pairwise_comm_dtype : torch.dtype
        Wire dtype for the member exchange.  ``bfloat16`` halves the volume;
        the arithmetic upcasts to float64 immediately on arrival.  Safe
        because the exchange ships residuals rather than raw fields (see
        :class:`PairwiseExchange`) — on raw fields bf16's *relative*
        precision would land at ~1e2 for ``z500``, swamping the ensemble
        spread the pairwise term measures.
    defer_pairwise_one_step : bool
        Issue the exchange for lead step *n* asynchronously and finalize it
        during step *n+1*'s model forward.  The abs-diff compute is trivial
        on GPU, so overlapping the transfer against the *model* — not
        against its own compute — is what hides it.
    variable_chunk : int
        Variables per exchange + pairwise-compute step.  Bounds the
        collective payload and the float64 abs-diff working set.
    pairwise_variables : list[str] | None
        Restrict CRPS to a subset of the scored variables.  ``None`` scores
        all of them.  The moment-based statistics stay on the full set
        either way, so this trades CRPS coverage against the only expensive
        communication in the run.
    regions : dict[str, list[dict] | None] | None
        Named spatial splits.  ``None`` (the default) keeps the store
        region-free.  Each entry is ``None`` (the whole grid), one
        ``{lat: [min, max], lon: [min, max]}`` box on the scored grid, or
        a list of boxes whose union defines the region; longitude boxes
        may wrap (``min > max`` after normalizing to [0, 360)).
        Every weighted sum gains a ``region`` axis.
    pairwise_member_tile : int
        Members per float64 abs-diff tile in :func:`_abs_diff_weighted_sum`
        — the scorer's own dominant memory term, and the one with no other
        knob.  It binds when a model is large enough to force
        ``members_per_rank=1, ensemble_group_size=ensemble_size`` (one
        sample per GPU): the exchange's receive buffer arrives in a narrow
        wire dtype, but the abs-diff arithmetic upcasts to float64
        immediately, so the *temporary* — sized
        ``pairwise_member_tile * variable_chunk * field * 8 bytes`` — is
        the actual constraint.  Defaults to 8, preserving prior behavior;
        lowering it (e.g. to 2) alongside ``variable_chunk=1`` trades
        throughput for a much smaller working set.  ``moment_comm_dtype``
        is the other memory lever for this regime — it halves the
        field-sized ensemble-moment reduction, a separate ~120 MB term.
    """

    ensemble_group_size: int | None
    members_per_rank: int
    stats_store: str
    scores_store: str | None
    moment_comm_dtype: torch.dtype
    climatology: Any
    verification_cache_size: int
    validate_coords: bool
    nccl_timeout_s: int
    crps: bool
    mae: bool
    lsd: bool
    lsd_wavenumber_cutoff: int | None
    pairwise_comm_dtype: torch.dtype
    defer_pairwise_one_step: bool
    variable_chunk: int
    pairwise_variables: list[str] | None
    pairwise_member_tile: int
    regions: dict[str, list[dict] | None] | None


def online_enabled(cfg: DictConfig) -> bool:
    """Whether ``cfg`` asks for online scoring during inference.

    Raises on an unrecognized mode rather than falling back to offline —
    a typo that silently downgrades a campaign to the offline path would
    surface only as a surprise ``forecast.zarr`` the size of a filesystem.
    """
    mode = str(cfg.get("scoring", {}).get("mode", "offline")).lower()
    if mode == "both":
        raise ValueError(
            "scoring.mode='both' has been removed.  An online run keeps its "
            "raw fields whenever output.retain=all (the default), so score "
            "both ways with:\n"
            "    main.py  ... scoring.mode=online\n"
            "    score.py ... scoring.mode=offline "
            "scoring.output.store_name=scores_offline.zarr\n"
            "and set scoring.online.scores_store to name the online store."
        )
    if mode not in ("offline", "online"):
        raise ValueError(
            f"Invalid scoring.mode '{mode}'; expected 'offline' or 'online'."
        )
    return mode == "online"


def retain_raw_output(cfg: DictConfig) -> bool:
    """Whether ``main.py`` should still write ``forecast.zarr``.

    ``output.retain`` defaults to ``all`` (today's behavior).  ``none``
    drops the raw store entirely and is only legal alongside online
    scoring.  ``sample`` (thinned retention for report visualizations) is
    a later phase.
    """
    retain = str(cfg.get("output", {}).get("retain", "all")).lower()
    if retain == "sample":
        raise NotImplementedError(
            "output.retain='sample' (thinned raw retention) is not implemented "
            "yet. Use 'all' or 'none'."
        )
    if retain not in ("all", "none"):
        raise ValueError(f"Invalid output.retain '{retain}'; expected 'all' or 'none'.")
    if retain == "none" and not online_enabled(cfg):
        raise ValueError(
            "output.retain='none' discards the raw forecast store, but "
            "scoring.mode is not 'online' — the run would produce nothing.  "
            "Set scoring.mode=online."
        )
    return retain == "all"


def parse_online_settings(cfg: DictConfig) -> OnlineSettings:
    """Read and validate the ``scoring.online`` config block.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.

    Returns
    -------
    OnlineSettings

    Raises
    ------
    ValueError
        On an invalid dtype or count, or on a setting that has since been
        removed (which is rejected rather than ignored, so a stale config
        cannot silently change what a run does).
    """
    block = cfg.get("scoring", {}).get("online", {}) or {}

    # `x or default` would fold 0 into the default and silently accept it.
    members_per_rank = _int_or(block.get("members_per_rank", 1), 1)
    if members_per_rank < 1:
        raise ValueError(
            f"scoring.online.members_per_rank must be >= 1, got {members_per_rank}."
        )

    moment_dtypes = {"float64": torch.float64, "float32": torch.float32}
    moment_dtype = _parse_dtype(
        block.get("moment_comm_dtype", "float64"),
        moment_dtypes,
        "moment_comm_dtype",
    )
    pairwise_dtypes = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    pairwise_dtype = _parse_dtype(
        block.get("pairwise_comm_dtype", "bfloat16"),
        pairwise_dtypes,
        "pairwise_comm_dtype",
    )

    if "pairwise_exchange" in block:
        raise ValueError(
            "scoring.online.pairwise_exchange has been removed; the member "
            "exchange is always an all-gather.  The ring alternative held "
            "K members of buffer instead of M, but variable_chunk bounds "
            "the same footprint without giving up one-step deferral."
        )
    defer = bool(block.get("defer_pairwise_one_step", True))

    # 0 means "no chunking" here, so a null and an explicit 0 agree.
    variable_chunk = _int_or(block.get("variable_chunk", 4), 0)
    if variable_chunk < 0:
        raise ValueError(
            f"scoring.online.variable_chunk must be >= 0, got {variable_chunk}."
        )

    pairwise_variables = block.get("pairwise_variables", None)
    if pairwise_variables is not None:
        pairwise_variables = [str(v) for v in pairwise_variables]

    pairwise_member_tile = _int_or(
        block.get("pairwise_member_tile", _PAIRWISE_MEMBER_TILE), _PAIRWISE_MEMBER_TILE
    )
    if pairwise_member_tile < 1:
        raise ValueError(
            "scoring.online.pairwise_member_tile must be >= 1, got "
            f"{pairwise_member_tile}."
        )

    # Regions live at scoring.regions — shared with the offline pathway.
    regions = parse_regions(cfg.scoring.get("regions", None))
    lsd_cfg = block.get("lsd", False)
    lsd_cutoff = None
    if isinstance(lsd_cfg, (dict, DictConfig)):
        lsd_opts = (
            OmegaConf.to_container(lsd_cfg, resolve=True)
            if isinstance(lsd_cfg, DictConfig)
            else dict(lsd_cfg)
        )
        unknown = set(lsd_opts) - {"wavenumber_cutoff"}
        if unknown:
            raise ValueError(
                f"Unknown scoring.online.lsd option(s) {sorted(unknown)}; "
                "expected only 'wavenumber_cutoff'."
            )
        lsd = True
        if lsd_opts.get("wavenumber_cutoff") is not None:
            lsd_cutoff = int(lsd_opts["wavenumber_cutoff"])
    else:
        lsd = bool(lsd_cfg)
    if lsd and regions is not None and not any(v is None for v in regions.values()):
        raise ValueError(
            "scoring.online.lsd needs a whole-grid region (an entry with a "
            "null box) to land its values in — spectra cannot be computed "
            "on a spatial sub-box.  Add e.g. 'global: null' to "
            "scoring.regions."
        )

    group_size = block.get("ensemble_group_size", None)
    return OnlineSettings(
        ensemble_group_size=None if group_size is None else int(group_size),
        members_per_rank=members_per_rank,
        stats_store=str(block.get("stats_store", "stats.zarr")),
        scores_store=(
            None
            if block.get("scores_store", None) is None
            else str(block["scores_store"])
        ),
        moment_comm_dtype=moment_dtype,
        climatology=block.get("climatology", None),
        verification_cache_size=int(block.get("verification_cache_size", 4)),
        validate_coords=bool(block.get("validate_coords", True)),
        nccl_timeout_s=int(block.get("nccl_timeout_s", 1800)),
        crps=bool(block.get("crps", True)),
        pairwise_comm_dtype=pairwise_dtype,
        defer_pairwise_one_step=defer,
        variable_chunk=variable_chunk,
        pairwise_variables=pairwise_variables,
        pairwise_member_tile=pairwise_member_tile,
        regions=regions,
        mae=bool(block.get("mae", False)),
        lsd=lsd,
        lsd_wavenumber_cutoff=lsd_cutoff,
    )


def _int_or(value: Any, default: int) -> int:
    """``int(value)``, mapping only ``None`` to *default*.

    Guards against ``int(value or default)``, which would fold a
    deliberate ``0`` into the default and accept an out-of-range setting
    as if it had never been written.
    """
    return default if value is None else int(value)


def _parse_dtype(value: Any, allowed: dict[str, torch.dtype], key: str) -> torch.dtype:
    """Resolve a config dtype name against the options valid for *key*."""
    name = str(value).lower()
    if name not in allowed:
        raise ValueError(
            f"Invalid scoring.online.{key} '{name}'; "
            f"expected one of {sorted(allowed)}."
        )
    return allowed[name]


# ---------------------------------------------------------------------------
# Group communication
# ---------------------------------------------------------------------------


class GroupComm:
    """Collectives over one ensemble group.

    Wraps the group's ``torch.distributed`` process group and degrades to a
    no-op when the group holds a single rank (``G = 1``), which is the
    deterministic / single-GPU path.  Every method is collective: all ranks
    of the group must call them in the same order.

    Parameters
    ----------
    group : EnsembleGroup
        This rank's group membership.
    process_group : Any
        The ``torch.distributed`` process group covering ``group.ranks``,
        or ``None`` when no communication is needed.
    """

    def __init__(self, group: EnsembleGroup, process_group: Any) -> None:
        self._group = group
        self._pg = process_group

    @property
    def group(self) -> EnsembleGroup:
        """The :class:`~src.work.EnsembleGroup` this comm serves."""
        return self._group

    @property
    def is_root(self) -> bool:
        """Whether this rank writes the group's statistics slabs."""
        return self._group.is_root

    @property
    def active(self) -> bool:
        """Whether collectives actually cross the wire."""
        return self._pg is not None

    def reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sum-reduce *tensor* onto the group root, in place.

        Non-root ranks are left holding an unspecified value and must not
        read the result.

        Parameters
        ----------
        tensor : torch.Tensor
            Contribution from this rank.

        Returns
        -------
        torch.Tensor
            The same tensor object, for call chaining.
        """
        if self._pg is not None:
            dist.reduce(
                tensor,
                dst=self._group.root_rank,
                op=dist.ReduceOp.SUM,
                group=self._pg,
            )
        return tensor

    def all_gather(
        self, tensor: torch.Tensor, async_op: bool = False
    ) -> tuple[list[torch.Tensor], Any]:
        """Gather every rank's *tensor* onto every rank of the group.

        Parameters
        ----------
        tensor : torch.Tensor
            This rank's contribution.  Must be identically shaped on every
            rank of the group.
        async_op : bool
            Return without waiting.  The caller must ``wait()`` the returned
            handle before reading the buffers — this is what lets the
            pairwise exchange overlap a model forward.

        Returns
        -------
        tuple[list[torch.Tensor], Any]
            ``(per_rank_buffers, work_handle)``.  The handle is ``None``
            when nothing was communicated or when ``async_op`` is false.
        """
        if self._pg is None:
            return [tensor], None
        buffers = [torch.empty_like(tensor) for _ in range(self._group.group_size)]
        work = dist.all_gather(buffers, tensor, group=self._pg, async_op=async_op)
        return buffers, (work if async_op else None)

    def all_agree(self, value: int, device: torch.device) -> bool:
        """Whether every rank in the group passed the same *value*.

        Used to catch a pipeline whose ``run_item`` yields different coords
        for different ensemble members — a mismatch that would otherwise
        deadlock the group at the next reduction rather than fail.

        Parameters
        ----------
        value : int
            Rank-local value to compare (e.g. a lead time in nanoseconds).
        device : torch.device
            Device to stage the collective on.

        Returns
        -------
        bool
            ``True`` when all ranks agree (always ``True`` for ``G = 1``).
        """
        if self._pg is None:
            return True
        t = torch.tensor([value, -value], dtype=torch.int64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self._pg)
        return bool(t[0].item() == value and -t[1].item() == value)


def build_group_comm(
    settings: OnlineSettings,
    rank: int,
    world_size: int,
    ensemble_size: int,
) -> GroupComm | None:
    """Create this rank's :class:`GroupComm`, or ``None`` if it idles.

    ``torch.distributed.new_group`` is itself collective over the whole
    world, so *every* rank — including ranks that end up in no group —
    must call this function, and the groups are created in a fixed order.

    Parameters
    ----------
    settings : OnlineSettings
        Parsed online settings.
    rank : int
        This process's global rank.
    world_size : int
        Total ranks in the job.
    ensemble_size : int
        Ensemble size ``M``.

    Returns
    -------
    GroupComm | None
        ``None`` for leftover ranks with no group to join.
    """
    rank_groups = plan_ensemble_groups(
        world_size,
        ensemble_size,
        group_size=settings.ensemble_group_size,
        members_per_rank=settings.members_per_rank,
    )
    group = ensemble_group_for_rank(rank, rank_groups, settings.members_per_rank)

    distributed = dist.is_available() and dist.is_initialized() and world_size > 1
    process_group = None

    if distributed:
        timeout = timedelta(seconds=settings.nccl_timeout_s)
        for ranks in rank_groups:
            # Collective over the world: every rank enters every call.
            pg = dist.new_group(ranks=list(ranks), timeout=timeout)
            if group is not None and tuple(ranks) == group.ranks:
                process_group = pg

    if group is None:
        logger.warning(f"Rank {rank}: no ensemble group assigned; will idle.")
        return None

    # A single-rank group needs no communication at all.
    if group.group_size == 1:
        process_group = None

    logger.info(
        f"Rank {rank}: ensemble group {group.group_id}/{group.n_groups} "
        f"(G={group.group_size}, K={group.members_per_rank}, "
        f"members={list(group.member_ids)}, root={group.is_root})"
    )
    return GroupComm(group, process_group)


# ---------------------------------------------------------------------------
# Verification / climatology access
# ---------------------------------------------------------------------------


class FieldCache:
    """Valid-time keyed LRU cache over a gridded ``DataSource``.

    Serves verification (and, when configured, climatology) fields to the
    scoring loop.  Every rank in a group requests the same valid times, so
    with a node-local group the OS page cache absorbs most of the read
    amplification; the in-process cache exists to avoid re-decoding the
    same chunk across the members a rank carries and across neighbouring
    lead steps.

    Fields are normalized once on the way in — variable order, spatial dim
    order, NaN policy and valid-range clamping — so the scoring loop can
    treat a cached field as directly comparable to a model output.

    Parameters
    ----------
    source : DataSource
        Source to read from.  Required to be local (predownloaded); online
        scoring does not tolerate in-loop remote fetches.
    variables : list[str]
        Variables to fetch, in the order the scorer expects them.
    spatial_dims : tuple[str, ...]
        Spatial dimension names in the order the scorer expects them.
    device : torch.device
        Device to stage fields on.
    max_size : int
        Number of valid times to retain.
    nan_policy : str
        ``"propagate"`` or ``"zero_fill"`` — mirrors ``scoring.nan_policy``
        so online and offline scores stay comparable.
    valid_ranges : dict | None
        Per-variable ``{min, max}`` clamps, mirroring
        ``scoring.valid_ranges``.
    """

    def __init__(
        self,
        source: DataSource,
        variables: list[str],
        spatial_dims: tuple[str, ...],
        device: torch.device,
        max_size: int = 4,
        nan_policy: str = "propagate",
        valid_ranges: dict | None = None,
    ) -> None:
        self._source = source
        self._variables = list(variables)
        self._spatial_dims = tuple(spatial_dims)
        self._device = device
        self._max_size = max(1, max_size)
        self._nan_policy = nan_policy
        self._valid_ranges = valid_ranges or {}
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()
        self._coords: CoordSystem = OrderedDict({"variable": np.array(self._variables)})

    def get(self, valid_time: np.datetime64) -> torch.Tensor:
        """Return the field at *valid_time* as a ``[variable, <spatial...>]`` tensor.

        Parameters
        ----------
        valid_time : np.datetime64
            Valid time to read.

        Returns
        -------
        torch.Tensor
            Float32 tensor on the configured device.
        """
        key = int(np.datetime64(valid_time, "ns").astype("int64"))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        field = self._fetch(valid_time)
        self._cache[key] = field
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return field

    def _fetch(self, valid_time: np.datetime64) -> torch.Tensor:
        da = self._source([valid_time], list(self._variables))
        da = da.transpose("time", "variable", *self._spatial_dims)
        tensor = torch.from_numpy(np.asarray(da.values[0]).copy()).to(
            device=self._device, dtype=torch.float32
        )

        if self._nan_policy == "zero_fill":
            tensor = torch.nan_to_num(tensor, nan=0.0)
        if self._valid_ranges:
            tensor = _apply_valid_ranges(tensor, self._coords, self._valid_ranges)
        return tensor


def available_times(source: DataSource) -> np.ndarray | None:
    """Return the valid times a local zarr-backed source can serve.

    Inspects the ``xr.DataArray`` that :class:`~src.data.PredownloadedSource`
    and :class:`~src.data.CompositeSource` wrap.  Returns ``None`` for
    sources whose coverage cannot be determined cheaply (e.g. a live remote
    source or a BYO wrapper) — callers should treat that as "unknown", not
    as "empty".

    Parameters
    ----------
    source : DataSource
        Verification source to inspect.

    Returns
    -------
    np.ndarray | None
        Sorted ``datetime64[ns]`` array, or ``None`` when unknown.
    """
    da = getattr(source, "_da", None)
    if da is not None and "time" in getattr(da, "coords", {}):
        return np.asarray(da.coords["time"].values).astype("datetime64[ns]")

    # CompositeSource: the intersection of its components' coverage is what
    # a multi-variable request can actually be served from.
    components = getattr(source, "_sources", None)
    if isinstance(components, dict) and components:
        per_source = [available_times(s) for s in components.values()]
        if any(t is None for t in per_source):
            return None
        common = per_source[0]
        for t in per_source[1:]:
            common = np.intersect1d(common, t)
        return common
    return None


def check_verification_coverage(
    source: DataSource,
    ic_times: Sequence[np.datetime64],
    lead_times: np.ndarray,
    variables: Sequence[str],
) -> None:
    """Fail fast when verification does not cover every scored valid time.

    Because verification is required to be predownloaded, coverage is
    knowable before any model weights load.  This converts what would
    otherwise be an in-loop failure — hanging an entire ensemble group
    hours into a campaign — into a startup error.

    Parameters
    ----------
    source : DataSource
        Verification source.
    ic_times : Sequence[np.datetime64]
        IC times this rank's group will run.
    lead_times : np.ndarray
        All lead times in the forecast.
    variables : Sequence[str]
        Variables that will be scored.

    Raises
    ------
    ValueError
        If any required valid time is missing from the store.
    """
    have = available_times(source)
    if have is None:
        logger.warning(
            "Verification coverage could not be verified up front (the source "
            "does not expose a time index).  A missing valid time will surface "
            "as an in-loop failure instead."
        )
        return

    needed = np.unique(
        np.asarray(
            [
                np.datetime64(t, "ns") + np.timedelta64(lt, "ns")
                for t in ic_times
                for lt in lead_times.astype("timedelta64[ns]")
            ],
            dtype="datetime64[ns]",
        )
    )
    missing = np.setdiff1d(needed, have)
    if missing.size:
        preview = ", ".join(str(t) for t in missing[:5])
        raise ValueError(
            f"Verification store is missing {missing.size}/{needed.size} valid "
            f"times required to score variables {list(variables)} "
            f"(first missing: {preview}).\n"
            "Re-run predownload.py with predownload.verification.enabled=true "
            "for the full forecast window, or narrow the campaign's ICs / "
            "nsteps."
        )
    logger.info(
        f"Verification coverage check passed: {needed.size} valid times "
        f"x {len(variables)} variables."
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


@dataclass
class StepContext:
    """Everything one lead step exposes to the configured statistics.

    Group products (``s1``, ``s2``, ``below``) are materialized *eagerly*
    by :class:`OnlineScorer` for the union of every statistic's
    :meth:`OnlineStatistic.requires`, in a fixed order, before any
    ``update`` runs.  They are only valid on the group root; non-root ranks
    see whatever the reduction left in their buffers and must not read them.

    Parameters
    ----------
    lead_index : int
        Index of this lead step in the store's ``lead_time`` axis.
    valid_time : np.datetime64
        ``IC time + lead time``.
    y : torch.Tensor
        Verification field, ``[variable, <spatial...>]``.
    clim : torch.Tensor | None
        Climatology on the same grid, or ``None``.
    f_local : torch.Tensor
        This rank's member block, ``[member, variable, <spatial...>]``.
    d_local : torch.Tensor
        ``f_local - y``, precomputed once (every statistic wants it).
    weights : torch.Tensor
        Spatial weights.  Region-free runs: broadcastable over the spatial
        dims.  With regions configured: ``[region, <spatial...>]``, the
        latitude weights multiplied by each region's mask.
    w_flat : torch.Tensor | None
        ``None`` for region-free runs.  Otherwise the same weights
        flattened to ``[n_spatial_points, region]`` so :meth:`wsum` is one
        matmul; its output then carries a trailing ``region`` axis.
    valid : torch.Tensor
        Boolean ``[variable, <spatial...>]`` mask, ``True`` where every
        member's *original* (pre-``nan_policy``) forecast was finite.  A
        pipeline like ``DLESyMPipeline`` NaN-masks whole (lead, variable)
        slices outside their validity cadence; this mask is what keeps
        that from either propagating as NaN through every derived
        statistic or, under ``nan_policy=zero_fill``, being scored as a
        spurious zero-valued forecast.  :meth:`wsum` excludes masked
        gridpoints from every weighted sum it forms.
    n_spatial : int
        Number of trailing spatial dimensions.
    ensemble_size : int
        Total ensemble size ``M``.
    member_ids : tuple[int, ...]
        Global member indices this rank carries.
    comm : GroupComm
        Group collectives.
    s1 : torch.Tensor | None
        Root-only ``sum_i (f_i - y)`` over the whole ensemble.
    s2 : torch.Tensor | None
        Root-only ``sum_i (f_i - y)^2``.
    below : torch.Tensor | None
        Root-only ``sum_i 1[f_i < y]``, the rank of ``y`` per gridpoint.
    """

    lead_index: int
    valid_time: np.datetime64
    y: torch.Tensor
    clim: torch.Tensor | None
    f_local: torch.Tensor
    d_local: torch.Tensor
    weights: torch.Tensor
    valid: torch.Tensor
    n_spatial: int
    ensemble_size: int
    member_ids: tuple[int, ...]
    comm: GroupComm
    w_flat: torch.Tensor | None = None
    s1: torch.Tensor | None = None
    s2: torch.Tensor | None = None
    below: torch.Tensor | None = None

    @property
    def n_regions(self) -> int:
        """Number of configured regions; 0 keeps the store region-free."""
        return 0 if self.w_flat is None else self.w_flat.shape[-1]

    def wsum(self, t: torch.Tensor) -> torch.Tensor:
        """Weighted sum of *t* over its trailing spatial dimensions.

        Gridpoints :attr:`valid` marks invalid for their variable are
        excluded — treated as absent, not zero — regardless of what *t*
        actually holds there (NaN under ``nan_policy=propagate``, or a
        finite but meaningless value under ``zero_fill``).

        With regions configured the result carries a trailing ``region``
        axis (one masked reduction per region, via a single matmul).
        """
        td = torch.where(self.valid, t.double(), 0.0)
        if self.w_flat is not None:
            lead = td.shape[: td.ndim - self.n_spatial]
            return td.reshape(*lead, -1) @ self.w_flat
        axes = tuple(range(t.ndim - self.n_spatial, t.ndim))
        return (td * self.weights).sum(dim=axes)

    def wsum_valid(self) -> torch.Tensor:
        """Per-variable weighted sum of :attr:`valid` — the mask-aware
        weighted-mean normalizer for whatever :meth:`wsum` excluded.
        Equals the old constant ``sum(weights)`` wherever nothing is
        masked."""
        vd = self.valid.double()
        if self.w_flat is not None:
            lead = vd.shape[: vd.ndim - self.n_spatial]
            return vd.reshape(*lead, -1) @ self.w_flat
        axes = tuple(range(self.valid.ndim - self.n_spatial, self.valid.ndim))
        return (vd * self.weights).sum(dim=axes)


class OnlineStatistic(Protocol):
    """Accumulator emitting sufficient statistics for one metric family.

    Implementations declare which group-reduced products they need, so the
    scorer only forms the expensive ones when something actually asks.
    ``update`` may issue its own small group collectives — the statistic
    list is built identically on every rank, so their order is well
    defined.
    """

    name: str

    def requires(self) -> set[str]:
        """Group products needed, from :data:`_PRODUCT_ORDER`."""
        ...

    def fields(self) -> dict[str, str]:
        """Map each emitted field name to its layout (see ``_LAYOUT_*``)."""
        ...

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        """Create per-IC buffers for a fresh initial condition.

        ``n_regions > 0`` prepends a region axis to every buffer,
        matching the trailing region axis :meth:`StepContext.wsum`
        produces."""
        ...

    def update(self, ctx: StepContext) -> None:
        """Accumulate this lead step's contribution."""
        ...

    def state(self) -> dict[str, torch.Tensor]:
        """Return the accumulated buffers, keyed by field name."""
        ...

    # Optional hooks, consulted via ``getattr`` so statistics that need
    # neither stay boilerplate-free:
    #
    #   flush(ctx)     -> complete deferred work at the end of an IC.
    #                     Collective; called on every rank in list order.
    #   variables()    -> narrow which variables this statistic emits
    #                     fields for (see :func:`statistic_variables`).


def _empty(*shape: int | torch.device) -> torch.Tensor:
    """NaN-filled float64 accumulator; the last argument is the device.

    Buffers start as NaN so that a lead step the pipeline never yields is
    visibly absent in ``stats.zarr`` rather than reading as a zero score.
    """
    *dims, device = shape
    return torch.full(
        tuple(int(d) for d in dims),  # type: ignore[arg-type]
        float("nan"),
        dtype=torch.float64,
        device=device,  # type: ignore[arg-type]
    )


def _region_first(value: torch.Tensor, n_regions: int) -> torch.Tensor:
    """Move :meth:`StepContext.wsum`'s trailing region axis to the front,
    matching the buffer layout; identity for region-free runs."""
    return value.movedim(-1, 0) if n_regions else value


class WeightSum:
    """The weighted-mean normalizer ``W = sum_s w``.

    Stored per ``(IC, lead, variable)`` — rather than assumed constant —
    precisely so a masked / regional accumulation can vary it without
    changing the store schema or the finalizers: a fully NaN-masked
    (lead, variable) gets ``W = 0`` here, matching the zeroed numerator
    every other statistic's :meth:`StepContext.wsum` produces for it, so
    ``finalize_stats`` divides ``0 / 0 = NaN`` rather than a wrong,
    finite value against the unmasked grid weight.
    """

    name = "weight_sum"

    def requires(self) -> set[str]:
        return set()

    def fields(self) -> dict[str, str]:
        return {"w_sum": _LAYOUT_SCALAR}

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        self._nr = n_regions
        self._w = _empty(
            *((n_regions,) if n_regions else ()), n_leads, n_variables, device
        )

    def update(self, ctx: StepContext) -> None:
        if not ctx.comm.is_root:
            return
        self._w[..., ctx.lead_index, :] = _region_first(ctx.wsum_valid(), self._nr)

    def state(self) -> dict[str, torch.Tensor]:
        return {"w_sum": self._w}


class MemberSquaredError:
    """Per-member weighted sum of squared error.

    Entirely local — each rank owns whole members — so only a tiny
    ``[M, variable]`` block crosses the wire, zero-padded outside this
    rank's members.  Finalizes to ``mse__{var}``, which matches the offline
    ``src.metrics.mse`` (and thus ``rmse`` after ``sqrt(mean_t(.))``).
    """

    name = "member_mse"

    def requires(self) -> set[str]:
        return set()

    def fields(self) -> dict[str, str]:
        return {"sse_member": _LAYOUT_MEMBER}

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        self._nr = n_regions
        self._sse = _empty(
            *((n_regions,) if n_regions else ()),
            ensemble_size,
            n_leads,
            n_variables,
            device,
        )

    def update(self, ctx: StepContext) -> None:
        block = torch.zeros(
            (ctx.ensemble_size, self._sse.shape[-1])
            + ((self._nr,) if self._nr else ()),
            dtype=torch.float64,
            device=self._sse.device,
        )
        block[list(ctx.member_ids)] = ctx.wsum(
            ctx.d_local**2
        )  # [K, variable(, region)]
        ctx.comm.reduce(block)

        if ctx.comm.is_root:
            # [M, var(, R)] -> buffer slice [(R,) M, var]
            self._sse[..., ctx.lead_index, :] = _region_first(block, self._nr)

    def state(self) -> dict[str, torch.Tensor]:
        return {"sse_member": self._sse}


class MemberAbsError:
    """Per-member weighted sum of absolute error (finalizes to ``mae``).

    Same communication shape as :class:`MemberSquaredError`: entirely
    local per member, one small zero-padded ``[M, variable]`` reduce.
    Enabled by ``scoring.online.mae``.
    """

    name = "member_mae"

    def requires(self) -> set[str]:
        return set()

    def fields(self) -> dict[str, str]:
        return {"sae_member": _LAYOUT_MEMBER}

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        self._nr = n_regions
        self._sae = _empty(
            *((n_regions,) if n_regions else ()),
            ensemble_size,
            n_leads,
            n_variables,
            device,
        )

    def update(self, ctx: StepContext) -> None:
        block = torch.zeros(
            (ctx.ensemble_size, self._sae.shape[-1])
            + ((self._nr,) if self._nr else ()),
            dtype=torch.float64,
            device=self._sae.device,
        )
        block[list(ctx.member_ids)] = ctx.wsum(ctx.d_local.abs())
        ctx.comm.reduce(block)
        if ctx.comm.is_root:
            self._sae[..., ctx.lead_index, :] = _region_first(block, self._nr)

    def state(self) -> dict[str, torch.Tensor]:
        return {"sae_member": self._sae}


class LogSpectralDistance:
    """Per-member radially averaged 2D log spectral distance vs verification.

    Delegates the whole computation — radial power spectra, optional
    wavenumber cutoff, and the dB form — to
    ``earth2studio.statistics.log_spectral_distance``; this class
    reimplements nothing.  Unlike every other field in the store this is a
    *final* per-(IC, member, lead) value rather than a mergeable sum — its
    aggregation over ICs is a plain mean, so storing the value keeps it
    exact.  Spectra are global by construction: with regions configured
    the values land only in whole-grid (``null``-box) regions and stay NaN
    elsewhere.  Enabled by ``scoring.online.lsd``.

    Parameters
    ----------
    whole_grid_regions : Sequence[int] | None
        Region indices with a ``null`` box, or ``None`` when the run is
        region-free.
    wavenumber_cutoff : int | None
        Forwarded to the earth2studio metric: keep only the first N radial
        modes (negative trims from the end; ``None`` uses all).
    """

    name = "lsd"

    def __init__(
        self,
        whole_grid_regions: Sequence[int] | None = None,
        wavenumber_cutoff: int | None = None,
    ) -> None:
        from earth2studio.statistics import log_spectral_distance

        self._metric = log_spectral_distance(
            reduction_dimensions=[],
            ensemble_dimension="ensemble",
            wavenumber_cutoff=wavenumber_cutoff,
        )
        self._whole_grid = (
            None if whole_grid_regions is None else list(whole_grid_regions)
        )

    def requires(self) -> set[str]:
        return set()

    def fields(self) -> dict[str, str]:
        return {"lsd": _LAYOUT_MEMBER}

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        self._nr = n_regions
        self._lsd = _empty(
            *((n_regions,) if n_regions else ()),
            ensemble_size,
            n_leads,
            n_variables,
            device,
        )

    def update(self, ctx: StepContext) -> None:
        if ctx.n_spatial != 2:
            raise ValueError(
                "scoring.online.lsd needs a 2D (lat, lon) spatial grid; "
                f"got {ctx.n_spatial} spatial dimensions."
            )
        # Entirely local per member.  The metric only uses the coordinate
        # system to check x/y compatibility, so index coords suffice.
        n_var = ctx.f_local.shape[1]
        y_coords: CoordSystem = OrderedDict(
            {
                "variable": np.arange(n_var),
                "ilat": np.arange(ctx.f_local.shape[-2]),
                "ilon": np.arange(ctx.f_local.shape[-1]),
            }
        )
        x_coords: CoordSystem = OrderedDict(
            {"ensemble": np.array(ctx.member_ids), **y_coords}
        )
        values, _ = self._metric(ctx.f_local, x_coords, ctx.y, y_coords)
        values = values.double()  # [K, variable]

        block = torch.zeros(
            (ctx.ensemble_size, self._lsd.shape[-1]),
            dtype=torch.float64,
            device=self._lsd.device,
        )
        block[list(ctx.member_ids), :] = values
        ctx.comm.reduce(block)
        if not ctx.comm.is_root:
            return
        if self._nr:
            for r in self._whole_grid or []:
                self._lsd[r, :, ctx.lead_index, :] = block
        else:
            self._lsd[:, ctx.lead_index, :] = block

    def state(self) -> dict[str, torch.Tensor]:
        return {"lsd": self._lsd}


class EnsembleMeanSquaredError:
    """Weighted sum of squared error of the ensemble mean.

    ``(fbar - y) = (1/M) * sum_i (f_i - y)`` follows directly from the
    reduced residual sum, so no separate communication is needed beyond the
    shared ``ens_moments`` product.
    """

    name = "ensemble_mean_mse"

    def requires(self) -> set[str]:
        return {"ens_moments"}

    def fields(self) -> dict[str, str]:
        return {"sse_ensmean": _LAYOUT_SCALAR}

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        self._nr = n_regions
        self._sse = _empty(
            *((n_regions,) if n_regions else ()), n_leads, n_variables, device
        )

    def update(self, ctx: StepContext) -> None:
        if not ctx.comm.is_root:
            return
        if ctx.s1 is None:
            raise RuntimeError(
                "StepContext.s1 is unset — 'ens_moments' must be requested "
                "before this accumulator's update() runs."
            )
        mean_dev = ctx.s1 / ctx.ensemble_size
        self._sse[..., ctx.lead_index, :] = _region_first(
            ctx.wsum(mean_dev**2), self._nr
        )

    def state(self) -> dict[str, torch.Tensor]:
        return {"sse_ensmean": self._sse}


class EnsembleSpread:
    """Weighted spatial sum of the unbiased ensemble variance.

    Matches ``src.metrics.ensemble_variance`` (Bessel-corrected variance
    across members, then a weighted spatial mean).  Computed from residual
    moments — variance is shift-invariant, so
    ``var_i(f_i) = var_i(f_i - y)`` exactly, while the summands stay at
    error magnitude.
    """

    name = "ensemble_variance"

    def requires(self) -> set[str]:
        return {"ens_moments"}

    def fields(self) -> dict[str, str]:
        return {"var_ens": _LAYOUT_SCALAR}

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        self._nr = n_regions
        self._var = _empty(
            *((n_regions,) if n_regions else ()), n_leads, n_variables, device
        )

    def update(self, ctx: StepContext) -> None:
        if not ctx.comm.is_root:
            return
        if ctx.s1 is None or ctx.s2 is None:
            raise RuntimeError(
                "StepContext.s1/s2 are unset — 'ens_moments' must be "
                "requested before this accumulator's update() runs."
            )
        m = ctx.ensemble_size
        var = (ctx.s2 - ctx.s1**2 / m) / (m - 1)
        self._var[..., ctx.lead_index, :] = _region_first(ctx.wsum(var), self._nr)

    def state(self) -> dict[str, torch.Tensor]:
        return {"var_ens": self._var}


class RankHistogram:
    """Area-weighted histogram of the verification's rank in the ensemble.

    The rank of ``y`` at a gridpoint is ``#{i : f_i < y}``, giving ``M + 1``
    bins.  Ties are counted into the lower bin rather than randomized;
    for continuous fields they are measure-zero, and for quantized fields a
    deterministic rule keeps runs reproducible.

    Counts are weighted by the same spatial weights as every other
    statistic so that a lat/lon grid's poles do not dominate the histogram.
    """

    name = "rank_histogram"

    def requires(self) -> set[str]:
        return {"rank_counts"}

    def fields(self) -> dict[str, str]:
        return {"rank_counts": _LAYOUT_RANK}

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        self._nr = n_regions
        self._counts = _empty(
            *((n_regions,) if n_regions else ()),
            ensemble_size + 1,
            n_leads,
            n_variables,
            device,
        )

    def update(self, ctx: StepContext) -> None:
        if not ctx.comm.is_root:
            return
        if ctx.below is None:
            raise RuntimeError(
                "StepContext.below is unset — 'rank_counts' must be "
                "requested before this accumulator's update() runs."
            )
        n_bins, _, n_variables = self._counts.shape[-3:]
        device = self._counts.device

        # `below` is meaningless at a masked gridpoint — NaN < y is always
        # False, so a masked member's non-comparison would otherwise land
        # in bin 0 as if it were a genuine low-value forecast.  Rather than
        # try to fix its value, zero its weight so it contributes to no
        # bin at all; `ctx.valid` was captured before nan_policy could
        # have replaced the NaN with something that compares "normally".
        ranks = ctx.below.reshape(n_variables, -1).long()
        valid_flat = ctx.valid.reshape(n_variables, -1)
        # One weight row per region (the region-free run is one region of
        # everything); scatter_add per region keeps the kernel simple.
        if self._nr:
            region_weights = ctx.weights.reshape(self._nr, 1, -1)
        else:
            region_weights = (
                ctx.weights.expand(ctx.y.shape[1:]).reshape(1, 1, -1).contiguous()
            )
        for r in range(region_weights.shape[0]):
            w_flat = torch.where(
                valid_flat, region_weights[r].expand(n_variables, -1), 0.0
            )
            hist = torch.zeros(
                (n_variables, n_bins), dtype=torch.float64, device=device
            )
            hist.scatter_add_(1, ranks, w_flat)
            if self._nr:
                self._counts[r, :, ctx.lead_index, :] = hist.transpose(0, 1)
            else:
                self._counts[:, ctx.lead_index, :] = hist.transpose(0, 1)

    def state(self) -> dict[str, torch.Tensor]:
        return {"rank_counts": self._counts}


class AnomalyMoments:
    """Weighted cross-moments of the ensemble mean against verification.

    Five sums per ``(IC, lead, variable)`` are enough to reconstruct bias,
    Pearson correlation and (with a climatology) ACC.  None of those are
    linear in ICs, which is precisely why the sums — not the per-IC scores
    — are the durable artifact.

    Field names encode whether a climatology was subtracted (``a``/``b``
    for anomalies vs. ``f``/``y`` for raw fields), so
    :func:`finalize_stats` can tell whether ACC is derivable without
    consulting the run's config.
    """

    name = "anomaly_moments"

    def __init__(self, anomaly: bool) -> None:
        self._anomaly = anomaly
        f, y = ("a", "b") if anomaly else ("f", "y")
        self._names = (
            f"sum_w{f}",
            f"sum_w{y}",
            f"sum_w{f}2",
            f"sum_w{y}2",
            f"sum_w{f}{y}",
        )

    def requires(self) -> set[str]:
        return {"ens_moments"}

    def fields(self) -> dict[str, str]:
        return {name: _LAYOUT_SCALAR for name in self._names}

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        self._nr = n_regions
        self._buf = {
            name: _empty(
                *((n_regions,) if n_regions else ()), n_leads, n_variables, device
            )
            for name in self._names
        }

    def update(self, ctx: StepContext) -> None:
        if not ctx.comm.is_root:
            return
        if ctx.s1 is None:
            raise RuntimeError(
                "StepContext.s1 is unset — 'ens_moments' must be requested "
                "before this accumulator's update() runs."
            )
        # fbar = y + mean_i(f_i - y); reconstructing it this way keeps the
        # reduction on residuals without changing the result.
        fbar = ctx.y.double() + ctx.s1 / ctx.ensemble_size
        obs = ctx.y.double()
        if ctx.clim is not None:
            clim = ctx.clim.double()
            fbar = fbar - clim
            obs = obs - clim

        i = ctx.lead_index
        n = self._names
        for name, term in (
            (n[0], fbar),
            (n[1], obs),
            (n[2], fbar**2),
            (n[3], obs**2),
            (n[4], fbar * obs),
        ):
            self._buf[name][..., i, :] = _region_first(ctx.wsum(term), self._nr)

    def state(self) -> dict[str, torch.Tensor]:
        return dict(self._buf)


# ---------------------------------------------------------------------------
# Pairwise member exchange (fair CRPS)
# ---------------------------------------------------------------------------

# Default members per float64 abs-diff tile, preserved by
# scoring.online.pairwise_member_tile when unset.  The exchange arrives in a
# narrow wire dtype but the arithmetic upcasts immediately, so the temporary
# — not the receive buffer — is the memory constraint: this bounds it to
# pairwise_member_tile * variable_chunk * field * 8 bytes.
_PAIRWISE_MEMBER_TILE = 8


def _abs_diff_weighted_sum(
    local: torch.Tensor,
    other: torch.Tensor,
    weights: torch.Tensor,
    n_spatial: int,
    member_tile: int = _PAIRWISE_MEMBER_TILE,
    w_flat: torch.Tensor | None = None,
) -> torch.Tensor:
    """``sum_{i in local} sum_{j in other} sum_s w_s |f_i(s) - f_j(s)|``.

    Tiled over both member axes so the float64 temporary stays bounded
    regardless of ensemble size.  The subtraction upcasts straight from the
    wire dtype, so a bf16 exchange never does bf16 arithmetic.  The weight
    multiply runs in place on the abs-diff tile (``mul_``) rather than
    forming a separate ``diff * weights`` temporary, dropping one of the
    three float64 buffers live at once.

    Parameters
    ----------
    local : torch.Tensor
        This rank's members, ``[K, variable, <spatial...>]``.
    other : torch.Tensor
        Members to pair against, ``[L, variable, <spatial...>]``.
    weights : torch.Tensor
        Spatial weights broadcastable over the spatial dims.
    n_spatial : int
        Number of trailing spatial dimensions.
    member_tile : int
        Members of *other* per tile — bounds the float64 working set.
        Defaults to :data:`_PAIRWISE_MEMBER_TILE`; runs pass
        ``settings.pairwise_member_tile`` explicitly.
    w_flat : torch.Tensor | None
        Regional weights ``[n_spatial_points, region]`` (see
        :attr:`StepContext.w_flat`).  When set the spatial reduction is a
        matmul against it and the result carries a region axis.

    Returns
    -------
    torch.Tensor
        Float64 ``[variable]`` tensor (``[variable, region]`` regional).
    """
    axes = tuple(range(local.ndim - n_spatial, local.ndim))
    out_shape: tuple[int, ...] = (local.shape[1],)
    if w_flat is not None:
        out_shape = (local.shape[1], w_flat.shape[-1])
    out = torch.zeros(out_shape, dtype=torch.float64, device=local.device)
    for i in range(local.shape[0]):
        a = local[i].double()
        for j0 in range(0, other.shape[0], member_tile):
            tile = other[j0 : j0 + member_tile].double()
            diff = (a.unsqueeze(0) - tile).abs_()
            if w_flat is not None:
                flat = diff.reshape(diff.shape[0], diff.shape[1], -1)
                out += (flat @ w_flat).sum(dim=0)
            else:
                diff.mul_(weights)
                out += diff.sum(dim=axes).sum(dim=0)
    return out


@dataclass
class _PendingExchange:
    """An in-flight (or ready) member exchange for one lead step."""

    lead_index: int
    local_chunks: list[torch.Tensor]
    gathered: list[list[torch.Tensor]]
    work: list[Any]


class PairwiseExchange:
    """Forms ``sum_{i<j} sum_s w_s |f_i(s) - f_j(s)|`` across an ensemble group.

    This is the only quantity in the online metric set needing field-sized
    member-to-member communication; everything else reduces spatially first
    and puts scalars on the wire.

    Members meet via an all-gather: every rank receives the whole ensemble,
    costing ``M`` members of receive buffer per variable chunk but taking a
    *single* collective to do it.  That is what makes it deferrable — the
    exchange can be issued asynchronously and completed during the next
    lead step's model forward, which is what actually hides the transfer.
    ``variable_chunk`` bounds the buffer.

    The exchange uses the identity

    .. code-block:: text

        sum_{i<j} d(i,j) = (1/2) sum_g sum_{i in B_g} sum_{j in ALL} d(i,j)

    so each rank computes the same expression against every member
    (including its own block, where ``d(i,i) = 0`` contributes nothing) and
    the group reduction halves the total.  That keeps the work balanced —
    every rank does exactly ``K * M`` pairs — and removes the index
    bookkeeping a "count each pair once" split would need.

    What crosses the wire is the **residual** ``f_i - y``, not ``f_i``.
    ``|f_i - f_j| = |(f_i - y) - (f_j - y)|``, so the result is identical,
    but the exchanged values sit at forecast-error magnitude rather than
    field magnitude — and the wire dtype's precision is *relative*.  For
    ``z500`` (~5.5e4 m^2/s^2, ensemble spread ~1e2) bf16 rounding of the
    raw field is ~1e2, comparable to the very spread the pairwise term
    measures; on residuals the same rounding is ~100x smaller.  Verification
    is already local on every rank, so this costs nothing.

    Parameters
    ----------
    settings : OnlineSettings
        Supplies the wire dtype, deferral and chunk size.
    var_index : list[int]
        Indices (into the scorer's variable axis) that CRPS covers.
    """

    def __init__(self, settings: OnlineSettings, var_index: list[int]) -> None:
        self._settings = settings
        self._var_index = list(var_index)
        self._pending: _PendingExchange | None = None

    @property
    def defers(self) -> bool:
        """Whether results arrive one lead step late."""
        return self._settings.defer_pairwise_one_step

    def _chunks(self) -> list[list[int]]:
        """Variable index chunks for one exchange."""
        size = self._settings.variable_chunk
        if size <= 0 or size >= len(self._var_index):
            return [self._var_index]
        return [
            self._var_index[i : i + size] for i in range(0, len(self._var_index), size)
        ]

    def step(self, ctx: StepContext) -> tuple[int, torch.Tensor] | None:
        """Advance the exchange by one lead step.

        Parameters
        ----------
        ctx : StepContext
            The lead step being accumulated.

        Returns
        -------
        tuple[int, torch.Tensor] | None
            ``(lead_index, values)`` for whichever lead step just completed
            — the current one, or the previous one when deferring — or
            ``None`` when nothing is ready yet.  ``values`` is only
            meaningful on the group root.
        """
        ready = self._complete(ctx) if self.defers else None
        self._submit(ctx)
        if self.defers:
            return ready
        return self._complete(ctx)

    def drain(self, ctx: StepContext) -> tuple[int, torch.Tensor] | None:
        """Complete any exchange still in flight at the end of an IC.

        Collective: every rank of the group must call this in the same
        place.  They always agree on whether something is pending, since
        the yield sequence is identical across the group.
        """
        return self._complete(ctx)

    def _submit(self, ctx: StepContext) -> None:
        """Issue the all-gather for this lead step's member block."""
        if self._pending is not None:
            raise RuntimeError(
                "PairwiseExchange: a previous exchange is still pending — two "
                "lead steps were submitted without an intervening completion."
            )
        dtype = self._settings.pairwise_comm_dtype
        # Masked once for the whole exchange, before chunking or the wire
        # cast: a gridpoint invalid for its variable (StepContext.valid)
        # must compare as identical "absent" state across every member,
        # not whatever nan_policy left there.  Under zero_fill every
        # member already lands on the same replacement value, so its
        # pairwise diff happens to self-cancel to 0 — but under propagate
        # a lone NaN would poison the entire per-variable sum for this
        # lead, not just the masked gridpoints.  This makes both policies
        # agree, and matches what StepContext.wsum already does for
        # crps_t1.
        d_local = torch.where(ctx.valid, ctx.d_local, 0.0)
        local_chunks: list[torch.Tensor] = []
        gathered: list[list[torch.Tensor]] = []
        work: list[Any] = []
        for chunk in self._chunks():
            index = torch.tensor(chunk, device=ctx.d_local.device, dtype=torch.long)
            block = d_local.index_select(1, index).to(dtype).contiguous()
            buffers, handle = ctx.comm.all_gather(block, async_op=self.defers)
            local_chunks.append(block)
            gathered.append(buffers)
            work.append(handle)
        self._pending = _PendingExchange(
            lead_index=ctx.lead_index,
            local_chunks=local_chunks,
            gathered=gathered,
            work=work,
        )

    def _complete(self, ctx: StepContext) -> tuple[int, torch.Tensor] | None:
        """Wait on the pending all-gather and reduce the pairwise sums."""
        pending, self._pending = self._pending, None
        if pending is None:
            return None

        out_shape: tuple[int, ...] = (len(self._var_index),)
        if ctx.w_flat is not None:
            out_shape = (len(self._var_index), ctx.n_regions)
        total = torch.zeros(out_shape, dtype=torch.float64, device=ctx.d_local.device)
        offset = 0
        for local, buffers, handle in zip(
            pending.local_chunks, pending.gathered, pending.work
        ):
            if handle is not None:
                handle.wait()
            members = torch.cat(buffers, dim=0)
            width = local.shape[1]
            total[offset : offset + width] = _abs_diff_weighted_sum(
                local,
                members,
                ctx.weights,
                ctx.n_spatial,
                member_tile=self._settings.pairwise_member_tile,
                w_flat=ctx.w_flat,
            )
            offset += width

        total *= 0.5
        ctx.comm.reduce(total)
        return pending.lead_index, total


class FairCRPS:
    """Fair (unbiased) CRPS, accumulated as its two sufficient terms.

    .. code-block:: text

        CRPS = (1/M) sum_i E_s|f_i - y|
             - (1/(M(M-1))) sum_{i<j} E_s|f_i - f_j|

    The first term is embarrassingly local — each rank owns whole members —
    so only a ``[variable]`` scalar crosses the wire.  The second is the
    expensive one and is delegated to :class:`PairwiseExchange`.

    Both terms are stored as raw weighted sums (not divided through by
    ``M`` or ``W``) so they stay mergeable across ICs and jobs like every
    other field in the store; :func:`finalize_stats` normalizes.

    When the exchange defers by one lead step, term 2 for step *n* lands
    while step *n+1* is being accumulated — hence the write at the lead
    index the exchange reports rather than at ``ctx.lead_index``.
    """

    name = "crps"

    def __init__(self, settings: OnlineSettings, variables: list[str]) -> None:
        self._settings = settings
        all_variables = list(variables)
        selected = settings.pairwise_variables
        if selected is None:
            self._variables = all_variables
        else:
            missing = [v for v in selected if v not in all_variables]
            if missing:
                raise ValueError(
                    f"scoring.online.pairwise_variables {missing} are not in "
                    f"scoring.variables {all_variables}."
                )
            chosen = set(selected)
            self._variables = [v for v in all_variables if v in chosen]
        self._var_index = [all_variables.index(v) for v in self._variables]
        self._exchange = PairwiseExchange(settings, self._var_index)

    def requires(self) -> set[str]:
        return {"pairwise"}

    def variables(self) -> list[str]:
        """CRPS may cover a subset of the scored variables."""
        return list(self._variables)

    def fields(self) -> dict[str, str]:
        return {"crps_t1": _LAYOUT_SCALAR, "crps_t2": _LAYOUT_SCALAR}

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        self._nr = n_regions
        n = len(self._variables)
        dims = (n_regions,) if n_regions else ()
        self._t1 = _empty(*dims, n_leads, n, device)
        self._t2 = _empty(*dims, n_leads, n, device)

    def update(self, ctx: StepContext) -> None:
        index = torch.tensor(
            self._var_index, device=ctx.f_local.device, dtype=torch.long
        )

        # Term 1: local per member, then a scalar reduce.  ctx.valid is
        # sized to the full scored-variable set, not the (possibly
        # narrower) pairwise `self._variables` subset, so masking must
        # happen before index_select — same order PairwiseExchange._submit
        # uses for term 2 — rather than after, via ctx.wsum.
        d_local = (
            torch.where(ctx.valid, ctx.d_local, 0.0).index_select(1, index).abs()
        ).double()
        if ctx.w_flat is not None:
            flat = d_local.reshape(d_local.shape[0], d_local.shape[1], -1)
            t1 = (flat @ ctx.w_flat).sum(dim=0)  # [var_sub, region]
        else:
            axes = tuple(range(d_local.ndim - ctx.n_spatial, d_local.ndim))
            t1 = (d_local * ctx.weights).sum(dim=axes).sum(dim=0)
        ctx.comm.reduce(t1)
        if ctx.comm.is_root:
            self._t1[..., ctx.lead_index, :] = _region_first(t1, self._nr)

        # Term 2: the member exchange, possibly resolving an earlier step.
        self._store_pairwise(ctx, self._exchange.step(ctx))

    def flush(self, ctx: StepContext) -> None:
        """Complete the deferred exchange left over from the last lead step."""
        self._store_pairwise(ctx, self._exchange.drain(ctx))

    def _store_pairwise(
        self, ctx: StepContext, result: tuple[int, torch.Tensor] | None
    ) -> None:
        if result is None or not ctx.comm.is_root:
            return
        lead_index, values = result
        self._t2[..., lead_index, :] = _region_first(values, self._nr)

    def state(self) -> dict[str, torch.Tensor]:
        return {"crps_t1": self._t1, "crps_t2": self._t2}


def statistic_variables(stat: OnlineStatistic, variables: Sequence[str]) -> list[str]:
    """Variables *stat* emits fields for — all of them unless it says otherwise.

    Only :class:`FairCRPS` narrows this today (via
    ``scoring.online.pairwise_variables``), because it is the only
    statistic whose cost scales with the variable count on the wire.
    """
    selector = getattr(stat, "variables", None)
    if selector is None:
        return list(variables)
    return list(selector())


def build_statistics(
    ensemble_size: int,
    has_climatology: bool = False,
    settings: OnlineSettings | None = None,
    variables: Sequence[str] | None = None,
) -> list[OnlineStatistic]:
    """Assemble the statistics an online run should accumulate.

    Ensemble-only statistics (spread, rank histogram, per-member breakdown,
    CRPS) are omitted for ``M = 1``, so a deterministic online run produces
    exactly the deterministic offline metric set.

    Parameters
    ----------
    ensemble_size : int
        Ensemble size ``M``.
    has_climatology : bool
        Whether a climatology source is configured (enables ACC).
    settings : OnlineSettings | None
        Parsed online settings.  ``None`` omits CRPS — for callers that
        only want the communication-free statistics.
    variables : Sequence[str] | None
        Scored variables, needed to resolve ``pairwise_variables``.
        Required when CRPS is enabled.

    Returns
    -------
    list[OnlineStatistic]
        Statistics in a fixed order — identical on every rank, which is
        what makes the collectives they issue well-ordered.
    """
    stats: list[OnlineStatistic] = [WeightSum(), EnsembleMeanSquaredError()]
    if ensemble_size > 1:
        stats.extend([MemberSquaredError(), EnsembleSpread(), RankHistogram()])
    stats.append(AnomalyMoments(anomaly=has_climatology))

    if settings is not None and settings.mae:
        stats.append(MemberAbsError())
    if settings is not None and settings.lsd:
        whole_grid = None
        if settings.regions is not None:
            whole_grid = [
                i for i, spec in enumerate(settings.regions.values()) if spec is None
            ]
        stats.append(
            LogSpectralDistance(
                whole_grid, wavenumber_cutoff=settings.lsd_wavenumber_cutoff
            )
        )

    if ensemble_size > 1 and settings is not None and settings.crps:
        if variables is None:
            raise ValueError(
                "build_statistics: 'variables' is required when CRPS is enabled."
            )
        stats.append(FairCRPS(settings, list(variables)))
    return stats


# ---------------------------------------------------------------------------
# Statistics store
# ---------------------------------------------------------------------------


def stats_array_groups(
    statistics: Sequence[OnlineStatistic],
    variables: Sequence[str],
    times: np.ndarray,
    lead_times: np.ndarray,
    ensemble_size: int,
    region_names: Sequence[str] | None = None,
) -> tuple[CoordSystem, list[tuple[CoordSystem, list[str]]]]:
    """Build the ``stats.zarr`` schema for the configured statistics.

    Mirrors the offline scorer's layout: a superset coordinate system
    defines the store's axes, and arrays are grouped by dimension
    structure so each group is one ``ZarrBackend.add_array`` call.  Array
    names follow the same ``{field}__{variable}`` convention as
    ``scores.zarr``.

    Parameters
    ----------
    statistics : Sequence[OnlineStatistic]
        Configured statistics.
    variables : Sequence[str]
        Scored variables.
    times : np.ndarray
        All IC times in the campaign.
    lead_times : np.ndarray
        All lead times in the forecast.
    ensemble_size : int
        Ensemble size ``M`` (sets the ``ensemble`` and ``rank_bin`` axes).
    region_names : Sequence[str] | None
        Names of the configured regional splits; every array gains a
        ``region`` axis, whose integer indices map to names kept in the
        store's attributes (:func:`finalize_stats` reattaches them).
        ``None`` keeps the store region-free.

    Returns
    -------
    tuple[CoordSystem, list[tuple[CoordSystem, list[str]]]]
        ``(superset_coords, array_groups)``.
    """
    n_regions = 0 if region_names is None else len(region_names)
    axes: dict[str, np.ndarray] = {
        "time": times,
        "region": np.arange(n_regions),
        "ensemble": np.arange(ensemble_size),
        "rank_bin": np.arange(ensemble_size + 1),
        "lead_time": lead_times,
    }

    # Array names accumulate per layout.  A statistic may cover only a
    # subset of the scored variables (CRPS under `pairwise_variables`), so
    # the name list is built per statistic rather than as a cross product.
    used_layouts: dict[str, list[str]] = {}
    for stat in statistics:
        stat_variables = statistic_variables(stat, variables)
        for field_name, layout in stat.fields().items():
            used_layouts.setdefault(layout, []).extend(
                f"{field_name}__{v}" for v in stat_variables
            )

    superset: CoordSystem = OrderedDict()
    for dim in ("time", "region", "ensemble", "rank_bin", "lead_time"):
        if dim == "region":
            if n_regions:
                superset[dim] = axes[dim]
        elif dim in ("time", "lead_time") or any(
            dim in _LAYOUT_DIMS[layout] for layout in used_layouts
        ):
            superset[dim] = axes[dim]

    groups: list[tuple[CoordSystem, list[str]]] = []
    for layout in (_LAYOUT_SCALAR, _LAYOUT_MEMBER, _LAYOUT_RANK):
        names = used_layouts.get(layout)
        if not names:
            continue
        coords: CoordSystem = OrderedDict(
            (dim, axes[dim]) for dim in _layout_dims(layout, n_regions)
        )
        groups.append((coords, names))
    return superset, groups


def add_stats_arrays(
    io: Any,
    array_groups: list[tuple[CoordSystem, list[str]]],
    region_names: Sequence[str] | None = None,
) -> None:
    """Create the ``stats.zarr`` data arrays, skipping any that exist.

    Deliberately does not go through ``ZarrBackend.add_array``: that helper
    defaults new arrays to ``float32``, which would silently discard the
    float64 accumulation the reductions are built around.  The coordinate
    arrays themselves are already created by
    :meth:`OutputManager.validate_output_store`.

    Safe to call on every rank via ``run_on_rank0_first`` — rank 0 creates,
    later ranks find the arrays present and no-op.

    Parameters
    ----------
    io : ZarrBackend
        Backend from ``OutputManager.io``.
    array_groups : list[tuple[CoordSystem, list[str]]]
        Groups from :func:`stats_array_groups`.
    region_names : Sequence[str] | None
        Regional split names, recorded in the store's attributes so the
        integer ``region`` axis stays interpretable (and so
        :func:`finalize_stats` can reattach them as coordinate labels).
    """
    if region_names is not None:
        existing = io.root.attrs.get("regions")
        if existing is not None and list(existing) != list(region_names):
            raise ValueError(
                f"stats store already holds regions {list(existing)} but the "
                f"current configuration defines {list(region_names)}. "
                "Resuming would relabel previously accumulated statistics; "
                "restore the original scoring.regions or clear the "
                "run's output directory."
            )
        io.root.attrs["regions"] = list(region_names)
    for coords, names in array_groups:
        shape = [len(io.coords[dim]) for dim in coords]
        chunks = [io.chunks.get(dim, len(io.coords[dim])) for dim in coords]
        for name in names:
            if name in io:
                continue
            io.root.create_array(
                name,
                shape=shape,
                chunks=chunks,
                dtype="float64",
                dimension_names=list(coords),
                fill_value=float("nan"),
                compressors=io.zarr_codecs,
            )


def open_stats_store(
    cfg: DictConfig,
    statistics: Sequence[OnlineStatistic],
    variables: Sequence[str],
    times: np.ndarray,
    lead_times: np.ndarray,
    ensemble_size: int,
) -> OutputManager:
    """Create (or validate) ``stats.zarr`` and return its manager.

    Collective across the whole world: every rank must call this, including
    ranks that belong to no ensemble group, because
    :meth:`OutputManager.validate_output_store` barriers internally.

    The store is chunked one IC per chunk (whole lead/member/bin axes) —
    each chunk is a few hundred KB, so per-lead chunking would only produce
    metadata overhead.  That also makes ``(IC, group)`` the natural unit of
    durability: a group root writes one IC's chunks and then its marker.

    Always opened on the synchronous ``ZarrBackend``, regardless of
    ``output.io_backend`` — :func:`add_stats_arrays` creates its arrays
    directly against ``io.root`` (to control dtype and skip
    :meth:`OutputManager.add_array`'s float32 default), which only
    ``ZarrBackend`` exposes synchronously.  ``AsyncZarrBackend.root`` is an
    async zarr group and tracks its own ``parallel_coords``/
    ``chunked_coords`` state from arrays created through :meth:`add_array`,
    so writes against arrays created behind its back would not go through
    the async write path correctly even if the attribute access itself were
    fixed.  stats.zarr is a few hundred KB per IC, so none of
    ``async_zarr``'s threaded/sharded-write machinery would matter here
    anyway.

    The returned manager is **not** entered as a context manager here — the
    caller owns its lifetime so that metadata consolidation happens after
    all groups have finished writing.
    """
    settings = parse_online_settings(cfg)
    region_names = None if settings.regions is None else list(settings.regions)
    superset, groups = stats_array_groups(
        statistics, variables, times, lead_times, ensemble_size, region_names
    )
    mgr = OutputManager(
        cfg, store_name=settings.stats_store, chunks={"time": 1}, io_backend="zarr"
    )
    mgr.validate_output_store(superset, [])
    run_on_rank0_first(add_stats_arrays, mgr.io, groups, region_names)
    return mgr


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------


class OnlineScorer:
    """Drives per-lead-step accumulation for one ensemble group.

    Instantiated on every rank of a group and driven by
    :meth:`src.pipelines.base.Pipeline.run`:
    :meth:`begin_item` per IC, :meth:`update` per yielded chunk,
    :meth:`finish_item` once the IC's rollout completes.  Only the group
    root writes.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.
    settings : OnlineSettings
        Parsed ``scoring.online`` block.
    comm : GroupComm
        This rank's group collectives.
    verification : FieldCache
        Verification field access.
    climatology : FieldCache | None
        Climatology field access, or ``None``.
    variables : list[str]
        Variables to score, in store order.
    lead_times : np.ndarray
        All lead times in the forecast, in store order.
    spatial_coords : CoordSystem
        Spatial coordinate arrays of the scored grid.
    weights : torch.Tensor
        Spatial weights, shaped to the spatial dims.
    stats_mgr : OutputManager
        Manager for ``stats.zarr``.
    device : torch.device
        Device for accumulation.
    known_missing_leads : Iterable[np.timedelta64]
        Lead times the pipeline structurally never yields — e.g.
        ``DLESyMPipeline`` skips its IC-window yield, so ``lead_time=0``
        never arrives (see :meth:`~src.pipelines.base.Pipeline.known_missing_leads`).
        Excluded from :meth:`finish_item`'s missing-lead-time warning so it
        fires only for a genuine gap (a crashed or partially resumed IC),
        not every single IC.
    """

    def __init__(
        self,
        cfg: DictConfig,
        settings: OnlineSettings,
        comm: GroupComm,
        verification: FieldCache,
        climatology: FieldCache | None,
        variables: list[str],
        lead_times: np.ndarray,
        spatial_coords: CoordSystem,
        weights: torch.Tensor,
        stats_mgr: OutputManager,
        device: torch.device,
        known_missing_leads: Iterable[np.timedelta64] = (),
    ) -> None:
        self._cfg = cfg
        self._settings = settings
        self._comm = comm
        self._verification = verification
        self._climatology = climatology
        self._variables = list(variables)
        self._lead_times = np.asarray(lead_times).astype("timedelta64[ns]")
        self._lead_index = {
            int(lt.astype("int64")): i for i, lt in enumerate(self._lead_times)
        }
        self._known_missing_indices = {
            self._lead_index[int(np.timedelta64(lt, "ns").astype("int64"))]
            for lt in known_missing_leads
            if int(np.timedelta64(lt, "ns").astype("int64")) in self._lead_index
        }
        self._expected_leads = set(range(len(self._lead_times))) - (
            self._known_missing_indices
        )
        self._spatial_dims = tuple(d for d in spatial_coords if d not in _NON_SPATIAL)
        self._weights = weights.to(device=device, dtype=torch.float64)
        # Regional runs carry the weights twice: full-shaped for the rank
        # histogram, and flattened [n_points, region] for the matmul path
        # every other reduction takes (see StepContext.w_flat).
        self._n_regions = 0 if settings.regions is None else len(settings.regions)
        self._w_flat = (
            self._weights.reshape(self._n_regions, -1).T.contiguous()
            if self._n_regions
            else None
        )
        self._stats_mgr = stats_mgr
        self._device = device

        self._ensemble_size = comm.group.ensemble_size
        self._statistics = build_statistics(
            self._ensemble_size,
            climatology is not None,
            settings,
            self._variables,
        )
        self._products = {
            product for stat in self._statistics for product in stat.requires()
        }
        unsupported = self._products - set(_PRODUCT_ORDER)
        if unsupported:
            raise ValueError(f"Unknown group products requested: {sorted(unsupported)}")

        self._nan_policy = str(cfg.scoring.get("nan_policy", "propagate")).lower()
        valid_ranges_cfg = cfg.scoring.get("valid_ranges", None)
        self._valid_ranges: dict = (
            OmegaConf.to_container(valid_ranges_cfg, resolve=True) or {}
            if valid_ranges_cfg is not None
            else {}
        )
        # `_apply_valid_ranges` locates the variable axis by name, so the
        # member block needs a coord system with its leading member axis.
        self._member_coords: CoordSystem = OrderedDict(
            {
                "ensemble": np.array(comm.group.member_ids),
                "variable": np.array(self._variables),
            }
        )

        self._item: WorkItem | None = None
        self._seen_leads: set[int] = set()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def begin_item(self, item: WorkItem) -> None:
        """Start accumulating a new initial condition."""
        self._item = item
        self._seen_leads = set()
        for stat in self._statistics:
            stat.reset(
                len(self._lead_times),
                len(self._variables),
                self._ensemble_size,
                self._device,
                n_regions=self._n_regions,
            )

    def update(self, x: torch.Tensor, coords: CoordSystem) -> None:
        """Accumulate one yielded chunk.

        Parameters
        ----------
        x : torch.Tensor
            Model output already filtered to the scored variables and
            regridded to the output grid.  Carries a ``member`` axis of
            size ``members_per_rank`` when the pipeline runs a batched
            rollout.
        coords : CoordSystem
            Matching coordinate system; must carry ``lead_time`` and
            ``variable``.
        """
        if self._item is None:
            raise RuntimeError("OnlineScorer.update called before begin_item.")

        x, lead_times = self._normalize(x, coords)
        for i, lead in enumerate(lead_times):
            self._step(x[i], lead)

    def finish_item(self, item: WorkItem) -> None:
        """Complete deferred work, write the IC's slab, and mark it done.

        The group is already synchronized by the final lead step's
        reduction, so no extra barrier is needed — but the flush below
        *is* collective (it drains the deferred CRPS exchange), so every
        rank of the group runs it before the root writes.
        """
        missing = self._expected_leads - self._seen_leads
        if missing:
            logger.warning(
                f"IC {item.time}: {len(missing)}/{len(self._expected_leads)} "
                "lead times were never yielded; their statistics stay NaN."
            )

        self._flush_statistics()

        if self._comm.is_root:
            self._write_slab(item.time)
            self._stats_mgr.flush()
            write_online_marker(item.time, self._cfg)
        self._item = None

    def _flush_statistics(self) -> None:
        """Let statistics finish deferred work at the end of an IC.

        Only :class:`FairCRPS` uses this today, to drain the member
        exchange the last lead step left in flight.  The hook is
        collective, so it runs in the same fixed statistic order on every
        rank — and unconditionally, since a statistic with nothing pending
        returns without communicating.

        The context carries no per-step fields: there is no step left to
        describe, only the comm, weights and grid metadata the drain needs.
        Holding on to the last real :class:`StepContext` instead would pin
        a few hundred MB of forecast and verification fields until the next
        IC started.
        """
        ctx = self._flush_context()
        for stat in self._statistics:
            flush = getattr(stat, "flush", None)
            if flush is not None:
                flush(ctx)

    def _flush_context(self) -> StepContext:
        """A field-free :class:`StepContext` for end-of-IC flushes."""
        empty = torch.empty(
            (0, len(self._variables)), dtype=torch.float32, device=self._device
        )
        return StepContext(
            lead_index=-1,
            valid_time=np.datetime64("NaT"),
            y=empty,
            clim=None,
            f_local=empty,
            d_local=empty,
            weights=self._weights,
            valid=torch.ones(
                (0, len(self._variables)), dtype=torch.bool, device=self._device
            ),
            n_spatial=len(self._spatial_dims),
            ensemble_size=self._ensemble_size,
            member_ids=self._comm.group.member_ids,
            comm=self._comm,
            w_flat=self._w_flat,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _normalize(
        self, x: torch.Tensor, coords: CoordSystem
    ) -> tuple[torch.Tensor, np.ndarray]:
        """Reshape a yielded chunk to ``(lead_time, member, variable, <spatial...>)``.

        The ``member`` axis is this rank's block: size 1 for the default
        one-member-per-rank layout, and ``members_per_rank`` when the
        pipeline ran a batched rollout (which carries members on the
        ``ensemble`` axis).
        """
        if "lead_time" not in coords or "variable" not in coords:
            raise ValueError(
                "Online scoring requires yielded coords to carry 'lead_time' "
                f"and 'variable'; got {list(coords)}.  Pipelines with a "
                "different yield shape must override the online hook."
            )
        if x.ndim != len(coords):
            raise ValueError(
                f"Yielded tensor has {x.ndim} dimensions but its coords "
                f"describe {len(coords)} ({list(coords)}).  Every coord entry "
                "must correspond to an axis of the tensor."
            )

        dims = list(coords.keys())
        # Structural axes the scorer does not consume must be singletons.
        # 'ensemble' is the exception: a batched rollout carries this rank's
        # members there.  A size-1 ensemble axis is squeezed out and
        # re-added below, so batched and unbatched chunks converge on one
        # layout regardless.
        for dim in ("batch", "time", "ensemble"):
            if dim not in dims:
                continue
            axis = dims.index(dim)
            if x.shape[axis] != 1:
                if dim == "ensemble":
                    continue
                raise ValueError(
                    f"Online scoring expects a singleton '{dim}' axis in "
                    f"yielded chunks; got size {x.shape[axis]}."
                )
            x = x.squeeze(axis)
            dims.pop(axis)

        has_members = "ensemble" in dims
        if has_members:
            self._check_member_block(coords)

        target = ["lead_time", *(["ensemble"] if has_members else []), "variable"]
        target.extend(self._spatial_dims)
        if sorted(target) != sorted(dims):
            raise ValueError(
                f"Yielded coords {dims} do not match the expected online "
                f"layout {target}.  Pipelines with a different yield shape "
                "must be validated before enabling online scoring."
            )
        x = x.permute(*[dims.index(d) for d in target]).contiguous()
        if not has_members:
            x = x.unsqueeze(1)

        # Sub-select the scored variables (and their order) from the chunk.
        chunk_vars = [str(v) for v in coords["variable"]]
        if chunk_vars != self._variables:
            try:
                idx = [chunk_vars.index(v) for v in self._variables]
            except ValueError as exc:
                raise ValueError(
                    f"Scored variables {self._variables} are not all present in "
                    f"the yielded chunk ({chunk_vars})."
                ) from exc
            x = x.index_select(2, torch.tensor(idx, device=x.device, dtype=torch.long))

        return x, np.asarray(coords["lead_time"]).astype("timedelta64[ns]")

    def _check_member_block(self, coords: CoordSystem) -> None:
        """Verify a batched chunk carries exactly this rank's members.

        Member identity is load-bearing: ``sse_member`` is written at
        absolute member indices and the group reduction assumes the blocks
        partition the ensemble.  A pipeline that reordered or renumbered
        them would corrupt the store rather than fail, so check once per
        chunk (it is a handful of integer comparisons).
        """
        got = tuple(int(m) for m in coords["ensemble"])
        expected = self._comm.group.member_ids
        if got != expected:
            raise ValueError(
                f"Batched chunk carries members {got}, but this rank owns "
                f"{expected}.  run_item_batched must yield its items' "
                "ensemble_ids, in order."
            )

    def _step(self, f_local: torch.Tensor, lead: np.timedelta64) -> None:
        """Accumulate one (IC, lead step) contribution across the group.

        Parameters
        ----------
        f_local : torch.Tensor
            This rank's member block, ``[member, variable, <spatial...>]``.
        lead : np.timedelta64
            Lead time of this step.
        """
        if self._item is None:
            raise RuntimeError("OnlineScorer._step called before begin_item.")
        key = int(lead.astype("int64"))
        if self._settings.validate_coords and not self._comm.all_agree(
            key, self._device
        ):
            raise RuntimeError(
                f"Ensemble group {self._comm.group.group_id} desynchronized at "
                f"IC {self._item.time}: this rank is at lead {lead} while "
                "another is elsewhere.  The pipeline's run_item must yield an "
                "identical coord sequence for every ensemble member."
            )
        if key not in self._lead_index:
            raise ValueError(
                f"Yielded lead time {lead} is not in the store's lead_time "
                f"axis ({self._lead_times[0]} .. {self._lead_times[-1]})."
            )
        lead_index = self._lead_index[key]
        self._seen_leads.add(lead_index)

        valid_time = np.datetime64(self._item.time, "ns") + lead
        y = self._verification.get(valid_time)
        clim = self._climatology.get(valid_time) if self._climatology else None

        if f_local.shape[1:] != y.shape:
            raise ValueError(
                f"Forecast field {tuple(f_local.shape[1:])} and verification "
                f"field {tuple(y.shape)} disagree at {valid_time}.  "
                "Verification must be on the same grid as the (possibly "
                "regridded) output."
            )

        # Captured before nan_policy can replace a masked NaN with a value
        # that would otherwise look like a genuine (if wrong) forecast —
        # see StepContext.valid.  Members share the same masked-invalid
        # gridpoints by construction (a pipeline like DLESyMPipeline masks
        # a whole (lead, variable) slice, not per-member), so this is
        # already identical across the group without needing a reduction.
        valid = torch.isfinite(f_local).all(dim=0)

        # Apply the same conditioning the offline scorer applies, so online
        # and offline scores are directly comparable.
        if self._nan_policy == "zero_fill":
            f_local = torch.nan_to_num(f_local, nan=0.0)
        if self._valid_ranges:
            f_local = _apply_valid_ranges(
                f_local.clone(), self._member_coords, self._valid_ranges
            )

        ctx = StepContext(
            lead_index=lead_index,
            valid_time=valid_time,
            y=y,
            clim=clim,
            f_local=f_local,
            d_local=f_local - y,
            weights=self._weights,
            valid=valid,
            n_spatial=len(self._spatial_dims),
            ensemble_size=self._ensemble_size,
            member_ids=self._comm.group.member_ids,
            comm=self._comm,
            w_flat=self._w_flat,
        )
        self._materialize(ctx)
        for stat in self._statistics:
            stat.update(ctx)

    def _materialize(self, ctx: StepContext) -> None:
        """Form the shared group products, in a fixed collective order.

        ``"pairwise"`` is declared in :data:`_PRODUCT_ORDER` but has no
        branch here: unlike the moment and rank products, which several
        statistics share, it is owned entirely by :class:`FairCRPS` (whose
        exchange spans lead steps when deferring, so it cannot be
        materialized per step).  Declaring it still lets the scorer reject
        an unsupported request up front.
        """
        for product in _PRODUCT_ORDER:
            if product not in self._products:
                continue
            if product == "ens_moments":
                dtype = self._settings.moment_comm_dtype
                d = ctx.d_local.double()
                s1 = d.sum(dim=0).to(dtype)
                s2 = (d**2).sum(dim=0).to(dtype)
                ctx.comm.reduce(s1)
                ctx.comm.reduce(s2)
                ctx.s1 = s1.double()
                ctx.s2 = s2.double()
            elif product == "rank_counts":
                below = (ctx.f_local < ctx.y).sum(dim=0, dtype=torch.int32)
                ctx.comm.reduce(below)
                ctx.below = below

    def _axis_values(self, dim: str) -> np.ndarray:
        """Coordinate values of a statistics-store axis."""
        if dim == "region":
            return np.arange(self._n_regions)
        if dim == "ensemble":
            return np.arange(self._ensemble_size)
        if dim == "rank_bin":
            return np.arange(self._ensemble_size + 1)
        if dim == "lead_time":
            return self._lead_times
        raise ValueError(f"Unknown statistics store dimension '{dim}'.")

    def _write_slab(self, time: np.datetime64) -> None:
        """Append this IC's statistics to ``stats.zarr`` (root only).

        Buffers are batched into as few writes as possible: fields sharing
        a layout *and* a variable set already share every axis, so
        concatenating them along the trailing variable axis yields their
        ``{field}__{variable}`` arrays in one call.  CRPS under
        ``pairwise_variables`` covers a narrower set than the rest, which
        is why the variable tuple is part of the key.
        """
        batches: dict[tuple[str, tuple[str, ...]], list[tuple[str, torch.Tensor]]]
        batches = {}
        for stat in self._statistics:
            layouts = stat.fields()
            stat_variables = tuple(statistic_variables(stat, self._variables))
            for name, tensor in stat.state().items():
                key = (layouts[name], stat_variables)
                batches.setdefault(key, []).append((name, tensor))

        for (layout, stat_variables), fields in batches.items():
            names: list[str] = []
            parts: list[torch.Tensor] = []
            for field_name, tensor in fields:
                names.extend(f"{field_name}__{v}" for v in stat_variables)
                parts.append(tensor)

            # (lead, var) -> (1, lead, n_names); (extra, lead, var) ->
            # (1, extra, lead, n_names); a leading region axis on the
            # buffers slots in the same way.
            data = torch.cat(parts, dim=-1).unsqueeze(0).cpu()

            write_coords: CoordSystem = OrderedDict()
            write_coords["time"] = np.array([time])
            for dim in _layout_dims(layout, self._n_regions)[1:]:
                write_coords[dim] = self._axis_values(dim)
            write_coords["variable"] = np.array(names)
            self._stats_mgr.write(data, write_coords)


# ---------------------------------------------------------------------------
# Finalize: stats.zarr -> scores.zarr
# ---------------------------------------------------------------------------


def _finalize_variable(
    ds: xr.Dataset,
    var: str,
    ensemble_size: int,
) -> dict[str, xr.DataArray]:
    """Derive one variable's score arrays from its sufficient statistics."""
    out: dict[str, xr.DataArray] = {}
    w = ds[f"w_sum__{var}"]

    # Per-member MSE.  For M = 1 there is no member breakdown to store, so
    # the ensemble-mean sum *is* the deterministic MSE.
    member_key = f"sse_member__{var}"
    if member_key in ds:
        out[f"mse__{var}"] = ds[member_key] / w
        out[f"ensemble_mean_mse__{var}"] = ds[f"sse_ensmean__{var}"] / w
    else:
        out[f"mse__{var}"] = ds[f"sse_ensmean__{var}"] / w

    if f"var_ens__{var}" in ds:
        out[f"ensemble_variance__{var}"] = ds[f"var_ens__{var}"] / w

    # Optional per-member additions (scoring.online.mae / .lsd).  The
    # store keeps LSD as a final per-(IC, member, lead) value rather than
    # a sum — its IC aggregation is a plain mean, so nothing normalizes it.
    if f"sae_member__{var}" in ds:
        out[f"mae__{var}"] = ds[f"sae_member__{var}"] / w
    if f"lsd__{var}" in ds:
        out[f"lsd__{var}"] = ds[f"lsd__{var}"]

    # Fair CRPS.  t1 and t2 are stored as raw weighted sums over members
    # and pairs respectively, so the normalization lives here:
    #   CRPS = (1/M) mean_s|f_i - y| - (1/(M(M-1))) sum_{i<j} mean_s|f_i - f_j|
    # The (M-1) denominator is what makes the estimator *fair* (unbiased
    # for the infinite-ensemble CRPS) rather than the biased 1/(2M^2) form.
    if f"crps_t1__{var}" in ds and ensemble_size > 1:
        m = ensemble_size
        out[f"crps__{var}"] = ds[f"crps_t1__{var}"] / (m * w) - ds[
            f"crps_t2__{var}"
        ] / (m * (m - 1) * w)

    if f"rank_counts__{var}" in ds:
        counts = ds[f"rank_counts__{var}"]
        total = counts.sum(dim="rank_bin")
        uniform = 1.0 / (ensemble_size + 1)
        # skipna=False: xarray's DataArray.sum() defaults to skipping NaN,
        # which would silently turn a fully-masked (lead, variable) --
        # counts/total = 0/0 = NaN in every bin -- into a "perfectly
        # calibrated" 0.0 instead of propagating the NaN every other
        # metric already gives it there.
        out[f"rank_reliability__{var}"] = np.abs(counts / total - uniform).sum(
            dim="rank_bin", skipna=False
        )

    # Moment-derived scores.  Anomaly-space sums (a/b) additionally support
    # ACC; raw sums (f/y) give bias and correlation only.
    anomaly = f"sum_wa__{var}" in ds
    f_key, y_key = ("a", "b") if anomaly else ("f", "y")
    sum_f = ds[f"sum_w{f_key}__{var}"]
    sum_y = ds[f"sum_w{y_key}__{var}"]
    sum_f2 = ds[f"sum_w{f_key}2__{var}"]
    sum_y2 = ds[f"sum_w{y_key}2__{var}"]
    sum_fy = ds[f"sum_w{f_key}{y_key}__{var}"]

    out[f"bias__{var}"] = (sum_f - sum_y) / w
    cov = sum_fy - sum_f * sum_y / w
    var_f = sum_f2 - sum_f**2 / w
    var_y = sum_y2 - sum_y**2 / w
    out[f"corr__{var}"] = cov / np.sqrt(var_f * var_y)
    if anomaly:
        out[f"acc__{var}"] = sum_fy / np.sqrt(sum_f2 * sum_y2)

    return out


def finalize_scores_store_name(cfg: DictConfig) -> str:
    """Filename :func:`finalize_stats` writes under ``output.path``.

    An explicit ``scoring.online.scores_store`` wins; otherwise the online
    scores land in ``scoring.output.store_name`` like any other scores.
    Set it when validating a run against an offline pass over the same
    retained fields, so the two stores sit side by side instead of one
    silently overwriting the other.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.

    Returns
    -------
    str
        Store filename (not a path).
    """
    settings = parse_online_settings(cfg)
    if settings.scores_store is not None:
        return settings.scores_store
    return str(cfg.scoring.output.get("store_name", "scores.zarr"))


def finalize_stats(cfg: DictConfig) -> str:
    """Derive ``scores.zarr`` from the accumulated ``stats.zarr``.

    Single-process and cheap (the whole statistics store is ~500 KB per
    IC), so it runs at the end of an online inference job and can be re-run
    at any time — including after merging statistics from several jobs.

    The emitted arrays use the offline scorer's ``{metric}__{variable}``
    naming and the metric keys the report package already understands, so
    ``report.py`` is untouched:

    ==========================  =========================================
    Array                       Definition
    ==========================  =========================================
    ``mse__{v}``                per-member MSE (``ensemble`` axis when M>1)
    ``ensemble_mean_mse__{v}``  MSE of the ensemble mean
    ``ensemble_variance__{v}``  Bessel-corrected ensemble variance
    ``crps__{v}``               fair (unbiased) CRPS
    ``rank_reliability__{v}``   ``sum_k |p_k - 1/(M+1)|`` of the histogram
    ``bias__{v}``               mean forecast minus mean verification
    ``corr__{v}``               centered spatial Pearson correlation
    ``acc__{v}``                anomaly correlation (climatology only)
    ``mae__{v}``                per-member MAE (``scoring.online.mae``)
    ``lsd__{v}``                per-member log spectral distance, dB
                                (``scoring.online.lsd``)
    ==========================  =========================================

    With ``scoring.regions`` configured every array carries a
    ``region`` axis, labeled with the configured region names.

    Spread and SSR are not written: the report derives them from
    ``ensemble_mean_mse`` and ``ensemble_variance`` with the correct
    ``sqrt(mean_t(.))`` ordering.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.

    Returns
    -------
    str
        Path of the written score store.

    Raises
    ------
    FileNotFoundError
        If the statistics store does not exist.
    """
    settings = parse_online_settings(cfg)
    stats_path = os.path.join(cfg.output.path, settings.stats_store)
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Statistics store not found at '{stats_path}'.\n"
            "Run inference (main.py) with scoring.mode=online first."
        )

    scores_path = os.path.join(cfg.output.path, finalize_scores_store_name(cfg))

    ds = xr.open_zarr(stats_path)
    ensemble_size = int(ds.sizes.get("ensemble", 1))
    # Regional runs store the region axis as integer indices; the names
    # live in the store attributes and become coordinate labels here.
    region_names = ds.attrs.get("regions")
    if region_names is not None and "region" in ds.dims:
        ds = ds.assign_coords(region=np.array([str(r) for r in region_names]))
    variables = sorted(
        {str(name).split("__", 1)[1] for name in ds.data_vars if "__" in str(name)}
    )

    arrays: dict[str, xr.DataArray] = {}
    for var in variables:
        arrays.update(_finalize_variable(ds, var, ensemble_size))

    # Statistics for ICs that were never run stay NaN rather than being
    # silently dropped, matching the offline scorer's partial-run behavior.
    scores = xr.Dataset(arrays).compute()
    if region_names is not None:
        scores.attrs["regions"] = list(region_names)
    # Rebuild from scratch: the derived metric set depends on what is in
    # stats.zarr (member breakdown, rank counts, climatology), so an
    # in-place overwrite could leave stale arrays from an earlier config.
    if os.path.exists(scores_path):
        shutil.rmtree(scores_path)
    scores.to_zarr(scores_path, mode="w", consolidated=True)
    logger.success(
        f"Finalized {len(arrays)} score arrays for {len(variables)} variables "
        f"→ {scores_path}"
    )
    return scores_path


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


def build_online_scorer(
    cfg: DictConfig,
    settings: OnlineSettings,
    comm: GroupComm,
    verification_source: DataSource,
    variables: list[str],
    lead_times: np.ndarray,
    spatial_coords: CoordSystem,
    stats_mgr: OutputManager,
    device: torch.device,
    known_missing_leads: Iterable[np.timedelta64] = (),
) -> OnlineScorer:
    """Assemble an :class:`OnlineScorer` from config and resolved sources.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.
    settings : OnlineSettings
        Parsed ``scoring.online`` block.
    comm : GroupComm
        This rank's ensemble group communicator.
    verification_source : DataSource
        Predownloaded verification source.
    variables : list[str]
        Variables to score.
    known_missing_leads : Iterable[np.timedelta64]
        Forwarded to :class:`OnlineScorer` — lead times the driving
        pipeline structurally never yields (see
        :meth:`~src.pipelines.base.Pipeline.known_missing_leads`).
    lead_times : np.ndarray
        Full lead-time axis of the forecast.
    spatial_coords : CoordSystem
        Spatial coordinates of the scored (post-regrid) grid.
    stats_mgr : OutputManager
        Manager for ``stats.zarr``.
    device : torch.device
        Device for accumulation.

    Returns
    -------
    OnlineScorer
    """
    spatial_dims = tuple(d for d in spatial_coords if d not in _NON_SPATIAL)
    nan_policy = str(cfg.scoring.get("nan_policy", "propagate")).lower()
    if nan_policy not in ("propagate", "zero_fill"):
        raise ValueError(
            f"Invalid scoring.nan_policy '{nan_policy}'; "
            "expected 'propagate' or 'zero_fill'."
        )
    valid_ranges_cfg = cfg.scoring.get("valid_ranges", None)
    valid_ranges = (
        OmegaConf.to_container(valid_ranges_cfg, resolve=True) or {}
        if valid_ranges_cfg is not None
        else {}
    )

    verification = FieldCache(
        verification_source,
        variables,
        spatial_dims,
        device,
        max_size=settings.verification_cache_size,
        nan_policy=nan_policy,
        valid_ranges=valid_ranges,
    )

    climatology = None
    if settings.climatology is not None:
        logger.info("Instantiating climatology source for ACC.")
        climatology = FieldCache(
            hydra.utils.instantiate(settings.climatology),
            variables,
            spatial_dims,
            device,
            max_size=settings.verification_cache_size,
            nan_policy=nan_policy,
        )

    weights = build_spatial_weights(
        spatial_coords,
        bool(cfg.scoring.get("lat_weights", False)),
        regions=settings.regions,
    )

    return OnlineScorer(
        cfg=cfg,
        settings=settings,
        comm=comm,
        verification=verification,
        climatology=climatology,
        variables=variables,
        lead_times=lead_times,
        spatial_coords=spatial_coords,
        weights=weights,
        stats_mgr=stats_mgr,
        device=device,
        known_missing_leads=known_missing_leads,
    )
