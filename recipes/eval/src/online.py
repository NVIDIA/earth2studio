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

"""Online (in-line) scoring: score forecasts inside the inference loop.

The default recipe flow is *store-then-score*: ``main.py`` writes every
(member, IC, lead, variable) field to ``forecast.zarr`` and ``score.py``
re-reads it.  At campaign scale that raw store is the binding constraint
(~170 TB for a 50-member, 730-IC, 14-day global campaign), even though the
scores derived from it are ~20 MB.

Online scoring removes the raw store from the critical path.  It plugs into
:class:`earth2studio.run.Pipeline` through its :class:`~earth2studio.run.pipeline.Scorer`
hook: the pipeline hands every filtered, regridded field to
:meth:`OnlineScorer.update`, which reads the verification field for
``IC + lead``, forms this rank's contributions, joins one reduction over its
ensemble group, and lets the group root append an IC slab of
mergeable statistics to ``stats.zarr`` in :meth:`OnlineScorer.finish_item`.

Ensemble groups
---------------
Contiguous blocks of ``G = ensemble_size / members_per_rank`` ranks form the
ensemble groups.  :func:`src.campaign.order_for_groups` lays the
work items out so that the pipeline's contiguous rank slices give every rank
of a block the same initial conditions in the same rounds, each rank carrying
one member group; rank ``b * G + g`` carries member group ``g``.  Collectives
inside the scorer hooks run over ``torch.distributed`` sub-groups of one
block only; the pipeline's per-round ``sync`` hook is a no-op here, and
exists solely to switch the pipeline into lockstep rounds.

Mergeable statistics, not scores
--------------------------------
Nothing final is written during inference.  ``stats.zarr`` holds mergeable
per-``(IC, lead, variable)`` sums (squared errors, ensemble moments, weighted
rank counts, anomaly cross-moments) from which :func:`finalize_stats` derives
a ``scores.zarr`` that is schema-compatible with the offline scorer's output
(``{metric}__{variable}`` arrays).  The sums carry the spatial weighting, so
the set of spatial regions (``scoring.regions`` plus the boxes of
``scoring.events``) is a run-time input.

Numerics
--------
All spatial reductions accumulate in ``float64``.  Group reductions are
formed on the **residual** ``d_i = f_i - y`` rather than on ``f_i``: the
ensemble variance is shift-invariant, so this is algebraically identical
while keeping the summands at forecast-error size.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections import OrderedDict
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
import torch.distributed as dist
import xarray as xr
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from earth2studio.data import fetch_data
from earth2studio.grids import Regridder, infer_grid, same_geometry
from earth2studio.grids.base import GridDefinition
from earth2studio.io import ZarrBackend
from earth2studio.run import ProgressStore, WorkItem
from earth2studio.run.work import MANIFEST_NAME, broadcast, rank_info
from earth2studio.utils.coords import CoordSystem, coord_array
from earth2studio.utils.cupy import from_torch
from earth2studio.utils.type import CoordinateSystem

from .campaign import online_scoring, order_for_groups
from .data import instantiate_source
from .logs import all_agree, rank0_first_checked
from .output import (
    OutputRegridder,
    output_regridder,
    prepare_store,
    progress_path,
    progress_tags,
    recorded_items,
)
from .regions import (
    build_spatial_weights,
    events_to_attrs,
    parse_events,
    scoring_regions,
)
from .scoring import open_verification_source

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

# Default members per float64 abs-diff tile; see OnlineSettings.
_PAIRWISE_MEMBER_TILE = 8


def _layout_dims(layout: str, n_regions: int) -> tuple[str, ...]:
    """Store dims for ``layout``; a ``region`` axis follows ``time`` with regions."""
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
        ``G = ensemble_size / members_per_rank``; an explicit value must
        equal that quotient.
    members_per_rank : int
        Ensemble members per rank ``K``; every item carries this many members.
    stats_store : str
        Filename of the statistics store under ``output.path``.
    scores_store : str | None
        Where :func:`finalize_stats` writes.  ``None`` falls back to
        ``scoring.output.store_name``.
    moment_comm_dtype : torch.dtype
        Dtype of the field-sized ensemble-moment reduction.
    climatology : Any
        Optional source node supplying climatology on the scored grid.
        Required for ACC; ``None`` disables it.
    verification_cache_size : int
        Number of verification valid times held per rank.
    validate_coords : bool
        Whether to cross-check each step's lead time across the group.
    nccl_timeout_s : int
        Watchdog timeout on the group process groups.
    crps : bool
        Whether to accumulate fair CRPS, the only statistic that needs
        field-sized member-to-member communication.
    mae : bool
        Accumulate per-member weighted absolute error.
    lsd : bool
        Compute the per-member log spectral distance.
    lsd_wavenumber_cutoff : int | None
        Optional radial-mode cutoff forwarded to the earth2studio metric.
    pairwise_comm_dtype : torch.dtype
        Wire dtype for the member exchange; arithmetic upcasts to float64.
    defer_pairwise_one_step : bool
        Complete the exchange for one update during the next update, so the
        transfer overlaps the model forward.
    variable_chunk : int
        Variables per exchange and pairwise-compute step; ``0`` disables
        chunking.
    pairwise_variables : list[str] | None
        Restrict CRPS to a subset of the scored variables.
    pairwise_member_tile : int
        Members per float64 abs-diff tile in :func:`_abs_diff_weighted_sum`.
    regions : dict[str, list[dict] | None] | None
        Named spatial splits (see :func:`src.regions.scoring_regions`).
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

    Raises on an unrecognized mode rather than falling back to offline: a
    typo that silently downgraded a campaign to the offline path would
    surface only as a surprise ``forecast.zarr`` the size of a filesystem.
    """
    return online_scoring(cfg)


def parse_online_settings(cfg: DictConfig) -> OnlineSettings:
    """Read and validate the ``scoring.online`` config block.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.

    Returns
    -------
    OnlineSettings
        Parsed settings.

    Raises
    ------
    ValueError
        On an invalid dtype or count, or on a setting that has since been
        removed, which raises rather than passing silently.
    """
    scoring = cfg.get("scoring", None) or {}
    block = scoring.get("online", None) or {}

    # `x or default` would fold 0 into the default and silently accept it.
    members_per_rank = _int_or(block.get("members_per_rank", 1), 1)
    if members_per_rank < 1:
        raise ValueError(
            f"scoring.online.members_per_rank must be >= 1, got {members_per_rank}."
        )

    moment_dtype = _parse_dtype(
        block.get("moment_comm_dtype", "float64"),
        {"float64": torch.float64, "float32": torch.float32},
        "moment_comm_dtype",
    )
    pairwise_dtype = _parse_dtype(
        block.get("pairwise_comm_dtype", "bfloat16"),
        {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
        },
        "pairwise_comm_dtype",
    )
    if "pairwise_exchange" in block:
        raise ValueError(
            "scoring.online.pairwise_exchange has been removed; the member "
            "exchange is always an all-gather bounded by variable_chunk."
        )

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

    regions = scoring_regions(scoring)
    lsd_cfg = block.get("lsd", False)
    lsd_cutoff = None
    if isinstance(lsd_cfg, (dict, DictConfig)):
        lsd_opts = (
            OmegaConf.to_container(lsd_cfg, resolve=True)
            if isinstance(lsd_cfg, DictConfig)
            else dict(lsd_cfg)
        )
        unknown = set(lsd_opts) - {"wavenumber_cutoff"}  # type: ignore[arg-type]
        if unknown:
            raise ValueError(
                f"Unknown scoring.online.lsd option(s) {sorted(unknown)}; "
                "expected only 'wavenumber_cutoff'."
            )
        lsd = True
        if lsd_opts.get("wavenumber_cutoff") is not None:  # type: ignore[union-attr]
            lsd_cutoff = int(lsd_opts["wavenumber_cutoff"])  # type: ignore[index]
    else:
        lsd = bool(lsd_cfg)
    if lsd and regions is not None and not any(v is None for v in regions.values()):
        raise ValueError(
            "scoring.online.lsd needs a whole-grid region (an entry with a "
            "null box) to land its values in; add e.g. 'global: null' to "
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
        mae=bool(block.get("mae", False)),
        lsd=lsd,
        lsd_wavenumber_cutoff=lsd_cutoff,
        pairwise_comm_dtype=pairwise_dtype,
        defer_pairwise_one_step=bool(block.get("defer_pairwise_one_step", True)),
        variable_chunk=variable_chunk,
        pairwise_variables=pairwise_variables,
        pairwise_member_tile=pairwise_member_tile,
        regions=regions,
    )


def _int_or(value: Any, default: int) -> int:
    """``int(value)``, mapping only ``None`` to ``default``."""
    return default if value is None else int(value)


def _parse_dtype(value: Any, allowed: dict[str, torch.dtype], key: str) -> torch.dtype:
    """Resolve a config dtype name against the options valid for ``key``."""
    name = str(value).lower()
    if name not in allowed:
        raise ValueError(
            f"Invalid scoring.online.{key} '{name}'; expected one of {sorted(allowed)}."
        )
    return allowed[name]


# ---------------------------------------------------------------------------
# Ensemble groups and group communication
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnsembleGroup:
    """One rank's place in the block layout of ensemble groups.

    Ranks ``b * G .. (b + 1) * G - 1`` form block ``b``; rank ``b * G + g``
    carries member group ``g``, members ``[g * K, (g + 1) * K)``.

    Attributes
    ----------
    block : int
        Block index ``b``.
    group_rank : int
        Position ``g`` of this rank within its block.
    group_size : int
        Ranks per block ``G``.
    members_per_rank : int
        Members per rank ``K``.
    ranks : tuple[int, ...]
        Global ranks of the block, in order.
    """

    block: int
    group_rank: int
    group_size: int
    members_per_rank: int
    ranks: tuple[int, ...]

    @property
    def ensemble_size(self) -> int:
        """Total ensemble size ``M = G * K``."""
        return self.group_size * self.members_per_rank

    @property
    def member_ids(self) -> tuple[int, ...]:
        """Global member indices this rank carries."""
        start = self.group_rank * self.members_per_rank
        return tuple(range(start, start + self.members_per_rank))

    @property
    def is_root(self) -> bool:
        """Whether this rank writes the group's statistics slabs."""
        return self.group_rank == 0

    @property
    def root_rank(self) -> int:
        """Global rank of the group root."""
        return self.ranks[0]


def plan_groups(settings: OnlineSettings, ensemble_size: int) -> tuple[int, int]:
    """Derive the group size ``G`` and members per rank ``K``.

    Parameters
    ----------
    settings : OnlineSettings
        Parsed online settings.
    ensemble_size : int
        Ensemble size ``M`` (1 for a deterministic campaign).

    Returns
    -------
    tuple[int, int]
        ``(G, K)`` with ``G * K == M``.

    Raises
    ------
    ValueError
        If ``K`` does not divide ``M`` or an explicit ``ensemble_group_size``
        disagrees with ``M / K``.
    """
    members_per_rank = settings.members_per_rank
    if ensemble_size % members_per_rank:
        raise ValueError(
            f"scoring.online.members_per_rank={members_per_rank} must divide "
            f"ensemble_size={ensemble_size}."
        )
    group_size = ensemble_size // members_per_rank
    if (
        settings.ensemble_group_size is not None
        and settings.ensemble_group_size != group_size
    ):
        raise ValueError(
            f"scoring.online.ensemble_group_size={settings.ensemble_group_size} "
            f"must equal ensemble_size / members_per_rank = {group_size}."
        )
    return group_size, members_per_rank


def group_for_rank(
    rank: int, world_size: int, group_size: int, members_per_rank: int
) -> EnsembleGroup:
    """This rank's :class:`EnsembleGroup` under the block layout.

    Parameters
    ----------
    rank : int
        Global rank.
    world_size : int
        Number of ranks; ``group_size`` must divide it.
    group_size : int
        Ranks per block ``G``.
    members_per_rank : int
        Members per rank ``K``.

    Returns
    -------
    EnsembleGroup
        The group membership of ``rank``.

    Raises
    ------
    ValueError
        If ``group_size`` does not divide ``world_size``.
    """
    if group_size < 1 or world_size % group_size:
        raise ValueError(
            f"world_size={world_size} must be a multiple of the ensemble group "
            f"size {group_size} (ensemble_size / members_per_rank); launch "
            f"a multiple of {group_size} ranks or raise members_per_rank."
        )
    block, group_rank = divmod(rank, group_size)
    return EnsembleGroup(
        block=block,
        group_rank=group_rank,
        group_size=group_size,
        members_per_rank=members_per_rank,
        ranks=tuple(range(block * group_size, (block + 1) * group_size)),
    )


class GroupComm:
    """Collectives over one ensemble group.

    Wraps the group's ``torch.distributed`` process group and degrades to a
    no-op when the group holds a single rank, which is the deterministic and
    single-process path.  Every method is collective over the group.

    Parameters
    ----------
    group : EnsembleGroup
        This rank's group membership.
    process_group : Any
        The ``torch.distributed`` process group covering ``group.ranks``, or
        ``None`` when the group needs no communication.
    """

    def __init__(self, group: EnsembleGroup, process_group: Any = None) -> None:
        self._group = group
        self._pg = process_group

    @property
    def group(self) -> EnsembleGroup:
        """The :class:`EnsembleGroup` this comm serves."""
        return self._group

    @property
    def is_root(self) -> bool:
        """Whether this rank writes the group's statistics slabs."""
        return self._group.is_root

    @property
    def active(self) -> bool:
        """Whether collectives cross the wire."""
        return self._pg is not None

    def reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sum-reduce ``tensor`` onto the group root, in place.

        Non-root ranks end up holding an unspecified value.
        """
        if self._pg is not None:
            dist.reduce(
                tensor, dst=self._group.root_rank, op=dist.ReduceOp.SUM, group=self._pg
            )
        return tensor

    def all_gather(
        self, tensor: torch.Tensor, async_op: bool = False
    ) -> tuple[list[torch.Tensor], Any]:
        """Gather every rank's ``tensor`` onto every rank of the group.

        Returns ``(per_rank_buffers, work_handle)``; the handle is ``None``
        when nothing crossed the wire or ``async_op`` is false.
        """
        if self._pg is None:
            return [tensor], None
        buffers = [torch.empty_like(tensor) for _ in range(self._group.group_size)]
        work = dist.all_gather(buffers, tensor, group=self._pg, async_op=async_op)
        return buffers, (work if async_op else None)

    def all_and(self, mask: torch.Tensor) -> torch.Tensor:
        """Logical AND of a boolean ``mask`` over the group, on every rank."""
        if self._pg is None:
            return mask
        flags = mask.to(torch.int32)
        dist.all_reduce(flags, op=dist.ReduceOp.MIN, group=self._pg)
        return flags.bool()

    def all_agree(self, value: int, device: torch.device) -> bool:
        """Whether every rank in the group passed the same ``value``."""
        if self._pg is None:
            return True
        t = torch.tensor([value, -value], dtype=torch.int64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self._pg)
        return bool(t[0].item() == value and -t[1].item() == value)


def build_group_comm(
    settings: OnlineSettings, rank: int, world_size: int, ensemble_size: int
) -> GroupComm:
    """Create this rank's :class:`GroupComm`.  Collective over the world.

    ``torch.distributed.new_group`` is itself collective, so every rank must
    call this function, which creates the groups in a fixed order.

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
    GroupComm
        Group collectives for this rank.
    """
    group_size, members_per_rank = plan_groups(settings, ensemble_size)
    group = group_for_rank(rank, world_size, group_size, members_per_rank)
    process_group = None
    if group_size > 1 and dist.is_available() and dist.is_initialized():
        timeout = timedelta(seconds=settings.nccl_timeout_s)
        for block in range(world_size // group_size):
            ranks = list(range(block * group_size, (block + 1) * group_size))
            pg = dist.new_group(ranks=ranks, timeout=timeout)
            if block == group.block:
                process_group = pg
    logger.info(
        f"Rank {rank}: ensemble group {group.block}/{world_size // group_size} "
        f"(G={group_size}, K={members_per_rank}, members={list(group.member_ids)}, "
        f"root={group.is_root})"
    )
    return GroupComm(group, process_group)


# ---------------------------------------------------------------------------
# Verification / climatology access
# ---------------------------------------------------------------------------


def _apply_valid_ranges(
    x: torch.Tensor, variables: Sequence[str], valid_ranges: Mapping[str, Any], dim: int
) -> torch.Tensor:
    """Clamp per-variable slices of ``x`` along axis ``dim`` to their physical range.

    ``valid_ranges`` maps variable names to ``{min, max}``; either bound may
    be absent for a one-sided clamp.  Variables not listed keep their values.
    """
    for index, name in enumerate(variables):
        bounds = valid_ranges.get(name)
        if not bounds:
            continue
        low = bounds.get("min", None)
        high = bounds.get("max", None)
        if low is None and high is None:
            continue
        view = x.select(dim, index)
        view.clamp_(
            min=None if low is None else float(low),
            max=None if high is None else float(high),
        )
    return x


def _check_spatial_coords(
    da: xr.DataArray, spatial_coords: CoordSystem, label: str
) -> None:
    """Require ``da`` to sit on the scored grid, dimension by dimension."""
    for dim, expected in spatial_coords.items():
        expected = np.asarray(expected)
        if dim in da.coords and da.coords[dim].dims == (dim,):
            actual = np.asarray(da.coords[dim].values)
            if actual.shape != expected.shape or not np.allclose(
                actual.astype(float), expected.astype(float), atol=1e-6
            ):
                raise ValueError(
                    f"{label} field coordinate '{dim}' does not match the scored "
                    f"grid ({actual.shape} vs {expected.shape}).  Verification "
                    "must be on the same grid as the (regridded) output."
                )
        elif da.sizes.get(dim) != len(expected):
            raise ValueError(
                f"{label} field has {da.sizes.get(dim)} points along '{dim}'; the "
                f"scored grid has {len(expected)}."
            )


class FieldCache:
    """Valid-time keyed LRU cache over a gridded data source.

    Serves verification (and climatology) fields to the scoring loop through
    :func:`earth2studio.data.fetch_data`.  With an ``output.regrid`` block, its
    method brings the fields onto the scored grid; without one the library
    crops, or regrids by nearest neighbor, a source on a larger grid.
    The cache normalizes fields once on the way in (variable order, spatial
    dimension order, NaN policy and valid-range clamping) so the scoring loop
    treats a cached field as directly comparable to a model output.

    Parameters
    ----------
    source : Any
        Data source to read from; expected to be local (predownloaded).
    variables : Sequence[str]
        Variables to fetch, in the order the scorer expects them.
    grid : GridDefinition
        Scored grid to request fields on.
    spatial_coords : CoordSystem
        Spatial coordinate arrays of the scored grid, in dimension order.
    device : torch.device
        Device to stage fields on.
    max_size : int, optional
        Number of valid times to keep, by default 4
    nan_policy : str, optional
        ``"propagate"``, ``"zero_fill"`` or ``"mask"`` (see
        :class:`OnlineScorer`); the cache zero-fills NaN under ``zero_fill``
        and leaves it in place otherwise, by default ``"propagate"``
    valid_ranges : Mapping | None, optional
        Per-variable ``{min, max}`` clamps, by default None
    label : str, optional
        Name used in error messages, by default ``"Verification"``
    regrid : OutputRegridder, optional
        The campaign's ``output.regrid`` regridder, by default None
    """

    def __init__(
        self,
        source: Any,
        variables: Sequence[str],
        grid: GridDefinition,
        spatial_coords: CoordSystem,
        device: torch.device,
        max_size: int = 4,
        nan_policy: str = "propagate",
        valid_ranges: Mapping[str, Any] | None = None,
        label: str = "Verification",
        regrid: OutputRegridder | None = None,
    ) -> None:
        self._source = source
        self._variables = [str(v) for v in variables]
        self._grid = grid
        self._regrid = regrid
        self._spatial_coords = spatial_coords
        self._spatial_dims = tuple(spatial_coords)
        self._device = torch.device(device)
        self._max_size = max(1, max_size)
        self._nan_policy = nan_policy
        self._valid_ranges = dict(valid_ranges or {})
        self._label = label
        self._cache: OrderedDict[int, torch.Tensor] = OrderedDict()

    @property
    def variables(self) -> list[str]:
        """Variables in field order."""
        return list(self._variables)

    def get(self, valid_time: np.datetime64) -> torch.Tensor:
        """Return the field at ``valid_time`` as a ``[variable, <spatial...>]`` tensor."""
        key = int(np.datetime64(valid_time, "ns").astype("int64"))
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached
        field = self._fetch(np.datetime64(valid_time, "ns"))
        self._cache[key] = field
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return field

    def _fetch(self, valid_time: np.datetime64) -> torch.Tensor:
        da = fetch_data(
            self._source,
            np.array([valid_time], dtype="datetime64[ns]"),
            np.array(self._variables),
            device=self._device,
            target_grid=None if self._regrid is not None else self._grid,
        )
        if self._regrid is not None:
            da = self._regrid(da)
        for dim in ("time", "lead_time"):
            if dim in da.dims:
                da = da.isel({dim: 0}, drop=True)
        da = da.transpose("variable", *self._spatial_dims)
        _check_spatial_coords(da, self._spatial_coords, self._label)
        tensor, _ = da.e2s.to_torch()
        tensor = tensor.to(device=self._device, dtype=torch.float32)
        if self._nan_policy == "zero_fill":
            tensor = torch.nan_to_num(tensor, nan=0.0)
        if self._valid_ranges:
            tensor = _apply_valid_ranges(
                tensor.clone(), self._variables, self._valid_ranges, dim=0
            )
        return tensor


def available_times(source: Any) -> np.ndarray | None:
    """Return the valid times a local zarr-backed source can serve, if knowable.

    Inspects the ``xr.DataArray`` that :class:`~src.data.PredownloadedSource`
    wraps.  Returns ``None`` for sources that do not expose their coverage
    cheaply; callers should treat that as "unknown", not as "empty".
    """
    da = getattr(source, "_da", None)
    if da is not None and "time" in getattr(da, "coords", {}):
        return np.asarray(da.coords["time"].values).astype("datetime64[ns]")
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
    source: Any,
    ic_times: Sequence[np.datetime64],
    lead_times: np.ndarray,
    variables: Sequence[str],
) -> None:
    """Fail fast when verification does not cover every scored valid time.

    Parameters
    ----------
    source : Any
        Verification source.
    ic_times : Sequence[np.datetime64]
        Initial times of the campaign.
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
            "does not expose a time index); a missing valid time will surface "
            "as an in-loop failure instead."
        )
        return
    needed = np.unique(
        np.asarray(
            [
                np.datetime64(t, "ns") + np.timedelta64(lt, "ns")
                for t in ic_times
                for lt in np.asarray(lead_times).astype("timedelta64[ns]")
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
            f"(first missing: {preview}).  Re-run predownload.py with "
            "predownload.verification.enabled=true for the full forecast window."
        )
    logger.info(
        f"Verification coverage check passed: {needed.size} valid times x "
        f"{len(variables)} variables."
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


NAN_POLICIES = ("propagate", "zero_fill", "mask")
"""``scoring.nan_policy`` values the online path accepts; see :class:`OnlineScorer`."""


def _valid_mask(
    finite: torch.Tensor, nan_policy: str, comm: GroupComm | None = None
) -> torch.Tensor:
    """Gridpoints entering a step's sums under ``nan_policy``, agreed over the group.

    Each rank sees only its own members, so the ranks of an ensemble group
    combine their masks: a point one member leaves non-finite drops out on
    every rank, and the root's weight normalizer matches the reduced sums.

    Parameters
    ----------
    finite : torch.Tensor
        Boolean ``[variable, <spatial...>]``, ``True`` where this rank's
        members' forecasts and the verification are finite.
    nan_policy : str
        One of :data:`NAN_POLICIES`.
    comm : GroupComm, optional
        Group collectives; ``None`` for a single-rank group, by default None.

    Returns
    -------
    torch.Tensor
        Boolean mask of the same shape.  ``mask`` keeps the points finite
        on every rank; ``propagate`` drops a variable's whole field once any
        point of any member is not finite, so its statistics come out NaN
        as the offline metrics' do; ``zero_fill`` keeps every point.
    """
    if nan_policy == "zero_fill":
        return torch.ones_like(finite)
    if nan_policy == "mask":
        return finite if comm is None else comm.all_and(finite)
    whole = finite.reshape(finite.shape[0], -1).all(dim=1)
    if comm is not None:
        whole = comm.all_and(whole)
    shape = (finite.shape[0],) + (1,) * (finite.ndim - 1)
    return whole.reshape(shape).expand_as(finite).clone()


@dataclass
class StepContext:
    """Everything one lead step exposes to the configured statistics.

    :class:`OnlineScorer` materializes the group products (``s1``, ``s2``,
    ``below``) eagerly for the union of every statistic's
    :meth:`OnlineStatistic.requires`, in a fixed order, before any ``update``
    runs.  They are only valid on the group root.

    A step may carry a subset of the scored variables (each stream of a
    runner carries its own): ``variables`` names them and ``var_index`` gives
    their positions on the store's variable axis, so accumulators scatter
    the step's values into full-width buffers.

    Attributes
    ----------
    lead_index : int
        Index of this lead step in the store's ``lead_time`` axis.
    valid_time : np.datetime64
        ``IC time + lead time``.
    variables : tuple[str, ...]
        Variables of this step, in field order.
    var_index : list[int]
        Positions of ``variables`` on the store's variable axis.
    y : torch.Tensor
        Verification field, ``[variable, <spatial...>]``.
    clim : torch.Tensor | None
        Climatology on the same grid, or ``None``.
    f_local : torch.Tensor
        This rank's member block, ``[member, variable, <spatial...>]``.
    d_local : torch.Tensor
        ``f_local - y``.
    weights : torch.Tensor
        Spatial weights, broadcastable over the spatial dims, or
        ``[region, <spatial...>]`` with regions configured.
    valid : torch.Tensor
        Boolean ``[variable, <spatial...>]`` mask of the gridpoints entering
        this step's sums under ``scoring.nan_policy``, agreed over the group
        (see :func:`_valid_mask`).
    n_spatial : int
        Number of trailing spatial dimensions.
    ensemble_size : int
        Total ensemble size ``M``.
    member_ids : tuple[int, ...]
        Global member indices this rank carries.
    comm : GroupComm
        Group collectives.
    w_flat : torch.Tensor | None
        Regional weights flattened to ``[n_points, region]``, or ``None``.
    s1, s2 : torch.Tensor | None
        Root-only residual sums over the whole ensemble.
    below : torch.Tensor | None
        Root-only ``sum_i 1[f_i < y]``, the rank of ``y`` per gridpoint.
    """

    lead_index: int
    valid_time: np.datetime64
    variables: tuple[str, ...]
    var_index: list[int]
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

    @property
    def n_variables(self) -> int:
        """Variables in this step."""
        return len(self.var_index)

    def wsum(self, t: torch.Tensor) -> torch.Tensor:
        """Weighted sum of ``t`` over its trailing spatial dimensions.

        Gridpoints :attr:`valid` marks invalid for their variable are
        excluded.  With regions configured the result carries a trailing
        ``region`` axis.
        """
        td = torch.where(self.valid, t.double(), 0.0)
        if self.w_flat is not None:
            lead = td.shape[: td.ndim - self.n_spatial]
            return td.reshape(*lead, -1) @ self.w_flat
        axes = tuple(range(t.ndim - self.n_spatial, t.ndim))
        return (td * self.weights).sum(dim=axes)

    def wsum_valid(self) -> torch.Tensor:
        """Per-variable weighted sum of :attr:`valid`, the weighted-mean normalizer."""
        vd = self.valid.double()
        if self.w_flat is not None:
            lead = vd.shape[: vd.ndim - self.n_spatial]
            return vd.reshape(*lead, -1) @ self.w_flat
        axes = tuple(range(self.valid.ndim - self.n_spatial, self.valid.ndim))
        return (vd * self.weights).sum(dim=axes)


class OnlineStatistic(Protocol):
    """Accumulator emitting mergeable statistics for one metric family.

    Implementations declare which group-reduced products they need, so the
    scorer only forms the expensive ones when something asks.  ``update``
    may issue its own small group collectives; every rank builds the same
    statistic list, so their order is well defined.
    """

    name: str

    def requires(self) -> set[str]:
        """Group products needed, from :data:`_PRODUCT_ORDER`."""
        ...

    def fields(self) -> dict[str, str]:
        """Map each emitted field name to its layout."""
        ...

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        """Create per-IC buffers for a fresh initial condition."""
        ...

    def update(self, ctx: StepContext) -> None:
        """Accumulate this lead step's contribution."""
        ...

    def state(self) -> dict[str, torch.Tensor]:
        """Return the accumulated buffers, keyed by field name."""
        ...

    # Optional hooks, consulted via ``getattr``:
    #   flush(ctx)  -> complete deferred work at the end of an IC (collective).
    #   variables() -> narrow which variables this statistic emits fields for.


def _empty(*shape: int | torch.device) -> torch.Tensor:
    """NaN-filled float64 accumulator; the last argument is the device.

    Buffers start as NaN so that a lead step never yielded is visibly absent
    in ``stats.zarr`` rather than reading as a zero score.
    """
    *dims, device = shape
    return torch.full(
        tuple(int(d) for d in dims),  # type: ignore[arg-type]
        float("nan"),
        dtype=torch.float64,
        device=device,  # type: ignore[arg-type]
    )


def _region_first(value: torch.Tensor, n_regions: int) -> torch.Tensor:
    """Move :meth:`StepContext.wsum`'s trailing region axis to the front."""
    return value.movedim(-1, 0) if n_regions else value


def _put(
    buf: torch.Tensor, lead_index: int, var_index: Sequence[int], value: torch.Tensor
) -> None:
    """Write ``value`` at ``buf[..., lead_index, var_index]`` (trailing axes ``lead, variable``)."""
    buf[..., lead_index, :][..., list(var_index)] = value


class WeightSum:
    """The weighted-mean normalizer ``W = sum_s w`` per ``(IC, lead, variable)``.

    Stored rather than assumed constant, so a masked accumulation gets
    ``W = 0`` where every gridpoint is invalid and :func:`finalize_stats`
    divides ``0 / 0 = NaN`` instead of producing a wrong finite value.
    """

    name = "weight_sum"

    def requires(self) -> set[str]:
        """Group products needed, from :data:`_PRODUCT_ORDER`."""
        return set()

    def fields(self) -> dict[str, str]:
        """Map each emitted field name to its layout."""
        return {"w_sum": _LAYOUT_SCALAR}

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        """Create per-IC buffers for a fresh initial condition."""
        self._nr = n_regions
        self._w = _empty(
            *((n_regions,) if n_regions else ()), n_leads, n_variables, device
        )

    def update(self, ctx: StepContext) -> None:
        """Add this lead step's weight normalizer per variable."""
        if not ctx.comm.is_root:
            return
        _put(
            self._w,
            ctx.lead_index,
            ctx.var_index,
            _region_first(ctx.wsum_valid(), self._nr),
        )

    def state(self) -> dict[str, torch.Tensor]:
        """Return the accumulated buffers, keyed by field name."""
        return {"w_sum": self._w}


class _MemberSum:
    """Per-member weighted sum of a function of the residual.

    Entirely local (each rank owns whole members), so only a small
    ``[M, variable]`` block crosses the wire, zero-padded outside this
    rank's members.
    """

    name = "member_sum"
    field_name = "sum_member"

    def requires(self) -> set[str]:
        return set()

    def fields(self) -> dict[str, str]:
        return {self.field_name: _LAYOUT_MEMBER}

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        self._nr = n_regions
        self._buf = _empty(
            *((n_regions,) if n_regions else ()),
            ensemble_size,
            n_leads,
            n_variables,
            device,
        )

    def _term(self, ctx: StepContext) -> torch.Tensor:
        raise NotImplementedError

    def update(self, ctx: StepContext) -> None:
        block = torch.zeros(
            (ctx.ensemble_size, ctx.n_variables) + ((self._nr,) if self._nr else ()),
            dtype=torch.float64,
            device=self._buf.device,
        )
        block[list(ctx.member_ids)] = ctx.wsum(self._term(ctx))
        ctx.comm.reduce(block)
        if ctx.comm.is_root:
            _put(
                self._buf, ctx.lead_index, ctx.var_index, _region_first(block, self._nr)
            )

    def state(self) -> dict[str, torch.Tensor]:
        return {self.field_name: self._buf}


class MemberSquaredError(_MemberSum):
    """Per-member weighted sum of squared error; finalizes to ``mse``."""

    name = "member_mse"
    field_name = "sse_member"

    def _term(self, ctx: StepContext) -> torch.Tensor:
        return ctx.d_local**2


class MemberAbsError(_MemberSum):
    """Per-member weighted sum of absolute error; finalizes to ``mae``."""

    name = "member_mae"
    field_name = "sae_member"

    def _term(self, ctx: StepContext) -> torch.Tensor:
        return ctx.d_local.abs()


class LogSpectralDistance:
    """Per-member radially averaged 2D log spectral distance against verification.

    Delegates the computation to ``earth2studio.statistics.log_spectral_distance``.
    Unlike every other field in the store this is a final per-(IC, member,
    lead) value rather than a mergeable sum.  Spectra are global, so with
    regions configured the values land only in whole-grid regions.

    Parameters
    ----------
    whole_grid_regions : Sequence[int] | None
        Region indices with a ``null`` box, or ``None`` for a region-free run.
    wavenumber_cutoff : int | None
        Forwarded to the earth2studio metric.
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
        """Group products needed, from :data:`_PRODUCT_ORDER`."""
        return set()

    def fields(self) -> dict[str, str]:
        """Map each emitted field name to its layout."""
        return {"lsd": _LAYOUT_MEMBER}

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        """Create per-IC buffers for a fresh initial condition."""
        self._nr = n_regions
        self._lsd = _empty(
            *((n_regions,) if n_regions else ()),
            ensemble_size,
            n_leads,
            n_variables,
            device,
        )

    def update(self, ctx: StepContext) -> None:
        """Compute each member's log spectral distance at this lead."""
        if ctx.n_spatial != 2:
            raise ValueError(
                "scoring.online.lsd needs a 2D spatial grid; got "
                f"{ctx.n_spatial} spatial dimensions."
            )
        y_coords: CoordSystem = OrderedDict(
            {
                "variable": np.array(ctx.variables),
                "ilat": np.arange(ctx.f_local.shape[-2]),
                "ilon": np.arange(ctx.f_local.shape[-1]),
            }
        )
        x_coords: CoordSystem = OrderedDict(
            {"ensemble": np.array(ctx.member_ids), **y_coords}
        )
        values = self._metric(
            from_torch(ctx.f_local, x_coords), from_torch(ctx.y, y_coords)
        ).e2s.to_torch()[0]
        values = values.to(device=self._lsd.device, dtype=torch.float64)
        block = torch.zeros(
            (ctx.ensemble_size, ctx.n_variables),
            dtype=torch.float64,
            device=self._lsd.device,
        )
        block[list(ctx.member_ids), :] = values
        ctx.comm.reduce(block)
        if not ctx.comm.is_root:
            return
        if self._nr:
            for r in self._whole_grid or []:
                self._lsd[r, :, ctx.lead_index, :][:, list(ctx.var_index)] = block
        else:
            self._lsd[:, ctx.lead_index, :][:, list(ctx.var_index)] = block

    def state(self) -> dict[str, torch.Tensor]:
        """Return the accumulated buffers, keyed by field name."""
        return {"lsd": self._lsd}


class EnsembleMeanSquaredError:
    """Weighted sum of squared error of the ensemble mean.

    ``(fbar - y) = (1/M) sum_i (f_i - y)`` follows from the reduced residual
    sum, so it needs no communication beyond ``ens_moments``.
    """

    name = "ensemble_mean_mse"

    def requires(self) -> set[str]:
        """Group products needed, from :data:`_PRODUCT_ORDER`."""
        return {"ens_moments"}

    def fields(self) -> dict[str, str]:
        """Map each emitted field name to its layout."""
        return {"sse_ensmean": _LAYOUT_SCALAR}

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        """Create per-IC buffers for a fresh initial condition."""
        self._nr = n_regions
        self._sse = _empty(
            *((n_regions,) if n_regions else ()), n_leads, n_variables, device
        )

    def update(self, ctx: StepContext) -> None:
        """Add the squared ensemble-mean error of this lead step."""
        if not ctx.comm.is_root:
            return
        if ctx.s1 is None:
            raise RuntimeError("StepContext.s1 is unset; 'ens_moments' was not formed.")
        mean_dev = ctx.s1 / ctx.ensemble_size
        _put(
            self._sse,
            ctx.lead_index,
            ctx.var_index,
            _region_first(ctx.wsum(mean_dev**2), self._nr),
        )

    def state(self) -> dict[str, torch.Tensor]:
        """Return the accumulated buffers, keyed by field name."""
        return {"sse_ensmean": self._sse}


class EnsembleSpread:
    """Weighted spatial sum of the unbiased (Bessel-corrected) ensemble variance.

    Computed from residual moments: variance is shift-invariant, so
    ``var_i(f_i) = var_i(f_i - y)`` exactly.
    """

    name = "ensemble_variance"

    def requires(self) -> set[str]:
        """Group products needed, from :data:`_PRODUCT_ORDER`."""
        return {"ens_moments"}

    def fields(self) -> dict[str, str]:
        """Map each emitted field name to its layout."""
        return {"var_ens": _LAYOUT_SCALAR}

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        """Create per-IC buffers for a fresh initial condition."""
        self._nr = n_regions
        self._var = _empty(
            *((n_regions,) if n_regions else ()), n_leads, n_variables, device
        )

    def update(self, ctx: StepContext) -> None:
        """Add the weighted ensemble variance of this lead step."""
        if not ctx.comm.is_root:
            return
        if ctx.s1 is None or ctx.s2 is None:
            raise RuntimeError(
                "StepContext.s1/s2 are unset; 'ens_moments' was not formed."
            )
        m = ctx.ensemble_size
        var = (ctx.s2 - ctx.s1**2 / m) / (m - 1)
        _put(
            self._var,
            ctx.lead_index,
            ctx.var_index,
            _region_first(ctx.wsum(var), self._nr),
        )

    def state(self) -> dict[str, torch.Tensor]:
        """Return the accumulated buffers, keyed by field name."""
        return {"var_ens": self._var}


class RankHistogram:
    """Area-weighted histogram of the verification's rank in the ensemble.

    The rank of ``y`` at a gridpoint is ``#{i : f_i < y}``, giving ``M + 1``
    bins; ties count into the lower bin.  The same spatial weights as every
    other statistic weight the counts.
    """

    name = "rank_histogram"

    def requires(self) -> set[str]:
        """Group products needed, from :data:`_PRODUCT_ORDER`."""
        return {"rank_counts"}

    def fields(self) -> dict[str, str]:
        """Map each emitted field name to its layout."""
        return {"rank_counts": _LAYOUT_RANK}

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        """Create per-IC buffers for a fresh initial condition."""
        self._nr = n_regions
        self._counts = _empty(
            *((n_regions,) if n_regions else ()),
            ensemble_size + 1,
            n_leads,
            n_variables,
            device,
        )

    def update(self, ctx: StepContext) -> None:
        """Count where the verification ranks among the members at this lead."""
        if not ctx.comm.is_root:
            return
        if ctx.below is None:
            raise RuntimeError(
                "StepContext.below is unset; 'rank_counts' was not formed."
            )
        n_bins = self._counts.shape[-3]
        n_variables = ctx.n_variables
        device = self._counts.device
        # A masked gridpoint contributes to no bin: `below` is meaningless
        # there (NaN < y is always False), so it gets zero weight.
        ranks = ctx.below.reshape(n_variables, -1).long()
        valid_flat = ctx.valid.reshape(n_variables, -1)
        if self._nr:
            region_weights = ctx.weights.reshape(self._nr, 1, -1)
        else:
            region_weights = (
                ctx.weights.expand(ctx.y.shape[1:]).reshape(1, 1, -1).contiguous()
            )
        columns = list(ctx.var_index)
        for r in range(region_weights.shape[0]):
            w_flat = torch.where(
                valid_flat, region_weights[r].expand(n_variables, -1), 0.0
            )
            hist = torch.zeros(
                (n_variables, n_bins), dtype=torch.float64, device=device
            )
            hist.scatter_add_(1, ranks, w_flat)
            if self._nr:
                self._counts[r, :, ctx.lead_index, :][:, columns] = hist.transpose(0, 1)
            else:
                self._counts[:, ctx.lead_index, :][:, columns] = hist.transpose(0, 1)

    def state(self) -> dict[str, torch.Tensor]:
        """Return the accumulated buffers, keyed by field name."""
        return {"rank_counts": self._counts}


class AnomalyMoments:
    """Weighted cross-moments of the ensemble mean against verification.

    Five sums per ``(IC, lead, variable)`` reconstruct bias, Pearson
    correlation and, with a climatology, ACC.  Field names encode whether
    the step subtracted a climatology (``a``/``b`` for anomalies, ``f``/``y``
    for raw fields), so :func:`finalize_stats` can tell whether ACC is derivable.
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
        """Group products needed, from :data:`_PRODUCT_ORDER`."""
        return {"ens_moments"}

    def fields(self) -> dict[str, str]:
        """Map each emitted field name to its layout."""
        return {name: _LAYOUT_SCALAR for name in self._names}

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        """Create per-IC buffers for a fresh initial condition."""
        self._nr = n_regions
        self._buf = {
            name: _empty(
                *((n_regions,) if n_regions else ()), n_leads, n_variables, device
            )
            for name in self._names
        }

    def update(self, ctx: StepContext) -> None:
        """Add the cross-moments of this lead step."""
        if not ctx.comm.is_root:
            return
        if ctx.s1 is None:
            raise RuntimeError("StepContext.s1 is unset; 'ens_moments' was not formed.")
        fbar = ctx.y.double() + ctx.s1 / ctx.ensemble_size
        obs = ctx.y.double()
        if ctx.clim is not None:
            clim = ctx.clim.double()
            fbar = fbar - clim
            obs = obs - clim
        n = self._names
        for name, term in (
            (n[0], fbar),
            (n[1], obs),
            (n[2], fbar**2),
            (n[3], obs**2),
            (n[4], fbar * obs),
        ):
            _put(
                self._buf[name],
                ctx.lead_index,
                ctx.var_index,
                _region_first(ctx.wsum(term), self._nr),
            )

    def state(self) -> dict[str, torch.Tensor]:
        """Return the accumulated buffers, keyed by field name."""
        return dict(self._buf)


# ---------------------------------------------------------------------------
# Pairwise member exchange (fair CRPS)
# ---------------------------------------------------------------------------


def _abs_diff_weighted_sum(
    local: torch.Tensor,
    other: torch.Tensor,
    weights: torch.Tensor,
    n_spatial: int,
    member_tile: int = _PAIRWISE_MEMBER_TILE,
    w_flat: torch.Tensor | None = None,
) -> torch.Tensor:
    """``sum_{i in local} sum_{j in other} sum_s w_s |f_i(s) - f_j(s)|``.

    Tiled over the ``other`` member axis so the float64 temporary stays
    bounded regardless of ensemble size.  The subtraction upcasts straight
    from the wire dtype, so a bf16 exchange never does bf16 arithmetic.

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
    member_tile : int, optional
        Members of ``other`` per tile, by default :data:`_PAIRWISE_MEMBER_TILE`
    w_flat : torch.Tensor | None, optional
        Regional weights ``[n_points, region]``, by default None

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
    """An in-flight (or ready) member exchange for one step."""

    lead_index: int
    buf_index: list[int]
    local_chunks: list[torch.Tensor]
    gathered: list[list[torch.Tensor]]
    work: list[Any]


class PairwiseExchange:
    """Forms ``sum_{i<j} sum_s w_s |f_i(s) - f_j(s)|`` across an ensemble group.

    Members meet via an all-gather of the residual ``f_i - y`` (so the
    wire dtype's relative precision applies to error-sized values), using

    .. code-block:: text

        sum_{i<j} d(i,j) = (1/2) sum_g sum_{i in B_g} sum_{j in ALL} d(i,j)

    so every rank computes the same expression against every member and the
    group reduction halves the total.  With ``defer_pairwise_one_step`` the
    all-gather issued by one update completes during the next, overlapping
    the model forward.

    Parameters
    ----------
    settings : OnlineSettings
        Supplies the wire dtype, deferral and chunk size.
    """

    def __init__(self, settings: OnlineSettings) -> None:
        self._settings = settings
        self._pending: _PendingExchange | None = None

    @property
    def defers(self) -> bool:
        """Whether results arrive one update late."""
        return self._settings.defer_pairwise_one_step

    def _chunks(self, positions: Sequence[int]) -> list[list[int]]:
        size = self._settings.variable_chunk
        positions = list(positions)
        if size <= 0 or size >= len(positions):
            return [positions]
        return [positions[i : i + size] for i in range(0, len(positions), size)]

    def step(
        self, ctx: StepContext, local_index: Sequence[int], buf_index: Sequence[int]
    ) -> tuple[int, list[int], torch.Tensor] | None:
        """Advance the exchange by one update.

        Parameters
        ----------
        ctx : StepContext
            The lead step being accumulated.
        local_index : Sequence[int]
            Positions in ``ctx.d_local``'s variable axis that CRPS covers.
        buf_index : Sequence[int]
            Positions on the CRPS buffers' variable axis for ``local_index``.

        Returns
        -------
        tuple[int, list[int], torch.Tensor] | None
            ``(lead_index, buf_index, values)`` for whichever step just
            completed, or ``None``.  ``values`` is meaningful on the root only.
        """
        ready = self._complete(ctx) if self.defers else None
        if len(local_index):
            self._submit(ctx, local_index, buf_index)
        if self.defers:
            return ready
        return self._complete(ctx)

    def drain(self, ctx: StepContext) -> tuple[int, list[int], torch.Tensor] | None:
        """Complete any exchange still in flight at the end of an IC. Collective."""
        return self._complete(ctx)

    def _submit(
        self, ctx: StepContext, local_index: Sequence[int], buf_index: Sequence[int]
    ) -> None:
        if self._pending is not None:
            raise RuntimeError(
                "PairwiseExchange: a previous exchange is still pending; two "
                "steps were submitted without an intervening completion."
            )
        dtype = self._settings.pairwise_comm_dtype
        # Masked gridpoints compare as identical "absent" state across
        # members, whatever nan_policy left there.
        d_local = torch.where(ctx.valid, ctx.d_local, 0.0)
        local_chunks: list[torch.Tensor] = []
        gathered: list[list[torch.Tensor]] = []
        work: list[Any] = []
        for chunk in self._chunks(local_index):
            index = torch.tensor(chunk, device=d_local.device, dtype=torch.long)
            block = d_local.index_select(1, index).to(dtype).contiguous()
            buffers, handle = ctx.comm.all_gather(block, async_op=self.defers)
            local_chunks.append(block)
            gathered.append(buffers)
            work.append(handle)
        self._pending = _PendingExchange(
            lead_index=ctx.lead_index,
            buf_index=list(buf_index),
            local_chunks=local_chunks,
            gathered=gathered,
            work=work,
        )

    def _complete(self, ctx: StepContext) -> tuple[int, list[int], torch.Tensor] | None:
        pending, self._pending = self._pending, None
        if pending is None:
            return None
        n = len(pending.buf_index)
        out_shape: tuple[int, ...] = (n,) if ctx.w_flat is None else (n, ctx.n_regions)
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
        return pending.lead_index, pending.buf_index, total


class FairCRPS:
    """Fair (unbiased) CRPS, accumulated as its two mergeable terms.

    .. code-block:: text

        CRPS = (1/M) sum_i E_s|f_i - y| - (1/(M(M-1))) sum_{i<j} E_s|f_i - f_j|

    The first term is local per member; :class:`PairwiseExchange` forms the
    second.  The store keeps both as raw weighted sums so they stay
    mergeable; :func:`finalize_stats` normalizes.

    Parameters
    ----------
    settings : OnlineSettings
        Parsed online settings (``pairwise_variables`` narrows coverage).
    variables : Sequence[str]
        All scored variables, in store order.
    """

    name = "crps"

    def __init__(self, settings: OnlineSettings, variables: Sequence[str]) -> None:
        self._settings = settings
        all_variables = [str(v) for v in variables]
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
        self._position = {v: i for i, v in enumerate(self._variables)}
        self._exchange = PairwiseExchange(settings)

    def requires(self) -> set[str]:
        """Group products needed, from :data:`_PRODUCT_ORDER`."""
        return {"pairwise"}

    def variables(self) -> list[str]:
        """CRPS may cover a subset of the scored variables."""
        return list(self._variables)

    def fields(self) -> dict[str, str]:
        """Map each emitted field name to its layout."""
        return {"crps_t1": _LAYOUT_SCALAR, "crps_t2": _LAYOUT_SCALAR}

    def reset(
        self,
        n_leads: int,
        n_variables: int,
        ensemble_size: int,
        device: torch.device,
        n_regions: int = 0,
    ) -> None:
        """Create per-IC buffers for a fresh initial condition."""
        self._nr = n_regions
        dims = (n_regions,) if n_regions else ()
        self._t1 = _empty(*dims, n_leads, len(self._variables), device)
        self._t2 = _empty(*dims, n_leads, len(self._variables), device)

    def _positions(self, ctx: StepContext) -> tuple[list[int], list[int]]:
        """Positions of this step's CRPS variables in the field and the buffers."""
        local = [k for k, v in enumerate(ctx.variables) if v in self._position]
        buf = [self._position[ctx.variables[k]] for k in local]
        return local, buf

    def update(self, ctx: StepContext) -> None:
        """Add the member term and advance the pairwise exchange for this lead."""
        local_index, buf_index = self._positions(ctx)
        if local_index:
            index = torch.tensor(
                local_index, device=ctx.d_local.device, dtype=torch.long
            )
            d_local = (
                torch.where(ctx.valid, ctx.d_local, 0.0).index_select(1, index).abs()
            ).double()
            if ctx.w_flat is not None:
                flat = d_local.reshape(d_local.shape[0], d_local.shape[1], -1)
                t1 = (flat @ ctx.w_flat).sum(dim=0)
            else:
                axes = tuple(range(d_local.ndim - ctx.n_spatial, d_local.ndim))
                t1 = (d_local * ctx.weights).sum(dim=axes).sum(dim=0)
            ctx.comm.reduce(t1)
            if ctx.comm.is_root:
                _put(self._t1, ctx.lead_index, buf_index, _region_first(t1, self._nr))
        self._store_pairwise(ctx, self._exchange.step(ctx, local_index, buf_index))

    def flush(self, ctx: StepContext) -> None:
        """Complete the deferred exchange left over from the last update."""
        self._store_pairwise(ctx, self._exchange.drain(ctx))

    def _store_pairwise(
        self, ctx: StepContext, result: tuple[int, list[int], torch.Tensor] | None
    ) -> None:
        if result is None or not ctx.comm.is_root:
            return
        lead_index, buf_index, values = result
        _put(self._t2, lead_index, buf_index, _region_first(values, self._nr))

    def state(self) -> dict[str, torch.Tensor]:
        """Return the accumulated buffers, keyed by field name."""
        return {"crps_t1": self._t1, "crps_t2": self._t2}


def statistic_variables(stat: OnlineStatistic, variables: Sequence[str]) -> list[str]:
    """Variables ``stat`` emits fields for; every scored one unless it says otherwise."""
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
    """Assemble the statistics an online run accumulates, in a fixed order.

    The list omits ensemble-only statistics for ``M = 1``, so a deterministic
    online run produces the deterministic offline metric set.

    Parameters
    ----------
    ensemble_size : int
        Ensemble size ``M``.
    has_climatology : bool, optional
        Whether the run has a climatology source (enables ACC), by default False
    settings : OnlineSettings | None, optional
        Parsed online settings; ``None`` omits CRPS, MAE and LSD, by default None
    variables : Sequence[str] | None, optional
        Scored variables, required with CRPS, by default None

    Returns
    -------
    list[OnlineStatistic]
        Statistics in the order every rank uses.
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
) -> list[tuple[CoordSystem, list[str]]]:
    """Build the ``stats.zarr`` array groups for the configured statistics.

    Arrays with the same dimension structure share a group, so each group is one
    :meth:`~earth2studio.io.ZarrBackend.add_array` call.  Array names follow
    the ``{field}__{variable}`` convention of ``scores.zarr``.

    Parameters
    ----------
    statistics : Sequence[OnlineStatistic]
        Configured statistics.
    variables : Sequence[str]
        Scored variables.
    times : np.ndarray
        All initial times of the campaign.
    lead_times : np.ndarray
        All lead times of the store.
    ensemble_size : int
        Ensemble size ``M`` (sets the ``ensemble`` and ``rank_bin`` axes).
    region_names : Sequence[str] | None, optional
        Regional split names; every array gains a ``region`` axis of integer
        indices, by default None

    Returns
    -------
    list[tuple[CoordSystem, list[str]]]
        ``(coords, array_names)`` per layout in use.
    """
    n_regions = 0 if region_names is None else len(region_names)
    axes: dict[str, np.ndarray] = {
        "time": np.asarray(times).astype("datetime64[ns]"),
        "region": np.arange(n_regions),
        "ensemble": np.arange(ensemble_size),
        "rank_bin": np.arange(ensemble_size + 1),
        "lead_time": np.asarray(lead_times).astype("timedelta64[ns]"),
    }
    used_layouts: dict[str, list[str]] = {}
    for stat in statistics:
        stat_variables = statistic_variables(stat, variables)
        for field_name, layout in stat.fields().items():
            used_layouts.setdefault(layout, []).extend(
                f"{field_name}__{v}" for v in stat_variables
            )
    groups: list[tuple[CoordSystem, list[str]]] = []
    for layout in (_LAYOUT_SCALAR, _LAYOUT_MEMBER, _LAYOUT_RANK):
        names = used_layouts.get(layout)
        if not names:
            continue
        coords: CoordSystem = OrderedDict(
            (dim, axes[dim]) for dim in _layout_dims(layout, n_regions)
        )
        groups.append((coords, names))
    return groups


def add_stats_arrays(
    backend: ZarrBackend,
    array_groups: Sequence[tuple[CoordSystem, Sequence[str]]],
    region_names: Sequence[str] | None = None,
) -> None:
    """Create the ``stats.zarr`` arrays from float64 templates, skipping existing ones.

    Parameters
    ----------
    backend : ZarrBackend
        Open statistics store.
    array_groups : Sequence[tuple[CoordSystem, Sequence[str]]]
        Groups from :func:`stats_array_groups`.
    region_names : Sequence[str] | None, optional
        Regional split names, recorded in the store attributes so the
        integer ``region`` axis stays interpretable, by default None

    Raises
    ------
    ValueError
        If the store already holds a different region set, or an existing
        array or coordinate differs from the current layout (other initial
        times or lead times): the backend checks every array of the
        template and leaves a matching one alone, so a resume into a store
        laid out for another campaign fails here instead of at its first write.
    """
    if region_names is not None:
        existing = backend.root.attrs.get("regions")
        if existing is not None and list(existing) != list(region_names):
            raise ValueError(
                f"stats store already holds regions {list(existing)} but the "
                f"current configuration defines {list(region_names)}; restore "
                "the original scoring.regions or clear the run's output directory."
            )
        backend.root.attrs["regions"] = list(region_names)
    for coords, names in array_groups:
        template = coord_array(
            (*coords, "variable"),
            {**dict(coords), "variable": np.array(list(names))},
            dtype=np.float64,
        )
        backend.add_array(template)


def open_stats_store(
    path: str,
    array_groups: Sequence[tuple[CoordSystem, Sequence[str]]],
    region_names: Sequence[str] | None,
    resume: bool,
    overwrite: bool,
    recorded: int = 0,
) -> ZarrBackend:
    """Create or reopen a statistics store on every rank.  Collective.

    :func:`src.output.prepare_store` applies the rerun policy on every rank
    first, then rank 0 creates the store before the other ranks open it,
    through :func:`src.logs.rank0_first_checked`, so a failure on rank 0
    reaches every rank.  Each chunk holds one initial condition, on the
    synchronous :class:`~earth2studio.io.ZarrBackend`.

    Parameters
    ----------
    path : str
        Store path.
    array_groups : Sequence[tuple[CoordSystem, Sequence[str]]]
        Groups from :func:`stats_array_groups`.
    region_names : Sequence[str] | None
        Regional split names, or ``None``.
    resume : bool
        Whether to continue an existing store.
    overwrite : bool
        Whether to replace an existing store when not resuming.
    recorded : int, optional
        Items the progress store records as done, by default 0;
        :func:`src.output.prepare_store` refuses a resume into a missing
        store with records.

    Returns
    -------
    ZarrBackend
        The open store.

    Raises
    ------
    FileExistsError
        If the store exists without ``resume`` or ``overwrite``.
    FileNotFoundError
        With ``resume``, a missing store and a nonzero ``recorded``.
    ValueError
        If the store holds other regions or another layout
        (:func:`add_stats_arrays`).
    """
    rank, _ = rank_info()
    prepare_store(path, resume, overwrite, rank, recorded)

    def _open() -> ZarrBackend:
        backend = ZarrBackend(path, chunks={"time": 1})
        add_stats_arrays(backend, array_groups, region_names)
        return backend

    return rank0_first_checked(_open)


# ---------------------------------------------------------------------------
# The scorer
# ---------------------------------------------------------------------------


class OnlineScorer:
    """Per-reference :class:`~earth2studio.run.pipeline.Scorer` for one ensemble group.

    Instantiated on every rank of a group and driven by
    :class:`earth2studio.run.Pipeline`: :meth:`begin_item` per item,
    :meth:`update` per yielded stream field, :meth:`finish_item` once the
    rollout completes.  Only the group root writes.

    Parameters
    ----------
    settings : OnlineSettings
        Parsed ``scoring.online`` block.
    comm : GroupComm
        This rank's group collectives.
    verification : FieldCache
        Verification field access.
    climatology : FieldCache | None
        Climatology field access, or ``None``.
    variables : Sequence[str]
        Variables to score, in store order.
    lead_times : np.ndarray
        Lead-time axis of the store.
    spatial_dims : Sequence[str]
        Spatial dimension names of the scored grid, in order.
    weights : torch.Tensor
        Spatial weights from :func:`src.regions.build_spatial_weights`.
    store : ZarrBackend
        Open statistics store.
    device : torch.device
        Device for accumulation.
    nan_policy : str, optional
        What a non-finite gridpoint does to the step's sums, matching the
        offline path where both offer the policy: ``"propagate"`` turns
        every statistic of that variable at that lead NaN, ``"zero_fill"``
        counts it as zero in forecast and verification, and ``"mask"``
        (online only) drops it from the sums and from the weight normalizer,
        so sea-only fields score over the sea; by default ``"propagate"``
    valid_ranges : Mapping | None, optional
        Per-variable ``{min, max}`` clamps applied to forecasts, by default None
    streams : Mapping[str, Sequence[str]] | None, optional
        Output variables per stream; a variable scores from the stream that
        owns it, by default every stream scores every variable it carries
    """

    def __init__(
        self,
        settings: OnlineSettings,
        comm: GroupComm,
        verification: FieldCache,
        climatology: FieldCache | None,
        variables: Sequence[str],
        lead_times: np.ndarray,
        spatial_dims: Sequence[str],
        weights: torch.Tensor,
        store: ZarrBackend,
        device: torch.device,
        nan_policy: str = "propagate",
        valid_ranges: Mapping[str, Any] | None = None,
        streams: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        if nan_policy not in NAN_POLICIES:
            raise ValueError(
                f"Invalid scoring.nan_policy '{nan_policy}'; expected one of "
                f"{', '.join(NAN_POLICIES)}."
            )
        self._settings = settings
        # Output variables per stream (src.output.stream_variables): a variable
        # scores from the stream that owns it, so a diagnostic re-emitting a
        # forecast variable never overwrites the forecast's statistics.
        self._streams: dict[str, frozenset[str]] | None = (
            None
            if streams is None
            else {s: frozenset(str(v) for v in names) for s, names in streams.items()}
        )
        self._comm = comm
        self._verification = verification
        self._climatology = climatology
        self._variables = [str(v) for v in variables]
        self._var_pos = {v: i for i, v in enumerate(self._variables)}
        self._lead_times = np.asarray(lead_times).astype("timedelta64[ns]")
        self._lead_index = {
            int(lt.astype("int64")): i for i, lt in enumerate(self._lead_times)
        }
        self._spatial_dims = tuple(spatial_dims)
        self._device = torch.device(device)
        self._weights = weights.to(device=self._device, dtype=torch.float64)
        self._n_regions = 0 if settings.regions is None else len(settings.regions)
        self._w_flat = (
            self._weights.reshape(self._n_regions, -1).T.contiguous()
            if self._n_regions
            else None
        )
        self._store = store
        self._nan_policy = nan_policy
        self._valid_ranges = dict(valid_ranges or {})
        self._ensemble_size = comm.group.ensemble_size
        self._statistics = build_statistics(
            self._ensemble_size, climatology is not None, settings, self._variables
        )
        self._products = {p for stat in self._statistics for p in stat.requires()}
        unsupported = self._products - set(_PRODUCT_ORDER)
        if unsupported:
            raise ValueError(f"Unknown group products requested: {sorted(unsupported)}")
        self._item: WorkItem | None = None
        self._seen_leads: set[int] = set()

    @property
    def statistics(self) -> list[OnlineStatistic]:
        """The configured statistics, in collective order."""
        return list(self._statistics)

    # -- lifecycle -----------------------------------------------------------

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

    def update(self, item: WorkItem, stream: str, x: xr.DataArray) -> None:
        """Accumulate one filtered, regridded stream field.

        Parameters
        ----------
        item : WorkItem
            The item under way.
        stream : str
            Stream name.  With ``streams`` given, only that stream's output
            variables score, and the scorer skips every other stream.
        x : xr.DataArray
            Field with ``lead_time`` and ``variable`` dimensions, the scored
            grid's spatial dimensions, optional singleton ``time``/``batch``
            axes and an optional ``ensemble`` axis labelled by the item's
            members.  The scorer skips variables outside ``scoring.variables``.
        """
        if self._item is None:
            raise RuntimeError("OnlineScorer.update called before begin_item.")
        names = [str(v) for v in np.asarray(x.coords["variable"].values)]
        allowed: frozenset[str] | None = None
        if self._streams is not None:
            allowed = self._streams.get(stream)
            if not allowed:
                return
        mine = [
            (k, v)
            for k, v in enumerate(names)
            if v in self._var_pos and (allowed is None or v in allowed)
        ]
        if not mine:
            return
        if "lead_time" not in x.dims:
            raise ValueError(
                f"Online scoring needs a 'lead_time' dimension on stream {stream!r}; "
                f"got {tuple(x.dims)}."
            )
        tensor, _ = x.e2s.to_torch()
        dims = list(map(str, x.dims))
        for dim in ("batch", "time"):
            if dim in dims:
                axis = dims.index(dim)
                if tensor.shape[axis] != 1:
                    raise ValueError(
                        f"Online scoring expects a singleton '{dim}' axis on stream "
                        f"{stream!r}; got size {tensor.shape[axis]}."
                    )
                tensor = tensor.squeeze(axis)
                dims.pop(axis)
        has_members = "ensemble" in dims
        if has_members:
            members = tuple(int(m) for m in np.asarray(x.coords["ensemble"].values))
        elif len(item.member_ids) > 1:
            # Scored as every member, one field would pass for a collapsed
            # ensemble; the pipeline's own check runs only for written streams.
            raise ValueError(
                f"Stream {stream!r} carries no 'ensemble' axis for an item with "
                f"members {tuple(item.member_ids)}; a runner batching members must "
                "label them on a leading ensemble dimension."
            )
        else:
            members = tuple(int(m) for m in item.member_ids) or (0,)
        expected = self._comm.group.member_ids
        if members != expected:
            raise ValueError(
                f"Stream {stream!r} carries members {members}, but this rank owns "
                f"{expected}; the items must be ordered with order_for_groups so "
                "rank b*G+g carries member group g."
            )
        target = [
            "lead_time",
            *(["ensemble"] if has_members else []),
            "variable",
            *self._spatial_dims,
        ]
        if sorted(target) != sorted(dims):
            raise ValueError(
                f"Stream {stream!r} dims {dims} do not match the online layout {target}."
            )
        tensor = tensor.permute(*[dims.index(d) for d in target])
        if not has_members:
            tensor = tensor.unsqueeze(1)
        tensor = tensor.to(self._device)
        local = torch.tensor(
            [k for k, _ in mine], device=tensor.device, dtype=torch.long
        )
        tensor = tensor.index_select(2, local).contiguous()
        variables = tuple(v for _, v in mine)
        var_index = [self._var_pos[v] for v in variables]
        leads = np.asarray(x.coords["lead_time"].values).astype("timedelta64[ns]")
        for i, lead in enumerate(leads):
            self._step(tensor[i], lead, variables, var_index)

    def finish_item(self, item: WorkItem) -> None:
        """Complete deferred work and write the IC's slab (group root).

        Collective over the group: the flush drains the deferred CRPS
        exchange on every rank before the root writes.
        """
        missing = set(range(len(self._lead_times))) - self._seen_leads
        if missing:
            logger.warning(
                f"IC {item.time}: {len(missing)}/{len(self._lead_times)} lead times "
                "were never yielded; their statistics stay NaN."
            )
        ctx = self._flush_context()
        for stat in self._statistics:
            flush = getattr(stat, "flush", None)
            if flush is not None:
                flush(ctx)
        if self._comm.is_root:
            self._write_slab(item.time)
            self._store.flush()
        self._item = None

    # -- internals -----------------------------------------------------------

    def _flush_context(self) -> StepContext:
        """A field-free :class:`StepContext` for end-of-IC flushes."""
        empty = torch.empty((0, 0), dtype=torch.float32, device=self._device)
        return StepContext(
            lead_index=-1,
            valid_time=np.datetime64("NaT"),
            variables=(),
            var_index=[],
            y=empty,
            clim=None,
            f_local=empty,
            d_local=empty,
            weights=self._weights,
            valid=torch.ones((0, 0), dtype=torch.bool, device=self._device),
            n_spatial=len(self._spatial_dims),
            ensemble_size=self._ensemble_size,
            member_ids=self._comm.group.member_ids,
            comm=self._comm,
            w_flat=self._w_flat,
        )

    def _step(
        self,
        f_local: torch.Tensor,
        lead: np.timedelta64,
        variables: tuple[str, ...],
        var_index: list[int],
    ) -> None:
        """Accumulate one (IC, lead step, variable subset) across the group."""
        assert self._item is not None  # noqa: S101 - checked by update()
        key = int(lead.astype("int64"))
        if self._settings.validate_coords and not self._comm.all_agree(
            key, self._device
        ):
            raise RuntimeError(
                f"Ensemble group {self._comm.group.block} desynchronized at IC "
                f"{self._item.time}: this rank is at lead {lead} while another is "
                "elsewhere; every member must yield an identical step sequence."
            )
        if key not in self._lead_index:
            raise ValueError(
                f"Yielded lead time {lead} is not on the store's lead_time axis "
                f"({self._lead_times[0]} .. {self._lead_times[-1]})."
            )
        lead_index = self._lead_index[key]
        self._seen_leads.add(lead_index)

        valid_time = np.datetime64(self._item.time, "ns") + lead
        index = torch.tensor(var_index, device=self._device, dtype=torch.long)
        y = self._verification.get(valid_time).index_select(0, index)
        clim = (
            self._climatology.get(valid_time).index_select(0, index)
            if self._climatology is not None
            else None
        )
        if f_local.shape[1:] != y.shape:
            raise ValueError(
                f"Forecast field {tuple(f_local.shape[1:])} and verification field "
                f"{tuple(y.shape)} disagree at {valid_time}; verification must be on "
                "the same grid as the (regridded) output."
            )
        # Non-finite in any member's forecast or in the verification (the
        # cache zero-filled the latter under zero_fill); captured before the
        # policy replaces anything.
        finite = torch.isfinite(f_local).all(dim=0) & torch.isfinite(y)
        valid = _valid_mask(finite, self._nan_policy, self._comm)
        if self._nan_policy == "zero_fill":
            f_local = torch.nan_to_num(f_local, nan=0.0)
        if self._valid_ranges:
            f_local = _apply_valid_ranges(
                f_local.clone(), variables, self._valid_ranges, dim=1
            )

        ctx = StepContext(
            lead_index=lead_index,
            valid_time=valid_time,
            variables=variables,
            var_index=list(var_index),
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
        """Form the shared group products, in a fixed collective order."""
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
        """Append this IC's statistics to the store (root only).

        Fields sharing a layout and a variable set go out in one write,
        concatenated along the trailing variable axis.
        """
        batches: dict[tuple[str, tuple[str, ...]], list[tuple[str, torch.Tensor]]] = {}
        for stat in self._statistics:
            layouts = stat.fields()
            stat_variables = tuple(statistic_variables(stat, self._variables))
            for name, tensor in stat.state().items():
                batches.setdefault((layouts[name], stat_variables), []).append(
                    (name, tensor)
                )
        for (layout, stat_variables), fields in batches.items():
            names: list[str] = []
            parts: list[torch.Tensor] = []
            for field_name, tensor in fields:
                names.extend(f"{field_name}__{v}" for v in stat_variables)
                parts.append(tensor)
            data = torch.cat(parts, dim=-1).unsqueeze(0).cpu().numpy()
            dims = (*_layout_dims(layout, self._n_regions), "variable")
            coords: dict[str, np.ndarray] = {
                "time": np.array([time], dtype="datetime64[ns]"),
                "variable": np.array(names),
            }
            for dim in dims[1:-1]:
                coords[dim] = self._axis_values(dim)
            self._store.write(xr.DataArray(data, dims=dims, coords=coords))


class MultiRefScorer:
    """Fan one rollout's fields out to one :class:`OnlineScorer` per reference.

    Each named reference gets its own scorer, verification cache, buffers
    and store; the pipeline sees one scorer.

    Parameters
    ----------
    scorers : Mapping[str | None, OnlineScorer]
        Reference name -> scorer, in the fixed order every rank agrees on.
    """

    def __init__(self, scorers: Mapping[str | None, OnlineScorer]) -> None:
        if not scorers:
            raise ValueError("MultiRefScorer needs at least one child scorer.")
        self._scorers = dict(scorers)

    @property
    def scorers(self) -> dict[str | None, OnlineScorer]:
        """The child scorers by reference name."""
        return dict(self._scorers)

    def begin_item(self, item: WorkItem) -> None:
        """Start a new initial condition in every reference's scorer."""
        for scorer in self._scorers.values():
            scorer.begin_item(item)

    def update(self, item: WorkItem, stream: str, x: xr.DataArray) -> None:
        """Hand the same field to every reference's scorer."""
        for scorer in self._scorers.values():
            scorer.update(item, stream, x)

    def finish_item(self, item: WorkItem) -> None:
        """Write every reference's slab."""
        for scorer in self._scorers.values():
            scorer.finish_item(item)


# ---------------------------------------------------------------------------
# Finalize: stats.zarr -> scores.zarr
# ---------------------------------------------------------------------------


def _finalize_variable(
    ds: xr.Dataset, var: str, ensemble_size: int
) -> dict[str, xr.DataArray]:
    """Derive one variable's score arrays from its mergeable statistics."""
    out: dict[str, xr.DataArray] = {}
    w = ds[f"w_sum__{var}"]

    member_key = f"sse_member__{var}"
    if member_key in ds:
        out[f"mse__{var}"] = ds[member_key] / w
        out[f"ensemble_mean_mse__{var}"] = ds[f"sse_ensmean__{var}"] / w
    else:
        out[f"mse__{var}"] = ds[f"sse_ensmean__{var}"] / w
    if f"var_ens__{var}" in ds:
        out[f"ensemble_variance__{var}"] = ds[f"var_ens__{var}"] / w
    if f"sae_member__{var}" in ds:
        out[f"mae__{var}"] = ds[f"sae_member__{var}"] / w
    if f"lsd__{var}" in ds:
        out[f"lsd__{var}"] = ds[f"lsd__{var}"]

    # Fair CRPS: the (M-1) denominator makes the estimator unbiased for the
    # infinite-ensemble CRPS.
    if f"crps_t1__{var}" in ds and ensemble_size > 1:
        m = ensemble_size
        out[f"crps__{var}"] = ds[f"crps_t1__{var}"] / (m * w) - ds[
            f"crps_t2__{var}"
        ] / (m * (m - 1) * w)

    if f"rank_counts__{var}" in ds:
        counts = ds[f"rank_counts__{var}"]
        total = counts.sum(dim="rank_bin")
        uniform = 1.0 / (ensemble_size + 1)
        # skipna=False keeps a fully masked (lead, variable) NaN rather than
        # reading as perfectly calibrated.
        out[f"rank_reliability__{var}"] = np.abs(counts / total - uniform).sum(
            dim="rank_bin", skipna=False
        )

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
    scores land in ``scoring.output.store_name``.
    """
    settings = parse_online_settings(cfg)
    if settings.scores_store is not None:
        return settings.scores_store
    return str(cfg.scoring.output.get("store_name", "scores.zarr"))


def finalize_stats(
    cfg: DictConfig,
    stats_store: str | None = None,
    scores_store: str | None = None,
) -> str:
    """Derive ``scores.zarr`` from the accumulated ``stats.zarr``.

    Single-process, cheap and idempotent: it runs at the end of an online
    inference job and can be re-run at any time, including after merging
    statistics from more than one job.  The emitted arrays use the offline
    scorer's ``{metric}__{variable}`` naming:

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
    ==========================  =========================================

    With ``scoring.regions`` configured every array carries a labelled
    ``region`` axis, and the store's ``events`` attribute carries the
    ``scoring.events`` definitions.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra config.
    stats_store : str | None, optional
        Statistics store filename, by default ``scoring.online.stats_store``
    scores_store : str | None, optional
        Scores store filename, by default :func:`finalize_scores_store_name`

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
    stats_path = os.path.join(cfg.output.path, stats_store or settings.stats_store)
    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Statistics store not found at '{stats_path}'.\n"
            "Run inference (main.py) with scoring.mode=online first."
        )
    scores_path = os.path.join(
        cfg.output.path, scores_store or finalize_scores_store_name(cfg)
    )

    ds = xr.open_zarr(stats_path, consolidated=False)
    ensemble_size = int(ds.sizes.get("ensemble", 1))
    region_names = ds.attrs.get("regions")
    if region_names is not None and "region" in ds.dims:
        ds = ds.assign_coords(region=np.array([str(r) for r in region_names]))
    variables = sorted(
        {str(name).split("__", 1)[1] for name in ds.data_vars if "__" in str(name)}
    )

    arrays: dict[str, xr.DataArray] = {}
    for var in variables:
        arrays.update(_finalize_variable(ds, var, ensemble_size))

    scores = xr.Dataset(arrays).compute()
    for name in scores.data_vars:
        scores[name].attrs.clear()
    if region_names is not None:
        scores.attrs["regions"] = list(region_names)
    events = events_to_attrs(parse_events(cfg.scoring.get("events", None)))
    if events is not None:
        scores.attrs["events"] = events
    # Rebuild from scratch: the derived metric set depends on what is in
    # stats.zarr, so an in-place overwrite could leave stale arrays.
    if os.path.exists(scores_path):
        shutil.rmtree(scores_path)
    scores.to_zarr(scores_path, mode="w", consolidated=True)
    logger.success(
        f"Finalized {len(arrays)} score arrays for {len(variables)} variables -> {scores_path}"
    )
    return scores_path


# ---------------------------------------------------------------------------
# Multi-reference verification sources
# ---------------------------------------------------------------------------

# Reference names become filenames: a leading alphanumeric followed by word
# characters, no separators, no dots, no traversal.
_REF_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


def verification_source_names(cfg: DictConfig) -> list[str | None]:
    """Reference names in config order; ``[None]`` when no references are named.

    Raises
    ------
    ValueError
        If ``verification_sources`` is present but empty, or a name is not
        usable in a store filename.
    """
    scoring = cfg.get("scoring", None) or {}
    block = scoring.get("online", None) or {}
    named = block.get("verification_sources", None)
    if named is None:
        return [None]
    if not named:
        raise ValueError(
            "scoring.online.verification_sources is set but empty; remove it to "
            "score a single (unnamed) reference, or add at least one named entry."
        )
    names: list[str | None] = []
    for name in (str(key) for key in named):
        if not _REF_NAME_RE.match(name):
            raise ValueError(
                f"Invalid verification source name '{name}'. Names are part of "
                "store filenames, so they must start with an alphanumeric and "
                "contain only letters, digits, '_' or '-'."
            )
        names.append(name)
    return names


def named_store_filename(base: str, ref_name: str | None) -> str:
    """Filename ``base`` takes for reference ``ref_name``'s own store.

    ``None`` maps to ``base`` unchanged; a name goes before the extension, so
    ``("stats.zarr", "era5")`` gives ``"stats__era5.zarr"``.
    """
    if ref_name is None:
        return base
    root, ext = os.path.splitext(base)
    return f"{root}__{ref_name}{ext or '.zarr'}"


def default_verification_source(cfg: DictConfig) -> Any:
    """The run's verification source: :func:`src.scoring.open_verification_source`."""
    return open_verification_source(cfg)


def resolve_verification_sources(cfg: DictConfig) -> dict[str | None, Any]:
    """Resolve one or more named verification (truth) sources, in config order.

    ``scoring.online.verification_sources`` maps reference names to source
    nodes; a ``null`` entry reuses :func:`default_verification_source` under
    that name.  Unset, the single default source takes the key ``None``.
    """
    names = verification_source_names(cfg)
    if names == [None]:
        return {None: default_verification_source(cfg)}
    named = cfg.scoring.online.verification_sources
    sources: dict[str | None, Any] = {}
    for name in names:
        spec = named[name]
        if spec is None:
            logger.info(
                f"Verification source '{name}': reusing the default resolution."
            )
            sources[name] = default_verification_source(cfg)
        else:
            logger.info(
                f"Verification source '{name}': instantiating the configured source."
            )
            sources[name] = instantiate_source(spec)
    return sources


def scores_store_names(cfg: DictConfig) -> list[str]:
    """Filenames :func:`finalize_online_stats` writes under ``output.path``.

    One per reference, in :func:`verification_source_names` order.
    """
    base = finalize_scores_store_name(cfg)
    return [named_store_filename(base, name) for name in verification_source_names(cfg)]


def finalize_online_stats(cfg: DictConfig) -> None:
    """Derive every reference's scores store from its statistics store."""
    settings = parse_online_settings(cfg)
    names = verification_source_names(cfg)
    for name, scores_store in zip(names, scores_store_names(cfg)):
        finalize_stats(
            cfg,
            stats_store=named_store_filename(settings.stats_store, name),
            scores_store=scores_store,
        )


# ---------------------------------------------------------------------------
# Wiring: the scored grid, item ordering and OnlineRun
# ---------------------------------------------------------------------------


def scored_grid(
    schemas: Mapping[str, CoordinateSystem],
    regridders: Mapping[str, Regridder],
    streams: Sequence[str],
) -> tuple[GridDefinition, CoordSystem, np.ndarray]:
    """The grid, spatial coordinates and lead-time axis the scorer works on.

    Parameters
    ----------
    schemas : Mapping[str, CoordinateSystem]
        Stream schemas from ``runner.output_coords``.
    regridders : Mapping[str, Regridder]
        Per-stream output regridders; a regridded stream scores on its
        regridder's ``target_grid``.
    streams : Sequence[str]
        Streams carrying scored variables.

    Returns
    -------
    tuple[GridDefinition, CoordSystem, np.ndarray]
        The common grid, its spatial coordinate arrays in dimension order,
        and the union of the streams' lead times (``timedelta64[ns]``).

    Raises
    ------
    ValueError
        If the scored streams sit on different grids.
    """
    grid: GridDefinition | None = None
    leads: list[np.ndarray] = []
    for stream in streams:
        schema = schemas[stream]
        regridder = regridders.get(stream)
        stream_grid = (
            regridder.target_grid if regridder is not None else infer_grid(schema)
        )
        if grid is None:
            grid = stream_grid
        elif not same_geometry(grid, stream_grid):
            raise ValueError(
                f"Scored streams sit on different grids (stream {stream!r} differs); "
                "set output.regrid so every scored stream lands on one grid."
            )
        leads.append(
            np.asarray(schema.coords["lead_time"].values).astype("timedelta64[ns]")
        )
    if grid is None:
        raise ValueError("No stream carries a scored variable.")
    coords = grid.coords()
    spatial: CoordSystem = OrderedDict()
    for dim, size in zip(grid.dims, grid.shape):
        if dim in coords and coords[dim].dims == (dim,):
            spatial[dim] = np.asarray(coords[dim].values)
        else:
            spatial[dim] = np.arange(size)
    return grid, spatial, np.unique(np.concatenate(leads))


def order_items(
    cfg: DictConfig, items: Sequence[WorkItem], world_size: int, group_size: int
) -> list[WorkItem]:
    """Order the campaign's items for lockstep ensemble groups.  Collective.

    Fresh runs order every item with :func:`src.campaign.order_for_groups`.
    A resumed run (``resume: true``) orders only the items the progress
    store has not recorded, then appends the finished ones, so the list
    :class:`~earth2studio.run.Pipeline` runs after dropping finished items
    keeps its block structure.  Rank 0 plans and broadcasts.

    Parameters
    ----------
    cfg : DictConfig
        Full config (``resume``, ``output.path``).
    items : Sequence[WorkItem]
        Time-major campaign items.
    world_size : int
        Number of ranks.
    group_size : int
        Ranks per ensemble group.

    Returns
    -------
    list[WorkItem]
        The same items, re-ordered.

    An initial time with some member groups recorded as done and others
    not, left by a job that stopped between the ranks' records, reruns whole:
    :func:`forget_times` drops its records from the progress store and every
    member group runs again, which rewrites that time's statistics.

    Raises
    ------
    ValueError
        If ``group_size`` does not divide ``world_size``.
    RuntimeError
        If the progress store still records a forgotten time afterwards.
    """
    rank, _ = rank_info()
    if not bool(cfg.get("resume", False)):
        return order_for_groups(items, world_size, group_size)

    def plan() -> list[WorkItem]:
        store = ProgressStore(progress_path(cfg))
        done_set = {item for item in items if store.is_done(item)}
        partial = {i.time for i in items if i not in done_set} & {
            i.time for i in done_set
        }
        if partial:
            logger.warning(
                f"Initial times {sorted(str(t) for t in partial)} are only partly "
                "recorded as done; rerunning every member group of those times, "
                "which rewrites their statistics."
            )
            done_set = set(forget_times(progress_path(cfg), partial, items))
        done = [item for item in items if item in done_set]
        pending = [item for item in items if item not in done_set]
        if not pending or not done:
            return order_for_groups(items, world_size, group_size)
        return order_for_groups(pending, world_size, group_size) + done

    return broadcast(all_agree(plan, rank == 0))


def forget_times(
    path: Path, times: Collection[np.datetime64], items: Sequence[WorkItem]
) -> list[WorkItem]:
    """Rebuild the progress store at ``path`` without the records of ``times``.

    Uses the store's own API only: reads the manifest, clears the store,
    writes the manifest back and records the finished items of every other
    initial time again with ``mark_done``, so the log layout and durability
    stay the library's.  Call this on one rank before any rank opens the store.

    Parameters
    ----------
    path : Path
        The progress store directory.
    times : Collection[np.datetime64]
        Initial times to forget, as the items spell them.
    items : Sequence[WorkItem]
        Every work item of the campaign.

    Returns
    -------
    list[WorkItem]
        The items still recorded as done, in input order.

    Raises
    ------
    RuntimeError
        If an item of the forgotten times is still recorded afterwards.
    """
    store = ProgressStore(path)
    keep = [item for item in items if store.is_done(item) and item.time not in times]
    manifest_path = path / MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    store.clear()
    if manifest is not None:
        store.setup(manifest)
    for item in keep:
        store.mark_done(item)
    store.close()
    check = ProgressStore(path)
    stale = [item for item in items if item.time in times and check.is_done(item)]
    if stale:
        raise RuntimeError(
            f"Progress store '{path}' still records {len(stale)} item(s) of the "
            "initial times to rerun; clear it and resume again."
        )
    return keep


def _no_sync(item: WorkItem | None) -> None:
    """Per-round hook: nothing to do, the group collectives run in the scorer.

    Passing a hook still matters: :class:`earth2studio.run.Pipeline` runs
    its ranks in lockstep rounds only when given one.
    """
    return None


@dataclass
class OnlineRun:
    """An online-scoring run wired to :class:`earth2studio.run.Pipeline`.

    Build with :meth:`build` and pass ``items``, ``scorer`` and ``sync`` to the
    pipeline.  Once the pipeline returns, the last method of this class derives
    the scores stores from the statistics stores.

    Attributes
    ----------
    cfg : DictConfig
        Full config.
    settings : OnlineSettings
        Parsed online settings.
    items : list[WorkItem]
        The campaign's items, ordered for lockstep ensemble groups.
    scorer : OnlineScorer | MultiRefScorer
        The pipeline scorer.
    sync : Callable[[WorkItem | None], None]
        Per-round hook every rank joins (a no-op).
    comm : GroupComm
        This rank's ensemble group collectives.
    stores : dict[str | None, ZarrBackend]
        Open statistics stores by reference name.
    manifest_tags : dict[str, Any]
        What the run accumulates (scored variables, statistic fields,
        regions, climatology, reference names and store name), for the
        progress manifest: a resume that changes any of them must not skip
        finished items, whose new statistics would stay NaN.
    """

    cfg: DictConfig
    settings: OnlineSettings
    items: list[WorkItem]
    scorer: OnlineScorer | MultiRefScorer
    sync: Callable[[WorkItem | None], None]
    comm: GroupComm
    stores: dict[str | None, ZarrBackend]
    manifest_tags: dict[str, Any]

    @classmethod
    def build(
        cls,
        cfg: DictConfig,
        settings: OnlineSettings,
        runner: Any,
        items: Sequence[WorkItem],
        schemas: Mapping[str, CoordinateSystem],
        regridders: Mapping[str, Regridder],
        variables: Mapping[str, Sequence[str]],
        device: torch.device | str,
    ) -> OnlineRun:
        """Assemble the online run on every rank.  Collective over the world.

        Parameters
        ----------
        cfg : DictConfig
            Full config.
        settings : OnlineSettings
            Parsed ``scoring.online`` block.
        runner : Runner
            The campaign's runner; informational.
        items : Sequence[WorkItem]
            Time-major items built with ``members_per_rank`` members each.
        schemas : Mapping[str, CoordinateSystem]
            Stream schemas from ``runner.output_coords``.
        regridders : Mapping[str, Regridder]
            Per-stream output regridders.
        variables : Mapping[str, Sequence[str]]
            Output variables per stream, from :func:`src.output.stream_variables`.
        device : torch.device | str
            Device for accumulation.

        Returns
        -------
        OnlineRun
            The wired run.

        Raises
        ------
        ValueError
            If the group size does not divide the world size, no stream
            produces a scored variable, or the scored streams sit on
            different grids.
        """
        rank, world_size = rank_info()
        device = torch.device(device)
        ensemble_size = int(cfg.get("ensemble_size", 1) or 1)
        group_size, members_per_rank = plan_groups(settings, ensemble_size)
        if world_size % group_size:
            raise ValueError(
                f"world_size={world_size} must be a multiple of the ensemble group "
                f"size {group_size} (ensemble_size / members_per_rank)."
            )
        scored = [str(v) for v in cfg.scoring.variables]
        streams = [
            s for s, names in variables.items() if any(v in scored for v in names)
        ]
        produced = {v for s in streams for v in variables[s]}
        missing = [v for v in scored if v not in produced]
        if missing:
            raise ValueError(
                f"scoring.variables {missing} are not among the output variables of "
                "any stream; add them to output.variables."
            )
        grid, spatial_coords, lead_times = scored_grid(schemas, regridders, streams)
        spatial_dims = tuple(spatial_coords)
        ordered = order_items(cfg, items, world_size, group_size)
        times = np.unique(
            np.array([item.time for item in items], dtype="datetime64[ns]")
        )

        comm = build_group_comm(settings, rank, world_size, ensemble_size)
        sources = resolve_verification_sources(cfg)
        nan_policy = str(cfg.scoring.get("nan_policy", "propagate")).lower()
        valid_ranges_cfg = cfg.scoring.get("valid_ranges", None)
        valid_ranges = (
            OmegaConf.to_container(valid_ranges_cfg, resolve=True) or {}
            if valid_ranges_cfg is not None
            else {}
        )
        weights = build_spatial_weights(
            spatial_coords,
            bool(cfg.scoring.get("lat_weights", False)),
            regions=settings.regions,
        )
        region_names = None if settings.regions is None else list(settings.regions)
        statistics = build_statistics(
            ensemble_size, settings.climatology is not None, settings, scored
        )
        groups = stats_array_groups(
            statistics, scored, times, lead_times, ensemble_size, region_names
        )
        resume = bool(cfg.get("resume", False))
        overwrite = bool(cfg.output.get("overwrite", False))

        climatology = None
        if settings.climatology is not None:
            logger.info("Instantiating the climatology source for ACC.")
            climatology = FieldCache(
                instantiate_source(settings.climatology),
                scored,
                grid,
                spatial_coords,
                device,
                max_size=settings.verification_cache_size,
                nan_policy=nan_policy,
                label="Climatology",
                regrid=output_regridder(cfg),
            )

        manifest_tags: dict[str, Any] = {
            "scored_variables": list(scored),
            "statistics": sorted(f for stat in statistics for f in stat.fields()),
            "regions": region_names,
            "climatology": climatology is not None,
            "verification_sources": [*sources],
            "stats_store": settings.stats_store,
            "nan_policy": nan_policy,
        }
        # Items this run's progress store records as done, after order_items
        # dropped the partly recorded initial times: prepare_store refuses a
        # resume whose statistics store has gone missing while records remain.
        # Records under other tags belong to another run; the pipeline's
        # manifest check refuses that resume with the reason.
        tags = {**progress_tags(cfg, online=True), **manifest_tags}
        recorded = recorded_items(progress_path(cfg), ordered, tags) if resume else 0
        stores: dict[str | None, ZarrBackend] = {}
        scorers: dict[str | None, OnlineScorer] = {}
        store_names: list[str] = []
        for name, source in sources.items():
            check_verification_coverage(source, list(times), lead_times, scored)
            store_names.append(named_store_filename(settings.stats_store, name))
            path = os.path.join(cfg.output.path, store_names[-1])
            stores[name] = open_stats_store(
                path, groups, region_names, resume, overwrite, recorded
            )
            scorers[name] = OnlineScorer(
                settings=settings,
                comm=comm,
                verification=FieldCache(
                    source,
                    scored,
                    grid,
                    spatial_coords,
                    device,
                    max_size=settings.verification_cache_size,
                    nan_policy=nan_policy,
                    valid_ranges=valid_ranges,  # type: ignore[arg-type]
                    regrid=output_regridder(cfg),
                ),
                climatology=climatology,
                variables=scored,
                lead_times=lead_times,
                spatial_dims=spatial_dims,
                weights=weights,
                store=stores[name],
                device=device,
                nan_policy=nan_policy,
                valid_ranges=valid_ranges,  # type: ignore[arg-type]
                streams=variables,
            )
        scorer: OnlineScorer | MultiRefScorer = (
            scorers[None] if list(scorers) == [None] else MultiRefScorer(scorers)
        )
        logger.info(
            f"Online scoring: {len(scored)} variables on streams {streams}, "
            f"{len(lead_times)} lead times, {len(times)} initial conditions, "
            f"{len(sources)} reference(s) -> {store_names}"
        )
        return cls(
            cfg=cfg,
            settings=settings,
            items=list(ordered),
            scorer=scorer,
            sync=_no_sync,
            comm=comm,
            stores=stores,
            manifest_tags=manifest_tags,
        )

    def finalize(self) -> None:
        """Close the statistics stores and, on rank 0, derive the scores stores."""
        for store in self.stores.values():
            store.close()
        rank, _ = rank_info()
        if rank == 0:
            finalize_online_stats(self.cfg)
