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

"""Campaign planning: initial conditions, work items and their ordering.

The recipe describes a campaign in Hydra terms (``start_times`` or an
``ic_block_*`` sweep, ``nsteps``, ``ensemble_size``) and the library runs it in
:class:`earth2studio.run.WorkItem` terms (reference time, final lead time, member
group).  This module is the translation, plus the item ordering that keeps the
ranks of an online-scoring ensemble group in lockstep.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from loguru import logger
from omegaconf import DictConfig

from earth2studio.run import WorkItem, work_items
from earth2studio.run.work import distribute_work
from earth2studio.utils.time import to_time_array

from .regions import parse_events


def event_initial_times(events: dict[str, dict] | None) -> list[np.datetime64]:
    """Initial conditions added by ``scoring.events`` entries with an ``ics`` block.

    For each such event the times run every ``step_hours`` from
    ``start - lookback_hours`` up to and including ``end``.  Events without an
    ``ics`` block add nothing; the campaign's existing initial conditions score
    them.

    Parameters
    ----------
    events : dict[str, dict] | None
        Parsed events (see :func:`src.regions.parse_events`).

    Returns
    -------
    list[np.datetime64]
        Sorted, de-duplicated initial times at second resolution.
    """
    if not events:
        return []
    times: list[np.ndarray] = []
    for name, event in events.items():
        ics = event.get("ics")
        if ics is None:
            continue
        step = np.timedelta64(int(round(float(ics["step_hours"]) * 60)), "m")
        first = event["start"] - np.timedelta64(
            int(round(float(ics["lookback_hours"]) * 60)), "m"
        )
        grid = np.arange(first, event["end"] + step, step).astype("datetime64[s]")
        grid = grid[grid <= np.datetime64(event["end"], "s")]
        logger.info(
            f"Event '{name}': {len(grid)} initial conditions from {grid[0]} "
            f"to {grid[-1]} every {ics['step_hours']:g} h"
        )
        times.append(grid)
    if not times:
        return []
    return list(np.unique(np.concatenate(times)))


def initial_times(cfg: DictConfig) -> list[np.datetime64]:
    """Initial-condition times of a campaign.

    Parameters
    ----------
    cfg : DictConfig
        Config with ``start_times`` (explicit list) or ``ic_block_start`` /
        ``ic_block_end`` / ``ic_block_step`` (inclusive range in hours), or
        neither when ``scoring.events`` adds the initial conditions.

    Returns
    -------
    list[np.datetime64]
        Sorted initial times.

    Raises
    ------
    ValueError
        If the config sets both forms or neither.
    """
    has_list = bool(cfg.get("start_times"))
    has_block = cfg.get("ic_block_start") is not None
    scoring_cfg = cfg.get("scoring", None)
    events = parse_events(scoring_cfg.get("events", None)) if scoring_cfg else None
    event_times = event_initial_times(events)
    if has_list and has_block:
        raise ValueError(
            "Provide either 'start_times' or 'ic_block_start/end/step', not both."
        )
    if not has_list and not has_block and not event_times:
        raise ValueError(
            "Config must specify 'start_times', 'ic_block_start/end/step', or "
            "scoring.events entries with an 'ics' block."
        )
    base: list[np.datetime64]
    if has_list:
        base = list(to_time_array(sorted(cfg.start_times)))
    elif has_block:
        ics = to_time_array([cfg.ic_block_start, cfg.ic_block_end])
        step = np.timedelta64(cfg.ic_block_step, "h")
        base = list(np.arange(ics[0], ics[1] + step, step))
    else:
        base = []
    if not event_times:
        return [np.datetime64(t, "s") for t in base]
    merged = np.unique(np.array(base + event_times, dtype="datetime64[s]"))
    return list(merged)


def ensemble_members(cfg: DictConfig) -> int | None:
    """Ensemble member count, or ``None`` for a deterministic campaign."""
    size = int(cfg.get("ensemble_size", 1) or 1)
    if size < 1:
        raise ValueError(f"ensemble_size must be >= 1, got {size}")
    return size if size > 1 else None


def final_lead_time(cfg: DictConfig, runner: object) -> np.timedelta64:
    """Last lead time of every forecast: ``nsteps`` model steps.

    Parameters
    ----------
    cfg : DictConfig
        Config with ``nsteps``.
    runner : object
        A :class:`earth2studio.run.Runner` exposing its model step as
        ``step``; one without a step (``None``) only supports ``nsteps: 0``.

    Returns
    -------
    np.timedelta64
        Final lead time, ``0`` when ``nsteps`` is zero.
    """
    nsteps = int(cfg.get("nsteps", 0) or 0)
    if nsteps < 0:
        raise ValueError(f"nsteps must be >= 0, got {nsteps}")
    if nsteps == 0:
        return np.timedelta64(0, "h")
    step = getattr(runner, "step", None)
    if step is None:
        raise ValueError(
            f"{type(runner).__name__} declares no model step, so only nsteps=0 "
            "is supported."
        )
    return np.timedelta64(step) * nsteps


def build_items(
    cfg: DictConfig, runner: object, member_batch: int = 1
) -> list[WorkItem]:
    """Build the campaign's work items, time-major.

    Parameters
    ----------
    cfg : DictConfig
        Campaign config.
    runner : object
        The :class:`earth2studio.run.Runner` of the campaign.
    member_batch : int, optional
        Members per item, by default 1

    Returns
    -------
    list[WorkItem]
        One item per initial time and member group.
    """
    times = initial_times(cfg)
    items = work_items(
        times,
        final_lead_time(cfg, runner),
        members=ensemble_members(cfg),
        member_batch=member_batch,
    )
    logger.info(
        f"Built {len(items)} work items ({len(times)} initial conditions, "
        f"ensemble_size={cfg.get('ensemble_size', 1)}, member_batch={member_batch})"
    )
    return items


def order_for_groups(
    items: Sequence[WorkItem], world_size: int, group_size: int
) -> list[WorkItem]:
    """Order items so each ensemble group's ranks run the same time together.

    :meth:`earth2studio.run.Pipeline.run` gives every rank one contiguous slice
    of the item list and, with a ``sync`` hook, runs the ranks' items in
    lockstep rounds.  Online scoring needs the ``group_size`` ranks of one
    ensemble group to hold the same initial condition in the same round, each
    with its own member group.  Laying the items out rank-major achieves that:
    the ranks of group ``b`` get the times ``distribute_work(times, b, n_groups)``
    assigns to that group, in time order, each rank carrying one member group.

    Parameters
    ----------
    items : Sequence[WorkItem]
        Time-major items from :func:`build_items`, with ``group_size`` member
        groups per time.
    world_size : int
        Number of ranks; ``group_size`` must divide it.
    group_size : int
        Ranks per ensemble group, equal to the member groups per time.

    Returns
    -------
    list[WorkItem]
        The same items, ordered for :func:`earth2studio.run.distribute_work`.

    Raises
    ------
    ValueError
        If ``group_size`` does not divide ``world_size`` or the items do
        not hold exactly ``group_size`` member groups per time.
    """
    if group_size < 1 or world_size < group_size:
        raise ValueError(
            f"world_size={world_size} is smaller than the ensemble group size "
            f"{group_size}; launch at least {group_size} ranks."
        )
    if world_size % group_size:
        raise ValueError(
            f"world_size={world_size} must be a multiple of the ensemble group "
            f"size {group_size}, so every group is complete."
        )
    by_time: dict[np.datetime64, list[WorkItem]] = {}
    for item in items:
        by_time.setdefault(item.time, []).append(item)
    times = list(by_time)
    bad = {str(t): len(v) for t, v in by_time.items() if len(v) != group_size}
    if bad:
        raise ValueError(
            f"Every time needs exactly {group_size} member groups for lockstep "
            f"groups; got {bad}"
        )
    n_groups = world_size // group_size
    ordered: list[WorkItem] = []
    for block in range(n_groups):
        block_times = distribute_work(times, block, n_groups)
        for group_rank in range(group_size):
            ordered.extend(by_time[t][group_rank] for t in block_times)
    return ordered


def online_scoring(cfg: DictConfig) -> bool:
    """Whether the campaign scores online (``scoring.mode: online``)."""
    scoring = cfg.get("scoring", None)
    return bool(scoring) and str(scoring.get("mode", "offline")).lower() == "online"


def members_per_item(cfg: DictConfig) -> int:
    """Members rolled out together in one work item.

    Online scoring reads ``scoring.online.members_per_rank``; offline runs use
    ``member_batch`` (default one member per item).
    """
    if online_scoring(cfg):
        return int(cfg.scoring.online.get("members_per_rank", 1) or 1)
    return int(cfg.get("member_batch", 1) or 1)


def campaign_times(items: Sequence[WorkItem]) -> np.ndarray:
    """Sorted unique initial times of the items at nanosecond resolution."""
    return np.unique(np.array([item.time for item in items], dtype="datetime64[ns]"))
