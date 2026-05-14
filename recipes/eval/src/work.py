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

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import numpy as np
from loguru import logger
from omegaconf import DictConfig

from earth2studio.utils.time import to_time_array

T = TypeVar("T")


@dataclass(frozen=True)
class WorkItem:
    """A single unit of inference work.

    Parameters
    ----------
    time : np.datetime64
        Initial condition time for this forecast.
    ensemble_id : int
        Ensemble member index (0 for deterministic runs).
    seed : int
        Random seed for reproducibility of perturbations.
    """

    time: np.datetime64
    ensemble_id: int = 0
    seed: int = 0


def build_work_items(cfg: DictConfig) -> list[WorkItem]:
    """Build the full list of work items from the Hydra config.

    Generates one WorkItem per (initial_time, ensemble_member) pair.  When no
    ensemble is configured the result is one item per initial time.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with ``start_times`` or ``ic_block_*`` keys, and optional
        ``ensemble`` section.

    Returns
    -------
    list[WorkItem]
        Ordered list of work items to be distributed across ranks.
    """
    ics = _parse_initial_times(cfg)
    n_ensemble = cfg.get("ensemble_size", 1)
    base_seed = cfg.get("random_seed", 42)

    items: list[WorkItem] = []
    for ic in ics:
        for ens_id in range(n_ensemble):
            seed = _deterministic_seed(base_seed, ic, ens_id)
            items.append(WorkItem(time=ic, ensemble_id=ens_id, seed=seed))

    logger.info(
        f"Built {len(items)} work items "
        f"({len(ics)} ICs x {n_ensemble} ensemble members)"
    )
    return items


def distribute_work(
    items: list[T],
    rank: int,
    world_size: int,
) -> list[T]:
    """Partition a list of work items across ranks.

    Items are distributed as evenly as possible; the last rank absorbs any
    remainder.  Returns an empty list (rather than calling ``exit()``) if a
    rank has nothing to do — callers can skip gracefully.

    Parameters
    ----------
    items : list[T]
        Full list of items to distribute.
    rank : int
        Current process rank.
    world_size : int
        Total number of processes.

    Returns
    -------
    list[T]
        Subset of items assigned to this rank.
    """
    n = len(items)
    if world_size <= 1 or n == 0:
        return list(items)

    base, remainder = divmod(n, world_size)
    # First `remainder` ranks each get one extra item
    if rank < remainder:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = remainder * (base + 1) + (rank - remainder) * base
        end = start + base

    subset = items[start:end]
    if len(subset) == 0:
        logger.warning(f"Rank {rank} has no work items assigned; will idle.")
    else:
        logger.info(f"Rank {rank}: assigned {len(subset)}/{n} work items")
    return subset


# ---------------------------------------------------------------------------
# Resume / progress tracking
# ---------------------------------------------------------------------------


def progress_dir(cfg: DictConfig) -> Path:
    """Return the progress-tracking directory for this eval run.

    Completion markers are written here — one per finished work item — so
    that resumed or multi-job runs can skip already-completed items.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with an ``output.path`` key.

    Returns
    -------
    Path
        ``<output.path>/.progress``
    """
    return Path(cfg.output.path) / ".progress"


def _marker_name(item: WorkItem) -> str:
    """Return a filesystem-safe marker filename for *item*."""
    ts = str(item.time.astype("datetime64[s]")).replace("-", "").replace(":", "")
    return f"{ts}_ens{item.ensemble_id}.done"


def write_marker(item: WorkItem, cfg: DictConfig) -> None:
    """Write a completion marker for a finished work item.

    Parameters
    ----------
    item : WorkItem
        The completed work item.
    cfg : DictConfig
        Hydra config (used to locate the progress directory).
    """
    d = progress_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    (d / _marker_name(item)).write_text(np.datetime64("now", "s").item().isoformat())


def filter_completed_items(items: list[WorkItem], cfg: DictConfig) -> list[WorkItem]:
    """Remove work items that already have completion markers.

    Parameters
    ----------
    items : list[WorkItem]
        Full list of work items.
    cfg : DictConfig
        Hydra config (used to locate the progress directory).

    Returns
    -------
    list[WorkItem]
        Items whose markers are absent (i.e. still need to run).
    """
    d = progress_dir(cfg)
    if not d.exists():
        return items
    existing = {f.name for f in d.iterdir() if f.suffix == ".done"}
    remaining = [item for item in items if _marker_name(item) not in existing]
    skipped = len(items) - len(remaining)
    if skipped:
        logger.info(f"Resume: skipping {skipped}/{len(items)} completed work items")
    return remaining


def clear_progress(cfg: DictConfig) -> None:
    """Remove all completion markers for this eval run.

    Called on a fresh (non-resume) run with ``overwrite=true`` so that stale
    markers from a prior run are not picked up if the user later switches to
    ``resume=true``.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config (used to locate the progress directory).
    """
    d = progress_dir(cfg)
    if d.exists():
        import shutil

        shutil.rmtree(d)
        logger.debug(f"Cleared progress directory: {d}")


# ---------------------------------------------------------------------------
# Predownload progress tracking
# ---------------------------------------------------------------------------


def predownload_progress_dir(cfg: DictConfig, store_name: str) -> Path:
    """Return the progress-tracking directory for a predownload store.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with an ``output.path`` key.
    store_name : str
        Logical name of the store (e.g. ``"data"``, ``"verification"``).

    Returns
    -------
    Path
        ``<output.path>/.predownload_progress/<store_name>``
    """
    return Path(cfg.output.path) / ".predownload_progress" / store_name


def _predownload_marker_name(time: np.datetime64) -> str:
    """Return a filesystem-safe marker filename for a predownload timestamp."""
    ts = str(time.astype("datetime64[s]")).replace("-", "").replace(":", "")
    return f"{ts}.done"


def write_predownload_marker(
    time: np.datetime64, cfg: DictConfig, store_name: str
) -> None:
    """Write a completion marker for a predownloaded timestamp.

    Parameters
    ----------
    time : np.datetime64
        The completed timestamp.
    cfg : DictConfig
        Hydra config (used to locate the progress directory).
    store_name : str
        Logical name of the store.
    """
    d = predownload_progress_dir(cfg, store_name)
    d.mkdir(parents=True, exist_ok=True)
    (d / _predownload_marker_name(time)).write_text(
        np.datetime64("now", "s").item().isoformat()
    )


def filter_predownload_completed(
    times: list[np.datetime64], cfg: DictConfig, store_name: str
) -> list[np.datetime64]:
    """Remove timestamps that already have predownload completion markers.

    Parameters
    ----------
    times : list[np.datetime64]
        Full list of timestamps to check.
    cfg : DictConfig
        Hydra config (used to locate the progress directory).
    store_name : str
        Logical name of the store.

    Returns
    -------
    list[np.datetime64]
        Timestamps whose markers are absent (still need downloading).
    """
    d = predownload_progress_dir(cfg, store_name)
    if not d.exists():
        return times
    existing = {f.name for f in d.iterdir() if f.suffix == ".done"}
    remaining = [t for t in times if _predownload_marker_name(t) not in existing]
    skipped = len(times) - len(remaining)
    if skipped:
        logger.info(
            f"Predownload resume ({store_name}): "
            f"skipping {skipped}/{len(times)} completed times"
        )
    return remaining


def clear_predownload_progress(cfg: DictConfig) -> None:
    """Remove all predownload completion markers.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config (used to locate the progress directory).
    """
    d = Path(cfg.output.path) / ".predownload_progress"
    if d.exists():
        import shutil

        shutil.rmtree(d)
        logger.debug(f"Cleared predownload progress directory: {d}")


# ---------------------------------------------------------------------------
# Scoring progress tracking
# ---------------------------------------------------------------------------


def scoring_progress_dir(cfg: DictConfig) -> Path:
    """Return the progress-tracking directory for scoring.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config with an ``output.path`` key.

    Returns
    -------
    Path
        ``<output.path>/.scoring_progress``
    """
    return Path(cfg.output.path) / ".scoring_progress"


def _scoring_marker_name(time: np.datetime64) -> str:
    """Return a filesystem-safe marker filename for a scored timestamp."""
    ts = str(time.astype("datetime64[s]")).replace("-", "").replace(":", "")
    return f"{ts}.done"


def write_scoring_marker(time: np.datetime64, cfg: DictConfig) -> None:
    """Write a completion marker for a scored initial-condition time.

    Parameters
    ----------
    time : np.datetime64
        The completed IC time.
    cfg : DictConfig
        Hydra config (used to locate the progress directory).
    """
    d = scoring_progress_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    (d / _scoring_marker_name(time)).write_text(
        np.datetime64("now", "s").item().isoformat()
    )


def filter_scoring_completed(
    times: list[np.datetime64], cfg: DictConfig
) -> list[np.datetime64]:
    """Remove IC times that already have scoring completion markers.

    Parameters
    ----------
    times : list[np.datetime64]
        Full list of IC times to check.
    cfg : DictConfig
        Hydra config (used to locate the progress directory).

    Returns
    -------
    list[np.datetime64]
        Times whose markers are absent (still need scoring).
    """
    d = scoring_progress_dir(cfg)
    if not d.exists():
        return times
    existing = {f.name for f in d.iterdir() if f.suffix == ".done"}
    remaining = [t for t in times if _scoring_marker_name(t) not in existing]
    skipped = len(times) - len(remaining)
    if skipped:
        logger.info(f"Scoring resume: skipping {skipped}/{len(times)} completed times")
    return remaining


def clear_scoring_progress(cfg: DictConfig) -> None:
    """Remove all scoring completion markers.

    Parameters
    ----------
    cfg : DictConfig
        Hydra config (used to locate the progress directory).
    """
    d = scoring_progress_dir(cfg)
    if d.exists():
        import shutil

        shutil.rmtree(d)
        logger.debug(f"Cleared scoring progress directory: {d}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _parse_initial_times(cfg: DictConfig) -> list[np.datetime64]:
    """Extract initial condition times from config.

    Parameters
    ----------
    cfg : DictConfig
        Config containing either ``start_times`` (explicit list) or
        ``ic_block_start`` / ``ic_block_end`` / ``ic_block_step`` (range).
        For the block form, times are ``np.arange(start, end + step, step)``
        with ``step`` in hours, so ``ic_block_end`` is **inclusive**: the last
        IC equals ``ic_block_end`` when that timestamp lies on the grid from
        ``ic_block_start`` and ``ic_block_step``.

    Returns
    -------
    list[np.datetime64]
        Sorted array of initial condition times.

    Raises
    ------
    ValueError
        If both ``start_times`` and ``ic_block_start`` are provided, or neither.
    """
    has_list = bool(cfg.get("start_times"))
    has_block = cfg.get("ic_block_start") is not None

    if has_list and has_block:
        raise ValueError(
            "Provide either 'start_times' or 'ic_block_start/end/step', not both."
        )
    if not has_list and not has_block:
        raise ValueError(
            "Config must specify either 'start_times' or 'ic_block_start/end/step'."
        )

    if has_list:
        return list(to_time_array(sorted(cfg.start_times)))

    ics = to_time_array([cfg.ic_block_start, cfg.ic_block_end])
    step = np.timedelta64(cfg.ic_block_step, "h")
    return list(np.arange(ics[0], ics[1] + step, step))


def _deterministic_seed(base: int, time: np.datetime64, ensemble_id: int) -> int:
    """Produce a deterministic per-(time, ensemble) seed from a base seed.

    Uses a fixed byte-packing scheme so the result is identical across
    Python processes and runs (unlike ``hash()``, which is salted by
    default via ``PYTHONHASHSEED``).

    Parameters
    ----------
    base : int
        Base random seed from config.
    time : np.datetime64
        Initial condition time.
    ensemble_id : int
        Ensemble member index.

    Returns
    -------
    int
        Deterministic seed value in [0, 2**63).
    """
    time_int = int(time.astype("datetime64[s]").astype("int64"))
    packed = struct.pack(">qqq", base, time_int, ensemble_id)

    # FNV-1a 64-bit — simple, fast, no external deps, fully deterministic.
    FNV_OFFSET = 0xCBF29CE484222325
    FNV_PRIME = 0x00000100000001B3
    h = FNV_OFFSET
    for byte in packed:
        h ^= byte
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h % (2**63)
