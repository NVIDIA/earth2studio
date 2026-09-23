# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Find and remove old downloaded input data from the Earth2Studio cache."""

from __future__ import annotations

import math
import os
import time
from typing import Any

from loguru import logger

_DEFAULT_RETENTION_HOURS = 48.0

# Earth2Studio stores downloaded input data and model files under the same cache root.
# Search only the HRRR and GFS directories so cleanup cannot remove cached model files.
_DATA_CACHE_SUBDIRS = ("hrrr", "gfs")


def _validate_hours(value: Any, name: str) -> float:
    """Return ``value`` as a float after validating it as a non-negative hour count."""
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a non-negative, finite number, got {value!r}")
    return float(value)


def _log_walk_error(error: OSError) -> None:
    """Log a directory-scanning error without interrupting cache cleanup."""
    logger.opt(exception=error).debug(
        "unable to scan cache path {}", error.filename or "<unknown>"
    )


def data_cache_root() -> str:
    """Return the directory used for downloaded input data.

    ``EARTH2STUDIO_DATA_CACHE`` takes priority over ``EARTH2STUDIO_CACHE``. If neither is set, the
    default is ``~/.cache/earth2studio``. The directory is not created by this function.
    """
    return (
        os.environ.get("EARTH2STUDIO_DATA_CACHE")
        or os.environ.get("EARTH2STUDIO_CACHE")
        or os.path.join(os.path.expanduser("~"), ".cache", "earth2studio")
    )


def resolve_cache_retention_hours(
    run_cfg: dict[str, Any], minimum_hours: float
) -> float:
    """Validate and return the effective data-cache retention period.

    The configured period defaults to 48 hours. An explicit zero disables cleanup. Smaller positive
    values are automatically increased to ``minimum_hours``.

    Parameters
    ----------
    run_cfg : dict[str, Any]
        Workflow ``run`` settings. ``cache_retention_hours`` may specify the retention period.
    minimum_hours : float
        Shortest retention period that safely preserves data needed by the workflow.

    Returns
    -------
    float
        Zero when cleanup is disabled; otherwise, the configured or minimum retention period,
        whichever is greater.

    Raises
    ------
    ValueError
        If either retention value is negative, non-numeric, or not finite.
    """
    minimum_hours = _validate_hours(minimum_hours, "minimum_hours")
    configured = _validate_hours(
        run_cfg.get("cache_retention_hours", _DEFAULT_RETENTION_HOURS),
        "run.cache_retention_hours",
    )
    if configured == 0:
        return 0.0
    effective = max(configured, minimum_hours)
    if effective > configured:
        logger.info(
            "raising cache retention from {:g}h to {:g}h to preserve the input lookback window",
            configured,
            effective,
        )
    return effective


def prune_data_cache(
    retention_hours: float, subdirs: tuple[str, ...] = _DATA_CACHE_SUBDIRS
) -> None:
    """Delete old input-data downloads without touching cached model files.

    A file is considered old when its modification time is earlier than the retention cutoff.
    Filesystem errors are ignored so cleanup cannot fail a forecast cycle.

    Parameters
    ----------
    retention_hours : float
        Age in hours after which a file may be deleted. Zero disables cleanup.
    subdirs : tuple[str, ...]
        Data-cache directories to clean. Every directory must be included in the safe allowlist.

    Raises
    ------
    ValueError
        If ``retention_hours`` is negative, non-numeric, or not finite, or if ``subdirs`` contains
        an unsupported directory.
    """
    retention_hours = _validate_hours(retention_hours, "retention_hours")

    unsupported = [subdir for subdir in subdirs if subdir not in _DATA_CACHE_SUBDIRS]
    if unsupported:
        raise ValueError(f"unsupported data-cache subdirectory {unsupported[0]!r}")

    if retention_hours == 0:
        return
    root = data_cache_root()
    cutoff = time.time() - retention_hours * 3600.0
    removed = 0
    freed = 0
    for subdir in subdirs:
        subroot = os.path.join(root, subdir)
        if not os.path.isdir(subroot):
            continue
        for dirpath, _dirs, files in os.walk(subroot, onerror=_log_walk_error):
            for name in files:
                path = os.path.join(dirpath, name)
                try:
                    stat = os.stat(path)
                    if stat.st_mtime < cutoff:
                        os.remove(path)
                        removed += 1
                        freed += stat.st_size
                except OSError:
                    continue
    if removed:
        logger.info(
            "pruned {} cached file(s) ({:.2f} GB) older than {:g}h from {} under {}",
            removed,
            freed / 1e9,
            retention_hours,
            ", ".join(subdirs),
            root,
        )
