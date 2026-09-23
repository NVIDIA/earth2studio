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

"""Tests for the data-cache age pruner (torch-free)."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest
from src.shared.data_cache import (
    data_cache_root,
    prune_data_cache,
    resolve_cache_retention_hours,
)


def _write(path: Path, age_hours: float) -> None:
    """Create a file whose mtime is age_hours in the past."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * 16)
    old = time.time() - age_hours * 3600.0
    os.utime(path, (old, old))


def _set_cache_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EARTH2STUDIO_DATA_CACHE", raising=False)
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path))


def test_prunes_only_stale_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_cache_root(tmp_path, monkeypatch)
    stale = tmp_path / "hrrr" / "old.grib2"
    fresh = tmp_path / "gfs" / "new.grib2"
    _write(stale, age_hours=100)
    _write(fresh, age_hours=1)

    prune_data_cache(48)

    assert not stale.exists()  # 100h > 48h -> deleted
    assert fresh.exists()  # 1h < 48h -> kept
    assert (tmp_path / "hrrr").is_dir()  # directories are never removed


def test_never_prunes_model_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_cache_root(tmp_path, monkeypatch)
    # Model packages cache under the SAME root as data (Package.default_cache), one subdir each.
    weights = [
        tmp_path / "stormcast-conus" / "model.mdlus",
        tmp_path / "sfno" / "weights.tar",
        tmp_path / "fcn3" / "best_ckpt.tar",
        tmp_path / "root_level_file.txt",  # anything directly under the root, too
    ]
    for w in weights:
        _write(
            w, age_hours=10_000
        )  # ancient, would be deleted by a whole-root age walk
    stale_data = tmp_path / "hrrr" / "old.grib2"
    _write(stale_data, age_hours=10_000)

    prune_data_cache(48)

    for w in weights:
        assert w.exists(), f"pruner must never touch model cache {w}"
    assert not stale_data.exists()  # but data blobs are still pruned


def test_zero_retention_disables_pruning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_cache_root(tmp_path, monkeypatch)
    ancient = tmp_path / "hrrr" / "ancient.grib2"
    _write(ancient, age_hours=10_000)

    prune_data_cache(0)

    assert ancient.exists()


@pytest.mark.parametrize("retention_hours", [-1, True, float("inf"), "48"])
def test_prune_rejects_invalid_retention(retention_hours: object) -> None:
    with pytest.raises(
        ValueError,
        match="retention_hours must be a non-negative, finite number",
    ):
        prune_data_cache(retention_hours)  # type: ignore[arg-type]


def test_prune_validates_all_subdirectories_before_deleting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_cache_root(tmp_path, monkeypatch)
    stale = tmp_path / "hrrr" / "old.grib2"
    _write(stale, age_hours=100)

    with pytest.raises(ValueError, match="unsupported data-cache subdirectory 'other'"):
        prune_data_cache(48, ("hrrr", "other"))

    assert stale.exists()


def test_missing_cache_root_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_cache_root(tmp_path / "does-not-exist", monkeypatch)
    prune_data_cache(48)  # no exception


def test_data_cache_env_precedence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(tmp_path / "a"))
    monkeypatch.setenv("EARTH2STUDIO_DATA_CACHE", str(tmp_path / "b"))
    assert data_cache_root() == str(tmp_path / "b")


def test_retention_defaults_to_48_hours_and_preserves_minimum() -> None:
    assert resolve_cache_retention_hours({}, 30) == 48
    assert resolve_cache_retention_hours({"cache_retention_hours": 12}, 30) == 30
    assert resolve_cache_retention_hours({"cache_retention_hours": 0}, 30) == 0


@pytest.mark.parametrize("value", [-1, True, float("inf"), "48"])
def test_invalid_retention_is_rejected(value: object) -> None:
    with pytest.raises(ValueError):
        resolve_cache_retention_hours({"cache_retention_hours": value}, 30)


@pytest.mark.parametrize("minimum_hours", [-1, True, float("inf"), "30"])
def test_invalid_minimum_retention_is_rejected(minimum_hours: object) -> None:
    with pytest.raises(
        ValueError,
        match="minimum_hours must be a non-negative, finite number",
    ):
        resolve_cache_retention_hours(
            {},
            minimum_hours,  # type: ignore[arg-type]
        )
