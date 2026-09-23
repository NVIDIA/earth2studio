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

"""Tests for the center->box subregion derivation (torch-free; the grid lookup is not tested)."""

from __future__ import annotations

import pytest
from src.stormcast.workflow import (
    _derive_subregion_box,
    _nearest_hrrr_index,
    _validate_subregion_box,
)


def _aligned(box: tuple[tuple[int, int], tuple[int, int]]) -> bool:
    """The model's load check: offset-from-origin (17,3) and extent both divisible by 4."""
    (lat0, lat1), (lon0, lon1) = box
    return not (
        (lat0 - 17) % 4 or (lat1 - lat0) % 4 or (lon0 - 3) % 4 or (lon1 - lon0) % 4
    )


def test_centered_box_is_aligned_sized_and_near_center() -> None:
    box = _derive_subregion_box(500, 900, (512, 640))
    (lat0, lat1), (lon0, lon1) = box
    assert (lat1 - lat0, lon1 - lon0) == (512, 640)
    assert _aligned(box)
    assert abs((lat0 + lat1) // 2 - 500) < 4
    assert abs((lon0 + lon1) // 2 - 900) < 4


def test_clamps_to_low_edge_keeping_size_and_alignment() -> None:
    box = _derive_subregion_box(20, 6, (512, 640))
    assert box == ((17, 529), (3, 643))


def test_far_north_center_clamps_to_domain_edge() -> None:
    box = _derive_subregion_box(1040, 1794, (512, 640))
    assert box == ((529, 1041), (1155, 1795))


@pytest.mark.parametrize("y,x", [(16, 900), (1041, 900), (500, 2), (500, 1795)])
def test_center_outside_domain_raises(y: int, x: int) -> None:
    with pytest.raises(ValueError, match="outside the model domain"):
        _derive_subregion_box(y, x, (512, 640))


@pytest.mark.parametrize(
    "size", [(513, 640), (512, 641), (0, 640), (2000, 640), (512, 2000)]
)
def test_bad_size_raises(size: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="must be a positive multiple"):
        _derive_subregion_box(500, 900, size)


def test_validate_accepts_shipped_box() -> None:
    assert _validate_subregion_box((273, 785), (579, 1219)) == ((273, 785), (579, 1219))


def test_validate_accepts_crop_near_northern_edge() -> None:
    assert _validate_subregion_box((517, 1029), (603, 1243)) == (
        (517, 1029),
        (603, 1243),
    )


def test_validate_rejects_below_min_size() -> None:
    with pytest.raises(ValueError, match="below the 128"):
        _validate_subregion_box((101, 165), (603, 731))  # lat extent 64


def test_validate_rejects_misaligned() -> None:
    with pytest.raises(ValueError, match="divisible by 4"):
        _validate_subregion_box((100, 612), (603, 1243))  # (100-17) % 4 != 0


def test_validate_rejects_out_of_domain() -> None:
    with pytest.raises(ValueError, match="must satisfy"):
        _validate_subregion_box((17, 1045), (603, 1243))  # lat1 > 1041


@pytest.mark.parametrize(
    ("lat", "lon"),
    [(float("nan"), 0.0), (0.0, float("inf"))],
)
def test_nearest_hrrr_index_rejects_nonfinite_coordinates(
    lat: float, lon: float
) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _nearest_hrrr_index(lat, lon)


@pytest.mark.parametrize("lat", [-90.1, 90.1])
def test_nearest_hrrr_index_rejects_invalid_latitude(lat: float) -> None:
    with pytest.raises(ValueError, match="between -90 and 90"):
        _nearest_hrrr_index(lat, 0.0)
