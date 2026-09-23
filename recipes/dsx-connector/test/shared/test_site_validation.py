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

"""Tests for startup site-config validation (torch-free)."""

from __future__ import annotations

import pytest
from src.shared.site_extraction import validate_and_normalize_sites


def test_rejects_empty_list() -> None:
    with pytest.raises(ValueError, match="at least one site"):
        validate_and_normalize_sites([])


@pytest.mark.parametrize("bad_id", ["", "   ", None, 5])
def test_rejects_bad_id(bad_id: object) -> None:
    with pytest.raises(ValueError, match="non-empty string"):
        validate_and_normalize_sites([{"id": bad_id, "lat": 40.0, "lon": -100.0}])


def test_rejects_duplicate_id() -> None:
    with pytest.raises(ValueError, match="duplicate site id"):
        validate_and_normalize_sites(
            [
                {"id": "dc-1", "lat": 40.0, "lon": -100.0},
                {"id": "dc-1", "lat": 41.0, "lon": -99.0},
            ]
        )


@pytest.mark.parametrize(
    "coord,value",
    [
        ("lat", float("nan")),
        ("lat", 91.0),
        ("lat", -91.0),
        ("lon", float("inf")),
        ("lon", 361.0),
        ("lon", -181.0),
    ],
)
def test_rejects_out_of_range_or_nonfinite(coord: str, value: float) -> None:
    site = {"id": "dc-1", "lat": 40.0, "lon": -100.0}
    site[coord] = value
    with pytest.raises(ValueError, match=coord):
        validate_and_normalize_sites([site])


@pytest.mark.parametrize(
    ("coord", "value"),
    [
        pytest.param("lat", "not-a-number", id="nonnumeric-lat"),
        pytest.param("lon", "not-a-number", id="nonnumeric-lon"),
        pytest.param("lat", True, id="boolean-lat"),
        pytest.param("lon", True, id="boolean-lon"),
    ],
)
def test_rejects_nonnumeric_or_boolean_coordinate(coord: str, value: object) -> None:
    site = {"id": "dc-1", "lat": 40.0, "lon": -100.0}
    site[coord] = value
    with pytest.raises(ValueError, match="finite number"):
        validate_and_normalize_sites([site])


@pytest.mark.parametrize("coord", ["lat", "lon"])
def test_rejects_missing_coordinate(coord: str) -> None:
    site = {"id": "dc-1", "lat": 40.0, "lon": -100.0}
    del site[coord]

    with pytest.raises(ValueError, match="finite number"):
        validate_and_normalize_sites([site])


def test_returns_normalized_copies_without_changing_input() -> None:
    sites = [
        {"id": "dc-1", "lat": "40.0", "lon": "-100.0"},
        {"id": "dc-2", "lat": 41.26, "lon": 264.06},
    ]

    normalized = validate_and_normalize_sites(sites)

    assert normalized == [
        {"id": "dc-1", "lat": 40.0, "lon": -100.0},
        {"id": "dc-2", "lat": 41.26, "lon": 264.06},
    ]
    assert all(copy is not original for copy, original in zip(normalized, sites))
    assert sites == [
        {"id": "dc-1", "lat": "40.0", "lon": "-100.0"},
        {"id": "dc-2", "lat": 41.26, "lon": 264.06},
    ]
