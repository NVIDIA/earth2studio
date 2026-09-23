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

"""Sanity checks for the internal forecast representation in ``src/shared/forecast.py``."""

from __future__ import annotations

import dataclasses

import pytest
from src.shared.forecast import ForecastSeries


def _series(**kw: object) -> ForecastSeries:
    base = dict(
        site_id="dc-omaha-1",
        variable="Temperature",
        init_ms=1000,
        lead_seconds=[0, 1, 2],
        member_values=[[300.0, 299.5, None]],  # single (deterministic) member
        model="stormcast-conus",
    )
    base.update(kw)
    return ForecastSeries(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("member_values", "expected_count", "is_probabilistic"),
    [
        ([[300.0, 299.5, None]], 1, False),
        ([[300.0, 299.5, 299.0], [300.2, 299.7, 299.1]], 2, True),
    ],
    ids=["deterministic", "ensemble"],
)
def test_derives_member_count_and_probability_flag(
    member_values: list[list[float | None]],
    expected_count: int,
    is_probabilistic: bool,
) -> None:
    series = _series(member_values=member_values)

    assert series.member_count == expected_count
    assert series.is_probabilistic is is_probabilistic


def test_rejects_empty_members() -> None:
    # At least one member is required (deterministic == exactly one).
    with pytest.raises(ValueError, match="at least one member"):
        _series(member_values=[])


@pytest.mark.parametrize(
    ("member_values", "member_index"),
    [
        ([[300.0, 299.5]], 0),
        ([[300.0, 299.5, 299.0], [300.2, 299.7]], 1),
    ],
    ids=["first-member", "later-member"],
)
def test_rejects_ragged_members(
    member_values: list[list[float]], member_index: int
) -> None:
    with pytest.raises(
        ValueError, match=rf"member {member_index} has 2 values.*3 lead times"
    ):
        _series(member_values=member_values)


def test_field_reassignment_is_blocked() -> None:
    # frozen=True guards against reassigning a field (not against mutating list contents).
    s = _series()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.member_values = []  # type: ignore[misc]
