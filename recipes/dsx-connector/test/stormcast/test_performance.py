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

"""Tests for the opt-in performance.num_diffusion_steps override (torch-free)."""

from __future__ import annotations

import pytest
from src.stormcast.workflow import _resolve_num_diffusion_steps


def test_missing_or_null_returns_none() -> None:
    assert _resolve_num_diffusion_steps({}) is None
    assert _resolve_num_diffusion_steps({"num_diffusion_steps": None}) is None


def test_valid_value_is_returned() -> None:
    assert _resolve_num_diffusion_steps({"num_diffusion_steps": 2}) == 2
    assert _resolve_num_diffusion_steps({"num_diffusion_steps": 1001}) == 1001


def test_below_minimum_raises() -> None:
    with pytest.raises(ValueError, match="num_diffusion_steps"):
        _resolve_num_diffusion_steps({"num_diffusion_steps": 1})


@pytest.mark.parametrize("bad", [18.0, "8", True])
def test_non_integer_raises(bad: object) -> None:
    with pytest.raises(ValueError, match="integer"):
        _resolve_num_diffusion_steps({"num_diffusion_steps": bad})
