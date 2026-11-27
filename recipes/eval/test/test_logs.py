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

"""Process setup helpers of ``src.logs``."""

from __future__ import annotations

import os

import pytest
from src.logs import all_agree, isolate_data_cache, rank0_first_checked


def test_isolate_data_cache_gives_the_process_its_own_root(monkeypatch) -> None:
    monkeypatch.delenv("EARTH2STUDIO_DATA_CACHE", raising=False)
    isolate_data_cache()
    path = os.environ["EARTH2STUDIO_DATA_CACHE"]
    assert os.path.isdir(path) and "e2s-data-cache-" in path
    # A second call keeps the root this process already owns.
    isolate_data_cache()
    assert os.environ["EARTH2STUDIO_DATA_CACHE"] == path


def test_isolate_data_cache_keeps_a_user_root(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("EARTH2STUDIO_DATA_CACHE", str(tmp_path))
    isolate_data_cache()
    assert os.environ["EARTH2STUDIO_DATA_CACHE"] == str(tmp_path)


def test_all_agree_runs_once_and_keeps_the_exception() -> None:
    calls: list[int] = []

    def work() -> int:
        calls.append(1)
        return 7

    assert all_agree(work) == 7
    assert all_agree(work, run=False) is None
    assert calls == [1]
    error = RuntimeError("boom")

    def failing() -> None:
        raise error

    with pytest.raises(RuntimeError) as info:
        all_agree(failing)
    assert info.value is error


def test_rank0_first_checked_returns_the_result_of_one_call() -> None:
    calls: list[int] = []

    def work() -> str:
        calls.append(1)
        return "store"

    assert rank0_first_checked(work) == "store"
    assert calls == [1]

    def failing() -> None:
        raise ValueError("nope")

    with pytest.raises(ValueError, match="nope"):
        rank0_first_checked(failing)
