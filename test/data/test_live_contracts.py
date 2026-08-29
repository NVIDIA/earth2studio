# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

"""Required real-provider contracts and zero-network registry checks."""

from __future__ import annotations

import inspect
from time import perf_counter

import pytest

import earth2studio.data as e2data
from test.data.live_cases import (
    LIVE_CASES,
    NETWORK_SOURCE_CLASSES,
    NON_NETWORK_SOURCES,
    LiveCase,
)


@pytest.mark.live_network
@pytest.mark.parametrize("case", LIVE_CASES, ids=lambda case: case.id)
def test_live_contract(case: LiveCase, tmp_path, monkeypatch) -> None:
    """Make one cold public call and validate a small real result."""
    cache_root = tmp_path / case.id
    cache_root.mkdir()
    assert not any(cache_root.iterdir())
    monkeypatch.setenv("EARTH2STUDIO_CACHE", str(cache_root))
    monkeypatch.setenv("EARTH2STUDIO_DATA_CACHE", str(cache_root))

    started = perf_counter()
    result = case.run()
    elapsed = perf_counter() - started
    case.validate(result)

    cache_bytes = sum(
        path.stat().st_size for path in cache_root.rglob("*") if path.is_file()
    )
    print(
        f"live-contract id={case.id} provider={case.provider_group} "
        f"elapsed_seconds={elapsed:.3f} cache_bytes={cache_bytes}"
    )


def test_public_data_sources_are_classified() -> None:
    """Fail when a public source is added without an explicit classification."""
    public_classes = {
        value
        for name, value in vars(e2data).items()
        if not name.startswith("_")
        and inspect.isclass(value)
        and value.__module__.startswith("earth2studio.data")
    }
    assert not NETWORK_SOURCE_CLASSES & NON_NETWORK_SOURCES
    assert public_classes == NETWORK_SOURCE_CLASSES | NON_NETWORK_SOURCES


def test_live_case_registry_is_well_formed() -> None:
    """Keep pilot cases unique and tied to classified network sources."""
    ids = [case.id for case in LIVE_CASES]
    source_types = [case.source_type for case in LIVE_CASES]

    assert len(ids) == len(set(ids))
    assert len(source_types) == len(set(source_types))
    assert set(source_types) <= NETWORK_SOURCE_CLASSES
