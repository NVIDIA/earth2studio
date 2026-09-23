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

"""Pure tests for ``src/dsx/contract_adapter.py``: no GPU, no model, no bus.

Covers deterministic (single-member) serialization, ensemble (multi-member) reduction to the
DSX summary fields, cleaning/parallelism guards, the copy-at-staging (replay-safety) invariant,
and metadata building.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from src.dsx.contract_adapter import (
    TOPIC_PREFIX,
    check_topics_config,
    series_to_bundle,
    site_metadata_message,
)
from src.dsx.schema import WeatherSchema
from src.shared.forecast import ForecastSeries
from src.stormcast.variables import VARIABLES


@pytest.fixture(scope="module")
def schema() -> WeatherSchema:
    return WeatherSchema.load()


def _series(**kw: object) -> ForecastSeries:
    base = dict(
        site_id="dc-omaha-1",
        variable="Temperature",
        init_ms=1000,
        lead_seconds=[0, 1, 2],
        member_values=[[300.0, 299.5, 299.0]],  # single (deterministic) member
        model="stormcast-conus",
    )
    base.update(kw)
    return ForecastSeries(**base)  # type: ignore[arg-type]


# --- deterministic ------------------------------------------------------------
def test_deterministic_bundle_shape(schema: WeatherSchema) -> None:
    topic, payload = series_to_bundle(
        _series(), "Weather/v1/PUB", "test-product", pub_ms=42
    )
    assert topic == "Weather/v1/PUB/Forecast/test-product/dc-omaha-1/Temperature"
    assert payload == {
        "initTime": 1000,
        "leadSeconds": [0, 1, 2],
        "values": [300.0, 299.5, 299.0],
        "memberCount": 1,
        "publicationTime": 42,
        "model": "stormcast-conus",
    }
    # A single-member series carries memberCount 1 and no summary fields.
    for k in ("standardDeviation", "minimum", "maximum", "percentiles", "members"):
        assert k not in payload
    schema.validate_payload("ForecastBundleMessage", payload)


def test_values_cleaned_nan_and_rounded() -> None:
    _, payload = series_to_bundle(
        _series(
            member_values=[
                [1.23456, np.float64("nan"), np.float32("inf"), -np.inf, None]
            ],
            lead_seconds=[0, 1, 2, 3, 4],
        ),
        "Weather/v1/PUB",
        "test-product",
        pub_ms=1,
    )
    assert payload["values"] == [1.235, None, None, None, None]


@pytest.mark.parametrize(
    ("member_values", "member_index", "match"),
    [
        ([[300.0, 299.5]], 0, "values length"),
        ([[300.0, 299.5], [302.0, 301.0]], 1, "member 1 length"),
    ],
)
def test_member_length_mismatch_raises(
    member_values: list[list[float]], member_index: int, match: str
) -> None:
    s = _series(member_values=member_values, lead_seconds=[0, 1])
    s.member_values[member_index].pop()
    with pytest.raises(ValueError, match=match):
        series_to_bundle(s, "Weather/v1/PUB", "test-product", pub_ms=1)


def test_no_members_is_invalid_input() -> None:
    # An empty member_values reaches series_to_bundle only via post-construction mutation
    # (construction rejects it); the adapter's own n == 0 guard then fires.
    s = _series()
    s.member_values.clear()
    with pytest.raises(ValueError, match="no members"):
        series_to_bundle(s, "Weather/v1/PUB", "test-product", pub_ms=1)


# --- ensemble reduction -------------------------------------------------------
def test_ensemble_reduction_exact(schema: WeatherSchema) -> None:
    # Two members over three leads; every summary is hand-checkable.
    s = _series(
        member_values=[[300.0, 299.5, 299.0], [302.0, 299.5, 301.0]],
        lead_seconds=[0, 1, 2],
    )
    _, payload = series_to_bundle(s, "Weather/v1/PUB", "test-product", pub_ms=7)
    assert payload["memberCount"] == 2
    assert payload["values"] == [301.0, 299.5, 300.0]  # mean
    assert payload["standardDeviation"] == [1.0, 0.0, 1.0]  # population SD (ddof=0)
    assert payload["minimum"] == [300.0, 299.5, 299.0]
    assert payload["maximum"] == [302.0, 299.5, 301.0]
    assert payload["percentiles"] == {
        "p10": [300.2, 299.5, 299.2],
        "p50": [301.0, 299.5, 300.0],
        "p90": [301.8, 299.5, 300.8],
    }
    assert "members" not in payload  # not emitted unless include_members
    schema.validate_payload("ForecastBundleMessage", payload)


def test_ensemble_null_when_any_member_invalid(schema: WeatherSchema) -> None:
    # At lead 1, member 0 is NaN -> every summary at that lead is null (N is not shrunk);
    # raw members keep the per-value null for that member only.
    s = _series(member_values=[[300.0, math.nan], [302.0, 301.0]], lead_seconds=[0, 1])
    _, payload = series_to_bundle(
        s, "Weather/v1/PUB", "test-product", pub_ms=1, include_members=True
    )
    assert payload["values"] == [301.0, None]
    assert payload["standardDeviation"] == [1.0, None]
    assert payload["minimum"] == [300.0, None]
    assert payload["maximum"] == [302.0, None]
    assert payload["percentiles"]["p50"] == [301.0, None]
    assert payload["members"] == [[300.0, None], [302.0, 301.0]]
    schema.validate_payload("ForecastBundleMessage", payload)


# --- replay safety: copy-at-staging -------------------------------------------
def test_payload_is_decoupled_from_source_series() -> None:
    member = [300.0, 299.5, 299.0]
    leads = [0, 1, 2]
    s = _series(member_values=[member], lead_seconds=leads)
    _, payload = series_to_bundle(s, "Weather/v1/PUB", "test-product", pub_ms=1)
    # Mutating the source containers must NOT change the already-built payload (replay-safe).
    member.append(999.0)
    leads.append(3)
    assert payload["values"] == [300.0, 299.5, 299.0]
    assert payload["leadSeconds"] == [0, 1, 2]


# --- metadata -----------------------------------------------------------------
def test_metadata_message(schema: WeatherSchema) -> None:
    site = {"id": "dc-omaha-1", "lat": "41.26", "lon": "264.06"}
    topic, meta = site_metadata_message(
        site,
        "WindU",
        VARIABLES["WindU"],
        "Weather/v1/PUB",
        "test-product",
        model_id="stormcast-conus",
        horizon_seconds=24,
        cadence_seconds=600,
        inputs=[{"source": "HRRR", "role": "initialCondition"}],
    )
    assert topic == "Weather/v1/PUB/Metadata/test-product/dc-omaha-1/WindU"
    assert meta["unit"] == "m s-1"
    assert meta["standardName"] == "eastward_wind"
    assert meta["heightMeters"] == 10
    assert meta["lat"] == 41.26
    assert meta["lon"] == -95.94
    schema.validate_payload("ForecastMetadataMessage", meta)


def test_metadata_inputs_copied() -> None:
    inputs = [{"source": "HRRR", "role": "initialCondition"}]
    _, meta = site_metadata_message(
        {"id": "s", "lat": 0.0, "lon": 0.0},
        "Temperature",
        VARIABLES["Temperature"],
        "Weather/v1/PUB",
        "test-product",
        "m",
        24,
        600,
        inputs,
    )
    # Neither appending to the outer list nor mutating a nested dict may leak into the payload.
    inputs.append({"source": "GFS", "role": "conditioning"})
    inputs[0]["role"] = "MUTATED"
    assert meta["inputs"] == [{"source": "HRRR", "role": "initialCondition"}]


def test_metadata_unit_is_stripped_and_must_not_be_empty() -> None:
    _, metadata = site_metadata_message(
        {"id": "s", "lat": 0.0, "lon": 0.0},
        "Temperature",
        {"unit": " K "},
        "Weather/v1/PUB",
        "test-product",
        "m",
        24,
        600,
        [],
    )
    assert metadata["unit"] == "K"
    with pytest.raises(ValueError, match="unit"):
        site_metadata_message(
            {"id": "s", "lat": 0.0, "lon": 0.0},
            "Temperature",
            {"unit": "   "},
            "Weather/v1/PUB",
            "test-product",
            "m",
            24,
            600,
            [],
        )


def test_topic_prefix_matches_the_contract_addresses(schema: WeatherSchema) -> None:
    for channel in ("forecast", "metadata"):
        address = schema._channels[channel]["address"]
        assert address.startswith(f"{TOPIC_PREFIX}/"), address


@pytest.mark.parametrize("topics", [{}, {"forecast_prefix": TOPIC_PREFIX}])
def test_check_topics_config_accepts_missing_or_contract_prefix(topics: dict) -> None:
    check_topics_config(topics)


def test_check_topics_config_rejects_a_custom_prefix() -> None:
    with pytest.raises(ValueError, match="fixed by the DSX weather contract"):
        check_topics_config({"forecast_prefix": "Other/v1/PUB"})
