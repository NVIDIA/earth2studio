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

"""Direct tests for ``DSXCoordinator`` in ``src/dsx/coordinator.py``.

No GPU, no model, no bus: a mock publisher records what the coordinator stages/emits, and the
real vendored schema validates the payloads.
"""

from __future__ import annotations

import pytest
from src.dsx.coordinator import DSXCoordinator
from src.dsx.schema import WeatherSchema
from src.shared.forecast import ForecastSeries
from src.stormcast.variables import VARIABLES

SITES = {"dc-omaha-1": {"id": "dc-omaha-1", "lat": 41.26, "lon": -95.94}}


@pytest.fixture(scope="module")
def schema() -> WeatherSchema:
    return WeatherSchema.load()


class _FakePublisher:
    def __init__(self) -> None:
        self.staged: list | None = None
        self.metadata: list | None = None

    def stage_batch(self, messages: list) -> None:
        self.staged = messages

    def publish_metadata(self, messages: list) -> None:
        self.metadata = messages


def _coord(schema: WeatherSchema, publisher: _FakePublisher) -> DSXCoordinator:
    return DSXCoordinator(
        publisher,
        schema,
        "Weather/v1/PUB",
        "test-product",
        SITES,
        "test-model",
        24,
        600,
        [],
        VARIABLES,
    )


def _series(
    variable: str = "Temperature", values: list | None = None
) -> ForecastSeries:
    return ForecastSeries(
        site_id="dc-omaha-1",
        variable=variable,
        init_ms=1000,
        lead_seconds=[0, 1],
        member_values=[values if values is not None else [300.0, 299.5]],
        model="test-model",
    )


def test_stage_forecasts_converts_validates_and_stages(schema: WeatherSchema) -> None:
    pub = _FakePublisher()
    _coord(schema, pub).stage_forecasts([_series()])
    assert pub.staged is not None and len(pub.staged) == 1
    topic, payload = pub.staged[0]
    assert topic == "Weather/v1/PUB/Forecast/test-product/dc-omaha-1/Temperature"
    assert payload["values"] == [300.0, 299.5]
    schema.validate_payload(
        "ForecastBundleMessage", payload
    )  # coordinator already validated


def test_stage_forecasts_is_atomic_on_failure(schema: WeatherSchema) -> None:
    # A later series raises during staging -> NOTHING is staged (the batch is built locally,
    # then staged in one call). Trigger: a variable outside the governed enum, which the
    # coordinator rejects at topic validation before staging.
    pub = _FakePublisher()
    good = _series()
    bad = ForecastSeries(
        site_id="dc-omaha-1",
        variable="NotAGovernedVariable",  # not in the contract enum -> topic match fails
        init_ms=1000,
        lead_seconds=[0, 1],
        member_values=[[300.0, 299.5]],
        model="test-model",
    )
    with pytest.raises(ValueError, match="does not match the contract"):
        _coord(schema, pub).stage_forecasts([good, bad])
    assert pub.staged is None  # atomic: nothing staged when a later series fails


def test_stage_forecasts_rejects_leads_over_horizon(
    schema: WeatherSchema,
) -> None:
    pub = _FakePublisher()
    coordinator = DSXCoordinator(
        pub,
        schema,
        "Weather/v1/PUB",
        "test-product",
        SITES,
        "test-model",
        3600,
        3600,
        [],
        VARIABLES,
    )
    forecast = ForecastSeries(
        site_id="dc-omaha-1",
        variable="Temperature",
        init_ms=1000,
        lead_seconds=[0, 7200],
        member_values=[[300.0, 299.5]],
        model="test-model",
    )

    with pytest.raises(ValueError, match="exceeds advertised horizonSeconds"):
        coordinator.stage_forecasts([forecast])
    assert pub.staged is None


def test_publish_metadata_builds_all_pairs(schema: WeatherSchema) -> None:
    pub = _FakePublisher()
    _coord(schema, pub).publish_metadata()
    assert pub.metadata is not None
    assert len(pub.metadata) == len(SITES) * 5  # five advertised variables
    topics = {t for t, _ in pub.metadata}
    assert "Weather/v1/PUB/Metadata/test-product/dc-omaha-1/WetBulb" in topics
    for _, meta in pub.metadata:
        schema.validate_payload(
            "ForecastMetadataMessage", meta
        )  # coordinator already validated
