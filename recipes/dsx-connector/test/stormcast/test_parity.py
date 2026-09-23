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

"""Structured-output parity coverage for the StormCast publishing pipeline.

The representative cycle includes multiple sites and leads, every published
variable, a NaN value, and off-meridian winds. Messages are captured at the
transport boundary and compared structurally after populating ``_buffer``
directly. Collector ``write()`` and site extraction are covered separately.

Regenerate the golden deliberately with ``UPDATE_PARITY_GOLDEN=1`` ONLY when a wire change is
intended and reviewed. A missing golden fails (it is not silently regenerated).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from src.dsx.coordinator import DSXCoordinator
from src.dsx.publisher import DSXPublisher
from src.dsx.schema import WeatherSchema
from src.stormcast.collector import StormCastCollector
from src.stormcast.variables import VARIABLES

GOLDEN = Path(__file__).resolve().parent / "golden" / "parity_cycle.json"
FIXED_CLOCK_MS = (
    9_999_000  # every publicationTime is stamped with this, so output is stable
)

# Two sites at off-meridian longitudes (so wind rotation is non-trivial and site-dependent).
SITES = [
    {"id": "dc-omaha-1", "lat": 41.26, "lon": -95.94},
    {"id": "dc-dallas-1", "lat": 32.78, "lon": -96.80},
]


@pytest.fixture(scope="module")
def schema() -> WeatherSchema:
    return WeatherSchema.load()


class _CaptureTransport:
    """Capturing stand-in for BusTransport: records the encoded bytes each publish sends."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes, bool]] = []

    def publish(self, topic: str, payload: bytes, retain: bool = False) -> bool:
        self.sent.append((topic, payload, retain))
        return True


def _capture(schema: WeatherSchema) -> list[dict]:
    """Serialize one representative cycle and return decoded messages sorted by topic."""
    extractor = SimpleNamespace(site_lon={s["id"]: s["lon"] % 360.0 for s in SITES})
    tx = _CaptureTransport()
    # real _emit runs (match_topic + json.dumps); tx captures the bytes
    publisher = DSXPublisher(tx, schema, dry_run=False)
    publisher._now_ms = (
        lambda: FIXED_CLOCK_MS
    )  # fixed clock -> deterministic publicationTime
    collector = StormCastCollector(extractor, "test-model", init_ms=1000)
    coordinator = DSXCoordinator(
        publisher,
        schema,
        "Weather/v1/PUB",
        "conus-site-weather",
        {s["id"]: s for s in SITES},
        "test-model",
        6,
        3600,
        [{"source": "HRRR", "role": "initialCondition"}],
        VARIABLES,
    )

    # Representative input buffer: all five source vars, three leads, and a NaN temperature
    # lead (which must serialize as null). Distinct per-site temperature.
    src = {"t2m": 300.0, "u10m": 3.0, "v10m": 4.0, "q1hl": 0.010, "p1hl": 96000.0}
    collector._buffer[("dc-omaha-1", 0)] = dict(src)
    collector._buffer[("dc-omaha-1", 3)] = {**src, "t2m": 295.0}
    collector._buffer[("dc-omaha-1", 6)] = {**src, "t2m": float("nan")}
    collector._buffer[("dc-dallas-1", 0)] = {**src, "t2m": 305.0}

    coordinator.stage_forecasts(collector.collect())
    publisher.publish_pending()  # captures the Forecast bundles (retain=False)
    coordinator.publish_metadata()  # captures the retained Metadata (retain=True)

    messages = [
        {"topic": topic, "retain": retain, "payload": json.loads(payload)}
        for topic, payload, retain in tx.sent
    ]
    messages.sort(key=lambda m: m["topic"])
    return messages


def test_parity_cycle(schema: WeatherSchema) -> None:
    got = _capture(schema)
    if os.environ.get("UPDATE_PARITY_GOLDEN") == "1":
        GOLDEN.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN.write_text(json.dumps(got, indent=2, sort_keys=True) + "\n")
        pytest.skip(f"parity golden regenerated at {GOLDEN}")
    assert GOLDEN.exists(), (
        f"parity golden missing at {GOLDEN}; regenerate deliberately with "
        "UPDATE_PARITY_GOLDEN=1 (only when an intended, reviewed wire change occurs)"
    )
    want = json.loads(GOLDEN.read_text())
    assert got == want


def test_publisher_retry_restamps_and_preserves_forecast(
    schema: WeatherSchema, monkeypatch, caplog
) -> None:
    sent: list[dict] = []
    outcomes = iter([False, True])

    class _RetryTransport:
        def publish(self, topic: str, payload: bytes, retain: bool = False) -> bool:
            sent.append(json.loads(payload))
            return next(outcomes)

    publisher = DSXPublisher(_RetryTransport(), schema, dry_run=False)
    publisher.stage_batch(
        [
            (
                "Weather/v1/PUB/Forecast/test-product/dc-omaha-1/Temperature",
                {"publicationTime": 0, "leadSeconds": [0], "values": [300.0]},
            )
        ]
    )
    clock = iter([2_000, 3_000])
    monkeypatch.setattr(publisher, "_now_ms", lambda: next(clock))

    with caplog.at_level(logging.WARNING):
        assert not publisher.publish_pending()
    assert publisher.has_pending
    assert any("remain queued" in record.message for record in caplog.records)

    assert publisher.publish_pending()
    assert not publisher.has_pending
    assert [message["publicationTime"] for message in sent] == [2_000, 3_000]
    assert sent[0]["values"] == sent[1]["values"]
    assert sent[0]["leadSeconds"] == sent[1]["leadSeconds"]
