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

"""Broker integration tests against a local MQTT broker (torch-free; no model needed).

SKIPPED unless a broker is reachable at ``$DSX_TEST_BROKER`` (default 127.0.0.1:1883), so the
default ``make test`` run stays service-free. A CI job that wants these as a gate must start the
broker itself (they are not self-provisioning). To run locally (bind to loopback only — the broker
is anonymous, so do NOT expose it on all interfaces):

    docker run -d --rm --name dsx-mosq -p 127.0.0.1:1883:1883 \
        -v "$PWD/test/dsx/mosquitto-anon.conf:/mosquitto/config/mosquitto.conf" eclipse-mosquitto
    pytest test/dsx/test_broker_integration.py

They exercise the real DSX bus layer (``BusTransport`` + ``DSXPublisher``) end-to-end: retained
Metadata vs non-retained Forecast semantics, and that distinct client ids coexist. They do NOT
cover mid-flight reconnect/replay (unit-tested in ``test/dsx/test_reliability.py``) or
workflow-level metadata gating (needs the model stack); those remain for the full runtime/parallel
test.
"""

from __future__ import annotations

import os
import time
import uuid

import paho.mqtt.client as mqtt
import pytest
from src.dsx.bus import BusTransport
from src.dsx.publisher import DSXPublisher

_HOST, _PORT = (
    os.environ.get("DSX_TEST_BROKER", "127.0.0.1:1883").split(":") + ["1883"]
)[:2]
_PORT = int(_PORT)


def _broker_up() -> bool:
    """True if an MQTT broker accepts a connection at the configured host/port."""
    try:
        c = mqtt.Client(
            client_id=f"probe-{uuid.uuid4().hex[:8]}",
            protocol=mqtt.MQTTv311,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
    except (TypeError, AttributeError):
        c = mqtt.Client(
            client_id=f"probe-{uuid.uuid4().hex[:8]}", protocol=mqtt.MQTTv311
        )
    try:
        c.connect(_HOST, _PORT, keepalive=5)
        return True
    except OSError:
        return False
    finally:
        c.disconnect()


pytestmark = pytest.mark.skipif(
    not _broker_up(), reason=f"no MQTT broker at {_HOST}:{_PORT} (set DSX_TEST_BROKER)"
)


def _collect(topics: list[str], seconds: float = 1.5) -> list[tuple[str, bool]]:
    """Subscribe fresh and collect (topic, retain) for a window. Retained msgs arrive on connect."""
    got: list[tuple[str, bool]] = []
    try:
        c = mqtt.Client(
            client_id=f"sub-{uuid.uuid4().hex[:8]}",
            protocol=mqtt.MQTTv311,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
    except (TypeError, AttributeError):
        c = mqtt.Client(client_id=f"sub-{uuid.uuid4().hex[:8]}", protocol=mqtt.MQTTv311)
    c.on_message = lambda cl, u, m: got.append((m.topic, bool(m.retain)))
    c.on_connect = lambda cl, u, f, rc, props=None: [
        cl.subscribe(t, qos=1) for t in topics
    ]
    c.connect(_HOST, _PORT, keepalive=10)
    c.loop_start()
    try:
        time.sleep(seconds)
    finally:
        c.loop_stop()
        c.disconnect()
    return got


def _publisher(client_id: str) -> tuple[DSXPublisher, BusTransport]:
    schema = _AllowSchema()
    transport = BusTransport(_HOST, _PORT, "noauth", qos=1, client_id=client_id)
    transport.connect()
    return DSXPublisher(transport, schema, dry_run=False), transport


class _AllowSchema:
    """Minimal schema stand-in: accept any topic (contract validation is unit-tested elsewhere)."""

    def match_topic(self, channel: str, topic: str) -> bool:
        return True


def test_metadata_retained_forecast_not_retained() -> None:
    prefix = f"Weather/v1/PUB/it-{uuid.uuid4().hex[:8]}"
    meta_topic = f"{prefix}/Metadata/s/Temperature"
    fcst_topic = f"{prefix}/Forecast/s/Temperature"
    pub, transport = _publisher(f"pub-{uuid.uuid4().hex[:8]}")
    try:
        # Metadata is retained; forecast is live (not retained).
        assert pub.publish_metadata([(meta_topic, {"unit": "K"})]) is True
        pub.stage_batch([(fcst_topic, {"leadSeconds": [0], "values": [1.0]})])
        assert pub.publish_pending() is True
        # A subscriber connecting AFTER both were published gets the retained Metadata only.
        got = _collect([f"{prefix}/#"])
        topics = {t for t, _r in got}
        assert (
            meta_topic in topics
        ), "retained Metadata should be delivered to a later subscriber"
        assert fcst_topic not in topics, "live Forecast must NOT be retained"
        assert all(
            r for t, r in got if t == meta_topic
        ), "Metadata must arrive with the retain flag"
    finally:
        # Clear the retained Metadata so the broker isn't left with test state.
        transport.publish(meta_topic, b"", retain=True)
        transport.close()


def test_distinct_client_ids_coexist() -> None:
    prefix = f"Weather/v1/PUB/it-{uuid.uuid4().hex[:8]}"
    pub_a, ta = _publisher(f"conus-{uuid.uuid4().hex[:8]}")
    pub_b, tb = _publisher(f"sfno-{uuid.uuid4().hex[:8]}")
    try:
        # Distinct client ids: both sessions stay connected and both publishes are broker-confirmed
        # (a shared id would have made the broker drop the first session).
        ta_topic = f"{prefix}/Forecast/a/Temperature"
        tb_topic = f"{prefix}/Forecast/b/Temperature"
        pub_a.stage_batch([(ta_topic, {"leadSeconds": [0], "values": [1.0]})])
        pub_b.stage_batch([(tb_topic, {"leadSeconds": [0], "values": [2.0]})])
        assert pub_a.publish_pending() is True
        assert pub_b.publish_pending() is True
    finally:
        ta.close()
        tb.close()
