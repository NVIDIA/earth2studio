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

"""Test the queueing, staging, and retry safeguards in ``src/dsx/publisher.py``."""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock

import pytest
from src.dsx.publisher import DSXPublisher


def _schema() -> MagicMock:
    s = MagicMock()
    s.match_topic.return_value = True
    return s


class _OKTransport:
    def __init__(self) -> None:
        self.sent: list = []
        self.publish_failures = 0

    def publish(self, topic: str, payload: bytes, retain: bool = False) -> bool:
        self.sent.append((topic, payload, retain))
        return True


class _FailTransport:
    def __init__(self) -> None:
        self.publish_failures = 0

    def publish(self, topic: str, payload: bytes, retain: bool = False) -> bool:
        self.publish_failures += 1
        return False


class _RaiseOnTransport:
    """Publishes successfully until it hits a topic containing ``raise_marker``, then raises."""

    def __init__(self, raise_marker: str) -> None:
        self.raise_marker = raise_marker
        self.sent: list = []
        self.publish_failures = 0

    def publish(self, topic: str, payload: bytes, retain: bool = False) -> bool:
        if self.raise_marker in topic:
            raise RuntimeError("boom")
        self.sent.append((topic, payload, retain))
        return True


def test_transport_is_required_outside_dry_run() -> None:
    with pytest.raises(ValueError, match="transport is required"):
        DSXPublisher(None, _schema(), dry_run=False)


def test_stage_batch_deep_copies_payload() -> None:
    # A staged message must be immune to later mutation of the caller's nested containers.
    pub = DSXPublisher(_OKTransport(), _schema(), dry_run=False)
    values = [1.0, 2.0]
    pct = {"p50": [1.0, 2.0]}
    payload = {"leadSeconds": [0, 1], "values": values, "percentiles": pct}
    pub.stage_batch([("Weather/v1/PUB/Forecast/s/Temperature", payload)])

    values.append(999.0)  # mutate the caller's nested list
    pct["p50"].append(999.0)  # mutate a nested dict's list
    payload["leadSeconds"].append(9)
    pub.stage_batch([("t/b", {"v": 2}), ("t/c", {"v": 3})])

    queued = pub._pending[0][1]
    assert queued["values"] == [1.0, 2.0]
    assert queued["percentiles"] == {"p50": [1.0, 2.0]}
    assert queued["leadSeconds"] == [0, 1]
    assert [topic for topic, _payload in pub._pending[1:]] == ["t/b", "t/c"]


def test_channel_selects_retention_behavior() -> None:
    transport = _OKTransport()
    pub = DSXPublisher(transport, _schema(), dry_run=False)
    pub.stage_batch([("forecast-topic", {"publicationTime": 0})])

    assert pub.publish_pending()
    assert pub.publish_metadata([("metadata-topic", {"unit": "K"})])
    assert [retain for _topic, _payload, retain in transport.sent] == [False, True]


def test_empty_parameter_match_is_accepted() -> None:
    schema = _schema()
    schema.match_topic.return_value = {}
    pub = DSXPublisher(_OKTransport(), schema, dry_run=False)
    pub.stage_batch([("Weather/v1/PUB/Health", {"publicationTime": 0})])

    assert pub.publish_pending()


def test_metadata_failure_is_log_only_not_replayed(caplog) -> None:
    # Metadata failures are counted + warned, but NEVER enter the forecast replay queue.
    pub = DSXPublisher(_FailTransport(), _schema(), dry_run=False)
    with caplog.at_level(logging.WARNING):
        pub.publish_metadata([("Weather/v1/PUB/Metadata/s/Temperature", {"unit": "K"})])
    assert pub._pending == []  # metadata is not queued for replay
    assert any("metadata" in r.message for r in caplog.records)


def test_publish_pending_exception_safe_drops_sent_keeps_rest() -> None:
    # If a publish raises mid-pass, the already-sent bundle must be dropped (not left queued
    # to republish as a duplicate), while the raising bundle and every un-attempted bundle
    # after it stay pending, in order, for the reconnect/replay path to retry.
    tx = _RaiseOnTransport("t/b")
    pub = DSXPublisher(tx, _schema(), dry_run=False)
    pub.stage_batch(
        [
            ("t/a", {"v": 1}),  # sent OK -> must be dropped
            ("t/b", {"v": 2}),  # raises  -> must stay queued
            ("t/c", {"v": 3}),  # never attempted -> must stay queued
        ]
    )
    with pytest.raises(RuntimeError):
        pub.publish_pending()
    assert [t for t, _p, _r in tx.sent] == ["t/a"]  # only the first was sent
    assert [topic for topic, _payload in pub._pending] == [
        "t/b",
        "t/c",
    ]  # rest still queued, in order


def test_republish_forecasts_resends_latest_bytes_unchanged() -> None:
    # The contract requires a republished bundle to be identical, including publicationTime.
    transport = _OKTransport()
    pub = DSXPublisher(transport, _schema(), dry_run=False)
    pub._now_ms = lambda: 1000  # type: ignore[method-assign]
    topic = "Weather/v1/PUB/Forecast/p/s/Temperature"
    pub.stage_batch([(topic, {"initTime": 1, "values": [1.0]})])
    assert pub.publish_pending()

    pub._now_ms = lambda: 2000  # type: ignore[method-assign]
    assert pub.republish_forecasts()
    (_, first, _), (_, again, retain) = transport.sent
    assert again == first
    assert b'"publicationTime": 1000' in again
    assert retain is False


def test_republish_forecasts_keeps_only_newest_cycle_per_topic() -> None:
    transport = _OKTransport()
    pub = DSXPublisher(transport, _schema(), dry_run=False)
    t_topic = "Weather/v1/PUB/Forecast/p/s/Temperature"
    w_topic = "Weather/v1/PUB/Forecast/p/s/WindU"
    pub.stage_batch([(t_topic, {"initTime": 1}), (w_topic, {"initTime": 1})])
    pub.publish_pending()
    pub.stage_batch([(t_topic, {"initTime": 2})])
    pub.publish_pending()
    transport.sent.clear()

    pub.republish_forecasts()
    resent = {topic: payload for topic, payload, _ in transport.sent}
    assert set(resent) == {t_topic, w_topic}
    assert b'"initTime": 2' in resent[t_topic]
    assert b'"initTime": 1' in resent[w_topic]


def test_republish_forecasts_skips_failed_publishes_and_metadata() -> None:
    # Only forecasts the broker accepted are republished; metadata has its own heartbeat path.
    pub = DSXPublisher(_FailTransport(), _schema(), dry_run=False)
    pub.stage_batch([("Weather/v1/PUB/Forecast/p/s/Temperature", {"initTime": 1})])
    assert not pub.publish_pending()
    transport = _OKTransport()
    pub.transport = transport  # type: ignore[assignment]
    pub.publish_metadata([("Weather/v1/PUB/Metadata/p/s/Temperature", {"unit": "K"})])
    transport.sent.clear()

    assert pub.republish_forecasts()
    assert transport.sent == []


def test_republish_forecasts_reports_failure() -> None:
    transport = _OKTransport()
    pub = DSXPublisher(transport, _schema(), dry_run=False)
    pub.stage_batch([("Weather/v1/PUB/Forecast/p/s/Temperature", {"initTime": 1})])
    pub.publish_pending()
    pub.transport = _FailTransport()  # type: ignore[assignment]
    assert not pub.republish_forecasts()


def test_republish_forecasts_stops_at_first_failure() -> None:
    # On a dead connection each publish can wait for a timeout, so do not try every topic.
    transport = _OKTransport()
    pub = DSXPublisher(transport, _schema(), dry_run=False)
    pub.stage_batch(
        [
            ("Weather/v1/PUB/Forecast/p/s/Temperature", {"initTime": 1}),
            ("Weather/v1/PUB/Forecast/p/s/WindU", {"initTime": 1}),
        ]
    )
    pub.publish_pending()
    failing = _FailTransport()
    pub.transport = failing  # type: ignore[assignment]

    assert not pub.republish_forecasts()
    assert failing.publish_failures == 1


def test_republish_forecasts_stops_when_shutdown_is_requested() -> None:
    transport = _OKTransport()
    pub = DSXPublisher(transport, _schema(), dry_run=False)
    pub.stage_batch([("Weather/v1/PUB/Forecast/p/s/Temperature", {"initTime": 1})])
    pub.publish_pending()
    transport.sent.clear()
    stop = threading.Event()
    stop.set()

    assert not pub.republish_forecasts(stop)
    assert transport.sent == []


def test_republish_forecasts_is_noop_in_dry_run(capsys) -> None:
    pub = DSXPublisher(None, _schema(), dry_run=True)
    pub.stage_batch([("Weather/v1/PUB/Forecast/p/s/Temperature", {"initTime": 1})])
    pub.publish_pending()
    capsys.readouterr()
    assert pub.republish_forecasts()
    assert capsys.readouterr().out == ""


class _PausingTransport:
    """Pauses the heartbeat's first publish until the test releases it.

    Sends are recorded in the order ``publish`` returns. ``new_cycle_progress`` is set when the
    new-cycle thread sends, so an implementation that does not serialize sends is caught.
    """

    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes]] = []
        self.heartbeat_in_publish = threading.Event()
        self.release_heartbeat = threading.Event()
        self.new_cycle_progress = threading.Event()
        self._paused = False

    def publish(self, topic: str, payload: bytes, retain: bool = False) -> bool:
        thread = threading.current_thread().name
        if thread == "heartbeat" and not self._paused:
            self._paused = True
            self.heartbeat_in_publish.set()
            assert self.release_heartbeat.wait(5)
        self.sent.append((topic, payload))
        if thread == "new-cycle":
            self.new_cycle_progress.set()
        return True


class _SignalingLock:
    """Lock that sets ``event`` when the new-cycle thread starts waiting for it."""

    def __init__(self, event: threading.Event) -> None:
        self._lock = threading.Lock()
        self._event = event

    def __enter__(self) -> None:
        if threading.current_thread().name == "new-cycle":
            self._event.set()
        self._lock.acquire()

    def __exit__(self, *exc: object) -> None:
        self._lock.release()


def test_republish_never_sends_older_bytes_after_a_newer_forecast() -> None:
    # A heartbeat republish that overlaps a new cycle must not deliver new -> old on a topic.
    transport = _PausingTransport()
    pub = DSXPublisher(transport, _schema(), dry_run=False)  # type: ignore[arg-type]
    pub._forecast_lock = _SignalingLock(transport.new_cycle_progress)  # type: ignore[assignment]
    topic = "Weather/v1/PUB/Forecast/p/s/Temperature"
    pub.stage_batch([(topic, {"initTime": 1})])
    pub.publish_pending()

    heartbeat = threading.Thread(target=pub.republish_forecasts, name="heartbeat")
    heartbeat.start()
    assert transport.heartbeat_in_publish.wait(5)

    def _publish_new_cycle() -> None:
        pub.stage_batch([(topic, {"initTime": 2})])
        pub.publish_pending()

    new_cycle = threading.Thread(target=_publish_new_cycle, name="new-cycle")
    new_cycle.start()
    # Wait until the new cycle has either sent (unsafe) or is waiting for the forecast lock
    # (safe) before letting the paused heartbeat finish its send.
    assert transport.new_cycle_progress.wait(5)
    transport.release_heartbeat.set()
    heartbeat.join(5)
    new_cycle.join(5)

    init_times = [
        2 if b'"initTime": 2' in payload else 1
        for t, payload in transport.sent
        if t == topic
    ]
    assert init_times == [1, 1, 2]
