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


"""DSX publish and reconnect reliability."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

from src.dsx import publish_loop


def _stop() -> threading.Event:
    return threading.Event()


def test_publish_succeeds_first_try() -> None:
    publisher = MagicMock()
    publisher.publish_pending.return_value = True
    transport = MagicMock()
    coordinator = MagicMock()
    assert publish_loop.publish_with_reconnect(
        publisher, coordinator, transport, one_shot=False, stop=_stop()
    )
    transport.reconnect.assert_not_called()  # no reconnect when the first publish works
    # Metadata is only re-published after a reconnect; the happy path leaves it to the heartbeat.
    coordinator.publish_metadata.assert_not_called()


def test_once_mode_fails_fast_without_reconnect() -> None:
    publisher = MagicMock()
    publisher.publish_pending.return_value = False
    transport = MagicMock()
    assert not publish_loop.publish_with_reconnect(
        publisher, MagicMock(), transport, one_shot=True, stop=_stop()
    )
    transport.reconnect.assert_not_called()  # --once never retries


def test_persistent_mode_does_not_reconnect_during_shutdown() -> None:
    publisher = MagicMock()
    publisher.publish_pending.return_value = False
    transport = MagicMock()
    stop = _stop()
    stop.set()

    assert not publish_loop.publish_with_reconnect(
        publisher, MagicMock(), transport, one_shot=False, stop=stop
    )
    transport.reconnect.assert_not_called()


def test_missing_transport_fails_without_waiting() -> None:
    publisher = MagicMock()
    publisher.publish_pending.return_value = False
    stop = MagicMock()

    assert not publish_loop.publish_with_reconnect(
        publisher, MagicMock(), None, one_shot=False, stop=stop
    )
    stop.wait.assert_not_called()


def test_reconnect_refreshes_metadata_before_replay() -> None:
    order: list[str] = []
    publisher = MagicMock()
    pending = iter([False, True])

    def _pending() -> bool:
        order.append("pending")
        return next(pending)

    publisher.publish_pending.side_effect = _pending
    transport = MagicMock()
    transport.reconnect.return_value = True
    coordinator = MagicMock()

    def _metadata() -> bool:
        order.append("metadata")
        return True

    coordinator.publish_metadata.side_effect = _metadata

    assert publish_loop.publish_with_reconnect(
        publisher, coordinator, transport, one_shot=False, stop=_stop()
    )
    transport.reconnect.assert_called_once()
    assert publisher.publish_pending.call_count == 2  # replayed, not rebuilt
    coordinator.publish_metadata.assert_called_once()
    assert order == ["pending", "metadata", "pending"]


def test_reconnect_exhaustion_preserves_count_and_capped_waits(monkeypatch) -> None:
    monkeypatch.setattr(publish_loop, "RECONNECT_BASE_DELAY", 60.0)
    monkeypatch.setattr(publish_loop, "RECONNECT_MAX_DELAY", 60.0)
    monkeypatch.setattr(publish_loop.random, "uniform", lambda _start, _end: 30.0)
    publisher = MagicMock()
    publisher.publish_pending.return_value = False
    transport = MagicMock()
    transport.reconnect.return_value = False
    stop = MagicMock()
    stop.is_set.return_value = False
    stop.wait.return_value = False

    assert not publish_loop.publish_with_reconnect(
        publisher, MagicMock(), transport, one_shot=False, stop=stop
    )
    assert transport.reconnect.call_count == publish_loop.RECONNECT_MAX_ATTEMPTS
    assert stop.wait.call_count == publish_loop.RECONNECT_MAX_ATTEMPTS - 1
    assert all(call.args[0] <= 60.0 for call in stop.wait.call_args_list)


def test_reconnect_failed_metadata_prevents_replay(monkeypatch) -> None:
    monkeypatch.setattr(publish_loop, "RECONNECT_BASE_DELAY", 0.0)
    monkeypatch.setattr(publish_loop, "RECONNECT_MAX_DELAY", 0.0)
    publisher = MagicMock()
    publisher.publish_pending.return_value = False  # top attempt fails
    coordinator = MagicMock()
    coordinator.publish_metadata.return_value = False  # metadata never re-confirms
    transport = MagicMock()
    transport.reconnect.return_value = True

    assert not publish_loop.publish_with_reconnect(
        publisher, coordinator, transport, one_shot=False, stop=_stop()
    )
    # publish_pending called only once (the initial attempt); replay is gated on metadata, which
    # never confirmed, so it is never re-called.
    assert publisher.publish_pending.call_count == 1
    assert (
        coordinator.publish_metadata.call_count == publish_loop.RECONNECT_MAX_ATTEMPTS
    )


def _heartbeat_parts(
    *, published: bool, reconnected: bool
) -> tuple[MagicMock, MagicMock, MagicMock]:
    coordinator = MagicMock()
    coordinator.publish_metadata.return_value = True
    publisher = MagicMock()
    publisher.republish_forecasts.side_effect = [published, True]
    transport = MagicMock()
    transport.reconnect_if_disconnected.return_value = reconnected
    return coordinator, publisher, transport


def test_heartbeat_republishes_metadata_and_forecasts() -> None:
    coordinator, publisher, transport = _heartbeat_parts(
        published=True, reconnected=False
    )
    stop = _stop()
    publish_loop.republish_on_heartbeat(coordinator, publisher, transport, stop)

    coordinator.publish_metadata.assert_called_once()
    publisher.republish_forecasts.assert_called_once_with(stop)
    transport.reconnect_if_disconnected.assert_not_called()


def test_heartbeat_reconnects_lost_connection_and_republishes_metadata_first() -> None:
    coordinator, publisher, transport = _heartbeat_parts(
        published=False, reconnected=True
    )
    calls = MagicMock()
    calls.attach_mock(coordinator.publish_metadata, "metadata")
    calls.attach_mock(publisher.republish_forecasts, "forecasts")
    calls.attach_mock(transport.reconnect_if_disconnected, "reconnect")
    publish_loop.republish_on_heartbeat(coordinator, publisher, transport, _stop())

    assert [c[0] for c in calls.mock_calls] == [
        "metadata",
        "forecasts",
        "reconnect",
        "metadata",
        "forecasts",
    ]


def test_heartbeat_does_not_retry_without_reconnect() -> None:
    # If a publish failed but no reconnect happened, the next heartbeat retries instead of
    # repeating the whole pass immediately.
    coordinator, publisher, transport = _heartbeat_parts(
        published=False, reconnected=False
    )
    publish_loop.republish_on_heartbeat(coordinator, publisher, transport, _stop())

    assert publisher.republish_forecasts.call_count == 1
    assert coordinator.publish_metadata.call_count == 1


def test_heartbeat_does_not_reconnect_during_shutdown() -> None:
    coordinator, publisher, transport = _heartbeat_parts(
        published=False, reconnected=True
    )
    stop = _stop()
    stop.set()
    publish_loop.republish_on_heartbeat(coordinator, publisher, transport, stop)

    transport.reconnect_if_disconnected.assert_not_called()


def test_heartbeat_skips_forecasts_when_metadata_fails() -> None:
    coordinator, publisher, transport = _heartbeat_parts(
        published=True, reconnected=False
    )
    coordinator.publish_metadata.return_value = False
    publish_loop.republish_on_heartbeat(coordinator, publisher, transport, _stop())

    publisher.republish_forecasts.assert_not_called()
    transport.reconnect_if_disconnected.assert_called_once()
