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

"""Queue and publish DSX forecast and metadata messages.

``DSXPublisher`` sits between ``DSXCoordinator`` and ``BusTransport``. Forecast payloads are copied
before they are added to the queue, so later changes by the caller cannot affect queued messages.
``publish_pending`` attempts each queued forecast once and leaves failed messages in the queue for
another attempt. Reconnection timing is handled in ``publish_loop.py``.

Metadata is sent as retained MQTT messages and is not added to the forecast retry queue. A publish
that returns ``False`` is logged and reported to the caller; exceptions propagate.

The encoded bytes of the latest published forecast for each topic are kept so the heartbeat can
republish them unchanged until the next cycle replaces them, as the DSX contract requires.

Immediately before sending any message, this class validates its MQTT topic. For forecasts, it
also updates ``publicationTime`` to the current send time.
"""

from __future__ import annotations

import copy
import json
import threading
import time
from threading import Event
from typing import Any, Literal

from loguru import logger

from .bus import BusTransport
from .schema import WeatherSchema


class DSXPublisher:
    """Queue and publish DSX forecast and metadata messages.

    Parameters
    ----------
    transport : BusTransport | None
        MQTT transport used to publish messages. Use ``None`` only in dry-run mode.
    schema : WeatherSchema
        Loaded DSX weather contract used to validate MQTT topics.
    dry_run : bool, optional
        If ``True``, print messages instead of publishing them. Default is ``False``.

    Raises
    ------
    ValueError
        If no transport is provided outside dry-run mode.
    """

    def __init__(
        self,
        transport: BusTransport | None,
        schema: WeatherSchema,
        dry_run: bool = False,
    ) -> None:
        if transport is None and not dry_run:
            raise ValueError("transport is required unless dry_run is enabled")
        self.transport = transport
        self.schema = schema
        self.dry_run = dry_run
        # Forecast messages waiting to be published: (topic, payload).
        # Failed messages remain here so they can be retried after reconnecting.
        self._pending: list[tuple[str, dict[str, Any]]] = []
        # Latest published forecast per topic, as sent: republished by the heartbeat thread
        # while the main thread publishes new cycles. Each forecast send and its cache update
        # happen under one lock, per message, so a republish can never follow a newer forecast
        # on the same topic with older bytes.
        self._latest_forecasts: dict[str, bytes] = {}
        self._forecast_lock = threading.Lock()

    def _now_ms(self) -> int:
        """Return the current Unix time in milliseconds.

        Kept as a separate method so tests can replace the clock.
        """
        return time.time_ns() // 1_000_000

    @property
    def has_pending(self) -> bool:
        """Return whether forecast messages are waiting to be published or retried."""
        return bool(self._pending)

    def stage_batch(self, messages: list[tuple[str, dict[str, Any]]]) -> None:
        """Copy validated forecast messages and add them to the pending queue.

        Every payload is copied before the queue is changed. If copying fails, the queue remains
        unchanged. Later changes to the caller's dictionaries cannot affect queued messages.

        Parameters
        ----------
        messages : list[tuple[str, dict[str, Any]]]
            Topic and payload pairs for the forecasts to queue.
        """
        copied_messages = [
            (topic, copy.deepcopy(payload)) for topic, payload in messages
        ]
        self._pending.extend(copied_messages)

    def publish_pending(self) -> bool:
        """Try to publish each queued forecast once.

        Messages are attempted in queue order. Before each attempt, ``publicationTime`` is updated
        to the current time. Successfully published messages are removed from the queue, while
        failed messages remain available for a later retry. Reconnection is handled by
        ``publish_loop.py``.

        If publishing raises an exception, the message that raised and all messages not yet
        attempted remain queued. Messages already published are removed to avoid unnecessary
        duplicates.

        Returns
        -------
        bool
            ``True`` if every queued forecast was published, otherwise ``False``.
        """
        failed_messages: list[tuple[str, dict[str, Any]]] = []
        pending_messages = self._pending
        next_pending_index = 0
        try:
            for message_index, (topic, payload) in enumerate(pending_messages):
                payload["publicationTime"] = self._now_ms()
                if not self._emit(topic, payload, channel="forecast"):
                    failed_messages.append(pending_messages[message_index])
                next_pending_index = message_index + 1
        finally:
            # Keep reported failures, the message that raised, and messages not yet attempted.
            self._pending = failed_messages + pending_messages[next_pending_index:]
        if failed_messages:
            logger.warning("{} forecast message(s) remain queued", len(failed_messages))
        return not failed_messages

    def publish_metadata(self, messages: list[tuple[str, dict[str, Any]]]) -> bool:
        """Publish metadata as retained MQTT messages.

        The broker stores the latest retained message for each topic and provides it to new
        subscribers. Metadata is attempted once and is not added to the forecast retry queue.
        Results of ``False`` are logged after all messages have been attempted. Exceptions propagate
        immediately.

        Parameters
        ----------
        messages : list[tuple[str, dict[str, Any]]]
            Topic and metadata payload pairs already built and validated by the caller.

        Returns
        -------
        bool
            ``True`` if every metadata message was published, otherwise ``False``.
        """
        all_published = True
        for topic, payload in messages:
            if not self._emit(topic, payload, channel="metadata"):
                all_published = False
        if not all_published:
            logger.warning("metadata: one or more retained publishes failed")
        return all_published

    def republish_forecasts(self, stop: Event | None = None) -> bool:
        """Republish the latest forecast for each topic, byte for byte.

        The payloads are the exact bytes of the last successful publish, so ``initTime`` and
        ``publicationTime`` are unchanged. Topics are republished until a newer forecast
        replaces them. Each topic's current bytes are read and sent under the forecast lock, so
        the lock is held for one publish at a time and never across the whole batch. The pass
        stops at the first failure, so a dead connection does not hold up the forecast loop for
        one publish timeout per topic.

        Parameters
        ----------
        stop : Event | None, optional
            Shutdown event; the pass stops between topics once it is set. Default is ``None``.

        Returns
        -------
        bool
            ``True`` if every forecast was republished (or none has been published yet),
            otherwise ``False``.
        """
        if self.dry_run or self.transport is None:
            return True
        transport = self.transport
        with self._forecast_lock:
            topics = list(self._latest_forecasts)
        for topic in topics:
            if stop is not None and stop.is_set():
                return False
            with self._forecast_lock:
                data = self._latest_forecasts[topic]
                if not transport.publish(topic, data, retain=False):
                    logger.warning("forecast republish failed on {}", topic)
                    return False
        return True

    def _emit(
        self,
        topic: str,
        payload: dict[str, Any],
        channel: Literal["forecast", "metadata"],
    ) -> bool:
        """Validate and send one DSX message.

        Forecast messages are not retained by the broker. Metadata messages are retained so new
        subscribers receive the latest metadata. In dry-run mode, the message is printed instead.

        Parameters
        ----------
        topic : str
            MQTT topic for the message.
        payload : dict[str, Any]
            Message content to encode as JSON.
        channel : {"forecast", "metadata"}
            DSX channel used to validate the topic and select its retention behavior.

        Returns
        -------
        bool
            Whether the message was printed or accepted for publishing.

        Raises
        ------
        ValueError
            If the topic does not match the selected DSX channel.
        TypeError
            If the payload cannot be encoded as JSON.
        RuntimeError
            If no transport is available outside dry-run mode.
        """
        if self.schema.match_topic(channel, topic) is None:
            raise ValueError(f"{channel} topic does not match the contract: {topic}")
        data = json.dumps(payload)
        retain = channel == "metadata"
        if self.dry_run:
            print(f"[{'RETAINED' if retain else 'live':8}] {topic}\n  {data}")
            return True
        transport = self.transport
        if transport is None:
            raise RuntimeError("transport is unavailable outside dry-run mode")
        encoded = data.encode("utf-8")
        if channel != "forecast":
            return transport.publish(topic, encoded, retain=retain)
        with self._forecast_lock:
            published = transport.publish(topic, encoded, retain=retain)
            if published:
                self._latest_forecasts[topic] = encoded
        return published
