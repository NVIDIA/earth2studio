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

"""Publish queued DSX forecasts and recover from connection failures.

This module provides model-independent reconnect logic shared by all workflows. When publishing
fails, persistent workflows retry the broker connection with increasing delays, publish metadata
again, and retry the queued forecasts. One-shot workflows return after the first failure.

The module does not depend on StormCast, SFNO, or other model-specific code.
"""

from __future__ import annotations

import random
from threading import Event

from loguru import logger

from .bus import BusTransport
from .coordinator import DSXCoordinator
from .publisher import DSXPublisher

# Persistent workflows retry queued forecasts after a connection failure without rerunning the
# model. The first reconnect is immediate. Delays between later attempts start at 2 seconds and
# increase to at most 60 seconds. One-shot workflows do not reconnect. If all attempts fail, this
# function returns False so the workflow can exit and allow its process supervisor to restart it.
RECONNECT_MAX_ATTEMPTS = 6
RECONNECT_BASE_DELAY = 2.0
RECONNECT_MAX_DELAY = 60.0


def republish_on_heartbeat(
    coordinator: DSXCoordinator,
    publisher: DSXPublisher,
    transport: BusTransport,
    stop: Event,
) -> None:
    """Republish metadata and the latest forecasts once, as one heartbeat.

    At QoS 0 a message can be lost; the next heartbeat restores it. If a publish fails because the
    broker dropped the connection (for example an idle connection between forecast cycles), this
    reconnects once and republishes, metadata first. A connection that another thread has already
    restored is left alone.

    Parameters
    ----------
    coordinator : DSXCoordinator
        Coordinator that publishes metadata.
    publisher : DSXPublisher
        Publisher holding the latest forecast for each topic.
    transport : BusTransport
        Broker connection.
    stop : Event
        Shutdown event; republishing stops between topics once it is set.
    """
    # Forecasts follow metadata, as at startup; skipping them after a metadata failure also avoids
    # waiting for another publish timeout on a dead connection.
    published = coordinator.publish_metadata() and publisher.republish_forecasts(stop)
    if not published and not stop.is_set() and transport.reconnect_if_disconnected():
        if coordinator.publish_metadata():
            publisher.republish_forecasts(stop)


def publish_with_reconnect(
    publisher: DSXPublisher,
    coordinator: DSXCoordinator,
    transport: BusTransport | None,
    one_shot: bool,
    stop: Event,
) -> bool:
    """Publish queued forecasts and retry after a connection failure.

    During retries, the existing queued forecast data is reused rather than rebuilt. The publisher
    still updates ``publicationTime`` before each send. A process restart is different: the
    workflow runs the producer again.

    Parameters
    ----------
    publisher : DSXPublisher
        Publisher containing the queued forecasts.
    coordinator : DSXCoordinator
        Coordinator used to publish metadata again after reconnecting.
    transport : BusTransport | None
        Broker connection. ``None`` is used in dry-run mode.
    one_shot : bool
        If ``True``, return after the first publish failure without reconnecting.
    stop : Event
        Shutdown event used to stop retries and interrupt waits.

    Returns
    -------
    bool
        ``True`` when all queued forecasts are accepted. ``False`` when one-shot publishing fails,
        no broker transport is available, shutdown is requested, or all reconnect attempts fail.
    """
    if publisher.publish_pending():
        return True
    if one_shot:
        return False
    if transport is None:
        return False

    delay = RECONNECT_BASE_DELAY
    for attempt in range(1, RECONNECT_MAX_ATTEMPTS + 1):
        if stop.is_set():
            return False

        logger.warning(
            "bus publish failed; reconnecting ({}/{})",
            attempt,
            RECONNECT_MAX_ATTEMPTS,
        )
        reconnected = transport.reconnect()
        if reconnected:
            # Publish metadata first so consumers can interpret the queued forecasts.
            metadata_published = coordinator.publish_metadata()
            if metadata_published and publisher.publish_pending():
                logger.info(
                    "reconnected; refreshed metadata and replayed staged bundles"
                )
                return True

        if attempt == RECONNECT_MAX_ATTEMPTS:
            return False

        # Add a small random variation so multiple connectors do not reconnect simultaneously.
        wait_seconds = min(
            delay + random.uniform(0.0, delay * 0.5), RECONNECT_MAX_DELAY
        )
        if stop.wait(wait_seconds):
            return False
        delay = min(delay * 2, RECONNECT_MAX_DELAY)
    return False  # unreachable: the last attempt returns above
