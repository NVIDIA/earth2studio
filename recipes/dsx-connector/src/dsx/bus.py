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

"""Publish messages to the DSX Exchange bus using MQTT.

DSX currently accepts MQTT 3.1.1 connections through NATS. This module handles connections,
authentication, TLS, and message delivery, but is not intended to be a general MQTT library.
Message queuing and replay are implemented in ``publisher.py``; DSX weather messages are created
in ``contract_adapter.py``.

Authentication can be disabled with ``noauth`` for local testing or enabled with ``oauth2``.
OAuth2 sends username ``oauthtoken`` and the bearer token as the MQTT password over TLS. The token
provider is called again before reconnecting, allowing file-based tokens to be updated. Connection
and authentication failures raise errors, while failed publishes are reported to the caller.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import paho.mqtt.client as mqtt
from loguru import logger

from .auth import (
    EnvTokenProvider,
    FileTokenProvider,
    StaticTokenProvider,
    TokenProvider,
)


def _reason_code_is_failure(reason_code: Any) -> bool:
    # Paho 2.x uses a ReasonCode object; Paho 1.x uses an integer where 0 means success.
    if hasattr(reason_code, "is_failure"):
        return bool(reason_code.is_failure)
    return int(reason_code) != 0


def build_transport(bus_cfg: dict[str, Any], dry_run: bool) -> BusTransport | None:
    """Build and connect a :class:`BusTransport` from the ``bus:`` config.

    All workflows use this function to configure the broker connection. It validates the shared bus
    settings before loading the model, so configuration errors fail quickly.

    The example configuration uses ``broker_host: CHANGE-ME`` because the broker address depends on
    the deployment. A dry-run accepts this placeholder because it does not contact the broker.
    Before publishing, replace it with the actual DSX broker address.

    With OAuth2, a token file is read during setup and again before reconnecting. Changes to that
    file are therefore used on the next reconnect. An environment-based token requires restarting
    the connector. Obtaining and refreshing tokens remains the operator's responsibility.

    Parameters
    ----------
    bus_cfg : dict[str, Any]
        The ``bus:`` section of the workflow configuration. It contains the broker connection,
        authentication, message-delivery, and timeout settings.
    dry_run : bool
        If True, validate the config but do not connect (returns None).

    Returns
    -------
    BusTransport | None
        A connected transport, or None in dry-run.

    Raises
    ------
    ValueError
        If the bus configuration is invalid.
    OSError
        If the network or TLS connection cannot be established.
    ConnectionError
        If the broker rejects the connection (e.g. an auth failure).
    TimeoutError
        If the broker does not respond to the connection request within ``connect_timeout``.
    """
    # Validate settings that do not require a broker connection, including during dry-runs.
    # Booleans and floats can compare equal to integers in Python, so reject them explicitly.
    qos = bus_cfg.get("qos", 0)
    if not isinstance(qos, int) or isinstance(qos, bool) or qos not in (0, 1):
        raise ValueError(f"bus.qos must be the integer 0 or 1, got {qos!r}")
    # Each connector instance must have a unique client ID. If two running connectors use the same
    # ID, the broker disconnects the first one. Require the ID explicitly and validate it during
    # dry-runs.
    client_id = bus_cfg.get("client_id")
    if not isinstance(client_id, str) or not client_id.strip():
        raise ValueError(
            "bus.client_id is required and must be a non-empty string, unique per connector "
            "instance (a shared id makes the broker drop the older session)"
        )
    # Both timeouts default to 10 seconds and must be positive, finite numbers.
    for key in ("connect_timeout", "publish_timeout"):
        value = bus_cfg.get(key, 10.0)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(
                f"bus.{key} must be a positive, finite number, got {value!r}"
            )
    # Validate authentication before returning from a dry-run, so invalid settings are always
    # reported.
    auth = bus_cfg.get("auth", "noauth")
    if auth not in ("noauth", "oauth2"):
        raise ValueError(f"bus.auth must be 'noauth' or 'oauth2', got {auth!r}")
    if auth == "oauth2" and bus_cfg.get("tls") is False:
        raise ValueError(
            "bus.auth 'oauth2' requires TLS (set tls: true): the bearer token is sent as the "
            "MQTT password and must never be transmitted in cleartext"
        )
    token_file: str | None = None
    if auth == "oauth2":
        configured_token_file = bus_cfg.get("token_file")
        if configured_token_file is not None:
            if (
                not isinstance(configured_token_file, str)
                or not configured_token_file.strip()
            ):
                raise ValueError(
                    "bus.token_file must be a non-empty string when provided"
                )
            token_file = configured_token_file.strip()

    if dry_run:
        return None

    # A real run requires a configured broker host and a valid port.
    # Example configurations use CHANGE-ME until the deployment's broker is configured.
    host = bus_cfg.get("broker_host")
    if not isinstance(host, str) or not host.strip() or host.strip() == "CHANGE-ME":
        raise ValueError("bus.broker_host is required (set it to your DSX broker host)")
    host = host.strip()
    port = bus_cfg.get("broker_port", 1883)
    if not isinstance(port, int) or isinstance(port, bool) or not (1 <= port <= 65535):
        raise ValueError(
            f"bus.broker_port must be an integer in [1, 65535], got {port!r}"
        )

    provider: TokenProvider | None = None
    if auth == "oauth2":
        provider = FileTokenProvider(token_file) if token_file else EnvTokenProvider()
    transport = BusTransport(
        host,
        port,
        auth,
        token_provider=provider,
        qos=qos,
        tls=bus_cfg.get("tls"),
        ca_certs=bus_cfg.get("ca_certs"),
        connect_timeout=bus_cfg.get("connect_timeout", 10.0),
        publish_timeout=bus_cfg.get("publish_timeout", 10.0),
        client_id=client_id,
    )
    try:
        transport.connect()
    except Exception:
        try:
            transport.close()
        except Exception:
            logger.opt(exception=True).warning(
                "failed to close transport after connection failure"
            )
        raise
    return transport


class BusTransport:
    """Publish MQTT messages to the DSX bus.

    Parameters
    ----------
    host : str
        Broker hostname or address.
    port : int
        Broker MQTT port.
    auth_mode : str, optional
        Authentication method: ``noauth`` or ``oauth2``, by default ``noauth``. OAuth2 sends the
        bearer token as the MQTT password and requires TLS.
    token : str | None, optional
        Fixed bearer token used when ``token_provider`` is not provided, by default None. Supply
        tokens through deployment configuration and never commit them.
    token_provider : TokenProvider | None, optional
        Source for the bearer token, by default None. It takes precedence over ``token`` and is
        called during setup and before reconnecting, allowing file-based tokens to be updated.
    qos : int, optional
        MQTT delivery level, by default 0. Level 0 sends without waiting for broker confirmation;
        level 1 waits for confirmation.
    client_id : str, optional
        Identifier for this broker connection, by default ``earth2-dsx-connector``. Each running
        connector must use a unique value.
    tls : bool | None, optional
        Whether to encrypt the connection with TLS, by default None. None enables TLS for
        ``oauth2`` and disables it for ``noauth``.
    ca_certs : str | None, optional
        Path to a certificate-authority bundle for TLS verification, by default None. None uses the
        system certificate store.
    connect_timeout : float, optional
        Seconds to wait for the broker to respond to a connection request, by default 10.0
    publish_timeout : float, optional
        Seconds to wait for the broker to confirm a level 1 message, by default 10.0. This setting
        is not used for level 0 messages.

    Raises
    ------
    ValueError
        If ``auth_mode`` is unknown, OAuth2 has no token source, or a token would be used without
        TLS.
    """

    def __init__(
        self,
        host: str,
        port: int,
        auth_mode: str = "noauth",
        token: str | None = None,
        token_provider: TokenProvider | None = None,
        qos: int = 0,
        client_id: str = "earth2-dsx-connector",
        tls: bool | None = None,
        ca_certs: str | None = None,
        connect_timeout: float = 10.0,
        publish_timeout: float = 10.0,
    ) -> None:
        self.qos = qos
        self.host, self.port = host, port
        self.connect_timeout = connect_timeout
        self.publish_timeout = publish_timeout
        self.publish_failures = 0
        # Allow only one thread at a time to publish, reconnect, or close the connection. This
        # prevents the heartbeat and forecast loop from using the MQTT client
        # simultaneously. Use RLock because reconnect() calls connect() while already holding this
        # lock.
        self._io_lock = threading.RLock()
        self._provider = token_provider
        self._connected = threading.Event()
        self._connect_failure: str | None = None
        try:  # paho-mqtt 2.x requires the callback API version
            self._client = mqtt.Client(
                client_id=client_id,
                protocol=mqtt.MQTTv311,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                reconnect_on_failure=False,
            )
        except (TypeError, AttributeError):  # Compatibility with paho-mqtt 1.x
            self._client = mqtt.Client(
                client_id=client_id,
                protocol=mqtt.MQTTv311,
                reconnect_on_failure=False,
            )
        # The workflow handles reconnects so it can reload the token and retry pending messages.
        if auth_mode == "oauth2":
            if self._provider is None and token:
                self._provider = StaticTokenProvider(token)
            if self._provider is None:
                raise ValueError("oauth2 requires a bearer token or token_provider")
            if tls is False:
                raise ValueError(
                    "oauth2 requires TLS (the token is sent as the MQTT password); "
                    "set tls: true and connect over the broker's TLS MQTT endpoint"
                )
            tls = True
        elif auth_mode != "noauth":
            raise ValueError(f"unknown auth_mode: {auth_mode}")
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        if tls:
            self._client.tls_set(ca_certs=ca_certs)
        # Never configure bearer-token credentials for an unencrypted connection.
        if self._provider is not None:
            if not tls:
                raise ValueError(
                    "a token provider requires TLS; refusing to set credentials on a "
                    "non-TLS client"
                )
            self._apply_token()

    def _apply_token(self) -> None:
        """Set the current bearer token (from the provider) as the MQTT password."""
        if self._provider is not None:
            self._client.username_pw_set("oauthtoken", self._provider.token())

    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Any,
        reason_code: Any,
        properties: Any = None,
    ) -> None:
        """Record whether the broker accepted the connection and wake the waiting connect() call."""
        if _reason_code_is_failure(reason_code):
            self._connect_failure = str(reason_code)
            logger.error("DSX bus rejected connection: {}", reason_code)
        else:
            self._connect_failure = None
            logger.info("DSX bus connected")
        self._connected.set()

    def _on_disconnect(self, client: mqtt.Client, userdata: Any, *args: Any) -> None:
        """Log whether the broker connection closed normally or unexpectedly."""
        # Paho 2.x passes flags, reason, and properties; Paho 1.x passes only the reason.
        reason_code = args[-2] if len(args) > 1 else args[0]
        if _reason_code_is_failure(reason_code):
            logger.warning("DSX bus disconnected unexpectedly: {}", reason_code)
        else:
            logger.info("DSX bus disconnected")

    def connect(self) -> None:
        """Connect to the broker and wait for its response.

        Raises
        ------
        OSError
            If the TCP/TLS connection to the broker cannot be established (e.g. connection refused,
            DNS failure, or a TLS handshake error).
        TimeoutError
            If the broker does not respond within ``connect_timeout``.
        ConnectionError
            If the broker rejects the connection, for example because authentication fails.
        """
        with self._io_lock:
            logger.info("connecting to DSX bus {}:{}", self.host, self.port)
            self._connected.clear()
            self._connect_failure = None
            # Bound paho's own socket-connect / TLS-handshake phase by connect_timeout too (it
            # otherwise uses paho's ~5s default), so a single deadline governs TCP+TLS and the
            # CONNACK wait below rather than only the latter. paho 2.x exposes a public
            # ``connect_timeout`` property; paho 1.6.x only the private field.
            if isinstance(
                getattr(type(self._client), "connect_timeout", None), property
            ):
                self._client.connect_timeout = self.connect_timeout
            else:
                self._client._connect_timeout = self.connect_timeout
            deadline = time.monotonic() + self.connect_timeout
            try:
                self._client.connect(self.host, self.port)
                self._client.loop_start()
            except Exception:
                self._stop_and_disconnect()
                raise
            # Wait for CONNACK with whatever remains of the single connect_timeout deadline.
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not self._connected.wait(remaining):
                self._stop_and_disconnect()
                raise TimeoutError(
                    f"broker {self.host}:{self.port} did not respond within "
                    f"{self.connect_timeout}s"
                )
            if self._connect_failure is not None:
                self._stop_and_disconnect()
                raise ConnectionError(
                    f"DSX bus rejected connection: {self._connect_failure}"
                )

    def _stop_and_disconnect(self) -> None:
        """Close the connection and stop the network loop without raising cleanup errors."""
        try:
            self._client.disconnect()
        except Exception:
            logger.opt(exception=True).debug("error disconnecting MQTT client")
        try:
            self._client.loop_stop()
        except Exception:
            logger.opt(exception=True).debug("error stopping MQTT network loop")

    def publish(self, topic: str, payload: bytes, retain: bool = False) -> bool:
        """Publish an MQTT message using the configured delivery level.

        For level 0, True means the MQTT client accepted the message for sending; the broker does
        not confirm receipt. For level 1, True means the broker acknowledged the message. It does
        not confirm that a downstream consumer received it.

        Parameters
        ----------
        topic : str
            MQTT topic to publish to.
        payload : bytes
            Encoded message content.
        retain : bool, optional
            Whether the broker should store the message as the latest value for this topic, by
            default False

        Returns
        -------
        bool
            True when the MQTT client accepts a level 0 message for sending or the broker
            acknowledges a level 1 message. False when publishing fails. Failures are counted in
            ``publish_failures`` so the caller can reconnect and retry.
        """
        with self._io_lock:
            info = self._client.publish(topic, payload, qos=self.qos, retain=retain)
            ok = info.rc == mqtt.MQTT_ERR_SUCCESS
            if ok and self.qos > 0:
                try:
                    info.wait_for_publish(self.publish_timeout)
                    ok = bool(info.is_published())
                except (ValueError, RuntimeError):
                    ok = False
            if not ok:
                self.publish_failures += 1
                logger.debug("publish to {} failed (rc={})", topic, info.rc)
            return ok

    def reconnect(self) -> bool:
        """Reconnect using the latest available token.

        The existing connection is closed first. The token provider is then called again before
        opening a new connection, allowing an updated token file to take effect. The caller
        controls the delay between reconnection attempts.

        Returns
        -------
        bool
            True if reconnection succeeds; otherwise False.
        """
        with self._io_lock:
            self._stop_and_disconnect()  # tear down any stale connection first
            try:
                self._apply_token()
                self.connect()
                return True
            except Exception as exc:
                logger.warning("DSX bus reconnect failed: {}", exc)
                return False

    def reconnect_if_disconnected(self) -> bool:
        """Reconnect only if the client has lost its connection.

        The check and the reconnect happen under one lock, so a thread cannot tear down a
        connection that another thread has just restored.

        Returns
        -------
        bool
            True if a reconnect was needed and succeeded. False if the client was already
            connected or the reconnect failed.
        """
        with self._io_lock:
            if self._client.is_connected():
                return False
            return self.reconnect()

    def close(self) -> None:
        """Wait for any active publish, then disconnect and stop the network loop."""
        with self._io_lock:
            self._stop_and_disconnect()
