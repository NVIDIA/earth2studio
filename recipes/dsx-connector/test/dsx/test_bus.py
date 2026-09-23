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


"""MQTT transport and token-provider behavior."""

from __future__ import annotations

from unittest.mock import MagicMock

import paho.mqtt.client as mqtt
import pytest
from src.dsx.auth import EnvTokenProvider, FileTokenProvider, StaticTokenProvider
from src.dsx.bus import BusTransport


class _RC:
    """Fake paho ReasonCode."""

    def __init__(self, is_failure: bool) -> None:
        self.is_failure = is_failure

    def __str__(self) -> str:
        return "failure" if self.is_failure else "success"


def test_bus_oauth2_requires_token() -> None:
    with pytest.raises(ValueError):
        BusTransport("h", 8883, "oauth2")


def test_bus_oauth2_rejects_disabled_tls() -> None:
    with pytest.raises(ValueError):
        BusTransport("h", 8883, "oauth2", token="tok", tls=False)  # noqa: S106


def test_bus_unknown_auth_mode() -> None:
    with pytest.raises(ValueError):
        BusTransport("h", 1883, "bogus")


@pytest.mark.parametrize("ca_certs", [None, "ca.pem"])
def test_bus_oauth2_configures_credentials_and_ca(monkeypatch, ca_certs) -> None:
    fake = MagicMock()
    monkeypatch.setattr("src.dsx.bus.mqtt.Client", MagicMock(return_value=fake))
    BusTransport("h", 8883, "oauth2", token="tok", ca_certs=ca_certs)  # noqa: S106
    fake.username_pw_set.assert_called_once_with("oauthtoken", "tok")
    fake.tls_set.assert_called_once_with(ca_certs=ca_certs)


def test_bus_noauth_no_tls_no_creds(monkeypatch) -> None:
    fake = MagicMock()
    monkeypatch.setattr("src.dsx.bus.mqtt.Client", MagicMock(return_value=fake))
    BusTransport("h", 1883, "noauth")
    fake.tls_set.assert_not_called()
    fake.username_pw_set.assert_not_called()


def test_bus_disables_paho_automatic_reconnect(monkeypatch) -> None:
    client_factory = MagicMock(return_value=MagicMock())
    monkeypatch.setattr("src.dsx.bus.mqtt.Client", client_factory)
    BusTransport("h", 1883, "noauth")
    assert client_factory.call_args.kwargs["reconnect_on_failure"] is False


def test_bus_disables_paho_automatic_reconnect_with_paho1(monkeypatch) -> None:
    client_factory = MagicMock(side_effect=[TypeError, MagicMock()])
    monkeypatch.setattr("src.dsx.bus.mqtt.Client", client_factory)
    BusTransport("h", 1883, "noauth")
    assert client_factory.call_count == 2
    assert client_factory.call_args.kwargs["reconnect_on_failure"] is False


def test_static_token_provider() -> None:
    assert StaticTokenProvider("tok").token() == "tok"  # noqa: S106
    assert StaticTokenProvider("  tok  ").token() == "tok"  # noqa: S106
    with pytest.raises(ValueError):
        StaticTokenProvider("")
    with pytest.raises(ValueError):
        StaticTokenProvider("   ")


def test_env_token_provider(monkeypatch) -> None:
    monkeypatch.delenv("DSX_OAUTH_TOKEN", raising=False)
    with pytest.raises(ValueError, match="unset or empty"):
        EnvTokenProvider().token()
    monkeypatch.setenv("DSX_OAUTH_TOKEN", "   ")
    with pytest.raises(ValueError, match="unset or empty"):
        EnvTokenProvider().token()
    monkeypatch.setenv("DSX_OAUTH_TOKEN", "  tok-1  ")
    assert EnvTokenProvider().token() == "tok-1"


def test_bus_connect_raises_on_auth_failure(monkeypatch) -> None:
    fake = MagicMock()
    monkeypatch.setattr("src.dsx.bus.mqtt.Client", MagicMock(return_value=fake))
    t = BusTransport("h", 1883, "noauth", connect_timeout=1.0)
    fake.loop_start.side_effect = lambda: t._on_connect(fake, None, None, _RC(True))
    with pytest.raises(ConnectionError):
        t.connect()


def test_bus_connect_uses_io_lock(monkeypatch) -> None:
    fake = MagicMock()
    monkeypatch.setattr("src.dsx.bus.mqtt.Client", MagicMock(return_value=fake))
    transport = BusTransport("h", 1883, "noauth", connect_timeout=1.0)
    lock = MagicMock()
    transport._io_lock = lock
    fake.loop_start.side_effect = lambda: transport._on_connect(
        fake, None, None, _RC(False)
    )
    transport.connect()
    fake.connect.assert_called_once_with("h", 1883)
    fake.loop_start.assert_called_once()
    fake.disconnect.assert_not_called()
    fake.loop_stop.assert_not_called()
    lock.__enter__.assert_called_once()
    lock.__exit__.assert_called_once()


def test_bus_qos0_reports_local_acceptance(monkeypatch) -> None:
    fake = MagicMock()
    monkeypatch.setattr("src.dsx.bus.mqtt.Client", MagicMock(return_value=fake))
    transport = BusTransport("h", 1883, "noauth", qos=0)
    info = MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)
    fake.publish.return_value = info

    assert transport.publish("topic", b"{}", retain=True)
    fake.publish.assert_called_once_with("topic", b"{}", qos=0, retain=True)
    info.wait_for_publish.assert_not_called()

    fake.publish.return_value = MagicMock(rc=mqtt.MQTT_ERR_NO_CONN)
    assert not transport.publish("topic", b"{}")
    assert transport.publish_failures == 1


def test_bus_qos1_requires_broker_confirmation(monkeypatch) -> None:
    fake = MagicMock()
    monkeypatch.setattr("src.dsx.bus.mqtt.Client", MagicMock(return_value=fake))
    transport = BusTransport("h", 1883, "noauth", qos=1, publish_timeout=2.0)
    info = MagicMock(rc=mqtt.MQTT_ERR_SUCCESS)
    info.is_published.return_value = True
    fake.publish.return_value = info

    assert transport.publish("topic", b"{}")
    info.wait_for_publish.assert_called_once_with(2.0)

    info.is_published.return_value = False
    assert not transport.publish("topic", b"{}")
    assert transport.publish_failures == 1


def test_bus_connect_propagates_tcp_error(monkeypatch) -> None:
    fake = MagicMock()
    fake.connect.side_effect = OSError("connection refused")
    monkeypatch.setattr("src.dsx.bus.mqtt.Client", MagicMock(return_value=fake))
    t = BusTransport("h", 1883, "noauth")
    with pytest.raises(OSError):
        t.connect()
    fake.loop_start.assert_not_called()  # fails before starting the loop
    fake.disconnect.assert_called()
    fake.loop_stop.assert_called()


def test_bus_on_connect_int_rc_paho1x(monkeypatch) -> None:
    # paho 1.x delivers an int rc (no ReasonCode); non-zero must register failure.
    fake = MagicMock()
    monkeypatch.setattr("src.dsx.bus.mqtt.Client", MagicMock(return_value=fake))
    t = BusTransport("h", 1883, "noauth")
    t._on_connect(fake, None, None, 5)  # 5 = not authorized
    assert t._connect_failure is not None
    t._on_connect(fake, None, None, 0)  # success
    assert t._connect_failure is None


def test_bus_logs_normal_disconnect_as_info(monkeypatch, caplog) -> None:
    fake = MagicMock()
    monkeypatch.setattr("src.dsx.bus.mqtt.Client", MagicMock(return_value=fake))
    transport = BusTransport("h", 1883, "noauth")
    with caplog.at_level("INFO"):
        transport._on_disconnect(fake, None, object(), _RC(False), None)
    assert any(
        record.levelname == "INFO" and "disconnected" in record.message
        for record in caplog.records
    )


def test_bus_logs_unexpected_paho1_disconnect_as_warning(monkeypatch, caplog) -> None:
    fake = MagicMock()
    monkeypatch.setattr("src.dsx.bus.mqtt.Client", MagicMock(return_value=fake))
    transport = BusTransport("h", 1883, "noauth")
    transport._on_disconnect(fake, None, 1)
    assert any(
        record.levelname == "WARNING" and "unexpectedly" in record.message
        for record in caplog.records
    )


def test_file_token_provider(tmp_path) -> None:
    p = tmp_path / "token"
    p.write_text("  tok-file\n")  # whitespace stripped
    assert FileTokenProvider(str(p)).token() == "tok-file"
    p.write_text("")  # empty
    with pytest.raises(ValueError):
        FileTokenProvider(str(p)).token()
    with pytest.raises(ValueError):
        FileTokenProvider(str(tmp_path / "missing")).token()


def test_bus_provider_without_tls_rejected() -> None:
    # A token provider on a non-TLS client must be refused (never stage creds in cleartext).
    with pytest.raises(ValueError):
        BusTransport(
            "h", 1883, "noauth", token_provider=StaticTokenProvider("tok")
        )  # noqa: S106


def test_bus_reconnect_refreshes_token(monkeypatch, tmp_path) -> None:
    fake = MagicMock()
    monkeypatch.setattr("src.dsx.bus.mqtt.Client", MagicMock(return_value=fake))
    p = tmp_path / "token"
    p.write_text("tok-1")
    t = BusTransport("h", 8883, "oauth2", token_provider=FileTokenProvider(str(p)))
    fake.username_pw_set.assert_called_with("oauthtoken", "tok-1")
    p.write_text("tok-2")  # mounted-secret file rotated out-of-band
    monkeypatch.setattr(
        t, "connect", lambda: None
    )  # don't really reconnect in the test
    assert t.reconnect() is True
    fake.username_pw_set.assert_called_with(
        "oauthtoken", "tok-2"
    )  # fresh token on reconnect


def test_connect_timeout_tears_down_socket(monkeypatch) -> None:
    # On a connect timeout, connect() must disconnect the socket (not just stop the loop) so it
    # isn't left open until the caller's next close()/reconnect().
    fake = MagicMock()
    teardown_order = []
    fake.disconnect.side_effect = lambda: teardown_order.append("disconnect")
    fake.loop_stop.side_effect = lambda: teardown_order.append("loop_stop")
    monkeypatch.setattr("src.dsx.bus.mqtt.Client", MagicMock(return_value=fake))
    t = BusTransport(
        "h", 1883, "noauth", connect_timeout=0.01
    )  # on_connect never fires
    with pytest.raises(TimeoutError):
        t.connect()
    fake.loop_stop.assert_called()
    fake.disconnect.assert_called()  # socket torn down on the failure path
    assert teardown_order == ["disconnect", "loop_stop"]


def test_stop_and_disconnect_swallows_teardown_errors(monkeypatch) -> None:
    # A loop-stop error must not prevent the disconnect attempt, and neither teardown error may
    # mask the original error.
    fake = MagicMock()
    fake.loop_stop.side_effect = RuntimeError("loop-stop boom")
    fake.disconnect.side_effect = RuntimeError("teardown boom")
    monkeypatch.setattr("src.dsx.bus.mqtt.Client", MagicMock(return_value=fake))
    t = BusTransport("h", 1883, "noauth", connect_timeout=0.01)
    with pytest.raises(TimeoutError):  # TimeoutError, NOT the teardown RuntimeError
        t.connect()
    fake.disconnect.assert_called()


def test_connect_applies_timeout_to_socket_phase(monkeypatch) -> None:
    # connect_timeout must bound paho's socket/TLS connect phase, not only the CONNACK wait, so a
    # firewalled host or stalled handshake fails within budget instead of on paho's ~5s default.
    fake = MagicMock()
    monkeypatch.setattr("src.dsx.bus.mqtt.Client", MagicMock(return_value=fake))
    transport = BusTransport("h", 1883, "noauth", connect_timeout=3.0)
    # Drive a successful CONNACK so connect() returns: firing on_connect sets the _connected event.
    fake.connect.side_effect = lambda *a, **k: transport._on_connect(
        fake, None, {}, _RC(False)
    )
    transport.connect()
    # MagicMock's class exposes no real property, so the code takes the private-field fallback.
    assert fake._connect_timeout == 3.0


def test_reconnect_if_disconnected_leaves_a_live_connection_alone(monkeypatch) -> None:
    # Another thread may have just reconnected; tearing that connection down would drop the QoS 0
    # messages it has queued.
    fake = MagicMock()
    fake.is_connected.return_value = True
    monkeypatch.setattr("src.dsx.bus.mqtt.Client", MagicMock(return_value=fake))
    transport = BusTransport("h", 1883, "noauth")
    transport.reconnect = MagicMock()  # type: ignore[method-assign]

    assert not transport.reconnect_if_disconnected()
    transport.reconnect.assert_not_called()


def test_reconnect_if_disconnected_reconnects_a_lost_connection(monkeypatch) -> None:
    fake = MagicMock()
    fake.is_connected.return_value = False
    monkeypatch.setattr("src.dsx.bus.mqtt.Client", MagicMock(return_value=fake))
    transport = BusTransport("h", 1883, "noauth")
    transport.reconnect = MagicMock(return_value=True)  # type: ignore[method-assign]

    assert transport.reconnect_if_disconnected()
    transport.reconnect.assert_called_once()
