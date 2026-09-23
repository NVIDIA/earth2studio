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

"""build_transport config validation (torch-free; validation raises before any broker connect)."""

from __future__ import annotations

import pytest
from src.dsx.bus import build_transport


def test_dry_run_returns_none_for_valid_structural_config() -> None:
    # client_id is a structural check now (validated even in dry-run), so it must be present.
    assert (
        build_transport(
            {"qos": 1, "connect_timeout": 5, "client_id": "x"}, dry_run=True
        )
        is None
    )


@pytest.mark.parametrize("client_id", [None, "  "])
def test_client_id_must_be_nonempty_string(client_id) -> None:
    config = {"qos": 1}
    if client_id is not None:
        config["client_id"] = client_id
    with pytest.raises(ValueError, match="client_id"):
        build_transport(config, dry_run=True)


@pytest.mark.parametrize("bad_qos", [True, 1.0, 2, "1"])
def test_qos_must_be_real_int_0_or_1(bad_qos) -> None:
    # bool/float slip through `in (0, 1)` via equality, so they must be rejected explicitly.
    with pytest.raises(ValueError, match="qos"):
        build_transport({"qos": bad_qos}, dry_run=True)


@pytest.mark.parametrize("key", ["connect_timeout", "publish_timeout"])
@pytest.mark.parametrize("bad", [0, float("nan"), float("inf"), "10", True])
def test_timeouts_must_be_positive_finite_numbers(key, bad) -> None:
    # NaN slips past `<= 0`, and a string raises TypeError on comparison — both must be rejected.
    with pytest.raises(ValueError, match=key):
        build_transport({"client_id": "x", key: bad}, dry_run=True)


def test_dry_run_rejects_unknown_auth_mode() -> None:
    # auth is a structural check now: an invalid mode must fail in dry-run, not only on connect.
    with pytest.raises(ValueError, match="auth"):
        build_transport({"client_id": "x", "auth": "bogus"}, dry_run=True)


def test_dry_run_rejects_oauth2_without_tls() -> None:
    # oauth2 sends the token as the MQTT password, so tls=False must be rejected up front.
    with pytest.raises(ValueError, match="TLS"):
        build_transport(
            {"client_id": "x", "auth": "oauth2", "tls": False}, dry_run=True
        )


@pytest.mark.parametrize("token_file", ["   ", 1])
def test_dry_run_rejects_invalid_token_file(token_file) -> None:
    with pytest.raises(ValueError, match="token_file"):
        build_transport(
            {
                "client_id": "x",
                "auth": "oauth2",
                "tls": True,
                "token_file": token_file,
            },
            dry_run=True,
        )


def test_broker_host_required_when_connecting() -> None:
    with pytest.raises(ValueError, match="broker_host"):
        build_transport({"client_id": "x"}, dry_run=False)  # host missing
    with pytest.raises(ValueError, match="broker_host"):
        build_transport({"broker_host": " CHANGE-ME ", "client_id": "x"}, dry_run=False)


def test_broker_port_range_checked() -> None:
    with pytest.raises(ValueError, match="broker_port"):
        build_transport(
            {"broker_host": "h", "broker_port": 70000, "client_id": "x"}, dry_run=False
        )


def test_cleanup_error_does_not_hide_connection_error(monkeypatch) -> None:
    class FailingTransport:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def connect(self) -> None:
            raise RuntimeError("connection failed")

        def close(self) -> None:
            raise RuntimeError("cleanup failed")

    monkeypatch.setattr("src.dsx.bus.BusTransport", FailingTransport)
    with pytest.raises(RuntimeError, match="connection failed"):
        build_transport(
            {"broker_host": "h", "client_id": "x"},
            dry_run=False,
        )
